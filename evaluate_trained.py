#!/usr/bin/env python3
"""
Post-Training Evaluation: Trained LoRA Adapter vs All Baselines.

Runs the trained model against all tasks with multi-episode averaging,
then compares to expert, heuristic, naive, and do-nothing baselines.

Usage:
    python evaluate_trained.py                            # full eval (needs adapter)
    python evaluate_trained.py --no-model                 # baselines-only
    python evaluate_trained.py --episodes 5 --verbose     # 5 episodes, detailed
    python evaluate_trained.py --adapter trained_model_full_0p5b --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import get_task

from evaluate import EXPERT_STRATEGIES, NAIVE_STRATEGIES, run_strategy
from train_grpo import build_obs_prompt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVALUABLE_TASKS = [
    "single_service_failure",
    "cascading_failure",
    "hidden_root_cause",
    "chaos_cascade",
    "multi_root_cause",
]

DEP_ORDER = ["database", "cache", "notification", "auth", "payments", "checkout"]


# ---------------------------------------------------------------------------
# Parse model JSON output -> IncidentAction
# ---------------------------------------------------------------------------

def parse_action(text: str) -> Optional[IncidentAction]:
    """Extract a valid IncidentAction from model output text."""
    text = text.strip()
    # Strip code fences
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    # Try direct parse
    try:
        return IncidentAction(**json.loads(text))
    except Exception:
        pass

    # Try to find first JSON object
    idx = text.find("{")
    if idx >= 0:
        depth = 0
        for i in range(idx, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return IncidentAction(**json.loads(text[idx:i+1]))
                    except Exception:
                        break
    return None


# ---------------------------------------------------------------------------
# Heuristic fallback (deterministic, dependency-aware)
# ---------------------------------------------------------------------------

def heuristic_action(obs_dict: Dict, action_history: List[str]) -> IncidentAction:
    """Rule-based fallback matching HeuristicAgent from run_baselines.py."""
    services = obs_dict.get("services", {})
    inspected, restarted, rolled_back = set(), set(), set()
    cleared = False

    for a in action_history:
        if a.startswith("inspect_"):
            inspected.add(a.split(":", 1)[1] if ":" in a else "")
        elif a.startswith("restart_service:"):
            restarted.add(a.split(":", 1)[1])
        elif a.startswith("rollback:"):
            rolled_back.add(a.split(":", 1)[1])
        elif a == "clear_cache":
            cleared = True

    # Rank by severity
    ranked = []
    for name, svc in services.items():
        st = svc.get("status", "healthy")
        ranked.append((0 if st == "down" else (0.5 if st == "degraded" else 1), name, svc))
    ranked.sort()

    # Phase 1: inspect unhealthy
    for _, n, s in ranked:
        if s.get("status") != "healthy" and n not in inspected:
            return IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name=n)

    # Phase 2: rollback version mismatches
    for _, n, s in ranked:
        if s.get("version", "v1.0.0") != "v1.0.0" and n not in rolled_back:
            return IncidentAction(action_type=ActionType.ROLLBACK, service_name=n)

    # Phase 3: clear cache
    if not cleared:
        for _, n, s in ranked:
            if n in ("auth", "checkout") and s.get("status") != "healthy":
                return IncidentAction(action_type=ActionType.CLEAR_CACHE)

    # Phase 4: restart in dependency order
    for n in DEP_ORDER:
        svc = services.get(n, {})
        if svc.get("status") != "healthy" and n not in restarted:
            return IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name=n)

    return IncidentAction(action_type=ActionType.DO_NOTHING)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(base_name: str, adapter_path: str, device: str):
    """Load base model + optional LoRA adapter. Returns (model, tokenizer).
    
    If adapter_path is 'none', loads just the base model (useful for
    evaluating SFT-merged models without a GRPO adapter).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"ERROR: {e}\nInstall: pip install transformers peft torch accelerate", file=sys.stderr)
        sys.exit(1)

    skip_adapter = adapter_path.lower() == "none"

    if not skip_adapter:
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            print(f"\n❌ Adapter not found at '{adapter_path}'", file=sys.stderr)
            print(f"   Get it from your teammate and place in: {Path(__file__).parent / adapter_path}", file=sys.stderr)
            print(f"   Or run with --no-model for baselines-only.\n", file=sys.stderr)
            sys.exit(1)

        for f in ["adapter_config.json", "adapter_model.safetensors"]:
            if not (adapter_dir / f).exists():
                print(f"❌ Missing {f} in {adapter_path}/", file=sys.stderr)
                sys.exit(1)

    # Precision
    dtype = torch.float32
    if device == "mps":
        dtype = torch.float16
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    label = f"{base_name}" if skip_adapter else f"{base_name} + {adapter_path}"
    print(f"\n  Loading {label} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(base_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_name, torch_dtype=dtype,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        model = model.to("mps")

    if not skip_adapter:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    params = sum(p.numel() for p in model.parameters())
    print(f"  ✅ Loaded: {params:,} params on {next(model.parameters()).device}")
    if skip_adapter:
        print(f"  ℹ️  No adapter loaded — evaluating base/merged model directly")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Generate action from trained model
# ---------------------------------------------------------------------------

def generate_action(
    model, tokenizer, obs_dict: Dict, step: int,
    action_history: List[str], deterministic: bool = True,
) -> str:
    """Generate action using EXACT training prompt format (build_obs_prompt).
    
    Args:
        deterministic: If True, use greedy decoding for reproducible benchmarks.
                       If False, use sampling (temperature=0.3) for stress testing.
    """
    import torch

    prompt_text = build_obs_prompt(obs_dict, step, action_history)
    messages = [{"role": "user", "content": prompt_text}]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": 64,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if deterministic:
        gen_kwargs["do_sample"] = False
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.3
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Run heuristic baseline episode
# ---------------------------------------------------------------------------

def run_heuristic_episode(task_name: str, seed: Optional[int] = None) -> Dict[str, Any]:
    """Run one episode with the deterministic heuristic agent."""
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed, chaos_mode=True)
    obs_dict = obs.model_dump()
    action_history: List[str] = []

    while not obs.done:
        action = heuristic_action(obs_dict, action_history)
        a_str = action.action_type.value
        if action.service_name:
            a_str += f":{action.service_name}"
        action_history.append(a_str)
        obs = env.step(action)
        obs_dict = obs.model_dump()

    grade = env.grade()
    env.close()
    return {
        "score": grade["score"], "breakdown": grade["breakdown"],
        "is_resolved": grade["is_resolved"], "steps_taken": grade["steps_taken"],
    }


# ---------------------------------------------------------------------------
# Run trained model episode
# ---------------------------------------------------------------------------

def run_trained_episode(
    task_name: str, model, tokenizer,
    seed: Optional[int] = None, verbose: bool = False,
    deterministic: bool = True,
) -> Dict[str, Any]:
    """Run one episode with the trained LoRA model."""
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed, episode_id=f"eval-{task_name}", chaos_mode=True)
    obs_dict = obs.model_dump()

    action_history: List[str] = []
    parse_fails = 0
    fallback_count = 0
    raw_outputs: List[str] = []
    t0 = time.time()

    while not obs.done:
        step = len(action_history) + 1
        raw = generate_action(
            model, tokenizer, obs_dict, step, action_history,
            deterministic=deterministic,
        )
        raw_outputs.append(raw)
        action = parse_action(raw)

        if action is None:
            parse_fails += 1
            fallback_count += 1
            action = heuristic_action(obs_dict, action_history)
            if verbose:
                print(f"    Step {step:2d}: ⚠️  PARSE FAIL raw={raw[:60]!r} → fallback={action.action_type.value}")

        a_str = action.action_type.value
        if action.service_name:
            a_str += f":{action.service_name}"
        action_history.append(a_str)

        obs = env.step(action)
        obs_dict = obs.model_dump()
        r = obs.reward if obs.reward is not None else 0.0

        if verbose and (parse_fails == 0 or step <= 3):
            print(f"    Step {step:2d}: {a_str:28s} health={obs.system_health_score:.4f} r={r:+.4f}")

    elapsed = time.time() - t0
    grade = env.grade()
    env.close()

    return {
        "score": grade["score"], "breakdown": grade["breakdown"],
        "is_resolved": grade["is_resolved"], "steps_taken": grade["steps_taken"],
        "parse_fails": parse_fails, "fallback_pct": fallback_count / max(len(action_history), 1),
        "actions": action_history, "elapsed_s": round(elapsed, 2),
        "raw_outputs": raw_outputs[:3] if verbose else [],
    }


# ---------------------------------------------------------------------------
# Multi-episode runner
# ---------------------------------------------------------------------------

def run_multi_episode(runner_fn, task_name: str, episodes: int, **kwargs) -> Dict[str, Any]:
    """Run N episodes and return averaged results + per-episode scores."""
    results = []
    for ep in range(episodes):
        seed = 42 + ep
        r = runner_fn(task_name, seed=seed, **kwargs)
        results.append(r)

    scores = [r["score"] for r in results]
    avg_score = sum(scores) / len(scores)
    resolved_count = sum(1 for r in results if r["is_resolved"])

    # Average breakdown
    avg_breakdown = {}
    if results and "breakdown" in results[0]:
        for key in results[0]["breakdown"]:
            avg_breakdown[key] = round(
                sum(r["breakdown"].get(key, 0) for r in results) / len(results), 4
            )

    out = {
        "avg_score": round(avg_score, 4),
        "scores": scores,
        "resolved": f"{resolved_count}/{episodes}",
        "avg_breakdown": avg_breakdown,
        "episodes": episodes,
    }

    # Trained-model specific stats
    if "parse_fails" in results[0]:
        total_pf = sum(r["parse_fails"] for r in results)
        total_steps = sum(r["steps_taken"] for r in results)
        out["parse_fail_rate"] = round(total_pf / max(total_steps, 1), 4)
        out["avg_fallback_pct"] = round(
            sum(r["fallback_pct"] for r in results) / len(results), 4
        )
        out["avg_elapsed_s"] = round(
            sum(r.get("elapsed_s", 0) for r in results) / len(results), 2
        )

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate trained LoRA vs baselines")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter", default="trained_model_full_0p5b")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--task", default=None, help="Single task name or alias (easy/medium/hard)")
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per agent per task")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--no-model", action="store_true", help="Skip trained model eval")
    parser.add_argument("--sample", action="store_true",
                        help="Use sampling instead of deterministic decoding (for stress testing)")
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                args.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                args.device = "mps"
            else:
                args.device = "cpu"
        except ImportError:
            args.device = "cpu"

    # Select tasks
    tasks = [t for t in EVALUABLE_TASKS if t in EXPERT_STRATEGIES]
    if args.task:
        alias = {"easy": "single_service_failure", "medium": "cascading_failure", "hard": "hidden_root_cause"}
        selected = alias.get(args.task, args.task)
        tasks = [t for t in tasks if t == selected]
        if not tasks:
            print(f"Error: Unknown task '{args.task}'", file=sys.stderr)
            sys.exit(1)

    eps = args.episodes

    print("=" * 72)
    print("  POST-TRAINING EVALUATION — Trained LoRA vs All Baselines")
    print("=" * 72)
    print(f"  Tasks:    {', '.join(tasks)}")
    print(f"  Episodes: {eps} per agent per task")
    print(f"  Device:   {args.device}")
    print(f"  Adapter:  {args.adapter}")
    print(f"  Model:    {'SKIP' if args.no_model else args.base_model}")
    print(f"  Time:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # ── 1. Expert baselines (deterministic, 1 run) ──
    print("┌──────────────────────────────────────────────────────────────────────┐")
    print("│  EXPERT BASELINE (optimal action sequence, 1 run)                   │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    expert = {}
    for t in tasks:
        r = run_strategy(t, EXPERT_STRATEGIES[t], verbose=False, label="expert")
        expert[t] = {"avg_score": r["score"], "resolved": "1/1" if r["is_resolved"] else "0/1",
                      "breakdown": r["breakdown"]}
        icon = "✅" if r["is_resolved"] else "❌"
        print(f"  {t:<30s} {r['score']:.4f}  {icon}")

    # ── 2. Naive baselines (deterministic, 1 run) ──
    print()
    print("┌──────────────────────────────────────────────────────────────────────┐")
    print("│  NAIVE BASELINE (restart everything, 1 run)                         │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    naive = {}
    for t in tasks:
        r = run_strategy(t, NAIVE_STRATEGIES[t], verbose=False, label="naive")
        naive[t] = {"avg_score": r["score"], "resolved": "1/1" if r["is_resolved"] else "0/1",
                     "breakdown": r["breakdown"]}
        icon = "✅" if r["is_resolved"] else "❌"
        print(f"  {t:<30s} {r['score']:.4f}  {icon}")

    # ── 3. Heuristic baseline (multi-episode) ──
    print()
    print("┌──────────────────────────────────────────────────────────────────────┐")
    print(f"│  HEURISTIC BASELINE (rule-based, {eps} episodes)                       │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    heuristic = {}
    for t in tasks:
        r = run_multi_episode(run_heuristic_episode, t, eps)
        heuristic[t] = r
        print(f"  {t:<30s} {r['avg_score']:.4f}  resolved={r['resolved']}")

    # ── 4. Do-nothing floor ──
    print()
    print("┌──────────────────────────────────────────────────────────────────────┐")
    print("│  DO-NOTHING FLOOR (1 run)                                           │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    nothing = {}
    for t in tasks:
        td = get_task(t)
        strategy = [IncidentAction(action_type=ActionType.DO_NOTHING)] * td.max_steps
        r = run_strategy(t, strategy, verbose=False, label="nothing")
        nothing[t] = {"avg_score": r["score"]}
        print(f"  {t:<30s} {r['score']:.4f}")

    # ── 5. Trained model ──
    trained = {}
    if not args.no_model:
        print()
        print("┌──────────────────────────────────────────────────────────────────────┐")
        print(f"│  TRAINED MODEL ({eps} episodes per task)                               │")
        print("└──────────────────────────────────────────────────────────────────────┘")
        model, tokenizer = load_model(args.base_model, args.adapter, args.device)
        print()

        for t in tasks:
            if args.verbose:
                print(f"  ── {t} ──")
            r = run_multi_episode(
                run_trained_episode, t, eps, model=model, tokenizer=tokenizer,
                verbose=args.verbose, deterministic=not args.sample,
            )
            trained[t] = r
            pf = f"parse_fail={r.get('parse_fail_rate', 0):.0%}"
            fb = f"fallback={r.get('avg_fallback_pct', 0):.0%}"
            spd = f"{r.get('avg_elapsed_s', 0):.1f}s/ep"
            print(f"  {t:<30s} {r['avg_score']:.4f}  resolved={r['resolved']}  {pf}  {fb}  {spd}")

    # ── Comparison Table ──
    print()
    print("┌──────────────────────────────────────────────────────────────────────┐")
    print("│  COMPARISON TABLE                                                    │")
    print("└──────────────────────────────────────────────────────────────────────┘")
    hdr = f"  {'Task':<24s} {'Expert':>7s} {'Heuris':>7s} {'Trained':>8s} {'Naive':>7s} {'Nothing':>8s}  {'vs Heur':>10s}"
    print(hdr)
    print(f"  {'─'*24} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*8}  {'─'*10}")

    for t in tasks:
        e = expert[t]["avg_score"]
        h = heuristic[t]["avg_score"]
        n = naive[t]["avg_score"]
        d = nothing[t]["avg_score"]
        if t in trained:
            tr = trained[t]["avg_score"]
            gap = tr - h
            icon = '✅' if gap > 0 else '❌'
            g = f"{icon} {gap:+.3f}"
            print(f"  {t:<24s} {e:>7.4f} {h:>7.4f} {tr:>8.4f} {n:>7.4f} {d:>8.4f}  {g}")
        else:
            print(f"  {t:<24s} {e:>7.4f} {h:>7.4f} {'N/A':>8s} {n:>7.4f} {d:>8.4f}  {'---':>10s}")

    # Averages
    avg_e = sum(expert[t]["avg_score"] for t in tasks) / len(tasks)
    avg_h = sum(heuristic[t]["avg_score"] for t in tasks) / len(tasks)
    avg_n = sum(naive[t]["avg_score"] for t in tasks) / len(tasks)
    avg_d = sum(nothing[t]["avg_score"] for t in tasks) / len(tasks)
    print(f"  {'─'*24} {'─'*7} {'─'*7} {'─'*8} {'─'*7} {'─'*8}  {'─'*10}")

    if trained:
        avg_t = sum(trained[t]["avg_score"] for t in tasks if t in trained) / max(len(trained), 1)
        gap = avg_t - avg_h
        icon = '✅' if gap > 0 else '❌'
        g = f"{icon} {gap:+.3f}"
        print(f"  {'AVERAGE':<24s} {avg_e:>7.4f} {avg_h:>7.4f} {avg_t:>8.4f} {avg_n:>7.4f} {avg_d:>8.4f}  {g}")
    else:
        print(f"  {'AVERAGE':<24s} {avg_e:>7.4f} {avg_h:>7.4f} {'N/A':>8s} {avg_n:>7.4f} {avg_d:>8.4f}")

    # ── Breakdown (if trained) ──
    if trained and args.verbose:
        print()
        print("┌──────────────────────────────────────────────────────────────────────┐")
        print("│  SCORE BREAKDOWN (trained model, averaged)                           │")
        print("└──────────────────────────────────────────────────────────────────────┘")
        components = ["recovery", "efficiency", "diagnostics", "ordering", "memory"]
        hdr2 = f"  {'Task':<24s}" + "".join(f" {c:>11s}" for c in components)
        print(hdr2)
        print(f"  {'─'*24}" + " ───────────" * len(components))
        for t in tasks:
            if t in trained and "avg_breakdown" in trained[t]:
                bd = trained[t]["avg_breakdown"]
                vals = "".join(f" {bd.get(c, 0):>11.4f}" for c in components)
                print(f"  {t:<24s}{vals}")

    # ── Save results ──
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episodes": eps, "device": args.device,
        "expert": {t: {"score": expert[t]["avg_score"]} for t in tasks},
        "heuristic": {t: {"score": heuristic[t]["avg_score"]} for t in tasks},
        "naive": {t: {"score": naive[t]["avg_score"]} for t in tasks},
        "nothing": {t: {"score": nothing[t]["avg_score"]} for t in tasks},
    }
    if trained:
        output["trained"] = {}
        for t in tasks:
            if t in trained:
                output["trained"][t] = {
                    "avg_score": trained[t]["avg_score"],
                    "scores": trained[t]["scores"],
                    "resolved": trained[t]["resolved"],
                    "parse_fail_rate": trained[t].get("parse_fail_rate", 0),
                    "avg_fallback_pct": trained[t].get("avg_fallback_pct", 0),
                }

    out_path = results_dir / "evaluation_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  📊 Results saved to {out_path}")

    # ── Verdict ──
    print()
    print("=" * 72)
    if trained:
        resolved_total = sum(
            int(trained[t]["resolved"].split("/")[0]) for t in tasks if t in trained
        )
        total_eps = sum(
            int(trained[t]["resolved"].split("/")[1]) for t in tasks if t in trained
        )
        print(f"  TRAINED: {resolved_total}/{total_eps} episodes resolved | Avg score: {avg_t:.4f}")
        if avg_t > avg_h:
            print("  ✅ OUTPERFORMS heuristic baseline — ready for deployment!")
            ret = 0
        elif avg_t > avg_n:
            print("  🟡 Beats naive but not heuristic — more training recommended.")
            ret = 0
        elif avg_t > avg_d:
            print("  🟠 Beats do-nothing but underperforms naive — check prompt alignment.")
            ret = 1
        else:
            print("  ❌ UNDERPERFORMS all baselines — likely prompt mismatch or insufficient training.")
            ret = 1

        total_pf = sum(trained[t].get("parse_fail_rate", 0) for t in tasks if t in trained) / max(len(trained), 1)
        if total_pf > 0.3:
            print(f"  ⚠️  High parse failure rate ({total_pf:.0%}) — model may need more training steps.")
    else:
        print("  Baselines-only run complete. Run without --no-model to evaluate the adapter.")
        ret = 0
    print("=" * 72)
    sys.exit(ret)


if __name__ == "__main__":
    main()
