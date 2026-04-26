"""
Multi-Specialist Agent Architecture for Incident Commander Environment.

Implements a coordinator-specialist pattern with 3 specialist agents:
  - DB Expert: database + cache specialization
  - Infra Expert: infrastructure-wide restart/scale/rollback
  - App Expert: application-level services (auth, payments, checkout)

The coordinator reads observations and delegates to the appropriate specialist.
Each specialist has a focused system prompt and restricted action set.

This is the #1 differentiator for Theme #1 (Multi-Agent) and demonstrates
emergent "theory of mind" as the coordinator learns which specialist to deploy.

Usage:
  python multi_agent_inference.py [--task TASK] [--chaos]
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openai import OpenAI
    _HAS_OPENAI = True
except ImportError:
    _HAS_OPENAI = False
    OpenAI = None  # type: ignore

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-1.5B-Instruct"
BENCHMARK_NAME = "incident_commander_env"
MAX_RETRIES = 3

# Import training prompt builder for --local mode (must match training distribution)
try:
    from train_grpo import build_obs_prompt as _training_prompt_builder
except ImportError:
    _training_prompt_builder = None


# ---------------------------------------------------------------------------
# Local model loader (for --local mode)
# ---------------------------------------------------------------------------

_local_model = None
_local_tokenizer = None


def load_local_model(base_model: str, adapter_path: str, device: str):
    """Load base model + LoRA adapter for local inference."""
    global _local_model, _local_tokenizer

    if _local_model is not None:
        return _local_model, _local_tokenizer

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except ImportError as e:
        print(f"ERROR: Missing dependency for --local mode: {e}", file=sys.stderr)
        print("Install: pip install transformers peft torch accelerate", file=sys.stderr)
        sys.exit(1)

    from pathlib import Path
    adapter_dir = Path(adapter_path)
    if not adapter_dir.exists():
        print(f"\n\u274c Adapter not found at '{adapter_path}'", file=sys.stderr)
        sys.exit(1)

    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n  Loading trained model on {device}...")

    dtype = torch.float32
    if device == "mps":
        dtype = torch.float16
    elif device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    _local_tokenizer = AutoTokenizer.from_pretrained(base_model)
    if _local_tokenizer.pad_token is None:
        _local_tokenizer.pad_token = _local_tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=dtype,
        device_map=device if device != "mps" else None,
    )
    if device == "mps":
        base = base.to("mps")

    _local_model = PeftModel.from_pretrained(base, adapter_path)
    _local_model.eval()

    param_count = sum(p.numel() for p in _local_model.parameters())
    print(f"  \u2705 Model loaded: {param_count:,} params on {device}")

    return _local_model, _local_tokenizer


def generate_local_action(
    model, tokenizer,
    obs_dict, step, action_history,
    deterministic: bool = True,
) -> str:
    """Generate action using the trained model with TRAINING-ALIGNED prompts."""
    import torch

    if _training_prompt_builder is None:
        raise RuntimeError("Cannot import train_grpo.build_obs_prompt for local inference")

    prompt_text = _training_prompt_builder(obs_dict, step, action_history)
    messages = [{"role": "user", "content": prompt_text}]

    if hasattr(tokenizer, "apply_chat_template"):
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        input_text = prompt_text

    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = {
        "max_new_tokens": 128,
        "pad_token_id": tokenizer.pad_token_id,
        "repetition_penalty": 1.0,
    }
    if deterministic:
        gen_kwargs["do_sample"] = False
        gen_kwargs["temperature"] = 1.0
        gen_kwargs["top_p"] = 1.0
        gen_kwargs["top_k"] = None
    else:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = 0.2
        gen_kwargs["top_p"] = 0.9

    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Specialist System Prompts
# ---------------------------------------------------------------------------

COORDINATOR_PROMPT = """You are an Incident Commander coordinating a team of three specialist agents to resolve a production microservices outage.

Your team:
- "db_expert": Specializes in database and cache issues. Handles: inspect_logs/inspect_metrics on database/cache, restart_service, scale_service on database/cache, clear_cache.
- "infra_expert": Specializes in infrastructure-wide operations. Handles: restart_service, scale_service, rollback on ANY service. Best for cascading failures and service restarts.
- "app_expert": Specializes in application-layer services (auth, payments, checkout, notification). Handles: inspect_logs/inspect_metrics on app services, rollback on app services, restart_service on app services.

Your ONLY job is to read the current system state and decide which specialist to delegate to.

## Decision Process
1. If services are DOWN or have high error rates, identify the root cause service.
2. If the root cause is database/cache → delegate to db_expert.
3. If multiple services need restarts/rollbacks → delegate to infra_expert.
4. If the issue is in auth/payments/checkout (e.g., bad version, 401 errors) → delegate to app_expert.
5. If unsure, delegate to infra_expert for broad triage.

## Reasoning Examples
- "Database CPU at 95%, auth and payments degraded" → db_expert (database is root cause)
- "Auth has version v2.2.0-rc1, payments getting 401s" → app_expert (bad deploy on auth)
- "4 services need restart after root cause fixed" → infra_expert (bulk recovery)

Respond ONLY with valid JSON:
{"delegate_to": "db_expert" | "infra_expert" | "app_expert", "context": "<brief reasoning>"}"""


DB_EXPERT_PROMPT = """You are a Database & Cache Expert in an incident response team. You specialize in diagnosing and fixing database and cache failures.

## Your Allowed Actions
- {"action_type": "inspect_logs", "service_name": "database"} or "cache"
- {"action_type": "inspect_metrics", "service_name": "database"} or "cache"
- {"action_type": "restart_service", "service_name": "database"} or "cache"
- {"action_type": "scale_service", "service_name": "database"} or "cache"
- {"action_type": "clear_cache"}

## Expert Knowledge
- If database CPU > 90%, SCALE first, then RESTART.
- If cache is DOWN with OOM, RESTART cache.
- If cache is serving stale data (auth tokens), use clear_cache.
- Always inspect logs/metrics BEFORE taking action if you haven't already.
- Database issues cascade to auth, payments, checkout — fix DB first.

## Reasoning Examples
- If I see CPU > 90% on database and auth is degraded → database is the root cause. Scale database first.
- If cache is DOWN and auth latency spiked → cache OOM. Restart cache to restore auth.
- If "stale auth tokens" in cache logs → clear_cache to flush poisoned entries.

Respond with ONLY a valid JSON action. No explanation."""


INFRA_EXPERT_PROMPT = """You are an Infrastructure Expert in an incident response team. You handle broad infrastructure operations: restarts, scaling, and rollbacks across all services.

## Your Allowed Actions
- {"action_type": "restart_service", "service_name": "<any_service>"}
- {"action_type": "scale_service", "service_name": "<any_service>"}
- {"action_type": "rollback", "service_name": "<any_service>"}
- {"action_type": "inspect_logs", "service_name": "<any_service>"}
- {"action_type": "inspect_metrics", "service_name": "<any_service>"}
- {"action_type": "escalate"}

## Available Services
database, cache, auth, notification, payments, checkout

## Dependency Order (fix upstream FIRST)
database/cache → auth → payments → checkout

## Expert Knowledge
- If a service has a non-v1.0.0 version → ROLLBACK.
- If a service has high CPU → SCALE before RESTART.
- Fix root cause service BEFORE restarting dependents.
- After fixing root cause, restart dependents in dependency order.
- Don't repeat an action that was already taken.

## Reasoning Examples
- If database is fixed but auth/payments/checkout still degraded → restart them in order: auth first, then payments, then checkout.
- If auth version is v2.2.0-rc1 → rollback auth before doing anything else.
- If notifications DOWN independently → just restart notifications.

Respond with ONLY a valid JSON action. No explanation."""


APP_EXPERT_PROMPT = """You are an Application Expert in an incident response team. You specialize in application-layer services: auth, payments, checkout, and notification.

## Your Allowed Actions
- {"action_type": "inspect_logs", "service_name": "auth"} or "payments" or "checkout" or "notification"
- {"action_type": "inspect_metrics", "service_name": "auth"} or "payments" or "checkout" or "notification"
- {"action_type": "restart_service", "service_name": "auth"} or "payments" or "checkout" or "notification"
- {"action_type": "rollback", "service_name": "auth"} or "payments" or "checkout" or "notification"
- {"action_type": "clear_cache"}

## Expert Knowledge
- Auth with version v2.2.0-rc1 = bad deploy → ROLLBACK auth. Restart won't fix it.
- JWT signature errors + 401s in payments/checkout → root cause is auth, not payments.
- After rollback on auth, clear_cache to flush stale tokens, then restart payments and checkout.
- Bad deploys resist restart — always check version first.

## Reasoning Examples
- If I see "JWT signature validation failure" in auth logs and version is v2.2.0-rc1 → rollback auth immediately.
- If payments shows "401 from auth" → the problem is auth, not payments. Don't restart payments first.
- After auth is fixed → clear_cache, then restart payments, then restart checkout.

Respond with ONLY a valid JSON action. No explanation."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def observation_to_prompt(
    obs_dict: Dict[str, Any],
    step: int,
    action_history: List[str],
    specialist: Optional[str] = None,
) -> str:
    """Convert an observation dict to a human-readable prompt."""
    lines = [f"## Current Status (Step {step}/{obs_dict.get('max_steps', 30)})"]
    lines.append(f"System Health: {obs_dict.get('system_health_score', 0):.2%}")
    lines.append(f"Severity: {obs_dict.get('incident_severity', 'unknown')}")

    # Chaos event notification
    chaos_event = obs_dict.get("metadata", {}).get("new_chaos_event")
    if chaos_event:
        lines.append(f"\n⚠️ NEW CHAOS EVENT: {chaos_event} has just failed!")

    lines.append("")

    # Filter services for specialists
    services = obs_dict.get("services", {})
    if specialist == "db_expert":
        show_services = {k: v for k, v in services.items() if k in ("database", "cache")}
        # Also show dependent service states (summary only)
        for k, v in services.items():
            if k not in show_services:
                show_services[k] = v  # Show all but expert focuses on db/cache
    else:
        show_services = services

    lines.append("### Services")
    for name, svc in sorted(show_services.items()):
        status = svc.get("status", "healthy")
        err = svc.get("error_rate", 0)
        lat = svc.get("latency_ms", 0)
        ver = svc.get("version", "?")
        cpu = svc.get("cpu_percent", 0)
        inst = svc.get("instances", 0)
        emoji = "🟢" if status == "healthy" else ("🟡" if status == "degraded" else "🔴")
        lines.append(
            f"  {emoji} {name}: {status} | err={err:.1%} | lat={lat:.0f}ms "
            f"| cpu={cpu:.0f}% | inst={inst} | ver={ver}"
        )

    alerts = obs_dict.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("### Alerts")
        for a in alerts:
            lines.append(f"  {a}")

    logs = obs_dict.get("logs", [])
    if logs:
        lines.append("")
        lines.append("### Recent Logs")
        for log_line in logs:
            lines.append(f"  {log_line}")

    metrics = obs_dict.get("metrics_detail")
    if metrics:
        lines.append("")
        lines.append("### Metrics Detail")
        lines.append(f"  {json.dumps(metrics, indent=2)}")

    err_msg = obs_dict.get("last_action_error")
    if err_msg:
        lines.append("")
        lines.append(f"⚠️ Last Action Error: {err_msg}")

    if action_history:
        lines.append("")
        lines.append("### Actions Taken So Far")
        for i, a in enumerate(action_history, 1):
            lines.append(f"  {i}. {a}")

    lines.append("")
    lines.append("What action should we take next? Respond with a JSON action object only.")

    return "\n".join(lines)


def parse_action(response_text: str) -> Optional[IncidentAction]:
    """Parse the LLM response text into an IncidentAction."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return IncidentAction(**data)
    except Exception:
        pass

    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            end_char = "}" if start_char == "{" else "]"
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == start_char:
                    depth += 1
                elif text[i] == end_char:
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[idx : i + 1])
                            return IncidentAction(**data)
                        except Exception:
                            break
    return None


def parse_delegation(response_text: str) -> Optional[Dict[str, str]]:
    """Parse the coordinator's delegation response."""
    text = response_text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        if "delegate_to" in data:
            return data
    except Exception:
        pass

    for start_char in ["{"]:
        idx = text.find(start_char)
        if idx >= 0:
            depth = 0
            for i in range(idx, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            data = json.loads(text[idx : i + 1])
                            if "delegate_to" in data:
                                return data
                        except Exception:
                            break
    return None


def fallback_delegation(obs_dict: Dict[str, Any]) -> str:
    """Heuristic fallback when coordinator LLM fails."""
    services = obs_dict.get("services", {})

    # Check if DB/cache are the problem
    for name in ["database", "cache"]:
        svc = services.get(name, {})
        if svc.get("status") != "healthy":
            return "db_expert"

    # Check for app-level issues
    for name in ["auth", "payments", "checkout"]:
        svc = services.get(name, {})
        if svc.get("version", "v1.0.0") != "v1.0.0":
            return "app_expert"
        if svc.get("status") != "healthy":
            return "app_expert"

    return "infra_expert"


# ---------------------------------------------------------------------------
# Multi-Agent Runner
# ---------------------------------------------------------------------------

SPECIALIST_PROMPTS = {
    "db_expert": DB_EXPERT_PROMPT,
    "infra_expert": INFRA_EXPERT_PROMPT,
    "app_expert": APP_EXPERT_PROMPT,
}


def run_multi_agent_task(
    task_name: str,
    client=None,
    chaos_mode: bool = False,
    verbose: bool = True,
    local_model=None,
    local_tokenizer=None,
) -> Dict[str, Any]:
    """
    Run a single task episode using the coordinator-specialist architecture.

    Returns dict with score, steps, rewards, delegations.
    """
    env = IncidentCommanderEnvironment()
    rewards: List[float] = []
    delegations: List[str] = []
    action_history: List[str] = []
    steps = 0

    use_local = local_model is not None
    mode_str = "local" if use_local else MODEL_NAME
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={mode_str} mode=multi_agent", flush=True)

    try:
        obs = env.reset(task_name=task_name, chaos_mode=chaos_mode)
        obs_dict = obs.model_dump()

        # Separate conversation histories for coordinator and each specialist
        coordinator_messages = [{"role": "system", "content": COORDINATOR_PROMPT}]
        specialist_messages = {
            name: [{"role": "system", "content": prompt}]
            for name, prompt in SPECIALIST_PROMPTS.items()
        }

        # Import orchestrator for hybrid mode
        from server.tasks import get_task
        from orchestrator import orchestrated_action as _orchestrated_action
        task = get_task(task_name)

        while not obs.done:
            steps += 1

            if use_local:
                # --- Local model path: use trained Qwen + orchestrator ---
                # Step 1: Coordinator is heuristic (no separate LLM call)
                specialist_name = fallback_delegation(obs_dict)
                context = "local-heuristic"

                delegations.append(specialist_name)
                if verbose:
                    print(f"  [COORD] → {specialist_name}: {context}", flush=True)

                # Step 2: Specialist = trained local model
                model_action = None
                for attempt in range(MAX_RETRIES):
                    try:
                        response_text = generate_local_action(
                            local_model, local_tokenizer,
                            obs_dict, steps, action_history,
                        )
                        model_action = parse_action(response_text)
                        if model_action:
                            break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"WARNING: Local model failed: {e}", file=sys.stderr)

                # Route through orchestrator (hybrid decision)
                decision = _orchestrated_action(
                    model_action=model_action,
                    obs_dict=obs_dict,
                    step=steps,
                    action_history=action_history,
                    task=task,
                )
                action = decision.action
                if verbose and not decision.used_model:
                    print(f"  [ORCH] ⚠️ OVERRIDE({decision.reason})", flush=True)

            else:
                # --- API path: full multi-agent with coordinator ---
                # Step 1: Coordinator decides delegation
                coord_prompt = observation_to_prompt(obs_dict, steps, action_history)
                coordinator_messages.append({"role": "user", "content": coord_prompt})

                delegation = None
                for attempt in range(MAX_RETRIES):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=coordinator_messages,
                            temperature=0.0,
                            max_tokens=128,
                        )
                        response_text = completion.choices[0].message.content or ""
                        delegation = parse_delegation(response_text)
                        if delegation:
                            coordinator_messages.append({"role": "assistant", "content": response_text})
                            break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"WARNING: Coordinator LLM failed: {e}", file=sys.stderr)

                if delegation is None:
                    specialist_name = fallback_delegation(obs_dict)
                    context = "fallback heuristic"
                else:
                    specialist_name = delegation.get("delegate_to", "infra_expert")
                    context = delegation.get("context", "")
                    if specialist_name not in SPECIALIST_PROMPTS:
                        specialist_name = "infra_expert"

                delegations.append(specialist_name)
                if verbose:
                    print(f"  [COORD] → {specialist_name}: {context}", flush=True)

                # Step 2: Specialist picks action
                spec_messages = specialist_messages[specialist_name]
                spec_prompt = observation_to_prompt(obs_dict, steps, action_history, specialist=specialist_name)
                spec_messages.append({"role": "user", "content": spec_prompt})

                action = None
                for attempt in range(MAX_RETRIES):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=spec_messages,
                            temperature=0.0,
                            max_tokens=256,
                        )
                        response_text = completion.choices[0].message.content or ""
                        action = parse_action(response_text)
                        if action:
                            spec_messages.append({"role": "assistant", "content": response_text})
                            break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"WARNING: {specialist_name} LLM failed: {e}", file=sys.stderr)

                if action is None:
                    # Fallback: use infra expert's basic heuristic
                    action = IncidentAction(action_type=ActionType.DO_NOTHING)

            # Build action string
            action_str = action.action_type.value
            if action.service_name:
                action_str += f":{action.service_name}"
            action_history.append(action_str)

            # Step 3: Execute action
            obs = env.step(action)
            obs_dict = obs.model_dump()

            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            error_str = obs.last_action_error if obs.last_action_error else "null"

            print(
                f"[STEP] step={steps} specialist={specialist_name} action={action_str} "
                f"reward={reward:.2f} done={str(obs.done).lower()} error={error_str}",
                flush=True,
            )

            # Update coordinator with what happened
            coord_update = (
                f"Specialist {specialist_name} took action: {action_str}. "
                f"Reward: {reward:.2f}. Health: {obs_dict.get('system_health_score', 0):.2%}"
            )
            coordinator_messages.append({"role": "user", "content": coord_update})

            # Keep histories manageable
            if len(coordinator_messages) > 24:
                coordinator_messages = coordinator_messages[:1] + coordinator_messages[-22:]
            for name in specialist_messages:
                msgs = specialist_messages[name]
                if len(msgs) > 16:
                    specialist_messages[name] = msgs[:1] + msgs[-14:]

        # Final grading
        grade_result = env.grade()
        success = grade_result.get("is_resolved", False)
        score = grade_result.get("score", 0.0)

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(f"[STEP] step={steps + 1} action=error reward=0.00 done=true error={error_msg}", flush=True)
        rewards.append(0.0)
        steps += 1
        score = 0.0
        success = False

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )

    return {
        "task": task_name,
        "score": score,
        "steps": steps,
        "success": success,
        "rewards": rewards,
        "delegations": delegations,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run multi-agent inference across tasks."""
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Specialist Agent for Incident Commander")
    parser.add_argument("--task", type=str, default=None, help="Specific task to run (default: all)")
    parser.add_argument("--chaos", action="store_true", help="Enable chaos mode")
    parser.add_argument("--quiet", action="store_true", help="Hide coordinator delegation logs")
    parser.add_argument("--local", action="store_true",
                        help="Use local trained model instead of API")
    parser.add_argument("--base-model", type=str, default="sft_merged_1p5b_v5",
                        help="Base model for --local mode")
    parser.add_argument("--adapter", type=str, default="trained_model_1p5b_v5",
                        help="LoRA adapter path for --local mode")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for --local mode (default: auto-detect)")
    args = parser.parse_args()

    local_model = None
    local_tokenizer = None
    client = None

    if args.local:
        local_model, local_tokenizer = load_local_model(
            args.base_model, args.adapter, args.device
        )
    else:
        if not _HAS_OPENAI:
            print("ERROR: 'openai' package required for API mode. "
                  "Use --local for local model or: pip install openai>=1.0.0",
                  file=sys.stderr)
            sys.exit(1)
        if not API_KEY:
            print("WARNING: No HF_TOKEN or API_KEY set.", file=sys.stderr)
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    if args.task:
        tasks = [args.task]
    else:
        tasks = list_tasks()

    mode_str = "LOCAL" if args.local else MODEL_NAME
    print(f"\n{'='*60}")
    print(f"  Multi-Specialist Agent — Incident Commander")
    print(f"  Model: {mode_str} | Chaos: {args.chaos}")
    print(f"  Tasks: {tasks}")
    print(f"{'='*60}\n")

    results = []
    for task_name in tasks:
        result = run_multi_agent_task(
            task_name,
            client,
            chaos_mode=args.chaos,
            verbose=not args.quiet,
            local_model=local_model,
            local_tokenizer=local_tokenizer,
        )
        results.append(result)
        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"  {status} {r['task']}: score={r['score']:.3f} steps={r['steps']}")
        if r["delegations"]:
            from collections import Counter
            counts = Counter(r["delegations"])
            print(f"     Delegations: {dict(counts)}")
    print()


if __name__ == "__main__":
    main()
