"""
Baseline Agent Benchmarking for Incident Commander Environment.

Runs 4 agent types across all tasks and saves reward data for comparison:
  1. RandomAgent    — picks random valid action each step
  2. HeuristicAgent — inspect first, then fix root cause (rule-based)
  3. LLMAgent       — existing inference.py agent (requires API key)
  4. TrainedAgent   — loads LoRA adapter and generates actions locally

Usage:
  python run_baselines.py [--episodes 20] [--skip-llm] [--tasks all]
  python run_baselines.py --episodes 5 --tasks all --skip-llm \
      --trained-base-model Qwen/Qwen2.5-0.5B-Instruct \
      --trained-adapter-dir trained_model_full_0p5b \
      --trained-max-new-tokens 64
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks
from server.services import ALL_SERVICES

# Import the EXACT training prompt builder so inference matches training
from train_grpo import build_obs_prompt


# ---------------------------------------------------------------------------
# Agent Implementations
# ---------------------------------------------------------------------------

class RandomAgent:
    """Picks a random valid action each step."""

    name = "random"

    def act(self, obs_dict: Dict[str, Any], action_history: List[str]) -> IncidentAction:
        action_type = random.choice([
            ActionType.INSPECT_LOGS,
            ActionType.INSPECT_METRICS,
            ActionType.RESTART_SERVICE,
            ActionType.SCALE_SERVICE,
            ActionType.ROLLBACK,
            ActionType.CLEAR_CACHE,
            ActionType.DO_NOTHING,
        ])
        service = None
        if action_type in (
            ActionType.INSPECT_LOGS, ActionType.INSPECT_METRICS,
            ActionType.RESTART_SERVICE, ActionType.SCALE_SERVICE,
            ActionType.ROLLBACK,
        ):
            service = random.choice(ALL_SERVICES)
        return IncidentAction(action_type=action_type, service_name=service)


class HeuristicAgent:
    """
    Rule-based agent that follows a structured incident response procedure:
    1. Inspect unhealthy services (logs first)
    2. Check for version mismatches → rollback
    3. Restart root cause / most degraded service
    4. Clear cache if relevant
    5. Restart remaining unhealthy services
    """

    name = "heuristic"

    def act(self, obs_dict: Dict[str, Any], action_history: List[str]) -> IncidentAction:
        services = obs_dict.get("services", {})

        # Build sets of what we've already done
        inspected = set()
        restarted = set()
        scaled = set()
        rolled_back = set()
        cleared_cache = False

        for a in action_history:
            if a.startswith("inspect_logs:") or a.startswith("inspect_metrics:"):
                inspected.add(a.split(":", 1)[1])
            elif a.startswith("restart_service:"):
                restarted.add(a.split(":", 1)[1])
            elif a.startswith("scale_service:"):
                scaled.add(a.split(":", 1)[1])
            elif a.startswith("rollback:"):
                rolled_back.add(a.split(":", 1)[1])
            elif a == "clear_cache":
                cleared_cache = True

        # Rank services by severity (worst first)
        ranked = []
        for name, svc in services.items():
            status = svc.get("status", "healthy")
            if status == "down":
                score = 0.0
            elif status == "degraded":
                score = 0.5
            else:
                score = 1.0
            ranked.append((score, name, svc))
        ranked.sort()

        # Phase 1: Inspect unhealthy services we haven't inspected
        for _, name, svc in ranked:
            if svc.get("status") != "healthy" and name not in inspected:
                return IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name=name)

        # Phase 2: Version mismatch → rollback
        for _, name, svc in ranked:
            version = svc.get("version", "v1.0.0")
            if version != "v1.0.0" and name not in rolled_back:
                return IncidentAction(action_type=ActionType.ROLLBACK, service_name=name)

        # Phase 3: Clear cache if not done and cache-dependent issues exist
        if not cleared_cache:
            for _, name, svc in ranked:
                if name in ("auth", "checkout") and svc.get("status") != "healthy":
                    return IncidentAction(action_type=ActionType.CLEAR_CACHE)

        # Phase 4: Scale high-CPU services
        for _, name, svc in ranked:
            if svc.get("cpu_percent", 0) > 85 and name not in scaled:
                return IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name=name)

        # Phase 5: Restart unhealthy services (dependency order)
        dep_order = ["database", "cache", "notification", "auth", "payments", "checkout"]
        for name in dep_order:
            svc = services.get(name, {})
            if svc.get("status") != "healthy" and name not in restarted:
                return IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name=name)

        # Phase 6: Try restarting anything still broken
        for _, name, svc in ranked:
            if svc.get("status") != "healthy":
                return IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name=name)

        return IncidentAction(action_type=ActionType.DO_NOTHING)


class TrainedAgent:
    """
    Trained LoRA agent — loads base model + adapter and generates actions.

    CRITICAL: Uses the EXACT same prompt format as train_grpo.py (build_obs_prompt)
    to match the training distribution. Prompt mismatch was the #1 cause of
    underperformance in the initial evaluation.
    """

    name = "trained"

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
        adapter_dir: str = "trained_model_full_0p5b",
        device: str = "auto",
        max_new_tokens: int = 64,
    ):
        self._base_model_name = base_model
        self._adapter_dir = adapter_dir
        self._max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        self._device = device
        self._loaded = False
        self._step_count = 0

    def _ensure_loaded(self):
        """Lazy-load the model on first call."""
        if self._loaded:
            return

        adapter_path = Path(self._adapter_dir)
        if not adapter_path.exists():
            print(f"\n❌ Adapter not found at '{self._adapter_dir}'", file=sys.stderr)
            print(f"   Get trained_model_full_0p5b/ from your teammate.", file=sys.stderr)
            print(f"   Falling back to heuristic agent.\n", file=sys.stderr)
            self._loaded = True
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as e:
            print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
            print("Install: pip install transformers peft torch accelerate", file=sys.stderr)
            self._loaded = True
            return

        # Auto-detect device
        if self._device == "auto":
            if torch.cuda.is_available():
                self._device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = "mps"
            else:
                self._device = "cpu"

        print(f"\n  Loading trained model on {self._device}...")

        # Precision
        dtype = torch.float32
        if self._device == "mps":
            dtype = torch.float16
        elif self._device == "cuda":
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self._tokenizer = AutoTokenizer.from_pretrained(self._base_model_name)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self._base_model_name,
            torch_dtype=dtype,
            device_map=self._device if self._device != "mps" else None,
        )
        if self._device == "mps":
            base = base.to("mps")

        self._model = PeftModel.from_pretrained(base, self._adapter_dir)
        self._model.eval()

        print(f"  ✅ Model loaded: {sum(p.numel() for p in self._model.parameters()):,} params")
        self._loaded = True

    def act(self, obs_dict: Dict[str, Any], action_history: List[str]) -> IncidentAction:
        """Generate action using the trained model with TRAINING-ALIGNED prompts."""
        self._ensure_loaded()
        self._step_count += 1

        # Fallback if model failed to load
        if self._model is None:
            return HeuristicAgent().act(obs_dict, action_history)

        import torch

        # CRITICAL: Use build_obs_prompt from train_grpo.py — EXACT same format
        # the model was trained on. This is the #1 fix for underperformance.
        step = len(action_history) + 1
        prompt_text = build_obs_prompt(obs_dict, step, action_history)

        # Training used: [{"role": "user", "content": prompt_text}]
        # No system prompt — the model was trained with user-role-only prompts
        messages = [{"role": "user", "content": prompt_text}]

        # Apply chat template (same as TRL GRPOTrainer does internally)
        if hasattr(self._tokenizer, "apply_chat_template"):
            input_text = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt_text

        inputs = self._tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        # Decode only new tokens
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON action from response
        action = self._parse_action(response)
        if action is None:
            # Fallback to heuristic on parse failure
            return HeuristicAgent().act(obs_dict, action_history)

        return action

    def _parse_action(self, response_text: str) -> Optional[IncidentAction]:
        """Parse model output into an IncidentAction."""
        text = response_text.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            data = json.loads(text)
            return IncidentAction(**data)
        except Exception:
            pass

        # Try to extract JSON from the text
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

    def reset_step_count(self):
        """Reset step counter between episodes."""
        self._step_count = 0


# ---------------------------------------------------------------------------
# Episode Runner
# ---------------------------------------------------------------------------

def run_episode(
    agent: Any,
    task_name: str,
    seed: int = None,
    chaos_mode: bool = False,
) -> Dict[str, Any]:
    """Run a single episode and return results."""
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed, chaos_mode=chaos_mode)
    obs_dict = obs.model_dump()

    rewards = []
    action_history = []
    steps = 0

    # Reset step counter for trained agent
    if hasattr(agent, "reset_step_count"):
        agent.reset_step_count()

    while not obs.done:
        steps += 1
        action = agent.act(obs_dict, action_history)

        action_str = action.action_type.value
        if action.service_name:
            action_str += f":{action.service_name}"
        action_history.append(action_str)

        obs = env.step(action)
        obs_dict = obs.model_dump()
        reward = obs.reward if obs.reward is not None else 0.0
        rewards.append(reward)

    grade = env.grade()
    env.close()

    return {
        "score": grade.get("score", 0.0),
        "is_resolved": grade.get("is_resolved", False),
        "steps": steps,
        "rewards": rewards,
        "cumulative_reward": sum(rewards),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run baseline benchmarks")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per agent per task")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM agent (no API key needed)")
    parser.add_argument("--tasks", type=str, default="core", help="'core' for 3 original tasks, 'all' for everything")
    parser.add_argument("--chaos", action="store_true", help="Enable chaos mode")

    # Trained model options
    parser.add_argument("--trained-base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model for trained agent")
    parser.add_argument("--trained-adapter-dir", type=str, default="trained_model_full_0p5b",
                        help="Path to LoRA adapter directory")
    parser.add_argument("--trained-device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for trained agent")
    parser.add_argument("--trained-max-new-tokens", type=int, default=64,
                        help="Max new tokens for trained agent generation")
    parser.add_argument("--skip-trained", action="store_true",
                        help="Skip trained agent evaluation")

    args = parser.parse_args()

    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    if args.tasks == "core":
        tasks = ["single_service_failure", "cascading_failure", "hidden_root_cause"]
    elif args.tasks == "all":
        tasks = list_tasks()
    else:
        tasks = [args.tasks]

    agents: List[Any] = [RandomAgent(), HeuristicAgent()]

    if not args.skip_trained:
        agents.append(TrainedAgent(
            base_model=args.trained_base_model,
            adapter_dir=args.trained_adapter_dir,
            device=args.trained_device,
            max_new_tokens=args.trained_max_new_tokens,
        ))

    # Optionally add LLM agent
    if not args.skip_llm:
        try:
            from inference import run_task
            # LLM agent is handled separately via inference.py
            pass
        except ImportError:
            print("WARNING: Could not import inference module, skipping LLM agent", file=sys.stderr)

    all_results: Dict[str, Dict[str, List[float]]] = {}

    for agent in agents:
        print(f"\n{'='*50}")
        print(f"  Running: {agent.name}")
        print(f"{'='*50}")

        agent_results: Dict[str, List[float]] = {}

        for task_name in tasks:
            print(f"\n  Task: {task_name}")
            scores = []

            for ep in range(args.episodes):
                # Use different seed for random_incident, fixed seed for others
                seed = ep if task_name == "random_incident" else None

                result = run_episode(
                    agent,
                    task_name,
                    seed=seed,
                    chaos_mode=args.chaos,
                )
                scores.append(result["score"])

                status = "✅" if result["is_resolved"] else "❌"
                if (ep + 1) % 5 == 0 or ep == 0:
                    print(
                        f"    Episode {ep+1:3d}/{args.episodes}: "
                        f"score={result['score']:.3f} {status} "
                        f"steps={result['steps']:2d} "
                        f"cum_reward={result['cumulative_reward']:.3f}"
                    )

            agent_results[task_name] = scores
            avg = sum(scores) / len(scores)
            print(f"  → {task_name} avg score: {avg:.3f}")

        all_results[agent.name] = agent_results

    # Save results
    output_path = results_dir / "baseline_rewards.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to {output_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"  BASELINE SUMMARY")
    print(f"{'='*70}")
    header = f"  {'Agent':<15}"
    for task in tasks:
        short = task[:12]
        header += f" {short:>14}"
    header += f" {'AVERAGE':>10}"
    print(header)
    print(f"  {'-'*65}")

    for agent_name, results in all_results.items():
        row = f"  {agent_name:<15}"
        task_avgs = []
        for task in tasks:
            scores = results.get(task, [])
            avg = sum(scores) / len(scores) if scores else 0.0
            task_avgs.append(avg)
            row += f" {avg:>14.3f}"
        overall_avg = sum(task_avgs) / len(task_avgs) if task_avgs else 0.0
        row += f" {overall_avg:>10.3f}"
        print(row)

    print()


if __name__ == "__main__":
    main()