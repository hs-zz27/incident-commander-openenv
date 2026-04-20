"""
GRPO Training Script for Incident Commander Environment.

Uses HuggingFace TRL to fine-tune a small language model via GRPO (Group Relative 
Policy Optimization) on our environment.

This is a MANDATORY hackathon requirement.

Target model: Qwen2.5-1.5B-Instruct (small, trains fast on free compute)
Training: 200-500 steps with reward curve logging every 50 steps

Usage (in Bangalore with compute credits):
  python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300

Prerequisites:
  pip install trl transformers torch accelerate peft bitsandbytes
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# These imports will be available on the training machine
# Commented out for local dev where GPU libs aren't installed
# from trl import GRPOTrainer, GRPOConfig
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks


# ---------------------------------------------------------------------------
# Environment wrapper for TRL
# ---------------------------------------------------------------------------

class IncidentCommanderRewardFunction:
    """
    Reward function wrapper for TRL GRPO training.
    
    Takes model-generated text, parses it as an action, steps the environment,
    and returns the reward.
    """

    def __init__(self, tasks: List[str] = None):
        self.tasks = tasks or ["single_service_failure", "cascading_failure", "hidden_root_cause"]
        self.env = IncidentCommanderEnvironment()
        self._current_task_idx = 0
        self._episode_step = 0
        self._obs_dict = None
        self._action_history = []

    def reset(self, task_idx: int = None):
        """Reset to a new episode."""
        if task_idx is not None:
            self._current_task_idx = task_idx % len(self.tasks)
        task = self.tasks[self._current_task_idx]
        obs = self.env.reset(task_name=task)
        self._obs_dict = obs.model_dump()
        self._episode_step = 0
        self._action_history = []
        return self._obs_dict

    def get_reward(self, response_text: str) -> float:
        """
        Parse model response as action, step environment, return reward.
        
        Returns negative reward for unparseable actions.
        """
        import json as _json

        try:
            # Try to parse as JSON action
            text = response_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()

            data = _json.loads(text)
            action = IncidentAction(**data)
        except Exception:
            return -0.1  # Penalize unparseable responses

        # Step environment
        obs = self.env.step(action)
        self._obs_dict = obs.model_dump()
        self._episode_step += 1

        action_str = action.action_type.value
        if action.service_name:
            action_str += f":{action.service_name}"
        self._action_history.append(action_str)

        return obs.reward if obs.reward is not None else 0.0


# ---------------------------------------------------------------------------
# Training Loop (GRPO)
# ---------------------------------------------------------------------------

def build_training_prompts(env_wrapper: IncidentCommanderRewardFunction) -> List[str]:
    """Generate training prompts from environment observations."""
    prompts = []

    for task_idx in range(len(env_wrapper.tasks)):
        obs_dict = env_wrapper.reset(task_idx)
        
        # Build observation prompt
        prompt = build_obs_prompt(obs_dict, 1, [])
        prompts.append(prompt)

    return prompts


def build_obs_prompt(obs_dict: Dict, step: int, action_history: List[str]) -> str:
    """Convert observation to training prompt."""
    lines = [
        "You are an SRE Incident Commander. Diagnose and fix the following incident.",
        f"Step {step}/{obs_dict.get('max_steps', 30)}",
        f"System Health: {obs_dict.get('system_health_score', 0):.2%}",
        "",
        "Services:"
    ]

    for name, svc in sorted(obs_dict.get("services", {}).items()):
        status = svc.get("status", "unknown")
        lines.append(
            f"  {name}: {status} | err={svc.get('error_rate', 0):.1%} "
            f"| lat={svc.get('latency_ms', 0):.0f}ms | ver={svc.get('version', '?')}"
        )

    if action_history:
        lines.append("")
        lines.append("Previous actions: " + ", ".join(action_history))

    lines.append("")
    lines.append("Respond with a JSON action: {\"action_type\": \"...\", \"service_name\": \"...\"}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Incident Commander")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="trained_model",
                        help="Output directory for trained model")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log metrics every N steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without GPU (uses mock training)")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"  GRPO Training — Incident Commander")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE — no GPU training, testing environment wrapper only\n")
        
        reward_fn = IncidentCommanderRewardFunction()
        tasks = reward_fn.tasks

        training_log = []
        for step in range(1, min(args.steps, 10) + 1):
            task_idx = step % len(tasks)
            obs_dict = reward_fn.reset(task_idx)
            
            # Simulate a model response
            test_actions = [
                '{"action_type": "inspect_logs", "service_name": "database"}',
                '{"action_type": "inspect_metrics", "service_name": "cache"}',
                '{"action_type": "restart_service", "service_name": "database"}',
            ]
            
            total_reward = 0
            for action_text in test_actions:
                r = reward_fn.get_reward(action_text)
                total_reward += r

            training_log.append({
                "step": step,
                "mean_reward": total_reward / len(test_actions),
                "task": tasks[task_idx],
            })

            if step % args.log_every == 0 or step == 1:
                print(f"  Step {step}: mean_reward={total_reward/len(test_actions):.4f} task={tasks[task_idx]}")

        # Save training log
        log_path = results_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"\n✅ Training log saved to {log_path}")
        print("✅ Dry run complete. Run on GPU with --no-dry-run for actual training.")
        return

    # --- Actual GRPO Training ---
    # This section runs on the Bangalore compute cluster with GPU
    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
    except ImportError:
        print("ERROR: TRL/transformers not installed. Required for training.", file=sys.stderr)
        print("Install with: pip install trl transformers torch accelerate peft", file=sys.stderr)
        sys.exit(1)

    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Set up reward function
    reward_fn = IncidentCommanderRewardFunction()

    def compute_rewards(responses: List[str]) -> List[float]:
        """Compute rewards for a batch of model responses."""
        rewards = []
        for resp in responses:
            r = reward_fn.get_reward(resp)
            rewards.append(r)
        return rewards

    # GRPO config
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.log_every,
        save_steps=args.steps,  # Save at end
        gradient_accumulation_steps=2,
        bf16=True,
    )

    # Build training prompts
    prompts = build_training_prompts(reward_fn)

    print(f"\nStarting GRPO training for {args.steps} steps...")
    
    # Note: The actual TRL GRPO integration will be adapted in Bangalore
    # based on the exact TRL version available on the compute cluster.
    # The structure is ready — just need to connect the reward_fn
    # to the GRPOTrainer's reward computation pipeline.

    print("⚠️ Full GRPO training requires GPU compute. Use --dry-run for testing.")
    print(f"Training will be completed in Bangalore with compute credits.")


if __name__ == "__main__":
    main()
