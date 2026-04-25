#!/usr/bin/env python3
"""
SFT Warm-Start: Supervised fine-tuning on expert trajectories before GRPO.

Generates (prompt, response) pairs from expert strategies across all tasks,
then runs a short SFT pass to teach the model:
  1. Valid JSON action format
  2. Reasonable action priors (inspect before fix, dependency ordering)

This eliminates the cold-start / mode-collapse problem in GRPO.

Usage:
  # Step 1: Generate dataset only
  python sft_warmstart.py --generate-only

  # Step 2: Full SFT (Colab/Kaggle GPU)
  python sft_warmstart.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 3 --batch-size 4 --lr 2e-5 \
    --use-lora --use-4bit \
    --output-dir sft_warmstart_adapter

  # Step 3: Then run GRPO starting from the SFT adapter
  python train_grpo.py \
    --model sft_warmstart_adapter \
    --steps 300 --num-generations 8 ...

Prerequisites:
  pip install trl>=0.15.0 transformers>=4.46.0 datasets peft bitsandbytes
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

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from evaluate import EXPERT_STRATEGIES
from train_grpo import build_obs_prompt


# ---------------------------------------------------------------------------
# All tasks to generate trajectories for
# ---------------------------------------------------------------------------

ALL_TASKS = list(EXPERT_STRATEGIES.keys())

# Additional heuristic strategies for more diverse training data
HEURISTIC_STRATEGIES: Dict[str, List[IncidentAction]] = {
    "single_service_failure": [
        IncidentAction(action_type=ActionType.INSPECT_METRICS, service_name="cache"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"),
    ],
    "cascading_failure": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="checkout"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"),
        IncidentAction(action_type=ActionType.INSPECT_METRICS, service_name="database"),
        IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="auth"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
    ],
    "hidden_root_cause": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="auth"),
        IncidentAction(action_type=ActionType.INSPECT_METRICS, service_name="auth"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="checkout"),
        IncidentAction(action_type=ActionType.ROLLBACK, service_name="auth"),
        IncidentAction(action_type=ActionType.CLEAR_CACHE),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
    ],
    "chaos_cascade": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="checkout"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="auth"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="notification"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="notification"),
    ],
    "multi_root_cause": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="auth"),
        IncidentAction(action_type=ActionType.INSPECT_METRICS, service_name="database"),
        IncidentAction(action_type=ActionType.ROLLBACK, service_name="auth"),
        IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.CLEAR_CACHE),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
    ],
}


# ---------------------------------------------------------------------------
# Generate SFT dataset from expert/heuristic trajectories
# ---------------------------------------------------------------------------

def action_to_json(action: IncidentAction) -> str:
    """Convert an IncidentAction to the exact JSON format the model should produce."""
    d: Dict[str, str] = {"action_type": action.action_type.value}
    if action.service_name:
        d["service_name"] = action.service_name
    return json.dumps(d)


def generate_trajectory_pairs(
    task_name: str,
    actions: List[IncidentAction],
    seed: int = 42,
    label: str = "expert",
) -> List[Dict[str, Any]]:
    """
    Run a trajectory and collect (prompt, response) pairs at each step.

    Returns list of dicts with:
      - prompt: conversational format [{role: user, content: ...}]
      - response: the correct JSON action string
      - task_name: for metadata
      - seed: for metadata
      - step: step number
      - label: expert or heuristic
    """
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)
    obs_dict = obs.model_dump()
    action_history: List[str] = []
    pairs = []

    for i, action in enumerate(actions):
        if obs.done:
            break

        step = i + 1
        prompt_text = build_obs_prompt(obs_dict, step, action_history)
        response_text = action_to_json(action)

        pairs.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "completion": [{"role": "assistant", "content": response_text}],
            "task_name": task_name,
            "seed": seed,
            "step": step,
            "label": label,
        })

        # Step environment and update history
        obs = env.step(action)
        obs_dict = obs.model_dump()
        a_str = action.action_type.value
        if action.service_name:
            a_str += f":{action.service_name}"
        action_history.append(a_str)

    env.close()
    return pairs


def build_sft_dataset(num_seeds: int = 5) -> List[Dict[str, Any]]:
    """
    Build the full SFT dataset from expert + heuristic trajectories.

    For each task × seed, we generate step-by-step (prompt, response) pairs
    from both expert and heuristic strategies. This gives the model diverse
    examples of correct actions at different environment states.
    """
    all_pairs = []

    strategies = {
        "expert": EXPERT_STRATEGIES,
        "heuristic": HEURISTIC_STRATEGIES,
    }

    for label, strategy_dict in strategies.items():
        for task_name in ALL_TASKS:
            if task_name not in strategy_dict:
                continue
            actions = strategy_dict[task_name]
            for s in range(num_seeds):
                seed = 42 + s
                pairs = generate_trajectory_pairs(
                    task_name, actions, seed=seed, label=label,
                )
                all_pairs.extend(pairs)

    return all_pairs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SFT Warm-Start for Incident Commander")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="sft_warmstart_adapter",
                        help="Output directory for SFT adapter")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of SFT training epochs")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate for SFT")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Number of seeds per task for trajectory generation")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA adapters")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Load in 4-bit quantization")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate and save the dataset, don't train")
    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # ── Step 1: Generate SFT dataset ──
    print("=" * 60)
    print("  SFT Warm-Start — Incident Commander")
    print("=" * 60)
    print(f"  Tasks: {', '.join(ALL_TASKS)}")
    print(f"  Seeds per task: {args.num_seeds}")
    print(f"  Strategies: expert + heuristic")
    print()

    print("Generating expert/heuristic trajectories...")
    all_pairs = build_sft_dataset(num_seeds=args.num_seeds)
    print(f"  ✅ Generated {len(all_pairs)} (prompt, response) pairs")

    # Count by task
    task_counts: Dict[str, int] = {}
    for p in all_pairs:
        task_counts[p["task_name"]] = task_counts.get(p["task_name"], 0) + 1
    for task, count in sorted(task_counts.items()):
        print(f"     {task}: {count} pairs")

    # Save dataset for inspection
    dataset_path = results_dir / "sft_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"  📁 Dataset saved to {dataset_path}")

    if args.generate_only:
        print("\n✅ Dataset generation complete. Use without --generate-only to train.")
        # Print a few examples
        print("\n── Example pairs ──")
        for p in all_pairs[:3]:
            prompt_preview = p["prompt"][0]["content"][:80] + "..."
            response = p["completion"][0]["content"]
            print(f"  [{p['task_name']}] step {p['step']}")
            print(f"    Prompt: {prompt_preview}")
            print(f"    Response: {response}")
            print()
        return

    # ── Step 2: Load model ──
    try:
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}", file=sys.stderr)
        print("Install: pip install trl>=0.15.0 transformers>=4.46.0 datasets peft bitsandbytes torch", file=sys.stderr)
        sys.exit(1)

    from train_grpo import load_model_and_tokenizer

    model, tokenizer, peft_config = load_model_and_tokenizer(
        model_name=args.model,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # ── Step 3: Build HF dataset ──
    # Format as conversational messages for SFTTrainer
    hf_rows = []
    for p in all_pairs:
        hf_rows.append({
            "messages": p["prompt"] + p["completion"],
        })
    dataset = Dataset.from_list(hf_rows)
    print(f"\n  HF Dataset: {len(dataset)} samples, columns={dataset.column_names}")

    # ── Step 4: Configure SFT ──
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    sft_config = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        max_seq_length=2048,
        gradient_accumulation_steps=2,
        warmup_ratio=0.1,
        report_to="none",
    )

    # ── Step 5: Train ──
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    print(f"\n{'=' * 60}")
    print(f"  Starting SFT warm-start ({args.epochs} epochs, {len(dataset)} samples)...")
    print(f"{'=' * 60}\n")

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    # ── Step 6: Save ──
    print(f"\nSaving adapter to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metrics
    metrics = {
        "train_result": {
            "global_step": train_result.global_step,
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics,
        },
        "config": {
            "model": args.model,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "num_seeds": args.num_seeds,
            "dataset_size": len(dataset),
            "use_lora": args.use_lora,
            "use_4bit": args.use_4bit,
        },
        "elapsed_seconds": round(elapsed, 1),
    }
    metrics_path = results_dir / "sft_training_log.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n{'=' * 60}")
    print(f"  ✅ SFT warm-start complete!")
    print(f"  📁 Adapter saved to: {args.output_dir}/")
    print(f"  📊 Metrics saved to: {metrics_path}")
    print(f"  ⏱️  Elapsed: {elapsed:.0f}s")
    print(f"  📈 Final loss: {train_result.training_loss:.4f}")
    print(f"")
    print(f"  Next step — run GRPO starting from this adapter:")
    print(f"  python train_grpo.py \\")
    print(f"    --model {args.output_dir} \\")
    print(f"    --steps 300 --num-generations 8 --num-seeds 5 \\")
    print(f"    --use-lora --use-4bit --gradient-checkpointing")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
