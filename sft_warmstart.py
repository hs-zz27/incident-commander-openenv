#!/usr/bin/env python3
"""
SFT Warm-Start: Supervised fine-tuning on expert trajectories before GRPO.

This script solves the cold-start problem in GRPO for small language models.
Without SFT, a 0.5B model produces mostly unparseable outputs, giving GRPO
a flat reward landscape (all -0.1) and causing mode collapse.

The SFT phase teaches the model two things:
  1. Output format — valid JSON matching {"action_type": "...", "service_name": "..."}
  2. Action priors — inspect before fix, dependency ordering, rollback bad versions

Pipeline:
  SFT warm-start (this script) → merge adapter → GRPO fine-tuning (train_grpo.py)

Usage:
  # Generate dataset only (inspect before committing GPU time)
  python sft_warmstart.py --generate-only

  # Full SFT on Colab T4 / Kaggle P100 (~10-15 min)
  python sft_warmstart.py \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --epochs 3 --batch-size 4 --lr 2e-5 \
    --use-lora --use-4bit --gradient-checkpointing \
    --output-dir sft_warmstart_adapter

  # Then GRPO from the merged SFT model
  python train_grpo.py \
    --model sft_merged_model \
    --steps 300 --num-generations 8 ...

Prerequisites:
  pip install -e ".[train]"
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from evaluate import EXPERT_STRATEGIES
from train_grpo import build_obs_prompt


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_TASKS = list(EXPERT_STRATEGIES.keys())

# Heuristic strategies: slightly different action orderings that still resolve
# the incident. This teaches the model that there are multiple valid approaches.
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
# Trajectory generation
# ---------------------------------------------------------------------------

def action_to_json(action: IncidentAction) -> str:
    """Serialize action to the exact JSON format we want the model to learn."""
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
    Roll out a trajectory through the environment and emit one
    (prompt, completion) pair per step.

    Each pair captures:
      - The *exact* observation the model would see at inference time
        (via build_obs_prompt, same builder used in train_grpo.py)
      - The correct JSON action as the target completion
      - The reward the action produced (for data quality inspection)
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

        # Step the environment BEFORE recording reward
        obs = env.step(action)
        obs_dict = obs.model_dump()
        reward = obs.reward if obs.reward is not None else 0.0

        # Build action string for history (same format as inference.py)
        a_str = action.action_type.value
        if action.service_name:
            a_str += f":{action.service_name}"
        action_history.append(a_str)

        pairs.append({
            "prompt": [{"role": "user", "content": prompt_text}],
            "completion": [{"role": "assistant", "content": response_text}],
            "task_name": task_name,
            "seed": seed,
            "step": step,
            "label": label,
            "reward": round(reward, 4),
            "health_after": round(obs.system_health_score, 4),
        })

    # Record episode score for data quality audit
    grade = env.grade()
    for p in pairs:
        p["episode_score"] = round(grade["score"], 4)
    env.close()
    return pairs


def build_sft_dataset(num_seeds: int = 5) -> List[Dict[str, Any]]:
    """
    Build the full SFT dataset from expert + heuristic trajectories.

    Design rationale:
      - Two strategy types (expert, heuristic) teach the model that
        multiple action orderings can be correct.
      - Multiple seeds per task expose the model to stochastic
        variations in the environment state.
      - Step-by-step pairs mean the model sees observations from
        early, mid, and late episode phases — not just initial states.
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

    # Shuffle to avoid task-clustering in batches
    random.seed(42)
    random.shuffle(all_pairs)
    return all_pairs


def print_dataset_stats(pairs: List[Dict[str, Any]]) -> None:
    """Print data quality statistics for inspection before training."""
    print(f"\n  ── Dataset Statistics ──")
    print(f"  Total pairs: {len(pairs)}")

    # By task
    task_counts = Counter(p["task_name"] for p in pairs)
    for task, count in sorted(task_counts.items()):
        scores = [p["episode_score"] for p in pairs if p["task_name"] == task]
        avg_score = sum(scores) / len(scores) if scores else 0
        print(f"    {task}: {count} pairs  (episode_score={avg_score:.3f})")

    # By label
    label_counts = Counter(p["label"] for p in pairs)
    for label, count in sorted(label_counts.items()):
        print(f"    {label}: {count} pairs")

    # Action distribution (what the model will learn to output)
    action_counts = Counter()
    for p in pairs:
        action_data = json.loads(p["completion"][0]["content"])
        action_counts[action_data["action_type"]] += 1
    print(f"\n  Action distribution in training data:")
    for action, count in action_counts.most_common():
        pct = count / len(pairs) * 100
        print(f"    {action}: {count} ({pct:.0f}%)")

    # Reward statistics
    rewards = [p["reward"] for p in pairs]
    print(f"\n  Per-step reward: min={min(rewards):.3f} max={max(rewards):.3f} "
          f"mean={sum(rewards)/len(rewards):.3f}")


# ---------------------------------------------------------------------------
# Post-SFT validation: quick JSON parse-rate check
# ---------------------------------------------------------------------------

def validate_sft_model(model, tokenizer, num_samples: int = 10) -> float:
    """
    Generate actions from the SFT'd model on held-out observations
    and measure JSON parse success rate.

    Returns parse rate (0.0 to 1.0). Above 0.8 means SFT worked.
    """
    import torch

    parse_ok = 0
    tasks = ALL_TASKS[:3]  # Quick check on first 3 tasks

    for task_name in tasks:
        env = IncidentCommanderEnvironment()
        obs = env.reset(task_name=task_name, seed=99)
        prompt_text = build_obs_prompt(obs.model_dump(), 1, [])
        env.close()

        messages = [{"role": "user", "content": prompt_text}]
        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        else:
            input_text = prompt_text

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=64, temperature=0.3,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Try parsing as valid action
        try:
            data = json.loads(text)
            if "action_type" in data:
                parse_ok += 1
                print(f"    ✅ {task_name}: {text}")
            else:
                print(f"    ⚠️  {task_name}: valid JSON but no action_type: {text[:60]}")
        except json.JSONDecodeError:
            print(f"    ❌ {task_name}: unparseable: {text[:60]}")

    rate = parse_ok / len(tasks)
    return rate


# ---------------------------------------------------------------------------
# Adapter merge: SFT LoRA → full merged model for GRPO
# ---------------------------------------------------------------------------

def merge_adapter_to_full_model(
    base_model_name: str,
    adapter_path: str,
    output_path: str,
) -> None:
    """
    Merge LoRA adapter into the base model weights and save as a
    standalone HuggingFace model.

    This is critical for the SFT→GRPO handoff:
      - GRPO needs to apply its OWN fresh LoRA on top of the model
      - If we pass the SFT adapter dir to GRPO, it would try to
        stack two LoRAs (messy, unstable, often crashes)
      - Instead, we merge SFT weights into the base → GRPO gets a
        clean model with SFT knowledge baked in
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print(f"\n  Merging adapter into base model...")
    print(f"    Base:    {base_model_name}")
    print(f"    Adapter: {adapter_path}")
    print(f"    Output:  {output_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Load in float16 for merge (no quantization — we need full precision weights)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="cpu",  # Merge on CPU to avoid VRAM pressure
    )

    # Apply and merge the LoRA
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()

    # Save the merged model
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    # Calculate size
    total_bytes = sum(
        f.stat().st_size for f in Path(output_path).rglob("*") if f.is_file()
    )
    print(f"    ✅ Merged model saved ({total_bytes / 1e6:.0f} MB)")
    print(f"    This model has SFT knowledge baked in — ready for GRPO.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SFT Warm-Start for Incident Commander",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect dataset before training
  python sft_warmstart.py --generate-only --num-seeds 5

  # Full SFT on Colab T4 (16 GB VRAM)
  python sft_warmstart.py \\
    --model Qwen/Qwen2.5-0.5B-Instruct \\
    --epochs 3 --batch-size 4 --lr 2e-5 \\
    --use-lora --use-4bit --gradient-checkpointing \\
    --output-dir sft_warmstart_adapter

  # Then run GRPO from the merged model
  python train_grpo.py \\
    --model sft_merged_model \\
    --steps 300 --num-generations 8 --use-lora ...
        """,
    )

    # Model
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base HuggingFace model to fine-tune")
    parser.add_argument("--output-dir", type=str, default="sft_warmstart_adapter",
                        help="Output directory for SFT LoRA adapter")
    parser.add_argument("--merged-output-dir", type=str, default="sft_merged_model",
                        help="Output directory for merged model (base + SFT adapter)")

    # Training
    parser.add_argument("--epochs", type=int, default=3,
                        help="SFT epochs (2-4 is usually sufficient)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device training batch size")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate (2e-5 is standard for SFT)")
    parser.add_argument("--num-seeds", type=int, default=5,
                        help="Seeds per task for trajectory generation")

    # Memory
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA adapters (recommended)")
    parser.add_argument("--use-4bit", action="store_true",
                        help="4-bit quantization (recommended for T4)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Gradient checkpointing (saves ~1GB VRAM)")

    # Workflow
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate and inspect dataset, no training")
    parser.add_argument("--skip-merge", action="store_true",
                        help="Skip the adapter merge step after training")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip post-SFT JSON parse-rate validation")

    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    # ==================================================================
    # Phase 1: Generate SFT dataset
    # ==================================================================
    print("=" * 64)
    print("  SFT Warm-Start — Incident Commander RL Pipeline")
    print("=" * 64)
    print(f"  Base model:  {args.model}")
    print(f"  Tasks:       {len(ALL_TASKS)} ({', '.join(ALL_TASKS)})")
    print(f"  Seeds/task:  {args.num_seeds}")
    print(f"  Strategies:  expert + heuristic")
    print()

    print("Phase 1: Generating expert/heuristic trajectories...")
    all_pairs = build_sft_dataset(num_seeds=args.num_seeds)
    print_dataset_stats(all_pairs)

    # Save dataset for reproducibility
    dataset_path = results_dir / "sft_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(all_pairs, f, indent=2)
    print(f"\n  📁 Dataset saved to {dataset_path}")

    if args.generate_only:
        print("\n" + "=" * 64)
        print("  ✅ Dataset generation complete.")
        print("  Run without --generate-only to train.")
        print("=" * 64)
        # Show examples
        print("\n  ── Sample training pairs ──")
        for p in all_pairs[:3]:
            prompt_preview = p["prompt"][0]["content"].split("\n")[0]
            response = p["completion"][0]["content"]
            print(f"  [{p['label']}] {p['task_name']} step {p['step']}  "
                  f"reward={p['reward']:+.3f}")
            print(f"    → {response}")
        return

    # ==================================================================
    # Phase 2: Load model and train
    # ==================================================================
    try:
        from trl import SFTTrainer, SFTConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"\nERROR: Missing dependency: {e}", file=sys.stderr)
        print("Install: pip install -e '.[train]'", file=sys.stderr)
        sys.exit(1)

    from train_grpo import load_model_and_tokenizer

    print(f"\nPhase 2: Loading model...")
    model, tokenizer, peft_config = load_model_and_tokenizer(
        model_name=args.model,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # Build HuggingFace Dataset in conversational format
    hf_rows = [{"messages": p["prompt"] + p["completion"]} for p in all_pairs]
    dataset = Dataset.from_list(hf_rows)
    print(f"  HF Dataset: {len(dataset)} rows, columns={dataset.column_names}")

    # ==================================================================
    # Phase 3: Configure and run SFT
    # ==================================================================
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    # SFT hyperparameters — tuned for short warm-start, not full fine-tuning
    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_steps=50,
        save_total_limit=2,
        bf16=use_bf16,
        fp16=not use_bf16 and torch.cuda.is_available(),
        gradient_accumulation_steps=2,
        report_to="none",
        packing=False,
    )
    # max_seq_length was removed in TRL >= 0.17; try with it, fall back without
    try:
        sft_config = SFTConfig(max_seq_length=2048, **sft_kwargs)
    except TypeError:
        sft_config = SFTConfig(**sft_kwargs)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": sft_config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = SFTTrainer(**trainer_kwargs)

    total_train_samples = len(dataset) * args.epochs
    effective_batch = args.batch_size * sft_config.gradient_accumulation_steps
    est_steps = total_train_samples // effective_batch

    print(f"\nPhase 3: SFT Training")
    print(f"  {'─' * 50}")
    print(f"  Epochs:           {args.epochs}")
    print(f"  Dataset size:     {len(dataset)}")
    print(f"  Effective batch:  {effective_batch}")
    print(f"  Estimated steps:  {est_steps}")
    print(f"  Precision:        {'bf16' if use_bf16 else 'fp16'}")
    print(f"  LoRA:             {'yes' if peft_config else 'no'}")
    print(f"  {'─' * 50}\n")

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    # Save adapter
    print(f"\n  Saving adapter to {args.output_dir}/...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save metrics
    metrics = {
        "phase": "sft_warmstart",
        "train_result": {
            "global_step": train_result.global_step,
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics,
        },
        "log_history": trainer.state.log_history,
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

    print(f"\n  SFT Results:")
    print(f"    Loss:    {train_result.training_loss:.4f}")
    print(f"    Steps:   {train_result.global_step}")
    print(f"    Time:    {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"    Metrics: {metrics_path}")

    # ==================================================================
    # Phase 4: Post-SFT validation
    # ==================================================================
    if not args.skip_validation:
        print(f"\nPhase 4: Validating SFT model (JSON parse-rate check)...")
        # Re-load the saved adapter for clean validation
        del trainer  # Free VRAM
        import gc; gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        val_model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        from peft import PeftModel
        val_model = PeftModel.from_pretrained(val_model, args.output_dir)
        val_model.eval()

        parse_rate = validate_sft_model(val_model, tokenizer)
        print(f"\n  JSON parse rate: {parse_rate:.0%}")
        if parse_rate >= 0.8:
            print("  ✅ SFT successful — model produces valid JSON.")
        elif parse_rate >= 0.5:
            print("  🟡 Partial success — consider more epochs or higher LR.")
        else:
            print("  ❌ SFT may need tuning — low parse rate. Try --epochs 5 --lr 5e-5")

        del val_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ==================================================================
    # Phase 5: Merge adapter into base model for GRPO
    # ==================================================================
    if not args.skip_merge and args.use_lora:
        print(f"\nPhase 5: Merging SFT adapter into base model...")
        merge_adapter_to_full_model(
            base_model_name=args.model,
            adapter_path=args.output_dir,
            output_path=args.merged_output_dir,
        )
    elif not args.use_lora:
        # Full fine-tune — the output dir IS the merged model
        args.merged_output_dir = args.output_dir

    # ==================================================================
    # Done
    # ==================================================================
    grpo_model_path = args.merged_output_dir if (args.use_lora and not args.skip_merge) else args.output_dir

    print(f"\n{'=' * 64}")
    print(f"  ✅ SFT Warm-Start Complete!")
    print(f"  {'─' * 60}")
    print(f"  Adapter:       {args.output_dir}/")
    if args.use_lora and not args.skip_merge:
        print(f"  Merged model:  {args.merged_output_dir}/")
    print(f"  Training log:  {metrics_path}")
    print(f"  Loss:          {train_result.training_loss:.4f}")
    print(f"  Time:          {elapsed:.0f}s")
    print(f"  {'─' * 60}")
    print(f"")
    print(f"  Next step — GRPO fine-tuning:")
    print(f"  python train_grpo.py \\")
    print(f"    --model {grpo_model_path} \\")
    print(f"    --steps 300 --batch-size 2 --lr 1e-5 \\")
    print(f"    --num-generations 8 --num-seeds 5 \\")
    print(f"    --use-lora --use-4bit --gradient-checkpointing \\")
    print(f"    --output-dir trained_model_v2")
    print(f"{'=' * 64}")


if __name__ == "__main__":
    main()
