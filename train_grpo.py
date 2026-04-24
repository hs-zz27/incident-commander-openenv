"""
GRPO Training Script for Incident Commander Environment.

Uses HuggingFace TRL to fine-tune a small language model via GRPO (Group Relative
Policy Optimization) on our environment.

This is a MANDATORY hackathon requirement.

Target model: Qwen2.5-1.5B-Instruct (small, trains fast on free compute)
Training: 200-500 steps with reward curve logging every 50 steps

Usage (GPU required):
  python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300

Usage (Colab T4, memory-safe):
  python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 50 \
      --batch-size 1 --use-lora --use-4bit --gradient-checkpointing

Usage (dry-run, no GPU):
  python train_grpo.py --dry-run --steps 5

Prerequisites:
  pip install trl>=0.15.0 transformers>=4.46.0 torch accelerate>=1.0.0 \
      peft>=0.14.0 bitsandbytes>=0.45.0 datasets>=3.0.0
"""

from __future__ import annotations

import argparse
import copy
import json
import locale
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks


# ---------------------------------------------------------------------------
# Observation → prompt (includes all Tier 2 fields — Audit Fix #14)
# ---------------------------------------------------------------------------

def build_obs_prompt(obs_dict: Dict, step: int, action_history: List[str]) -> str:
    """Convert observation to training prompt with ALL observation fields."""
    lines = [
        "You are an SRE Incident Commander. Diagnose and fix the following incident.",
        f"Step {step}/{obs_dict.get('max_steps', 30)}",
        f"System Health: {obs_dict.get('system_health_score', 0):.2%}",
        f"Severity: {obs_dict.get('incident_severity', 'unknown')}",
        f"Escalation Tier: {obs_dict.get('escalation_tier', 1)}/4",
        "",
        "Services:",
    ]

    for name, svc in sorted(obs_dict.get("services", {}).items()):
        status = svc.get("status", "unknown")
        emoji = "🟢" if status == "healthy" else ("🟡" if status == "degraded" else "🔴")
        lines.append(
            f"  {emoji} {name}: {status} | err={svc.get('error_rate', 0):.1%} "
            f"| lat={svc.get('latency_ms', 0):.0f}ms | cpu={svc.get('cpu_percent', 0):.0f}% "
            f"| ver={svc.get('version', '?')}"
        )

    # Alerts
    alerts = obs_dict.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("Alerts:")
        for a in alerts:
            lines.append(f"  {a}")

    # Services at risk (Tier 2 field)
    at_risk = obs_dict.get("services_at_risk", [])
    if at_risk:
        lines.append("")
        lines.append(f"⚠️ Services at risk of degradation: {', '.join(at_risk)}")

    # Runbook memory (Tier 2 field)
    runbook = obs_dict.get("runbook_memory", [])
    if runbook:
        lines.append("")
        lines.append("📖 Runbook memory (past similar incidents):")
        for entry in runbook:
            lines.append(
                f"  - {entry.get('incident_type', '?')}: "
                f"fix=[{', '.join(entry.get('fix_sequence', []))}] "
                f"score={entry.get('score', 0):.2f} "
                f"({entry.get('episodes_ago', '?')} episodes ago)"
            )

    # Log quality warning (Tier 2 field)
    metadata = obs_dict.get("metadata", {})
    log_quality = metadata.get("log_quality")
    if log_quality and log_quality != "full":
        lines.append("")
        lines.append(f"⚠️ Log quality: {log_quality} — logs may be incomplete or misleading")

    # Logs from last inspect
    logs = obs_dict.get("logs", [])
    if logs:
        lines.append("")
        lines.append("Recent Logs:")
        for log_line in logs[:10]:  # Cap to avoid prompt explosion
            lines.append(f"  {log_line}")

    if action_history:
        lines.append("")
        lines.append("Previous actions: " + ", ".join(action_history))

    lines.append("")
    lines.append('Respond with a JSON action: {"action_type": "...", "service_name": "..."}')

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Episode rollout for reward computation (Audit Fix #3 + #4)
# ---------------------------------------------------------------------------

def rollout_episode(
    task_name: str,
    actions_text: List[str],
    seed: Optional[int] = None,
) -> float:
    """
    Run a COMPLETE episode rollout and return the FINAL episode score.
    
    Each call creates a fresh environment — no shared state (Audit Fix #3).
    Returns grade()["score"] (0.0-1.0), not per-step reward (Audit Fix #4).
    
    Args:
        task_name: Which task to run.
        actions_text: List of JSON action strings from the model.
        seed: Optional seed for reproducibility.
    
    Returns:
        Episode score in [0.0, 1.0].
    """
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)

    for action_text in actions_text:
        if obs.done:
            break

        try:
            text = action_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            data = json.loads(text)
            action = IncidentAction(**data)
        except Exception:
            # Unparseable action → do_nothing
            action = IncidentAction(action_type=ActionType.DO_NOTHING)

        obs = env.step(action)

    # Return FINAL episode score, not per-step reward (Audit Fix #4)
    grade = env.grade()
    env.close()
    return grade["score"]


def compute_single_action_reward(
    task_name: str,
    obs_dict: Dict,
    action_text: str,
    action_history: List[str],
    seed: Optional[int] = None,
) -> float:
    """
    Compute reward for a single action by running a full rollout.
    
    Replays the action history up to this point, then executes the new action,
    then uses a heuristic fallback to complete the episode.
    Returns the final episode grade (0.0-1.0).
    
    This ensures GRPO gets episode-level rewards that are comparable
    across different completions from the same prompt.
    """
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, seed=seed)

    # Replay history
    for past_action_str in action_history:
        if obs.done:
            break
        parts = past_action_str.split(":", 1)
        action_type = parts[0]
        service_name = parts[1] if len(parts) > 1 else None
        try:
            action = IncidentAction(action_type=action_type, service_name=service_name)
        except Exception:
            action = IncidentAction(action_type=ActionType.DO_NOTHING)
        obs = env.step(action)

    # Execute the NEW action
    if not obs.done:
        try:
            text = action_text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            data = json.loads(text)
            action = IncidentAction(**data)
        except Exception:
            return -0.1  # Parse failure penalty

        obs = env.step(action)

    # Complete episode with do_nothing to get final score
    while not obs.done:
        obs = env.step(IncidentAction(action_type=ActionType.DO_NOTHING))

    grade = env.grade()
    env.close()
    return grade["score"]


# ---------------------------------------------------------------------------
# GRPO Reward Function Wrapper (Audit Fix #3 + #4)
# ---------------------------------------------------------------------------

class IncidentCommanderRewardFunction:
    """
    Reward function wrapper for TRL GRPO training.
    
    CRITICAL: Each reward computation runs a FRESH, INDEPENDENT rollout.
    No shared environment state between completions (Audit Fix #3).
    Returns EPISODE-LEVEL score, not per-step reward (Audit Fix #4).
    """

    TASKS = ["single_service_failure", "cascading_failure", "hidden_root_cause"]

    def __init__(self):
        self._current_task_idx = 0
        self._seed = 42

    def next_task(self) -> str:
        """Cycle through tasks."""
        task = self.TASKS[self._current_task_idx % len(self.TASKS)]
        self._current_task_idx += 1
        return task

    def get_initial_obs(self, task_name: str) -> Dict:
        """Get initial observation for a task (for prompt building)."""
        env = IncidentCommanderEnvironment()
        obs = env.reset(task_name=task_name, seed=self._seed)
        obs_dict = obs.model_dump()
        env.close()
        return obs_dict

    def score_completions(
        self,
        task_name: str,
        completions: List[str],
        action_history: List[str],
    ) -> List[float]:
        """
        Score multiple GRPO completions for the same prompt.
        
        Each completion is scored via an independent full rollout.
        Returns list of episode-level scores (0.0-1.0).
        """
        scores = []
        for completion in completions:
            score = compute_single_action_reward(
                task_name=task_name,
                obs_dict={},  # Not used in current implementation
                action_text=completion,
                action_history=action_history,
                seed=self._seed,  # Same seed = same starting state
            )
            scores.append(score)
        return scores


# ---------------------------------------------------------------------------
# TRL-compatible reward function (P0 fix: uses explicit metadata, no guessing)
# ---------------------------------------------------------------------------

def incident_reward_func(completions, task_name, seed, **kwargs):
    """
    TRL GRPOTrainer-compatible reward function.

    Called by the trainer for each batch of completions. Receives extra
    dataset columns (task_name, seed) via **kwargs — no string matching.

    Args:
        completions: List of model outputs. Each is a list of message dicts
                     e.g. [{"role": "assistant", "content": "..."}].
        task_name:   List of task name strings from the dataset.
        seed:        List of seed ints from the dataset.
        **kwargs:    Any other dataset columns (unused).

    Returns:
        List of float rewards, one per completion.
    """
    rewards = []
    for completion, tn, s in zip(completions, task_name, seed):
        # Extract text content from the completion message(s)
        if isinstance(completion, list) and len(completion) > 0:
            # TRL sends completions as list of message dicts
            text = completion[0].get("content", "") if isinstance(completion[0], dict) else str(completion[0])
        else:
            text = str(completion)

        score = compute_single_action_reward(
            task_name=tn,
            obs_dict={},
            action_text=text,
            action_history=[],
            seed=int(s),
        )
        rewards.append(float(score))
    return rewards


# ---------------------------------------------------------------------------
# Dataset builder (P0 fix: explicit task_name + seed metadata per sample)
# ---------------------------------------------------------------------------

def build_training_dataset(reward_fn: IncidentCommanderRewardFunction, num_seeds: int = 3):
    """
    Build a HuggingFace Dataset with prompt + explicit task metadata.

    Each row contains:
      - prompt:    Conversational format [{"role": "user", "content": "..."}]
      - task_name: Explicit task identifier for reward routing
      - seed:      Seed used to generate this scenario

    Multiple seeds per task diversify the training distribution.
    """
    from datasets import Dataset

    rows = []
    for task_name in reward_fn.TASKS:
        for s in range(num_seeds):
            seed_val = s + 42
            env = IncidentCommanderEnvironment()
            obs = env.reset(task_name=task_name, seed=seed_val)
            prompt_text = build_obs_prompt(obs.model_dump(), 1, [])
            env.close()
            rows.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "task_name": task_name,
                "seed": seed_val,
            })

    return Dataset.from_list(rows)


# ---------------------------------------------------------------------------
# Memory-aware model loading (P1 fix: LoRA, 4-bit, gradient checkpointing)
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name, use_lora, use_4bit, gradient_checkpointing):
    """
    Load model and tokenizer with memory-aware configuration.

    Returns:
        (model, tokenizer, peft_config) — peft_config is None if LoRA is off.
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch

    print(f"\nLoading model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {"device_map": "auto"}

    # Precision: bf16 if supported, else fp16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model_kwargs["torch_dtype"] = torch.bfloat16
        print("  Precision: bfloat16")
    else:
        model_kwargs["torch_dtype"] = torch.float16
        print("  Precision: float16")

    # 4-bit quantization via bitsandbytes
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
                bnb_4bit_use_double_quant=True,
            )
            print("  Quantization: 4-bit NF4")
        except ImportError:
            print("  ⚠️ bitsandbytes not installed — skipping 4-bit quantization")

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("  Gradient checkpointing: enabled")

    # LoRA adapter
    peft_config = None
    if use_lora:
        try:
            from peft import LoraConfig
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
                task_type="CAUSAL_LM",
            )
            print("  LoRA: r=16, alpha=32")
        except ImportError:
            print("  ⚠️ peft not installed — skipping LoRA")

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    return model, tokenizer, peft_config


# ---------------------------------------------------------------------------
# Training Loop (GRPO)
# ---------------------------------------------------------------------------

def build_training_prompts(reward_fn: IncidentCommanderRewardFunction) -> List[Dict]:
    """Generate training prompt + task pairs from environment observations."""
    prompts = []

    for task in reward_fn.TASKS:
        obs_dict = reward_fn.get_initial_obs(task)
        prompt = build_obs_prompt(obs_dict, 1, [])
        prompts.append({"prompt": prompt, "task": task})

    return prompts


# ---------------------------------------------------------------------------
# Runtime preflight / platform guards
# ---------------------------------------------------------------------------

def _force_utf8_locale_for_trl_on_windows() -> None:
    """
    Work around TRL import issues on Windows CP1252 locales.

    Some TRL files are read with default locale encoding at import time.
    On certain Windows setups this causes UnicodeDecodeError unless UTF-8
    mode is enabled before process start. This patch forces UTF-8 encoding
    lookup in-process as a fallback.
    """
    if platform.system().lower() != "windows":
        return

    preferred = locale.getpreferredencoding(False)
    if preferred.lower() == "utf-8":
        return

    # Best-effort in-process fallback for libraries that call locale APIs.
    locale.getpreferredencoding = lambda do_setlocale=True: "utf-8"  # type: ignore[assignment]
    if hasattr(locale, "getencoding"):
        locale.getencoding = lambda: "utf-8"  # type: ignore[assignment]

    print(
        "⚠️ Windows non-UTF8 locale detected; applying UTF-8 compatibility mode for TRL import.",
        file=sys.stderr,
    )
    print(
        "   Recommended in PowerShell: $env:PYTHONUTF8='1'",
        file=sys.stderr,
    )


def _preflight_training_environment(allow_cpu: bool) -> None:
    """Validate Python/GPU prerequisites before expensive model loading."""
    py = sys.version_info
    if py.major == 3 and py.minor >= 13:
        print(
            "⚠️ Python 3.13 detected. Training ecosystem support is less mature on 3.13.",
            file=sys.stderr,
        )
        print(
            "   Recommended for stable GPU training: Python 3.10 or 3.11.",
            file=sys.stderr,
        )

    try:
        import torch
    except Exception as e:
        raise RuntimeError(
            f"PyTorch import failed: {e}. Install torch before non-dry training."
        ) from e

    cuda_ok = torch.cuda.is_available()
    if not cuda_ok and not allow_cpu:
        raise RuntimeError(
            "CUDA GPU not detected. Non-dry GRPO training is expected to run on GPU. "
            "Use --allow-cpu only for debug, or run on Colab/RunPod/Lambda."
        )

    if cuda_ok:
        print(f"GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Running without CUDA GPU (--allow-cpu). This is only suitable for debug.")


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Incident Commander")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct",
                        help="HuggingFace model name or local checkpoint path")
    parser.add_argument("--steps", type=int, default=300, help="Training steps")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="trained_model",
                        help="Output directory for trained model/adapter")
    parser.add_argument("--log-every", type=int, default=50,
                        help="Log metrics every N steps")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test without GPU (uses mock training)")

    # Memory-aware options (P1)
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA adapters for memory-efficient training")
    parser.add_argument("--use-4bit", action="store_true",
                        help="Load model in 4-bit quantization (requires bitsandbytes)")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce VRAM usage")

    # GRPO-specific options
    parser.add_argument("--num-generations", type=int, default=4,
                        help="Number of completions per prompt (GRPO group size)")
    parser.add_argument("--num-seeds", type=int, default=3,
                        help="Number of seeds per task for dataset diversity")

    # Checkpointing options (P1)
    parser.add_argument("--save-steps", type=int, default=50,
                        help="Save checkpoint every N steps")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None,
                        help="Path to checkpoint directory to resume training from")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="Allow non-dry training on CPU for debugging (very slow)")

    args = parser.parse_args()

    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)

    print(f"{'='*60}")
    print(f"  GRPO Training — Incident Commander")
    print(f"  Model: {args.model}")
    print(f"  Steps: {args.steps}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    if args.use_lora:
        print(f"  LoRA: enabled")
    if args.use_4bit:
        print(f"  4-bit quantization: enabled")
    if args.gradient_checkpointing:
        print(f"  Gradient checkpointing: enabled")
    print(f"  Generations per prompt: {args.num_generations}")
    print(f"  Save every: {args.save_steps} steps")
    if args.resume_from_checkpoint:
        print(f"  Resuming from: {args.resume_from_checkpoint}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE — no GPU training, testing environment wrapper only\n")
        
        reward_fn = IncidentCommanderRewardFunction()

        training_log = []
        for step in range(1, min(args.steps, 10) + 1):
            task = reward_fn.next_task()
            
            # Simulate K=4 model completions for GRPO
            mock_completions = [
                '{"action_type": "inspect_logs", "service_name": "database"}',
                '{"action_type": "restart_service", "service_name": "cache"}',
                '{"action_type": "do_nothing"}',
                '{"action_type": "inspect_metrics", "service_name": "auth"}',
            ]
            
            # Score all completions independently (Audit Fix #3)
            scores = reward_fn.score_completions(
                task_name=task,
                completions=mock_completions,
                action_history=[],
            )

            mean_score = sum(scores) / len(scores)
            best_score = max(scores)
            
            training_log.append({
                "step": step,
                "task": task,
                "mean_score": round(mean_score, 4),
                "best_score": round(best_score, 4),
                "scores": [round(s, 4) for s in scores],
            })

            if step % args.log_every == 0 or step == 1:
                print(
                    f"  Step {step}: task={task} "
                    f"mean_score={mean_score:.4f} best={best_score:.4f} "
                    f"scores={[round(s,3) for s in scores]}"
                )

        # Save training log
        log_path = results_dir / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(training_log, f, indent=2)
        print(f"\n✅ Training log saved to {log_path}")
        print("✅ Dry run complete — reward function produces independent episode-level scores.")
        print("✅ Run on GPU with actual model for GRPO training.")
        return

    # -----------------------------------------------------------------------
    # Full GRPO Training (GPU required)
    # -----------------------------------------------------------------------
    _force_utf8_locale_for_trl_on_windows()

    try:
        from trl import GRPOTrainer, GRPOConfig
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from datasets import Dataset
        import torch
    except ImportError as e:
        print(f"ERROR: Missing training dependency: {e}", file=sys.stderr)
        print(
            "Install with: pip install trl>=0.15.0 transformers>=4.46.0 "
            "torch accelerate>=1.0.0 peft>=0.14.0 bitsandbytes>=0.45.0 datasets>=3.0.0",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        _preflight_training_environment(allow_cpu=args.allow_cpu)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    # --- Step 1: Load model + tokenizer (memory-aware) ---
    model, tokenizer, peft_config = load_model_and_tokenizer(
        model_name=args.model,
        use_lora=args.use_lora,
        use_4bit=args.use_4bit,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # --- Step 2: Build training dataset with explicit metadata ---
    reward_fn = IncidentCommanderRewardFunction()
    print(f"\nBuilding training dataset ({len(reward_fn.TASKS)} tasks × {args.num_seeds} seeds)...")
    dataset = build_training_dataset(reward_fn, num_seeds=args.num_seeds)
    print(f"  Dataset size: {len(dataset)} samples")
    print(f"  Columns: {dataset.column_names}")

    # --- Step 3: Configure GRPO ---
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    config = GRPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=1,
        max_steps=args.steps,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.log_every,
        save_steps=args.save_steps,
        gradient_accumulation_steps=2,
        bf16=use_bf16,
        fp16=not use_bf16,
        num_generations=args.num_generations,
        max_completion_length=256,
        log_level="info",
        report_to="none",
    )

    # --- Step 4: Instantiate GRPOTrainer ---
    print(f"\nInitializing GRPOTrainer...")
    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "reward_funcs": [incident_reward_func],
        "args": config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
    }
    if peft_config is not None:
        trainer_kwargs["peft_config"] = peft_config

    trainer = GRPOTrainer(**trainer_kwargs)

    # --- Step 5: Train ---
    print(f"\n{'='*60}")
    print(f"  Starting GRPO training for {args.steps} steps...")
    print(f"  Training on {len(dataset)} prompts with {args.num_generations} generations each")
    print(f"{'='*60}\n")

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # --- Step 6: Save final model/adapter + tokenizer ---
    print(f"\nSaving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # --- Step 7: Write training metrics log ---
    metrics_log = {
        "train_result": {
            "global_step": train_result.global_step,
            "training_loss": train_result.training_loss,
            "metrics": train_result.metrics,
        },
        "log_history": trainer.state.log_history,
        "config": {
            "model": args.model,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "use_lora": args.use_lora,
            "use_4bit": args.use_4bit,
            "gradient_checkpointing": args.gradient_checkpointing,
            "num_generations": args.num_generations,
            "num_seeds": args.num_seeds,
        },
    }
    log_path = results_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(metrics_log, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  ✅ Training complete!")
    print(f"  📁 Model saved to: {args.output_dir}/")
    print(f"  📊 Training log saved to: {log_path}")
    print(f"  📈 Final loss: {train_result.training_loss:.4f}")
    print(f"  🔢 Total steps: {train_result.global_step}")
    if args.use_lora:
        print(f"  💡 To load: use PeftModel.from_pretrained(base_model, '{args.output_dir}')")
    else:
        print(f"  💡 To load: use AutoModelForCausalLM.from_pretrained('{args.output_dir}')")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
