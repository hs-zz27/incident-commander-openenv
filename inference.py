"""
Baseline inference script for the Incident Commander Environment.

Uses the OpenAI client to run an LLM agent that interacts with the environment
across all three tasks (easy, medium, hard).

Configuration via environment variables:
  API_BASE_URL    — OpenAI-compatible API base (default: https://api.openai.com/v1)
  MODEL_NAME      — Model name (default: gpt-4o-mini)
  OPENAI_API_KEY  — API key for the LLM service
  HF_TOKEN        — Hugging Face token (optional)

Stdout format (exact):
  [START] task=<task_name> env=<benchmark> model=<model_name>
  [STEP] step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
  [END] success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from typing import Any, Dict, List, Optional

# Guard openai import — give a clear error instead of a stack trace
try:
    from openai import OpenAI
except ImportError:
    print(
        "ERROR: 'openai' package is required. Install with: pip install openai>=1.0.0",
        file=sys.stderr,
    )
    sys.exit(1)

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks

# Import training prompt builder for --local mode (must match training distribution)
try:
    from train_grpo import build_obs_prompt as _training_prompt_builder
except ImportError:
    _training_prompt_builder = None


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://api.openai.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "gpt-4o-mini"

BENCHMARK_NAME = "incident_commander_env"
MAX_RETRIES = 3


# ---------------------------------------------------------------------------
# System prompt for the LLM agent
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert Site Reliability Engineer acting as Incident Commander for a microservices production system that is experiencing an outage.

Your goal is to diagnose the root cause and restore the system to full health as quickly as possible.

## Available Services
- database: Foundational data store
- cache: Redis cache layer
- auth: Authentication service (depends on database, cache)
- notification: Messaging service (standalone)
- payments: Payment processing (depends on database, notification)
- checkout: User-facing checkout (depends on auth, payments, database)

## Available Actions (respond with JSON)
- {"action_type": "inspect_logs", "service_name": "<service>"}
- {"action_type": "inspect_metrics", "service_name": "<service>"}
- {"action_type": "restart_service", "service_name": "<service>"}
- {"action_type": "scale_service", "service_name": "<service>"}
- {"action_type": "rollback", "service_name": "<service>"}
- {"action_type": "clear_cache"}
- {"action_type": "escalate"}
- {"action_type": "do_nothing"}
- {"action_type": "write_runbook", "metadata": {"summary": "<incident summary>"}}  (only on final step)

## Strategy
1. First INSPECT logs and metrics of services showing errors
2. Identify the ROOT CAUSE (the upstream service causing cascading failures)
3. Fix the root cause FIRST (restart, scale, or rollback as appropriate)
4. Then restart dependent services in dependency order
5. Verify the system is recovering

## Key Hints
- If a service has a non-standard version (not v1.0.0), consider ROLLBACK
- If a service has high CPU/memory, consider SCALE before RESTART
- Fix dependencies BEFORE dependents (e.g. fix database before auth)
- Don't repeat the same action — try something different each step
- Check logs for specific error messages pointing to root cause

## Response Format
Respond with ONLY a valid JSON object representing your action. No explanation, no markdown.
Example: {"action_type": "inspect_logs", "service_name": "database"}
"""


# ---------------------------------------------------------------------------
# Observation → user prompt
# ---------------------------------------------------------------------------

def observation_to_prompt(
    obs_dict: Dict[str, Any],
    step: int,
    action_history: List[str],
) -> str:
    """Convert an observation dict to a human-readable prompt for the LLM."""
    lines = [f"## Current Status (Step {step}/{obs_dict.get('max_steps', 30)})"]
    lines.append(f"System Health: {obs_dict.get('system_health_score', 0):.2%}")
    lines.append(f"Severity: {obs_dict.get('incident_severity', 'unknown')}")
    lines.append("")

    # Service table
    lines.append("### Services")
    services = obs_dict.get("services", {})
    for name, svc in sorted(services.items()):
        status = svc.get("status", "unknown")
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

    # Alerts
    alerts = obs_dict.get("alerts", [])
    if alerts:
        lines.append("")
        lines.append("### Alerts")
        for a in alerts:
            lines.append(f"  {a}")

    # Logs (from last inspect)
    logs = obs_dict.get("logs", [])
    if logs:
        lines.append("")
        lines.append("### Recent Logs")
        for log_line in logs:
            lines.append(f"  {log_line}")

    # Metrics detail
    metrics = obs_dict.get("metrics_detail")
    if metrics:
        lines.append("")
        lines.append("### Metrics Detail")
        lines.append(f"  {json.dumps(metrics, indent=2)}")

    # Error
    err_msg = obs_dict.get("last_action_error")
    if err_msg:
        lines.append("")
        lines.append(f"⚠️ Last Action Error: {err_msg}")

    # Action history (W-07 fix)
    if action_history:
        lines.append("")
        lines.append("### Actions Taken So Far")
        for i, a in enumerate(action_history, 1):
            lines.append(f"  {i}. {a}")

    lines.append("")
    lines.append("What action should we take next? Respond with a JSON action object only.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Parse LLM response → IncidentAction
# ---------------------------------------------------------------------------

def parse_action(response_text: str) -> Optional[IncidentAction]:
    """Parse LLM response into IncidentAction with fuzzy action matching."""
    # Use the shared fuzzy parser from evaluate_trained.py
    try:
        from evaluate_trained import parse_action as fuzzy_parse
        return fuzzy_parse(response_text)
    except ImportError:
        pass

    # Inline fallback if evaluate_trained not available
    text = response_text.strip()
    if "```" in text:
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

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
# Smart fallback action (W-06 fix)
# ---------------------------------------------------------------------------

def fallback_action(
    obs_dict: Dict[str, Any],
    step: int,
    action_history: List[str],
) -> IncidentAction:
    """
    Deterministic fallback when LLM fails.

    Strategy:
    - Steps 1-3: inspect the most degraded services
    - Steps 4+: try restarting the most degraded service
    - After restart fails: try scaling
    """
    services = obs_dict.get("services", {})

    # Rank services by health (worst first)
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

    # Find services we haven't inspected yet
    inspected = set()
    restarted = set()
    for a in action_history:
        if a.startswith("inspect_logs:") or a.startswith("inspect_metrics:"):
            inspected.add(a.split(":", 1)[1])
        elif a.startswith("restart_service:"):
            restarted.add(a.split(":", 1)[1])

    # Phase 1: Inspect un-inspected unhealthy services
    for _, name, svc in ranked:
        if svc.get("status") != "healthy" and name not in inspected:
            return IncidentAction(
                action_type=ActionType.INSPECT_LOGS,
                service_name=name,
            )

    # Phase 2: Check for version mismatch → rollback
    for _, name, svc in ranked:
        version = svc.get("version", "v1.0.0")
        if version != "v1.0.0" and svc.get("status") != "healthy":
            if f"rollback:{name}" not in action_history:
                return IncidentAction(
                    action_type=ActionType.ROLLBACK,
                    service_name=name,
                )

    # Phase 3: Restart the worst un-restarted unhealthy service
    for _, name, svc in ranked:
        if svc.get("status") != "healthy" and name not in restarted:
            return IncidentAction(
                action_type=ActionType.RESTART_SERVICE,
                service_name=name,
            )

    # Phase 4: Scale the worst unhealthy service
    for _, name, svc in ranked:
        if svc.get("status") != "healthy":
            if f"scale_service:{name}" not in action_history:
                return IncidentAction(
                    action_type=ActionType.SCALE_SERVICE,
                    service_name=name,
                )

    # Phase 5: Try restarting anything still unhealthy
    for _, name, svc in ranked:
        if svc.get("status") != "healthy":
            return IncidentAction(
                action_type=ActionType.RESTART_SERVICE,
                service_name=name,
            )

    # Everything healthy or nothing to do
    return IncidentAction(action_type=ActionType.DO_NOTHING)


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
        print(f"\n❌ Adapter not found at '{adapter_path}'", file=sys.stderr)
        print(f"   Get trained_model_full_0p5b/ from your teammate.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"\n  Loading trained model on {device}...")

    # Precision
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
    print(f"  ✅ Model loaded: {param_count:,} params on {device}")

    return _local_model, _local_tokenizer


def generate_local_action(
    model, tokenizer,
    obs_dict, step, action_history,
) -> str:
    """
    Generate action using the trained model with TRAINING-ALIGNED prompts.

    Uses build_obs_prompt (user-role only, no system prompt) to match
    the exact distribution the model was trained on.
    """
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

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Run a single task (updated for --local support)
# ---------------------------------------------------------------------------

def run_task(task_name: str, client=None, local_model=None, local_tokenizer=None) -> None:
    """Run a single task episode and print results in exact format."""
    use_local = local_model is not None
    env = IncidentCommanderEnvironment()
    rewards: List[float] = []
    success = False
    score = 0.0
    steps = 0
    action_history: List[str] = []

    mode_str = "local" if use_local else MODEL_NAME
    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={mode_str}", flush=True)

    try:
        obs = env.reset(task_name=task_name)
        obs_dict = obs.model_dump()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        MAX_REPEATS = 3  # repeat guard threshold

        while not obs.done:
            steps += 1

            action = None

            if use_local:
                # Local model path — uses training-aligned prompts
                for attempt in range(MAX_RETRIES):
                    try:
                        response_text = generate_local_action(
                            local_model, local_tokenizer,
                            obs_dict, steps, action_history,
                        )
                        action = parse_action(response_text)
                        if action:
                            break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(f"WARNING: Local model failed: {e}", file=sys.stderr)
            else:
                # API path — uses OpenAI client with SYSTEM_PROMPT
                prompt = observation_to_prompt(obs_dict, steps, action_history)
                messages.append({"role": "user", "content": prompt})

                for attempt in range(MAX_RETRIES):
                    try:
                        completion = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=messages,
                            temperature=0.0,
                            max_tokens=256,
                        )
                        response_text = completion.choices[0].message.content or ""
                        action = parse_action(response_text)
                        if action:
                            messages.append({"role": "assistant", "content": response_text})
                            break
                    except Exception as e:
                        if attempt == MAX_RETRIES - 1:
                            print(
                                f"WARNING: LLM failed after {MAX_RETRIES} retries: {e}",
                                file=sys.stderr,
                            )

            # Repeat guard: if model outputs same action 3x in a row, use fallback
            if action is not None:
                a_check = action.action_type.value
                if action.service_name:
                    a_check += f":{action.service_name}"
                recent = action_history[-MAX_REPEATS:] if len(action_history) >= MAX_REPEATS else []
                if len(recent) == MAX_REPEATS and all(a == a_check for a in recent):
                    action = fallback_action(obs_dict, steps, action_history)

            if action is None:
                action = fallback_action(obs_dict, steps, action_history)

            # Build action string for history
            action_str = action.action_type.value
            if action.service_name:
                action_str += f":{action.service_name}"
            action_history.append(action_str)

            # Execute action
            obs = env.step(action)
            obs_dict = obs.model_dump()

            reward = obs.reward if obs.reward is not None else 0.0
            rewards.append(reward)
            error_str = obs.last_action_error if obs.last_action_error else "null"

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={str(obs.done).lower()} "
                f"error={error_str}",
                flush=True,
            )

            # Keep conversation history manageable (API mode only)
            if not use_local and len(messages) > 20:
                messages = messages[:1] + messages[-18:]

        # Final grading
        grade_result = env.grade()
        success = grade_result.get("is_resolved", False)
        score = grade_result.get("score", 0.0)

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(
            f"[STEP] step={steps + 1} action=error reward=0.00 "
            f"done=true error={error_msg}",
            flush=True,
        )
        rewards.append(0.0)
        steps += 1
        score = 0.0

    finally:
        env.close()
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run inference across all three tasks."""
    import argparse

    parser = argparse.ArgumentParser(description="Incident Commander Inference")
    parser.add_argument("--local", action="store_true",
                        help="Use local trained model instead of API")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct",
                        help="Base model for --local mode")
    parser.add_argument("--adapter", type=str, default="trained_model_full_0p5b",
                        help="LoRA adapter path for --local mode")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="Device for --local mode (default: auto-detect)")
    parser.add_argument("--task", type=str, default=None,
                        help="Run a specific task (default: all)")
    args = parser.parse_args()

    local_model = None
    local_tokenizer = None
    client = None

    if args.local:
        local_model, local_tokenizer = load_local_model(
            args.base_model, args.adapter, args.device
        )
    else:
        if not API_KEY:
            print("WARNING: No HF_TOKEN or API_KEY set.", file=sys.stderr)
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    tasks = [args.task] if args.task else list_tasks()
    for task_name in tasks:
        run_task(
            task_name,
            client=client,
            local_model=local_model,
            local_tokenizer=local_tokenizer,
        )


if __name__ == "__main__":
    main()

