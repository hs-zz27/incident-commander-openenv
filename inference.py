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


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-mini")
# HF_TOKEN is the primary key per hackathon spec; OPENAI_API_KEY as fallback
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY") or "AIzaSyDyygKcNXZBMwuRY4MhAGX0B_ZzERKCSPg"

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
    """Parse the LLM response text into an IncidentAction."""
    text = response_text.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
        return IncidentAction(**data)
    except Exception:
        pass

    # Try to extract JSON from the response
    for start_char in ["{", "["]:
        idx = text.find(start_char)
        if idx >= 0:
            # Find matching close
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
# Run a single task
# ---------------------------------------------------------------------------

def run_task(task_name: str, client: OpenAI) -> None:
    """Run a single task episode and print results in exact format."""
    env = IncidentCommanderEnvironment()
    rewards: List[float] = []
    success = False
    score = 0.0
    steps = 0
    action_history: List[str] = []

    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={MODEL_NAME}", flush=True)

    try:
        obs = env.reset(task_name=task_name)
        obs_dict = obs.model_dump()
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        while not obs.done:
            steps += 1
            prompt = observation_to_prompt(obs_dict, steps, action_history)
            messages.append({"role": "user", "content": prompt})

            # Call LLM
            action = None
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

            # Keep conversation history manageable (sliding window)
            if len(messages) > 20:
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
            f"score={score:.2f} rewards={rewards_str}",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Run inference across all three tasks."""
    if not API_KEY:
        print("WARNING: No API key set. Set HF_TOKEN or OPENAI_API_KEY.", file=sys.stderr)

    client = OpenAI(
        api_key=API_KEY or "sk-placeholder",
        base_url=API_BASE_URL,
    )

    tasks = list_tasks()
    for task_name in tasks:
        run_task(task_name, client)


if __name__ == "__main__":
    main()
