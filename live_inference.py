"""
Live inference for Incident Commander via FastAPI backend (Qwen-only).

This script is structured similarly to inference.py but uses HTTP calls to the
local backend so the frontend dashboard can track state/timeline/log updates.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List, Optional

import requests

# Add project root to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.models import ActionType, IncidentAction
from server.tasks import list_tasks


BENCHMARK_NAME = "incident_commander_env"
MAX_RETRIES = 3
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
QWEN_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_ADAPTER = os.getenv("ADAPTER_PATH", "trained_model_0p5b_v2")


def parse_backend_action(action_data: Optional[Dict[str, Any]]) -> Optional[IncidentAction]:
    """Parse raw JSON data from backend into IncidentAction."""
    if not action_data:
        return None
    try:
        return IncidentAction(**action_data)
    except Exception:
        return None


def fallback_action(
    obs_dict: Dict[str, Any],
    step: int,
    action_history: List[str],
) -> IncidentAction:
    """
    Deterministic fallback when backend model fails.
    Mirrors inference.py fallback behavior.
    """
    services = obs_dict.get("services", {})

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

    inspected = set()
    restarted = set()
    for a in action_history:
        if a.startswith("inspect_logs:") or a.startswith("inspect_metrics:"):
            inspected.add(a.split(":", 1)[1])
        elif a.startswith("restart_service:"):
            restarted.add(a.split(":", 1)[1])

    for _, name, svc in ranked:
        if svc.get("status") != "healthy" and name not in inspected:
            return IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name=name)

    for _, name, svc in ranked:
        version = svc.get("version", "v1.0.0")
        if version != "v1.0.0" and svc.get("status") != "healthy":
            if f"rollback:{name}" not in action_history:
                return IncidentAction(action_type=ActionType.ROLLBACK, service_name=name)

    for _, name, svc in ranked:
        if svc.get("status") != "healthy" and name not in restarted:
            return IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name=name)

    for _, name, svc in ranked:
        if svc.get("status") != "healthy" and f"scale_service:{name}" not in action_history:
            return IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name=name)

    for _, name, svc in ranked:
        if svc.get("status") != "healthy":
            return IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name=name)

    return IncidentAction(action_type=ActionType.DO_NOTHING)


def _request_json(method: str, path: str, **kwargs) -> Dict[str, Any]:
    """Issue a backend request and return JSON payload."""
    response = requests.request(method, f"{BASE_URL}{path}", timeout=20, **kwargs)
    response.raise_for_status()
    return response.json()


def _touch_dashboard_endpoints() -> None:
    """
    Touch state/timeline endpoints used by frontend dashboard.
    This keeps API traffic aligned with live UI expectations.
    """
    _request_json("GET", "/state")
    _request_json("GET", "/timeline")


def run_live_task(task_name: str, adapter_path: str, device: str, delay_seconds: float) -> None:
    """Run a single task via backend Qwen policy + FastAPI endpoints."""
    rewards: List[float] = []
    action_history: List[str] = []
    success = False
    score = 0.0
    steps = 0
    done = False
    max_repeats = 3

    print(f"[START] task={task_name} env={BENCHMARK_NAME} model={QWEN_MODEL}", flush=True)

    try:
        _request_json("POST", "/reset", json={"task_name": task_name})
        _touch_dashboard_endpoints()

        while not done:
            steps += 1
            error_str = "null"

            state_data = _request_json("GET", "/state")
            obs_dict = state_data.get("state", {})

            action: Optional[IncidentAction] = None
            for attempt in range(MAX_RETRIES):
                try:
                    pred_data = _request_json(
                        "POST",
                        "/predict",
                        json={
                            "base_model": QWEN_MODEL,
                            "adapter_path": adapter_path,
                            "device": device,
                        },
                    )
                    action = parse_backend_action(pred_data.get("parsed_action"))
                    if action:
                        break
                except Exception as e:
                    if attempt == MAX_RETRIES - 1:
                        error_str = f"predict_error:{str(e).replace(chr(10), ' ')}"

            if action is not None:
                a_check = action.action_type.value
                if action.service_name:
                    a_check += f":{action.service_name}"
                recent = action_history[-max_repeats:] if len(action_history) >= max_repeats else []
                if len(recent) == max_repeats and all(a == a_check for a in recent):
                    action = fallback_action(obs_dict, steps, action_history)

            if action is None:
                action = fallback_action(obs_dict, steps, action_history)

            action_str = action.action_type.value
            if action.service_name:
                action_str += f":{action.service_name}"
            action_history.append(action_str)

            step_data = _request_json("POST", "/step", json={"action": action.model_dump()})
            reward = float(step_data.get("reward", 0.0))
            done = bool(step_data.get("done", False))
            rewards.append(reward)

            observation = step_data.get("observation", {}) if isinstance(step_data, dict) else {}
            if observation.get("last_action_error"):
                error_str = str(observation["last_action_error"]).replace("\n", " ")

            # Print inspect_logs output for live debugging while frontend updates via state/timeline.
            for log_line in observation.get("logs", []):
                print(f"[LOG] step={steps} action={action_str} {log_line}", flush=True)

            _touch_dashboard_endpoints()

            print(
                f"[STEP] step={steps} action={action_str} "
                f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
                flush=True,
            )

            if delay_seconds > 0:
                time.sleep(delay_seconds)

        grade_data = _request_json("GET", "/grade")
        success = grade_data.get("is_resolved", False)
        score = grade_data.get("score", 0.0)

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(
            f"[STEP] step={steps + 1} action=error reward=0.00 done=true error={error_msg}",
            flush=True,
        )
        rewards.append(0.0)
        steps += 1
        score = 0.0
        success = False

    finally:
        rewards_str = ",".join(f"{r:.2f}" for r in rewards)
        print(
            f"[END] success={str(success).lower()} steps={steps} "
            f"score={score:.3f} rewards={rewards_str}",
            flush=True,
        )


def main() -> None:
    """Run live inference across one or all tasks."""
    parser = argparse.ArgumentParser(
        description="Incident Commander live inference via FastAPI (Qwen only)"
    )
    parser.add_argument("--task", type=str, default=None, help="Run specific task (default: all)")
    parser.add_argument(
        "--adapter",
        type=str,
        default=DEFAULT_ADAPTER,
        help="LoRA adapter path for backend /predict",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use on backend model inference",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between steps for easier live dashboard viewing",
    )
    args = parser.parse_args()

    try:
        _request_json("GET", "/health")
    except Exception:
        print(
            f"ERROR: FastAPI server not found at {BASE_URL}. Run 'python server/app.py' first.",
            file=sys.stderr,
        )
        sys.exit(1)

    try:
        task_resp = _request_json("GET", "/tasks")
        available_tasks = task_resp.get("tasks", [])
    except Exception:
        available_tasks = list_tasks()

    if args.task:
        if available_tasks and args.task not in available_tasks:
            print(
                f"ERROR: Unknown task '{args.task}'. Available: {', '.join(available_tasks)}",
                file=sys.stderr,
            )
            sys.exit(1)
        tasks = [args.task]
    else:
        tasks = available_tasks if available_tasks else list_tasks()

    for task_name in tasks:
        run_live_task(task_name, args.adapter, args.device, args.delay)


if __name__ == "__main__":
    main()
