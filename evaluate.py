#!/usr/bin/env python3
"""
Self-contained evaluation script for the Incident Commander Environment.

Runs deterministic expert strategies against all three tasks WITHOUT any
external API key. Produces a detailed report proving the environment works
end-to-end with correct grading, determinism, and reward shaping.

Usage:
    python evaluate.py              # full evaluation
    python evaluate.py --task easy   # single task
    python evaluate.py --verbose     # detailed output
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, ".")

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks, get_task


# ---------------------------------------------------------------------------
# Expert strategies (optimal action sequences)
# ---------------------------------------------------------------------------

EXPERT_STRATEGIES: Dict[str, List[IncidentAction]] = {
    "single_service_failure": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"),
    ],
    "cascading_failure": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"),
        IncidentAction(action_type=ActionType.INSPECT_METRICS, service_name="checkout"),
        IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="auth"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
    ],
    "hidden_root_cause": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="checkout"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="payments"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="auth"),
        IncidentAction(action_type=ActionType.ROLLBACK, service_name="auth"),
        IncidentAction(action_type=ActionType.CLEAR_CACHE),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
    ],
    "chaos_cascade": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="checkout"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="auth"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
        # After step 8 chaos injection, notification goes down
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="notification"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="notification"),
        # May need extra restarts for dependents of notification
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
    ],
    "multi_root_cause": [
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="auth"),
        IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"),
        IncidentAction(action_type=ActionType.ROLLBACK, service_name="auth"),
        IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
        IncidentAction(action_type=ActionType.CLEAR_CACHE),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
        IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
    ],
}

# Naive strategy: just restart everything (for comparison)
_NAIVE_RESTART_ALL = [
    IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"),
    IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"),
    IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="auth"),
    IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="notification"),
    IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"),
    IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"),
]

NAIVE_STRATEGIES: Dict[str, List[IncidentAction]] = {
    "single_service_failure": list(_NAIVE_RESTART_ALL),
    "cascading_failure": list(_NAIVE_RESTART_ALL),
    "hidden_root_cause": list(_NAIVE_RESTART_ALL),
    "chaos_cascade": list(_NAIVE_RESTART_ALL) + list(_NAIVE_RESTART_ALL),  # double pass
    "multi_root_cause": list(_NAIVE_RESTART_ALL) + list(_NAIVE_RESTART_ALL),
}

# Do-nothing strategy (baseline floor — always times out)
DO_NOTHING_STRATEGY: List[IncidentAction] = [
    IncidentAction(action_type=ActionType.DO_NOTHING)
] * 30


# ---------------------------------------------------------------------------
# Run strategy
# ---------------------------------------------------------------------------

def run_strategy(
    task_name: str,
    actions: List[IncidentAction],
    verbose: bool = False,
    label: str = "strategy",
) -> Dict:
    """Run a strategy and return the result dict."""
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name, episode_id=f"eval-{task_name}-{label}")

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  {label.upper()} on {task_name}")
        print(f"{'─'*60}")
        print(f"  Initial health: {obs.system_health_score:.4f}")
        print(f"  Severity: {obs.incident_severity.value}")

    rewards = []
    for i, action in enumerate(actions):
        obs = env.step(action)
        r = obs.reward if obs.reward is not None else 0.0
        rewards.append(r)

        if verbose:
            svc = action.service_name or ""
            print(
                f"  Step {i+1:2d}: {action.action_type.value:18s} "
                f"{svc:14s} → health={obs.system_health_score:.4f}  "
                f"reward={r:+.4f}  done={obs.done}"
            )

        if obs.done:
            break

    grade = env.grade()
    env.close()

    result = {
        "task": task_name,
        "label": label,
        "score": grade["score"],
        "breakdown": grade["breakdown"],
        "is_resolved": grade["is_resolved"],
        "steps_taken": grade["steps_taken"],
        "escalated": grade["escalated"],
        "rewards": rewards,
        "cumulative_reward": sum(rewards),
        "final_health": obs.system_health_score,
    }

    if verbose:
        print(f"  ────────────────────────")
        print(f"  Score: {grade['score']:.4f}  Resolved: {grade['is_resolved']}")
        print(f"  Breakdown: {json.dumps(grade['breakdown'], indent=4)}")

    return result


# ---------------------------------------------------------------------------
# Determinism check
# ---------------------------------------------------------------------------

def check_determinism(task_name: str, actions: List[IncidentAction], runs: int = 5) -> bool:
    """Run the same strategy N times and verify identical results."""
    results = []
    for i in range(runs):
        r = run_strategy(task_name, actions, verbose=False, label=f"det-{i}")
        results.append(r)

    # Compare all runs
    baseline = results[0]
    for i, r in enumerate(results[1:], 1):
        if abs(r["score"] - baseline["score"]) > 1e-6:
            return False
        if r["steps_taken"] != baseline["steps_taken"]:
            return False
        if r["is_resolved"] != baseline["is_resolved"]:
            return False
        for j, (a, b) in enumerate(zip(r["rewards"], baseline["rewards"])):
            if abs(a - b) > 1e-6:
                return False
    return True


# ---------------------------------------------------------------------------
# Full evaluation
# ---------------------------------------------------------------------------

def run_full_evaluation(
    task_filter: Optional[str] = None,
    verbose: bool = False,
) -> None:
    """Run the complete evaluation suite."""
    tasks = list_tasks()
    # Only evaluate tasks that have defined expert strategies
    # (random_incident is non-deterministic and has no fixed expert strategy)
    evaluable_tasks = [t for t in tasks if t in EXPERT_STRATEGIES]
    if task_filter:
        task_map = {"easy": "single_service_failure", "medium": "cascading_failure", "hard": "hidden_root_cause"}
        selected = task_map.get(task_filter, task_filter)
        evaluable_tasks = [t for t in evaluable_tasks if t == selected]
        if not evaluable_tasks:
            print(f"Error: Unknown task '{task_filter}'")
            sys.exit(1)
    tasks = evaluable_tasks

    print("=" * 70)
    print("  INCIDENT COMMANDER ENVIRONMENT — EVALUATION REPORT")
    print("=" * 70)
    print(f"  Tasks: {', '.join(tasks)}")
    print(f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    all_results = {}

    # ── Expert Strategy ──
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  EXPERT STRATEGY (optimal actions)                              │")
    print("└──────────────────────────────────────────────────────────────────┘")

    expert_results = {}
    for task_name in tasks:
        r = run_strategy(task_name, EXPERT_STRATEGIES[task_name], verbose=verbose, label="expert")
        expert_results[task_name] = r
        status = "✅ RESOLVED" if r["is_resolved"] else "❌ UNRESOLVED"
        print(f"  {task_name:30s}  score={r['score']:.4f}  steps={r['steps_taken']:2d}  {status}")
    all_results["expert"] = expert_results

    # ── Naive Strategy ──
    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  NAIVE STRATEGY (restart everything)                            │")
    print("└──────────────────────────────────────────────────────────────────┘")

    naive_results = {}
    for task_name in tasks:
        r = run_strategy(task_name, NAIVE_STRATEGIES[task_name], verbose=verbose, label="naive")
        naive_results[task_name] = r
        status = "✅ RESOLVED" if r["is_resolved"] else "❌ UNRESOLVED"
        print(f"  {task_name:30s}  score={r['score']:.4f}  steps={r['steps_taken']:2d}  {status}")
    all_results["naive"] = naive_results

    # ── Do-Nothing Baseline ──
    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  DO-NOTHING BASELINE (floor)                                    │")
    print("└──────────────────────────────────────────────────────────────────┘")

    nothing_results = {}
    for task_name in tasks:
        task = get_task(task_name)
        strategy = [IncidentAction(action_type=ActionType.DO_NOTHING)] * task.max_steps
        r = run_strategy(task_name, strategy, verbose=False, label="nothing")
        nothing_results[task_name] = r
        status = "✅ RESOLVED" if r["is_resolved"] else "❌ UNRESOLVED"
        print(f"  {task_name:30s}  score={r['score']:.4f}  steps={r['steps_taken']:2d}  {status}")
    all_results["nothing"] = nothing_results

    # ── Determinism ──
    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  DETERMINISM VERIFICATION (5 runs each)                         │")
    print("└──────────────────────────────────────────────────────────────────┘")

    all_deterministic = True
    # Only check determinism for non-random tasks
    determinism_tasks = [t for t in tasks if t != "random_incident"]
    for task_name in determinism_tasks:
        ok = check_determinism(task_name, EXPERT_STRATEGIES[task_name], runs=5)
        icon = "✅" if ok else "❌"
        print(f"  {task_name:30s}  {icon} {'DETERMINISTIC' if ok else 'NON-DETERMINISTIC'}")
        if not ok:
            all_deterministic = False

    # ── Score Differentiation ──
    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  SCORE DIFFERENTIATION                                          │")
    print("└──────────────────────────────────────────────────────────────────┘")

    for task_name in tasks:
        e = expert_results[task_name]["score"]
        n = naive_results[task_name]["score"]
        d = nothing_results[task_name]["score"]
        gap_en = e - n
        gap_nd = n - d
        print(f"  {task_name}:")
        print(f"    Expert={e:.4f}  Naive={n:.4f}  Nothing={d:.4f}")
        print(f"    Expert-Naive gap: {gap_en:+.4f}  Naive-Nothing gap: {gap_nd:+.4f}")
        if e > n > d:
            print(f"    ✅ Properly ordered: Expert > Naive > Nothing")
        else:
            print(f"    ⚠️  Ordering issue detected")

    # ── Summary ──
    print()
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)

    all_ok = True
    checks = [
        ("Expert solves all tasks", all(r["is_resolved"] for r in expert_results.values())),
        ("Expert scores > 0.80 on all", all(r["score"] >= 0.80 for r in expert_results.values())),
        ("Naive scores lower than expert", all(
            naive_results[t]["score"] <= expert_results[t]["score"]
            for t in tasks
        )),
        ("Do-nothing scores near zero", all(r["score"] < 0.2 for r in nothing_results.values())),
        ("Environment is deterministic", all_deterministic),
        ("Grader scores in [0, 1]", all(
            0 <= r["score"] <= 1
            for group in all_results.values()
            for r in group.values()
        )),
    ]

    for label, passed in checks:
        icon = "✅" if passed else "❌"
        print(f"  {icon} {label}")
        if not passed:
            all_ok = False

    print()
    if all_ok:
        print("  🎉  ALL CHECKS PASSED — Environment is ready for submission!")
    else:
        print("  ⚠️   SOME CHECKS FAILED — Review the results above.")

    print("=" * 70)

    # ── Stdout format demo ──
    print()
    print("┌──────────────────────────────────────────────────────────────────┐")
    print("│  INFERENCE STDOUT FORMAT DEMO                                   │")
    print("└──────────────────────────────────────────────────────────────────┘")
    task_name = tasks[0]
    strategy = EXPERT_STRATEGIES[task_name]
    env = IncidentCommanderEnvironment()
    obs = env.reset(task_name=task_name)
    rewards = []
    steps = 0

    print(f"[START] task={task_name} env=incident_commander_env model=expert-baseline")
    for action in strategy:
        obs = env.step(action)
        steps += 1
        r = obs.reward if obs.reward is not None else 0.0
        rewards.append(r)
        err = obs.last_action_error if obs.last_action_error else "null"
        a_str = action.action_type.value
        if action.service_name:
            a_str += f":{action.service_name}"
        print(f"[STEP] step={steps} action={a_str} reward={r:.2f} done={str(obs.done).lower()} error={err}")
        if obs.done:
            break

    grade = env.grade()
    success = grade.get("is_resolved", False)
    score = grade.get("score", 0.0)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}")
    env.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Incident Commander Environment")
    parser.add_argument("--task", choices=["easy", "medium", "hard", "chaos_cascade", "multi_root_cause"], help="Run a single task")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed step output")
    args = parser.parse_args()

    run_full_evaluation(task_filter=args.task, verbose=args.verbose)
