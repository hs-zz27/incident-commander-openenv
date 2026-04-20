"""
Baseline Agent Benchmarking for Incident Commander Environment.

Runs 4 agent types across all tasks and saves reward data for comparison:
  1. RandomAgent    — picks random valid action each step
  2. HeuristicAgent — inspect first, then fix root cause (rule-based)
  3. LLMAgent       — existing inference.py agent (requires API key)
  4. TrainedAgent   — placeholder for post-training comparison

Usage:
  python run_baselines.py [--episodes 20] [--skip-llm] [--tasks all]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType
from server.tasks import list_tasks
from server.services import ALL_SERVICES


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
    """Placeholder for the trained RL agent (to be filled in Bangalore)."""

    name = "trained"

    def act(self, obs_dict: Dict[str, Any], action_history: List[str]) -> IncidentAction:
        # TODO: Load trained model and generate action
        # For now, uses heuristic as placeholder
        return HeuristicAgent().act(obs_dict, action_history)


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

    agents = [RandomAgent(), HeuristicAgent(), TrainedAgent()]

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
