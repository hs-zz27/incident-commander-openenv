"""
Incident Commander Environment — core environment logic.

Implements reset(), step(), and state() following the OpenEnv specification.
All state transitions are deterministic given the same seed and action sequence.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Dict, List, Optional
from uuid import uuid4

from .models import (
    ActionType,
    IncidentAction,
    IncidentObservation,
    IncidentState,
    ServiceState,
    ServiceStatusEnum,
    SeverityLevel,
)
from .services import (
    ALL_SERVICES,
    DEPENDENCY_GRAPH,
    apply_clear_cache,
    apply_restart,
    apply_rollback,
    apply_scale,
    compute_health_score,
    classify_severity,
    generate_alerts,
    generate_logs,
    generate_metrics,
    propagate_dependencies,
)
from .tasks import TaskDefinition, get_task, list_tasks
from .grader import compute_step_reward, grade_episode
from .chaos import ChaosAgent


class IncidentCommanderEnvironment:
    """
    OpenEnv-compatible environment for production incident response simulation.

    The agent plays Incident Commander and must triage, diagnose, and resolve
    a microservices outage across three difficulty levels.
    """

    def __init__(self) -> None:
        self._state: Optional[IncidentState] = None
        self._services: Dict[str, ServiceState] = {}
        self._task: Optional[TaskDefinition] = None
        self._actions_history: List[str] = []
        self._rewards: List[float] = []
        self._escalated: bool = False
        self._is_done: bool = False
        self._last_action_error: Optional[str] = None
        self._prev_health: float = 0.0
        self._inspected_services: set = set()
        self._timeline: List[Dict[str, Any]] = []
        self._chaos_agent: ChaosAgent = ChaosAgent()
        self._chaos_mode: bool = False
        self._chaos_rng: random.Random = random.Random()
        self._last_chaos_event: Optional[str] = None

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_name: Optional[str] = None,
        chaos_mode: bool = False,
        **kwargs: Any,
    ) -> IncidentObservation:
        """
        Reset the environment to start a new episode.

        Args:
            seed: Random seed for reproducibility. For random_incident tasks,
                  pass None for a fresh random scenario each time.
            episode_id: Optional episode identifier.
            task_name: Name of the task to run. Defaults to "single_service_failure".
            chaos_mode: If True, enables background chaos injection during the episode.

        Returns:
            Initial observation of the incident scenario.
        """
        task_name = task_name or "single_service_failure"

        # For random_incident, pass seed through to get_task for parameterized generation
        if task_name == "random_incident":
            self._task = get_task(task_name, seed=seed)
        else:
            self._task = get_task(task_name)

        # Deep-copy initial services so we don't mutate the task definition
        self._services = {
            k: v.model_copy() for k, v in self._task.initial_services.items()
        }

        self._actions_history = []
        self._rewards = []
        self._escalated = False
        self._is_done = False
        self._last_action_error = None
        self._inspected_services = set()
        self._prev_health = compute_health_score(self._services)
        self._last_chaos_event = None

        # Initialize chaos agent
        self._chaos_mode = chaos_mode
        self._chaos_agent = ChaosAgent()
        self._chaos_agent.reset()
        self._chaos_rng = random.Random(seed if seed is not None else random.randint(0, 99999))
        self._timeline = [{
            "step": 0,
            "event": "incident_detected",
            "severity": classify_severity(self._services),
            "health": self._prev_health,
            "description": f"Incident detected: {self._task.description}",
            "affected_services": [
                s.name for s in self._services.values()
                if s.status != ServiceStatusEnum.HEALTHY
            ],
        }]

        self._state = IncidentState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_name=task_name,
            is_resolved=False,
            cumulative_reward=0.0,
            actions_taken=[],
            services={k: v.model_copy() for k, v in self._services.items()},
            incident_timeline=list(self._timeline),
        )

        return self._build_observation(reward=0.0, logs=[], metrics_detail=None)

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(
        self,
        action: IncidentAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IncidentObservation:
        """
        Execute one action and return the resulting observation.

        Args:
            action: The IncidentAction to execute.
            timeout_s: Optional timeout (unused).

        Returns:
            Observation after executing the action.
        """
        if self._state is None or self._task is None:
            raise RuntimeError("Environment not initialised. Call reset() first.")

        if self._is_done:
            return self._build_observation(
                reward=0.0,
                logs=[],
                metrics_detail=None,
            )

        self._state.step_count += 1
        self._last_action_error = None

        action_type = action.action_type.value
        service_name = action.service_name

        # Build action string for history
        action_str = f"{action_type}:{service_name}" if service_name else action_type

        # Validate service name for service-specific actions
        needs_service = action_type in (
            "inspect_logs", "inspect_metrics",
            "restart_service", "scale_service", "rollback",
        )

        if needs_service and not service_name:
            self._last_action_error = (
                f"Action '{action_type}' requires a service_name."
            )
            self._actions_history.append(action_str)
            reward = -0.02
            self._rewards.append(reward)
            self._state.cumulative_reward += reward
            self._state.actions_taken.append(action_str)
            return self._build_observation(reward=reward, logs=[], metrics_detail=None)

        if needs_service and service_name not in ALL_SERVICES:
            self._last_action_error = (
                f"Unknown service '{service_name}'. "
                f"Valid: {ALL_SERVICES}"
            )
            self._actions_history.append(action_str)
            reward = -0.02
            self._rewards.append(reward)
            self._state.cumulative_reward += reward
            self._state.actions_taken.append(action_str)
            return self._build_observation(reward=reward, logs=[], metrics_detail=None)

        # Execute the action
        logs: List[str] = []
        metrics_detail = None

        if action_type == "inspect_logs":
            logs = generate_logs(
                service_name, self._services, self._task.name, self._state.step_count
            )
            self._inspected_services.add(service_name)

        elif action_type == "inspect_metrics":
            metrics_detail = generate_metrics(
                service_name, self._services, self._task.name
            )
            self._inspected_services.add(service_name)

        elif action_type == "restart_service":
            self._services = apply_restart(
                self._services, service_name, self._task.name
            )
            # Propagate dependency effects
            self._services = propagate_dependencies(self._services, self._task.name)

        elif action_type == "scale_service":
            self._services = apply_scale(self._services, service_name)
            self._services = propagate_dependencies(self._services, self._task.name)

        elif action_type == "rollback":
            self._services = apply_rollback(
                self._services, service_name, good_version="v1.0.0"
            )
            self._services = propagate_dependencies(self._services, self._task.name)

        elif action_type == "clear_cache":
            self._services = apply_clear_cache(self._services)
            self._services = propagate_dependencies(self._services, self._task.name)

        elif action_type == "escalate":
            self._escalated = True
            self._is_done = True

        elif action_type == "do_nothing":
            pass  # literally nothing

        self._actions_history.append(action_str)

        # --- Chaos injection (before computing reward) ---
        self._last_chaos_event = None
        if self._chaos_mode:
            chaos_result = self._chaos_agent.maybe_inject(
                step=self._state.step_count,
                current_services=self._services,
                rng=self._chaos_rng,
                inspected_services=self._inspected_services,
            )
            if chaos_result:
                self._last_chaos_event = chaos_result
                self._services = propagate_dependencies(self._services, self._task.name)

        # Scripted chaos for chaos_cascade task: notification fails at step 8
        if (
            self._task.name == "chaos_cascade"
            and self._state.step_count == 8
            and "notification" not in self._chaos_agent.injected_services
        ):
            self._chaos_agent.force_inject("notification", self._services, "oom_crash")
            self._last_chaos_event = "notification"
            self._services = propagate_dependencies(self._services, self._task.name)

        # Compute health and reward
        curr_health = compute_health_score(self._services)

        # Check if system is fully resolved
        all_healthy = all(
            s.status == ServiceStatusEnum.HEALTHY for s in self._services.values()
        )
        if all_healthy and curr_health >= 0.95:
            self._state.is_resolved = True
            self._is_done = True

        # Check step limit
        if self._state.step_count >= self._task.max_steps:
            self._is_done = True

        reward = compute_step_reward(
            prev_health=self._prev_health,
            curr_health=curr_health,
            action_type=action_type,
            service_name=service_name,
            task=self._task,
            actions_history=self._actions_history,
            services=self._services,
            is_done=self._is_done,
            steps_taken=self._state.step_count,
        )

        # Chaos survival bonus: if agent handled chaos without further health loss
        health_delta = curr_health - self._prev_health
        if self._last_chaos_event and health_delta >= 0:
            reward += 0.05

        # Record timeline event
        event = {
            "step": self._state.step_count,
            "event": action_str,
            "health": round(curr_health, 4),
            "health_delta": round(health_delta, 4),
            "reward": round(reward, 4),
        }
        if self._last_chaos_event:
            event["chaos_event"] = self._last_chaos_event
        if self._last_action_error:
            event["error"] = self._last_action_error
        if self._state.is_resolved:
            event["event_type"] = "incident_resolved"
        elif self._is_done:
            event["event_type"] = "episode_ended"
        else:
            event["event_type"] = "action"
        self._timeline.append(event)

        self._prev_health = curr_health
        self._rewards.append(reward)
        self._state.cumulative_reward += reward
        self._state.actions_taken = list(self._actions_history)
        self._state.services = {
            k: v.model_copy() for k, v in self._services.items()
        }
        self._state.incident_timeline = list(self._timeline)

        return self._build_observation(
            reward=reward, logs=logs, metrics_detail=metrics_detail
        )

    # ------------------------------------------------------------------
    # state (property)
    # ------------------------------------------------------------------

    @property
    def state(self) -> IncidentState:
        """Return the current environment state."""
        if self._state is None:
            return IncidentState(episode_id=None, step_count=0)
        return self._state

    @property
    def timeline(self) -> List[Dict[str, Any]]:
        """Return the incident timeline for the current episode."""
        return list(self._timeline)

    # ------------------------------------------------------------------
    # grade() — final scoring
    # ------------------------------------------------------------------

    def grade(self) -> Dict[str, Any]:
        """
        Grade the completed episode. Returns score and breakdown.

        Should be called after the episode is done.
        """
        if self._task is None:
            return {
                "score": 0.0,
                "breakdown": {"recovery": 0.0, "efficiency": 0.0, "diagnostics": 0.0, "ordering": 0.0},
                "steps_taken": 0,
                "is_resolved": False,
                "escalated": False,
                "rewards": [],
            }

        score, breakdown = grade_episode(
            task=self._task,
            final_services=self._services,
            actions_history=self._actions_history,
            steps_taken=self._state.step_count if self._state else 0,
            is_resolved=self._state.is_resolved if self._state else False,
            escalated=self._escalated,
        )
        return {
            "score": score,
            "breakdown": breakdown,
            "steps_taken": self._state.step_count if self._state else 0,
            "is_resolved": self._state.is_resolved if self._state else False,
            "escalated": self._escalated,
            "rewards": list(self._rewards),
        }

    # ------------------------------------------------------------------
    # close()
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Clean up resources (no-op for this environment)."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_observation(
        self,
        reward: float,
        logs: List[str],
        metrics_detail: Optional[Dict[str, Any]],
    ) -> IncidentObservation:
        """Construct an observation from current state."""
        health = compute_health_score(self._services)
        severity_str = classify_severity(self._services)
        alerts = generate_alerts(self._services)

        severity_map = {
            "critical": SeverityLevel.CRITICAL,
            "high": SeverityLevel.HIGH,
            "medium": SeverityLevel.MEDIUM,
            "low": SeverityLevel.LOW,
            "resolved": SeverityLevel.RESOLVED,
        }

        # Build metadata with optional chaos event info
        obs_metadata: Dict[str, Any] = {}
        if self._last_chaos_event:
            obs_metadata["new_chaos_event"] = self._last_chaos_event

        return IncidentObservation(
            done=self._is_done,
            reward=round(reward, 4),
            metadata=obs_metadata,
            services={k: v.model_copy() for k, v in self._services.items()},
            alerts=alerts,
            logs=logs,
            metrics_detail=metrics_detail,
            incident_severity=severity_map.get(severity_str, SeverityLevel.LOW),
            system_health_score=health,
            step_count=self._state.step_count if self._state else 0,
            max_steps=self._task.max_steps if self._task else 30,
            last_action_error=self._last_action_error,
            task_name=self._task.name if self._task else "",
        )
