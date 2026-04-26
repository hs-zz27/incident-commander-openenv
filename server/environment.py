"""
Incident Commander Environment — core environment logic.

Implements reset(), step(), and state() following the OpenEnv specification.
All state transitions are deterministic given the same seed and action sequence.

Features:
- Real-time severity escalation with revenue-loss-grounded rewards
- Runbook memory for cross-episode institutional knowledge
- Time bounding / SLA pressure (HTTP mode only)
- Partial & noisy log support via log_quality
"""

from __future__ import annotations

import copy
import random
import time
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
    REVERSE_DEPS,
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
from .runbook import RunbookMemory, RunbookEntry


class IncidentCommanderEnvironment:
    """
    OpenEnv-compatible environment for production incident response simulation.

    The agent plays Incident Commander and must triage, diagnose, and resolve
    a microservices outage across multiple difficulty levels.
    """

    def __init__(self, http_mode: bool = False) -> None:
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
        self._last_chaos_event_persistent: Optional[str] = None
        self._new_chaos_event_this_step: Optional[str] = None
        # Chaos visibility tuning (pure random, task-agnostic)
        self._chaos_guarantee_step: int = 8  # if chaos_mode and no chaos by this step, force one random inject
        # NOTE: keep chaos task-agnostic; no scripted per-task chaos.

        # Write-runbook grace step: when incident resolves, allow one extra step
        # so the agent can explicitly write a runbook before termination.
        self._resolved_grace_step: bool = False

        # Runbook memory — persistent across episodes (T2-7)
        self._runbook_memory: RunbookMemory = RunbookMemory()
        self._incident_fingerprint: str = ""
        self._runbook_written: bool = False
        self._runbook_correct: bool = False
        self._runbook_available: bool = False
        self._runbook_used: bool = False
        self._runbook_suggestions: List[Dict[str, Any]] = []

        # Severity escalation state (T2-8)
        self._escalation_tier: int = 1
        self._services_at_risk: List[str] = []

        # Time bounding / SLA pressure (T2-5)
        self._http_mode: bool = http_mode
        self._episode_start_time: float = 0.0
        self._last_step_time: float = 0.0

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
        self._last_chaos_event_persistent = None
        self._new_chaos_event_this_step = None
        self._resolved_grace_step = False

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

        # Severity escalation reset (T2-8)
        self._escalation_tier = 1
        self._services_at_risk = []

        # Time bounding reset (T2-5)
        self._episode_start_time = time.time()
        self._last_step_time = self._episode_start_time

        # Runbook memory — advance episode counter and do lookup (T2-7)
        self._runbook_memory.advance_episode()
        self._incident_fingerprint = self._runbook_memory.build_fingerprint(
            self._task.root_cause_service, task_name
        )
        self._runbook_suggestions = self._runbook_memory.lookup(self._incident_fingerprint)
        self._runbook_available = len(self._runbook_suggestions) > 0
        self._runbook_written = False
        self._runbook_correct = False
        self._runbook_used = False

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

        return self._build_observation(reward=0.0, logs=[], metrics_detail=None, log_quality="full")

    # ------------------------------------------------------------------
    # _tick() — Real-time severity escalation (T2-8)
    # ------------------------------------------------------------------

    def _tick(self, step: int) -> None:
        """
        Escalate damage based on how long incident has been unaddressed.

        Called at the START of each step before returning observation.
        Implements 4 escalation tiers with progressive cascading damage.
        """
        # Count unhealthy services
        degraded = []
        critical = []
        for name, svc in self._services.items():
            if svc.status == ServiceStatusEnum.DEGRADED:
                degraded.append(name)
            elif svc.status == ServiceStatusEnum.DOWN:
                critical.append(name)

        # Determine escalation tier
        if step <= 3:
            self._escalation_tier = 1
        elif step <= 7:
            self._escalation_tier = 2
        elif step <= 12:
            self._escalation_tier = 3
        else:
            self._escalation_tier = 4

        # Tier 2+: Root cause degrades further (only if still unhealthy)
        if step > 3 and self._task:
            root_svc = self._services.get(self._task.root_cause_service)
            if root_svc and root_svc.status != ServiceStatusEnum.HEALTHY:
                new_health_penalty = 0.04 if step > 5 else 0.02
                # Degrade the root cause further — increase error rate, latency
                current_err = root_svc.error_rate
                new_err = min(1.0, current_err + new_health_penalty * 0.5)
                current_lat = root_svc.latency_ms
                new_lat = min(30000.0, current_lat * 1.05)  # 5% increase, capped (Audit Fix #12)

                updates = {"error_rate": round(new_err, 4), "latency_ms": round(new_lat, 1)}

                # Tier 3+: Root cause may become critical if not already
                if step > 8 and root_svc.status == ServiceStatusEnum.DEGRADED:
                    if root_svc.error_rate > 0.6:
                        updates["status"] = ServiceStatusEnum.DOWN

                self._services[self._task.root_cause_service] = root_svc.model_copy(update=updates)

        # Tier 3+: Direct dependents start degrading (only if root is still broken)
        if step > 7 and self._task:
            root_name = self._task.root_cause_service
            root_svc = self._services.get(root_name)
            if root_svc and root_svc.status != ServiceStatusEnum.HEALTHY:
                dependents = REVERSE_DEPS.get(root_name, [])
                for dep_name in dependents:
                    dep_svc = self._services.get(dep_name)
                    if dep_svc and dep_svc.status == ServiceStatusEnum.HEALTHY:
                        self._services[dep_name] = dep_svc.model_copy(update={
                            "status": ServiceStatusEnum.DEGRADED,
                            "error_rate": max(dep_svc.error_rate, 0.15),
                            "latency_ms": max(dep_svc.latency_ms, 500.0),
                        })

        # Tier 4: Full cascade — all unhealthy services worsen
        if step > 12:
            for name, svc in list(self._services.items()):
                if svc.status != ServiceStatusEnum.HEALTHY:
                    new_err = min(1.0, svc.error_rate + 0.03)
                    self._services[name] = svc.model_copy(update={
                        "error_rate": round(new_err, 4),
                    })

        # Determine services at risk from ALL unhealthy services (Audit Fix #8)
        # Derive from all unhealthy, not just root cause — avoids leaking root cause
        self._services_at_risk = []
        if step >= 6:
            unhealthy_names = [
                name for name, svc in self._services.items()
                if svc.status != ServiceStatusEnum.HEALTHY
            ]
            for unh_name in unhealthy_names:
                for dep_name in REVERSE_DEPS.get(unh_name, []):
                    dep_svc = self._services.get(dep_name)
                    if dep_svc and dep_svc.status == ServiceStatusEnum.HEALTHY:
                        if dep_name not in self._services_at_risk:
                            self._services_at_risk.append(dep_name)

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
                log_quality="full",
            )

        # If we were resolved last step, this is the one grace step.
        grace_active = self._resolved_grace_step
        self._resolved_grace_step = False

        self._state.step_count += 1
        self._last_action_error = None

        # NOTE: _tick() moved to AFTER action execution (Audit Fix #5)
        # This ensures health_delta only reflects the agent's action, not escalation damage

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
            return self._build_observation(reward=reward, logs=[], metrics_detail=None, log_quality="full")

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
            return self._build_observation(reward=reward, logs=[], metrics_detail=None, log_quality="full")

        # Execute the action
        logs: List[str] = []
        metrics_detail = None
        log_quality = "full"
        action_description = ""

        if action_type == "inspect_logs":
            logs_result = generate_logs(
                service_name, self._services, self._task.name, self._state.step_count
            )
            logs, log_quality = logs_result
            self._inspected_services.add(service_name)
            if logs:
                action_description = "\n".join(logs)
            else:
                action_description = "No logs found."

        elif action_type == "inspect_metrics":
            metrics_detail = generate_metrics(
                service_name, self._services, self._task.name
            )
            self._inspected_services.add(service_name)
            if metrics_detail:
                # Format metrics simply
                action_description = ", ".join(f"{k}={v}" for k, v in metrics_detail.items())

        elif action_type == "restart_service":
            self._services = apply_restart(
                self._services, service_name, self._task.name
            )
            # Propagate dependency effects
            self._services = propagate_dependencies(self._services, self._task.name)
            action_description = f"Restarted {service_name}."

        elif action_type == "scale_service":
            self._services = apply_scale(self._services, service_name)
            self._services = propagate_dependencies(self._services, self._task.name)
            action_description = f"Scaled {service_name} instances."

        elif action_type == "rollback":
            self._services = apply_rollback(
                self._services, service_name, good_version="v1.0.0"
            )
            self._services = propagate_dependencies(self._services, self._task.name)
            action_description = f"Rolled back {service_name} to v1.0.0."

        elif action_type == "clear_cache":
            self._services = apply_clear_cache(self._services)
            self._services = propagate_dependencies(self._services, self._task.name)
            action_description = "Cleared global cache."

        elif action_type == "escalate":
            self._escalated = True
            self._is_done = True
            action_description = "Incident escalated to human."

        elif action_type == "write_runbook":
            # Agent writes runbook — allow when: episode done, near step limit,
            # OR system is already fully healthy (resolved but _is_done not set yet)
            curr_health_pre = compute_health_score(self._services)
            all_healthy_pre = all(
                s.status == ServiceStatusEnum.HEALTHY for s in self._services.values()
            )
            allowed = (
                self._is_done
                or self._state.step_count >= self._task.max_steps - 1
                or (all_healthy_pre and curr_health_pre >= 0.95)
            )
            if allowed:
                summary = action.metadata.get("summary", "") if action.metadata else ""
                self._runbook_written = True
                if self._task.root_cause_service.lower() in summary.lower():
                    self._runbook_correct = True
                # Treat write_runbook as terminal when system is already resolved
                if all_healthy_pre and curr_health_pre >= 0.95:
                    self._state.is_resolved = True
                    self._is_done = True
                action_description = f"Runbook summary: {summary}"
            else:
                self._last_action_error = "write_runbook only allowed on final step or after resolution"

        elif action_type == "do_nothing":
            action_description = "Agent took no action."

        self._actions_history.append(action_str)

        # --- Check if fix action matches any runbook suggestion (Audit Fix #9) ---
        # Match any recovery action against any entry in the fix sequence
        if (
            not self._runbook_used
            and self._runbook_available
            and action_type in ("restart_service", "scale_service", "rollback", "clear_cache")
        ):
            for suggestion in self._runbook_suggestions:
                fix_seq = suggestion.get("fix_sequence", [])
                if action_str in fix_seq:
                    self._runbook_used = True
                    break

        # --- Chaos injection (before computing reward) ---
        self._last_chaos_event = None
        self._new_chaos_event_this_step = None
        if self._chaos_mode:
            chaos_result = self._chaos_agent.maybe_inject(
                step=self._state.step_count,
                current_services=self._services,
                rng=self._chaos_rng,
                inspected_services=self._inspected_services,
            )
            if chaos_result:
                self._last_chaos_event = chaos_result
                self._last_chaos_event_persistent = chaos_result
                self._new_chaos_event_this_step = chaos_result
                self._services = propagate_dependencies(self._services, self._task.name)
            elif (
                self._state.step_count >= self._chaos_guarantee_step
                and len(self._chaos_agent.injected_services) == 0
            ):
                forced = self._chaos_agent.force_random_inject(
                    step=self._state.step_count,
                    current_services=self._services,
                    rng=self._chaos_rng,
                    inspected_services=self._inspected_services,
                )
                if forced:
                    self._last_chaos_event = forced
                    self._last_chaos_event_persistent = forced
                    self._new_chaos_event_this_step = forced
                    self._services = propagate_dependencies(self._services, self._task.name)

        # NOTE: No task-specific scripted chaos. Chaos mode is purely random injection
        # via ChaosAgent.maybe_inject() (plus the single random guarantee), and should
        # apply consistently to any task.

        # --- Severity escalation tick AFTER action (Audit Fix #5) ---
        # Run tick after action so health_delta reflects agent's impact, not escalation
        self._tick(self._state.step_count)

        # Compute health and reward
        curr_health = compute_health_score(self._services)

        # Check if system is fully resolved
        all_healthy = all(
            s.status == ServiceStatusEnum.HEALTHY for s in self._services.values()
        )
        if all_healthy and curr_health >= 0.95:
            self._state.is_resolved = True
            # End if runbook already written; otherwise allow one grace step to write it.
            if action_type == "write_runbook" or self._runbook_written:
                self._is_done = True
            else:
                self._resolved_grace_step = True

        # If we were already resolved coming into this step (grace step),
        # end the episode after allowing this one action.
        if grace_active and not self._is_done:
            self._is_done = True

        # Check step limit
        if self._state.step_count >= self._task.max_steps:
            self._is_done = True

        # Time bounding: check SLA time limit (HTTP mode only) (T2-5)
        elapsed = time.time() - self._episode_start_time if self._http_mode else 0.0
        if self._http_mode and elapsed > self._task.time_limit_seconds and not self._is_done:
            self._is_done = True
            # Apply time-exceeded terminal penalty in reward below

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
            escalation_tier=self._escalation_tier,
            runbook_used=self._runbook_used and action_type in ("restart_service", "scale_service", "rollback", "clear_cache"),
            elapsed_seconds=elapsed - (self._last_step_time - self._episode_start_time) if self._http_mode else 0.0,
            http_mode=self._http_mode,
        )

        # Time-exceeded penalty (T2-5)
        if self._http_mode and elapsed > self._task.time_limit_seconds:
            reward -= 0.10

        # Chaos survival bonus: if agent handled chaos without further health loss
        health_delta = curr_health - self._prev_health
        if self._last_chaos_event and health_delta >= 0:
            reward += 0.05

        # Runbook write reward (T2-7)
        if action_type == "write_runbook":
            if self._runbook_correct:
                reward += 0.05
            elif self._runbook_written:
                reward += 0.02

        # Record timeline event
        event: Dict[str, Any] = {
            "step": self._state.step_count,
            "event": action_str,
            "health": round(curr_health, 4),
            "health_delta": round(health_delta, 4),
            "reward": round(reward, 4),
            "escalation_tier": self._escalation_tier,
        }
        if action_description:
            event["description"] = action_description
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
        self._last_step_time = time.time()
        self._rewards.append(reward)
        self._state.cumulative_reward += reward
        self._state.actions_taken = list(self._actions_history)
        self._state.services = {
            k: v.model_copy() for k, v in self._services.items()
        }
        self._state.incident_timeline = list(self._timeline)

        # Auto-write runbook at episode end if agent didn't (Audit Fix #10)
        # Only auto-record successful episodes to avoid polluting memory
        if self._is_done and not self._runbook_written and self._task and self._state.is_resolved:
            self._auto_write_runbook()

        return self._build_observation(
            reward=reward, logs=logs, metrics_detail=metrics_detail, log_quality=log_quality
        )

    # ------------------------------------------------------------------
    # _auto_write_runbook() — auto-save episode result to memory (T2-7)
    # ------------------------------------------------------------------

    def _auto_write_runbook(self) -> None:
        """Automatically save episode result to runbook memory at episode end."""
        if not self._task or not self._state:
            return

        # Build fix sequence from recovery actions taken
        fix_actions = [
            a for a in self._actions_history
            if not a.startswith("inspect_") and a != "do_nothing"
            and a != "escalate" and a != "write_runbook"
        ]

        entry = RunbookEntry(
            incident_type=self._incident_fingerprint,
            root_cause_service=self._task.root_cause_service,
            fix_sequence=fix_actions,
            steps_taken=self._state.step_count,
            score=self._state.cumulative_reward,
            summary=f"Auto-recorded: {self._task.root_cause_description}",
        )
        self._runbook_memory.write(entry)

        # Award memory points for auto-written runbook.
        # The agent resolved the incident — auto-runbook captures the fix.
        # Since we know the root cause, mark as correct for grading.
        self._runbook_written = True
        self._runbook_correct = True

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
                "breakdown": {
                    "recovery": 0.0, "efficiency": 0.0,
                    "diagnostics": 0.0, "ordering": 0.0, "memory": 0.0,
                },
                "steps_taken": 0,
                "is_resolved": False,
                "escalated": False,
                "rewards": [],
            }

        elapsed = time.time() - self._episode_start_time if self._http_mode else 0.0

        score, breakdown = grade_episode(
            task=self._task,
            final_services=self._services,
            actions_history=self._actions_history,
            steps_taken=self._state.step_count if self._state else 0,
            is_resolved=self._state.is_resolved if self._state else False,
            escalated=self._escalated,
            runbook_written=self._runbook_written,
            runbook_correct=self._runbook_correct,
            runbook_available=self._runbook_available,
            runbook_used=self._runbook_used,
            elapsed_seconds=elapsed,
            http_mode=self._http_mode,
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
        log_quality: str = "full",
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

        # Build metadata with optional chaos event info and log quality
        obs_metadata: Dict[str, Any] = {}
        if self._new_chaos_event_this_step:
            obs_metadata["new_chaos_event"] = self._new_chaos_event_this_step
        if self._last_chaos_event_persistent:
            obs_metadata["last_chaos_event"] = self._last_chaos_event_persistent
        if log_quality != "full":
            obs_metadata["log_quality"] = log_quality

        # Time bounding info (T2-5)
        if self._http_mode and self._task:
            elapsed = time.time() - self._episode_start_time
            remaining = max(0.0, self._task.time_limit_seconds - elapsed)
            obs_metadata["elapsed_seconds"] = round(elapsed, 2)
            obs_metadata["time_remaining"] = round(remaining, 2)
            # Time pressure classification
            pct_remaining = remaining / self._task.time_limit_seconds if self._task.time_limit_seconds > 0 else 0
            if pct_remaining < 0.2:
                obs_metadata["time_pressure"] = "critical"
            elif pct_remaining < 0.5:
                obs_metadata["time_pressure"] = "high"
            else:
                obs_metadata["time_pressure"] = "normal"

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
            runbook_memory=self._runbook_suggestions,
            escalation_tier=self._escalation_tier,
            services_at_risk=self._services_at_risk,
        )
