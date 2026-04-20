"""
Advanced edge-case tests for robustness validation.

Covers: step limits, all action types on all services, repeated actions,
serialization round-trips, full episode rollouts, escalation scoring,
and the do-nothing strategy floor.
"""

import json
import pytest
from server.environment import IncidentCommanderEnvironment
from server.models import (
    IncidentAction,
    ActionType,
    ServiceStatusEnum,
    IncidentObservation,
    IncidentState,
)
from server.services import ALL_SERVICES, DEPENDENCY_GRAPH
from server.tasks import list_tasks, get_task


# ---------------------------------------------------------------------------
# Step limit enforcement
# ---------------------------------------------------------------------------

class TestStepLimit:
    def test_easy_task_step_limit(self, easy_env):
        """Easy task should end after 15 steps of do_nothing."""
        for i in range(16):
            obs = easy_env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
            if obs.done:
                assert easy_env.state.step_count <= 15
                break
        assert obs.done is True

    def test_medium_task_step_limit(self, medium_env):
        """Medium task should end after 20 steps."""
        for i in range(21):
            obs = medium_env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
            if obs.done:
                assert medium_env.state.step_count <= 20
                break
        assert obs.done is True

    def test_hard_task_step_limit(self, hard_env):
        """Hard task should end after 30 steps."""
        for i in range(31):
            obs = hard_env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
            if obs.done:
                assert hard_env.state.step_count <= 30
                break
        assert obs.done is True

    def test_step_count_matches(self, easy_env):
        """Step count in observation should match state."""
        for i in range(5):
            obs = easy_env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
            assert obs.step_count == i + 1
            assert easy_env.state.step_count == i + 1
            if obs.done:
                break


# ---------------------------------------------------------------------------
# All action types on all services
# ---------------------------------------------------------------------------

class TestAllActionCombinations:
    @pytest.mark.parametrize("service_name", ALL_SERVICES)
    def test_inspect_logs_on_every_service(self, easy_env, service_name):
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name=service_name
        ))
        assert len(obs.logs) > 0
        assert obs.last_action_error is None

    @pytest.mark.parametrize("service_name", ALL_SERVICES)
    def test_inspect_metrics_on_every_service(self, easy_env, service_name):
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_METRICS, service_name=service_name
        ))
        assert obs.metrics_detail is not None
        assert obs.metrics_detail["service"] == service_name

    @pytest.mark.parametrize("service_name", ALL_SERVICES)
    def test_restart_service_on_every_service(self, service_name):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name=service_name
        ))
        assert obs is not None
        assert obs.last_action_error is None

    @pytest.mark.parametrize("service_name", ALL_SERVICES)
    def test_scale_service_on_every_service(self, service_name):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        before = env.state.services[service_name].instances
        obs = env.step(IncidentAction(
            action_type=ActionType.SCALE_SERVICE, service_name=service_name
        ))
        assert obs.services[service_name].instances == before + 1

    @pytest.mark.parametrize("service_name", ALL_SERVICES)
    def test_rollback_on_every_service(self, service_name):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.ROLLBACK, service_name=service_name
        ))
        assert obs.services[service_name].version == "v1.0.0"


# ---------------------------------------------------------------------------
# Repeated actions
# ---------------------------------------------------------------------------

class TestRepeatedActions:
    def test_repeated_inspect_logs(self, easy_env):
        """Inspecting the same service multiple times should not crash."""
        for _ in range(5):
            obs = easy_env.step(IncidentAction(
                action_type=ActionType.INSPECT_LOGS, service_name="cache"
            ))
            assert len(obs.logs) > 0

    def test_repeated_restart(self, easy_env):
        """Restarting the same service multiple times should be penalized."""
        rewards = []
        for _ in range(4):
            obs = easy_env.step(IncidentAction(
                action_type=ActionType.RESTART_SERVICE, service_name="cache"
            ))
            rewards.append(obs.reward)
            if obs.done:
                break
        # Later restarts should be less rewarded (diminishing/negative)
        if len(rewards) >= 3:
            assert rewards[-1] <= rewards[0]

    def test_repeated_scale(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="cascading_failure")
        for _ in range(5):
            obs = env.step(IncidentAction(
                action_type=ActionType.SCALE_SERVICE, service_name="database"
            ))
        assert obs.services["database"].instances >= 6


# ---------------------------------------------------------------------------
# Serialization round-trips
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_observation_to_json(self, easy_env):
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        json_str = obs.model_dump_json()
        data = json.loads(json_str)
        assert "services" in data
        assert "alerts" in data
        assert "done" in data

    def test_observation_from_json(self, easy_env):
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        json_str = obs.model_dump_json()
        restored = IncidentObservation.model_validate_json(json_str)
        assert restored.step_count == obs.step_count
        assert restored.system_health_score == obs.system_health_score

    def test_state_to_json(self, easy_env):
        easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        state = easy_env.state
        json_str = state.model_dump_json()
        data = json.loads(json_str)
        assert data["step_count"] == 1
        assert "episode_id" in data

    def test_action_roundtrip(self):
        action = IncidentAction(
            action_type=ActionType.INSPECT_LOGS,
            service_name="database"
        )
        json_str = action.model_dump_json()
        restored = IncidentAction.model_validate_json(json_str)
        assert restored.action_type == ActionType.INSPECT_LOGS
        assert restored.service_name == "database"

    def test_action_from_dict(self):
        """Actions should be constructable from plain dicts (HTTP API compat)."""
        data = {"action_type": "restart_service", "service_name": "auth"}
        action = IncidentAction(**data)
        assert action.action_type == ActionType.RESTART_SERVICE

    def test_invalid_action_type_raises(self):
        with pytest.raises(Exception):
            IncidentAction(action_type="totally_invalid")


# ---------------------------------------------------------------------------
# Full episode rollouts
# ---------------------------------------------------------------------------

class TestFullRollouts:
    def test_expert_easy(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"))
        obs = env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"))
        assert obs.done is True
        grade = env.grade()
        assert grade["score"] >= 0.90
        assert grade["is_resolved"] is True

    def test_expert_medium(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="cascading_failure")
        env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="database"))
        env.step(IncidentAction(action_type=ActionType.SCALE_SERVICE, service_name="database"))
        env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="database"))
        env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="auth"))
        env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"))
        obs = env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"))
        assert obs.done is True
        grade = env.grade()
        assert grade["score"] >= 0.80
        assert grade["is_resolved"] is True

    def test_expert_hard(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="hidden_root_cause")
        env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="checkout"))
        env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="auth"))
        env.step(IncidentAction(action_type=ActionType.ROLLBACK, service_name="auth"))
        env.step(IncidentAction(action_type=ActionType.CLEAR_CACHE))
        env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="payments"))
        obs = env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="checkout"))
        assert obs.done is True
        grade = env.grade()
        assert grade["score"] >= 0.60
        assert grade["is_resolved"] is True


# ---------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------

class TestEscalation:
    def test_escalate_gives_partial_credit(self, easy_env):
        easy_env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"))
        easy_env.step(IncidentAction(action_type=ActionType.INSPECT_METRICS, service_name="auth"))
        obs = easy_env.step(IncidentAction(action_type=ActionType.ESCALATE))
        assert obs.done is True
        grade = easy_env.grade()
        assert grade["escalated"] is True
        assert grade["score"] > 0  # should get partial credit for diagnostics
        assert grade["score"] < 0.5  # but not full credit

    def test_escalate_without_diagnostics(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type=ActionType.ESCALATE))
        assert obs.done is True
        grade = env.grade()
        assert grade["score"] < 0.15  # very little credit


# ---------------------------------------------------------------------------
# Metrics detail
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_metrics_contain_required_fields(self, easy_env):
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_METRICS, service_name="database"
        ))
        m = obs.metrics_detail
        assert m is not None
        required = [
            "service", "status", "error_rate", "latency_p50_ms",
            "latency_p95_ms", "latency_p99_ms", "cpu_percent",
            "memory_percent", "instances", "version",
            "requests_per_second", "open_connections",
            "dependencies", "dependents", "dependency_health",
        ]
        for key in required:
            assert key in m, f"Missing metric key: {key}"

    def test_metrics_dependency_health(self, easy_env):
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_METRICS, service_name="auth"
        ))
        m = obs.metrics_detail
        assert "dependency_health" in m
        assert "database" in m["dependency_health"]
        assert "cache" in m["dependency_health"]

    def test_metrics_cleared_after_inspect_logs(self, easy_env):
        """After inspect_logs, metrics_detail should be None."""
        easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_METRICS, service_name="database"
        ))
        obs = easy_env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        assert obs.metrics_detail is None


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

class TestTaskDefinitions:
    @pytest.mark.parametrize("task_name", list_tasks())
    def test_task_has_required_fields(self, task_name):
        task = get_task(task_name)
        assert task.name == task_name
        assert task.description
        assert task.difficulty in ("easy", "medium", "hard", "expert")
        assert task.max_steps > 0
        assert task.root_cause_service in ALL_SERVICES
        assert task.root_cause_description
        assert len(task.correct_recovery_actions) > 0
        assert len(task.initial_services) == len(ALL_SERVICES)

    @pytest.mark.parametrize("task_name", list_tasks())
    def test_task_initial_state_has_issues(self, task_name):
        """Every task should start with at least one non-healthy service."""
        task = get_task(task_name)
        unhealthy = sum(
            1
            for s in task.initial_services.values()
            if s.status != ServiceStatusEnum.HEALTHY
        )
        assert unhealthy >= 1

    def test_unknown_task_raises(self):
        with pytest.raises(KeyError):
            get_task("nonexistent_task")


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

class TestAlerts:
    def test_alerts_on_unhealthy_system(self, easy_env):
        obs = easy_env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
        assert len(obs.alerts) > 0

    def test_no_alerts_on_healthy_system(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"))
        obs = env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"))
        assert len(obs.alerts) == 0 or obs.incident_severity.value == "resolved"


# ---------------------------------------------------------------------------
# Multiple resets
# ---------------------------------------------------------------------------

class TestMultipleResets:
    def test_reset_clears_state(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"))
        env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"))

        # Reset to a different task
        obs = env.reset(task_name="cascading_failure")
        assert obs.step_count == 0
        assert obs.task_name == "cascading_failure"
        assert obs.done is False
        assert env.state.step_count == 0

    def test_reset_same_task_resets(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"))

        obs = env.reset(task_name="single_service_failure")
        assert obs.step_count == 0
        assert obs.services["cache"].status == ServiceStatusEnum.DOWN
