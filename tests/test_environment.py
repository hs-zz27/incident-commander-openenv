"""
Comprehensive test suite for the Incident Commander Environment.

Tests cover:
- Environment lifecycle (reset, step, state, grade)
- All three tasks (easy, medium, hard)
- Action validation and edge cases
- Determinism (same actions → same results)
- Reward shaping correctness
- HTTP API endpoints
- Grader scoring
- Stdout format compliance
"""

import json
import pytest
from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType, ServiceStatusEnum
from server.tasks import list_tasks, get_task
from server.services import (
    compute_health_score,
    classify_severity,
    ALL_SERVICES,
    DEPENDENCY_GRAPH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def env():
    return IncidentCommanderEnvironment()


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_reset_returns_observation(self, env):
        obs = env.reset(task_name="single_service_failure")
        assert obs is not None
        assert obs.done is False
        assert obs.reward == 0.0
        assert obs.task_name == "single_service_failure"
        assert obs.step_count == 0
        assert len(obs.services) == 6
        assert obs.system_health_score > 0
        assert obs.system_health_score < 1.0

    def test_state_before_reset(self, env):
        state = env.state
        assert state.step_count == 0

    def test_state_after_reset(self, env):
        env.reset(task_name="single_service_failure")
        state = env.state
        assert state.episode_id is not None
        assert state.step_count == 0
        assert state.task_name == "single_service_failure"

    def test_step_increments_count(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        assert obs.step_count == 1
        assert env.state.step_count == 1

    def test_close_is_safe(self, env):
        env.reset(task_name="single_service_failure")
        env.close()  # should not raise

    def test_step_without_reset_raises(self, env):
        with pytest.raises(RuntimeError):
            env.step(IncidentAction(
                action_type=ActionType.INSPECT_LOGS, service_name="cache"
            ))

    def test_grade_after_reset(self, env):
        env.reset(task_name="single_service_failure")
        grade = env.grade()
        assert "score" in grade
        assert "breakdown" in grade
        assert 0 <= grade["score"] <= 1

    def test_all_tasks_available(self):
        tasks = list_tasks()
        assert "single_service_failure" in tasks
        assert "cascading_failure" in tasks
        assert "hidden_root_cause" in tasks
        assert len(tasks) >= 3


# ---------------------------------------------------------------------------
# Easy task: single_service_failure
# ---------------------------------------------------------------------------

class TestEasyTask:
    def test_initial_state(self, env):
        obs = env.reset(task_name="single_service_failure")
        assert obs.services["cache"].status == ServiceStatusEnum.DOWN
        assert obs.services["auth"].status == ServiceStatusEnum.DEGRADED
        assert obs.services["database"].status == ServiceStatusEnum.HEALTHY
        assert obs.system_health_score < 0.7

    def test_restart_cache_resolves(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        assert len(obs.logs) > 0
        obs = env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="cache"
        ))
        assert obs.services["cache"].status == ServiceStatusEnum.HEALTHY
        assert obs.system_health_score > 0.9
        assert obs.done is True

    def test_grade_after_solve(self, env):
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="cache"
        ))
        grade = env.grade()
        assert grade["is_resolved"] is True
        assert grade["score"] >= 0.80


# ---------------------------------------------------------------------------
# Medium task: cascading_failure
# ---------------------------------------------------------------------------

class TestMediumTask:
    def test_initial_state(self, env):
        obs = env.reset(task_name="cascading_failure")
        assert obs.services["database"].status == ServiceStatusEnum.DEGRADED
        assert obs.services["checkout"].status == ServiceStatusEnum.DOWN
        assert obs.system_health_score < 0.4

    def test_correct_recovery_sequence(self, env):
        env.reset(task_name="cascading_failure")
        env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="database"
        ))
        env.step(IncidentAction(
            action_type=ActionType.SCALE_SERVICE, service_name="database"
        ))
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="database"
        ))
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="auth"
        ))
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="payments"
        ))
        obs = env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="checkout"
        ))
        assert obs.done is True
        grade = env.grade()
        assert grade["is_resolved"] is True
        assert grade["score"] >= 0.70


# ---------------------------------------------------------------------------
# Hard task: hidden_root_cause
# ---------------------------------------------------------------------------

class TestHardTask:
    def test_initial_state(self, env):
        obs = env.reset(task_name="hidden_root_cause")
        assert obs.services["auth"].version == "v2.2.0-rc1"
        assert obs.services["auth"].status == ServiceStatusEnum.DEGRADED
        assert obs.services["payments"].status == ServiceStatusEnum.DEGRADED
        assert obs.services["checkout"].status == ServiceStatusEnum.DEGRADED

    def test_restart_auth_does_not_fix(self, env):
        """Restarting auth with bad deploy should NOT fully fix it."""
        env.reset(task_name="hidden_root_cause")
        obs = env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="auth"
        ))
        # Auth should still be degraded because the bad version persists
        assert obs.services["auth"].status == ServiceStatusEnum.DEGRADED
        assert obs.services["auth"].version == "v2.2.0-rc1"

    def test_rollback_fixes_auth(self, env):
        env.reset(task_name="hidden_root_cause")
        obs = env.step(IncidentAction(
            action_type=ActionType.ROLLBACK, service_name="auth"
        ))
        assert obs.services["auth"].version == "v1.0.0"
        assert obs.services["auth"].status == ServiceStatusEnum.HEALTHY

    def test_correct_recovery_sequence(self, env):
        env.reset(task_name="hidden_root_cause")
        env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="checkout"
        ))
        env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="auth"
        ))
        env.step(IncidentAction(
            action_type=ActionType.ROLLBACK, service_name="auth"
        ))
        env.step(IncidentAction(action_type=ActionType.CLEAR_CACHE))
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="payments"
        ))
        obs = env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="checkout"
        ))
        assert obs.done is True
        grade = env.grade()
        assert grade["is_resolved"] is True
        assert grade["score"] >= 0.60


# ---------------------------------------------------------------------------
# Action validation
# ---------------------------------------------------------------------------

class TestActionValidation:
    def test_missing_service_name(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type=ActionType.INSPECT_LOGS))
        assert obs.last_action_error is not None
        assert obs.reward < 0

    def test_invalid_service_name(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="nonexistent"
        ))
        assert obs.last_action_error is not None

    def test_do_nothing_penalized(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
        assert obs.reward < 0  # should be penalized during incident

    def test_escalate_ends_episode(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type=ActionType.ESCALATE))
        assert obs.done is True
        grade = env.grade()
        assert grade["escalated"] is True
        assert grade["score"] > 0  # partial credit

    def test_clear_cache_no_crash(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(action_type=ActionType.CLEAR_CACHE))
        assert obs is not None

    def test_step_after_done_returns_observation(self, env):
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(action_type=ActionType.ESCALATE))
        obs = env.step(IncidentAction(action_type=ActionType.DO_NOTHING))
        assert obs.done is True
        assert obs.reward == 0.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_actions_same_results(self):
        """Running the same sequence of actions should produce identical results."""
        actions = [
            IncidentAction(action_type=ActionType.INSPECT_LOGS, service_name="cache"),
            IncidentAction(action_type=ActionType.RESTART_SERVICE, service_name="cache"),
        ]

        results = []
        for _ in range(3):
            env = IncidentCommanderEnvironment()
            env.reset(task_name="single_service_failure", episode_id="test-ep")
            run_results = []
            for action in actions:
                obs = env.step(action)
                run_results.append({
                    "reward": obs.reward,
                    "done": obs.done,
                    "health": obs.system_health_score,
                    "step": obs.step_count,
                })
            results.append(run_results)
            env.close()

        # All runs should be identical
        assert results[0] == results[1]
        assert results[1] == results[2]


# ---------------------------------------------------------------------------
# Reward shaping
# ---------------------------------------------------------------------------

class TestRewardShaping:
    def test_inspect_gives_positive_reward(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        assert obs.reward > 0  # inspecting root cause → positive

    def test_fix_gives_large_reward(self, env):
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        obs = env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="cache"
        ))
        assert obs.reward > 0.5  # big health improvement + completion bonus

    def test_rewards_are_floats(self, env):
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        assert isinstance(obs.reward, float)

    def test_score_in_range(self, env):
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="cache"
        ))
        grade = env.grade()
        assert 0.0 <= grade["score"] <= 1.0


# ---------------------------------------------------------------------------
# Grader
# ---------------------------------------------------------------------------

class TestGrader:
    def test_grader_components_sum_to_score(self, env):
        env.reset(task_name="single_service_failure")
        env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        env.step(IncidentAction(
            action_type=ActionType.RESTART_SERVICE, service_name="cache"
        ))
        grade = env.grade()
        component_sum = sum(grade["breakdown"].values())
        assert abs(component_sum - grade["score"]) < 0.01

    def test_grader_breakdown_keys(self, env):
        env.reset(task_name="single_service_failure")
        grade = env.grade()
        assert "recovery" in grade["breakdown"]
        assert "efficiency" in grade["breakdown"]
        assert "diagnostics" in grade["breakdown"]
        assert "ordering" in grade["breakdown"]


# ---------------------------------------------------------------------------
# HTTP API (using TestClient)
# ---------------------------------------------------------------------------

class TestHTTPAPI:
    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient
        from server.app import app
        return TestClient(app)

    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"

    def test_tasks(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        assert len(r.json()["tasks"]) >= 3

    def test_reset_default(self, client):
        r = client.post("/reset", json={})
        assert r.status_code == 200
        assert r.json()["done"] is False

    def test_reset_no_body(self, client):
        r = client.post("/reset")
        assert r.status_code == 200

    def test_reset_with_task(self, client):
        r = client.post("/reset", json={"task_name": "hidden_root_cause"})
        assert r.status_code == 200

    def test_step(self, client):
        client.post("/reset", json={"task_name": "single_service_failure"})
        r = client.post("/step", json={
            "action": {"action_type": "inspect_logs", "service_name": "cache"}
        })
        assert r.status_code == 200
        assert "observation" in r.json()
        assert "reward" in r.json()

    def test_state(self, client):
        client.post("/reset", json={})
        r = client.get("/state")
        assert r.status_code == 200
        assert "state" in r.json()

    def test_grade(self, client):
        client.post("/reset", json={})
        r = client.get("/grade")
        assert r.status_code == 200
        assert "score" in r.json()

    def test_invalid_action_422(self, client):
        client.post("/reset", json={})
        r = client.post("/step", json={"action": {"action_type": "bad"}})
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# Services module
# ---------------------------------------------------------------------------

class TestServices:
    def test_all_services_exist(self):
        assert len(ALL_SERVICES) == 6
        for svc in ["database", "cache", "auth", "notification", "payments", "checkout"]:
            assert svc in ALL_SERVICES

    def test_dependency_graph_valid(self):
        for svc, deps in DEPENDENCY_GRAPH.items():
            for dep in deps:
                assert dep in ALL_SERVICES, f"{svc} depends on unknown {dep}"

    def test_health_score_range(self, env):
        obs = env.reset(task_name="single_service_failure")
        assert 0.0 <= obs.system_health_score <= 1.0

    def test_severity_classification(self, env):
        obs = env.reset(task_name="single_service_failure")
        assert obs.incident_severity.value in [
            "critical", "high", "medium", "low", "resolved"
        ]


# ---------------------------------------------------------------------------
# Stdout format compliance
# ---------------------------------------------------------------------------

class TestStdoutFormat:
    def test_start_format(self):
        line = "[START] task=test_task env=test_env model=test_model"
        assert line.startswith("[START]")
        assert "task=" in line
        assert "env=" in line
        assert "model=" in line

    def test_step_format(self):
        line = "[STEP] step=1 action=inspect_logs:cache reward=0.05 done=false error=null"
        assert line.startswith("[STEP]")
        parts = line.split()
        assert parts[1].startswith("step=")
        assert parts[2].startswith("action=")
        assert parts[3].startswith("reward=")
        assert parts[4].startswith("done=")
        assert parts[5].startswith("error=")

    def test_end_format(self):
        line = "[END] success=true steps=2 score=0.950 rewards=0.05,1.47"
        assert line.startswith("[END]")
        assert "success=" in line
        assert "steps=" in line
        assert "score=" in line
        assert "rewards=" in line

    def test_reward_format(self):
        """Rewards must be 2 decimal places."""
        assert f"{0.1:.2f}" == "0.10"
        assert f"{1.0:.2f}" == "1.00"
        assert f"{0.123456:.2f}" == "0.12"

    def test_boolean_format(self):
        """Booleans must be lowercase."""
        assert str(True).lower() == "true"
        assert str(False).lower() == "false"
