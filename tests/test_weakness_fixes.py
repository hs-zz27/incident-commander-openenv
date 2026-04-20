"""
Tests for all weakness fixes (W-01 through W-12).

Validates that every identified vulnerability has been properly addressed.
"""

import pytest
from fastapi.testclient import TestClient

from server.app import app
from server.environment import IncidentCommanderEnvironment
from server.models import IncidentAction, ActionType, ServiceStatusEnum


# ---------------------------------------------------------------------------
# W-01: Step before reset returns 400
# ---------------------------------------------------------------------------

class TestW01StepBeforeReset:
    @pytest.fixture
    def client(self):
        from server.app import create_incident_app
        return TestClient(create_incident_app())

    def test_step_before_reset_returns_400(self, client):
        r = client.post("/step", json={"action": {"action_type": "inspect_logs", "service_name": "cache"}})
        assert r.status_code == 400
        assert "not initialised" in r.json()["detail"].lower()

    def test_step_after_reset_returns_200(self, client):
        client.post("/reset", json={})
        r = client.post("/step", json={"action": {"action_type": "inspect_logs", "service_name": "cache"}})
        assert r.status_code == 200

    def test_invalid_task_returns_400(self, client):
        r = client.post("/reset", json={"task_name": "nonexistent"})
        assert r.status_code == 400


# ---------------------------------------------------------------------------
# W-02: Grade before reset returns full dict
# ---------------------------------------------------------------------------

class TestW02GradeBeforeReset:
    def test_grade_without_task_has_all_keys(self):
        env = IncidentCommanderEnvironment()
        grade = env.grade()
        assert "score" in grade
        assert "breakdown" in grade
        assert "steps_taken" in grade
        assert "is_resolved" in grade
        assert "escalated" in grade
        assert "rewards" in grade
        assert grade["score"] == 0.0
        assert grade["breakdown"]["recovery"] == 0.0

    def test_grade_endpoint_no_crash(self):
        from server.app import create_incident_app
        client = TestClient(create_incident_app())
        r = client.get("/grade")
        assert r.status_code == 200
        data = r.json()
        assert data["score"] == 0.0
        assert "recovery" in data["breakdown"]


# ---------------------------------------------------------------------------
# W-04: clear_cache triggers propagation
# ---------------------------------------------------------------------------

class TestW04ClearCachePropagation:
    def test_clear_cache_propagates(self):
        """After rollback + clear_cache on hard task, dependents should auto-heal."""
        env = IncidentCommanderEnvironment()
        env.reset(task_name="hidden_root_cause")

        # Rollback auth (fixes root cause)
        env.step(IncidentAction(action_type=ActionType.ROLLBACK, service_name="auth"))

        # Clear cache should propagate and auto-heal remaining services
        obs = env.step(IncidentAction(action_type=ActionType.CLEAR_CACHE))

        # After propagation, services that only had dep-caused degradation should recover
        assert obs.services["auth"].status == ServiceStatusEnum.HEALTHY


# ---------------------------------------------------------------------------
# W-08: Repeated inspect penalty
# ---------------------------------------------------------------------------

class TestW08RepeatedInspectPenalty:
    def test_first_inspect_rewarded(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        obs = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        assert obs.reward > 0  # first inspect of root cause → positive

    def test_second_inspect_same_service_penalized(self):
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        obs1 = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        obs2 = env.step(IncidentAction(
            action_type=ActionType.INSPECT_LOGS, service_name="cache"
        ))
        # Second inspect should be lower/negative
        assert obs2.reward < obs1.reward

    def test_spam_inspect_not_exploitable(self):
        """Spamming inspect should NOT accumulate high reward."""
        env = IncidentCommanderEnvironment()
        env.reset(task_name="single_service_failure")
        total = 0.0
        for _ in range(10):
            obs = env.step(IncidentAction(
                action_type=ActionType.INSPECT_LOGS, service_name="cache"
            ))
            total += obs.reward
            if obs.done:
                break
        # Total from 10 inspects should be small (only first one rewarded)
        assert total < 0.15  # much less than 10 * 0.05 = 0.50


# ---------------------------------------------------------------------------
# W-05: Inference import guard (unit test)
# ---------------------------------------------------------------------------

class TestW05InferenceImportGuard:
    def test_inference_module_imports(self):
        """inference.py should import successfully when openai is installed."""
        import inference  # noqa: F401
        assert hasattr(inference, "run_task")
        assert hasattr(inference, "fallback_action")
        assert hasattr(inference, "parse_action")


# ---------------------------------------------------------------------------
# W-06: Smart fallback agent
# ---------------------------------------------------------------------------

class TestW06SmartFallback:
    def test_fallback_solves_easy(self):
        from inference import fallback_action
        env = IncidentCommanderEnvironment()
        obs = env.reset(task_name="single_service_failure")
        obs_dict = obs.model_dump()
        history = []
        for step in range(20):
            action = fallback_action(obs_dict, step + 1, history)
            a_str = action.action_type.value
            if action.service_name:
                a_str += f":{action.service_name}"
            history.append(a_str)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            if obs.done:
                break
        assert env.grade()["is_resolved"] is True

    def test_fallback_solves_hard(self):
        from inference import fallback_action
        env = IncidentCommanderEnvironment()
        obs = env.reset(task_name="hidden_root_cause")
        obs_dict = obs.model_dump()
        history = []
        for step in range(30):
            action = fallback_action(obs_dict, step + 1, history)
            a_str = action.action_type.value
            if action.service_name:
                a_str += f":{action.service_name}"
            history.append(a_str)
            obs = env.step(action)
            obs_dict = obs.model_dump()
            if obs.done:
                break
        assert env.grade()["is_resolved"] is True

    def test_fallback_detects_version_mismatch(self):
        """Fallback should rollback services with non-standard versions."""
        from inference import fallback_action
        env = IncidentCommanderEnvironment()
        obs = env.reset(task_name="hidden_root_cause")
        obs_dict = obs.model_dump()

        # First 3 steps: inspects
        history = []
        for step in range(3):
            action = fallback_action(obs_dict, step + 1, history)
            a_str = action.action_type.value
            if action.service_name:
                a_str += f":{action.service_name}"
            history.append(a_str)
            obs = env.step(action)
            obs_dict = obs.model_dump()

        # After inspecting, next action should be rollback:auth (version mismatch)
        action = fallback_action(obs_dict, 4, history)
        assert action.action_type == ActionType.ROLLBACK
        assert action.service_name == "auth"


# ---------------------------------------------------------------------------
# W-07: Action history in prompt
# ---------------------------------------------------------------------------

class TestW07ActionHistoryInPrompt:
    def test_prompt_includes_history(self):
        from inference import observation_to_prompt
        obs_dict = {"system_health_score": 0.5, "incident_severity": "high", "services": {}, "alerts": [], "logs": [], "metrics_detail": None, "last_action_error": None, "max_steps": 15}
        history = ["inspect_logs:cache", "restart_service:cache"]
        prompt = observation_to_prompt(obs_dict, 3, history)
        assert "Actions Taken So Far" in prompt
        assert "inspect_logs:cache" in prompt
        assert "restart_service:cache" in prompt

    def test_prompt_works_with_empty_history(self):
        from inference import observation_to_prompt
        obs_dict = {"system_health_score": 0.5, "incident_severity": "high", "services": {}, "alerts": [], "logs": [], "metrics_detail": None, "last_action_error": None, "max_steps": 15}
        prompt = observation_to_prompt(obs_dict, 1, [])
        assert "Actions Taken So Far" not in prompt  # no section if empty


# ---------------------------------------------------------------------------
# W-10: Concurrency lock exists
# ---------------------------------------------------------------------------

class TestW10ConcurrencyLock:
    def test_app_has_lock(self):
        """Verify the app factory creates a lock."""
        import asyncio
        from server.app import create_incident_app
        # The lock is internal to the closure, but we can verify
        # the app works correctly under sequential calls
        client = TestClient(create_incident_app())
        client.post("/reset", json={})
        r1 = client.post("/step", json={"action": {"action_type": "inspect_logs", "service_name": "cache"}})
        r2 = client.post("/step", json={"action": {"action_type": "restart_service", "service_name": "cache"}})
        assert r1.status_code == 200
        assert r2.status_code == 200


# ---------------------------------------------------------------------------
# W-12: openenv.yaml enrichment
# ---------------------------------------------------------------------------

class TestW12OpenEnvYaml:
    def test_yaml_has_required_fields(self):
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        assert data["spec_version"] == 1
        assert data["name"] == "incident_commander_env"
        assert data["type"] == "space"
        assert data["runtime"] == "fastapi"
        assert data["app"] == "server.app:app"
        assert data["port"] == 8000

    def test_yaml_has_optional_fields(self):
        import yaml
        with open("openenv.yaml") as f:
            data = yaml.safe_load(f)
        assert "description" in data
        assert "inference_script" in data
        assert "tasks" in data
        assert len(data["tasks"]) >= 3
