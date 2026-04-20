"""
Task definitions for the Incident Commander Environment.

Each task defines:
- initial service states (the "broken" cluster)
- task metadata (name, max_steps, description)
- root cause information for the grader
- the sequence of correct recovery actions
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .models import ServiceState, ServiceStatusEnum
from .services import DEPENDENCY_GRAPH, REVERSE_DEPS


@dataclass(frozen=True)
class TaskDefinition:
    """Immutable description of a single task/scenario."""

    name: str
    description: str
    difficulty: str  # "easy", "medium", "hard"
    max_steps: int
    root_cause_service: str
    root_cause_description: str
    correct_recovery_actions: List[str]   # ordered list of action strings
    initial_services: Dict[str, ServiceState]


# ---------------------------------------------------------------------------
# Task 1 — Easy: Single Service Failure
# ---------------------------------------------------------------------------

def _build_easy_task() -> TaskDefinition:
    """Cache is DOWN → auth is DEGRADED. Fix: restart cache."""
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=20.0,
            cpu_percent=12.0, memory_percent=28.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.DOWN,
            error_rate=1.0, latency_ms=0.0,
            cpu_percent=0.0, memory_percent=0.0,
            instances=0, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.40, latency_ms=2500.0,
            cpu_percent=65.0, memory_percent=55.0,
            instances=2, version="v1.0.0",
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=30.0,
            cpu_percent=10.0, memory_percent=20.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=40.0,
            cpu_percent=18.0, memory_percent=32.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.30, latency_ms=3000.0,
            cpu_percent=55.0, memory_percent=50.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="single_service_failure",
        description=(
            "A single service (cache) has crashed, causing degraded performance "
            "in dependent services. Identify the failed service and restart it."
        ),
        difficulty="easy",
        max_steps=15,
        root_cause_service="cache",
        root_cause_description="Cache service OOM crash — needs restart",
        correct_recovery_actions=[
            "restart_service:cache",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Task 2 — Medium: Cascading Failure
# ---------------------------------------------------------------------------

def _build_medium_task() -> TaskDefinition:
    """Database overloaded → auth, payments, checkout cascade."""
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.25, latency_ms=4500.0,
            cpu_percent=92.0, memory_percent=85.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=15.0,
            cpu_percent=10.0, memory_percent=25.0,
            instances=2, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.30, latency_ms=3200.0,
            cpu_percent=70.0, memory_percent=60.0,
            instances=2, version="v1.0.0",
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=25.0,
            cpu_percent=8.0, memory_percent=18.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.35, latency_ms=5000.0,
            cpu_percent=75.0, memory_percent=65.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DOWN,
            error_rate=0.85, latency_ms=8000.0,
            cpu_percent=95.0, memory_percent=90.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="cascading_failure",
        description=(
            "The database is under heavy load, causing cascading failures across "
            "auth, payments, and checkout. Scale the database first, then restart "
            "dependent services in the correct dependency order."
        ),
        difficulty="medium",
        max_steps=20,
        root_cause_service="database",
        root_cause_description="Database overloaded — needs scaling and restart before dependents can recover",
        correct_recovery_actions=[
            "scale_service:database",
            "restart_service:database",
            "restart_service:auth",
            "restart_service:payments",
            "restart_service:checkout",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Task 3 — Hard: Hidden Root Cause
# ---------------------------------------------------------------------------

def _build_hard_task() -> TaskDefinition:
    """
    Symptoms in checkout/payments. Red herring: payments looks broken.
    True root cause: auth has bad deploy v2.2.0-rc1 causing 401 errors.
    Cache is serving stale tokens, masking the issue intermittently.
    """
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=22.0,
            cpu_percent=14.0, memory_percent=30.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=12.0,
            cpu_percent=15.0, memory_percent=35.0,
            instances=2, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.32, latency_ms=600.0,
            cpu_percent=45.0, memory_percent=40.0,
            instances=2, version="v2.2.0-rc1",  # <-- BAD DEPLOY
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=20.0,
            cpu_percent=8.0, memory_percent=15.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.38, latency_ms=2800.0,
            cpu_percent=60.0, memory_percent=55.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.45, latency_ms=4000.0,
            cpu_percent=70.0, memory_percent=60.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="hidden_root_cause",
        description=(
            "Checkout and payments are experiencing high error rates. "
            "The symptoms are misleading — investigate carefully to find "
            "the true root cause before taking recovery actions."
        ),
        difficulty="hard",
        max_steps=30,
        root_cause_service="auth",
        root_cause_description=(
            "Auth service deployed v2.2.0-rc1 with JWT signature algorithm mismatch. "
            "Cache is serving stale tokens masking the issue intermittently."
        ),
        correct_recovery_actions=[
            "rollback:auth",
            "clear_cache",
            "restart_service:payments",
            "restart_service:checkout",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Task 4 — Hard: Chaos Cascade (T2-1)
# ---------------------------------------------------------------------------

def _build_chaos_cascade_task() -> TaskDefinition:
    """
    DB fails first. At step 8 a chaos event fires: notification also fails.
    Agent must handle both failures. The "wow" demo task.
    """
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.DOWN,
            error_rate=0.90, latency_ms=9000.0,
            cpu_percent=98.0, memory_percent=95.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=15.0,
            cpu_percent=10.0, memory_percent=25.0,
            instances=2, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.35, latency_ms=3500.0,
            cpu_percent=72.0, memory_percent=60.0,
            instances=2, version="v1.0.0",
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=25.0,
            cpu_percent=8.0, memory_percent=18.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.40, latency_ms=5500.0,
            cpu_percent=80.0, memory_percent=70.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DOWN,
            error_rate=0.90, latency_ms=10000.0,
            cpu_percent=96.0, memory_percent=92.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="chaos_cascade",
        description=(
            "The database has crashed, taking auth, payments, and checkout down with it. "
            "A secondary chaos event will inject a notification failure mid-episode. "
            "Handle both the primary and secondary failures to fully restore the system."
        ),
        difficulty="hard",
        max_steps=35,
        root_cause_service="database",
        root_cause_description=(
            "Database crashed under extreme load. Notification service will independently "
            "fail at step 8 due to chaos injection."
        ),
        correct_recovery_actions=[
            "restart_service:database",
            "restart_service:auth",
            "restart_service:payments",
            "restart_service:checkout",
            "restart_service:notification",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Task 5 — Expert: Multi-Root Cause (T2-2)
# ---------------------------------------------------------------------------

def _build_multi_root_cause_task() -> TaskDefinition:
    """
    TWO simultaneous root causes: auth bad deploy + database CPU spike.
    Neither fully resolves without fixing both.
    """
    services = {
        "database": ServiceState(
            name="database", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.30, latency_ms=5000.0,
            cpu_percent=96.0, memory_percent=88.0,
            instances=2, version="v1.0.0",
        ),
        "cache": ServiceState(
            name="cache", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.02, latency_ms=18.0,
            cpu_percent=12.0, memory_percent=28.0,
            instances=2, version="v1.0.0",
        ),
        "auth": ServiceState(
            name="auth", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.40, latency_ms=1200.0,
            cpu_percent=55.0, memory_percent=48.0,
            instances=2, version="v2.2.0-rc1",  # BAD DEPLOY
        ),
        "notification": ServiceState(
            name="notification", status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01, latency_ms=22.0,
            cpu_percent=9.0, memory_percent=16.0,
            instances=1, version="v1.0.0",
        ),
        "payments": ServiceState(
            name="payments", status=ServiceStatusEnum.DEGRADED,
            error_rate=0.45, latency_ms=4500.0,
            cpu_percent=70.0, memory_percent=62.0,
            instances=2, version="v1.0.0",
        ),
        "checkout": ServiceState(
            name="checkout", status=ServiceStatusEnum.DOWN,
            error_rate=0.80, latency_ms=9000.0,
            cpu_percent=90.0, memory_percent=85.0,
            instances=2, version="v1.0.0",
        ),
    }
    return TaskDefinition(
        name="multi_root_cause",
        description=(
            "Multiple simultaneous root causes: auth has a bad deployment AND the database "
            "is experiencing a CPU spike. Both must be addressed — fixing only one will leave "
            "the system degraded. This requires careful investigation to discover both issues."
        ),
        difficulty="expert",
        max_steps=40,
        root_cause_service="auth",  # Primary root cause for grader
        root_cause_description=(
            "Auth deployed v2.2.0-rc1 with JWT mismatch AND database has CPU spike. "
            "Both must be fixed for full resolution."
        ),
        correct_recovery_actions=[
            "rollback:auth",
            "scale_service:database",
            "restart_service:database",
            "clear_cache",
            "restart_service:payments",
            "restart_service:checkout",
        ],
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Randomized Incident Generator (T1-1)
# ---------------------------------------------------------------------------

# Failure mode templates: how they affect a service's metrics
_FAILURE_PROFILES = {
    "oom": {
        "status": ServiceStatusEnum.DOWN,
        "error_rate": 1.0,
        "latency_ms": 0.0,
        "cpu_percent": 0.0,
        "memory_percent": 0.0,
        "instances": 0,
    },
    "bad_deploy": {
        "status": ServiceStatusEnum.DEGRADED,
        "error_rate": 0.35,
        "latency_ms": 800.0,
        "cpu_percent": 50.0,
        "memory_percent": 45.0,
        "version": "v2.2.0-rc1",
    },
    "cpu_spike": {
        "status": ServiceStatusEnum.DEGRADED,
        "error_rate": 0.28,
        "latency_ms": 4500.0,
        "cpu_percent": 95.0,
        "memory_percent": 82.0,
    },
    "network_partition": {
        "status": ServiceStatusEnum.DOWN,
        "error_rate": 0.95,
        "latency_ms": 15000.0,
        "cpu_percent": 5.0,
        "memory_percent": 20.0,
        "instances": 1,
    },
}

# Maps failure mode → correct primary fix action
_FAILURE_FIX = {
    "oom": "restart_service",
    "bad_deploy": "rollback",
    "cpu_spike": "scale_service",
    "network_partition": "restart_service",
}

# Degradation profile for downstream services affected by a failed dependency
_DOWNSTREAM_DEGRADATION = {
    "status": ServiceStatusEnum.DEGRADED,
    "error_rate": 0.30,
    "latency_ms": 3000.0,
    "cpu_percent": 65.0,
    "memory_percent": 55.0,
}


def _build_random_task(seed: Optional[int] = None) -> TaskDefinition:
    """
    Build a randomized incident task.

    Randomizes: root cause service, failure mode, and which 1-3
    downstream services are affected. Uses the dependency graph
    to determine realistic failure propagation.
    """
    rng = random.Random(seed)

    # Choose root cause service (any of the six)
    root_candidates = ["database", "cache", "auth", "payments", "notification", "checkout"]
    root_service = rng.choice(root_candidates)

    # Choose failure mode
    failure_mode = rng.choice(["oom", "bad_deploy", "cpu_spike", "network_partition"])

    # Build healthy baseline for all services
    services = {}
    for svc_name in root_candidates:
        services[svc_name] = ServiceState(
            name=svc_name,
            status=ServiceStatusEnum.HEALTHY,
            error_rate=0.01,
            latency_ms=25.0,
            cpu_percent=15.0,
            memory_percent=30.0,
            instances=2,
            version="v1.0.0",
        )

    # Apply failure to root cause service
    failure_profile = _FAILURE_PROFILES[failure_mode]
    root_updates = {
        "status": failure_profile["status"],
        "error_rate": failure_profile["error_rate"],
        "latency_ms": failure_profile["latency_ms"],
        "cpu_percent": failure_profile["cpu_percent"],
        "memory_percent": failure_profile["memory_percent"],
    }
    if "instances" in failure_profile:
        root_updates["instances"] = failure_profile["instances"]
    if "version" in failure_profile:
        root_updates["version"] = failure_profile["version"]

    services[root_service] = services[root_service].model_copy(update=root_updates)

    # Determine downstream services (those that depend on root_service)
    potential_downstream = REVERSE_DEPS.get(root_service, [])

    # Also consider transitive dependents
    all_downstream = set(potential_downstream)
    for ds in list(all_downstream):
        all_downstream.update(REVERSE_DEPS.get(ds, []))
    all_downstream.discard(root_service)

    # Pick 1-3 downstream services to show degradation
    if all_downstream:
        num_affected = rng.randint(1, min(3, len(all_downstream)))
        affected = rng.sample(sorted(all_downstream), num_affected)
    else:
        affected = []

    for ds_name in affected:
        ds_updates = dict(_DOWNSTREAM_DEGRADATION)
        # Add some randomized variance
        ds_updates["error_rate"] = round(rng.uniform(0.15, 0.50), 2)
        ds_updates["latency_ms"] = round(rng.uniform(1500.0, 5000.0), 1)
        ds_updates["cpu_percent"] = round(rng.uniform(50.0, 85.0), 1)
        ds_updates["memory_percent"] = round(rng.uniform(45.0, 75.0), 1)
        services[ds_name] = services[ds_name].model_copy(update=ds_updates)

    # Build recovery actions
    primary_fix = _FAILURE_FIX[failure_mode]
    recovery_actions = [f"{primary_fix}:{root_service}"]
    if failure_mode == "bad_deploy":
        recovery_actions.append("clear_cache")
    for ds_name in affected:
        recovery_actions.append(f"restart_service:{ds_name}")

    # Difficulty scales with number of affected services
    total_broken = 1 + len(affected)
    if total_broken <= 1:
        difficulty = "easy"
        max_steps = 15
    elif total_broken <= 2:
        difficulty = "medium"
        max_steps = 20
    else:
        difficulty = "hard"
        max_steps = 25

    mode_descriptions = {
        "oom": f"{root_service} crashed due to OOM — needs restart",
        "bad_deploy": f"{root_service} has bad deploy v2.2.0-rc1 — needs rollback",
        "cpu_spike": f"{root_service} experiencing CPU spike — needs scaling",
        "network_partition": f"{root_service} network partitioned — needs restart after resolution",
    }

    affected_str = ", ".join(affected) if affected else "none"
    return TaskDefinition(
        name="random_incident",
        description=(
            f"Randomized incident: {root_service} is failing ({failure_mode}), "
            f"affecting downstream services: {affected_str}. "
            f"Diagnose and resolve the incident."
        ),
        difficulty=difficulty,
        max_steps=max_steps,
        root_cause_service=root_service,
        root_cause_description=mode_descriptions[failure_mode],
        correct_recovery_actions=recovery_actions,
        initial_services=services,
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_REGISTRY: Dict[str, TaskDefinition] = {}


def _register_tasks() -> None:
    global TASK_REGISTRY
    TASK_REGISTRY = {
        "single_service_failure": _build_easy_task(),
        "cascading_failure": _build_medium_task(),
        "hidden_root_cause": _build_hard_task(),
        "chaos_cascade": _build_chaos_cascade_task(),
        "multi_root_cause": _build_multi_root_cause_task(),
        "random_incident": _build_random_task(seed=42),  # default deterministic
    }


_register_tasks()


def get_task(name: str, seed: Optional[int] = None) -> TaskDefinition:
    """
    Retrieve a task definition by name. Raises KeyError if not found.

    For 'random_incident', pass seed=None to get a fresh random task,
    or a specific int seed for reproducibility.
    """
    if name == "random_incident":
        return _build_random_task(seed=seed)
    if name not in TASK_REGISTRY:
        raise KeyError(
            f"Unknown task '{name}'. Available: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[name]


def list_tasks() -> List[str]:
    """Return list of available task names."""
    return list(TASK_REGISTRY.keys())
