"""
Microservices dependency graph and simulation engine.

Models a production microservices system with realistic dependency relationships.
All state transitions are deterministic given the same seed and action sequence.
"""

from __future__ import annotations

import copy
import random
import uuid
from typing import Any, Dict, List, Optional

from .models import ServiceState, ServiceStatusEnum


# ---------------------------------------------------------------------------
# Dependency graph  (service -> list of services it depends on)
# ---------------------------------------------------------------------------
DEPENDENCY_GRAPH: Dict[str, List[str]] = {
    "database": [],
    "cache": [],
    "auth": ["database", "cache"],
    "notification": [],
    "payments": ["database", "notification"],
    "checkout": ["auth", "payments", "database"],
}

# Reverse map: service -> list of services that depend on it
REVERSE_DEPS: Dict[str, List[str]] = {svc: [] for svc in DEPENDENCY_GRAPH}
for _svc, _deps in DEPENDENCY_GRAPH.items():
    for _d in _deps:
        REVERSE_DEPS[_d].append(_svc)

ALL_SERVICES = list(DEPENDENCY_GRAPH.keys())


# ---------------------------------------------------------------------------
# Healthy baseline for each service
# ---------------------------------------------------------------------------
def _healthy_service(name: str, version: str = "v1.0.0") -> ServiceState:
    return ServiceState(
        name=name,
        status=ServiceStatusEnum.HEALTHY,
        error_rate=0.01,
        latency_ms=25.0,
        cpu_percent=15.0,
        memory_percent=30.0,
        instances=2,
        version=version,
    )


def build_healthy_cluster(versions: Optional[Dict[str, str]] = None) -> Dict[str, ServiceState]:
    """Return a fully healthy service cluster."""
    versions = versions or {}
    return {
        name: _healthy_service(name, versions.get(name, "v1.0.0"))
        for name in ALL_SERVICES
    }


# ---------------------------------------------------------------------------
# Log generation (deterministic per service + scenario)
# ---------------------------------------------------------------------------

def generate_logs(
    service_name: str,
    services: Dict[str, ServiceState],
    task_name: str,
    step: int,
) -> List[str]:
    """Generate realistic log lines for a service based on its current state and the task."""
    svc = services.get(service_name)
    if svc is None:
        return [f"[ERROR] Service '{service_name}' not found in cluster."]

    logs: List[str] = []
    # Primary telemetry logs occur early in the step window
    sec = random.randint(0, 10)
    ts = f"2026-04-03T10:{step:02d}:{sec:02d}Z"

    if svc.status == ServiceStatusEnum.DOWN:
        logs.append(f"[{ts}] CRITICAL {service_name}: Service is not responding — health check failed.")
        logs.append(f"[{ts}] ERROR {service_name}: All instances terminated unexpectedly.")
        logs.append(f"[{ts}] WARN {service_name}: Last successful heartbeat was 120s ago.")

    elif svc.status == ServiceStatusEnum.DEGRADED:
        logs.append(f"[{ts}] WARN {service_name}: Elevated error rate detected — {svc.error_rate*100:.1f}% of requests failing.")
        logs.append(f"[{ts}] WARN {service_name}: p95 latency spike to {svc.latency_ms:.0f}ms (threshold: 200ms).")
        logs.append(f"[{ts}] INFO {service_name}: CPU utilisation at {svc.cpu_percent:.0f}%.")

    else:
        logs.append(f"[{ts}] INFO {service_name}: Service operating normally. Error rate: {svc.error_rate*100:.2f}%.")
        logs.append(f"[{ts}] INFO {service_name}: p95 latency: {svc.latency_ms:.0f}ms.")

    # Task-specific contextual logs
    if task_name == "single_service_failure":
        if service_name == "cache":
            logs.append(f"[{ts}] CRITICAL cache: Redis connection pool exhausted — no available connections.")
            logs.append(f"[{ts}] ERROR cache: OOM killer terminated redis-server (PID 1842).")
        elif service_name == "auth" and svc.status != ServiceStatusEnum.HEALTHY:
            logs.append(f"[{ts}] ERROR auth: Cache lookup failed — falling back to database (slow path).")
            logs.append(f"[{ts}] WARN auth: Token validation latency exceeds SLA.")

    elif task_name == "cascading_failure":
        if service_name == "database":
            logs.append(f"[{ts}] WARN database: Connection pool saturation at 95% — queries queueing.")
            logs.append(f"[{ts}] ERROR database: Slow query detected: SELECT * FROM transactions — 8200ms.")
            logs.append(f"[{ts}] WARN database: Disk I/O throughput at 98% capacity.")
        elif service_name == "auth" and svc.status != ServiceStatusEnum.HEALTHY:
            logs.append(f"[{ts}] ERROR auth: Database query timeout after 5000ms.")
            logs.append(f"[{ts}] WARN auth: Upstream dependency 'database' is degraded.")
        elif service_name == "payments" and svc.status != ServiceStatusEnum.HEALTHY:
            logs.append(f"[{ts}] ERROR payments: Transaction commit failed — database connection reset.")
            logs.append(f"[{ts}] WARN payments: Upstream dependency 'database' is degraded.")
        elif service_name == "checkout" and svc.status != ServiceStatusEnum.HEALTHY:
            logs.append(f"[{ts}] ERROR checkout: Auth token validation timed out.")
            logs.append(f"[{ts}] ERROR checkout: Payment processing failed for order #98421.")

    elif task_name == "hidden_root_cause":
        if service_name == "auth":
            if svc.version == "v2.2.0-rc1":
                logs.append(f"[{ts}] ERROR auth: Intermittent JWT signature validation failure (v2.2.0-rc1 regression).")
                logs.append(f"[{ts}] WARN auth: Deployed version v2.2.0-rc1 — release candidate, not production-validated.")
                logs.append(f"[{ts}] ERROR auth: Token validation returning 401 for valid tokens — signature algorithm mismatch.")
            else:
                logs.append(f"[{ts}] INFO auth: Version {svc.version} — all token validations passing.")
        elif service_name == "payments":
            logs.append(f"[{ts}] ERROR payments: Request authentication failed — received 401 from auth service.")
            logs.append(f"[{ts}] WARN payments: Retry budget exhausted for auth-dependent operations.")
            logs.append(f"[{ts}] ERROR payments: 32% of payment initiations failing due to auth errors.")
        elif service_name == "checkout":
            logs.append(f"[{ts}] ERROR checkout: Intermittent 401 errors from auth service during checkout flow.")
            logs.append(f"[{ts}] WARN checkout: Customer-facing error rate at {svc.error_rate*100:.1f}%.")
            logs.append(f"[{ts}] ERROR checkout: Payment step failing — upstream auth returning invalid_token.")
        elif service_name == "cache":
            logs.append(f"[{ts}] WARN cache: Serving stale auth tokens — TTL not expired but tokens may be invalid.")
            logs.append(f"[{ts}] INFO cache: Hit rate: 78% — some requests bypass cache and hit auth directly.")

    # Inject contextual noise (benign traffic) to test agent filtering
    num_noise = random.randint(6, 12)
    for _ in range(num_noise):
        n_sec = random.randint(0, 59)
        n_ts = f"2026-04-03T10:{step:02d}:{n_sec:02d}Z"
        trace_id = str(uuid.uuid4())[:8]
        status = random.choice([200, 200, 201, 204, 304, 404])
        ms = random.randint(15, 120)
        logs.append(f"[{n_ts}] INFO {service_name}: [trace={trace_id}] Handled incoming request -> HTTP {status} ({ms}ms)")

    # Sort lexicographically by timestamp string to interleave naturally
    logs.sort()

    return logs


def generate_metrics(
    service_name: str,
    services: Dict[str, ServiceState],
    task_name: str,
) -> Dict[str, Any]:
    """Generate detailed metrics dict for a service."""
    svc = services.get(service_name)
    if svc is None:
        return {"error": f"Service '{service_name}' not found"}

    # Add stochastic jitter to metrics
    j_factor = random.uniform(0.95, 1.05)
    j_add = random.uniform(-1.0, 1.0)
    
    # Degraded/down services experience chaotic jitter
    if svc.status != ServiceStatusEnum.HEALTHY:
        j_factor = random.uniform(0.8, 1.4)
        j_add = random.uniform(-5.0, 15.0)

    metrics: Dict[str, Any] = {
        "service": service_name,
        "status": svc.status.value,
        "error_rate": round(min(1.0, max(0.0, svc.error_rate * j_factor)), 4),
        "latency_p50_ms": round(max(1.0, svc.latency_ms * 0.6 * j_factor + j_add), 1),
        "latency_p95_ms": round(max(1.0, svc.latency_ms * j_factor + j_add), 1),
        "latency_p99_ms": round(max(1.0, svc.latency_ms * 1.4 * j_factor + j_add * 2), 1),
        "cpu_percent": round(min(100.0, max(0.0, svc.cpu_percent + random.gauss(0, 2.0))), 1),
        "memory_percent": round(min(100.0, max(0.0, svc.memory_percent + random.gauss(0, 0.5))), 1),
        "instances": svc.instances,
        "version": svc.version,
        "requests_per_second": int((500 if svc.status == ServiceStatusEnum.HEALTHY else (200 if svc.status == ServiceStatusEnum.DEGRADED else 0)) * j_factor),
        "open_connections": int((50 if svc.status == ServiceStatusEnum.HEALTHY else (180 if svc.status == ServiceStatusEnum.DEGRADED else 0)) * j_factor),
        "dependencies": DEPENDENCY_GRAPH.get(service_name, []),
        "dependents": REVERSE_DEPS.get(service_name, []),
    }

    # Add dependency health info
    dep_health = {}
    for dep in DEPENDENCY_GRAPH.get(service_name, []):
        dep_svc = services.get(dep)
        if dep_svc:
            dep_health[dep] = dep_svc.status.value
    metrics["dependency_health"] = dep_health

    return metrics


# ---------------------------------------------------------------------------
# Alert generation
# ---------------------------------------------------------------------------

def generate_alerts(services: Dict[str, ServiceState]) -> List[str]:
    """Generate alert strings based on current service states."""
    alerts: List[str] = []
    for name, svc in services.items():
        if svc.status == ServiceStatusEnum.DOWN:
            alerts.append(f"🔴 CRITICAL: {name} is DOWN — all instances unreachable.")
        elif svc.status == ServiceStatusEnum.DEGRADED:
            alerts.append(f"🟡 WARNING: {name} is DEGRADED — error rate {svc.error_rate*100:.1f}%, latency {svc.latency_ms:.0f}ms.")
    return alerts


# ---------------------------------------------------------------------------
# Health score computation
# ---------------------------------------------------------------------------

SERVICE_WEIGHTS: Dict[str, float] = {
    "database": 0.25,
    "cache": 0.10,
    "auth": 0.20,
    "notification": 0.05,
    "payments": 0.20,
    "checkout": 0.20,
}


def compute_health_score(services: Dict[str, ServiceState]) -> float:
    """Compute weighted system health score in [0.0, 1.0]."""
    score = 0.0
    for name, svc in services.items():
        w = SERVICE_WEIGHTS.get(name, 0.1)
        if svc.status == ServiceStatusEnum.HEALTHY:
            svc_score = 1.0 - svc.error_rate  # near 1.0
        elif svc.status == ServiceStatusEnum.DEGRADED:
            svc_score = max(0.0, 0.5 - svc.error_rate)
        else:
            svc_score = 0.0
        score += w * svc_score
    return round(min(1.0, max(0.0, score)), 4)


# ---------------------------------------------------------------------------
# Severity classification
# ---------------------------------------------------------------------------

def classify_severity(services: Dict[str, ServiceState]) -> str:
    """Return severity level string based on service states."""
    down_count = sum(1 for s in services.values() if s.status == ServiceStatusEnum.DOWN)
    degraded_count = sum(1 for s in services.values() if s.status == ServiceStatusEnum.DEGRADED)

    if down_count == 0 and degraded_count == 0:
        return "resolved"
    if down_count >= 2 or (down_count >= 1 and degraded_count >= 2):
        return "critical"
    if down_count >= 1 or degraded_count >= 3:
        return "high"
    if degraded_count >= 2:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Dependency-aware state propagation
# ---------------------------------------------------------------------------

def propagate_dependencies(
    services: Dict[str, ServiceState],
    task_name: str = "",
) -> Dict[str, ServiceState]:
    """
    Propagate health status through the dependency graph.

    If a dependency is DOWN, dependents become at least DEGRADED.
    If a dependency is DEGRADED, dependents may also degrade.
    If ALL dependencies are healthy, a degraded dependent auto-recovers
    (unless it is the root cause itself — e.g. bad deploy).
    """
    result = {k: v.model_copy() for k, v in services.items()}

    # Process in topological order (foundational first)
    order = ["database", "cache", "notification", "auth", "payments", "checkout"]

    for svc_name in order:
        deps = DEPENDENCY_GRAPH.get(svc_name, [])
        if not deps:
            continue

        svc = result[svc_name]

        # Check worst dependency status
        any_dep_down = any(result[d].status == ServiceStatusEnum.DOWN for d in deps)
        any_dep_degraded = any(result[d].status == ServiceStatusEnum.DEGRADED for d in deps)

        if any_dep_down and svc.status == ServiceStatusEnum.HEALTHY:
            # Healthy service with a down dependency becomes degraded
            result[svc_name] = svc.model_copy(update={
                "status": ServiceStatusEnum.DEGRADED,
                "error_rate": max(svc.error_rate, 0.35),
                "latency_ms": max(svc.latency_ms, 1500.0),
            })
        elif any_dep_degraded and svc.status == ServiceStatusEnum.HEALTHY:
            # Healthy service with degraded dependency gets slightly impacted
            result[svc_name] = svc.model_copy(update={
                "status": ServiceStatusEnum.DEGRADED,
                "error_rate": max(svc.error_rate, 0.15),
                "latency_ms": max(svc.latency_ms, 500.0),
            })
        elif not any_dep_down and not any_dep_degraded and svc.status == ServiceStatusEnum.DEGRADED:
            # All deps are healthy — auto-recover unless this service has an
            # intrinsic fault (e.g. bad deploy in hidden_root_cause).
            is_intrinsic_fault = (
                task_name == "hidden_root_cause"
                and svc_name == "auth"
                and svc.version == "v2.2.0-rc1"
            )
            if not is_intrinsic_fault:
                result[svc_name] = svc.model_copy(update={
                    "status": ServiceStatusEnum.HEALTHY,
                    "error_rate": 0.02,
                    "latency_ms": 35.0,
                    "cpu_percent": min(svc.cpu_percent, 20.0),
                    "memory_percent": min(svc.memory_percent, 35.0),
                })

    return result


# ---------------------------------------------------------------------------
# Action effects
# ---------------------------------------------------------------------------

def apply_restart(
    services: Dict[str, ServiceState],
    service_name: str,
    task_name: str,
) -> Dict[str, ServiceState]:
    """Apply a restart to a service. Returns updated services dict."""
    result = {k: v.model_copy() for k, v in services.items()}
    svc = result.get(service_name)
    if svc is None:
        return result

    # Special case: auth with bad deploy in hidden_root_cause can't be fixed by restart
    if (
        task_name == "hidden_root_cause"
        and service_name == "auth"
        and svc.version == "v2.2.0-rc1"
    ):
        result[service_name] = svc.model_copy(update={
            "status": ServiceStatusEnum.DEGRADED,
            "error_rate": 0.25,
            "latency_ms": 400.0,
            "cpu_percent": 40.0,
        })
        return result

    # Check if all dependencies are healthy
    deps = DEPENDENCY_GRAPH.get(service_name, [])
    all_deps_healthy = all(
        result[d].status == ServiceStatusEnum.HEALTHY for d in deps
    )

    if all_deps_healthy or not deps:
        # Restart brings service back to healthy baseline
        result[service_name] = svc.model_copy(update={
            "status": ServiceStatusEnum.HEALTHY,
            "error_rate": 0.01,
            "latency_ms": 25.0,
            "cpu_percent": 15.0,
            "memory_percent": 30.0,
        })
    else:
        # Restart helps somewhat but deps still broken
        result[service_name] = svc.model_copy(update={
            "error_rate": min(svc.error_rate, 0.20),
            "latency_ms": min(svc.latency_ms, 800.0),
        })

    return result


def apply_scale(
    services: Dict[str, ServiceState],
    service_name: str,
) -> Dict[str, ServiceState]:
    """Scale up a service by adding an instance."""
    result = {k: v.model_copy() for k, v in services.items()}
    svc = result.get(service_name)
    if svc is None:
        return result

    new_instances = svc.instances + 1
    # Scaling reduces load per instance
    new_cpu = max(10.0, svc.cpu_percent * (svc.instances / new_instances))
    new_memory = max(20.0, svc.memory_percent * 0.85)
    new_latency = max(20.0, svc.latency_ms * 0.6)
    new_error_rate = max(0.01, svc.error_rate * 0.5)

    new_status = svc.status
    if new_error_rate < 0.05 and new_latency < 200:
        new_status = ServiceStatusEnum.HEALTHY

    result[service_name] = svc.model_copy(update={
        "instances": new_instances,
        "cpu_percent": round(new_cpu, 1),
        "memory_percent": round(new_memory, 1),
        "latency_ms": round(new_latency, 1),
        "error_rate": round(new_error_rate, 4),
        "status": new_status,
    })

    return result


def apply_rollback(
    services: Dict[str, ServiceState],
    service_name: str,
    good_version: str = "v1.0.0",
) -> Dict[str, ServiceState]:
    """Roll back a service to a known good version."""
    result = {k: v.model_copy() for k, v in services.items()}
    svc = result.get(service_name)
    if svc is None:
        return result

    result[service_name] = svc.model_copy(update={
        "version": good_version,
        "status": ServiceStatusEnum.HEALTHY,
        "error_rate": 0.01,
        "latency_ms": 25.0,
        "cpu_percent": 15.0,
        "memory_percent": 30.0,
    })

    return result


def apply_clear_cache(
    services: Dict[str, ServiceState],
) -> Dict[str, ServiceState]:
    """Flush the cache service. Temporarily increases latency for cache-dependent services."""
    result = {k: v.model_copy() for k, v in services.items()}

    cache = result.get("cache")
    if cache is None:
        return result

    # Cache flush: cache itself stays healthy but briefly slower
    result["cache"] = cache.model_copy(update={
        "latency_ms": 50.0,  # slightly elevated after flush
        "memory_percent": 10.0,  # memory freed
    })

    # Auth uses cache — brief latency bump if it was relying on stale cache
    auth = result.get("auth")
    if auth and auth.status == ServiceStatusEnum.HEALTHY:
        result["auth"] = auth.model_copy(update={
            "latency_ms": max(auth.latency_ms, 80.0),
        })

    return result
