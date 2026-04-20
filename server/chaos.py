"""
Chaos Agent — Background failure injection for the Incident Commander Environment.

When chaos_mode is enabled, the ChaosAgent randomly injects new failures into
previously healthy services during an episode, simulating real-world incident
cascades where new problems emerge while you're fixing existing ones.
"""

from __future__ import annotations

import random
from typing import Dict, Optional, Set

from .models import ServiceState, ServiceStatusEnum


class ChaosAgent:
    """
    Probabilistic failure injector that runs alongside the main episode.

    After `min_step`, each environment step has `injection_probability` chance
    of injecting a new failure into a healthy service that isn't currently
    being investigated by the agent.
    """

    # Failure profiles the chaos agent can inject
    CHAOS_PROFILES = {
        "oom_crash": {
            "status": ServiceStatusEnum.DOWN,
            "error_rate": 0.95,
            "latency_ms": 0.0,
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "instances": 0,
        },
        "memory_leak": {
            "status": ServiceStatusEnum.DEGRADED,
            "error_rate": 0.30,
            "latency_ms": 3000.0,
            "cpu_percent": 80.0,
            "memory_percent": 95.0,
        },
        "cpu_overload": {
            "status": ServiceStatusEnum.DEGRADED,
            "error_rate": 0.25,
            "latency_ms": 4000.0,
            "cpu_percent": 98.0,
            "memory_percent": 70.0,
        },
    }

    def __init__(
        self,
        injection_probability: float = 0.15,
        min_step: int = 5,
    ) -> None:
        """
        Args:
            injection_probability: Probability (0-1) of injecting a failure
                                   at each step after min_step.
            min_step: First step at which chaos injection can occur.
        """
        self.injection_probability = injection_probability
        self.min_step = min_step
        self._injected_services: Set[str] = set()

    def reset(self) -> None:
        """Reset chaos agent state for a new episode."""
        self._injected_services = set()

    @property
    def injected_services(self) -> Set[str]:
        """Return set of services that have been chaos-injected this episode."""
        return set(self._injected_services)

    def maybe_inject(
        self,
        step: int,
        current_services: Dict[str, ServiceState],
        rng: random.Random,
        inspected_services: Optional[Set[str]] = None,
    ) -> Optional[str]:
        """
        Possibly inject a new failure into a healthy service.

        Args:
            step: Current step number.
            current_services: Current service states.
            rng: Random number generator for reproducibility.
            inspected_services: Services the agent is currently investigating
                                (prefer NOT injecting into these).

        Returns:
            Name of the newly failed service, or None if no injection.
        """
        if step < self.min_step:
            return None

        if rng.random() > self.injection_probability:
            return None

        inspected = inspected_services or set()

        # Find candidate services: healthy and not already chaos-injected
        candidates = []
        preferred = []  # prefer services NOT being investigated
        for name, svc in current_services.items():
            if svc.status == ServiceStatusEnum.HEALTHY and name not in self._injected_services:
                candidates.append(name)
                if name not in inspected:
                    preferred.append(name)

        if not candidates:
            return None

        # Prefer injecting into services the agent isn't looking at
        target_pool = preferred if preferred else candidates
        target = rng.choice(target_pool)

        # Pick a random chaos profile
        profile_name = rng.choice(list(self.CHAOS_PROFILES.keys()))
        profile = self.CHAOS_PROFILES[profile_name]

        # Apply the failure
        updates = {
            "status": profile["status"],
            "error_rate": profile["error_rate"],
            "latency_ms": profile["latency_ms"],
            "cpu_percent": profile["cpu_percent"],
            "memory_percent": profile["memory_percent"],
        }
        if "instances" in profile:
            updates["instances"] = profile["instances"]

        current_services[target] = current_services[target].model_copy(update=updates)
        self._injected_services.add(target)

        return target

    def force_inject(
        self,
        service_name: str,
        current_services: Dict[str, ServiceState],
        profile_name: str = "oom_crash",
    ) -> str:
        """
        Force-inject a specific failure into a named service.

        Used by pre-scripted chaos tasks (e.g., chaos_cascade at step 8).
        """
        profile = self.CHAOS_PROFILES.get(profile_name, self.CHAOS_PROFILES["oom_crash"])

        updates = {
            "status": profile["status"],
            "error_rate": profile["error_rate"],
            "latency_ms": profile["latency_ms"],
            "cpu_percent": profile["cpu_percent"],
            "memory_percent": profile["memory_percent"],
        }
        if "instances" in profile:
            updates["instances"] = profile["instances"]

        current_services[service_name] = current_services[service_name].model_copy(
            update=updates
        )
        self._injected_services.add(service_name)

        return service_name
