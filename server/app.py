"""
FastAPI application for the Incident Commander Environment.

Exposes the OpenEnv-compliant HTTP API:
  POST /reset  — start a new episode
  POST /step   — execute an action
  GET  /state  — get current state
  GET  /health — health check

Compatible with Hugging Face Spaces deployment.
"""

from __future__ import annotations

import asyncio
import logging
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .environment import IncidentCommanderEnvironment
from .models import IncidentAction, IncidentObservation, IncidentState
from .tasks import list_tasks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models for the HTTP layer
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    task_name: Optional[str] = None
    chaos_mode: bool = False


class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StepRequest(BaseModel):
    action: Dict[str, Any]


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: Optional[float] = None
    done: bool = False


class StateResponse(BaseModel):
    state: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str = "healthy"


class GradeResponse(BaseModel):
    score: float
    breakdown: Dict[str, float]
    steps_taken: int
    is_resolved: bool
    escalated: bool
    rewards: list


class TaskListResponse(BaseModel):
    tasks: list


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_incident_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    app = FastAPI(
        title="Incident Commander Environment",
        description=(
            "An OpenEnv-compliant RL environment simulating production "
            "microservices incident response. The agent plays Incident "
            "Commander and must triage, diagnose, and resolve outages."
        ),
        version="1.0.0",
    )

    # CORS for Hugging Face Spaces
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Single environment instance (stateful) with concurrency guard
    env = IncidentCommanderEnvironment()
    _lock = asyncio.Lock()
    _is_initialised = False

    # ---- Health check ----
    @app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(status="healthy")

    # ---- Task list ----
    @app.get("/tasks", response_model=TaskListResponse)
    async def tasks():
        return TaskListResponse(tasks=list_tasks())

    # ---- Reset ----
    @app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest = None):
        nonlocal _is_initialised
        req = request or ResetRequest()
        async with _lock:
            try:
                obs = env.reset(
                    seed=req.seed,
                    episode_id=req.episode_id,
                    task_name=req.task_name,
                    chaos_mode=req.chaos_mode,
                )
                _is_initialised = True
                obs_dict = obs.model_dump()
                return ResetResponse(
                    observation=obs_dict,
                    reward=obs.reward,
                    done=obs.done,
                )
            except KeyError as e:
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logger.error(f"Reset failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

    # ---- Step ----
    @app.post("/step", response_model=StepResponse)
    async def step(request: StepRequest):
        if not _is_initialised:
            raise HTTPException(
                status_code=400,
                detail="Environment not initialised. Call POST /reset first.",
            )
        try:
            action = IncidentAction(**request.action)
        except Exception as e:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid action: {e}",
            )

        async with _lock:
            try:
                obs = env.step(action)
                obs_dict = obs.model_dump()
                return StepResponse(
                    observation=obs_dict,
                    reward=obs.reward,
                    done=obs.done,
                )
            except Exception as e:
                logger.error(f"Step failed: {e}\n{traceback.format_exc()}")
                raise HTTPException(status_code=500, detail=str(e))

    # ---- State ----
    @app.get("/state", response_model=StateResponse)
    async def state():
        return StateResponse(state=env.state.model_dump())

    # ---- Grade ----
    @app.get("/grade", response_model=GradeResponse)
    async def grade():
        result = env.grade()
        return GradeResponse(**result)

    # ---- Timeline (post-mortem) ----
    @app.get("/timeline")
    async def timeline():
        """Return the incident timeline for post-mortem analysis."""
        state = env.state
        return {
            "episode_id": state.episode_id,
            "task_name": state.task_name,
            "timeline": state.incident_timeline,
            "total_steps": state.step_count,
            "is_resolved": state.is_resolved,
        }

    # ---- Environment Info ----
    @app.get("/info")
    async def info():
        """Return environment metadata and capabilities."""
        from .services import DEPENDENCY_GRAPH, ALL_SERVICES
        from .tasks import list_tasks as _list_tasks, get_task
        tasks_info = {}
        for t_name in _list_tasks():
            t = get_task(t_name)
            tasks_info[t_name] = {
                "difficulty": t.difficulty,
                "max_steps": t.max_steps,
                "description": t.description,
            }
        return {
            "name": "incident_commander_env",
            "version": "1.0.0",
            "services": ALL_SERVICES,
            "dependency_graph": DEPENDENCY_GRAPH,
            "action_types": [
                "inspect_logs", "inspect_metrics",
                "restart_service", "scale_service",
                "rollback", "clear_cache",
                "escalate", "do_nothing",
            ],
            "tasks": tasks_info,
        }
    # ---- Live Health Dashboard (T2-3) ----
    @app.get("/dashboard")
    async def dashboard():
        """Live auto-refreshing HTML dashboard for demo presentations."""
        from fastapi.responses import HTMLResponse
        state = env.state
        services = state.services
        from .services import compute_health_score
        health = compute_health_score(services) if services else 0.0

        rows = ""
        for name in ["database", "cache", "auth", "notification", "payments", "checkout"]:
            svc = services.get(name)
            if svc is None:
                continue
            status = svc.status.value
            if status == "healthy":
                color = "#22c55e"
                bg = "#052e16"
            elif status == "degraded":
                color = "#eab308"
                bg = "#422006"
            else:
                color = "#ef4444"
                bg = "#450a0a"
            health_pct = 0.0
            if status == "healthy":
                health_pct = max(0, (1.0 - svc.error_rate)) * 100
            elif status == "degraded":
                health_pct = max(0, (0.5 - svc.error_rate)) * 100
            rows += f"""
            <tr>
                <td style="font-weight:600;">{name}</td>
                <td style="background:{bg}; color:{color}; text-align:center; border-radius:4px; padding:4px 8px;">
                    {status.upper()}
                </td>
                <td>{health_pct:.0f}%</td>
                <td>{svc.error_rate*100:.1f}%</td>
                <td>{svc.latency_ms:.0f} ms</td>
                <td>{svc.cpu_percent:.0f}%</td>
                <td>{svc.instances}</td>
                <td>{svc.version}</td>
            </tr>"""

        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta http-equiv="refresh" content="2">
            <title>Incident Commander — Live Dashboard</title>
            <style>
                body {{ font-family: 'Inter', 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; margin: 0; padding: 20px; }}
                h1 {{ color: #f8fafc; margin-bottom: 4px; }}
                .meta {{ color: #94a3b8; margin-bottom: 20px; }}
                table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; }}
                th {{ background: #334155; padding: 12px 16px; text-align: left; font-size: 13px; text-transform: uppercase; letter-spacing: 0.05em; color: #94a3b8; }}
                td {{ padding: 10px 16px; border-top: 1px solid #334155; }}
                .health-bar {{ background: #1e293b; border-radius: 999px; height: 24px; width: 200px; overflow: hidden; margin-top: 8px; }}
                .health-fill {{ height: 100%; border-radius: 999px; transition: width 0.5s; }}
            </style>
        </head>
        <body>
            <h1>🚨 Incident Commander</h1>
            <div class="meta">
                Task: <b>{state.task_name or 'N/A'}</b> &nbsp;|&nbsp;
                Step: <b>{state.step_count}</b> &nbsp;|&nbsp;
                Score: <b>{state.cumulative_reward:.3f}</b> &nbsp;|&nbsp;
                Resolved: <b>{state.is_resolved}</b>
            </div>
            <div style="margin-bottom:12px;">
                System Health: <b>{health:.1%}</b>
                <div class="health-bar">
                    <div class="health-fill" style="width:{health*100:.0f}%; background: {'#22c55e' if health > 0.8 else '#eab308' if health > 0.5 else '#ef4444'};"></div>
                </div>
            </div>
            <table>
                <thead>
                    <tr>
                        <th>Service</th><th>Status</th><th>Health</th><th>Error Rate</th>
                        <th>Latency</th><th>CPU</th><th>Instances</th><th>Version</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </body>
        </html>
        """
        return HTMLResponse(content=html)

    # ---- Metadata (required by openenv validate) ----
    @app.get("/metadata")
    async def metadata():
        """Return environment name and description for openenv validation."""
        return {
            "name": "incident_commander_env",
            "description": (
                "AI SRE Incident Commander — diagnose and resolve production "
                "microservices outages across 5 difficulty levels with chaos injection"
            ),
        }

    # ---- Schema (required by openenv validate) ----
    @app.get("/schema")
    async def schema():
        """Return JSON schemas for action, observation, and state models."""
        return {
            "action": IncidentAction.model_json_schema(),
            "observation": IncidentObservation.model_json_schema(),
            "state": IncidentState.model_json_schema(),
        }

    return app


# Create the app instance (used by uvicorn / openenv.yaml)
app = create_incident_app()


def main():
    """Entry point for direct execution."""
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


# Alias for [project.scripts] entry point
run_server = main


if __name__ == "__main__":
    main()
