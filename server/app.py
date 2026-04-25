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
    env = IncidentCommanderEnvironment(http_mode=True)
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

    # ---- Trained Model Predict Endpoint ----
    # Lazy-loaded model state
    _model_state = {"model": None, "tokenizer": None, "loaded": False, "error": None}

    class PredictRequest(BaseModel):
        base_model: str = Field(default="Qwen/Qwen2.5-0.5B-Instruct", description="Base model name")
        adapter_path: str = Field(default="trained_model_full_0p5b", description="LoRA adapter path")
        device: str = Field(default="auto", description="Device: auto, cpu, cuda, mps")

    @app.post("/predict")
    async def predict(request: PredictRequest = None):
        """
        Generate the next action from the trained LoRA policy.

        Uses the current environment state and the EXACT training prompt format
        (build_obs_prompt from train_grpo.py) for distribution-matched inference.

        The model is lazy-loaded on first call to avoid slowing down server startup.
        """
        import os, sys
        req = request or PredictRequest()

        # Lazy-load model
        if not _model_state["loaded"]:
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
                from peft import PeftModel

                device = req.device
                if device == "auto":
                    if torch.cuda.is_available():
                        device = "cuda"
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = "cpu"

                from pathlib import Path
                adapter_dir = Path(req.adapter_path)
                if not adapter_dir.exists():
                    raise HTTPException(
                        status_code=404,
                        detail=f"Adapter not found at '{req.adapter_path}'. "
                               f"Get trained_model_full_0p5b/ from your teammate.",
                    )

                dtype = torch.float32
                if device == "mps":
                    dtype = torch.float16
                elif device == "cuda":
                    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

                tokenizer = AutoTokenizer.from_pretrained(req.base_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                base = AutoModelForCausalLM.from_pretrained(
                    req.base_model, torch_dtype=dtype,
                    device_map=device if device != "mps" else None,
                )
                if device == "mps":
                    base = base.to("mps")

                model = PeftModel.from_pretrained(base, req.adapter_path)
                model.eval()

                _model_state["model"] = model
                _model_state["tokenizer"] = tokenizer
                _model_state["loaded"] = True
                logger.info(f"Trained model loaded on {device}")

            except HTTPException:
                raise
            except Exception as e:
                _model_state["error"] = str(e)
                raise HTTPException(status_code=500, detail=f"Model loading failed: {e}")

        model = _model_state["model"]
        tokenizer = _model_state["tokenizer"]

        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Build prompt from current env state using EXACT training format
        try:
            sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            from train_grpo import build_obs_prompt
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Cannot import train_grpo.build_obs_prompt — needed for prompt alignment",
            )

        import torch

        state = env.state
        obs_dict = {
            "services": {k: v.model_dump() for k, v in state.services.items()},
            "system_health_score": sum(
                1 for s in state.services.values()
                if s.status.value == "healthy"
            ) / max(len(state.services), 1),
            "max_steps": 30,
            "incident_severity": "unknown",
            "alerts": [],
            "logs": [],
            "escalation_tier": 1,
            "services_at_risk": [],
            "runbook_memory": [],
            "metadata": {},
        }

        prompt_text = build_obs_prompt(obs_dict, state.step_count + 1, state.actions_taken)
        messages = [{"role": "user", "content": prompt_text}]

        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt_text

        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                temperature=0.3,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
            )

        generated = outputs[0][inputs["input_ids"].shape[1]:]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON action
        import json as json_mod
        action_data = None
        try:
            action_data = json_mod.loads(response)
        except Exception:
            # Try to extract JSON from response
            idx = response.find("{")
            if idx >= 0:
                end = response.rfind("}")
                if end > idx:
                    try:
                        action_data = json_mod.loads(response[idx:end+1])
                    except Exception:
                        pass

        return {
            "raw_response": response,
            "parsed_action": action_data,
            "step": state.step_count + 1,
            "model_device": str(next(model.parameters()).device),
        }

    @app.get("/model/info")
    async def model_info():
        """Return trained model status and metadata."""
        info = {
            "loaded": _model_state["loaded"],
            "error": _model_state["error"],
        }
        if _model_state["model"] is not None:
            model = _model_state["model"]
            info["device"] = str(next(model.parameters()).device)
            info["param_count"] = sum(p.numel() for p in model.parameters())
        return info

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
