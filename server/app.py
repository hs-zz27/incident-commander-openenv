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
import os
import sys
import traceback
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .environment import IncidentCommanderEnvironment
from .models import IncidentAction, IncidentObservation, IncidentState
from .tasks import list_tasks, get_task
from orchestrator import orchestrated_action as _orchestrated_action

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


class PredictRequest(BaseModel):
    base_model: str = Field(
        default="sft_merged_1p5b_v5", description="Base model name"
    )
    adapter_path: str = Field(
        default="trained_model_1p5b_v5", description="LoRA adapter path"
    )
    device: str = Field(default="auto", description="Device: auto, cpu, cuda, mps")


class StartSimRequest(BaseModel):
    task: str = Field(default="random_incident", description="Task/scenario to run")
    chaos: bool = Field(default=True, description="Enable chaos mode")


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
    _last_observation: Optional[IncidentObservation] = None

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
        nonlocal _is_initialised, _last_observation
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
                _last_observation = obs
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
        nonlocal _last_observation
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
                _last_observation = obs
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
        state_dict = env.state.model_dump()
        # Enrich state with data from the last observation
        if _last_observation is not None:
            state_dict["runbook_memory"] = [
                entry if isinstance(entry, dict) else entry
                for entry in (_last_observation.runbook_memory or [])
            ]
            state_dict["escalation_tier"] = _last_observation.escalation_tier
            state_dict["services_at_risk"] = _last_observation.services_at_risk
            # Expose last observation metadata (used for chaos injection indicator).
            state_dict["metadata"] = _last_observation.metadata or {}
        return StateResponse(state=state_dict)

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
                "inspect_logs",
                "inspect_metrics",
                "restart_service",
                "scale_service",
                "rollback",
                "clear_cache",
                "escalate",
                "do_nothing",
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
        for name in [
            "database",
            "cache",
            "auth",
            "notification",
            "payments",
            "checkout",
        ]:
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
                <td>{svc.error_rate * 100:.1f}%</td>
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
                Task: <b>{state.task_name or "N/A"}</b> &nbsp;|&nbsp;
                Step: <b>{state.step_count}</b> &nbsp;|&nbsp;
                Score: <b>{state.cumulative_reward:.3f}</b> &nbsp;|&nbsp;
                Resolved: <b>{state.is_resolved}</b>
            </div>
            <div style="margin-bottom:12px;">
                System Health: <b>{health:.1%}</b>
                <div class="health-bar">
                    <div class="health-fill" style="width:{health * 100:.0f}%; background: {"#22c55e" if health > 0.8 else "#eab308" if health > 0.5 else "#ef4444"};"></div>
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
                    elif (
                        hasattr(torch.backends, "mps")
                        and torch.backends.mps.is_available()
                    ):
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
                    dtype = (
                        torch.bfloat16
                        if torch.cuda.is_bf16_supported()
                        else torch.float16
                    )

                tokenizer = AutoTokenizer.from_pretrained(req.base_model)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token

                base = AutoModelForCausalLM.from_pretrained(
                    req.base_model,
                    torch_dtype=dtype,
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
                raise HTTPException(
                    status_code=500, detail=f"Model loading failed: {e}"
                )

        model = _model_state["model"]
        tokenizer = _model_state["tokenizer"]

        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Build prompt from current env state using EXACT training format
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from train_grpo import build_obs_prompt
        except ImportError:
            raise HTTPException(
                status_code=500,
                detail="Cannot import train_grpo.build_obs_prompt — needed for prompt alignment",
            )

        import torch

        state = env.state
        if _last_observation is None:
            raise HTTPException(
                status_code=400,
                detail="No observation available. Call POST /reset first.",
            )
        obs_dict = _last_observation.model_dump()

        prompt_text = build_obs_prompt(
            obs_dict, state.step_count + 1, state.actions_taken
        )
        messages = [{"role": "user", "content": prompt_text}]

        if hasattr(tokenizer, "apply_chat_template"):
            input_text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            input_text = prompt_text

        inputs = tokenizer(
            input_text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                repetition_penalty=1.0,
                pad_token_id=tokenizer.pad_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        # Compute confidence from generation scores
        confidence = 0.0
        if outputs.scores:
            probs_list = [torch.softmax(s[0], dim=-1) for s in outputs.scores]
            top_probs = [p.max().item() for p in probs_list]
            confidence = round(sum(top_probs) / len(top_probs), 4) if top_probs else 0.0

        generated = outputs.sequences[0][inputs["input_ids"].shape[1] :]
        response = tokenizer.decode(generated, skip_special_tokens=True).strip()

        # Parse JSON action using the shared fuzzy parser
        action_data = None
        heuristic_action = None
        parsed_action = None
        try:
            sys.path.insert(
                0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            from evaluate_trained import parse_action as fuzzy_parse
            from evaluate_trained import heuristic_action as eval_heuristic_action

            heuristic_action = eval_heuristic_action

            parsed_action = fuzzy_parse(response)
        except ImportError:
            # Fallback: basic JSON extraction
            import json as json_mod

            try:
                action_data = json_mod.loads(response)
                parsed_action = IncidentAction(**action_data)
            except Exception:
                idx = response.find("{")
                if idx >= 0:
                    end = response.rfind("}")
                    if end > idx:
                        try:
                            action_data = json_mod.loads(response[idx : end + 1])
                            parsed_action = IncidentAction(**action_data)
                        except Exception:
                            pass

        # Hybrid orchestrator: route between model output and deterministic expert policy
        task = get_task(state.task_name or "single_service_failure")
        decision = _orchestrated_action(
            model_action=parsed_action,
            obs_dict=obs_dict,
            step=state.step_count + 1,
            action_history=state.actions_taken,
            task=task,
        )
        final_action = decision.action
        action_data = {"action_type": final_action.action_type.value}
        if final_action.service_name:
            action_data["service_name"] = final_action.service_name

        return {
            "raw_response": response,
            "parsed_action": action_data,
            "routing": {"used_model": decision.used_model, "reason": decision.reason},
            "confidence": confidence,
            "step": state.step_count + 1,
            "model_device": str(next(model.parameters()).device),
        }

    @app.post("/predict_and_step")
    async def predict_and_step(request: PredictRequest = None):
        """
        One-shot endpoint: predict the next action AND execute it.

        This is the endpoint the frontend should call in a loop until done=true.
        Combines /predict + /step atomically so the caller doesn't need to
        make two separate requests per step.
        """
        nonlocal _last_observation

        # 1. Get the predicted action (reuses /predict logic)
        predict_result = await predict(request)
        action_data = predict_result["parsed_action"]

        # 2. Execute the action on the environment
        action = IncidentAction(**action_data)
        async with _lock:
            obs = env.step(action)
            _last_observation = obs

        obs_dict = obs.model_dump()
        return {
            "action_taken": action_data,
            "routing": predict_result["routing"],
            "confidence": predict_result.get("confidence", 0.0),
            "observation": obs_dict,
            "reward": obs.reward,
            "done": obs.done,
            "step": env.state.step_count,
            "system_health": obs_dict.get("system_health_score", 0.0),
            "is_resolved": env.state.is_resolved,
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

    @app.get("/score")
    async def live_score():
        """Return current score breakdown (can be called during an episode)."""
        result = env.grade()
        return {
            "score": result.get("score", 0.0),
            "breakdown": result.get("breakdown", {}),
            "steps_taken": result.get("steps_taken", 0),
            "is_resolved": result.get("is_resolved", False),
        }

    @app.get("/report")
    async def incident_report():
        """Generate a post-incident report for the completed episode."""
        state = env.state
        grade_result = env.grade()
        breakdown = grade_result.get("breakdown", {})
        timeline = env.timeline

        # Build markdown report
        lines = []
        lines.append("# 📋 Post-Incident Report")
        lines.append(f"\n**Task:** {state.task_name}")
        lines.append(f"**Episode ID:** {state.episode_id}")
        lines.append(f"**Status:** {'✅ Resolved' if state.is_resolved else '❌ Unresolved'}")
        lines.append(f"**Total Steps:** {state.step_count}")
        lines.append(f"**Overall Score:** {grade_result.get('score', 0):.3f} / 1.000")

        lines.append("\n## Score Breakdown")
        lines.append("| Component | Score | Max | Notes |")
        lines.append("|:----------|:------|:----|:------|")
        component_max = {"recovery": 0.35, "efficiency": 0.20, "diagnostics": 0.15, "ordering": 0.20, "memory": 0.10}
        for comp, max_val in component_max.items():
            val = breakdown.get(comp, 0)
            pct = (val / max_val * 100) if max_val > 0 else 0
            emoji = "🟢" if pct >= 90 else ("🟡" if pct >= 60 else "🔴")
            lines.append(f"| {emoji} {comp.title()} | {val:.3f} | {max_val:.2f} | {pct:.0f}% |")

        lines.append("\n## Action Timeline")
        lines.append("| Step | Action | Health | Reward |")
        lines.append("|:-----|:-------|:-------|:-------|")
        for evt in timeline:
            step = evt.get("step", "?")
            event = evt.get("event", "unknown")
            health = evt.get("health", 0)
            reward = evt.get("reward", 0)
            if health is not None:
                lines.append(f"| {step} | {event} | {health:.2%} | {reward:+.3f} |")

        lines.append("\n## Recovery Actions Taken")
        for i, action in enumerate(state.actions_taken, 1):
            lines.append(f"{i}. `{action}`")

        # Service final state
        lines.append("\n## Final Service State")
        lines.append("| Service | Status | Error Rate | Latency |")
        lines.append("|:--------|:-------|:-----------|:--------|")
        for name, svc_data in sorted(state.services.items()):
            svc = svc_data if isinstance(svc_data, dict) else svc_data.model_dump() if hasattr(svc_data, 'model_dump') else {}
            status = svc.get("status", "unknown")
            emoji = "🟢" if status in ("healthy", "ServiceStatusEnum.HEALTHY") else ("🟡" if "degraded" in str(status).lower() else "🔴")
            err = svc.get("error_rate", 0)
            lat = svc.get("latency_ms", 0)
            lines.append(f"| {emoji} {name} | {status} | {err:.1%} | {lat:.0f}ms |")

        lines.append("\n---")
        lines.append("*Report generated automatically by Incident Commander*")

        report_text = "\n".join(lines)
        return {"report": report_text, "score": grade_result.get("score", 0), "is_resolved": state.is_resolved}

    # ---- Simulation process management ----
    _sim_process: dict = {"proc": None}


    @app.post("/start-sim")
    async def start_sim(request: StartSimRequest = None):
        """Spawn live_inference.py as a background subprocess."""
        import subprocess
        req = request or StartSimRequest()

        # Kill existing sim if running
        if _sim_process["proc"] is not None:
            try:
                _sim_process["proc"].terminate()
                _sim_process["proc"].wait(timeout=3)
            except Exception:
                try:
                    _sim_process["proc"].kill()
                except Exception:
                    pass
            _sim_process["proc"] = None

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        cmd = [
            sys.executable, os.path.join(project_root, "live_inference.py"),
            "--task", req.task,
            "--delay", "1.5",
        ]
        logger.info(f"Starting sim: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd, cwd=project_root,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        )
        _sim_process["proc"] = proc
        return {"status": "started", "task": req.task, "pid": proc.pid}

    @app.post("/stop-sim")
    async def stop_sim():
        """Stop a running simulation subprocess."""
        if _sim_process["proc"] is None:
            return {"status": "not_running"}
        try:
            _sim_process["proc"].terminate()
            _sim_process["proc"].wait(timeout=3)
        except Exception:
            try:
                _sim_process["proc"].kill()
            except Exception:
                pass
        _sim_process["proc"] = None
        return {"status": "stopped"}

    @app.get("/sim-status")
    async def sim_status():
        """Check if simulation is currently running."""
        proc = _sim_process["proc"]
        if proc is None:
            return {"running": False}
        poll = proc.poll()
        if poll is not None:
            _sim_process["proc"] = None
            return {"running": False, "exit_code": poll}
        return {"running": True, "pid": proc.pid}

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
