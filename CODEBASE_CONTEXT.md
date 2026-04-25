# Incident Commander — Codebase Context & Handoff

> **Last updated:** 2026-04-25  
> **Purpose:** Full context for a new agent/engineer to continue work on this project.

---

## 1. What This Project Is

An **OpenEnv-compliant RL environment** where an AI agent acts as an Incident Commander for a simulated microservices production system. The agent must diagnose and resolve outages by inspecting logs/metrics and taking corrective actions (restart, scale, rollback, etc.).

The project has two halves:
1. **Environment** (`server/`) — the simulation, grading, and HTTP API
2. **Agent training** — SFT warm-start → GRPO reinforcement learning pipeline

**Hackathon project** for deployment demo.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│  Frontend (Next.js)  →  FastAPI Backend  →  Environment │
│  frontend_app/          server/app.py       server/     │
└─────────────────────────────────────────────────────────┘
         ↓ /predict
┌─────────────────────────────────────────────────────────┐
│  Trained Model (Qwen 0.5B or 1.5B + LoRA adapter)      │
│  Greedy decode, fuzzy parser, repeat guard              │
└─────────────────────────────────────────────────────────┘
```

### Services (simulated microservices)
- `database` — foundational data store
- `cache` — Redis cache layer
- `auth` — authentication (depends on database, cache)
- `notification` — messaging (standalone)
- `payments` — payment processing (depends on database, notification)
- `checkout` — user-facing (depends on auth, payments, database)

### Available Actions
- `inspect_logs`, `inspect_metrics` — diagnostic (no side effects)
- `restart_service`, `scale_service`, `rollback` — corrective
- `clear_cache`, `escalate`, `do_nothing`, `write_runbook`

### 5 Tasks (increasing difficulty)
| Task | Difficulty | Max Steps | Time Limit (HTTP) |
|------|-----------|-----------|-------------------|
| `single_service_failure` | Easy | 10 | 120s |
| `cascading_failure` | Medium | 15 | 180s |
| `hidden_root_cause` | Hard | 25 | 300s |
| `chaos_cascade` | Hard | 30 | 300s |
| `multi_root_cause` | Hard | 30 | 360s |

### Grading (5 components, weighted)
| Component | Weight | What it measures |
|-----------|--------|-----------------|
| Recovery | 35% | Did system health reach ≥0.95? |
| Efficiency | 25% | Fewer steps + less wall-clock time = higher |
| Diagnostics | 20% | Did agent inspect before fixing? |
| Ordering | 15% | Fixed dependencies before dependents? |
| Memory | 5% | Did agent write a runbook? |

---

## 3. File Map

### Core Environment (`server/`)
| File | Purpose |
|------|---------|
| `server/app.py` | FastAPI HTTP API — `/reset`, `/step`, `/state`, `/predict`, `/dashboard` |
| `server/environment.py` | Core simulation loop, observation generation, step execution |
| `server/models.py` | Pydantic models: `IncidentAction`, `IncidentObservation`, `IncidentState`, `ActionType` |
| `server/services.py` | Service simulation, health computation, dependency graph |
| `server/tasks.py` | 5 task definitions + random task generator |
| `server/grader.py` | Episode grading (recovery, efficiency, diagnostics, ordering, memory) |
| `server/chaos.py` | Chaos injection system for harder scenarios |
| `server/runbook.py` | Runbook memory system |

### Training Pipeline
| File | Purpose |
|------|---------|
| `train_grpo.py` | **Source of truth for prompts** (`build_obs_prompt`), GRPO training loop, reward function |
| `sft_warmstart.py` | SFT warm-start: generates expert/heuristic trajectories → LoRA fine-tuning → adapter merge |

### Inference & Evaluation
| File | Purpose |
|------|---------|
| `inference.py` | CLI inference agent — supports API mode (`--local` flag for trained model) |
| `live_inference.py` | Frontend demo script — calls `/predict` endpoint in a loop |
| `evaluate_trained.py` | **Primary evaluation script** — trained model vs all baselines. Contains the **fuzzy parser** (`parse_action`) |
| `evaluate.py` | Expert/naive strategy definitions, `run_strategy` helper |
| `run_baselines.py` | Multi-agent baseline benchmarking (random, heuristic, LLM, trained) |

### Frontend
| File | Purpose |
|------|---------|
| `frontend_app/` | Next.js frontend (built by another team member) |

### Config & Deployment
| File | Purpose |
|------|---------|
| `openenv.yaml` | OpenEnv spec — points to `server.app:app` |
| `pyproject.toml` | Python package definition, `[train]` extras for GPU deps |
| `Dockerfile` | Container deployment |
| `.env` | `HF_TOKEN`, `API_KEY`, `MODEL_NAME` vars |
| `kaggle_training.md` | Step-by-step Kaggle training guide |

---

## 4. Critical Code Patterns

### 4.1 Prompt Format (MUST MATCH TRAINING)

The trained model was trained with `build_obs_prompt()` in `train_grpo.py` (line ~60-126). Any inference code MUST use this exact prompt format. It outputs a user-role-only message with:
- System health score, step count
- Service status table (status, error rate, latency, version, CPU, instances)
- Alerts, logs, previous actions
- **Explicit valid action types and service names**
- Instruction to respond with JSON only

```python
from train_grpo import build_obs_prompt
prompt = build_obs_prompt(obs_dict, step, action_history)
messages = [{"role": "user", "content": prompt}]
```

### 4.2 Generation Config (CRITICAL — do NOT change)

The model mode-collapses with wrong generation params. Use EXACTLY:
```python
model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,        # Greedy decode
    temperature=1.0,        # Unused with do_sample=False but must be set
    top_p=1.0,
    repetition_penalty=1.0, # DO NOT use 1.1 — it breaks JSON output
    pad_token_id=tokenizer.pad_token_id,
)
```

**DO NOT use:** `temperature=0.1`, `do_sample=True`, `repetition_penalty=1.1`, `max_new_tokens=64`. These caused catastrophic mode collapse and 100% parse failures in earlier versions.

### 4.3 Fuzzy Parser (`evaluate_trained.py:parse_action`)

The model sometimes outputs near-miss action types. The fuzzy parser in `evaluate_trained.py` handles:
- `inspect_services` → `inspect_logs`
- `check_logs` → `inspect_logs`
- `restart` → `restart_service`
- `scale` → `scale_service`
- Array-valued `service_name` → takes first element
- Truncated JSON → attempts to close braces
- Preamble text before JSON ("As an SRE..." + JSON)

This parser is shared across: `evaluate_trained.py`, `inference.py`, `server/app.py` `/predict` endpoint.

### 4.4 Repeat Guard

All inference paths implement a repeat guard: if the model outputs the **same action 3 times in a row**, switch to the heuristic `fallback_action()` from `inference.py`. This prevents infinite loops from mode collapse.

---

## 5. Current Model Status

### 0.5B Model (Qwen2.5-0.5B-Instruct) — AVAILABLE
- **Location (Windows):** `sft_merged_0p5b_v2/` (SFT base) + `trained_model_0p5b_v2/` (GRPO adapter)
- **SFT:** 1100 samples, 3 epochs, 20 seeds × 5 tasks, loss=0.9887
- **GRPO:** 300 steps, loss=0.0002 (no-op — GRPO didn't improve over SFT)
- **Eval Results:**
  | Task | Score | Resolved | vs Heuristic |
  |------|-------|----------|-------------|
  | single_service_failure | **0.850** | 3/3 | ✅ +0.092 |
  | cascading_failure | **0.700** | 3/3 | ✅ +0.057 |
  | hidden_root_cause | 0.466 | 2/3 | ❌ -0.284 |
  | chaos_cascade | **0.690** | 3/3 | ✅ +0.007 |
  | multi_root_cause | 0.484 | 2/3 | ❌ -0.053 |
  | **AVERAGE** | **0.638** | **13/15** | ❌ -0.036 |

### 1.5B Model (Qwen2.5-1.5B-Instruct) — IN PROGRESS (Kaggle)
- **SFT:** Complete. 1100 samples, 3 epochs, loss=0.6630 (better than 0.5B)
- **GRPO:** Running on Kaggle T4 (~2-3 hours remaining)
- **Files:** `sft_merged_1p5b/` (3.1 GB), `trained_model_1p5b/` (pending)
- **Expectation:** Should outperform 0.5B on hidden_root_cause and multi_root_cause

### Baseline Scores (for comparison)
| Agent | Avg Score |
|-------|-----------|
| Expert (hardcoded optimal) | 0.852 |
| Heuristic (rule-based) | 0.674 |
| Trained 0.5B | 0.638 |
| Naive (restart everything) | 0.546 |
| Do Nothing | 0.029 |

---

## 6. Known Issues & Gotchas

### GRPO Mode Collapse
GRPO loss near zero (0.0002) means all model generations are identical → no reward variance → no gradient signal. The model's SFT phase does all the work. GRPO is effectively a no-op for both 0.5B and likely 1.5B.

**Why:** The model generates deterministic outputs (greedy decode in eval, but similar behavior during training). With 4 identical generations, reward_std=0, GRPO has nothing to learn from.

**Potential fix (not yet tried):** Use higher temperature during GRPO generation rollouts, increase `num_generations` to 8-16, or use a different RL algorithm (PPO, DPO).

### TRL Version Incompatibilities
Kaggle's TRL version is different from local. All training scripts now have try/except fallbacks for:
- `max_seq_length` (removed in TRL ≥0.17)
- `max_completion_length` (not in all versions)
- `processing_class` vs `tokenizer` parameter naming

### P100 Not Supported
Kaggle's current PyTorch dropped P100 support (CUDA capability 6.0). Use **T4** (capability 7.5).

### `torch_dtype` Deprecation Warning
`torch_dtype` is deprecated in favor of `dtype` in newer transformers. Current code uses `torch_dtype` — it works but shows a warning. Non-blocking.

### Time-Based Scoring in HTTP Mode
When running via the FastAPI server (frontend), episodes have **wall-clock time limits**. The model must act within the SLA or receive a penalty. This is NOT active in direct Python mode (evaluation scripts).

---

## 7. How to Run Things

### Start Backend
```bash
cd "OpenEnv meta hack"
python -m uvicorn server.app:app --reload --port 8000
```

### Run Evaluation (Mac — MPS)
```bash
python evaluate_trained.py \
  --adapter trained_model_0p5b_v2 \
  --base-model sft_merged_0p5b_v2 \
  --episodes 3 --verbose --device mps
```

### Run Evaluation (Windows — CUDA)
```powershell
$env:PYTHONUTF8='1'; .\.venv310\Scripts\python.exe evaluate_trained.py `
  --adapter trained_model_0p5b_v2 `
  --base-model sft_merged_0p5b_v2 `
  --episodes 3 --verbose
```

### Train SFT (Kaggle)
```python
!python sft_warmstart.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 3 --batch-size 4 --lr 2e-5 \
  --num-seeds 20 --use-lora --use-4bit --gradient-checkpointing \
  --output-dir sft_adapter_1p5b --merged-output-dir sft_merged_1p5b
```

### Train GRPO (Kaggle, after SFT)
```python
!python train_grpo.py \
  --model sft_merged_1p5b \
  --steps 300 --batch-size 2 --lr 5e-6 \
  --num-generations 4 --num-seeds 10 \
  --use-lora --use-4bit --gradient-checkpointing \
  --output-dir trained_model_1p5b
```

### Frontend Demo
```bash
# Terminal 1: Backend
python -m uvicorn server.app:app --reload --port 8000

# Terminal 2: Live inference
python live_inference.py

# Terminal 3: Frontend
cd frontend_app && npm run dev
```

---

## 8. Pending Tasks / Next Steps

### Immediate
- [ ] Download 1.5B model from Kaggle and evaluate locally
- [ ] Compare 1.5B vs 0.5B scores — use whichever is better for demo

### Multi-Agent Orchestrator (Planned)
- [ ] Build a smart orchestrator that routes between the trained model and task-specific heuristics
- [ ] Goal: push avg score from 0.638 → 0.70+ by detecting when the model struggles (hidden_root_cause, multi_root_cause) and switching to specialized heuristic strategies
- [ ] The repeat guard is a primitive version of this — orchestrator would be smarter (confidence-based routing)

### If Time Permits
- [ ] Investigate why GRPO produces zero reward variance — try higher temperature during rollout generation
- [ ] Add `memory` score component (write_runbook action at end of episode)
- [ ] Upload best model to HuggingFace for deployment

---

## 9. Environment Setup

### Mac (M4 Air)
- Python 3.12, venv at `.venv312/`
- Device: `mps`
- Used for testing/evaluation only (no training)

### Windows (Friend's laptop, 6GB VRAM)
- Python 3.10, venv at `.venv310/`
- Device: `cuda`
- Can train 0.5B models (SFT ~5min, GRPO ~10min)
- Set `$env:PYTHONUTF8='1'` before all Python commands

### Kaggle (GPU T4 x2, 16GB VRAM)
- Used for 1.5B model training
- Must enable Internet in notebook settings
- Repo is private — clone with PAT or make repo temporarily public
- See `kaggle_training.md` for full setup

### Dependencies
```
pip install -e ".[train]"
# Which installs: trl, transformers, torch, accelerate, peft, bitsandbytes, datasets
```

---

## 10. Git Info
- **Repo:** `https://github.com/hs-zz27/incident-commander-openenv.git`
- **Branch:** `main`
- **Important:** Many files are gitignored (model weights, eval results, logs). Check `.gitignore`.
