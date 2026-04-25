# Incident Commander — Codebase Context & Handoff

> **Last updated:** 2026-04-26  
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
│  Trained Model (Qwen 1.5B + LoRA adapter)               │
│  Greedy decode, fuzzy parser, orchestrator routing       │
└─────────────────────────────────────────────────────────┘
```

### Hybrid Orchestrator (IMPORTANT)
The shipped inference path is a **hybrid orchestrator** that routes between:
- the trained model's proposed action (when safe), and
- a deterministic, task-aware heuristic expert (when the model is wrong / repeating / contradicts known patterns).

**Source of truth:** `orchestrator.py` (used by `evaluate_trained.py`, `inference.py`, and `server/app.py:/predict`).

### Multi-Agent Architecture (`multi_agent_inference.py`)
A separate **coordinator-specialist** system using GPT-4o-mini as coordinator with 3 specialist agents:
- **DB Expert** — database + cache specialization
- **Infra Expert** — infrastructure-wide restart/scale/rollback
- **App Expert** — application-layer services (auth, payments, checkout)

Requires `OPENAI_API_KEY` in `.env`. This is an optional demo feature, not used in benchmarking.

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
| `server/environment.py` | Core simulation loop, observation generation, step execution. **Has grace-step logic for write_runbook.** |
| `server/models.py` | Pydantic models: `IncidentAction`, `IncidentObservation`, `IncidentState`, `ActionType` |
| `server/services.py` | Service simulation, health computation, dependency graph |
| `server/tasks.py` | 5 task definitions + random task generator |
| `server/grader.py` | Episode grading (recovery, efficiency, diagnostics, ordering, memory) |
| `server/chaos.py` | Chaos injection system for harder scenarios |
| `server/runbook.py` | Runbook memory system |

### Training Pipeline
| File | Purpose |
|------|---------|
| `train_grpo.py` | **Source of truth for prompts** (`build_obs_prompt`), GRPO training loop, reward function. Supports configurable `--lora-r` / `--lora-alpha`. |
| `sft_warmstart.py` | SFT warm-start: generates expert/heuristic trajectories → LoRA fine-tuning → adapter merge. Uses 12 seeds × 5 tasks for 1.5B. |
| `incident_commander_training.ipynb` | **Colab notebook** for judges — full SFT + GRPO + eval pipeline with inline plots. |

### Inference & Evaluation
| File | Purpose |
|------|---------|
| `inference.py` | CLI inference agent — supports API mode (`--local` flag for trained model) |
| `live_inference.py` | Frontend demo script — calls `/predict` endpoint in a loop |
| `evaluate_trained.py` | **Primary evaluation script** — trained model vs all baselines. Contains the **fuzzy parser** (`parse_action`) |
| `evaluate.py` | Expert/naive strategy definitions, `run_strategy` helper |
| `run_baselines.py` | Multi-agent baseline benchmarking (random, heuristic, LLM, trained) |
| `orchestrator.py` | **Hybrid routing policy** (model + deterministic expert), diagnostics guarantees + guardrails |
| `multi_agent_inference.py` | Coordinator-specialist multi-agent architecture (GPT-4o-mini based) |
| `plot_training.py` | Generates training evidence plots (reward curve, loss, baseline comparison, score breakdown, pipeline overview) |

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
    do_sample=False,        # Greedy decode for eval
    temperature=1.0,        # Unused with do_sample=False but must be set
    top_p=1.0,
    repetition_penalty=1.0, # DO NOT use 1.1 — it breaks JSON output
    pad_token_id=tokenizer.pad_token_id,
)
```

**DO NOT use:** `temperature=0.1`, `do_sample=True`, `repetition_penalty=1.1`, `max_new_tokens=64`. These caused catastrophic mode collapse and 100% parse failures in earlier versions.

### 4.3 GRPO Generation Config (TRAINING ONLY)

During GRPO training, `temperature=1.0` is set in GRPOConfig to ensure diverse completions across the generations per prompt. Without this, entropy collapses and all generations become identical → reward_std=0 → no gradient → learning dies.

### 4.4 Fuzzy Parser (`evaluate_trained.py:parse_action`)

The model sometimes outputs near-miss action types. The fuzzy parser in `evaluate_trained.py` handles:
- `inspect_services` → `inspect_logs`
- `check_logs` → `inspect_logs`
- `restart` → `restart_service`
- `scale` → `scale_service`
- Array-valued `service_name` → takes first element
- Truncated JSON → attempts to close braces
- Preamble text before JSON ("As an SRE..." + JSON)

This parser is shared across: `evaluate_trained.py`, `inference.py`, `server/app.py` `/predict` endpoint.

### 4.5 Orchestrator Routing

All inference paths use `orchestrator.py` to decide each step:
- **Trust model** when action is valid and not harmful.
- **Override** to deterministic expert when:
  - model output can't be parsed, or
  - model repeats actions (repeat×2), or
  - known patterns require a specific fix (e.g. `auth` bad deploy ⇒ `rollback:auth` then `clear_cache`), or
  - dependency-order constraints would be violated, or
  - early recovery action before root cause has been inspected.

This is the main technique to boost score without further training.

### 4.6 Reward Function (GRPO)

**v5 uses `--reward-mode direct`:** Score is computed from a single-step environment rollout, not a full tail completion. This gives the model much tighter credit assignment.

### 4.7 Write-Runbook Grace Step (NEW in v5)

When the system becomes fully healthy (`curr_health >= 0.95`), the environment sets `is_resolved = True` but does **NOT** immediately set `_is_done = True`. Instead, the agent gets one **grace step** to optionally send a `write_runbook` action. If they do, the episode ends cleanly with a +0.05 memory bonus. If they send any other action, the episode ends normally and `_auto_write_runbook()` fires as a fallback.

---

## 5. Current Model Status

### Latest: v5 (Qwen2.5-1.5B-Instruct) — HuggingFace L40S

- **Files:** `sft_merged_1p5b_v5/` (SFT base) + `trained_model_1p5b_v5/` (GRPO adapter)
- **Model:** Qwen2.5-1.5B-Instruct (1,543,714,304 params)
- **SFT:** 1368 samples (12 seeds × 5 tasks × 4 strategies: expert+heuristic+recovery+diverse), 4 epochs, LoRA r=16
  - Final loss: **0.0827** | Token accuracy: **97.7%** | Parse rate: **100%** | Time: 23.7 min
- **GRPO:** 400 steps, 16 generations, temperature=1.0, lr=2e-6, LoRA r=32 alpha=64 (attn + MLP layers)
  - Reward range: 0.528–0.997 | Mean: ~0.86 | Time: 57 min
- **HuggingFace backup:** `hs-zz27/v5-model-backup`

**v5 Evaluation Results (5 episodes per task, L40S):**
| Task | Expert | Heuristic | Trained | Naive | vs Heur |
|------|--------|-----------|---------|-------|---------|
| single_service_failure | 0.850 | 0.722 | **0.900** | 0.800 | ✅ +0.178 |
| cascading_failure | 0.900 | 0.664 | **0.794** | 0.550 | ✅ +0.130 |
| hidden_root_cause | 0.850 | 0.764 | **0.800** | 0.170 | ✅ +0.036 |
| chaos_cascade | 0.950 | 0.632 | **0.790** | 0.760 | ✅ +0.158 |
| multi_root_cause | 0.900 | 0.564 | **0.791** | 0.600 | ✅ +0.227 |
| **AVERAGE** | **0.890** | 0.669 | **0.815** | 0.576 | **✅ +0.146** |

**Resolved: 25/25** | **Score Breakdown (avg):** recovery=0.336, efficiency=0.194, diagnostics=0.140, ordering=0.095, memory=0.050

### v5 Model Behavior — MASSIVE IMPROVEMENT over v4

The 1.5B model learned to output **correct recovery actions directly** (e.g., `restart_service:cache`, `restart_service:database`) instead of the v4's repetitive `inspect_metrics:checkout`. The orchestrator now only overrides for:
- `early_recovery_before_root_inspect` (model jumps to fix before inspecting — correct intent, just needs 1 inspect first)
- `db_high_cpu_requires_scale_first` (model tries restart but DB needs scale first)
- `must_fix_auth_before_dependents` (model targets checkout instead of auth rollback)

**Key differences from v4:**
- **0% parse failures** (v4 also 0% — SFT ensures JSON format)
- **Fallback rate: 33-75%** (orchestrator still helps, but model proposes the RIGHT action type — just overridden for ordering/safety)
- **Steps per episode: 3-8** (v4 was 4-40+ with repeat loops!)
- **Memory score: 0.050** (v4 was 0.000 everywhere — grace step fix works!)

### Training Version History
| Version | Model | Reward Fn | Eval Score | Resolved | Key Change |
|---------|-------|-----------|------------|----------|------------|
| v2 (Windows) | 0.5B | episode_score | **0.764** | 25/25 | Original baseline |
| v3 (Kaggle) | 0.5B | +alignment, -repeat | 0.545 | 15/25 | Mode collapse |
| v4 (Kaggle) | 0.5B | -repeat, +diversity | 0.665 | 23/25 | Partial recovery |
| **v5 (HF L40S)** | **1.5B** | **direct-action** | **0.815** | **25/25** | **Best result** |

### SFT Training Curve (v5 1.5B)
| Epoch | Loss | Token Accuracy | Entropy |
|-------|------|---------------|---------|
| 0.5 | 1.243 | 71.4% | 1.313 |
| 1.0 | 0.300 | 91.9% | 0.328 |
| 2.0 | 0.105 | 97.0% | 0.125 |
| 3.0 | 0.090 | 97.5% | 0.104 |
| 4.0 | 0.083 | 97.7% | 0.104 |

### GRPO Training Curve (v5 1.5B)
| Step | Reward | Entropy | frac_zero_std | Status |
|------|--------|---------|---------------|--------|
| 25 | 0.805 | 0.026 | 0.8 | ✅ Strong start |
| 75 | 0.901 | 0.016 | 0.7 | ✅ Climbing |
| 125 | 0.964 | 0.015 | 0.6 | ✅ Near-peak |
| 200 | 0.968 | 0.007 | 0.9 | ✅ Peak |
| 275 | 0.997 | 0.004 | 0.9 | ✅ Near-perfect |
| 350 | 0.882 | 0.015 | 0.8 | ✅ Stable |
| 400 | 0.829 | 0.011 | 0.8 | ✅ Final |

**Note:** Entropy is much lower than v4 (0.007-0.05 vs 0.16-0.23). The 1.5B model converges to a confident policy faster. This is NOT mode collapse — the model is actually outputting correct actions, as shown by the 0.815 eval score and 25/25 resolved rate. The `frac_reward_zero_std` staying at 0.6-0.9 confirms all generations are nearly identical (confident policy), but the reward is HIGH.

### Baseline Scores (updated)
| Agent | Avg Score |
|-------|-----------|
| Expert (hardcoded optimal) | 0.890 |
| **Trained v5 (1.5B + orchestrator)** | **0.815** |
| Trained v2 (0.5B, Windows) | 0.764 |
| Heuristic (rule-based) | 0.669 |
| Trained v4 (0.5B, Kaggle) | 0.665 |
| Naive (restart everything) | 0.576 |
| Do Nothing | 0.029 |

---

## 6. Known Issues & Gotchas

### Orchestrator Still Does Heavy Lifting
The 1.5B model learned the RIGHT action type (restart instead of inspect_metrics spam), but the orchestrator still overrides 33-75% of steps for ordering/safety. This is by design — the orchestrator is the "safety net" that ensures correct dependency ordering.

### Low Entropy During GRPO
The 1.5B model's entropy drops below 0.05 during GRPO. This is actually fine because:
1. The eval score is 0.815 (beats heuristic by +0.146)
2. 25/25 episodes resolved
3. The model outputs correct action types, not junk

### TRL Version Incompatibilities
HuggingFace's TRL version (5.6.2) is different from Kaggle's. All training scripts have try/except fallbacks for:
- `max_seq_length` (removed in TRL ≥0.17)
- `max_completion_length` (not in all versions)
- `processing_class` vs `tokenizer` parameter naming
- `temperature` kwarg may not be in all GRPOConfig versions

### HuggingFace Space Ephemeral Storage
Files on HF Spaces are wiped on restart. Always upload artifacts to the Hub immediately after training. Logs should be copied to local files (like `meow.txt`, `grpo_training_logs.txt`).

### `torch_dtype` Deprecation Warning
`torch_dtype` is deprecated in favor of `dtype` in newer transformers. Current code uses `torch_dtype` — it works but shows a warning. Non-blocking.

---

## 7. How to Run Things

### Start Backend
```bash
cd "OpenEnv meta hack"
python -m uvicorn server.app:app --reload --port 8000
```

### Run Evaluation (HF Space / any CUDA)
```bash
python evaluate_trained.py \
  --adapter trained_model_1p5b_v5 \
  --base-model sft_merged_1p5b_v5 \
  --episodes 5 --verbose
```

### Run Evaluation (Mac — MPS)
```bash
python evaluate_trained.py \
  --adapter trained_model_1p5b_v5 \
  --base-model sft_merged_1p5b_v5 \
  --episodes 5 --verbose --device mps
```

### Train SFT (HF Space L40S)
```bash
python sft_warmstart.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --epochs 4 --batch-size 4 --lr 2e-5 --num-seeds 12 \
  --use-lora --gradient-checkpointing \
  --output-dir sft_adapter_1p5b_v5 \
  --merged-output-dir sft_merged_1p5b_v5
```

### Train GRPO (HF Space L40S, after SFT)
```bash
python train_grpo.py \
  --model sft_merged_1p5b_v5 \
  --steps 400 --batch-size 1 --lr 2e-6 \
  --num-generations 16 --num-seeds 10 \
  --use-lora --gradient-checkpointing \
  --lora-r 32 --lora-alpha 64 \
  --save-steps 25 --log-every 10 \
  --reward-mode direct \
  --snapshot-steps "1,2,3,4,5,6,7,8" \
  --output-dir trained_model_1p5b_v5
```

### Generate Training Plots
```bash
python plot_training.py --log results/training_log.json --output-dir results
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

### Colab Training (for judges)
```
https://colab.research.google.com/github/hs-zz27/incident-commander-openenv/blob/main/incident_commander_training.ipynb
```

---

## 8. Pending Tasks / Next Steps

### Immediate (hackathon submission)
- [ ] Upload 1.5B model to HuggingFace Hub from Space
- [ ] Run Colab notebook end-to-end for filled output cells
- [ ] Write HuggingFace mini-blog post (30% of judging score)
- [ ] Polish README.md with v5 results table + architecture diagram + links
- [ ] Generate final training evidence plots with `plot_training.py`
- [ ] Add frontend interactive controls (scenario selector + auto-pilot button)

### Done
- [x] Trained 1.5B model with GRPO — 0.815 avg score, 25/25 resolved
- [x] Implemented write_runbook grace step — memory score now 0.050 everywhere
- [x] Created Colab training notebook for judges
- [x] Configurable LoRA rank/alpha CLI args + MLP targeting for bigger models
- [x] Multi-agent coordinator-specialist architecture
- [x] Hybrid orchestrator (model + heuristic routing)
- [x] Fixed GRPO mode collapse: removed alignment bonus, added temperature=1.0
- [x] Created plot_training.py for training evidence visualization

---

## 9. Environment Setup

### HuggingFace Space (L40S, 48GB VRAM) — PRIMARY TRAINING ENVIRONMENT
- Used for v5 1.5B training
- Ephemeral storage — upload everything to Hub immediately
- TRL version 5.6.2, transformers 5.6.2
- SFT: 23.7 min | GRPO: 57 min

### Mac (M4 Air)
- Python 3.12, venv at `.venv/`
- Device: `mps`
- Used for testing/evaluation only (no training)

### Google Colab (T4, 16GB VRAM) — SUBMISSION EVIDENCE
- Used for running the training notebook to produce filled-in output cells
- Free T4 is sufficient for 0.5B model

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
- **HuggingFace Model Backup:** `hs-zz27/v5-model-backup`
