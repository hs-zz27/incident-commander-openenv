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
│  Trained Model (Qwen 0.5B + LoRA adapter)               │
│  Greedy decode, fuzzy parser, orchestrator routing       │
└─────────────────────────────────────────────────────────┘
```

### Hybrid Orchestrator (IMPORTANT)
The shipped inference path is a **hybrid orchestrator** that routes between:
- the trained model's proposed action (when safe), and
- a deterministic, task-aware heuristic expert (when the model is wrong / repeating / contradicts known patterns).

**Source of truth:** `orchestrator.py` (used by `evaluate_trained.py`, `inference.py`, and `server/app.py:/predict`).

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
| `train_grpo.py` | **Source of truth for prompts** (`build_obs_prompt`), GRPO training loop, reward function. Contains `temperature=1.0` to prevent entropy collapse. |
| `sft_warmstart.py` | SFT warm-start: generates expert/heuristic trajectories → LoRA fine-tuning → adapter merge. Uses 8 seeds × 5 tasks. |

### Inference & Evaluation
| File | Purpose |
|------|---------|
| `inference.py` | CLI inference agent — supports API mode (`--local` flag for trained model) |
| `live_inference.py` | Frontend demo script — calls `/predict` endpoint in a loop |
| `evaluate_trained.py` | **Primary evaluation script** — trained model vs all baselines. Contains the **fuzzy parser** (`parse_action`) |
| `evaluate.py` | Expert/naive strategy definitions, `run_strategy` helper |
| `run_baselines.py` | Multi-agent baseline benchmarking (random, heuristic, LLM, trained) |
| `orchestrator.py` | **Hybrid routing policy** (model + deterministic expert), diagnostics guarantees + guardrails |
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

During GRPO training, `temperature=1.0` is set in GRPOConfig to ensure diverse completions across the 4 generations per prompt. Without this, entropy collapses and all generations become identical → reward_std=0 → no gradient → learning dies.

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

The GRPO reward is: `episode_score + repeat_penalty + diversity_bonus`
- `episode_score`: 0.0–1.0 from `env.grade()["score"]` after heuristic tail completion
- `repeat_penalty`: -0.05 if action repeats the previous action
- `diversity_bonus`: +0.03 if action is novel (not seen in history)

**History of reward function changes (critical context for debugging):**
- v2: `episode_score` only → worked well (0.764 eval)
- v3: Added `alignment_bonus=+0.15` and `repeat_penalty=-0.15` → caused mode collapse (0.545 eval)
- v4: Removed alignment bonus, kept mild `repeat_penalty=-0.05` + `diversity_bonus=+0.03` → partial recovery (0.666 eval)

---

## 5. Current Model Status

### Latest: v4 (Qwen2.5-0.5B-Instruct) — Kaggle T4
- **Files:** `sft_merged_0p5b_v4/` (SFT base) + `trained_model_0p5b_v4/` (GRPO adapter)
- **SFT:** ~560 samples (8 seeds × 5 tasks × heuristic+expert), 3 epochs, LoRA r=16
- **GRPO:** 300 steps, 4 generations, temperature=1.0, lr=5e-6
- **Training time:** SFT ~10 min, GRPO ~33 min on T4
- **GRPO Health:** entropy stayed at 0.16–0.23 (no collapse!), frac_reward_zero_std 0.2–0.6 (healthy)
- **Reward trajectory:** 0.598 → 0.743 (peak step 150) → 0.696 (final step 300)

**v4 Evaluation Results (5 episodes per task, Kaggle T4):**
| Task | Expert | Heuristic | Trained | Naive | vs Heur |
|------|--------|-----------|---------|-------|---------|
| single_service_failure | 0.850 | 0.761 | **0.833** | 0.750 | ✅ +0.072 |
| cascading_failure | 0.900 | 0.643 | 0.516 | 0.550 | ❌ -0.127 |
| hidden_root_cause | 0.750 | 0.750 | **0.750** | 0.170 | ⬜ +0.000 |
| chaos_cascade | 0.860 | 0.683 | 0.588 | 0.710 | ❌ -0.095 |
| multi_root_cause | 0.900 | 0.596 | **0.640** | 0.550 | ✅ +0.044 |
| **AVERAGE** | 0.852 | 0.687 | **0.665** | 0.546 | ❌ -0.021 |

**Resolved: 23/25** | **Score Breakdown (avg):** recovery=0.332, efficiency=0.087, diagnostics=0.150, ordering=0.116, memory=0.000

### v4 Model Behavior
The model has learned to output `inspect_metrics:checkout` or `inspect_metrics:database` as its default action. It repeats this, and the orchestrator overrides (repeat×2) kick in to route to heuristic actions. The orchestrator does most of the actual decision-making — the model contributes diagnostics actions at steps 1–2 and the orchestrator handles recovery.

**Fallback rate:** ~31-50% of steps are orchestrator overrides. Lower fallback = model contributing more.

### Training Version History
| Version | Reward Fn | GRPO Temp | Eval Score | Resolved | Key Issue |
|---------|-----------|-----------|------------|----------|-----------|
| v2 (Windows) | episode_score only | model default | **0.764** | **25/25** | Best result |
| v3 (Kaggle) | +alignment(0.15), -repeat(0.15) | model default | 0.545 | 15/25 | Mode collapse: alignment bonus too strong, entropy→0.03 |
| v4 (Kaggle) | -repeat(0.05), +diversity(0.03) | 1.0 (fixed) | 0.665 | 23/25 | Better but model still defaults to inspect_metrics |

### Baseline Scores (for comparison)
| Agent | Avg Score |
|-------|-----------|
| Expert (hardcoded optimal) | 0.852 |
| Heuristic (rule-based, in evaluate_trained.py) | 0.687 |
| Trained v4 (model + orchestrator) | 0.665 |
| Trained v2 (model + orchestrator, Windows) | 0.764 |
| Naive (restart everything) | 0.546 |
| Do Nothing | 0.029 |

---

## 6. Known Issues & Gotchas

### Model Still Partially Mode-Collapsed (v4)
Despite fixing the alignment bonus and adding temperature=1.0, the v4 model still defaults to `inspect_metrics:checkout` or `inspect_metrics:database` as its go-to action, then repeats it. The orchestrator's repeat×2 override does the real work.

**Why v2 was better:** v2 was trained with the original SFT data (diverse strategies) and plain episode_score reward on a different TRL version (Windows). v3/v4 SFT strategies were aligned to always start with `inspect_logs:root_cause`, which may have reduced diversity.

**Potential fixes not yet tried:**
- Revert SFT heuristic strategies to more diverse orderings (v2-style)
- Increase `num_generations` to 8 (more diverse rollouts)
- Use a cosine KL coefficient schedule
- Try PPO or REINFORCE instead of GRPO

### TRL Version Incompatibilities
Kaggle's TRL version is different from local. All training scripts have try/except fallbacks for:
- `max_seq_length` (removed in TRL ≥0.17)
- `max_completion_length` (not in all versions)
- `processing_class` vs `tokenizer` parameter naming
- `temperature` kwarg may not be in all GRPOConfig versions

### P100 Not Supported
Kaggle's current PyTorch dropped P100 support (CUDA capability 6.0). Use **T4** (capability 7.5).

### `torch_dtype` Deprecation Warning
`torch_dtype` is deprecated in favor of `dtype` in newer transformers. Current code uses `torch_dtype` — it works but shows a warning. Non-blocking.

### Time-Based Scoring in HTTP Mode
When running via the FastAPI server (frontend), episodes have **wall-clock time limits**. The model must act within the SLA or receive a penalty. This is NOT active in direct Python mode (evaluation scripts).

### `CUDA_VISIBLE_DEVICES=0` Required on Kaggle T4×2
Multi-GPU causes TRL to use distributed training, which wastes memory on 0.5B model. Always set `CUDA_VISIBLE_DEVICES=0` to use single GPU.

---

## 7. How to Run Things

### Start Backend
```bash
cd "OpenEnv meta hack"
python -m uvicorn server.app:app --reload --port 8000
```

### Run Evaluation (Kaggle — CUDA)
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_trained.py \
  --adapter trained_model_0p5b_v4 \
  --base-model sft_merged_0p5b_v4 \
  --episodes 5 --verbose
```

### Run Evaluation (Mac — MPS)
```bash
python evaluate_trained.py \
  --adapter trained_model_0p5b_v4 \
  --base-model sft_merged_0p5b_v4 \
  --episodes 5 --verbose --device mps
```

### Train SFT (Kaggle T4)
```bash
CUDA_VISIBLE_DEVICES=0 python sft_warmstart.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --epochs 3 --batch-size 4 --lr 2e-5 --num-seeds 8 \
  --use-lora --gradient-checkpointing \
  --output-dir sft_adapter_0p5b_v4 \
  --merged-output-dir sft_merged_0p5b_v4
```

### Train GRPO (Kaggle T4, after SFT)
```bash
CUDA_VISIBLE_DEVICES=0 python train_grpo.py \
  --model sft_merged_0p5b_v4 \
  --steps 300 --batch-size 1 --lr 5e-6 \
  --num-generations 4 --num-seeds 5 \
  --use-lora --gradient-checkpointing \
  --save-steps 50 --log-every 10 \
  --output-dir trained_model_0p5b_v4
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

---

## 8. Pending Tasks / Next Steps

### Immediate
- [ ] Download v4 model zips from Kaggle and extract locally
- [ ] Evaluate v4 locally on Mac/MPS to confirm scores
- [ ] Generate final training evidence plots with `plot_training.py`
- [ ] Embed plots into README.md for hackathon judges

### If Time Permits (score improvement)
- [ ] Try reverting SFT strategies to v2-style diverse orderings
- [ ] Increase `num_generations` from 4 to 8 for more diverse GRPO rollouts
- [ ] Investigate training with `num_seeds=10` (more diverse prompts)
- [ ] Add `write_runbook` action at end of episodes to capture memory score (currently 0.000 everywhere)

### Done
- [x] Fixed GRPO mode collapse: removed alignment bonus, added temperature=1.0, diversity bonus
- [x] Implemented orchestrator.py (hybrid model + heuristic routing)
- [x] Wired orchestrator into evaluate_trained.py, inference.py, server/app.py
- [x] Created plot_training.py for training evidence visualization
- [x] Aligned SFT strategies with orchestrator diagnostic order

---

## 9. Environment Setup

### Mac (M4 Air)
- Python 3.12, venv at `.venv312/`
- Device: `mps`
- Used for testing/evaluation only (no training)

### Kaggle (GPU T4 x2, 16GB VRAM) — PRIMARY TRAINING ENVIRONMENT
- Used for all v3/v4 training
- Must enable Internet in notebook settings
- Always set `CUDA_VISIBLE_DEVICES=0` (single GPU)
- Don't use `--use-4bit` for 0.5B model (unnecessary, bf16 fits in 16GB)
- See `kaggle_training.md` for full setup

### Dependencies
```
pip install -e ".[train]"
# Which installs: trl, transformers, torch, accelerate, peft, bitsandbytes, datasets
```

**Note:** Unit tests expect `pyyaml` (for `openenv.yaml` validation) and it is now included in `pyproject.toml`.

---

## 10. Git Info
- **Repo:** `https://github.com/hs-zz27/incident-commander-openenv.git`
- **Branch:** `main`
- **Important:** Many files are gitignored (model weights, eval results, logs). Check `.gitignore`.

---

## 11. GRPO Training Metrics Reference (v4)

Key metrics to watch during GRPO training (from `grpo_training_logs.txt`):

| Step | Reward | Entropy | frac_zero_std | Status |
|------|--------|---------|---------------|--------|
| 10 | 0.598 | 0.163 | 0.3 | ✅ Healthy |
| 50 | 0.672 | 0.189 | 0.3 | ✅ Healthy |
| 100 | 0.667 | 0.186 | 0.1 | ✅ Healthy |
| 150 | 0.743 | 0.194 | 0.6 | ✅ Peak reward |
| 200 | 0.711 | 0.198 | 0.5 | ✅ Healthy |
| 250 | 0.714 | 0.198 | 0.5 | ✅ Stable |
| 300 | 0.696 | 0.168 | 0.2 | ✅ Final |

**Healthy training indicators:**
- `entropy` > 0.05 (v3 collapsed to 0.029 = dead)
- `frac_reward_zero_std` < 0.8 (v3 hit 1.0 = all generations identical)
- `reward_std` > 0.01 (if zero, no gradient signal)
