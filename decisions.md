# Decisions Log

Every design decision made during development, who made it, and why.

## Decisions Made Without User Approval

### D-01: MIT License
- REVERTED — removed after user flagged it

### D-02: Author Name — "Incident Commander Team"
- File: pyproject.toml — Change to your real name

### D-03: Default LLM Model — gpt-4o-mini
- File: inference.py — May not work with HuggingFace router

### D-04: Default API Endpoint — https://api.openai.com/v1
- Hackathon sample uses https://router.huggingface.co/v1

### D-05: 6 Microservices (database, cache, auth, notification, payments, checkout)
- Realistic e-commerce backend with enough complexity for 3 difficulty levels

### D-06: Health Score Weights
- database=25%, auth=20%, payments=20%, checkout=20%, cache=10%, notification=5%

### D-07: Reward Values
- inspect root cause: +0.05 | other: +0.02 | repeat: -0.01
- health delta x2.0 | correct recovery: +0.15 | wrong target: -0.03
- repeated action: -0.05 | do_nothing: -0.03 | completion: +0.20

### D-08: Grader Weights
- Recovery 40%, Efficiency 25%, Diagnostics 15%, Ordering 20%

### D-09: Max Steps — Easy=15, Medium=20, Hard=30

### D-10: Python >= 3.10

### D-11: Environment Name — incident_commander_env

### D-12: Port 8000

### D-13: Version v1.0.0 = healthy, anything else = bad deploy

### D-14: Dependency propagation is instant (same step)

### D-15: Bad deploys resist restart, require rollback

### D-16: Pydantic v2

### D-17: No heavy dependencies for core env

### D-18: Fully deterministic, no randomness
- PARTIALLY REVERSED — D-24 adds optional seed param. Default behavior stays deterministic for test compatibility.

## Decisions Dictated by Hackathon/OpenEnv Spec

- FastAPI server, openenv.yaml, inference.py in root
- [START]/[STEP]/[END] stdout format, HF_TOKEN, score= field, flush=True
- OpenAI client, Docker, HF Spaces, 3+ tasks, scores 0.0-1.0
- [project.scripts] server, uv.lock, /metadata, /schema endpoints

## New Decisions (Before Bangalore Sprint)

### D-19: Randomized Incident Generator
- File: server/tasks.py — `_build_random_task(seed)`
- Parameterizes: root service, failure mode (oom/bad_deploy/cpu_spike/network_partition), downstream
- Registered as `random_incident` in TASK_REGISTRY
- Uses dependency graph for realistic failure propagation

### D-20: Multi-Specialist Agent Architecture
- File: multi_agent_inference.py (NEW, does NOT modify inference.py)
- Coordinator + 3 specialists: db_expert, infra_expert, app_expert
- Each specialist has focused system prompt with reasoning examples
- Coordinator outputs JSON delegation, specialist outputs JSON action

### D-21: ChaosAgent — Background Failure Injection
- File: server/chaos.py (NEW), server/environment.py (modified)
- 15% probability per step after step 5
- Prefers injecting into services agent isn't currently investigating
- `chaos_mode=False` by default — opt-in via /reset param

### D-22: Chaos survival reward bonus
- +0.05 reward if agent handles chaos injection without health loss
- Stacks with normal step reward

### D-23: Task 4 — Chaos Cascade
- File: server/tasks.py — `chaos_cascade`
- DB crash + scripted notification failure at step 8
- Max steps: 35, difficulty: hard

### D-24: Task 5 — Multi-Root Cause
- File: server/tasks.py — `multi_root_cause`
- Auth bad deploy + database CPU spike simultaneously
- Max steps: 40, difficulty: expert

### D-25: D-18 Partial Reversal — Optional Seed
- Default `seed=None` uses `random.randint(0, 99999)` for chaos RNG
- Explicit seed makes chaos/random episodes reproducible
- Existing 3 fixed tasks remain fully deterministic

### D-26: Live Dashboard Endpoint
- File: server/app.py — `GET /dashboard`
- Auto-refreshing HTML (meta refresh 2s), dark theme
- Shows: service status, health bars, step count, score
- For live demo during pitch presentation

### D-27: Baseline Benchmark Framework
- File: run_baselines.py (NEW)
- RandomAgent, HeuristicAgent, TrainedAgent (placeholder)
- Runs N episodes per agent per task, saves JSON to results/

### D-28: Reward Curve Plotting
- File: plot_baselines.py (NEW)
- Dark-theme matplotlib charts for blog/pitch
- Bar chart per task + overall summary horizontal bar

### D-29: Training Script Structure
- File: train_grpo.py (NEW)
- GRPO via TRL library, target: Qwen2.5-1.5B-Instruct
- Dry-run mode for local testing without GPU
- Full training to be done in Bangalore with compute credits

