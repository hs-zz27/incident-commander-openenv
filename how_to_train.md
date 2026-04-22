# 🧠 How to Train the Incident Commander Model

> **Complete GRPO Training Guide** — From zero to a fine-tuned LLM that resolves production incidents autonomously.

---

## Table of Contents

1. [Overview](#1-overview)
2. [What is GRPO?](#2-what-is-grpo)
3. [Architecture — How Training Works](#3-architecture--how-training-works)
4. [Prerequisites](#4-prerequisites)
5. [Environment Setup](#5-environment-setup)
6. [Understanding the Reward Function](#6-understanding-the-reward-function)
7. [Training Prompts & Data Pipeline](#7-training-prompts--data-pipeline)
8. [Dry Run — Validate Before Training](#8-dry-run--validate-before-training)
9. [Full GRPO Training (GPU Required)](#9-full-grpo-training-gpu-required)
10. [Cloud GPU Setup](#10-cloud-gpu-setup)
11. [Hyperparameter Tuning](#11-hyperparameter-tuning)
12. [Monitoring & Logging](#12-monitoring--logging)
13. [Post-Training Evaluation](#13-post-training-evaluation)
14. [Inference with Trained Model](#14-inference-with-trained-model)
15. [Troubleshooting](#15-troubleshooting)
16. [FAQ](#16-faq)

---

## 1. Overview

This project trains a small language model (default: **Qwen2.5-1.5B-Instruct**) to act as an **SRE Incident Commander** — an AI agent that diagnoses and resolves production microservice outages.

### Why Train?

| Approach | Pros | Cons |
|----------|------|------|
| **Prompting GPT-4o** | High quality, no training needed | Expensive per-call, high latency, closed-source |
| **GRPO-trained small model** | Fast inference, low cost, self-hosted | Requires GPU time for training |

### What You'll Get

After training, you'll have a **fine-tuned model** that:
- Outputs valid JSON actions for the environment
- Diagnoses root causes by inspecting logs and metrics
- Fixes services in correct dependency order
- Scores **0.60–0.95** across all three difficulty levels

---

## 2. What is GRPO?

**Group Relative Policy Optimization (GRPO)** is a reinforcement learning algorithm from DeepSeek, designed to train language models using reward signals — without needing a separate critic model.

### How GRPO Differs from Other Methods

| Method | Needs Critic? | Needs Reference Data? | Our Use |
|--------|--------------|----------------------|---------|
| **SFT (Supervised Fine-Tuning)** | ❌ | ✅ Expert trajectories | Not used — we don't have expert data |
| **PPO** | ✅ Value model | ❌ | Too memory-heavy for small GPU |
| **DPO** | ❌ | ✅ Preference pairs | Not used — no paired data |
| **GRPO** ✅ | ❌ | ❌ | **Our choice** — uses environment reward directly |

### GRPO in Plain English

1. **Generate**: The model generates **N responses** (a "group") for each prompt
2. **Evaluate**: Each response is scored using our environment reward function
3. **Rank**: Responses are ranked *within the group* (relative comparison)
4. **Update**: The model is updated to produce responses more like the high-reward ones

This is powerful because the model learns from its own attempts — no expert demonstrations needed.

---

## 3. Architecture — How Training Works

```
┌─────────────────────────────────────────────────────────────┐
│                     GRPO Training Loop                      │
│                                                             │
│  ┌──────────┐     ┌─────────────────────┐                   │
│  │  Prompt   │────►│  LLM (Qwen 1.5B)   │                   │
│  │ Generator │     │  Generates N actions │                   │
│  └──────────┘     └─────────┬───────────┘                   │
│       ▲                     │                               │
│       │              N generated responses                  │
│       │                     │                               │
│       │           ┌─────────▼───────────┐                   │
│       │           │  Reward Function     │                   │
│       │           │  (env.step + score)  │                   │
│       │           └─────────┬───────────┘                   │
│       │                     │                               │
│       │              N reward values                        │
│       │                     │                               │
│       │           ┌─────────▼───────────┐                   │
│       │           │  GRPO Optimizer      │                   │
│       │           │  Rank within group   │                   │
│       │           │  Update model weights│                   │
│       │           └─────────┬───────────┘                   │
│       │                     │                               │
│       └─────────────────────┘                               │
│              Next training step                             │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **Training script** | `train_grpo.py` | Main entrypoint — CLI, training loop, logging |
| **Reward wrapper** | `train_grpo.py` → `IncidentCommanderRewardFunction` | Parses model text → action, steps env, returns reward |
| **Environment** | `server/environment.py` → `IncidentCommanderEnvironment` | Simulates the microservice outage |
| **Grader** | `server/grader.py` | Computes per-step rewards and final episode score |
| **Task definitions** | `server/tasks.py` | Defines the 3 training scenarios |

---

## 4. Prerequisites

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **GPU** | NVIDIA T4 (16 GB) | NVIDIA A100 (40/80 GB) |
| **GPU RAM** | 16 GB | 40 GB+ |
| **System RAM** | 16 GB | 32 GB+ |
| **Disk** | 20 GB free | 50 GB+ |
| **CUDA** | 11.8+ | 12.1+ |

> **No GPU?** Use `--dry-run` to validate everything works, then train on a cloud GPU (see [Section 10](#10-cloud-gpu-setup)).

### Software Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.10 | Runtime |
| PyTorch | ≥ 2.0 | Deep learning framework |
| TRL | ≥ 0.7.0 | GRPO trainer implementation |
| Transformers | ≥ 4.36.0 | Model loading & tokenizer |
| Accelerate | ≥ 0.25.0 | Multi-GPU / mixed precision |
| PEFT | ≥ 0.7.0 | LoRA adapters (memory-efficient training) |
| bitsandbytes | ≥ 0.41.0 | 4-bit/8-bit quantization |

---

## 5. Environment Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/hs-zz27/incident-commander-openenv.git
cd incident-commander-openenv
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate    # macOS / Linux
```

### Step 3: Install Core Project Dependencies

```bash
pip install -e ".[dev]"
```

### Step 4: Install Training Dependencies

```bash
pip install trl transformers torch accelerate peft bitsandbytes
```

### Step 5: Verify Environment Integration

```bash
# Run tests to ensure the environment works
python -m pytest tests/ -q
# Expected: 134 passed

# Run evaluation to validate grading
python evaluate.py
# Expected: 🎉 ALL CHECKS PASSED
```

### Step 6: Verify GPU Availability (if training)

```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

---

## 6. Understanding the Reward Function

The reward function is the **core** of training — it teaches the model what "good" incident response looks like.

### Reward Function Class: `IncidentCommanderRewardFunction`

Located in `train_grpo.py`, this class:

1. **Receives** the model's generated text response
2. **Parses** it as a JSON action (e.g., `{"action_type": "restart_service", "service_name": "cache"}`)
3. **Steps** the environment with that action
4. **Returns** the environment's reward signal

```python
class IncidentCommanderRewardFunction:
    def get_reward(self, response_text: str) -> float:
        """
        Parse model response → action → step environment → return reward.
        Returns -0.1 for unparseable responses.
        """
```

### Per-Step Reward Breakdown

| Action/Event | Reward | Rationale |
|-------------|--------|-----------|
| First inspect of **root cause** service | **+0.05** | Investigating the actual problem |
| First inspect of **other** service | **+0.02** | Any investigation is slightly good |
| Repeated inspect of same service | **-0.01** | Stop wasting time |
| Health improvement (Δ health × 2.0) | **+variable** | Biggest reward — actually fixing things |
| Correct recovery action | **+0.15** | Taking the right fix action |
| Recovery on wrong target | **-0.03** | Fixing the wrong thing |
| Repeated same action | **-0.05** | Don't spam the same action |
| `do_nothing` during outage | **-0.03** | There's a crisis — act! |
| Episode completion bonus | **+0.20** | Bonus for fully resolving the incident |
| **Unparseable response** | **-0.10** | Teaches the model to output valid JSON |

### Why This Reward Design Works

- **Shaped rewards** (many small signals per step) beat sparse rewards (only score at end)
- **Negative for bad JSON** forces the model to learn the output format quickly
- **Health delta × 2.0** makes actually fixing services the highest-reward action
- **Ordering bonus** rewards fixing dependencies bottom-up (database → auth → checkout)

---

## 7. Training Prompts & Data Pipeline

### How Prompts Are Generated

Training prompts are built dynamically from environment observations — **no static dataset required**.

```python
def build_obs_prompt(obs_dict, step, action_history) -> str:
    """
    Converts an environment observation into a training prompt.
    
    Example output:
    ─────────────────────────────────────────
    You are an SRE Incident Commander. Diagnose and fix the following incident.
    Step 1/15
    System Health: 55.00%

    Services:
      auth: degraded | err=25.0% | lat=500ms | ver=v1.0.0
      cache: down | err=100.0% | lat=0ms | ver=v1.0.0
      checkout: degraded | err=40.0% | lat=800ms | ver=v1.0.0
      database: healthy | err=1.0% | lat=20ms | ver=v1.0.0
      notification: healthy | err=2.0% | lat=30ms | ver=v1.0.0
      payments: degraded | err=30.0% | lat=600ms | ver=v1.0.0

    Respond with a JSON action: {"action_type": "...", "service_name": "..."}
    ─────────────────────────────────────────
    """
```

### Expected Model Output

The model should generate a valid JSON action:

```json
{"action_type": "inspect_logs", "service_name": "cache"}
```

### The 3 Training Scenarios

| Task | Difficulty | Max Steps | Root Cause | Optimal Solution |
|------|-----------|-----------|------------|------------------|
| `single_service_failure` | Easy | 15 | Cache OOM crash | Inspect cache → restart cache |
| `cascading_failure` | Medium | 20 | Database overload | Scale DB → restart DB → restart dependents in order |
| `hidden_root_cause` | Hard | 30 | Bad auth deployment (v2.2.0-rc1) | Rollback auth → clear cache → restart dependents |

### The 8 Available Actions

| Action | Requires Service Name | Description |
|--------|----------------------|-------------|
| `inspect_logs` | ✅ | Read log entries for a service |
| `inspect_metrics` | ✅ | Get CPU, memory, latency metrics |
| `restart_service` | ✅ | Restart a service (won't fix bad deploys) |
| `scale_service` | ✅ | Add another instance |
| `rollback` | ✅ | Roll back to previous stable version |
| `clear_cache` | ❌ | Flush all cached data |
| `escalate` | ❌ | Give up and call a human |
| `do_nothing` | ❌ | Skip your turn (penalized) |

---

## 8. Dry Run — Validate Before Training

**Always run a dry run first** to validate the training wrapper works without needing a GPU.

```bash
python train_grpo.py --dry-run
```

### What the Dry Run Does

1. ✅ Initializes the `IncidentCommanderRewardFunction`
2. ✅ Resets the environment for each of the 3 tasks
3. ✅ Simulates model responses with test actions
4. ✅ Steps the environment and collects rewards
5. ✅ Saves a training log to `results/training_log.json`

### Expected Dry Run Output

```
============================================================
  GRPO Training — Incident Commander
  Model: Qwen/Qwen2.5-1.5B-Instruct
  Steps: 300
  Batch size: 4
  Learning rate: 5e-06
============================================================

⚠️  DRY RUN MODE — no GPU training, testing environment wrapper only

  Step 1: mean_reward=0.0233 task=cascading_failure
  Step 2: mean_reward=0.0150 task=hidden_root_cause
  ...

✅ Training log saved to results/training_log.json
✅ Dry run complete. Run on GPU with --no-dry-run for actual training.
```

### Verify the Training Log

```bash
cat results/training_log.json | python -m json.tool | head -20
```

---

## 9. Full GRPO Training (GPU Required)

### Basic Training Command

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 300
```

### Full Training Command with All Options (Memory-Aware)

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 500 \
  --batch-size 4 \
  --lr 5e-6 \
  --output-dir trained_model \
  --log-every 50 \
  --use-lora \
  --use-4bit \
  --gradient-checkpointing \
  --save-steps 100 \
  --num-seeds 3
```

### CLI Arguments Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | `Qwen/Qwen2.5-1.5B-Instruct` | HuggingFace model name or local path |
| `--steps` | `300` | Total number of GRPO training steps |
| `--batch-size` | `4` | Per-device training batch size |
| `--lr` | `5e-6` | Learning rate for the optimizer |
| `--output-dir` | `trained_model` | Directory to save the fine-tuned model |
| `--log-every` | `50` | Print training metrics every N steps |
| `--dry-run` | `false` | Test wrapper without GPU |
| `--use-lora` | `false` | Use LoRA adapters for memory-efficient training |
| `--use-4bit` | `false` | Load model in 4-bit quantization (requires bitsandbytes) |
| `--gradient-checkpointing`| `false` | Enable gradient checkpointing to reduce VRAM usage |
| `--save-steps` | `50` | Save checkpoint every N steps |
| `--resume-from-checkpoint`| `None` | Path to checkpoint directory to resume training from |
| `--num-generations`| `4` | Number of completions per prompt (GRPO group size) |
| `--num-seeds` | `3` | Number of seeds per task for dataset diversity |

### What Happens During Training

```text
Step 1/300:
  1. Dataset generates N completions per prompt
  2. Model generates completions           → '{"action_type": "inspect_logs", "service_name": "cache"}'
  3. Each response parsed & env stepped     → reward = +0.05
  4. Rewards ranked within the group        → relative ranking computed
  5. Model weights updated via GRPO         → gradient step applied
  6. Repeat with new/same prompt

Step 50/300:  (--log-every 50)
  mean_reward=0.0450  ← model improving
  
Step 300/300:
  Model saved to trained_model/
  Training log saved to results/training_log.json
```

### GRPO Configuration Details

The training script uses these TRL `GRPOConfig` settings:

```python
GRPOConfig(
    output_dir="trained_model",
    num_train_epochs=1,
    max_steps=300,                       # Total training steps
    per_device_train_batch_size=4,       # Batch size per GPU
    learning_rate=5e-6,                  # Conservative LR for fine-tuning
    logging_steps=50,                    # Log every 50 steps
    save_steps=50,                       # Save checkpoints
    gradient_accumulation_steps=2,       # Effective batch = 4 × 2 = 8
    bf16=True,                           # BFloat16 mixed precision
    num_generations=4,                   # GRPO group size
)
```

### Training Output Files

| Path | Content |
|------|---------|
| `trained_model/` | HuggingFace-compatible model checkpoint or LoRA adapter |
| `trained_model/config.json` | Model configuration |
| `trained_model/tokenizer.json` | Tokenizer files |
| `trained_model/model.safetensors` | Model weights or adapter weights |
| `results/training_log.json` | Step-by-step reward data and metrics |

---

## 10. Cloud GPU Setup

### Option A: Google Colab (Free Tier — T4 GPU)

```python
# Cell 1: Clone and install
!git clone https://github.com/hs-zz27/incident-commander-openenv.git
%cd incident-commander-openenv
!pip install -e ".[train]"

# Cell 2: Verify GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Cell 3: Dry run
!python train_grpo.py --dry-run --steps 5

# Cell 4: Full training (memory-safe on T4)
!python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 50 \
  --batch-size 1 \
  --use-lora \
  --use-4bit \
  --gradient-checkpointing \
  --output-dir trained_model

# Cell 5: Download the trained adapter
from google.colab import files
!zip -r trained_model.zip trained_model/
files.download("trained_model.zip")
```

> **Note:** Because we used `--use-lora` and `--use-4bit`, the saved `trained_model` is a LoRA adapter, not the full model. See the Inference section on how to load it.

### Option B: Lambda Labs / RunPod / Vast.ai

```bash
# 1. Provision GPU instance (A100 recommended)
# 2. SSH in
ssh user@gpu-instance-ip

# 3. Clone & setup
git clone https://github.com/hs-zz27/incident-commander-openenv.git
cd incident-commander-openenv

pip install -e ".[train]"

# 4. Dry run
python train_grpo.py --dry-run --steps 5

# 5. Full training (A100 can handle full fine-tuning without LoRA)
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 500 \
  --batch-size 8 \
  --output-dir trained_model

# 6. Download results
scp -r user@gpu-instance-ip:~/incident-commander-openenv/trained_model ./
```

### Option C: Docker on GPU Machine

```bash
# Build
docker build -t incident-commander-train .

# Run with NVIDIA GPU passthrough
docker run --gpus all \
  -v $(pwd)/trained_model:/app/trained_model \
  -v $(pwd)/results:/app/results \
  incident-commander-train \
  python train_grpo.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --steps 300 \
    --output-dir trained_model
```

> **Note:** Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

### Option D: HuggingFace Spaces (Inference Only)

After training locally or on cloud GPU, deploy the trained model:

```bash
# Push trained model to HuggingFace Hub
pip install huggingface_hub
huggingface-cli upload your-username/incident-commander-trained ./trained_model
```

---

## 11. Hyperparameter Tuning

### Recommended Configurations

#### Quick Experiment (< 30 min on T4)

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 100 \
  --batch-size 2 \
  --lr 1e-5 \
  --log-every 10
```

#### Standard Training (~ 1–2 hours on A100)

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 300 \
  --batch-size 4 \
  --lr 5e-6 \
  --log-every 50
```

#### Extended Training (~ 3–5 hours on A100)

```bash
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 500 \
  --batch-size 8 \
  --lr 3e-6 \
  --log-every 50
```

### Hyperparameter Guide

| Parameter | Too Low | Sweet Spot | Too High |
|-----------|---------|------------|----------|
| **Learning Rate** | Slow convergence | `3e-6` to `1e-5` | Unstable / mode collapse |
| **Batch Size** | Noisy gradients | `4` (T4) / `8` (A100) | OOM errors |
| **Steps** | Undertrained | `200–500` | Overfitting / diminishing returns |
| **Gradient Accumulation** | — | `2–4` | Slower iteration |

### Alternative Base Models

| Model | Parameters | VRAM Needed | Notes |
|-------|-----------|-------------|-------|
| `Qwen/Qwen2.5-0.5B-Instruct` | 0.5B | ~4 GB | Fastest, lower quality |
| **`Qwen/Qwen2.5-1.5B-Instruct`** | 1.5B | ~8 GB | **Default — best balance** |
| `Qwen/Qwen2.5-3B-Instruct` | 3B | ~16 GB | Higher quality, needs more VRAM |
| `meta-llama/Llama-3.2-1B-Instruct` | 1B | ~6 GB | Alternative small model |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | ~20 GB | Strong reasoning |

---

## 12. Monitoring & Logging

### Training Log Format

The training log at `results/training_log.json` contains:

```json
[
  {
    "step": 1,
    "mean_reward": 0.0233,
    "task": "cascading_failure"
  },
  {
    "step": 2,
    "mean_reward": 0.0450,
    "task": "hidden_root_cause"
  }
]
```

### Plotting Reward Curves

After training, visualize the reward progression:

```python
import json
import matplotlib.pyplot as plt

with open("results/training_log.json") as f:
    log = json.load(f)

steps = [entry["step"] for entry in log]
rewards = [entry["mean_reward"] for entry in log]

plt.figure(figsize=(10, 6))
plt.plot(steps, rewards, marker='o', linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Mean Reward")
plt.title("GRPO Training — Reward Curve")
plt.grid(True, alpha=0.3)
plt.savefig("results/reward_curve.png", dpi=150, bbox_inches="tight")
plt.show()
```

### What to Look For

| Signal | Meaning |
|--------|---------|
| **Reward trending upward** ✅ | Model is learning |
| **Reward plateaus** ⚠️ | Try increasing LR or steps |
| **Reward oscillates wildly** ❌ | LR too high — reduce it |
| **Reward stays at -0.1** ❌ | Model can't produce valid JSON — check prompt format |
| **Reward peaks then drops** ❌ | Overfitting — reduce steps or add regularization |

---

## 13. Post-Training Evaluation

After training completes, evaluate the trained model against all tasks.

### Step 1: Run Evaluation with Trained Model

```bash
# Point inference at your trained model
export API_BASE_URL="http://localhost:11434/v1"   # or wherever you serve the model
export MODEL_NAME="trained_model"

python inference.py
```

### Step 2: Compare Against Baselines

```bash
# Run baselines for comparison
python run_baselines.py --episodes 20

# Plot results
python plot_baselines.py
```

### Step 3: Grade the Model

The final grading rubric (weights adding to 1.0):

| Component | Weight | What It Measures |
|-----------|--------|------------------|
| **Recovery** | 40% | Did the system get fixed? (health score improvement) |
| **Efficiency** | 25% | How quickly? (fewer steps = better) |
| **Diagnostics** | 15% | Did the agent investigate before acting? |
| **Ordering** | 20% | Were actions in the correct dependency order? |

### Expected Score Targets

| Task | Random Agent | Heuristic Agent | **Trained Model Target** | Expert |
|------|-------------|-----------------|-------------------------|--------|
| Easy (single_service_failure) | 0.10–0.30 | 0.60–0.80 | **0.75–0.95** | 0.95 |
| Medium (cascading_failure) | 0.05–0.20 | 0.50–0.70 | **0.60–1.00** | 1.00 |
| Hard (hidden_root_cause) | 0.02–0.10 | 0.20–0.40 | **0.40–0.85** | 0.85 |

---

## 14. Inference with Trained Model

### Option A: Serve with vLLM (Recommended for Production)

```bash
pip install vllm

# Serve the trained model
python -m vllm.entrypoints.openai.api_server \
  --model ./trained_model \
  --port 8080

# Run inference against it
export API_BASE_URL="http://localhost:8080/v1"
export MODEL_NAME="trained_model"
python inference.py
```

### Option B: Serve with Ollama (Easy Local Setup)

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM ./trained_model
PARAMETER temperature 0.3
SYSTEM "You are an SRE Incident Commander."
EOF

# Create and run
ollama create incident-commander -f Modelfile
ollama run incident-commander

# Use with inference script
export API_BASE_URL="http://localhost:11434/v1"
export MODEL_NAME="incident-commander"
python inference.py
```

### Option C: Direct Python Loading

If you trained with `--use-lora`, you need to load the base model first, then apply the adapter:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model = PeftModel.from_pretrained(base_model, "./trained_model")
tokenizer = AutoTokenizer.from_pretrained("./trained_model")

prompt = "You are an SRE Incident Commander..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.3)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

If you trained without LoRA (full fine-tune), you can just use `AutoModelForCausalLM.from_pretrained("./trained_model")`.

---

## 15. Troubleshooting

### Common Issues

| Issue | Cause | Fix |
|-------|-------|-----|
| `CUDA out of memory` | Batch size too large | Reduce `--batch-size`, add `--use-lora`, `--use-4bit`, and `--gradient-checkpointing` |
| `ImportError: No module named 'trl'` | Training deps not installed | `pip install -e ".[train]"` |
| `Model outputs gibberish` | Too few training steps | Increase `--steps` to 500+ |
| `Reward stuck at -0.1` | Model can't produce valid JSON | Check prompt format; try lower LR |
| `FileNotFoundError: results/` | First run | The script creates this automatically |
| `torch.cuda.is_available() = False` | No GPU or CUDA not installed | Install CUDA toolkit or use `--dry-run` |
| `ConnectionError` during model download | Network issue | Set `HF_HUB_OFFLINE=1` if model is cached |
| `ValueError: Unrecognized configuration class` | Loading LoRA without PeftModel | Use `PeftModel.from_pretrained` (see Section 14 Option C) |
| `bitsandbytes is required` | Missing 4-bit dependency | `pip install bitsandbytes` (or use `.[train]`) |

### Memory Optimization Tips

```bash
# 1. Use gradient checkpointing (add to GRPO config)
#    gradient_checkpointing=True

# 2. Use smaller batch size with more gradient accumulation
python train_grpo.py --batch-size 1 --steps 300

# 3. Use a smaller model
python train_grpo.py --model Qwen/Qwen2.5-0.5B-Instruct

# 4. Enable 4-bit quantization (modify train_grpo.py)
#    model = AutoModelForCausalLM.from_pretrained(
#        model_name, load_in_4bit=True
#    )
```

### Verifying a Trained Model Works

```bash
# Quick sanity check
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('./trained_model')
tokenizer = AutoTokenizer.from_pretrained('./trained_model')
print('✅ Model loaded successfully')
print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}')
"
```

---

## 16. FAQ

### Q: Do I need a GPU to run the project?
**A:** No. The environment server, tests, and evaluation all run on CPU. You only need a GPU for the `train_grpo.py` training step. Use `--dry-run` to validate without GPU.

### Q: How long does training take?
**A:** On a T4 GPU: ~30 min for 100 steps, ~2 hours for 300 steps. On an A100: ~15 min for 100 steps, ~45 min for 300 steps.

### Q: Can I use a different model besides Qwen?
**A:** Yes! Any HuggingFace causal LM works. Pass `--model <model_name>`. Recommended alternatives: `meta-llama/Llama-3.2-1B-Instruct`, `microsoft/Phi-3-mini-4k-instruct`.

### Q: What if I don't have a HuggingFace token?
**A:** For most models (including Qwen), no token is needed. Some gated models (Llama) require accepting terms at huggingface.co and setting `HF_TOKEN`.

### Q: How do I know if training worked?
**A:** Check the reward curve in `results/training_log.json`. Rewards should trend upward. Then run `python inference.py` with the trained model and compare scores against the baseline scores in this document.

### Q: Can I resume training from a checkpoint?
**A:** Yes. Point `--resume-from-checkpoint` at the saved checkpoint directory (e.g. `trained_model/checkpoint-100`):
```bash
python train_grpo.py --model Qwen/Qwen2.5-1.5B-Instruct --steps 300 --resume-from-checkpoint ./trained_model/checkpoint-100
```

### Q: What's the minimum training to see improvement?
**A:** Usually 100–200 steps shows measurable improvement in reward. 300+ steps for reliable task completion.

---

## Quick Start Cheat Sheet

```bash
# 1. Install everything
pip install -e ".[dev]"
pip install -e ".[train]"

# 2. Validate environment works
python -m pytest tests/ -q
python evaluate.py

# 3. Dry run (no GPU)
python train_grpo.py --dry-run

# 4. Train (on GPU machine)
python train_grpo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --steps 300 \
  --batch-size 4 \
  --output-dir trained_model

# 5. Evaluate trained model
python inference.py  # with trained model served
python run_baselines.py
python plot_baselines.py

# Done! 🎉
```
