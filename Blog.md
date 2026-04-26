# Training an LLM Incident Commander in a Live Microservice Outage Environment

Subtitle:
An OpenEnv environment where an LLM learns to diagnose cascading failures, find hidden root causes, and recover a live service system.

---

Production outages rarely fail one service at a time.

They spread across dependencies, surface misleading symptoms, and punish the wrong fix. A broken checkout flow might not be a checkout bug at all - it might be an overloaded database, a bad auth deployment, or a stale cache poisoning everything downstream.

For the OpenEnv Hackathon, we built an environment around exactly that problem.

Our project trains an LLM to act as an incident commander inside a live microservice outage simulator: inspecting logs, checking metrics, choosing recovery actions, and restoring the system in the right order before the incident gets worse.

Our core claim is simple:

**training improves the model's performance on this benchmark in a measurable way.**

## Why incident response?

Many agent benchmarks are still close to puzzles: answer a question, solve a static task, call a tool once, or complete a neat sandbox workflow.

Incident response is different.

It forces an agent to operate under:
- partial observability
- misleading symptoms
- dependency-aware reasoning
- long-horizon recovery planning
- penalties for premature or badly ordered actions

That makes it a strong fit for OpenEnv. It is not just about generating the right text. It is about choosing the right action in the right state, for the right reason.

We wanted to build something that teaches more than "restart everything."

## The environment

We built an OpenEnv-compliant environment around six interdependent services:

- `database`
- `cache`
- `auth`
- `notification`
- `payments`
- `checkout`

The agent can take actions such as:
- `inspect_logs`
- `inspect_metrics`
- `restart_service`
- `scale_service`
- `rollback`
- `clear_cache`
- `write_runbook`
- `do_nothing`

The environment includes several incident families:
- single-service failures
- cascading failures
- hidden root-cause incidents
- chaos scenarios
- multi-root-cause outages

This means the agent cannot win by memorizing one fixed repair script. It has to learn when to inspect, when to intervene, which service is actually upstream, and which recovery action is appropriate.

## What makes this benchmark hard

The challenge is not just "recover a broken system."

The challenge is recovering it **intelligently**.

A good policy has to learn:
- diagnostics before blind intervention
- root-cause recovery before downstream recovery
- service dependency ordering
- the difference between restart, scale-up, rollback, and cache clearing
- how to avoid loops, redundancy, and noisy but useless actions

In other words, this is not a grid-world with better branding. It is a compact operations-reasoning benchmark.

## A simple example

Suppose `checkout` is failing.

A weak policy sees checkout errors and restarts `checkout`.

But in our environment, that may be the wrong move. The real issue might be:
- `database` is overloaded, which degrades `auth`
- `auth` fails requests
- `payments` becomes unstable
- `checkout` looks broken even though it is not the real source

A better policy learns to:
1. inspect logs or metrics first
2. identify the upstream root cause
3. scale or restart the right dependency
4. recover downstream services in the correct order

That is the kind of behavior we wanted to train.

## Reward design

Reward design mattered as much as model choice.

We did not use a simple binary success signal. Instead, we shaped reward around behaviors we actually want from an incident commander:
- system recovery
- correct diagnostics
- root-cause-first repair
- efficient action sequences
- good ordering under dependencies
- useful memory and runbook behavior

This gives the model a denser learning signal than "fixed" vs "not fixed," and makes it harder to game the benchmark with shallow action spam.

## Training pipeline

We used a two-stage pipeline:

### 1. SFT warm-start
We first generated structured trajectories using:
- expert behavior
- heuristic behavior
- recovery-oriented traces
- diverse trajectories

This taught the model the action schema and basic operational priors.

### 2. GRPO fine-tuning
We then fine-tuned the model against the live environment reward, where it had to improve its actual behavior inside the benchmark.

For the 1.5B run, we trained on top of:

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- SFT-merged checkpoint: `sft_merged_1p5b_v5`
- GRPO adapter: `trained_model_1p5b_v5`

We are also providing:
- a **reproducible notebook** for judges and readers
- the **original logs and evaluation artifacts** from the strongest full runs

That distinction is important:
- the notebook is the clean rerunnable pipeline
- the headline metrics come from the original full training runs and their saved artifacts

## Training evidence

Our SFT warm-start used:
- `1368` training pairs
- `5` incident task families
- expert + heuristic + recovery + diverse data

Key SFT signals:
- final train loss: `0.3847`
- mean token accuracy: `0.969`
- JSON parse rate in validation: `100%`

That mattered because the environment depends on reliable structured actions. Before RL, the model had already learned to emit valid incident commands consistently.

During GRPO, training reward reached very strong regions of the policy space, including peaks near `0.99` reward on training episodes, while still surfacing harder pockets where the benchmark resisted collapse into a trivial strategy.

## Results

Our strongest 1.5B evaluation result was:

**25/25 episodes resolved in this evaluation set**  
**Average score: 0.9153**

That substantially outperformed the heuristic baseline.

Average score comparison:
- Heuristic baseline: `0.7127`
- Trained 1.5B model: `0.9153`

Per-task trained scores:
- `single_service_failure`: `0.9400`
- `cascading_failure`: `0.8740`
- `hidden_root_cause`: `0.9233`
- `chaos_cascade`: `0.9483`
- `multi_root_cause`: `0.8910`

This is the evidence we wanted most: not just that the model can act, but that training makes it measurably better on a benchmark with diagnostics, sequencing, and recovery tradeoffs.

## What the model learned

The biggest improvement was behavioral.

Compared with weaker baselines, the trained model became much better at:
- starting with diagnostics instead of immediate blind restarts
- inspecting the right upstream service
- choosing rollback vs restart vs scale-up based on context
- respecting dependency order during multi-service recovery

This mattered most in:
- hidden root-cause tasks
- cascading database-driven incidents
- multi-root-cause outages where naive policies break down

## What we learned building it

A few lessons stood out.

### 1. Realistic failure structure makes RL more meaningful
When the environment contains hidden causes, dependency chains, and misleading symptoms, the agent cannot brute-force its way to good behavior.

### 2. Reward design is crucial
A good reward function turned "busy-looking action sequences" into better diagnostics, better ordering, and better recovery.

### 3. Serving parity matters
One subtle challenge in LLM environments is making sure live inference sees the same quality of observation context as evaluation and training. Aligning that path was essential for getting behavior that matched the offline results.

### 4. The 1.5B jump was significant
Smaller models could learn parts of the task, but the 1.5B model was the first one that looked robust within this benchmark.

## Why this project matters

We think incident response is a compelling direction for LLM training because it combines:
- reasoning under uncertainty
- tool use
- structured action generation
- safety-sensitive decisions
- long-horizon recovery planning

An agent that improves here is not just getting better at a toy task. It is learning a workflow people actually care about.

Beyond major outages, we think the same framework could also support training on repetitive operational failures - the kinds of recurring SRE issues where consistent first response matters just as much as deep diagnosis.

That is what made this project exciting for us.

We were not trying to build another static benchmark. We were trying to build a trainable environment for one of the most stressful and operationally meaningful workflows in software: the worst five minutes in production.

## Try it yourself

We are publishing:
- the environment on Hugging Face Spaces: `[HF Space link]`
- the reproducible notebook: `[Colab / notebook link]`
- the codebase: `[GitHub repo link]`
- the original training logs and evaluation artifacts: `[artifact link]`

The environment is hosted on Hugging Face Spaces with a containerized deployment so the benchmark can be run consistently outside our local setup.

## Final takeaway

Our final result is an LLM benchmark where the agent has to diagnose, reason, sequence, and recover under pressure - and where training produces a clear gain.

**A trained 1.5B model reached a `0.9153` average score and resolved all `25/25` episodes in our evaluation set.**

That is the core of our submission.
