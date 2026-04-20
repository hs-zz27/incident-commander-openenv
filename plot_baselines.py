"""
Plot baseline reward curves for Incident Commander Environment.

Reads results/baseline_rewards.json and generates a publication-quality
reward comparison chart for the HuggingFace blog and pitch slides.

Usage:
  python plot_baselines.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    print("ERROR: matplotlib and numpy required. Install with: pip install matplotlib numpy", file=sys.stderr)
    sys.exit(1)


def main():
    results_dir = Path(__file__).parent / "results"
    input_path = results_dir / "baseline_rewards.json"

    if not input_path.exists():
        print(f"ERROR: {input_path} not found. Run `python run_baselines.py` first.", file=sys.stderr)
        sys.exit(1)

    with open(input_path) as f:
        data = json.load(f)

    # Color palette
    colors = {
        "random": "#ef4444",      # Red
        "heuristic": "#f59e0b",   # Amber
        "llm": "#3b82f6",         # Blue
        "trained": "#22c55e",     # Green
    }

    agent_labels = {
        "random": "Random Agent",
        "heuristic": "Heuristic Agent",
        "llm": "LLM Agent (GPT-4o-mini)",
        "trained": "Trained Agent (GRPO)",
    }

    # Collect all tasks
    all_tasks = set()
    for agent_data in data.values():
        all_tasks.update(agent_data.keys())
    tasks = sorted(all_tasks)
    task_short = [t.replace("_", "\n") if len(t) > 15 else t for t in tasks]

    # --- Figure 1: Bar chart comparison across tasks ---
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    agents = list(data.keys())
    n_agents = len(agents)
    n_tasks = len(tasks)
    bar_width = 0.18
    x = np.arange(n_tasks)

    for i, agent_name in enumerate(agents):
        agent_data = data[agent_name]
        means = []
        stds = []
        for task in tasks:
            scores = agent_data.get(task, [])
            means.append(np.mean(scores) if scores else 0)
            stds.append(np.std(scores) if scores else 0)

        offset = (i - n_agents / 2 + 0.5) * bar_width
        bars = ax.bar(
            x + offset,
            means,
            bar_width,
            yerr=stds,
            capsize=3,
            color=colors.get(agent_name, "#64748b"),
            label=agent_labels.get(agent_name, agent_name),
            alpha=0.9,
            edgecolor="none",
            error_kw={"ecolor": "#94a3b8", "linewidth": 1},
        )

    ax.set_xlabel("Task", fontsize=13, color="#e2e8f0", fontweight="bold")
    ax.set_ylabel("Average Score", fontsize=13, color="#e2e8f0", fontweight="bold")
    ax.set_title(
        "Incident Commander — Baseline Agent Performance",
        fontsize=16, color="#f8fafc", fontweight="bold", pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(task_short, fontsize=10, color="#cbd5e1")
    ax.tick_params(colors="#94a3b8")
    ax.set_ylim(0, 1.1)

    # Grid
    ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="#475569")
    ax.set_axisbelow(True)

    # Legend
    legend = ax.legend(
        loc="upper right",
        fontsize=11,
        facecolor="#334155",
        edgecolor="#475569",
        labelcolor="#e2e8f0",
    )

    # Score annotations
    for spine in ax.spines.values():
        spine.set_color("#334155")

    plt.tight_layout()
    bar_path = results_dir / "baseline_comparison.png"
    plt.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    print(f"✅ Bar chart saved to {bar_path}")
    plt.close()

    # --- Figure 2: Overall average per agent (reward curve style) ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig2.patch.set_facecolor("#0f172a")
    ax2.set_facecolor("#1e293b")

    agent_avgs = {}
    for agent_name in agents:
        agent_data = data[agent_name]
        all_scores = []
        for task_scores in agent_data.values():
            all_scores.extend(task_scores)
        agent_avgs[agent_name] = np.mean(all_scores) if all_scores else 0

    sorted_agents = sorted(agent_avgs.items(), key=lambda x: x[1])
    names = [agent_labels.get(a, a) for a, _ in sorted_agents]
    values = [v for _, v in sorted_agents]
    bar_colors = [colors.get(a, "#64748b") for a, _ in sorted_agents]

    bars = ax2.barh(names, values, color=bar_colors, alpha=0.9, edgecolor="none", height=0.5)

    # Value labels
    for bar, val in zip(bars, values):
        ax2.text(
            bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center", fontsize=12, color="#e2e8f0", fontweight="bold",
        )

    ax2.set_xlabel("Average Score Across All Tasks", fontsize=13, color="#e2e8f0", fontweight="bold")
    ax2.set_title(
        "Agent Performance Summary",
        fontsize=16, color="#f8fafc", fontweight="bold", pad=20,
    )
    ax2.set_xlim(0, 1.1)
    ax2.tick_params(colors="#94a3b8")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.3, color="#475569")
    ax2.set_axisbelow(True)

    for spine in ax2.spines.values():
        spine.set_color("#334155")
    for label in ax2.get_yticklabels():
        label.set_color("#e2e8f0")
        label.set_fontsize(11)

    plt.tight_layout()
    summary_path = results_dir / "agent_summary.png"
    plt.savefig(summary_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    print(f"✅ Summary chart saved to {summary_path}")
    plt.close()

    print(f"\nDone! Charts saved to {results_dir}/")


if __name__ == "__main__":
    main()
