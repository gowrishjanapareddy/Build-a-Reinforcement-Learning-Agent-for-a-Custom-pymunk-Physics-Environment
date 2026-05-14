"""
plot.py - Learning Curve Comparison Plot Generator.

Reads Monitor CSV logs from logs/baseline/ and logs/shaped/ and plots the
mean reward over timesteps for both reward functions side by side.

Usage:
    python plot.py
    # Saves reward_comparison.png in the project root.
"""

import glob
import os

import matplotlib.pyplot as plt
import pandas as pd


def load_monitor_data(log_dir: str) -> pd.DataFrame | None:
    """Load stable-baselines3 Monitor CSV file from a log directory."""
    file_path = os.path.join(log_dir, "monitor.csv")
    if not os.path.exists(file_path):
        print(f"[plot.py] WARNING: No monitor file found in '{log_dir}'. "
              "Run train.py first to generate logs.")
        return None

    # The first line is a JSON comment — skip it
    df = pd.read_csv(file_path, skiprows=1)
    df["cumulative_steps"] = df["l"].cumsum()
    return df


def main():
    baseline_df = load_monitor_data("logs/baseline")
    shaped_df = load_monitor_data("logs/shaped")

    if baseline_df is None and shaped_df is None:
        print("[plot.py] ERROR: No log data found. Train both reward types first.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if baseline_df is not None:
        smoothed = baseline_df["r"].rolling(window=50, min_periods=1).mean()
        ax.plot(baseline_df["cumulative_steps"], smoothed,
                label="Baseline Reward  [cos(θ₁) + cos(θ₂)]",
                color="#E74C3C", alpha=0.9, linewidth=1.5)
        ax.fill_between(baseline_df["cumulative_steps"],
                        baseline_df["r"].rolling(window=50, min_periods=1).min(),
                        baseline_df["r"].rolling(window=50, min_periods=1).max(),
                        color="#E74C3C", alpha=0.1)

    if shaped_df is not None:
        smoothed = shaped_df["r"].rolling(window=50, min_periods=1).mean()
        ax.plot(shaped_df["cumulative_steps"], smoothed,
                label="Shaped Reward  [+ center & velocity penalties]",
                color="#2ECC71", alpha=0.9, linewidth=1.5)
        ax.fill_between(shaped_df["cumulative_steps"],
                        shaped_df["r"].rolling(window=50, min_periods=1).min(),
                        shaped_df["r"].rolling(window=50, min_periods=1).max(),
                        color="#2ECC71", alpha=0.1)

    ax.set_title("Learning Curves: Baseline vs. Shaped Reward\n"
                 "Double Inverted Pendulum — PPO Agent", fontsize=14, fontweight="bold")
    ax.set_xlabel("Environment Steps (Timesteps)", fontsize=12)
    ax.set_ylabel("Mean Episode Reward (50-ep rolling avg)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()

    output_path = "reward_comparison.png"
    fig.savefig(output_path, dpi=150)
    print(f"[plot.py] Saved plot to {output_path}")


if __name__ == "__main__":
    main()
