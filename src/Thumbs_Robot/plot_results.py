"""
Plotting Script for Thumbs Robot PPO training and evaluation results.
This script reads the CSV logs (update_log.csv, episode_log.csv, and eval_results.csv)
and generates high-resolution comparative and summary plots.

CSCN8020 PPO 訓練與評估結果繪圖與對照腳本。
讀取 PPO 訓練日誌與評估結果 CSV，繪製高解析度訓練指標看板與基準手勢評估比較圖表。
"""

from __future__ import annotations
import os
import sys
import csv
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Apply clean and professional styling
# 嘗試載入現代化繪圖風格
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except:
    try:
        plt.style.use("ggplot")
    except:
        pass

def read_eval_results(csv_path: Path) -> dict[str, list] | None:
    """
    Read detailed evaluation CSV log and extract episode-level results.
    讀取詳細的評估 CSV 日誌並解析 Episode 級別的指標數據。
    """
    if not csv_path.exists():
        print(f"Warning: Evaluation CSV not found at {csv_path}")
        return None
        
    gestures = []
    successes = []
    steps = []
    rewards = []
    pose_errs = []
    orient_errs = []

    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            _header = next(reader) # skip headers
        except StopIteration:
            return None
            
        for row in reader:
            # Stop parsing when hitting empty divider or summary section
            if not row or row[0] == "" or row[0] == "Overall_Metric" or row[0] == "Gesture_Specific_Metric":
                break
            gestures.append(row[0])
            successes.append(int(row[3]))
            steps.append(int(row[4]))
            rewards.append(float(row[5]))
            pose_errs.append(float(row[6]))
            orient_errs.append(float(row[7]))
            
    return {
        "Gesture": gestures,
        "Success": np.array(successes),
        "Steps": np.array(steps),
        "Reward": np.array(rewards),
        "Pose_Error": np.array(pose_errs),
        "Orientation_Error": np.array(orient_errs)
    }

def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate moving average with padding protection.
    計算移動平均值，具備長度保護。
    """
    if len(data) == 0:
        return np.array([])
    actual_window = min(window, len(data))
    return np.convolve(data, np.ones(actual_window) / actual_window, mode="valid")

def plot_single_run_metrics(results_dir: Path) -> None:
    """
    Generate and save separate high-resolution training performance plots for a single configuration.
    為單一配置生成並儲存獨立的高解析度訓練效能圖表。
    """
    update_log_path = results_dir / "update_log.csv"
    episode_log_path = results_dir / "episode_log.csv"

    if not update_log_path.exists() or not episode_log_path.exists():
        print(f"Error: Required CSV logs missing in '{results_dir}'")
        return

    update_df = pd.read_csv(update_log_path)
    episode_df = pd.read_csv(episode_log_path)

    # Construct output image directory: e.g. results/img/ppo_config_a
    output_img_dir = results_dir.parent / "img" / results_dir.name
    os.makedirs(output_img_dir, exist_ok=True)

    # Colors for professional aesthetics
    c_blue = "#1f77b4"
    c_darkblue = "#0f3c5a"
    c_green = "#2ca02c"
    c_red = "#d62728"
    c_orange = "#ff7f0e"
    c_purple = "#9467bd"

    title_suffix = f" ({results_dir.name.upper()})"

    # 1. Plot 1: Episode Accumulated Returns (EMA smoothed)
    plt.figure(figsize=(8, 5))
    plt.plot(episode_df["Reward"], alpha=0.25, color=c_blue, label="Raw Return")
    plt.plot(episode_df["Reward"].ewm(span=30, adjust=False).mean(), color=c_darkblue, linewidth=2.5, label="EMA (span=30)")
    plt.title(f"Episode Accumulated Returns{title_suffix}", fontsize=12, fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Accumulated Return")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path1 = output_img_dir / "accumulated_returns.png"
    plt.savefig(output_path1, dpi=300)
    plt.close()
    print(f"Generated plot saved to: {output_path1}")

    # 2. Plot 2: Rolling Success Rate (window=20)
    plt.figure(figsize=(8, 5))
    rolling_success = episode_df["Success"].rolling(window=20, min_periods=1).mean() * 100.0
    plt.plot(rolling_success, color=c_green, linewidth=2.5, label="Rolling Success Rate")
    plt.title(f"Rolling Success Rate (Window=20){title_suffix}", fontsize=12, fontweight="bold")
    plt.xlabel("Episode")
    plt.ylabel("Success Rate (%)")
    plt.ylim(-5, 105)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path2 = output_img_dir / "success_rate.png"
    plt.savefig(output_path2, dpi=300)
    plt.close()
    print(f"Generated plot saved to: {output_path2}")

    # 3. Plot 3: Optimization Losses
    plt.figure(figsize=(8, 5))
    plt.plot(update_df["actor_loss"], color=c_red, alpha=0.8, label="Actor Loss")
    plt.plot(update_df["critic_loss"], color=c_orange, alpha=0.8, label="Critic Loss")
    plt.title(f"Actor & Critic Optimization Losses{title_suffix}", fontsize=12, fontweight="bold")
    plt.xlabel("Update Cycles")
    plt.ylabel("Loss Value (Log Scale)")
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path3 = output_img_dir / "optimization_losses.png"
    plt.savefig(output_path3, dpi=300)
    plt.close()
    print(f"Generated plot saved to: {output_path3}")

    # 4. Plot 4: Policy Entropy
    plt.figure(figsize=(8, 5))
    plt.plot(update_df["entropy"], color=c_purple, linewidth=2, label="Policy Entropy")
    plt.title(f"Policy Entropy (Exploration Schedule){title_suffix}", fontsize=12, fontweight="bold")
    plt.xlabel("Update Cycles")
    plt.ylabel("Entropy Value")
    plt.legend(loc="upper right")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    output_path4 = output_img_dir / "policy_entropy.png"
    plt.savefig(output_path4, dpi=300)
    plt.close()
    print(f"Generated plot saved to: {output_path4}")

def plot_eval_performance(eval_csv: Path, output_dir: Path) -> None:
    """
    Generate a bar chart visualization of PPO deterministic evaluation results per target gesture.
    產出 PPO 確定性評估結果的柱狀圖（按手勢類別區分）。
    """
    data = read_eval_results(eval_csv)
    if data is None:
        return

    # Group and compute metrics per gesture
    df = pd.DataFrame(data)
    grouped = df.groupby("Gesture").agg({
        "Success": "mean",
        "Steps": "mean",
        "Pose_Error": "mean",
        "Orientation_Error": "mean"
    })
    grouped["Success"] *= 100.0 # Convert to percentage

    gestures = grouped.index.tolist()
    x = np.arange(len(gestures))
    width = 0.35

    # 1. Plot Success Rate & Pose Error breakdown
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Custom professional colors
    color_succ = "#2ca02c" # Green
    color_err = "#d62728" # Red
    
    rects1 = ax1.bar(x - width/2, grouped["Success"], width, label="Success Rate (%)", color=color_succ, alpha=0.85)
    ax1.set_ylabel("Evaluation Success Rate (%)", color=color_succ, fontsize=12, fontweight="bold")
    ax1.tick_params(axis="y", labelcolor=color_succ)
    ax1.set_ylim(0, 115)
    
    ax2 = ax1.twinx()
    rects2 = ax2.bar(x + width/2, grouped["Pose_Error"], width, label="Pose Error (rad)", color=color_err, alpha=0.85)
    ax2.set_ylabel("Average Pose Error (rad)", color=color_err, fontsize=12, fontweight="bold")
    ax2.tick_params(axis="y", labelcolor=color_err)
    
    # Add values on top of success bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
        
    # Add values on top of error bars
    for rect in rects2:
        height = rect.get_height()
        ax2.annotate(
            f"{height:.4f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )

    plt.title("PPO Evaluation Performance Breakdown by Gesture", fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gestures, fontsize=11, fontweight="bold")
    ax1.set_xlabel("Target Gestures", fontsize=12)
    ax1.grid(True, axis="y", linestyle="--", alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / "eval_gesture_performance.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated evaluation performance plot saved to: {output_path}")

def plot_comparative_metrics(compare_dirs: list[str], output_dir: Path) -> None:
    """
    Generate comparative curves across multiple PPO training runs.
    在多個 PPO 訓練運行目錄之間生成比較曲線圖表。
    """
    plt.figure(figsize=(11, 6.5))
    
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    
    has_data = False
    for idx, d_str in enumerate(compare_dirs):
        d_path = Path(d_str)
        ep_log = d_path / "episode_log.csv"
        
        if ep_log.exists():
            df = pd.read_csv(ep_log)
            # Plot moving average of successes
            rolling_success = df["Success"].rolling(window=30, min_periods=1).mean() * 100.0
            plt.plot(rolling_success, color=colors[idx % len(colors)], linewidth=2.5, label=f"{d_path.name.upper()}")
            has_data = True
            
    if not has_data:
        plt.close()
        return

    plt.title("PPO Training Success Rate Comparison (Ablations)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Success Rate (%)", fontsize=12)
    plt.ylim(-5, 105)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower right", frameon=True)
    
    plt.tight_layout()
    output_path = output_dir / "ppo_comparison_success.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Generated comparative success plot saved to: {output_path}")

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO Performance Results Visualizer [Phase 6]")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="results/ppo_config_a", 
        help="Directory path containing PPO training CSVs"
    )
    parser.add_argument(
        "--eval_csv",
        type=str,
        default="results/ppo_evaluation/eval_results.csv",
        help="Path to the PPO evaluation CSV file"
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        help="List of results directories to generate ablation comparisons (e.g. results/ppo_config_a results/ppo_config_b)"
    )
    args = parser.parse_args()

    results_dir = Path(args.dir)
    eval_csv = Path(args.eval_csv)
    
    # 1. Plot Single Dashboard / 繪製單一結果看板
    if results_dir.exists():
        plot_single_run_metrics(results_dir)
        
    # 2. Plot Eval breakdown / 繪製評估分析柱狀圖
    if eval_csv.exists():
        plot_eval_performance(eval_csv, eval_csv.parent)
        
    # 3. Plot Comparison Dashboard / 繪製多組實驗比較圖
    if args.compare:
        compare_out = Path("results/plots")
        compare_out.mkdir(parents=True, exist_ok=True)
        plot_comparative_metrics(args.compare, compare_out)

if __name__ == "__main__":
    main()
