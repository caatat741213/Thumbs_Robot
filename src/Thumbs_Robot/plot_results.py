import os
import sys
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def plot_training_results(results_dir: str):
    """
    Read training CSV logs (update_log.csv and episode_log.csv) and generate
    high-resolution plots showcasing PPO optimization metrics.
    """
    update_log_path = os.path.join(results_dir, "update_log.csv")
    episode_log_path = os.path.join(results_dir, "episode_log.csv")

    if not os.path.exists(update_log_path) or not os.path.exists(episode_log_path):
        print(f"Error: Required CSV logs not found in directory '{results_dir}'")
        return

    print("========================================")
    print(f"Plotting Training Results from: {results_dir}")
    print("========================================")

    # 1. Load Data
    update_df = pd.read_csv(update_log_path)
    episode_df = pd.read_csv(episode_log_path)

    # 2. Setup matplotlib figure layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Unitree G1 Hand Continuous PPO Training Performance", fontsize=16)

    # Subplot 1: Episode Returns (Rollout Returns)
    # Applying exponential moving average (EMA) for smoother rendering
    axs[0, 0].plot(episode_df["total_return"], alpha=0.3, color="blue", label="Raw Returns")
    axs[0, 0].plot(episode_df["total_return"].ewm(span=20).mean(), color="darkblue", linewidth=2, label="EMA Smoothed")
    axs[0, 0].set_title("Episode Accumulated Returns")
    axs[0, 0].set_xlabel("Episodes")
    axs[0, 0].set_ylabel("Accumulated Return")
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Success Rate (Rolling mean)
    rolling_success = episode_df["success"].rolling(window=20, min_periods=1).mean() * 100.0
    axs[0, 1].plot(rolling_success, color="green", linewidth=2)
    axs[0, 1].set_title("Rolling Success Rate (Window=20)")
    axs[0, 1].set_xlabel("Episodes")
    axs[0, 1].set_ylabel("Success Rate (%)")
    axs[0, 1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Actor and Critic Losses
    axs[1, 0].plot(update_df["actor_loss"], color="red", label="Actor Loss")
    axs[1, 0].plot(update_df["critic_loss"], color="orange", label="Critic Loss")
    axs[1, 0].set_title("Optimization Losses")
    axs[1, 0].set_xlabel("Update Epochs")
    axs[1, 0].set_ylabel("Loss Value")
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 4: Policy Entropy
    axs[1, 1].plot(update_df["entropy"], color="purple", label="Entropy")
    axs[1, 1].set_title("Policy Entropy (Exploration Decay)")
    axs[1, 1].set_xlabel("Update Epochs")
    axs[1, 1].set_ylabel("Entropy Value")
    axs[1, 1].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    
    # Save high-resolution plot
    output_path = os.path.join(results_dir, "training_metrics.png")
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Generated metrics plot saved to: {output_path}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Results Plotter [Phase 6]")
    parser.add_argument(
        "--dir", 
        type=str, 
        default="results/ppo_config_a", 
        help="Directory path containing update_log.csv and episode_log.csv"
    )
    args = parser.parse_args()

    plot_training_results(results_dir=args.dir)
