import os
import sys
import argparse
import json
import csv
import time
import random
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Ensure src/ is in the PYTHONPATH with high robustness
# 以高健壯性將 'src' 目錄加入 Python 路徑，確保能獨立執行且順利匯入
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from Thumbs_Robot.rollout_buffer import RolloutBuffer
from Thumbs_Robot.agent import PPOAgent
from g1_rl.g1_hand_env import G1HandEnv

def set_seed(seed: int) -> None:
    """
    Set all random seeds to guarantee 100% reproducibility.
    設定所有隨機種子以確保完全的可重複性。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Force PyTorch to use deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_training(args: argparse.Namespace):
    """
    Main training loop running continuous Actor-Critic PPO 
    control on the G1 3-digit hand environment.
    """
    # 1. Establish logs directory
    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save training configuration
    config_path = os.path.join(args.results_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=4)

    # 2. Seed configuration
    set_seed(args.seed)

    # 3. Environment & Agent setup
    xml_path = os.path.join("assets", "g1_fixed_base", "scene_29dof_fixed_base.xml")
    if not os.path.exists(xml_path):
        xml_path = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"

    env = G1HandEnv(xml_path=xml_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    
    print("==================================================")
    print("Starting PPO Training: Unitree G1 Hand-Gesture Control")
    print(f"Device: {device} | State Dim: {state_dim} | Action Dim: {action_dim}")
    print(f"Random Seed: {args.seed}")
    print(f"Hyperparameters:")
    print(f" - Learning Rate (lr): {args.lr}")
    print(f" - Discount Factor (gamma): {args.gamma}")
    print(f" - GAE Lambda: {args.gae_lambda}")
    print(f" - PPO Clip Epsilon: {args.clip_epsilon}")
    print(f" - Batch Size: {args.batch_size}")
    print(f" - Rollout Length: {args.rollout_length}")
    print(f" - Epochs per Rollout: {args.ppo_epochs}")
    print(f" - Max Total Steps: {args.max_total_steps if not args.smoke_test else 20}")
    print("==================================================")

    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        clip_epsilon=args.clip_epsilon,
        device=device
    )

    buffer = RolloutBuffer(args.rollout_length, state_dim, action_dim, device=device)

    # 4. Open CSV Log files
    step_log_path = os.path.join(args.results_dir, "step_log.csv")
    update_log_path = os.path.join(args.results_dir, "update_log.csv")
    episode_log_path = os.path.join(args.results_dir, "episode_log.csv")

    step_log_file = open(step_log_path, "w", newline="", encoding="utf-8")
    update_log_file = open(update_log_path, "w", newline="", encoding="utf-8")
    episode_log_file = open(episode_log_path, "w", newline="", encoding="utf-8")

    step_writer = csv.writer(step_log_file)
    update_writer = csv.writer(update_log_file)
    episode_writer = csv.writer(episode_log_file)

    # Write headers
    step_writer.writerow(["episode", "step", "gesture", "pose_error", "orientation_error", "reward", "value", "advantage"])
    update_writer.writerow(["update", "total_steps", "actor_loss", "critic_loss", "entropy", "total_loss", "mean_reward", "success_rate", "completed_episodes"])
    episode_writer.writerow(["Episode", "Gesture", "Reward", "Success", "Steps", "Final Pose Error", "Hold Duration", "Safety Violation"])

    # Flush headers
    step_log_file.flush()
    update_log_file.flush()
    episode_log_file.flush()

    # 5. Execute Training Loop
    start_time = time.time()
    obs, info = env.reset(seed=args.seed)
    obs_t = torch.tensor(obs)
    
    episode_count = 0
    total_steps = 0
    episode_reward = 0.0
    episode_length = 0
    
    success_history = []
    reward_history = []

    # Table Header for Console Prints (Update level progress)
    print(f"\n{'Update':<6} | {'Steps':<8} | {'Ep_Done':<7} | {'Mean_Rwd':<8} | {'Succ_%':<6} | {'Loss_A':<8} | {'Loss_C':<8} | {'Entropy':<7} | {'Time':<6} | {'Last_Gesture':<12}")
    print("-" * 102)

    # If smoke test, override maximum execution length
    max_steps = 20 if args.smoke_test else args.max_total_steps

    while total_steps < max_steps:
        # Collect rollouts
        buffer.clear()
        
        # Temp storage for completed episodes metrics inside this rollout
        rollout_completed_rewards = []
        rollout_completed_successes = []
        rollout_last_gesture = "N/A"
        
        for step in range(args.rollout_length):
            action, log_prob, value = agent.select_action(obs_t)
            
            # Step simulation
            action_np = action.numpy()[0] if len(action.shape) > 1 else action.numpy()
            next_obs, reward, terminated, truncated, step_info = env.step(action_np)
            done = terminated or truncated
            
            # Record step metrics
            buffer.insert(obs_t, action, reward, done, log_prob, value)
            
            # Write to step log
            step_writer.writerow([
                episode_count, 
                episode_length, 
                step_info.get("target_gesture", "N/A"), 
                step_info.get("pose_error_norm", 0.0), 
                step_info.get("orientation_error", 0.0), 
                reward, 
                value.item(), 
                0.0 # Will compute advantage after rollout
            ])

            obs = next_obs
            obs_t = torch.tensor(obs)
            episode_reward += reward
            episode_length += 1
            total_steps += 1

            if done:
                wall_clock_time = time.time() - start_time
                is_success = terminated # terminated signals success gesture hold
                
                success_history.append(int(is_success))
                reward_history.append(episode_reward)
                
                rollout_completed_rewards.append(episode_reward)
                rollout_completed_successes.append(int(is_success))
                rollout_last_gesture = step_info.get("target_gesture", "N/A")

                # Write to episode log silently (no print statements here to avoid cluttering)
                episode_writer.writerow([
                    episode_count, 
                    step_info.get("target_gesture", "N/A"), 
                    episode_reward, 
                    float(is_success), 
                    episode_length,
                    step_info.get("pose_error_norm", 0.0),
                    step_info.get("hold_duration", 0),
                    step_info.get("safety_violations", 0)
                ])
                
                # Flush to avoid data loss on crash
                episode_log_file.flush()
                step_log_file.flush()

                # Reset env with deterministic incremented seed
                episode_count += 1
                obs, info = env.reset(seed=args.seed + episode_count)
                obs_t = torch.tensor(obs)
                episode_reward = 0.0
                episode_length = 0

                if total_steps >= max_steps:
                    break

        # Rollout finished, calculate advantages and update agent
        if total_steps >= max_steps and done:
            next_value = torch.zeros(1, device=device) # Terminal end
        else:
            _, _, next_value = agent.select_action(obs_t)
            
        buffer.compute_advantages(next_value, next_done=done, gamma=args.gamma, gae_lambda=args.gae_lambda)

        # Optimize network weights
        metrics = agent.update(buffer)
        
        # Calculate rollout summary metrics
        completed_ep = len(rollout_completed_rewards)
        rollout_mean_reward = np.mean(rollout_completed_rewards) if completed_ep > 0 else 0.0
        rollout_success_rate = np.mean(rollout_completed_successes) * 100.0 if completed_ep > 0 else 0.0
        
        # Write to update log
        update_writer.writerow([
            total_steps // args.rollout_length, 
            total_steps,
            metrics["actor_loss"], 
            metrics["critic_loss"], 
            metrics["entropy"], 
            metrics["total_loss"],
            rollout_mean_reward,
            rollout_success_rate,
            completed_ep
        ])
        update_log_file.flush()

        # Print Update level progress as a clean table row
        update_cycle = total_steps // args.rollout_length
        wall_clock_time = time.time() - start_time
        
        mean_reward_str = f"{rollout_mean_reward:.2f}" if completed_ep > 0 else "N/A"
        success_rate_str = f"{rollout_success_rate:.1f}%" if completed_ep > 0 else "0.0%"
        
        if not args.smoke_test:
            # Print every 5 cycles in standard mode to keep terminal extremely clean
            if update_cycle % 5 == 0 or update_cycle == 1:
                print(f"{update_cycle:<6d} | {total_steps:<8d} | {completed_ep:<7d} | {mean_reward_str:<8} | {success_rate_str:<6} | {metrics['actor_loss']:<8.4f} | {metrics['critic_loss']:<8.4f} | {metrics['entropy']:<7.3f} | {wall_clock_time:<5.1f}s | {rollout_last_gesture:<12}")
        else:
            # Print every cycle in smoke-test mode to verify output format
            print(f"{update_cycle:<6d} | {total_steps:<8d} | {completed_ep:<7d} | {mean_reward_str:<8} | {success_rate_str:<6} | {metrics['actor_loss']:<8.4f} | {metrics['critic_loss']:<8.4f} | {metrics['entropy']:<7.3f} | {wall_clock_time:<5.1f}s | {rollout_last_gesture:<12}")

        # Save checkpoint periodically
        if (total_steps // args.rollout_length) % 10 == 0 and not args.smoke_test:
            ckpt_path = os.path.join("models", f"checkpoint_step_{total_steps}.pt")
            agent.save_checkpoint(ckpt_path)

    # Save best/final weights
    final_model_path = os.path.join("models", "best_thumbs_robot.pt")
    agent.save_checkpoint(final_model_path)
    
    # Training completion summary
    total_time = time.time() - start_time
    if len(success_history) > 0:
        final_rolling_success = np.mean(success_history[-50:]) * 100.0
        mean_reward_final_20 = np.mean(reward_history[-20:])
    else:
        final_rolling_success = 0.0
        mean_reward_final_20 = 0.0
    
    print("-" * 102)
    print(f"Training completed in {total_time:.1f} seconds. Saved final model to: {final_model_path}")
    print(f"Final rolling success rate (last 50 episodes): {final_rolling_success:.2f}%")
    print(f"Mean cumulative reward (last 20 episodes): {mean_reward_final_20:.4f}")
    print("==================================================")

    # Close file handles and env
    step_log_file.close()
    update_log_file.close()
    episode_log_file.close()
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless continuous PPO training runner [Phase 3]")
    parser.add_argument("--seed", type=int, default=666, help="Random seed for reproducibility")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda coefficient")
    parser.add_argument("--clip-epsilon", type=float, default=0.2, help="PPO clipping threshold")
    parser.add_argument("--ppo-epochs", type=int, default=10, help="Updates epochs per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="PPO mini-batch size")
    parser.add_argument("--rollout-length", type=int, default=2048, help="Rollout steps collected per update cycle")
    parser.add_argument("--max-total-steps", type=int, default=100000, help="Maximum total training environment steps")
    parser.add_argument("--results-dir", type=str, default="results/ppo_config_a", help="Directory path to save results")
    parser.add_argument("--cpu", action="store_true", help="Force training on CPU")
    parser.add_argument("--smoke-test", action="store_true", help="Execute rapid smoke test override")

    args = parser.parse_args()
    
    if args.smoke_test:
        args.rollout_length = 10
        args.batch_size = 2
        args.ppo_epochs = 2
        args.max_total_steps = 20

    run_training(args)
