"""
PPO Policy Evaluator
Stage: Phase 6 (Evaluation Dashboard)
Primary Function:
- Loads the PPOAgent model weights checkpoint.
- Runs deterministic policy runs (zero exploration variance, dist.mean actions).
- Collects stats across benchmark target gestures (success rate, steps, pose & orient errors).
- Serializes detailed run metrics to evaluation CSV logs.
"""

import os
import sys
import argparse
import csv
import time
from pathlib import Path
import numpy as np
import torch

# Ensure src/ is in the PYTHONPATH with high robustness
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from Thumbs_Robot.agent import PPOAgent
from g1_rl.g1_hand_env import G1HandEnv

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    設定隨機種子以確保實驗的可重複性。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for policy evaluation.
    解析策略評估的命令列參數。
    """
    parser = argparse.ArgumentParser(description="Deterministic PPO Policy Evaluator [Phase 6]")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="models/best_thumbs_robot.pt", 
        help="Path to trained PPO model checkpoint weights"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/ppo_evaluation",
        help="Directory to save evaluation results CSV"
    )
    parser.add_argument(
        "--episodes_per_gesture", 
        type=int, 
        default=10, 
        help="Number of evaluation episodes per target gesture"
    )
    return parser.parse_args()

def evaluate_policy() -> None:
    """
    Load a trained continuous PPO controller, run deterministic greedy evaluation
    across benchmark gestures, print metrics, and save detailed results to CSV.
    """
    args = parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Establish base seed for reproducibility
    base_seed = 1000
    set_seed(base_seed)

    print("========================================")
    print("Evaluating G1 Hand-Gesture PPO Policy")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Episodes per gesture: {args.episodes_per_gesture} (Total: {args.episodes_per_gesture * 3})")
    print(f"Output directory: {output_dir}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("========================================")

    # 1. Environment initialization
    xml_path = os.path.join("assets", "g1_fixed_base", "scene_29dof_fixed_base.xml")
    if not os.path.exists(xml_path):
        xml_path = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"
        
    env = G1HandEnv(xml_path=xml_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Agent initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    # Robust checkpoint loading
    try:
        if not checkpoint_path.exists():
            print(f"Warning: Checkpoint '{checkpoint_path}' not found! Running evaluation on untrained network.")
        else:
            agent.load_checkpoint(str(checkpoint_path))
            print("Successfully loaded model checkpoint.\n")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        env.close()
        sys.exit(1)

    # Define the 3 target gestures to evaluate (THUMBS_UP, OPEN_STOP, THUMBS_DOWN)
    benchmark_gestures = [0, 1, 2] # correspond to IDs in G1HandEnv
    
    # Metrics tracking lists
    episode_results = []
    gesture_metrics = {gid: {"successes": 0, "steps": [], "pose_errs": [], "orient_errs": []} for gid in benchmark_gestures}
    
    print(f"{'Gesture':<12} | {'Episode':<8} | {'Seed':<6} | {'Success':<8} | {'Steps':<6} | {'Pose Error':<10} | {'Orient Error':<12}")
    print("-" * 80)

    # 3. Evaluation loop
    for gid in benchmark_gestures:
        gesture_name = env.gestures[gid]
        
        for ep in range(args.episodes_per_gesture):
            episode_seed = base_seed + gid * 100 + ep
            set_seed(episode_seed)
            
            # Reset environment with specific target gesture ID
            obs, info = env.reset(seed=episode_seed, options={"gesture_id": gid})
            
            done = False
            ep_steps = 0
            ep_success = False
            cumulative_reward = 0.0

            while not done:
                # Deterministic greedy action selection (uses mean of distribution, variance=0)
                action, _, _ = agent.select_action(torch.tensor(obs), deterministic=True)
                
                # Step the simulation forward
                action_np = action.numpy()[0] if len(action.shape) > 1 else action.numpy()
                next_obs, reward, terminated, truncated, step_info = env.step(action_np)
                done = terminated or truncated
                obs = next_obs
                ep_steps += 1
                cumulative_reward += reward

                # terminated indicates consecutive successful gesture hold streak achieved
                if terminated:
                    ep_success = True

            pose_err = step_info.get("pose_error_norm", 0.0)
            orient_err = step_info.get("orientation_error", 0.0)

            # Record metrics
            if ep_success:
                gesture_metrics[gid]["successes"] += 1
            gesture_metrics[gid]["steps"].append(ep_steps)
            gesture_metrics[gid]["pose_errs"].append(pose_err)
            gesture_metrics[gid]["orient_errs"].append(orient_err)

            # Print status to console
            print(f"{gesture_name:<12} | {ep+1:<8d} | {episode_seed:<6d} | {str(ep_success):<8} | {ep_steps:<6d} | {pose_err:<10.4f} | {orient_err:<12.4f}")

            episode_results.append({
                "Gesture": gesture_name,
                "Episode": ep + 1,
                "Seed": episode_seed,
                "Success": ep_success,
                "Steps": ep_steps,
                "Reward": cumulative_reward,
                "Pose_Error": pose_err,
                "Orientation_Error": orient_err
            })

    # Close environmental resources
    env.close()

    # 4. Compute overall and gesture-specific statistics
    total_episodes = len(benchmark_gestures) * args.episodes_per_gesture
    total_successes = sum(gesture_metrics[gid]["successes"] for gid in benchmark_gestures)
    overall_success_rate = (total_successes / total_episodes) * 100.0
    
    avg_steps_all = np.mean([s for gid in benchmark_gestures for s in gesture_metrics[gid]["steps"]])
    avg_pose_err_all = np.mean([e for gid in benchmark_gestures for e in gesture_metrics[gid]["pose_errs"]])
    avg_orient_err_all = np.mean([o for gid in benchmark_gestures for o in gesture_metrics[gid]["orient_errs"]])

    print("-" * 80)
    print("Evaluation Summary (PPO Deterministic Policy):")
    print(f"Total Successes: {total_successes} / {total_episodes}")
    print(f"Overall Success Rate: {overall_success_rate:.2f}% (Goal: >= 80.00%)")
    print(f"Average Steps to Completion: {avg_steps_all:.2f}")
    print(f"Average Final Pose Error: {avg_pose_err_all:.4f} rad")
    print(f"Average Final Orientation Error: {avg_orient_err_all:.4f} deg")
    print("-" * 80)
    print("Gesture-Specific Performance Breakdown:")
    for gid in benchmark_gestures:
        g_name = env.gestures[gid]
        g_success_rate = (gesture_metrics[gid]["successes"] / args.episodes_per_gesture) * 100.0
        g_avg_steps = np.mean(gesture_metrics[gid]["steps"])
        g_avg_pose = np.mean(gesture_metrics[gid]["pose_errs"])
        print(f" - {g_name:<11}: Success: {g_success_rate:.1f}% | Avg Steps: {g_avg_steps:.1f} | Avg Pose Error: {g_avg_pose:.4f}")
    print("========================================")

    # 5. Save detailed data to CSV file
    csv_path = output_dir / "eval_results.csv"
    print(f"Saving evaluation results to {csv_path}...")
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write detailed run headers
            writer.writerow([
                "Gesture",
                "Episode_Index",
                "Seed",
                "Success",
                "Steps",
                "Cumulative_Reward",
                "Final_Pose_Error",
                "Final_Orientation_Error"
            ])
            # Write data rows
            for res in episode_results:
                writer.writerow([
                    res["Gesture"],
                    res["Episode"],
                    res["Seed"],
                    int(res["Success"]),
                    res["Steps"],
                    f"{res['Reward']:.4f}",
                    f"{res['Pose_Error']:.6f}",
                    f"{res['Orientation_Error']:.6f}"
                ])
                
            # Write empty line and overall summary
            writer.writerow([])
            writer.writerow(["Overall_Metric", "Value"])
            writer.writerow(["Overall_Successes", f"{total_successes}/{total_episodes}"])
            writer.writerow(["Overall_Success_Rate_Percent", f"{overall_success_rate:.2f}"])
            writer.writerow(["Average_Steps", f"{avg_steps_all:.2f}"])
            writer.writerow(["Average_Pose_Error_rad", f"{avg_pose_err_all:.6f}"])
            writer.writerow(["Average_Orientation_Error_deg", f"{avg_orient_err_all:.6f}"])
            
            # Write gesture-specific breakdowns
            writer.writerow([])
            writer.writerow(["Gesture_Specific_Metric", "Success_Rate_Percent", "Average_Steps", "Average_Pose_Error_rad"])
            for gid in benchmark_gestures:
                g_name = env.gestures[gid]
                g_success_rate = (gesture_metrics[gid]["successes"] / args.episodes_per_gesture) * 100.0
                g_avg_steps = np.mean(gesture_metrics[gid]["steps"])
                g_avg_pose = np.mean(gesture_metrics[gid]["pose_errs"])
                writer.writerow([
                    g_name,
                    f"{g_success_rate:.2f}",
                    f"{g_avg_steps:.2f}",
                    f"{g_avg_pose:.6f}"
                ])
            
        print("Results successfully saved!")
    except Exception as e:
        print(f"Failed to save CSV file: {e}")
    print("========================================")

if __name__ == "__main__":
    evaluate_policy()
