import os
import sys
import argparse
import torch
import numpy as np

# Ensure src/ is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Thumbs_Robot.agent import PPOAgent
from g1_rl.g1_hand_env import G1HandEnv

def evaluate_policy(checkpoint_path: str, num_episodes: int = 20):
    """
    Load a trained checkpoint and evaluate the deterministic greedy policy
    (exploration turned off) over a fixed number of testing episodes.
    """
    print("========================================")
    print("Evaluating G1 Hand-Gesture PPO Policy")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluation episodes: {num_episodes}")
    print("========================================")

    # 1. Environment initialization
    xml_path = os.path.join("assets", "g1_hand.xml")
    env = G1HandEnv(xml_path=xml_path)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Agent initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Running evaluation on untrained network.")
    else:
        agent.load_checkpoint(checkpoint_path)
        print("Successfully loaded model checkpoint.")

    # 3. Evaluation loop
    success_count = 0
    total_steps = 0
    final_errors = []
    final_orientation_errors = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=1000 + ep) # Fixed evaluation seeds for consistency
        done = False
        ep_steps = 0
        ep_success = False

        while not done:
            # Deterministic greedy action selection (uses mean of distribution)
            action, _, _ = agent.select_action(torch.tensor(obs), deterministic=True)
            
            next_obs, reward, terminated, truncated, step_info = env.step(action.numpy()[0] if len(action.shape) > 1 else action.numpy())
            done = terminated or truncated
            obs = next_obs
            ep_steps += 1

            if terminated:
                ep_success = True

        success_count += int(ep_success)
        total_steps += ep_steps
        
        pose_err = step_info.get("pose_error_norm", 0.0)
        orient_err = step_info.get("orientation_error", 0.0)
        
        final_errors.append(pose_err)
        final_orientation_errors.append(orient_err)

        print(f"Episode {ep+1:02d} | Gesture: {step_info.get('target_gesture', 'N/A'):<11} | Steps: {ep_steps:<3} | Success: {ep_success} | Pose Error: {pose_err:.4f}")

    success_rate = (success_count / num_episodes) * 100.0
    avg_steps = total_steps / num_episodes
    avg_pose_error = np.mean(final_errors)
    avg_orient_error = np.mean(final_orientation_errors)

    print("\n--- Final Evaluation Metrics ---")
    print(f"Overall Success Rate: {success_rate:.2f}% (Goal: >= 80%)")
    print(f"Average Steps to Completion: {avg_steps:.2f}")
    print(f"Average Final Pose Error (Rad): {avg_pose_error:.4f}")
    print(f"Average Final Orientation Error (Deg): {avg_orient_error:.4f}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic PPO Policy Evaluator [Phase 6]")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="models/best_thumbs_robot.pt", 
        help="Path to trained PPO model checkpoint weights"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=20, 
        help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    evaluate_policy(checkpoint_path=args.checkpoint, num_episodes=args.episodes)
