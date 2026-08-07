import os
import sys
import argparse
import numpy as np

# Ensure src/ is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from g1_rl.g1_hand_env import G1HandEnv

def evaluate_baseline(mode: str, num_episodes: int = 20):
    """
    Run baseline controllers (Random Action or Scripted Heuristics) 
    over evaluation episodes to construct a control benchmark against PPO.
    """
    print("========================================")
    print(f"Evaluating G1 Hand-Gesture Baseline Benchmark")
    print(f"Mode: {mode}")
    print(f"Episodes: {num_episodes}")
    print("========================================")

    # 1. Environment initialization
    xml_path = os.path.join("assets", "g1_hand.xml")
    env = G1HandEnv(xml_path=xml_path)
    action_dim = env.action_space.shape[0]

    # 2. Evaluation loop
    success_count = 0
    total_steps = 0
    final_errors = []
    final_orientation_errors = []

    for ep in range(num_episodes):
        obs, info = env.reset(seed=2000 + ep) # Standard baseline evaluation seeds
        done = False
        ep_steps = 0
        ep_success = False

        while not done:
            if mode == "random":
                # Random noise selection
                action = env.action_space.sample()
            elif mode == "heuristic":
                # Static constant increment targeting joints
                # (Simple proportional rule based on observation error placeholder)
                # In real code: action = Kp * position_error
                action = np.zeros(action_dim, dtype=np.float32)
                # Assume observation indices 12-16 contain the pose errors
                if len(obs) >= 17:
                    action = np.clip(obs[12:17] * 0.5, -1.0, 1.0)
            else:
                action = np.zeros(action_dim, dtype=np.float32)

            next_obs, reward, terminated, truncated, step_info = env.step(action)
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

    print("\n--- Baseline Benchmark Metrics ---")
    print(f"Overall Success Rate: {success_rate:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Final Pose Error (Rad): {avg_pose_error:.4f}")
    print(f"Average Final Orientation Error (Deg): {avg_orient_error:.4f}")
    print("========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Baseline Controller Evaluator [Phase 6]")
    parser.add_argument(
        "--mode", 
        type=str, 
        choices=["random", "heuristic", "zero"], 
        default="random", 
        help="Type of baseline controller to evaluate"
    )
    parser.add_argument(
        "--episodes", 
        type=int, 
        default=20, 
        help="Number of evaluation episodes"
    )
    args = parser.parse_args()

    evaluate_baseline(mode=args.mode, num_episodes=args.episodes)
