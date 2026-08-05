import os
import sys
import argparse
import torch
import time

# Ensure src/ is in the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Thumbs_Robot.agent import PPOAgent
from g1_rl.g1_hand_env import G1HandEnv

def render_policy(checkpoint_path: str):
    """
    Launch interactive 3D simulation visualization.
    Loads the trained continuous PPO controller and commands
    gestures (Thumbs Up, Open/Stop, Thumbs Down) live.
    """
    print("========================================")
    print("Launching MuJoCo 3D Visualizer Renderer")
    print(f"Checkpoint: {checkpoint_path}")
    print("========================================")

    # 1. Environment initialization
    xml_path = os.path.join("assets", "g1_hand.xml")
    env = G1HandEnv(xml_path=xml_path, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 2. Agent initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    if not os.path.exists(checkpoint_path):
        print(f"Warning: Checkpoint '{checkpoint_path}' not found! Rendering untrained network.")
    else:
        agent.load_checkpoint(checkpoint_path)
        print("Successfully loaded model checkpoint.")

    # 3. Interactive visualization loop
    try:
        while True:
            obs, info = env.reset()
            done = False
            step_count = 0

            print(f"\n--- Episode Start | Target Gesture: {info.get('target_gesture', 'N/A')} ---")

            while not done:
                # Deterministic greedy policy execution
                action, _, _ = agent.select_action(torch.tensor(obs), deterministic=True)
                
                next_obs, reward, terminated, truncated, step_info = env.step(action.numpy()[0] if len(action.shape) > 1 else action.numpy())
                done = terminated or truncated
                obs = next_obs
                
                # Interactive sleep to mimic simulation frame rates
                time.sleep(1.0 / env.metadata.get("render_fps", 30))
                
                step_count += 1
                if step_count % 10 == 0:
                    print(f"Step: {step_count:03d} | Pose Error: {step_info.get('pose_error_norm', 0.0):.4f} | Reward: {reward:.4f}")

            print(f"Episode Finished! Total steps taken: {step_count} | Success: {terminated}")
            time.sleep(1.0) # Break sleep before starting the next episode

    except KeyboardInterrupt:
        print("\nVisualizer closed by user.")
    finally:
        env.close()
        print("Renderer shutdown.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPO Continuous Controller MuJoCo 3D Visualizer [Phase 6]")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="models/best_thumbs_robot.pt", 
        help="Path to trained PPO model checkpoint weights"
    )
    args = parser.parse_args()

    render_policy(checkpoint_path=args.checkpoint)
