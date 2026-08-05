import os
import sys
import argparse
import time
from pathlib import Path
import torch
import numpy as np

# Ensure src/ is in the PYTHONPATH with high robustness
# 以高健壯性將 'src' 目錄加入 Python 路徑，確保能獨立執行且順利匯入
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from Thumbs_Robot.agent import PPOAgent
from g1_rl.g1_hand_env import G1HandEnv

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    設定隨機種子以確保可重複性。
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def render_policy(checkpoint_path: str):
    """
    Launch interactive 3D simulation visualization.
    Loads the trained continuous PPO controller and commands
    gestures (Thumbs Up, Open/Stop, Thumbs Down) sequentially with recording prompts.
    啟動互動式 3D 模擬視覺化。載入訓練好的連續 PPO 控制器，並順序演示三種手勢（Thumbs Up, Open/Stop, Thumbs Down），同時提供錄影提示。
    """
    print("========================================")
    print("Launching MuJoCo 3D Visualizer Renderer")
    print(f"Checkpoint: {checkpoint_path}")
    print("========================================")

    # 1. Path calibration and check
    chk_path = Path(checkpoint_path)
    if not chk_path.exists():
        # Try parent directory relative path for convenience
        alt_path = Path(__file__).resolve().parent.parent.parent / checkpoint_path
        if alt_path.exists():
            chk_path = alt_path

    # 2. Environment initialization in Human Render mode
    # 對齊訓練使用的場景 XML，確保模型 morphology 正確（三指形態）
    xml_path = os.path.join("assets", "g1_fixed_base", "scene_29dof_fixed_base.xml")
    if not os.path.exists(xml_path):
        xml_path = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"

    env = G1HandEnv(xml_path=xml_path, render_mode="human")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    # 3. Agent initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = PPOAgent(state_dim=state_dim, action_dim=action_dim, device=device)
    
    try:
        if not chk_path.exists():
            print(f"Warning: Checkpoint '{checkpoint_path}' not found! Rendering untrained network.")
        else:
            agent.load_checkpoint(str(chk_path))
            print("Successfully loaded model checkpoint.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        env.close()
        sys.exit(1)

    set_seed(666)

    # Define the target gestures to render sequentially
    # 依序展示三種標準手勢：Thumbs Up, Open/Stop, Thumbs Down
    benchmark_gestures = [0, 1, 2]
    total_gestures = len(benchmark_gestures)

    # 4. Interactive visualization loop
    try:
        for idx, gid in enumerate(benchmark_gestures, 1):
            gesture_name = env.gestures[gid]
            
            # Reset environment and specify gesture via option
            obs, info = env.reset(seed=1000 + gid, options={"gesture_id": gid})
            
            # Interactive prompts for recording / 互動式錄影引導提示
            if idx == 1:
                print("\n" + "=" * 60)
                print("MuJoCo GUI window is now open.")
                print("1. Please adjust the camera view in the simulator window.")
                print("2. Start your screen/video recording software when ready.")
                print("3. Press [Enter] in this terminal to start Episode 1 (Thumbs Up).")
                print("=" * 60)
                input("Press [Enter] to start visualization...")
            else:
                print("\n" + "=" * 80)
                print(f"Preparing Gesture {idx}/{total_gestures}: {gesture_name}")
                print("You can adjust the camera now if needed.")
                input(f"Press [Enter] to render Gesture {idx}...")
            
            print("\nStarting simulation loop...\n")
            print("-" * 85)
            print(f"{'Ep #':<6} | {'Target Gesture':<15} | {'Step':<6} | {'Pose Error':<12} | {'Orient Error':<14} | {'Reward'}")
            print("-" * 85)

            done = False
            step_count = 0
            episode_reward = 0.0

            while not done:
                # Deterministic greedy policy execution
                action, _, _ = agent.select_action(torch.tensor(obs), deterministic=True)
                
                # Step simulation
                action_np = action.numpy()[0] if len(action.shape) > 1 else action.numpy()
                next_obs, reward, terminated, truncated, step_info = env.step(action_np)
                done = terminated or truncated
                obs = next_obs
                episode_reward += reward
                step_count += 1
                
                pose_err = step_info.get("pose_error_norm", 0.0)
                orient_err = step_info.get("orientation_error", 0.0)
                
                # Real-time console print for steps / 即時列印步進狀態
                if step_count % 10 == 0 or done:
                    print(
                        f"{idx:<6d} | "
                        f"{gesture_name:<15} | "
                        f"{step_count:<6d} | "
                        f"{pose_err:<12.4f} | "
                        f"{orient_err:<14.4f} | "
                        f"{reward:+.4f}"
                    )
                
                # Interactive sleep to mimic simulation frame rates
                time.sleep(1.0 / env.metadata.get("render_fps", 30))

            print("-" * 85)
            print(f"Episode {idx} Finished! | Gesture: {gesture_name} | Total Steps: {step_count} | Success: {terminated}")
            print(f"Cumulative Reward: {episode_reward:.4f}")
            print("=" * 80 + "\n")
            
            time.sleep(1.0) # Break sleep before starting the next gesture

        print("All benchmark gestures rendered successfully!")
        
        # Keep window open until user closes it / 等待用戶手動關閉視窗後才結束
        # This checks if the passive viewer is active in Gymnasium environment
        if hasattr(env, "viewer") and env.viewer is not None:
            print("Close the MuJoCo viewer window to exit.")
            while env.viewer is not None and env.viewer.is_running():
                time.sleep(0.1)

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
