import sys
import os
from pathlib import Path
import numpy as np
import torch

# Add 'src' directory to Python path for importing modules
# 將 'src' 目錄加入 Python 路徑，以匯入相關模組
current_dir = Path(__file__).resolve().parent
src_dir = current_dir.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from Thumbs_Robot.actor_critic_network import ActorCriticNetwork
from Thumbs_Robot.rollout_buffer import RolloutBuffer
from Thumbs_Robot.agent import PPOAgent
from g1_rl.g1_hand_env import G1HandEnv

def run_smoke_test() -> None:
    print("==================================================")
    print("Starting PPO Module Smoke Test...")
    print("==================================================")

    # 1. Environment initialization / 初始化真實 Gym 環境
    print("\n[Step 1] Initializing G1HandEnv in headless mode...")
    xml_path = os.path.join("assets", "g1_fixed_base", "scene_29dof_fixed_base.xml")
    if not os.path.exists(xml_path):
        # Fallback to absolute path or other locations if needed, but relative to repo root is standard
        print(f"   Warning: Model XML not found at {xml_path}. Trying fallback...")
        xml_path = "assets/g1_fixed_base/scene_29dof_fixed_base.xml"
        
    try:
        env = G1HandEnv(xml_path=xml_path, render_mode=None)
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        print(f"   Environment loaded successfully.")
        print(f"   State space dimension: {state_dim}")
        print(f"   Action space dimension: {action_dim}")
    except Exception as e:
        print(f"   Failed to load G1HandEnv: {e}")
        sys.exit(1)

    # 2. Agent & Network Initialization / 初始化代理人與網路
    print("\n[Step 2] Initializing PPOAgent with test hyperparameters...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            lr=3e-4,
            ppo_epochs=2,
            batch_size=2,
            device=device
        )
        print(f"   PPOAgent initialized on device: {agent.device}")
    except Exception as e:
        print(f"   Failed to initialize PPOAgent: {e}")
        env.close()
        sys.exit(1)

    # 3. Rollout Buffer and Real Episode Step Collection / 真實步進與 Rollout 緩衝區收集
    rollout_length = 5
    print(f"\n[Step 3] Running episode to collect {rollout_length} steps of transitions...")
    buffer = RolloutBuffer(rollout_length, state_dim, action_dim, device=device)
    
    try:
        obs, info = env.reset(seed=42)
        obs_t = torch.tensor(obs)
        
        for i in range(rollout_length):
            # Select action using agent
            action, log_prob, value = agent.select_action(obs_t)
            
            # Step simulation
            action_np = action.numpy()[0] if len(action.shape) > 1 else action.numpy()
            next_obs, reward, terminated, truncated, step_info = env.step(action_np)
            done = terminated or truncated
            
            # Insert transition into rollout buffer
            buffer.insert(obs_t, action, reward, terminated, truncated, log_prob, value)
            
            print(f"   Step {i+1:02d} | Reward: {reward:.4f} | Done: {done} | Target: {step_info.get('target_gesture')}")
            
            obs = next_obs
            obs_t = torch.tensor(obs)
            if done:
                obs, info = env.reset()
                obs_t = torch.tensor(obs)
                
        print(f"   Rollout buffer filled. Current pointer: {buffer.ptr}")
        assert buffer.ptr == rollout_length, "Rollout buffer size mismatch!"
        print("   Real episode step collection passed.")
    except Exception as e:
        print(f"   Error during rollout collection: {e}")
        env.close()
        sys.exit(1)

    # 4. GAE Advantages and PPO Optimization / 優勢計算與網路最佳化
    print("\n[Step 4] Computing GAE advantages and executing PPO agent update...")
    try:
        # Predict value of the final next state for bootstrapping
        _, _, next_value = agent.select_action(obs_t)
        
        # GAE calculation
        buffer.compute_advantages(next_value, next_done=terminated, next_truncated=truncated, gamma=0.99, gae_lambda=0.95)
        print(f"   Calculated advantages shape: {buffer.advantages.shape}")
        assert buffer.advantages.shape[0] == rollout_length, "Advantages dimension mismatch!"
        
        # Optimize networks
        metrics = agent.update(buffer)
        print("   Optimization complete. Computed Metrics:")
        for k, v in metrics.items():
            print(f"     - {k}: {v:.6f}")
            assert not np.isnan(v), f"NaN encountered in metric '{k}'!"
            
        print("   GAE advantage calculation and PPO agent updates test passed.")
    except Exception as e:
        print(f"   Error during PPO optimization: {e}")
        env.close()
        sys.exit(1)

    # 5. Checkpoint saving and loading / 檢查點儲存與載入
    print("\n[Step 5] Testing model save and load checkpoint...")
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)
    test_checkpoint = checkpoint_dir / "test_smoke_ppo_checkpoint.pt"
    
    try:
        # Save model checkpoint
        agent.save_checkpoint(test_checkpoint)
        print(f"   Checkpoint saved to: {test_checkpoint}")
        assert test_checkpoint.exists(), "Checkpoint file was not created!"
        
        # Load model checkpoint into a new clean agent
        new_agent = PPOAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            device=device
        )
        new_agent.load_checkpoint(test_checkpoint)
        print("   Checkpoint loaded into a new PPO agent successfully.")
        
        # Clean up temporary test checkpoint file
        if test_checkpoint.exists():
            test_checkpoint.unlink()
            print("   Temporary checkpoint file deleted.")
        print("   Save/Load checkpoint test passed.")
    except Exception as e:
        print(f"   Error during checkpoint save/load test: {e}")
        if test_checkpoint.exists():
            test_checkpoint.unlink()
        env.close()
        sys.exit(1)

    # Close MuJoCo environment / 關閉環境
    env.close()
    
    print("\n==================================================")
    print("ALL PPO MODULE SMOKE TESTS PASSED SUCCESSFULLY!")
    print("==================================================")

if __name__ == "__main__":
    run_smoke_test()
