#!/usr/bin/env python3
"""
G1 Initial Reset Posture Previewer & Calibrator
Allows developers to visualize and interactively adjust joint angles.
When sliders are dragged in the MuJoCo GUI, the script prints the updated
`pres_joints` dictionary to the terminal for easy copy-paste back to the code.
"""

import sys
from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

# Ensure src/ is in python path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.append(str(src_dir))

from g1_rl.g1_hand_env import G1HandEnv

def preview_reset_pose() -> None:
    print("==================================================")
    print("Loading G1HandEnv for reset posture preview...")
    print("==================================================")
    
    # Initialize the environment in human render mode
    env = G1HandEnv(render_mode="human")
    
    # Reset to apply the initial joint angles (pres_joints)
    env.reset(seed=42)
    
    print("\nEnvironment reset complete. Launching interactive MuJoCo viewer...")
    print("💡 HOW TO ADJUST JOINTS INTERACTIVELY:")
    print("1. In the MuJoCo GUI window, expand the right-hand side panel.")
    print("2. Expand the 'Joints' section to see sliders for all joints.")
    print("3. Drag the sliders to adjust G1's posture.")
    print("4. Watch this terminal: it will print the updated 'pres_joints' dictionary!")
    print("5. Once you are happy, copy-paste the dictionary to 'src/g1_rl/g1_hand_env.py'.")
    print("==================================================")
    
    last_print_time = 0.0
    last_qpos = env.data.qpos.copy()
    
    try:
        with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
            while viewer.is_running():
                current_time = time.time()
                
                # Check for changes (threshold 1e-3 rad) and rate limit prints to 1.5s
                if np.max(np.abs(env.data.qpos - last_qpos)) > 1e-3:
                    if (current_time - last_print_time) > 1.5:
                        last_qpos = env.data.qpos.copy()
                        last_print_time = current_time
                        
                        # Print formatted python dictionary
                        print("\n--- Copy and paste this into src/g1_rl/g1_hand_env.py ---")
                        print("        pres_joints = {")
                        
                        # List of joints grouped logically
                        joint_groups = [
                            ("Left Arm", [
                                "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
                                "left_shoulder_yaw_joint", "left_elbow_joint",
                                "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"
                            ]),
                            ("Right Arm", [
                                "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
                                "right_shoulder_yaw_joint", "right_elbow_joint",
                                "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint"
                            ]),
                            ("Left Hand (Dex3)", [
                                "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
                                "left_hand_middle_0_joint", "left_hand_middle_1_joint",
                                "left_hand_index_0_joint", "left_hand_index_1_joint"
                            ]),
                            ("Right Hand (Dex3)", [
                                "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
                                "right_hand_middle_0_joint", "right_hand_middle_1_joint",
                                "right_hand_index_0_joint", "right_hand_index_1_joint"
                            ]),
                            ("Waist", [
                                "waist_pitch_joint", "waist_roll_joint", "waist_yaw_joint"
                            ])
                        ]
                        
                        for group_name, joint_list in joint_groups:
                            print(f"            # --- {group_name} ---")
                            for jname in joint_list:
                                jid = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                                if jid >= 0:
                                    qadr = env.model.jnt_qposadr[jid]
                                    val = env.data.qpos[qadr]
                                    print(f'            "{jname}": {val:.4f},')
                        print("        }")
                        print("---------------------------------------------------------")
                
                # Update forward kinematics based on the joint positions set via GUI/sliders
                mujoco.mj_forward(env.model, env.data)
                
                # Sync viewer to show updates
                viewer.sync()
                # Run small sleep
                time.sleep(0.05)
                
    except KeyboardInterrupt:
        print("\nPreviewer closed by user.")
    finally:
        env.close()
        print("Viewer closed.")

if __name__ == "__main__":
    preview_reset_pose()
