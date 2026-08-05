import argparse
import os
from typing import Optional
import mujoco

def audit_g1_model(xml_path: str, use_viewer: bool = False):
    """
    Load the fixed-base Unitree G1 model and audit its joints, limits, and actuators.
    Helps calibrate target pose vectors for Thumbs Up, Open/Stop, and Thumbs Down.
    """
    if not os.path.exists(xml_path):
        print(f"Error: Model XML file not found at '{xml_path}'")
        return

    print("========================================")
    print(f"Loading Unitree G1 Model from: {xml_path}")
    print("========================================")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    print("\n--- Joint Audit ---")
    print(f"Total degrees of freedom (nv): {model.nv}")
    print(f"Total number of joints (njnt): {model.njnt}")
    
    for i in range(model.njnt):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        joint_type = model.jnt_type[i]
        joint_range = model.jnt_range[i]
        print(f"Joint ID {i:02d} | Name: '{joint_name}' | Type ID: {joint_type} | Limits: {joint_range}")

    print("\n--- Actuator Audit ---")
    print(f"Total number of actuators (nu): {model.nu}")
    
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        ctrl_range = model.actuator_ctrlrange[i]
        print(f"Actuator ID {i:02d} | Name: '{act_name}' | Control Range: {ctrl_range}")

    print("\n--- Calibration Checkpoints ---")
    print("1. Target Pose Vector: Thumbs Up")
    print("   - Fingers flexed, Thumb extended, Wrist oriented up.")
    print("2. Target Pose Vector: Open/Stop")
    print("   - All digits extended, Palm facing forward.")
    print("3. Target Pose Vector: Thumbs Down")
    print("   - Fingers flexed, Thumb extended, Wrist oriented down.")

    if use_viewer:
        print("\nLaunching MuJoCo passive viewer for manual posture calibration...")
        # mujoco.viewer.launch(model, data)
        # Note: Active viewer launch depends on mujoco visualizer availability.
    else:
        print("\nHeadless audit completed. Save these ranges to target vectors.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unitree G1 3-Digit Hand Model Auditor [Phase 0]")
    parser.add_argument(
        "--xml-path", 
        type=str, 
        default="assets/g1_hand.xml", 
        help="Path to G1 hand MuJoCo XML model file"
    )
    parser.add_argument(
        "--no-viewer", 
        action="store_true", 
        help="Disable interactive visualizer (headless mode)"
    )
    args = parser.parse_args()

    audit_g1_model(xml_path=args.xml_path, use_viewer=not args.no_viewer)
