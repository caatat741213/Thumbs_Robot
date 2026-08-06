"""
G1 Hand-Gesture Gymnasium Environment
Stage: Phase 1 (Environment Setup)
Primary Function:
- Defines state space s_t and action space a_t for controlling the wrist and 3-digit fingers of Unitree G1.
- Implements a composite multi-objective reward function including pose, orientation, smoothness, and hold components.
- Manages joint limits, gravity compensation, and target gesture check streaks.
"""

import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import time
from typing import Dict, Tuple, Any, Optional

# Ensure src/ is in the PYTHONPATH
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_dir not in sys.path:
    sys.path.append(src_dir)

class G1HandEnv(gym.Env):
    """
    Gymnasium environment for Unitree G1 3-digit hand-gesture control.
    Morphology: 2 main fingers (finger_1, finger_2) and 1 thumb.
    Gestures: Thumbs Up, Open/Stop, Thumbs Down.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self, 
        xml_path: str = "assets/g1_fixed_base/scene_29dof_fixed_base.xml", 
        render_mode: Optional[str] = None,
        action_increment: float = 0.05,
        finger_increment: float = 0.08,
        frame_skip: int = 10,
        maximum_episode_steps: int = 150,
        pose_tolerance: float = 0.06,
        orient_tolerance: float = 0.12,
        required_success_steps: int = 15,
        controlled_kp: float = 25.0,
        controlled_kd: float = 2.5,
        hold_kp: float = 35.0,
        hold_kd: float = 3.5,
    ):
        super(G1HandEnv, self).__init__()
        self.xml_path = xml_path
        self.render_mode = render_mode
        self.action_increment = action_increment
        self.finger_increment = finger_increment
        self.frame_skip = frame_skip
        self.maximum_episode_steps = maximum_episode_steps
        self.pose_tolerance = pose_tolerance
        self.orient_tolerance = orient_tolerance
        self.required_success_steps = required_success_steps
        self.controlled_kp = controlled_kp
        self.controlled_kd = controlled_kd
        self.hold_kp = hold_kp
        self.hold_kd = hold_kd

        # 1. Load MuJoCo Model & Data
        if not os.path.exists(self.xml_path):
            self.xml_path = os.path.join(src_dir, "..", xml_path)
            if not os.path.exists(self.xml_path):
                raise FileNotFoundError(f"MuJoCo model not found at: {xml_path}")
                
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        # 2. Setup Controlled Wrist Joints & Actuators (Left Wrist pitch, roll, yaw)
        self.wrist_joints = ["left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint"]
        self.wrist_actuators = ["left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw"]
        
        self.wrist_joint_ids = [self._require_id(mujoco.mjtObj.mjOBJ_JOINT, j) for j in self.wrist_joints]
        self.wrist_actuator_ids = [self._require_id(mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in self.wrist_actuators]
        
        self.wrist_qpos_indices = [int(self.model.jnt_qposadr[jid]) for jid in self.wrist_joint_ids]
        self.wrist_qvel_indices = [int(self.model.jnt_dofadr[jid]) for jid in self.wrist_joint_ids]
        
        # Load Joint Ranges for Wrist Hard-Clipping boundary limits
        self.wrist_lows = []
        self.wrist_highs = []
        for jid in self.wrist_joint_ids:
            low, high = self.model.jnt_range[jid]
            self.wrist_lows.append(float(low))
            self.wrist_highs.append(float(high))
        self.wrist_lows = np.array(self.wrist_lows, dtype=np.float32)
        self.wrist_highs = np.array(self.wrist_highs, dtype=np.float32)

        # 3. Setup Joint Actuator Mapping (for non-controlled shoulder/elbow lock)
        self.joint_actuator_mapping = self._build_joint_actuator_mapping()

        # 4. Observation Space (dim=32):
        # s_t = [q_t (6), q_dot_t (6), q_target (6), error_t (6), gesture_one_hot (3), last_action (5)]
        obs_dim = 32
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 5. Continuous Action Space: a_t = [dw_roll, dw_pitch, dw_yaw, d_thumb, d_fingers] (dim=5)
        # Action boundaries normalized to [-1.0, 1.0] for optimization stability
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )

        # 6. Gesture definitions and target postures [wrist_roll, wrist_pitch, wrist_yaw, thumb, finger_1, finger_2]
        self.gestures = {
            0: "THUMBS_UP",
            1: "OPEN_STOP",
            2: "THUMBS_DOWN"
        }
        self.target_vectors = {
            0: np.array([1.2, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),   # THUMBS_UP
            1: np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=np.float32),   # OPEN_STOP
            2: np.array([-1.2, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32),  # THUMBS_DOWN
        }

        # Simulated dynamic states for the virtual G1 3-digit fingers (scaled [0.0, 1.0])
        self.virtual_qpos = np.zeros(3, dtype=np.float32)
        self.virtual_qvel = np.zeros(3, dtype=np.float32)
        
        self.current_gesture_id = 0
        self.last_action = np.zeros(5, dtype=np.float32)
        self.controller_target = np.zeros(3, dtype=np.float32)
        self.hold_targets = np.zeros(self.model.nq, dtype=np.float64)
        
        self.episode_step = 0
        self.success_streak = 0
        self.viewer = None
        self.dt = self.model.opt.timestep * self.frame_skip

    def _require_id(self, object_type: mujoco.mjtObj, name: str) -> int:
        """Helper to safely retrieve MuJoCo ID."""
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"MuJoCo object was not found: {name}")
        return object_id

    def _build_joint_actuator_mapping(self) -> list[tuple[int, int, int]]:
        """Maps actuator IDs to corresponding joint positions and velocities."""
        mapping = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                raise RuntimeError(f"Actuator {actuator_id} is not attached to a joint.")
            qpos_index = int(self.model.jnt_qposadr[joint_id])
            qvel_index = int(self.model.jnt_dofadr[joint_id])
            mapping.append((actuator_id, qpos_index, qvel_index))
        return mapping

    @staticmethod
    def _calculate_pd_torque(target_angle: float, current_angle: float, current_velocity: float, kp: float, kd: float) -> float:
        """Computes Proportional-Derivative (PD) torque feedback command."""
        return kp * (target_angle - current_angle) - kd * current_velocity

    def _get_observation(self) -> np.ndarray:
        """Retrieves physical states and builds observation vector s_t."""
        q_wrist = np.array([self.data.qpos[idx] for idx in self.wrist_qpos_indices], dtype=np.float32)
        v_wrist = np.array([self.data.qvel[idx] for idx in self.wrist_qvel_indices], dtype=np.float32)

        # Full posture vector including simulated virtual fingers
        q_t = np.concatenate([q_wrist, self.virtual_qpos])
        v_t = np.concatenate([v_wrist, self.virtual_qvel])

        # Current target and error vectors
        q_target = self.target_vectors[self.current_gesture_id]
        error_t = q_target - q_t

        # Target gesture ID represented as a 3-dim one-hot vector
        one_hot = np.zeros(3, dtype=np.float32)
        one_hot[self.current_gesture_id] = 1.0

        # Construct observation: q_t(6) + v_t(6) + q_target(6) + error_t(6) + one_hot(3) + last_action(5) = 32
        obs = np.concatenate([q_t, v_t, q_target, error_t, one_hot, self.last_action])
        return obs.astype(np.float32)

    def _get_info(self) -> Dict[str, Any]:
        """Calculates hand-pose and orientation errors for stats."""
        q_wrist = np.array([self.data.qpos[idx] for idx in self.wrist_qpos_indices], dtype=np.float32)
        q_target = self.target_vectors[self.current_gesture_id]
        
        # Norm error for the virtual fingers
        pose_error_norm = float(np.linalg.norm(self.virtual_qpos - q_target[3:]))
        # Norm error for the wrist orientation
        orientation_error = float(np.linalg.norm(q_wrist - q_target[:3]))

        return {
            "target_gesture": self.gestures[self.current_gesture_id],
            "pose_error_norm": pose_error_norm,
            "orientation_error": orientation_error,
            "success_streak": self.success_streak,
            "episode_step": self.episode_step,
        }

    def _apply_controller(self) -> None:
        """Applies torque commands combining PD feedback with Coriolis/gravity compensation."""
        for actuator_id, qpos_index, qvel_index in self.joint_actuator_mapping:
            current_position = float(self.data.qpos[qpos_index])
            current_velocity = float(self.data.qvel[qvel_index])

            # Apply active PD control to wrist actuators; lock arm/waist joints to baseline posture
            if actuator_id in self.wrist_actuator_ids:
                list_idx = self.wrist_actuator_ids.index(actuator_id)
                desired_position = self.controller_target[list_idx]
                kp = self.controlled_kp
                kd = self.controlled_kd
            else:
                desired_position = float(self.hold_targets[qpos_index])
                kp = self.hold_kp
                kd = self.hold_kd

            pd_torque = self._calculate_pd_torque(desired_position, current_position, current_velocity, kp, kd)
            bias_torque = float(self.data.qfrc_bias[qvel_index])
            
            # Combine PD torque with passive gravity/Coriolis bias-forces compensation
            commanded_torque = pd_torque + bias_torque
            
            control_low, control_high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(commanded_torque, control_low, control_high)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Resets the environment, physical states, and samples new target gesture."""
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = 0.0

        # Scripted arm presentation posture (shoulder and elbow held stable to showcase gestures)
        pres_joints = {
            "left_shoulder_pitch_joint": -0.5,
            "left_shoulder_roll_joint": 0.4,
            "left_shoulder_yaw_joint": -0.2,
            "left_elbow_joint": 1.2
        }
        for name, pose_val in pres_joints.items():
            try:
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0:
                    qpos_addr = self.model.jnt_qposadr[jid]
                    self.data.qpos[qpos_addr] = pose_val
            except:
                pass

        mujoco.mj_forward(self.model, self.data)
        self.hold_targets = self.data.qpos.copy()

        # Configure target gesture configuration selection
        if options is not None and "gesture_id" in options:
            self.current_gesture_id = int(options["gesture_id"])
        elif options is not None and "target_gesture" in options:
            target_str = options["target_gesture"]
            matched_id = None
            for gid, name in self.gestures.items():
                if name.upper() == target_str.upper():
                    matched_id = gid
                    break
            self.current_gesture_id = matched_id if matched_id is not None else self.np_random.choice(list(self.gestures.keys()))
        else:
            self.current_gesture_id = self.np_random.choice(list(self.gestures.keys()))

        # Reset controller targets to match initial physical positions
        self.controller_target = np.array([self.data.qpos[idx] for idx in self.wrist_qpos_indices], dtype=np.float32)

        # Initialize simulated dynamic fingers with minor randomized noise for exploration
        self.virtual_qpos = self.np_random.uniform(0.1, 0.9, size=3).astype(np.float32)
        self.virtual_qvel.fill(0.0)

        self.last_action.fill(0.0)
        self.episode_step = 0
        self.success_streak = 0

        observation = self._get_observation()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Applies action steps, runs simulation, checks termination, and returns reward."""
        if action.shape[0] != 5:
            raise ValueError(f"Action dimension must be 5, but got {action.shape[0]}")

        # Hard clip continuous action to safe bounds [-1, 1]
        clipped_action = np.clip(action, self.action_space.low, self.action_space.high)
        self.last_action = clipped_action.copy()

        # Apply Continuous Increments to Wrists Controller Target
        self.controller_target = np.clip(
            self.controller_target + clipped_action[:3] * self.action_increment,
            self.wrist_lows,
            self.wrist_highs
        )

        # Step simulated finger dynamics
        prev_virtual_qpos = self.virtual_qpos.copy()
        
        # action[3] -> thumb, action[4] -> fingers (index and middle shared)
        self.virtual_qpos[0] = np.clip(self.virtual_qpos[0] + clipped_action[3] * self.finger_increment, 0.0, 1.0)
        self.virtual_qpos[1] = np.clip(self.virtual_qpos[1] + clipped_action[4] * self.finger_increment, 0.0, 1.0)
        self.virtual_qpos[2] = np.clip(self.virtual_qpos[2] + clipped_action[4] * self.finger_increment, 0.0, 1.0)
        
        # Calculate simulated finger velocity
        self.virtual_qvel = (self.virtual_qpos - prev_virtual_qpos) / self.dt

        # Step MuJoCo simulator physics
        for _ in range(self.frame_skip):
            self._apply_controller()
            mujoco.mj_step(self.model, self.data)

        self.episode_step += 1

        # Check Hold Success and Termination Criteria
        info = self._get_info()
        pose_error = info["pose_error_norm"]
        orient_error = info["orientation_error"]

        # Hold criteria: both finger pose error and wrist orientation error are below thresholds
        if pose_error <= self.pose_tolerance and orient_error <= self.orient_tolerance:
            self.success_streak += 1
        else:
            self.success_streak = 0

        # Terminated when target holds consecutively, truncated at max steps limit
        terminated = self.success_streak >= self.required_success_steps
        truncated = self.episode_step >= self.maximum_episode_steps

        # Compute Reward components
        reward, reward_info = self.compute_reward(clipped_action, pose_error, orient_error)
        
        # Success bonus injection
        if terminated:
            reward += 10.0
            reward_info["success_bonus"] = 10.0

        observation = self._get_observation()
        info["reward_info"] = reward_info
        info["is_success"] = terminated

        if self.render_mode == "human":
            self.render()

        return observation, float(reward), terminated, truncated, info

    def compute_reward(self, action: np.ndarray, pose_error: float, orientation_error: float) -> Tuple[float, Dict[str, float]]:
        """Computes multi-objective reward combining errors, smooth penalties, and bonuses."""
        # Weights for multi-objective optimization components
        w_progress = 1.5
        w_hand = 2.0
        w_orient = 1.0
        w_smooth = 0.05
        w_smooth_delta = 0.05
        w_joint_limit = 0.2
        b_hold = 0.5
        c_time = 0.1

        # Progress reward can be added during trajectory transitions
        r_progress = 0.0 
        r_hand = -pose_error
        r_orient = -orientation_error
        r_smooth = -float(np.sum(np.square(action)))
        
        # Smoothness delta penalizes change in action command to avoid jittering
        r_smooth_delta = -float(np.sum(np.square(action - self.last_action)))
        
        # Joint limit penalty: penalize approaching wrist physical limits
        q_wrist = np.array([self.data.qpos[idx] for idx in self.wrist_qpos_indices])
        limit_violations = 0.0
        for i, val in enumerate(q_wrist):
            margin = (self.wrist_highs[i] - self.wrist_lows[i]) * 0.08
            if val < self.wrist_lows[i] + margin or val > self.wrist_highs[i] - margin:
                limit_violations += 1.0
        r_joint_limit = -limit_violations

        # Hold bonus reward: encourage maintaining safe stable gesture posture
        r_hold = b_hold if (pose_error <= self.pose_tolerance and orientation_error <= self.orient_tolerance) else 0.0
        r_success = 0.0 # Success terminal bonus handled at step-level
        r_time = -c_time

        total_reward = (
            (w_progress * r_progress)
            + (w_hand * r_hand)
            + (w_orient * r_orient)
            + (w_smooth * r_smooth)
            + (w_smooth_delta * r_smooth_delta)
            + (w_joint_limit * r_joint_limit)
            + r_hold
            + r_success
            + r_time
        )

        reward_info = {
            "reward_total": total_reward,
            "progress": r_progress,
            "pose_error_penalty": r_hand,
            "orientation_penalty": r_orient,
            "smoothness_penalty": r_smooth,
            "smoothness_delta_penalty": r_smooth_delta,
            "joint_limit_penalty": r_joint_limit,
            "hold_bonus": r_hold,
            "success_bonus": r_success,
            "time_penalty": r_time
        }

        return total_reward, reward_info

    def render(self):
        """Launches passive viewer or syncs rendering steps."""
        if self.render_mode != "human":
            return

        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        if not self.viewer.is_running():
            return

        self.viewer.sync()
        time.sleep(1.0 / self.metadata["render_fps"])

    def close(self):
        """Safely closes passive viewer resources."""
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
