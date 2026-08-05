# CSCN8020 Final Project: Unitree G1 Three-Digit Hand-Gesture Control via PPO
## Continuous-Action PPO Controller for humanoid Hand Gesture Modulation

### Student Details
* **Team Members:** Emmanuel • Liggia • Cemil • Chao
* **Course:** CSCN8020 - Reinforcement Learning Programming (Final Project)
* **Instructor:** Prof. Enrique Espinosa
* **Repository GitHub URL:** https://github.com/caatat741213/Thumbs_Robot.git

---

### Project Description
This repository contains the complete implementation, verification, and evaluation of a continuous-action Actor-Critic PPO agent designed to control the 3-digit hand (comprising two main fingers and one thumb) of the Unitree G1 humanoid robot in a simulated MuJoCo environment.

The controller solves a multi-task hand-gesture target-conditioned modulation task. Specifically, the agent learns to output joint position increments for the 5-DOF wrist and fingers, steering the hand to achieve three distinct target gestures:
1. **Thumbs Up**: Thumb extended, wrist rotated upwards, main fingers flexed.
2. **Open/Stop**: All three digits fully extended, palm facing forward.
3. **Thumbs Down**: Thumb extended, wrist rotated downwards, main fingers flexed.

The repository aligns with the strict academic requirements of **"Math MDP $\rightarrow$ Algorithm Logic $\rightarrow$ Code Variables $\rightarrow$ Real-time logs"** 4-in-1 alignment mapping.

---

### Repository Directory Structure

```text
.
├── Thumbs_Robot.ipynb         # Interactive walkthrough notebook in English (cleared and parameterized)
├── Thumbs_Robot_TW.ipynb      # Interactive walkthrough notebook in Traditional Chinese
├── test_mujoco_viewer.py      # Basic MuJoCo rendering setup validation script
├── assets/                    # Simulation scene and meshes for fixed-base Unitree G1 robot
│   └── g1_fixed_base/
│       ├── scene_29dof_fixed_base.xml
│       └── meshes/
├── doc/                       # Project specifications, plans, and instructions
│   ├── Final_project_plan.md
│   ├── David_said.md
│   ├── Assignment3.md
│   └── Unitree_MuJoCo_G1_Primer_Workshop.md
├── models/                    # Saved neural network checkpoints and PPO agent weights (.pt)
├── results/                   # Training CSV logs and evaluation metrics
│   ├── ppo_config_a/          # Output folder for Config A logs (step_log, update_log, episode_log)
│   ├── img/                   # Folder for generated high-resolution convergence plot files
│   │   └── ppo_config_a/
│   │       ├── accumulated_returns.png
│   │       ├── success_rate.png
│   │       ├── optimization_losses.png
│   │       └── policy_entropy.png
│   └── ppo_evaluation/        # Evaluation results and CSV outputs
├── src/                       # Source code directory
│   ├── g1_rl/                 # Custom Gymnasium environment classes for G1 robot
│   │   ├── g1_hand_env.py     # Main 3-digit hand gesture Gymnasium environment wrapper
│   │   ├── g1_elbow_env.py    # Elbow single-joint environment wrapper (Assignment 3 Baseline)
│   │   └── g1_model_audit.py  # Model joint audit script
│   ├── Thumbs_Robot/          # PPO algorithm components and execution runners
│   │   ├── actor_critic_network.py  # Continuous Gaussian policy Actor-Critic network
│   │   ├── agent.py                 # PPO agent update and action-selection logic
│   │   ├── rollout_buffer.py        # Rollout transitions memory and GAE advantage calculator
│   │   ├── train_thumbs.py          # Continuous PPO training runner
│   │   ├── evaluate_thumbs.py       # Deterministic multi-gesture policy evaluator
│   │   ├── render_thumbs.py         # MuJoCo 3D interactive policy visualizer
│   │   ├── plot_results.py          # High-resolution convergence curves plotter
│   │   └── smoke_test.py            # Quick execution smoke test runner for PPO components
│   └── dqn/                   # Assignment 3 DQN single-joint elbow controller codebase
├── requirements.txt           # Python dependency requirements list
└── .gitignore                 # Version control exclude patterns
```

---

### Environment Setup Instructions

To configure the workspace and verify the reinforcement learning simulation environment:

1. **WSL Connection (Optional but Recommended for Windows Users):**
   * Open the command palette (`Ctrl + Shift + P`) in VS Code.
   * Select `WSL: Connect to WSL` and open your project directory inside the WSL filesystem.

2. **Install System-Level Compilation and OpenGL Dependencies:**
   Run the following commands in the WSL/Linux terminal to install packages required for python compilation, rendering, and GLFW window management:
   ```bash
   sudo apt update && sudo apt install -y \
       python3-venv \
       python3-dev \
       build-essential \
       git \
       cmake \
       ninja-build \
       pkg-config \
       libglfw3 \
       libglfw3-dev \
       libgl1-mesa-dev \
       libegl1-mesa-dev \
       libxinerama-dev \
       libxcursor-dev \
       libxrandr-dev
   ```

3. **Initialize Local Virtual Environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

4. **Upgrade Package Managers and Install Requirements:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

5. **Clone and Integrate External Unitree MuJoCo dependency:**
   ```bash
   git clone https://github.com/unitreerobotics/unitree_mujoco.git external/unitree_mujoco
   ```

6. **Verify Library Versions and CUDA Availability:**
   Check python runtime imports and CUDA hardware support:
   ```bash
   python -c "import mujoco, torch, gymnasium; print('MuJoCo Version:', mujoco.__version__); print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Gymnasium Version:', gymnasium.__version__)"
   ```

7. **Verify Physics Renderer & GUI Window Viewer:**
   Verify MuJoCo physics and UI window visualization work correctly by launching the basic test script:
   ```bash
   PYTHONPATH=src python src/test_mujoco_viewer.py
   ```

---

### PPO & Environment Implementation Details

The continuous action workspace incorporates the following modules:

1. **G1HandEnv (`g1_hand_env.py`):**
   * **State Space (32 Dimensions):** Features joint positions, joint velocities, target gesture coordinates, joint target error, one-hot target gesture representation, and the previous step action.
   * **Action Space (5 Dimensions):** Continuous position target increments ($\Delta \theta_t$) for the wrist joints (roll, pitch, yaw) and fingers (main fingers flex, thumb flex).
   * **Composite Reward Function:**
     $$r_t = w_p(e_{t-1} - e_t) - w_h E_{\text{hand}} - w_o E_{\text{orientation}} - w_v \|\dot{q}_t\|^2 - w_a \|a_t\|_2^2 - w_s \|a_t - a_{t-1}\|_2^2 + b_{\text{hold}} I_{\text{hold}} - c_{\text{time}}$$
     It combines pose error reduction progress, pose/orientation penalties, joint limit penalties, action smoothness penalties, step change smoothness penalties, hold bonus (for sustaining correct gesture for 15+ steps), and a time penalty.
   * **Success Conditions:** Evaluated via a temporal gesture hold validation loop. Success is declared when joint and wrist errors remain within tolerance for at least 15 steps.

2. **Actor-Critic Network (`actor_critic_network.py`):**
   * **Actor Network:** MLP with hidden sizes `[128, 128]` predicting the mean vector ($\mu$) and standard deviation ($\sigma$) of the continuous action Gaussian distribution. Action standard deviation is dynamically bounded to prevent exploding policy parameters.
   * **Critic Network:** MLP with hidden sizes `[128, 128]` outputting the state value estimate $V(s)$.

3. **Rollout Buffer (`rollout_buffer.py`):**
   * Stores on-policy trajectory batches of size `rollout_length = 2048` transitions.
   * Computes Generalized Advantage Estimation (GAE) with parameters $\gamma = 0.99$ and $\lambda = 0.95$.

4. **PPO Agent (`agent.py`):**
   * Manages neural network optimization using clipped surrogate objectives.
   * Implements clipped value objective and policy entropy regularization for continuous exploration.
   * Features robust NaN/Inf prevention clipping and gradient norm clipping.

---

### Execution Commands

All scripts should be executed from the repository root with `PYTHONPATH=src`:

* **Module Smoke Test:**
  Verify buffer collection, Actor-Critic forward propagation, GAE advantage estimation, and update optimization loops:
  ```bash
  PYTHONPATH=src python src/Thumbs_Robot/smoke_test.py
  ```

* **Training PPO policy (Headless):**
  Train a PPO model with a specific configuration name. Logs and configurations will write to `results/{CONFIG_NAME}`:
  ```bash
  # Train Config A (Baseline 100,000 steps)
  PYTHONPATH=src python src/Thumbs_Robot/train_thumbs.py --results-dir results/ppo_config_a
  
  # Train Config A for a quick test (e.g., 2000 steps)
  PYTHONPATH=src python src/Thumbs_Robot/train_thumbs.py --results-dir results/ppo_config_a --max-total-steps 2000
  ```

* **Evaluating the Policy:**
  Run deterministic greedy evaluation over 30 episodes (10 per gesture) to record detailed performance:
  ```bash
  PYTHONPATH=src python src/Thumbs_Robot/evaluate_thumbs.py --checkpoint models/ppo_config_a_best.pt --output_dir results/ppo_config_a_evaluation
  ```

* **Plotting Training Convergence Curves:**
  Generate separate high-resolution training metric plots (`accumulated_returns.png`, `success_rate.png`, `optimization_losses.png`, `policy_entropy.png`) saved under `results/img/{CONFIG_NAME}/`:
  ```bash
  PYTHONPATH=src python src/Thumbs_Robot/plot_results.py --dir results/ppo_config_a --eval_csv results/ppo_config_a_evaluation/eval_results.csv
  ```

* **Visualizing Gesture Rendering in MuJoCo GUI:**
  Launch the interactive 3D visualizer to command and render Thumbs Up, Open/Stop, and Thumbs Down gestures sequentially:
  ```bash
  PYTHONPATH=src python src/Thumbs_Robot/render_thumbs.py --checkpoint models/ppo_config_a_best.pt
  ```

---

### Academic Mapping & Core Discussions

1. **3-Digit Morphology Constraints:**
   The policy is constrained to the Unitree G1 humanoid hand morphology, containing only 1 wrist roll, 1 wrist pitch, 1 wrist yaw, and 2 finger joints (flexors) active per side. 

2. **MDP to Code Mapping Verification:**
   * **MDP State** $s_t$: Mapped to `obs` array returned by `env._get_obs()`.
   * **MDP Action** $a_t$: Mapped to `clipped_action` returned by `agent.select_action`.
   * **PPO Loss** $L_{\text{total}}$: Mapped to variables in `agent.update()` (`actor_loss`, `critic_loss`, `entropy`).
   * **Console Log Output**: Mapped to output table fields printed in `train_thumbs.py` (`Update`, `Steps`, `Mean_Rwd`, `Succ_%`, `Loss_A`, `Loss_C`).

3. **Generalization across Gestures:**
   Generalization is achieved by appending the target gesture vector explicitly to the observation space. The network represents a unified, target-conditioned policy mapping joint positions and relative error to the correct target increments.
