# Presentation Speech Draft: Thumbs_Robot Gesture Control via PPO

This document contains the presentation slides and speech scripts for the group presentation. The English level is designed to be clear, professional, and easy to present.

---

## Slide 1: Introduction & Project Overview

### Slide Content
* **Project Name**: Thumbs_Robot: Unitree G1 Three-Digit Hand-Gesture Control via PPO
* **Objective**: Control Unitree G1 robot's 3-digit hand using continuous-action PPO for three target gestures:
  1. **Thumbs Up**
  2. **Open / Stop**
  3. **Thumbs Down**
* **Key Academic Highlight**: Math-to-Code-to-Log Mapping (Mathematical MDP $\rightarrow$ Algorithmic Logic $\rightarrow$ Python Code Variables $\rightarrow$ CSV/Console logs).
* **Team Members**: Emmanuel • Liggia • Cemil • Chao

### Speech Script
"Hello everyone. Today, our team is presenting our project, **Thumbs_Robot**. In this project, we use Reinforcement Learning to control the hand-gesture movements of the Unitree G1 robot. 

Unitree G1 is a humanoid robot, and we focus on controlling its 3-digit hand to perform three main gestures: Thumbs Up, Open/Stop, and Thumbs Down in the MuJoCo simulation environment. 

A key highlight of our project is the **Four-in-One Mapping** required by the professor. We mapped our mathematical MDP formulas directly to algorithm logic, Python code variables, and console logs. Let's start with how we set up the project environment."

---

## Slide 2: Environment Installation & Setup (Step 0 to Step 3)

### Slide Content
* **Step 0: WSL & Linux System Setup**: Using Windows Subsystem for Linux, installing build packages (GCC, CMake, OpenGL, GLFW) and Python virtual environment.
* **Step 1: Dependency Installation**: Upgrading pip, installing `requirements.txt` (including `gymnasium`, `mujoco`, `numpy`, `torch`, `pandas`, `matplotlib`).
* **Step 2: Unitree MuJoCo Repo**: Cloning the official Unitree MuJoCo code into the `external` directory.
* **Step 3: Verification**: Running verification script to check MuJoCo (3.10.0), PyTorch (with CUDA), and Gymnasium versions.

### Speech Script
"For the first part, we completed the **Environment Installation and Setup**. 

In **Step 0**, we configured the workspace using WSL (Windows Subsystem for Linux) because MuJoCo and robot tools run best on Linux. We installed packages like python3-venv, GCC, CMake, and OpenGL graphics libraries to support 3D rendering. We also created a local virtual environment to keep our workspace clean.

In **Step 1**, we upgraded pip and installed all the Python packages from `requirements.txt`. These packages include Gymnasium, MuJoCo, PyTorch, Pandas, and Matplotlib.

In **Step 2**, we cloned the official Unitree MuJoCo repository. This repository provides the physical models, joint meshes, and XML files for the G1 robot.

Finally, in **Step 3**, we verified the installation. As you can see, our setup successfully loaded Python 3.12, MuJoCo 3.10, PyTorch with CUDA enabled, and Gymnasium 1.3. This ensured a stable baseline environment for training."

---

## Slide 3: Model Joint Audit & Gesture Calibration (Phase 0)

### Slide Content
* **Fixed-Base G1 Model Generation**: Removing floating-base joint (floating_base_joint) to fix G1 in space, simplifying hand training.
* **Joint and Actuator Audit**:
  * Total 29 joints & 29 actuators (hip, knee, ankle, waist, shoulder, elbow, wrist).
  * We focus on controlling the wrist and digits.
* **3-Digit Hand Rule**: Unitree G1 uses a 3-digit structure (**two primary fingers + one thumb**), NOT a 5-digit human-like hand.
* **Manually Calibrated Poses**:
  * **Thumbs Up**: Thumb extended, primary fingers curled, wrist turned up.
  * **Open/Stop**: All digits extended, palm facing forward.
  * **Thumbs Down**: Thumb extended, primary fingers curled, wrist turned down.

### Speech Script
"Moving on to **Phase 0: Model Joint Audit & Gesture Calibration**.

First, G1 is a full humanoid robot, but we only need to control its hands and wrists. So, we wrote a script called `create_fixed_base_g1.py` to remove the floating base joint. This fixes the G1 torso in the space and makes the simulation much simpler.

Second, we audited the G1 joints. The G1 model has 29 degrees of freedom. We printed the joint limits and actuator control ranges. For our task, we only control the wrist and finger actuators.

Third, we calibrated three target gestures. It is very important to note the **3-Digit Hand Rule**: G1's hand has only two primary fingers and one thumb. It is a three-digit hand, not a five-finger human hand. 
* For **Thumbs Up**, the thumb is out, fingers are curled, and the wrist points up.
* For **Open or Stop**, all three digits are fully open and flat.
* For **Thumbs Down**, the thumb is out, fingers are curled, and the wrist points down. 

These target joint positions serve as the goals for our reinforcement learning agent."

---

## Slide 4: Gymnasium Environment Development & Random Action Testing (Phase 1)

### Slide Content
* **State Space ($s_t$)**: 32 dimensions. Includes joint angles ($q_t$), velocities ($\dot{q}_t$), target angles ($q_{target}$), joint errors ($q_{target} - q_t$), target gesture one-hot vector, and previous action ($a_{t-1}$).
* **Action Space ($a_t$)**: 5 dimensions. Continuous joint increments control the wrist joints and fingers.
* **Composite Reward Function ($r_t$)**:
  $$r_t = w_p(e_{t-1} - e_t) - w_h E_{\text{hand}} - w_o E_{\text{orientation}} - w_v \|\dot{q}_t\|^2 - w_a \|a_t\|^2 - w_s \|a_t-a_{t-1}\|_2^2 + b_{\text{hold}} I_{\text{hold}} - c_{\text{time}}$$
  * *Pose and Orientation Error*: Penalizes deviation from the target gesture.
  * *Energy and Safety*: Penalizes large actions and joint velocities.
  * *Smoothness*: Penalizes joint jerking to prevent shaking.
  * *Hold Bonus*: Awarded for maintaining target pose.
* **Random Action Test**: Verified environment initialization, observation/action shapes, and step reward calculation.

### Speech Script
"Next is **Phase 1: Gymnasium Environment Development and Random Action Testing**.

We developed a custom Gymnasium environment called `G1HandEnv`.
Our state space has **32 dimensions**. It tells the agent the current joint angles, joint velocities, the target gesture vector, the error between current and target joints, and the previous actions. 

Our action space has **5 dimensions**, representing continuous joint increments for the wrist and fingers.

To teach the G1 hand, we designed a **Composite Reward Function**. This reward has several parts:
* First, a progress reward that praises the hand for getting closer to the target.
* Second, penalties for pose and orientation errors.
* Third, penalties for action magnitude and velocity to save energy and protect the robot.
* Fourth, a **smoothness penalty** based on action changes, which prevents the G1 fingers from shaking or vibrating.
* Lastly, a **hold bonus** when G1 holds the target gesture correctly.

We ran a random action smoke test. As shown, the environment initialized successfully, outputting the correct observation shape of 32, action shape of 5, and computing step rewards correctly."

---

## Slide 5: PPO Algorithm Architecture & Network Design (Phase 2)

### Slide Content
* **PPO Training Workflow**:
  * **Phase 1: Rollout Collection**: Actor predicts Gaussian mean and std. Action is sampled, clipped for safety, and run in MuJoCo. Experience is saved to Buffer.
  * **Phase 2: GAE Advantage Estimation**: Handles time truncation via Critic bootstrap $V(s_{N+1})$. Computes GAE advantages ($A_t$) and Returns ($R_t$).
  * **Phase 3: Network Updates**: Shuffles data into mini-batches, updates Actor and Critic using PPO Clipped Loss with Entropy Regularization:
    $$L_{total} = L_{actor} + c_1 L_{critic} - c_2 \mathcal{H}(\pi_\theta)$$
  * **Phase 4: Diagnostics**: Monitor policy divergence (KL), explained variance, policy entropy, and loss components.
* **Component Responsibilities**:
  * **Actor Network**: Outputs action mean and std dev; samples actions.
  * **Critic Network**: Predicts state values $V(s_t)$ and provides bootstrap targets.
  * **Rollout Buffer**: Stores trajectories and normalizes advantage signals.

### Speech Script
"Now, let's talk about **Phase 2: PPO Algorithm Architecture and Network Design**.

Our implementation uses a standard Actor-Critic PPO architecture, which runs in four phases:
1. **Rollout Collection**: The Actor network outputs a mean and standard deviation of continuous action distributions. We sample a raw action from a Gaussian distribution, clip it to make sure it respects G1's physical limits, apply it to MuJoCo, and store the transition.
2. **Advantage Estimation**: We read the rollout buffer and calculate Generalized Advantage Estimation (GAE). A key detail here is **Bootstrap Handling**: if a training episode is truncated because of the time limit, we use the Critic network to predict the value of the next state to avoid target bias.
3. **PPO Updates**: We shuffle the trajectory data and feed it into mini-batches. Over several epochs, we update the Actor and Critic networks using the PPO Clipped Surrogate Loss, Critic Mean Squared Error Loss, and Entropy Regularization to encourage exploration.
4. **Diagnostic Logs**: We monitor policy divergence using KL divergence, value function explained variance, and actor standard deviation to ensure stable learning.

Our core classes are: the **Actor Network** and **Critic Network** for policy and value prediction, and the **Rollout Buffer** for storing and processing on-policy trajectories."

---

## Slide 6: Four-in-One MDP to Log Mapping (Phase 2 Detail)

### Slide Content
* **Mandatory Alignment**: Math MDP $\rightarrow$ Algorithmic Logic $\rightarrow$ Code Variables $\rightarrow$ Console/CSV logs.

| Math Concept | Algorithmic Logic | Code Variable | Log Field |
| :--- | :--- | :--- | :--- |
| State $s_t$ | Physics and Task observations | `obs` / `state` | `pose_error`, `orientation_error` |
| Distribution $\pi_\theta(a \mid s_t)$ | Gaussian mean and std dev | `dist.mean` / `dist.stddev` | `actor_mean`, `actor_std` |
| Action $a_{raw}$ / $a_t$ | Sampled action & clipped action | `raw_action` / `clipped_action` | `action_sample` / `action_clipped` |
| Smoothness $r_{smooth}$ | Action change penalty | `r_smooth` | `smoothness_penalty` |
| Reward $r_t$ | Composite reward | `reward` / `total_reward` | `reward_total` |
| Value $V_\phi(s_t)$ | Expected return prediction | `value` / `value_t` | `value` / `V(s_t)` |
| Advantage $A_t$ | Normalized advantage signal | `advantages` / `adv` | `advantage` |
| Policy Loss $L_{actor}$ | Clipped surrogate loss | `actor_loss` | `actor_loss` |
| Value Loss $L_{critic}$ | Critic MSE loss | `critic_loss` | `critic_loss` |
| Entropy $\mathcal{H}$ | Exploration regularization | `entropy` | `entropy` |

### Speech Script
"To conclude this section, I want to highlight our **Four-in-One MDP to Log Mapping**. 

The professor emphasized that our code must align perfectly with mathematical models. To achieve this, we created a complete mapping table. 

For instance, the **State** $s_t$ in mathematics corresponds to the `obs` variable in our Python code, and we output the corresponding errors as `pose_error` and `orientation_error` in our CSV files. 

The **Action Distribution** $\pi$ maps to the Gaussian distribution's mean and standard deviation, which we log as `actor_mean` and `actor_std`. 

The **Clipped Action** $a_t$ maps to `clipped_action` in the code, logged as `action_clipped`. 

Similarly, our total reward $r_t$, advantage $A_t$, actor loss $L_{actor}$, critic loss $L_{critic}$, and entropy are all mapped directly to code variables and saved under corresponding names in our CSV training logs. This structure makes our reinforcement learning process completely transparent, verifiable, and easy to debug. 

This covers the setup, environment, and PPO architecture of our project. Thank you!"
