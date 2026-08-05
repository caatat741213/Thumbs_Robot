# CSCN8020 FINAL PROJECT ACTION PLAN
## From DQN to Actor-Critic for Unitree G1 Three-Digit Hand-Gesture Control in MuJoCo
**Final-project implementation plan extending Assignment 3**

---

### Team Members & Course Context
- **Team Members:** Emmanuel • Liggia • Cemil • Chao
- **Course / Context:** Reinforcement Learning Final Project — Unitree G1 in MuJoCo

**Core project objective:** Extend the validated Assignment 3 DQN pipeline into an Actor-Critic controller that learns continuous, coordinated wrist-and-hand actions for Thumbs Up, Open/Stop, and Thumbs Down. DQN remains the starting point and optional baseline; the final implementation is Actor-Critic / PPO.

*Prepared for team implementation, professor review, and final presentation*

---

## 1. Executive Summary and Required Corrections

This document is a corrected action plan for a project that evolves the completed Assignment 3 DQN work into an Actor-Critic final project. The original plan treated DQN and Actor-Critic as two equal implementations. The revised plan makes the development path explicit: reuse the tested Assignment 3 environment, training, logging, checkpoint, and evaluation structure; replace the discrete Q-learning controller with an Actor and a Critic; then demonstrate the mathematical, algorithmic, code, and console-log mapping requested by the professor.

| Issue in the earlier plan | Correction in Version 2.0 |
| :--- | :--- |
| **Project framing** | Primary deliverable is Actor-Critic / PPO. Assignment 3 DQN is the validated foundation and optional comparison baseline. |
| **Middle gesture** | Replace Medium / sideways thumb with Open / Stop. |
| **Robot hand image** | Use the real three-digit hand: two primary fingers and one thumb; do not show a five-finger human-like hand. |
| **Implementation flow** | Add the complete Actor-Critic learning cycle, including state, action, environment, reward, next state, value estimates, target, advantage, losses, backpropagation, and logging. |
| **Mathematics** | Add the finite MDP definitions, trajectory, return, value functions, advantage, TD target, PPO objective, critic loss, entropy term, and project reward equation. |
| **Evidence** | Specify console / CSV fields that connect each mathematical quantity to the implementation. |

### 1.1 Research question
How can the DQN control pipeline developed in Assignment 3 be evolved into an Actor-Critic controller for coordinated Unitree G1 three-digit hand gestures, and how effectively does the learned policy achieve accurate, stable, smooth, and safe motion?

### 1.2 Scope decision
- **Primary algorithm:** PPO-style Actor-Critic with continuous actions.
- **Baseline:** Assignment 3 DQN code and results. A direct DQN rerun on the new hand environment is optional unless the professor explicitly requires a controlled comparison.
- **Platform:** Fixed-base Unitree G1 in MuJoCo, one hand only, with the shoulder and elbow held in a scripted presentation pose.
- **Task:** Target-conditioned execution of three gestures from randomized safe starting configurations.

從作業 3 DQN 演化至期末 Actor-Critic / PPO 專案

作業 3 - DQN 基準線  
* 任務： 單關節肘部設定點控制  
* 動作： 離散動作索引  
* 策略： 基於 $Q(s,a)$ 的 $\epsilon$-greedy  
* 價值模型： Online Q 網路 + Target Q 網路  
* 經驗： 經驗重放池（Replay buffer）；Off-policy 小批次  
* 學習目標： $r + \gamma \max Q(s',a')$  
* 損失： Huber / 平方時序差分誤差（TD error）  
* 可複用基礎： MuJoCo + Gymnasium 架構；日誌；檢查點；評估與渲染

演進與改變（Evolve / Reuse / Replace）  
* 複用： 環境 API、日誌記錄、檢查點機制。  
* 替換/新增： Q 網路 $\rightarrow$ Actor + Critic；$\epsilon$-greedy $\rightarrow$ 策略採樣；Replay $\rightarrow$ Rollouts。  

期末專案 - Actor-Critic / PPO  
* 任務： 協調手腕 + 三指手勢  
* 動作： 所有受控關節的連續向量  
* 策略： Actor $\pi_\theta(a\vert{}s)$  
* 價值模型： Critic $V_\phi(s)$  
* 經驗： Rollout 緩衝區；On-policy 軌跡  
* 學習目標： Return / TD target + 優勢估計（Advantage estimate）  
* 損失： PPO Actor 損失 + Critic 損失 - 熵（Entropy）  
* 新實證數據： Actor 均值/動作；價值（Values）；優勢（Advantages）；組件獎勵；Actor/Critic 損失


---

## 2. Corrected Robot Hand and Gesture Definitions

The MuJoCo implementation must use the real three-digit hand: two primary fingers and one thumb. All target poses, actions, rewards, screenshots, and videos must match that morphology.

> **Physical-model rule:**
> All diagrams must show two fingers plus one thumb. Phase 0 must verify the exact MuJoCo joints, actuators, limits, and active degrees of freedom before the observation and action vectors are fixed.
> *Do not use five-finger or human-hand renders as technical evidence.*

### 2.1 Final gesture set

| Gesture | Required three-digit configuration | Main controlled elements |
| :--- | :--- | :--- |
| **Thumbs Up** | Two primary fingers flexed; thumb extended; wrist orients the thumb upward. | Finger 1, Finger 2, thumb, wrist orientation |
| **Open / Stop** | Two primary fingers and thumb extended; palm facing forward; wrist held stable. | Finger 1, Finger 2, thumb, wrist roll/pitch/yaw |
| **Thumbs Down** | Two primary fingers flexed; thumb extended; wrist orients the thumb downward. | Finger 1, Finger 2, thumb, wrist orientation |

### 2.2 Target-pose calibration
1. Load the exact fixed-base Unitree G1 model; print joint / actuator names, limits, control ranges, and transmission types.
2. Identify wrist axes and independently actuated thumb / finger joints; record the controlled dimension $N$.
3. Use a manual slider or scripted controller to verify all three gestures and calibrate their target vectors.
4. Save each vector, screenshot, and short render; use only these verified values in reward and success logic.

*Engineering checkpoint: Open / Stop intentionally changes the finger configuration. This prevents the task from collapsing into wrist-only setpoint tracking and makes the Actor-Critic controller coordinate multiple joints.*

---

## 3. Finite MDP Formulation and Mathematical Foundation

The hand-control environment is modeled as a finite-horizon episodic Markov Decision Process. The process repeats until the selected gesture is held successfully or the maximum step limit is reached.

| Symbol | Project interpretation |
| :--- | :--- |
| $s_t \in \mathcal{S}$ | Normalized state at time $t$: current joint positions and velocities, target pose, pose error, target gesture, and previous action. |
| $a_t \in \mathcal{A}(s_t)$ | Continuous wrist-and-digit action selected by the Actor and clipped to safe bounds. |
| $r_t \in \mathbb{R}$ | Scalar reward measuring progress, final pose accuracy, stability, smoothness, speed, and safety. |
| $p(s_{t+1}, r_t \mid s_t, a_t)$ | MuJoCo transition: probability / dynamics of the next state and reward given the current state and applied action. |
| $\pi_	heta(a\mid s)$ | Actor policy: state-dependent probability distribution over continuous actions. |
| $V_\phi(s)$ | Critic value: expected discounted return from state $s$ under the current policy. |

### 3.1 Trajectory, return, and objective
$$	τ = (s_0, a_0, r_1, s_1, a_1, r_2, \dots, s_T)$$
$$G_t = \sum_{k=t+1}^{T} \gamma^{k-t-1} r_k$$
$$	\text{Goal: maximize} J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[G_0]$$

A trajectory records one episode. The discounted return $G_t$ summarizes future rewards. The policy objective is to maximize expected return over many trajectories and randomized initial configurations.

### 3.2 Value, action-value, and advantage
$$V^\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$$
$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$$
$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

The Critic approximates $V^\pi(s)$. The advantage estimates whether an observed action produced a better or worse result than the Critic expected, and it supplies the learning signal for the Actor.

### 3.3 One-step temporal-difference target
$$y_t = r_t + \gamma (1 - d_t) V_\phi(s_{t+1})$$
$$\delta_t = y_t - V_\phi(s_t)$$
$$\hat{A}_t = \delta_t \quad \text{simple TD} \quad \text{or} \quad \hat{A}_t = \text{GAE}(\delta_t, \gamma, \lambda)$$

Here $d_t$ is 1 for a true terminal state and 0 otherwise. Time-limit truncation should be handled separately so that the implementation does not incorrectly remove bootstrapping.

### 3.4 PPO Actor-Critic losses

* 機率比率： $$ho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
* Actor 損失： $$L_{\text{actor}} = -\mathbb{E} \left[ \min\left( \rho_t \hat{A}_t, \, \text{clip}(\rho_t, 1-\varepsilon, 1+\varepsilon) \hat{A}_t \right) \right]$$
* Critic 損失：$$L_{\text{critic}} = \mathbb{E} \left[ (V_\phi(s_t) - \hat{R}_t)^2 \right]$$
* 總損失：$$L_{\text{total}} = L_{\text{actor}} + c_v L_{\text{critic}} - c_e \mathcal{H}(\pi_\theta(\cdot \mid s_t))$$

The clipped Actor objective limits abrupt policy changes. The Critic loss trains the value estimate. The entropy term supports exploration early in training.

*Connection to the professor’s DQN cycle: DQN uses $\max Q(s',a')$ in the Bellman target. Actor-Critic replaces that discrete max operation with a learned value $V(s')$, an advantage estimate, and separate Actor and Critic updates.*

---

## 4. Actor-Critic Learning Cycle and Implementation Flow

The following cycle is the Actor-Critic equivalent of the complete DQN learning cycle supplied by the professor. It should be used as the structure for the implementation, the presentation, and the diagnostic log.

期末專案的完整 Actor-Critic / PPO 學習循環

1. 當前狀態 $s_t$： 關節位置 + 速度、關節誤差 + 目標手勢、前一動作。  
2. Actor $\pi_\theta(a\vert{}s_t)$： 輸出連續動作分佈的均值 $\mu_\theta(s_t)$ 與標準差 $\sigma_\theta(s_t)$。  
3. 選擇並縮放動作 $a_t$： 訓練時進行採樣；評估時使用均值；剪裁（clip）至安全的手腕/手指極限。  
4. MuJoCo 環境： 施加致動器目標並推進模擬時間步。  
5. 觀察轉移： 取得獎勵 $r_t$、下一狀態 $s_{t+1}$、完成標記 $d_t$，並將數值寫入控制台 / CSV 日誌。  
6. Critic 價值： 計算 $V_\phi(s_t)$ 與 $V_\phi(s_{t+1})$。  
7. TD 目標與優勢： $y_t = r_t + \gamma(1-d_t)V_\phi(s_{t+1})$ 以及 $\hat{A}_t = y_t - V_\phi(s_t)$ (或 GAE)。  
8. 更新 Actor 與 Critic： Actor（PPO 剪裁目標）、Critic（價值迴歸）、熵（維持探索）。  
9. 反向傳播： 更新 $\theta$ 與 $\phi$；重複進行直到達到終止狀態。

### 4.1 Episode execution flow
1. Reset all controlled wrist and digit joints to randomized safe values.
2. Randomly select Thumbs Up, Open/Stop, or Thumbs Down and attach the corresponding verified target vector.
3. Construct and normalize $s_t$.
4. Run the Actor to obtain a continuous action distribution; sample during training and use the mean during deterministic evaluation.
5. Scale and clip the action to safe joint-target increments or actuator commands.
6. Step MuJoCo; obtain $r_t$, $s_{t+1}$, `terminated`, `truncated`, and reward components.
7. Use the Critic to calculate $V(s_t)$ and $V(s_{t+1})$; calculate TD target, return, and advantage.
8. Write the state, action, rewards, values, advantage, and losses to the log.
9. Update the Actor and Critic after the required rollout length; repeat until success or the episode limit.

---

## 5. Mapping Mathematics to Algorithm, Code, and Logs

The final presentation should not show equations in isolation. Every mathematical quantity should be linked to the algorithm step, the relevant code variable or function, and visible console / CSV evidence.

### 5.1 State and continuous action design
$$s_t = [q_t, \dot{q}_t, q_{	ext{target}}(g), q_{	ext{target}}(g)-q_t, 	ext{one\_hot}(g), a_{t-1}]$$
$$a_t = [\Delta q_{	ext{wrist\_1}}, \dots, \Delta q_{	ext{wrist\_m}}, \Delta q_{	ext{thumb}}, \Delta q_{	ext{finger\_1}}, \Delta q_{	ext{finger\_2}}]$$

The exact dimensions $m$ and $N$ depend on the audited MuJoCo model. If the actuator interface uses torque rather than target increments, the same conceptual vector applies but the scaling and safety constraints must be changed accordingly.

### 5.2 Mathematics-to-code mapping

| Mathematics / concept | Algorithm or function | Suggested code variable | Required evidence |
| :--- | :--- | :--- | :--- |
| **Current state $s_t$** | `env._get_obs()` | `obs` / `state` | Print normalized state summary, target ID, pose error |
| **Actor policy $\pi_	heta(a\mid s)$** | `actor(obs)` | `action_dist`, `mu`, `std` | Print actor mean / std and sampled action |
| **Selected action $a_t$** | `dist.sample()`; `scale_action()` | `raw_action`, `clipped_action` | Print raw, scaled, and clipped vectors |
| **Environment transition** | `env.step(action)` | `next_obs`, `reward`, `terminated`, `truncated`, `info` | Print next-state error and termination flags |
| **Reward $r_t$** | `compute_reward()` | `reward_total`, `reward_components` | Print progress, pose, orientation, smoothness, limits, bonuses |
| **Critic $V_\phi(s)$** | `critic(obs)` | `value_t`, `next_value` | Print $V(s_t)$ and $V(s_{t+1})$ |
| **TD target $y_t$** | $r + \gamma(1-d)V(s')$ | `td_target` | Print target value |
| **Advantage $\hat{A}_t$** | GAE or `td_target - value_t` | `advantage` | Print raw and normalized advantage |
| **Actor loss** | `ppo_policy_loss()` | `actor_loss`, `ratio`, `clip_fraction` | Print ratio statistics and actor loss |
| **Critic loss** | `value_loss()` | `critic_loss` | Print value loss and explained variance |
| **Entropy** | `dist.entropy()` | `entropy` | Print entropy / policy standard deviation |
| **Update** | `optimizer.step()` | `grad_norm`, `learning_rate` | Print actor / critic gradient norm and learning rate |

### 5.3 Example diagnostic log format

```text
[episode=018 step=042 gesture=OPEN_STOP]
state_error_norm=0.214  actor_mean=[0.18,-0.06,0.31,0.42,0.39]  actor_std=[0.21,...]
action_sample=[0.22,-0.10,0.28,0.51,0.33]  action_clipped=[0.22,-0.10,0.28,0.50,0.33]
reward_total=1.84  progress=0.62  pose=-0.21  orientation=-0.08  smoothness=-0.03  hold=1.50
V(s_t)=3.210  V(s_t+1)=3.480  td_target=4.286  advantage=1.076
actor_loss=-0.032  critic_loss=0.579  entropy=1.204  clip_fraction=0.08
```

### 5.4 Minimum log files
- `step_log.csv`: One row per environment step with state summaries, action values, reward components, next-state error, value, target, and advantage.
- `update_log.csv`: One row per optimizer update with actor loss, critic loss, entropy, KL divergence, clip fraction, explained variance, gradient norms, and learning rates.
- `episode_log.csv`: Gesture, total return, success, steps to success, final pose error, final orientation error, hold duration, and safety violations.
- `config.json` or `YAML`: Random seed, model asset, controlled joints, action scaling, reward weights, PPO hyperparameters, and success thresholds.

---

## 6. Environment, Reward, and Success Criteria

### 6.1 Shared environment logic
- Fixed-base robot; one hand controlled; shoulder and elbow held in a scripted pose.
- Randomized safe start across every controlled wrist and digit joint.
- Target-conditioned policy: one Actor-Critic model receives the selected gesture as part of the observation.
- Normalized observations and actions; hard clipping to verified joint / actuator limits.
- Episode terminates after a successful hold; time-limit expiration is recorded as truncation.

### 6.2 Project reward equation
$$e_t = \text{weighted pose-and-orientation error at step t} $$
$$r_t = w_p(e_{t-1}-e_t) - w_h E_{\text{hand}} - w_o E_{\text{orientation}} - w_v \Vert{}\dot{q}_t\Vert{}^2 - w_a \Vert{}a_t\Vert{}^2 - w_s \Vert{}a_t-a_{t-1}\Vert{}^2 - w_j P_{\text{joint\_limits}} + b_{\text{hold}} I_{\text{hold}} + b_{\text{success}} I_{\text{success}} - c_{\text{time}}$$

| Reward component | Purpose |
| :--- | :--- |
| **Progress $w_p(e_{t-1}-e_t)$** | Positive signal whenever the hand moves closer to the selected target. |
| **Hand-pose error $E_{	ext{hand}}$** | Matches thumb and both primary fingers to the target configuration. |
| **Orientation error $E_{	ext{orientation}}$** | Aligns the palm / wrist for Up, Stop, or Down. |
| **Velocity and smoothness penalties** | Reduce shaking, oscillation, and abrupt command changes. |
| **Action penalty** | Discourages unnecessarily large commands. |
| **Joint-limit penalty** | Penalizes unsafe or invalid configurations. |
| **Hold bonus** | Rewards maintaining the correct gesture for consecutive steps. |
| **Success bonus** | Provides a clear terminal reward after the required hold. |
| **Time penalty** | Encourages efficient completion. |

### 6.3 Initial success criteria
- Mean absolute error for the thumb and both primary fingers below a calibrated threshold, initially around 0.05 radians only as a starting point.
- Wrist / palm orientation error below a calibrated threshold, initially around 10-15 degrees only as a starting point.
- Both conditions held for 10-20 consecutive simulation steps.
- No actuator saturation, joint-limit violation, NaN, or unstable motion during the successful hold.
- Final thresholds selected from manual-controller tests so that success is visually correct and learnable.
- **Stop validation:** For Open / Stop, the two primary fingers and thumb must be visibly extended and the palm must face the viewer. A wrist-only pose with curled fingers is not valid.

---

## 7. Actor-Critic / PPO Implementation Plan

| Component | Role in the final project |
| :--- | :--- |
| **Actor** | Receives $s_t$ and outputs the mean and standard deviation of a continuous action distribution for all controlled joints. |
| **Critic** | Receives $s_t$ and estimates $V(s_t)$, the expected discounted return from the current state. |
| **Rollout buffer** | Stores current on-policy trajectories: states, actions, log probabilities, rewards, done flags, values, returns, and advantages. |
| **GAE** | Computes lower-variance advantage estimates using $\gamma$ and $\lambda$. |
| **PPO clipping** | Limits how far the new policy moves from the behavior policy during each update. |
| **Entropy regularization** | Prevents exploration from collapsing too early. |
| **Gradient clipping** | Reduces unstable updates. |
| **Checkpointing** | Saves policy / value weights, optimizers, normalization statistics, configuration, and training step. |

### 7.1 Reuse from Assignment 3
- Repository layout and reproducible environment setup.
- MuJoCo loading, fixed-base configuration, Gymnasium reset / step API, rendering, and seed management.
- Evaluation scripts, metric plotting, checkpoint naming, and final demonstration workflow.
- Existing output values and log conventions where they remain meaningful.

### 7.2 Replace or add
- Replace the online / target Q-networks and epsilon-greedy selection with an Actor, a Critic, and a continuous policy distribution.
- Replace replay-buffer Q-learning with on-policy rollouts, returns, TD values, and advantage estimates.
- Add continuous-action scaling, per-joint clipping, and verified safety bounds.
- Add PPO ratio, clipping, entropy, KL, explained variance, and actor / critic gradient logging.

### 7.3 Recommended starting hyperparameters

| Parameter | Starting value / range | Reason |
| :--- | :--- | :--- |
| $\gamma$ | 0.99 | Values future pose and hold rewards. |
| GAE $\lambda$ | 0.95 | Balances variance and bias. |
| PPO clip $ arepsilon$ | 0.2 | Common conservative update bound. |
| Learning rate | 3e-4; tune separately if needed | Stable starting point for Actor-Critic. |
| Rollout length | 512-2048 environment steps | Enough temporal diversity before updates. |
| Mini-batch size | 64-256 | Depends on rollout length and hardware. |
| Epochs / rollout | 5-10 | Multiple updates without excessive policy drift. |
| Entropy coefficient | 0.001-0.02 | Tune based on exploration collapse. |
| Gradient norm | 0.5-1.0 | Controls unstable gradients. |

*These are initial values, not final claims. The report must record all changes and explain them using learning curves and diagnostic logs.*

---

## 8. Implementation Roadmap and Team Exit Conditions

| Phase | Main tasks | Exit condition |
| :--- | :--- | :--- |
| **0. Model audit** | Load exact asset; print joints / actuators; verify two-finger-plus-thumb morphology; determine $N$; calibrate three target poses. | All gestures render correctly under manual control; target vectors and screenshots saved. |
| **1. Environment migration** | Adapt Assignment 3 environment to hand / wrist control; add gesture conditioning, random starts, reward components, success and safety logic. | Scripted controller reaches each target; unit tests and bounds checks pass. |
| **2. Actor-Critic skeleton** | Implement Actor, Critic, rollout buffer, action distribution, returns, GAE, PPO losses, checkpointing. | Short run completes without NaNs; tensor shapes and gradients are valid. |
| **3. Logging and math mapping** | Add step, update, and episode logs; print values from the complete cycle; connect variables to equations. | A single episode and optimizer update can be explained line by line from log to math to code. |
| **4. Gesture-by-gesture training** | Train and debug each gesture separately, beginning with Open / Stop or the easiest calibrated pose. | Each individual gesture reaches a stable success threshold. |
| **5. Unified target-conditioned policy** | Train one policy on all three gestures from randomized safe initial hand configurations. | Stable checkpoint with acceptable per-gesture success and no systematic failure. |
| **6. Evaluation** | Run deterministic fixed and randomized tests; calculate accuracy, speed, smoothness, stability, safety, and robustness. | Reproducible tables, plots, videos, and selected checkpoint. |
| **7. Presentation and repository** | Prepare diagrams, code mapping, logs, equations, demo video, README, environment file, and final report. | Every design choice is reproducible and each team member can explain the complete learning cycle. |

### 8.1 Suggested team allocation

| Workstream | Main responsibilities |
| :--- | :--- |
| **Model / environment** | Asset audit, target calibration, action bounds, random reset, reward and success logic. |
| **Actor-Critic algorithm** | Networks, distribution, rollout buffer, GAE, PPO update, optimization stability. |
| **Logging / analysis** | Console output, CSV schemas, plots, metrics, mathematical mapping, experiment tracking. |
| **Evaluation / presentation** | Deterministic tests, videos, comparison to Assignment 3, report integration, slides and rehearsal. |

### 8.2 Minimum viable final project
- Verified fixed-base three-digit Unitree hand and the three gestures: Thumbs Up, Open / Stop, and Thumbs Down.
- One trained target-conditioned Actor-Critic / PPO checkpoint with complete mathematical and logging evidence for at least one update.
- Per-gesture and overall evaluation results plus a rendered demonstration.
- Reproducible repository, configuration, dependencies, and execution commands.

---

## 9. Evaluation Metrics, Outputs, and Risks

| Metric | Interpretation |
| :--- | :--- |
| **Overall and per-gesture success rate** | Whether the policy completes the full task and whether one gesture is systematically harder. |
| **Return and reward components** | Whether improvement comes from pose accuracy, progress, holding, smoothness, or unintended reward exploitation. |
| **Steps / time to success** | Efficiency of gesture formation. |
| **Final digit-pose error** | Accuracy of thumb and two primary fingers. |
| **Final wrist / palm orientation error** | Directional correctness of Up, Stop, and Down. |
| **Hold stability** | Whether the gesture remains correct without oscillation. |
| **Action smoothness** | Mean squared action change, joint acceleration, or jerk. |
| **Joint-limit / saturation violations** | Safety and validity of the controller. |
| **Training environment steps** | Sample efficiency. |
| **Wall-clock time** | Practical computation cost. |
| **Robustness** | Success from unseen randomized safe initial configurations. |

### 9.1 Required deliverables
- Trained Actor-Critic / PPO checkpoint and deterministic evaluation script.
- Three verified target vectors and screenshots using the real three-digit hand.
- Training curves: return, success, actor loss, critic loss, entropy, KL, clip fraction, and explained variance.
- Evaluation table by gesture and overall.
- Step, update, and episode logs showing the complete math-algorithm-code mapping.
- Rendered video containing Thumbs Up, Open / Stop, and Thumbs Down.
- Technical report, README, environment / dependency file, configuration file, and reproducible commands.

### 9.2 Main risks and controls

| Risk | Control |
| :--- | :--- |
| **Wrong MuJoCo hand asset or rigid fingers** | Complete Phase 0 first; do not write the final action space before the joint audit. |
| **Stop gesture is visually incorrect** | Require all three digits extended and palm forward in manual and learned validation. |
| **Reward exploitation** | Plot each reward component and inspect rendered episodes at checkpoints. |
| **Policy collapse / low exploration** | Monitor entropy and action standard deviation; tune entropy coefficient and learning rate. |
| **Critic instability** | Normalize returns / advantages, clip gradients, monitor explained variance and value loss. |
| **Unsafe or saturated actions** | Scale actions conservatively, clip to verified limits, and terminate or penalize persistent violations. |
| **Results not reproducible** | Save seeds, full configuration, normalization statistics, commit hash, dependency versions, and exact commands. |

*Final acceptance rule: A model is not considered successful only because total reward increases. It must render the correct three-digit gesture, satisfy calibrated pose and orientation thresholds, hold the pose stably, and avoid safety violations.*

---

## 10. Presentation Structure and Technical References

### 10.1 Recommended final presentation flow
1. **Assignment 3 recap:** DQN elbow-control problem, state, discrete actions, Q-values, Bellman target, and reusable code.
2. **Problem evolution:** Why multi-joint three-digit hand control requires a continuous policy.
3. **Corrected physical asset and three gestures:** Thumbs Up, Open / Stop, Thumbs Down.
4. **Finite MDP formulation and project reward.**
5. **Actor-Critic learning cycle and the DQN-to-Actor-Critic mathematical mapping.**
6. **Code architecture:** Environment, Actor, Critic, rollout buffer, PPO update, logging.
7. **One complete console-log example mapped to equations and code variables.**
8. **Training curves, deterministic evaluation metrics, and rendered gestures.**
9. **Limitations, risks, and next improvements.**

### 10.2 Technical references
1. **Team Assignment 3 repository:** [https://github.com/chooksemmanuel/CSCN8020_Assignment3](https://github.com/chooksemmanuel/CSCN8020_Assignment3) — Existing DQN environment, training, evaluation, checkpoint, and reporting structure. 
2. **Unitree G1 product information:** [https://www.unitree.com/g1/](https://www.unitree.com/g1/) — Robot platform and hand options. The exact MuJoCo asset remains the implementation source of truth. 
3. **Unitree MuJoCo repository:** [https://github.com/unitreerobotics/unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco) — Official simulation assets and examples. 
4. **Proximal Policy Optimization Algorithms:** [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) — PPO clipped policy objective and Actor-Critic implementation basis. 
5. **Reinforcement Learning: An Introduction:** [http://incompleteideas.net/book/the-book-2nd.html](http://incompleteideas.net/book/the-book-2nd.html) — Finite MDP, return, value functions, policy gradients, and Actor-Critic foundations. 
6. **Course guidance from David:** Complete DQN learning cycle and requirement to connect mathematics, algorithm, code, and console values; adapted in this plan to Actor-Critic. 