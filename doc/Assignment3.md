### **DQN ASSIGNMENT: Deep Q-Network Control of the Unitree G1 Left Elbow**

**DQN 作業：Unitree G1 左肘之深度 Q 網路控制**

A Multi-Goal Reinforcement-Learning Assignment in MuJoCo and Gymnasium

**一項在 MuJoCo 與 Gymnasium 環境下的多目標強化學習作業**

* **Course / 課程**: CSCN8020 - Reinforcement Learning (強化學習)
* **Type / 類型**: Individual Assignment | 100 Marks (個人作業 | 滿分 100 分)
* **Prerequisite / 先決條件**: Unitree MuJoCo G1 Primer Workshop completed (已完成 Unitree MuJoCo G1 入門工作坊)



---

### **Assignment Goal / 作業目標**

Train a student-written PyTorch DQN to control the Unitree G1 left elbow across multiple target angles, compare it with the existing rule-based policy, evaluate success rate, and demonstrate the learned policy visually.

訓練由學生親自編寫的 PyTorch DQN 模型，以控制 Unitree G1 的左肘動作跨越多個目標角度，並與現有的基於規則策略（Rule-based policy）進行比較、評估成功率，最後以視覺化方式展示訓練好的策略。

---

### **1. Assignment Context / 1. 作業背景**

This assignment begins where the Unitree MuJoCo G1 Primer Workshop ends. Students have already inspected the raw model, implemented continuous elbow target control, validated the Gymnasium environment, and observed the live relationship among discrete actions, controller targets, actuator torques, and elbow movement.

本作業延續 Unitree MuJoCo G1 入門工作坊的結尾。學生先前已檢查過原始模型、實作連續的手肘目標控制、驗證 Gymnasium 環境，並觀察離散動作、控制器目標、致動器轉矩（torque）與手肘運動之間的即時關係。

| Stage (階段) | Purpose (目的) |
| --- | --- |
| `inspect_g1_model.py` | Inspect the raw model, joint state, actuator limits, and viewer torque controls. <br>

<br> (檢查原始模型、關節狀態、致動器極限與檢視器的轉矩控制。)

 |
| `control_single_joint.py` | Demonstrate continuous target-angle control using PD control and bias compensation. <br>

<br> (展示使用 PD 控制與偏置補償進行連續目標角度控制。)

 |
| `test_g1_elbow_env.py --render` | Validate the Gymnasium episode and success logic. <br>

<br> (驗證 Gymnasium 回合與成功判定邏輯。)

 |
| `demo_g1_elbow_env.py` | Prepare the camera and observe live actions, targets, torque values, and physical movement. <br>

<br> (準備攝影機並觀察即時動作、目標、轉矩值與物理運動。)

 |
| **DQN** | Replace the hand-written rule-based policy with a learned action-value policy. <br>

<br> (將手寫的基於規則策略替換為學習到的動作價值策略。)

 |

The assignment does not require students to redesign the MuJoCo model or the Gymnasium environment. The approved environment is the controlled experimental platform. The student task is to implement, train, evaluate, and explain a DQN agent that learns the three-action decision policy.

本作業不要求學生重新設計 MuJoCo 模型或 Gymnasium 環境。認可的環境即為受控的實驗平台。學生的任務是實作、訓練、評估並解釋一個學習三動作決策策略的 DQN 代理人（Agent）。

---

### **2. Learning Objectives / 2. 學習目標**

1. Explain why DQN is suitable for the environment's discrete three-action control problem.
(解釋為何 DQN 適合用於該環境的離散三動作控制問題。)

2. Map the four-value observation vector to three action-value estimates.
(將 4 個數值的觀察向量對映至 3 個動作價值估計值。)

3. Implement an online Q-network and a target Q-network in PyTorch.
(在 PyTorch 中實作線上 Q 網路與目標 Q 網路。)

4. Implement experience replay and mini-batch learning.
(實作經驗重放（Experience Replay）與小批次學習（Mini-batch Learning）。)

5. Apply epsilon-greedy exploration with a controlled decay schedule.
(應用帶有受控衰減排程的 $\epsilon$-greedy 探索策略。)

6. Compute Bellman targets and optimize a temporal-difference loss.
(計算貝爾曼目標值並最佳化時序差分（TD）損失。)

7. Handle Gymnasium terminated and truncated signals correctly.
(正確處理 Gymnasium 的終止（terminated）與截斷（truncated）訊號。)

8. Train the agent headlessly and keep the implementation compatible with CPU execution.
(採無介面模式（headless）訓練代理人，並確保實作內容相容於 CPU 執行。)

9. Save and reload trained model checkpoints.
(儲存並重新載入訓練好的模型檢查點。)

10. Evaluate the learned policy with exploration disabled.
(在關閉探索的情況下評估學習到的策略。)

11. Measure success rate separately from cumulative reward.
(將成功率與累積獎勵分開進行測量。)

12. Compare the learned DQN with the provided rule-based baseline.
(將學習到的 DQN 與提供的基線（Rule-based baseline）進行比較。)

13. Interpret training curves and explain the effect of exploration decay.
(解讀訓練曲線並解釋探索衰減帶來的影響。)

14. Demonstrate the learned policy in the MuJoCo viewer after training.
(訓練完成後，在 MuJoCo 檢視器中展示學習到的策略。)


---

### **3. Reinforcement-Learning Task / 3. 強化學習任務**

**Objective / 目標**: Train one DQN agent to move the fixed-base Unitree G1 left elbow to goals sampled from a limited multi-goal range and keep the elbow within the success tolerance for the required number of consecutive environment steps.

訓練一個 DQN 代理人，將固定基底的 Unitree G1 左肘移動至從特定多目標範圍中採樣的目標角度，並使手肘在連續所需的環境步數內保持在成功容許誤差範圍內。

#### **3.1 Environment / 3.1 環境**

* **Environment class / 環境類別**: `G1ElbowTargetEnv`
* **Controlled joint / 受控關節**: `left_elbow_joint`
* **Controlled actuator / 受控致動器**: left elbow (左肘)
* **Default execution mode / 預設執行模式**: headless (無介面模式)
* **Optional rendering / 可選繪製**: evaluation and demonstration only (僅用於評估與展示)
* **Low-level control / 低階控制**: approved PD controller plus MuJoCo `qfrc_bias` compensation (認可的 PD 控制器加 MuJoCo `qfrc_bias` 偏置補償)



#### **3.2 Observation / 3.2 觀察值（Observation）**

`[current elbow angle, current elbow velocity, goal angle, goal angle - current elbow angle]`

`[當前手肘角度, 當前手肘角速度, 目標角度, 目標角度 - 當前手肘角度]`

#### **3.3 Discrete Actions / 3.3 離散動作**

| Action (動作) | Meaning (意義) |
| --- | --- |
| `0` | Decrease the internal controller target (降低內部控制器目標值)

 |
| `1` | Hold the internal controller target (保持內部控制器目標值不變)

 |
| `2` | Increase the internal controller target (提高內部控制器目標值)

 |

#### **3.4 Multi-Goal Scope / 3.4 多目標範圍**

During training, goal angles must be sampled from the approved range: `[-0.8, +0.8]` rad.

訓練期間，目標角度必須從核可的範圍中採樣：`[-0.8, +0.8]` 弧度。

The final evaluation must include all four required benchmark goals: `-0.8` rad, `-0.4` rad, `+0.4` rad, `+0.8` rad. Each benchmark goal must be evaluated in five episodes, producing 20 total evaluation episodes.

最終評估必須包含所有 4 個要求的基準目標：`-0.8` rad、`-0.4` rad、`+0.4` rad、`+0.8` rad。每個基準目標必須評估 5 個回合（Episode），總共產生 20 個評估回合。

#### **3.5 Success Requirement / 3.5 成功標準**

Required threshold: at least 80% success over the 20 final evaluation episodes. This is equivalent to at least 16 successful episodes.

要求的門檻：在最終 20 個評估回合中達到至少 80% 的成功率（相當於至少 16 個成功回合）。

Evaluation must use a greedy policy with epsilon set to 0.0. Training success alone does not satisfy this requirement.

評估必須使用貪婪策略（Greedy policy），將 $\epsilon$ 設定為 0.0。僅憑訓練過程中的成功並不滿足此要求。

---

### **4. Medium-Scaffolding Model / 4. 中等鷹架模型**

Students begin from the completed primer repository. The following components are supplied and must remain compatible with the assignment:
學生從已完成的工作坊儲存庫開始。以下元件已由系統提供，且必須保持與本作業相容：

* The fixed-base G1 MuJoCo model and assets (固定基底的 G1 MuJoCo 模型與資源)
* The `G1ElbowTargetEnv` Gymnasium environment (Gymnasium 環境)
* The approved low-level PD and bias-compensation controller (認可的低階 PD 與偏置補償控制器)
* The rule-based validation policy (基於規則的驗證策略)
* Environment checking and visualization scripts (環境檢查與視覺化腳本)
* The successful multi-step success condition and reward function (成功的多步驟成功條件與獎勵函數)



Students must implement the DQN components themselves. Required student-written components are:
學生必須親自實作 DQN 的相關元件。要求學生自行編寫的元件包含：

* `ReplayBuffer` (經驗重放緩衝區)
* `QNetwork` (Q 網路)
* epsilon-greedy `select_action()` (探索動作選擇函數)
* `optimize_model()` or an equivalent optimization function (模型最佳化函數)
* target-network synchronization (目標網路同步)
* training loop (訓練迴圈)
* evaluation loop (評估迴圈)
* checkpoint saving and loading (檢查點儲存與載入)
* metrics recording and plotting (指標記錄與繪圖)



---

### **5. Required Methodology / 5. 要求的實作方法**

#### **5.1 PyTorch Network / 5.1 PyTorch 網路架構**

PyTorch is required. The minimum approved architecture is:

必須使用 PyTorch。最低核可的架構為：

* **Input / 輸入**: 4 values (4 個數值)


* **Hidden layer 1 / 隱藏層 1**: 64 units, ReLU (64 個單元，ReLU 活化函數)


* **Hidden layer 2 / 隱藏層 2**: 64 units, ReLU (64 個單元，ReLU 活化函數)


* **Output / 輸出**: 3 Q-values (3 個 Q 值)



Students may use a larger network only if they justify the change and remain within the five-hour training limit. The final layer must not apply softmax because DQN outputs unconstrained Q-values.

學生僅能在有合理理由且訓練時間不超過 5 小時限制的前提下使用更大的網路。最後一層**絕不可**使用 Softmax，因為 DQN 輸出的是不受限的 Q 值。

#### **5.2 Required DQN Elements / 5.2 必備的 DQN 元素**

1. Create an online Q-network and a separate target Q-network. (建立線上 Q 網路與獨立的目標 Q 網路。)

2. Initialize the target network from the online network. (使用線上 Q 網路初始化目標網路。)

3. Store transitions in a replay buffer. (將轉移狀態存入經驗重放緩衝區。)

4. Do not optimize until the replay buffer contains enough samples for one mini-batch. (在經驗重放緩衝區累積足夠一個小批次的樣本前，不進行最佳化。)

5. Select actions using epsilon-greedy exploration during training. (訓練期間使用 $\epsilon$-greedy 探索來選擇動作。)

6. Sample random mini-batches from replay memory. (從經驗記憶中隨機採樣小批次數據。)

7. Calculate the current Q-value for each selected action. (計算所選動作的當前 Q 值。)

8. Calculate the target using the reward, discount factor, target network, and next state. (使用獎勵、折扣因子、目標網路與下一狀態計算目標值。)

9. Prevent bootstrapping from true terminated states. (防止從真正的終止狀態（Terminated）進行自舉引導（Bootstrapping）。)

10. Handle time-limit truncation explicitly and explain the chosen treatment in the report. (明確處理時間限制截斷（Truncation），並在報告中解釋所採用的處理方式。)

11. Optimize a Huber loss or mean-squared-error loss. (最佳化 Huber 損失或均方誤差（MSE）損失。)

12. Clip gradients if necessary for stability. (若為了穩定性需要，進行梯度裁剪（Gradient Clipping）。)

13. Update the target network at the required interval. (按要求的間隔更新目標網路。)

14. Decay epsilon after each episode, subject to epsilon_min. (每個回合後衰減 $\epsilon$，且不低於 `epsilon_min`。)

15. Save the best or final trained model. (儲存最佳或最終訓練好的模型。)


#### **5.3 Required Baseline Hyperparameters / 5.3 要求的基線超參數**

| Parameter (參數) | Required baseline value (要求的基線值) |
| --- | --- |
| Discount factor, gamma ($\gamma$) | 0.95

 |
| Learning rate (學習率) | 0.001

 |
| Mini-batch size (小批次大小) | 64

 |
| Replay-buffer capacity (重放緩衝區容量) | 50,000 transitions (筆轉移資料)

 |
| Initial epsilon (初始 $\epsilon$) | 1.00

 |
| Minimum epsilon (最小 $\epsilon$) | 0.05

 |
| Baseline epsilon decay (基線 $\epsilon$ 衰減率) | 0.995 per episode (每個回合 0.995)

 |
| Target-network update (目標網路更新間隔) | Every 250 optimization steps (每 250 次最佳化步驟)

 |
| Warm-up before learning (學習前預熱階段) | At least 500 transitions (至少 500 筆轉移資料)

 |
| Maximum episode length (最大回合長度) | Environment default: 150 steps (環境預設：150 步)

 |
| Training goal range (訓練目標範圍) | `[-0.8, +0.8]` rad

 |
| Evaluation epsilon (評估時的 $\epsilon$) | 0.00

 |

#### **5.4 Training-Time Limit / 5.4 訓練時間限制**

The complete required training and comparison must finish within five hours on a student laptop. The implementation must run on CPU. CUDA may be detected and used optionally, but the submitted code must not require a GPU.

完整的訓練與比較必須在學生筆電上於 5 小時內完成。實作必須能在 CPU 上運行；程式可選擇性偵測並使用 CUDA，但提交的程式碼絕不能「強制需要」GPU。

* Run training headlessly. (以無介面模式進行訓練。)
* Record wall-clock training time for each experiment. (記錄每個實驗的實際牆上時鐘（Wall-clock）訓練時間。)
* Use an episode cap or time-based stop condition. (使用回合上限或基於時間的停止條件。)
* Do not render during training. (訓練期間請勿進行繪製/渲染。)
* Evaluation and the final video may use rendering after training is complete. (訓練完成後的評估與最終影片可使用繪製。)

---

### **6. Required Parameter Study: Exploration Decay / 6. 必備參數研究：探索衰減**

Conduct one controlled comparison. Keep every other hyperparameter and random-seed policy unchanged.

進行一項對照比較實驗。保持其他所有超參數與隨機種子策略不變。

| Configuration (設定) | Epsilon decay ($\epsilon$ 衰減) | Purpose (目的) |
| --- | --- | --- |
| **A - Baseline (基線)** | 0.995 | Longer exploration period (較長的探索期)

 |
| **B - Faster decay (快速衰減)** | 0.985 | Earlier transition toward exploitation (較早轉向利用/Exploitation)

 |

For both configurations, report:

針對兩種設定，均需回報以下項目：

* Total training episodes (總訓練回合數)
* Wall-clock training time (牆上時鐘訓練時間)
* Final epsilon (最終 $\epsilon$ 值)
* Mean cumulative reward over the final 20 training episodes (最後 20 個訓練回合的平均累積獎勵)
* Training success rate over the final 50 episodes (最後 50 個回合的訓練成功率)
* Final greedy evaluation success rate over 20 episodes (20 個回合的最終貪婪評估成功率)
* Mean evaluation reward (平均評估獎勵)
* Observations about stability, convergence, and action behaviour (關於穩定性、收斂性與動作行為的觀察)

**Recommendation / 建議**: Select the better exploration-decay setting using evidence, not only the highest single reward. Consider success rate, stability, training time, and consistency across target angles.

根據數據證據選擇較佳的探索衰減設定，而非僅憑單次最高獎勵。請考量成功率、穩定性、訓練時間以及跨目標角度的一致性。

---

### **7. Comparison with the Rule-Based Baseline / 7. 與基於規則基線的比較**

The existing `choose_rule_based_action()` implementation is the comparison baseline. Evaluate both the rule-based policy and the selected DQN policy on the same 20 benchmark episodes.

現有的 `choose_rule_based_action()` 實作為比較基線。在相同的 20 個基準回合上評估「基於規則的策略」與「獲選的 DQN 策略」。

| Metric (指標) | Rule-based policy (基於規則策略) | Selected DQN (獲選的 DQN) |
| --- | --- | --- |
| Successes/20 (成功數/20) |  |  |
| Success rate (成功率) |  |  |
| Mean cumulative reward (平均累積獎勵) |  |  |
| Mean episode length (平均回合長度) |  |  |
| Mean final absolute error (平均最終絕對誤差) |  |  |
| Main qualitative behaviour (主要定性行為表現) |  |  |

The discussion must address:

討論內容必須涵蓋：

* Which policy is more sample efficient? (哪種策略的樣本效率更高？)
* Which policy is more stable near the goal? (哪種策略在目標附近更穩定？)
* Does the DQN generalize across all four target angles? (DQN 是否能泛化至所有 4 個目標角度？)
* Does the DQN learn to use HOLD appropriately? (DQN 是否學會適當地使用 HOLD 保持動作？)
* Are there signs of oscillation or unnecessary target changes? (是否有振盪或不必要目標變更的跡象？)
* Why might a hand-written policy outperform a learned policy in this simple task? (在這項簡單任務中，為何手寫策略可能會優於學習策略？)

---

### **8. Required Experimental Workflow / 8. 要求的實驗工作流程**

1. Run the existing environment checker and rule-based baseline before training. (訓練前執行現有的環境檢查器與基於規則的基線。)

2. Set and record all random seeds used by Python, NumPy, PyTorch, and the Gymnasium environment. (設定並記錄 Python、NumPy、PyTorch 及 Gymnasium 環境使用的所有隨機種子。)

3. Implement and compile the DQN source files. (實作並編譯 DQN 原始碼檔案。)

4. Run a short smoke test to confirm replay insertion, batch sampling, action selection, and one optimization step. (執行短暫的冒煙測試，以確認重放寫入、批次採樣、動作選擇及單步最佳化皆正常。)

5. Train Configuration A headlessly. (以無介面模式訓練設定 A。)

6. Train Configuration B headlessly. (以無介面模式訓練設定 B。)

7. Save model checkpoints and metrics for both configurations. (儲存兩種設定的模型檢查點與指標數據。)

8. Evaluate both configurations greedily over 20 benchmark episodes. (使用貪婪策略在 20 個基準回合上評估兩種設定。)

9. Select the stronger DQN configuration using the required evidence. (利用數據證據選擇較優的 DQN 設定。)

10. Evaluate the rule-based baseline on the same benchmark goals. (在相同的基準目標上評估基於規則的基線。)

11. Generate plots and the comparison table. (繪製圖表並繪製比較表格。)

12. Record a short rendered video of the selected DQN after training. (訓練後錄製獲選 DQN 的短暫繪製展示影片。)

13. Write the technical report and verify reproducibility from the submitted instructions. (撰寫技術報告，並依提交的說明驗證可重複再現性。)

---

### **9. Required Outputs and Metrics / 9. 要求的產出與指標**

#### **9.1 Training Metrics / 9.1 訓練指標**

* Episode number (回合編號)
* Cumulative reward per episode (每回合累積獎勵)
* Episode success indicator (回合成功標記)
* Episode length (回合長度)
* Final absolute angle error (最終絕對角度誤差)
* Epsilon value ($\epsilon$ 值)
* Loss summary when available (損失值摘要（若有）)
* Training wall-clock time (訓練牆上時鐘時間)



#### **9.2 Required Plots / 9.2 要求的圖表**

* Raw and moving-average training reward (原始與滑動平均訓練獎勵圖)
* Training success rate using a rolling window (滾動視窗訓練成功率圖)
* Epsilon over episodes ($\epsilon$ 隨回合變化的曲線圖)
* Loss over optimization steps or episodes (損失值隨最佳化步驟或回合變化的曲線圖)
* Comparison plot or table for the two epsilon-decay configurations (兩種 $\epsilon$ 衰減設定的比較圖或表)
* Evaluation success rate by target angle (按目標角度區分的評估成功率圖)



#### **9.3 Final Evaluation Table / 9.3 最終評估表**

| Goal (目標) | Episodes (回合數) | Successes (成功數) | Success rate (成功率) | Mean reward (平均獎勵) |
| --- | --- | --- | --- | --- |
| -0.8 rad | 5 |  |  |  |
| -0.4 rad | 5 |  |  |  |
| +0.4 rad | 5 |  |  |  |
| +0.8 rad | 5 |  |  |  |
| **Overall (整體)** | **20** |  |  |  |

---

### **10. Deliverables / 10. 交付項目**

| Deliverable (交付物) | Required content (要求的內容) |
| --- | --- |
| **Python source code** <br>

<br> (Python 原始碼) | Complete student-written DQN implementation and any approved starter files that were modified. <br>

<br> (學生自行編寫的完整 DQN 實作及任何經修改的認可起始檔案。)

 |
| **Trained model** <br>

<br> (訓練好的模型) | Saved PyTorch checkpoint for the selected configuration. <br>

<br> (獲選設定所儲存的 PyTorch 檢查點檔案。)

 |
| **Metrics files** <br>

<br> (指標檔案) | CSV or equivalent structured output for both parameter configurations and final evaluation. <br>

<br> (包含兩種參數設定與最終評估結果的 CSV 或等效結構化輸出檔。)

 |
| **Plots** <br>

<br> (圖表) | All required training, exploration, loss, success, and evaluation visualizations. <br>

<br> (所有要求的訓練、探索、損失、成功率與評估視覺化圖表。)

 |
| **Technical report** <br>

<br> (技術報告) | A concise academic report, recommended length 6-10 pages excluding appendices. <br>

<br> (一份精簡的學術報告，建議長度為 6-10 頁，不含附錄。)

 |
| **Rendered video** <br>

<br> (繪製影片) | A short 2-3 minute demonstration of the selected trained policy after headless training. <br>

<br> (無介面訓練後，展示獲選策略運行的 2-3 分鐘短片。)

 |
| **README update** <br>

<br> (README 更新) | Exact commands for training, evaluation, checkpoint loading, and rendering. <br>

<br> (用於訓練、評估、載入檢查點及繪製運行的精確指令。)

 |

#### **10.1 Recommended File Structure / 10.1 建議的檔案結構**

```text
src/
  g1_rl/
    g1_elbow_env.py
  dqn/
    __init__.py
    q_network.py
    replay_buffer.py
    agent.py
    train_dqn.py
    evaluate_dqn.py
    render_dqn_policy.py
results/
  config_a/
  config_b/
models/
  selected_dqn.pt
report/
  DQN_Assignment_Report.pdf
README.md
```

#### **10.2 Video Requirements / 10.2 影片要求**
* Length: approximately 2-3 minutes (長度：約 2-3 分鐘)
* Load the saved model rather than retraining during the video (影片中應載入已儲存的模型而非重新訓練)
* Run with $\epsilon = 0.0$ (在 $\epsilon = 0.0$ 下執行)
* Show at least two different target angles (至少展示 2 個不同的目標角度)
* Show the MuJoCo viewer and relevant console metrics (顯示 MuJoCo 繪製畫面與相關的主控台數據指標)
* State whether each episode succeeded (說明每個回合是否成功)
* Do not use the rule-based policy in place of the DQN (切勿以基於規則的策略替代 DQN)

---

### **11. Technical Report Requirements / 11. 技術報告要求**
1. Introduction and connection to the G1 Primer Workshop (引言與 G1 入門工作坊之關聯)
2. Environment observation, action, reward, and success definitions (環境觀察、動作、獎勵與成功之定義)
3. Final Q-network architecture (最終 Q 網路架構)
4. Replay-buffer and target-network methodology (經驗重放緩衝區與目標網路之方法學)
5. Bellman target and loss formulation (貝爾曼目標與損失公式化)
6. Exploration strategy (探索策略)
7. Training methodology and reproducibility controls (訓練方法與可重複再現性控制)
8. Results for both epsilon-decay configurations (兩種 $\epsilon$ 衰減設定之結果)
9. Required plots and evaluation tables (必備圖表與評估表格)
10. Comparison with the rule-based baseline (與基於規則基線之比較)
11. Discussion of failures, oscillation, stability, and generalization (失敗案例、振盪、穩定性與泛化性之討論)
12. Evidence-based recommendation of the better exploration-decay setting (基於證據的較佳探索衰減設定建議)
13. Limitations and proposed future improvements (局限性與未來改進提案)

---

### **12. Technical and Academic Constraints / 12. 技術與學術限制**
* This is an individual assignment. (此為個人作業。)
* PyTorch is required. (必須使用 PyTorch。)
* The implementation must run on CPU; CUDA support is optional. (實作必須能於 CPU 上執行；CUDA 支援為可選項。)
* Training must run headlessly. (訓練過程必須以無介面模式執行。)
* The approved Gymnasium observation, actions, reward, and success condition must not be changed unless written instructor approval is obtained. (未獲得授課教師書面批准前，不得修改認可的 Gymnasium 觀察值、動作、獎勵與成功條件。)
* The Unitree files under `external/unitree_mujoco` must not be edited. (不得編輯 `external/unitree_mujoco` 下的 Unitree 檔案。)
* Do not replace DQN with another algorithm or a library implementation that hides the required DQN components. (不得將 DQN 替換為其他演算法或隱藏了要求之 DQN 元件的套件庫實作。)
* Stable-Baselines3 or equivalent turnkey RL training libraries are not permitted for the required implementation. (不允許使用 Stable-Baselines3 或類似的開箱即用 RL 訓練庫來完成核心實作。)
* Students must be able to explain every submitted DQN component. (學生必須能夠解釋所提交的每個 DQN 元件。)
* Any AI-assisted code or writing must be disclosed according to course and institutional requirements. (任何 AI 輔助的程式碼或寫作均須依課程與學校規定進行揭露。)
* The submitted video must demonstrate the learned DQN policy, not the existing rule-based policy. (提交的影片必須展示訓練好的 DQN 策略，而非現有的基於規則策略。)

---

### **13. Rubric - 100 Marks / 13. 評分標準 - 滿分 100 分**

| Criterion (配分項目) | Marks (分數) | Full-credit expectations (滿分期望標準) |
| :--- | :--- | :--- |
| **A. Environment understanding and baseline verification** <br> (環境理解與基線驗證) | 8 | Correctly validates the supplied environment and rule-based baseline; accurately explains observations, actions, targets, torque control, reward, termination, and truncation. <br> (正確驗證提供的環境與規則基線；精準解釋觀察值、動作、目標、轉矩控制、獎勵、終止與截斷。) |
| **B. Q-network and PyTorch implementation** <br> (Q 網路與 PyTorch 實作) | 12 | Correct 4-input/3-output network, appropriate activations, device handling, initialization, forward pass, and checkpoint support. <br> (正確的 4 輸入/3 輸出網路、適當的活化函數、裝置處理、初始化、前向傳播及檢查點支援。) |
| **C. Replay buffer and transition handling** <br> (經驗重放緩衝區與轉移資料處理) | 10 | Correct storage, bounded capacity, random batch sampling, tensor conversion, and handling of terminal information. <br> (正確的存儲、有界容量、隨機批次採樣、張量轉換及終止資訊處理。) |
| **D. DQN action selection and exploration** <br> (DQN 動作選擇與探索) | 10 | Correct epsilon-greedy behaviour, decay, minimum epsilon, deterministic greedy evaluation, and reproducible seeds. <br> (正確的 $\epsilon$-greedy 行為、衰減、最小 $\epsilon$、確定性貪婪評估及可重複的隨機種子。) |
| **E. Bellman update and optimization** <br> (貝爾曼更新與最佳化) | 15 | Correct selected Q-values, target-network bootstrap, terminal masking, detached targets, loss, optimizer use, and target-network update. <br> (正確選擇的 Q 值、目標網路自舉、終止遮罩、分離目標（detached targets）、損失函數、最佳化器使用及目標網路更新。) |
| **F. Training workflow and reproducibility** <br> (訓練工作流程與可重複再現性) | 10 | Headless training, CPU compatibility, time limit, metrics logging, checkpointing, clear commands, and repeatable execution. <br> (無介面訓練、CPU 相容性、時間限制、指標記錄、檢查點儲存、明確的指令與可重複執行性。) |
| **G. Required exploration-decay comparison** <br> (必備探索衰減對照比較) | 10 | Both configurations completed under controlled conditions; valid metrics and evidence-based comparison. <br> (兩種設定均在受控條件下完成；提供有效的指標數據與基於數據證據的比較。) |
| **H. Final evaluation and performance** <br> (最終評估與效能表現) | 10 | Correct 20-episode greedy evaluation across four goals. At least 80% success earns full performance credit; lower results are graded proportionally with consideration of correct methodology. <br> (跨 4 個目標進行正確的 20 回合貪婪評估。達到至少 80% 成功率可得滿分，較低者在考量方法正確性後按比例給分。) |
| **I. Rule-based versus DQN analysis** <br> (基於規則與 DQN 之分析) | 5 | Fair comparison using common goals and metrics; insightful discussion of sample efficiency, stability, and generalization. <br> (使用共同目標與指標進行公平比較；對樣本效率、穩定性及泛化性進行深入討論。) |
| **J. Report, plots, and interpretation** <br> (報告、圖表與解讀) | 7 | Complete academic report with readable plots, correct tables, technical interpretation, limitations, and recommendation. <br> (完整的學術報告，包含清晰易讀的圖表、正確的表格、技術解讀、局限性與建議。) |
| **K. Rendered video and submission quality** <br> (繪製影片與提交品質) | 3 | Clear 2-3 minute DQN demonstration, saved-model loading, multiple goals, organized files, and usable README. <br> (清晰的 2-3 分鐘 DQN 展示影片、載入已儲存模型、多目標展示、檔案組織良好且 README 實用。) |

#### **13.1 Performance Threshold Interpretation / 13.1 效能門檻解讀**
The 80% threshold is a required target, but grading will distinguish between an incorrect implementation and a correctly implemented agent that underperforms. The evaluation criterion will consider both achieved success and methodological correctness.  
80% 的門檻是要求達成的目標，但評分時會區分「不正確的實作」與「實作正確但效能未達預期」的情況。評估標準將同時考量實際取得的成功率與方法論的正確性。

| Final evaluation result (最終評估結果) | Performance interpretation (效能表現解讀) |
| :--- | :--- |
| **80-100% success (成功)** | Meets or exceeds the required target (達到或超越要求目標) |
| **60-79% success (成功)** | Partial performance credit; analysis of limitations required (獲得部分效能分數；需分析其局限性) |
| **1-59% success (成功)** | Limited performance credit; implementation and diagnosis examined closely (獲得有限效能分數；將嚴格審查實作與診斷分析) |
| **0% or invalid evaluation (0% 或無效評估)** | No performance credit; other rubric categories may still earn marks if valid (不給予效能分數；若其他評分項目有效仍可給分) |

---

### **14. Submission Checklist / 14. 提交前檢查清單**
* [ ] Environment checker passes. (環境檢查腳本通過。)
* [ ] Rule-based baseline results are recorded. (已記錄基於規則的基線結果。)
* [ ] Student-written DQN components are present. (包含學生編寫的 DQN 元件。)
* [ ] Both epsilon-decay experiments are complete. (完成兩項 $\epsilon$ 衰減實驗。)
* [ ] Training time is within five hours. (訓練時間在 5 小時內。)
* [ ] CPU execution is supported. (支援 CPU 執行。)
* [ ] Selected checkpoint loads successfully. (獲選的檢查點能成功載入。)
* [ ] Evaluation uses epsilon = 0.0. (評估時使用的 $\epsilon = 0.0$。)
* [ ] Twenty evaluation episodes are reported. (回報了 20 個評估回合的數據。)
* [ ] Overall success rate is calculated correctly. (正確計算整體成功率。)
* [ ] Rule-based and DQN policies are compared. (已比較基於規則與 DQN 的策略。)
* [ ] All required plots are included. (附上所有要求的圖表。)
* [ ] The rendered video shows the trained DQN. (展示影片呈現訓練好的 DQN。)
* [ ] README commands have been tested. (已測試過 README 中的指令。)
* [ ] The technical report and all required files are included. (包含技術報告與所有要求檔案。)

---

### **15. Final Academic Focus / 15. 最終學術重點**
The purpose of this assignment is not merely to produce a high reward. Students must demonstrate that they understand how the DQN transforms an observed robot state and goal into action values, how replay and target networks stabilize learning, how exploration affects data collection, and how a learned policy compares with a transparent rule-based controller.  
本作業的目的不僅僅是產生高額獎勵。學生必須展現其理解 DQN 如何將觀察到的機器人狀態與目標轉化為動作價值（Action values）、經驗重放與目標網路如何穩定學習過程、探索如何影響數據收集，以及學習到的策略如何與透明的基於規則控制器進行比較。

The assignment is complete only when the code, evidence, interpretation, and rendered demonstration collectively show that the submitted policy is a trained DQN operating in the approved Unitree G1 Gymnasium environment.  
只有當程式碼、數據證據、結果解讀及展示影片共同證明提交的策略是一個在核可的 Unitree G1 Gymnasium 環境中運行的訓練後 DQN 時，本作業才算完整完成。

---

### **16. Submission Requirements / 16. 提交詳細要求**
These requirements define the files, repository structure, access conditions, and Brightspace submission items that must accompany the technical work described in Sections 1-15.  
本節規定了配合第 1 至 15 節技術工作所需附帶的檔案、儲存庫結構、存取條件及 Brightspace 提交項目。

#### **16.1 Required GitHub Repository / 16.1 要求的 GitHub 儲存庫**
Create a GitHub repository with the exact name: `CSCN8020_Assignment 3`  
建立一個名稱完全相同的 GitHub 儲存庫：`CSCN8020_Assignment 3`

* Public repository URL: `[https://github.com/](https://github.com/)<USRID>/CSCN8020_Assignment 3`
* Cloneable URL ending in `.git`: `[https://github.com/](https://github.com/)<USRID>/CSCN8020_Assignment3.git`

#### **16.2 Required Repository Contents / 16.2 儲存庫必備內容**
* The completed Jupyter Notebook (已完成的 Jupyter Notebook)
* All embedded assets required by the Notebook (Notebook 所需的所有嵌入資源)
* Generated images or plots needed to reproduce the report (重現報告所需的生成圖表/圖片)
* Training and evaluation log files or structured metrics files (訓練與評估日誌檔或結構化指標檔)
* `README.md`
* `requirements.txt`
* `.gitignore`
* Student-written Python source files, saved model checkpoint, and any configuration files required by Sections 4-10 (學生編寫的 Python 原始碼、儲存的模型檢查點，以及第 4-10 節要求的任何設定檔)

#### **16.3 README.md Requirements / 16.3 README.md 規範**
Must include assignment title, student name, student ID, short project summary, environment setup instructions, exact installation and execution commands, repository file descriptions, description of student-written DQN implementation, GitHub URLs, and Python/OS environment details.  
必須包含：作業名稱、學生姓名、學號、專案簡述、環境設置說明、精確的安裝與執行指令、儲存庫檔案說明、學生撰寫的 DQN 實作說明、GitHub 網址及最終驗證使用的 Python/作業系統環境。

#### **16.4 requirements.txt Requirements / 16.4 requirements.txt 規範**
Must list packages needed to reproduce the environment (e.g., `torch`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `jupyter`, `mujoco`).  
必須列出重現環境所需的 Python 套件（例如：`torch`, `gymnasium`, `numpy`, `pandas`, `matplotlib`, `jupyter`, `mujoco` 等）。

#### **16.5 .gitignore Requirements / 16.5 .gitignore 規範**
Mandatory to exclude `.venv/`, `venv/`, `env/`, `__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`, `.DS_Store`, etc.  
強制排除虛擬環境、快取與暫存檔。儲存庫大小需確保可在一般網路條件下於 **3 分鐘內** 完成複製（Clone）。

#### **16.6 One-Page PDF Submission to Brightspace / 16.6 提交至 Brightspace 的單頁 PDF**
Submit a separate **one-page PDF** directly to Brightspace containing:  
直接上傳至 Brightspace 的獨立**單頁 PDF**，內容需包含：
* Student full name & ID (學生姓名與學號)
* Assignment title (作業名稱)
* A project summary of approximately 100 words (約 100 字的專案摘要)
* The GitHub repository URL and cloneable `.git` URL (GitHub 儲存庫網址與 Clone `.git` 網址)

#### **16.7 Brightspace Submission Requirements / 16.7 Brightspace 提交項目**
* The one-page PDF file (單頁 PDF 檔)
* The GitHub repository link & cloneable `.git` URL (GitHub 連結與 Clone 網址)
* The short rendered evaluation video, or a clearly accessible link to it (短繪製評估影片或其清晰可存取的連結)

#### **16.8 Academic Integrity and AI-Use Statement / 16.8 學術誠信與 AI 使用聲明**
AI tools are allowed for support, but submitted work must demonstrate original understanding. Students must be able to explain all submitted code and underlying principles. Submitting unexplainable code may be treated as an academic integrity violation.  
允許使用 AI 工具作為輔助，但提交的作品必須展現學生自己的理解。學生必須能夠解釋所有提交的程式碼與底層原理。提交無法解釋的程式碼可能會被視為違反學術誠信。

#### **16.9 Final Submission Validation / 16.9 最終提交驗證**
Before submitting, verify that the repo clones smoothly, requirements install cleanly, checkpoints load without retraining, plots/metrics match the report, and the PDF is exactly one page.  
提交前，請在新環境中確認儲存庫能順利 Clone、依賴套件可正常安裝、檢查點可直接載入運行、圖表指標與報告吻合，且 PDF 確實為單頁。

```