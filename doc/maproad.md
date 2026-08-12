
```text

======================= 階段一：Rollout 收集 (On-Policy 採樣與環境互動) =======================
當前狀態 s_t (包含關節狀態、目標手勢、前一步動作 a_{t-1})
  │
  ├────────────────────────────────────────────────┐
  ▼ (Actor 網路)                             ▼ (Critic 網路)
均值 μ_θ(s_t) 與標準差 σ_θ(s_t)               狀態價值估計 V_ϕ(s_t)
  │                                               │
  ▼ (高斯分佈採樣)                                 │
連續動作 a_raw ~ N(μ_θ, σ_θ)                       │
  │                                               │
  ▼ (動作安全極限剪裁)                              │
安全動作 a_t = clip(a_raw, a_min, a_max)           │
  │                                               │
  ▼ (施加動作至 MuJoCo 模擬器環境)                  │
計算平滑度懲罰與維持獎勵，獲得總獎勵 r_t              │
獲得下一狀態 s_{t+1} 與 terminated/truncated        │
  │                                               │
  ▼                                               ▼
將 (s_t, a_t, r_t, log_prob_old, V_ϕ(s_t), terminated, truncated) 存入 Buffer
                                (持續累積收集 N 步)

======================= 階段二：優勢估計與 Bootstrap 處理 =======================
從 Rollout Buffer 讀取整段軌跡數據 (States, Unclipped_Actions, Rewards, Masks...)
  │
  ▼ (Bootstrapping 處理)
計算末端狀態價值 V_ϕ(s_{N+1})：
  ├── 若 terminated = True  ──> Mask = 0，不進行未來的價值累加
  └── 若 truncated = True   ──> Mask = 1，使用 V_ϕ(s_{N+1}) 進行 Bootstrap 補充
  │
  ▼ (廣義優勢估計 GAE-λ)
計算 A_t (GAE) 與 回報目標 R_t = A_t + V_ϕ(s_t)

======================= 階段三：PPO 策略與價值網路更新 (當 Buffer 存滿時) =======================
打亂數據，拆分為多個 Mini-batch (進行 K 個 Epoch 更新)
  │
  ├────────────────────────────────────────┐
  ▼ (當前 Actor 網路)                       ▼ (當前 Critic 網路)
計算新 log_prob 與策略熵 H(π_θ)            預測新價值 V_ϕ(s_t)
  │                                        │
  ▼ (計算概率比率 r_t)                      │
r_t = exp(log_prob(a_raw) - log_prob_old(a_raw))
  │                                        │
  ▼ (優勢 Mini-batch 標準化)                │
A_norm = (A_batch - mean) / (std + 1e-8)   │
  │                                        │
  ├────────────────────────────────────────┼────────────────────────────────────────┐
  ▼ (Actor 損失計算)                        ▼ (Critic 損失計算)                      ▼ (策略熵正則)
L_actor = -min(ρ_t*A_norm,                 L_critic = (R_t - V_ϕ(s_t))²             H(π_θ)
               clip(ρ_t, 1-ε, 1+ε)*A_norm)
  │                                        │                                         │
  └───────────────────────────────────────┬─┴────────────────────────────────────────┘
                                          ▼
                          計算總體損失 (Total Loss)
                          L_total = L_actor + c₁*L_critic - c₂*H(π_θ)
                                          │
                                          ▼
                              反向傳播 (Backpropagation)
                                          │
                                          ▼ (梯度安全剪裁)
                              梯度剪裁 (L2 Norm Clipping)
                                          │
                                          ▼
                              更新 Actor (θ) 與 Critic (ϕ) 參數

======================= 階段四：診斷日誌與監控 (Diagnostics) =======================
計算並記錄以下指標至 Console / CSV / TensorBoard：
- 策略散度 (approx_kl)
- 價值預測解釋變異量 (explained_variance)
- 策略平均標準差 (mean_actor_std)
- 各項損失函數組件 (actor_loss, critic_loss, entropy)

```

---

- **下表展示了數學公式、演算法邏輯、程式碼變數與實時日誌的對應關係**

| 數學概念 (MDP Formula) | 演算法邏輯 (Algorithm Logic) | 程式碼變數 (Code Variable) | Console / CSV 日誌欄位 (Log Field) |
| :--- | :--- | :--- | :--- |
| **當前狀態** $s_t$ | 接收環境物理與任務觀測向量 | `obs` / `state` | `pose_error`, `orientation_error` |
| **動作分佈** $\pi_\theta(a \mid s_t)$ | Actor 網路預測的高斯均值與標準差 | `dist.mean` / `dist.stddev` | `actor_mean`, `actor_std` |
| **採樣動作** $a_{raw}$ | 從高斯分佈中隨機採樣探索 | `raw_action` / `action_unclipped` | `action_sample` |
| **安全動作** $a_t$ | 將採樣動作剪裁至物理安全邊界 | `clipped_action` | `action_clipped` |
| **平滑度懲罰** $r_{smooth}$ | 懲罰相鄰動作的平方差以消除抖動 | `r_smooth` / `r_smooth_delta` | `smoothness_penalty`, `smoothness_delta_penalty` |
| **總即時獎勵** $r_t$ | 多目標合成獎勵（姿態+維持+平滑） | `reward` / `total_reward` | `reward` / `reward_total` |
| **狀態價值** $V_\phi(s_t)$ | Critic 網路預測的當前期望回報 | `value` / `value_t` | `value` / `V(s_t)` |
| **Bootstrap 估計** $V(s_{t+1})$ | 下一狀態的價值預估 (區分時間截斷) | `next_value` / `next_val` | `next_value` |
| **回報目標** $R_t$ (TD Target) | 貝爾曼目標值估算 (GAE 加價值) | `returns` / `td_tgt` | `td_target` |
| **優勢估計** $A_t$ | 經過 Mini-batch 標準化後的優勢信號 | `advantages` / `adv` | `advantage` |
| **概率比率** $\rho_t(\theta)$ | 新舊策略重要性採樣比率 | `ratio` | `clip_fraction` (裁剪比例監控) |
| **策略散度** $KL(\pi_{\theta_{old}} \| \pi_\theta)$ | 監控新舊策略在更新過程中的 KL 散度 | `approx_kl` | `approx_kl`  |
| **策略損失** $L_{\text{actor}}$ | PPO Clipped Surrogate 損失 | `actor_loss` | `actor_loss` |
| **價值損失** $L_{\text{critic}}$ | 價值網路的均方誤差 (MSE) 損失 | `critic_loss` | `critic_loss` |
| **策略熵正則** $\mathcal{H}(\pi_\theta)$ | 策略的信息熵，用於鼓勵多樣化探索 | `entropy` | `entropy` |
| **總體損失** $L_{\text{total}}$ | 結合策略、價值、策略熵的加權總損失 | `loss` / `total_loss` | `total_loss` |
| **解釋變異量** $EV$ | 評估 Critic 預測準確度的指標 | `explained_var` | `explained_variance` |

- **組件職責**：
  1. **Actor Network**：接收 $s_t$，預測連續動作分佈的均值與標準差，使用高斯分佈進行採樣；在更新階段負責計算新策略的 Log 概率與策略熵，用以約束策略更新幅度並維持探索。
  2. **Critic Network**：接收 $s_t$，預測期望的狀態價值 $V(s_t)$；在環境因時間到而截斷（Truncated）時，負責預測末端狀態的 Bootstrap 價值以消除邊界偏差。
  3. **Rollout Buffer**：儲存 on-policy 軌跡數據（States, Actions, Log_probs, Rewards, Values, Terminated, Truncated），精確區分結束類型以計算正確的 GAE 優勢與 Returns，並在更新時進行優勢標準化。