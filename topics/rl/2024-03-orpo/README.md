<!--
ORPO (Odds Ratio Preference Optimization) 論文解讀
撰寫日期: 2026-05-21
撰寫方式: Hermes Agent 自主執行
-->

# ORPO (Odds Ratio Preference Optimization): 無需 Reference Model 的單階段偏好對齊方法

> **種子論文**: [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691) (2024-03)
> **作者**: Jiwoo Hong, Noah Lee, James Thorne et al.
> **機構**: KAIST AI

---

## TL;DR

ORPO（Odds Ratio Preference Optimization）想解決現有偏好對齊方法需要多階段訓練（SFT warm-up → alignment）且依賴 reference model 的問題。它將 odds ratio 懲罰項直接附加在 SFT 的 negative log-likelihood loss 上，讓模型在單一階段同時完成領域適應與偏好對齊，完全不需要額外的 reference model 或 RL 訓練循環。在 Phi-2 (2.7B)、Llama-2 (7B) 和 Mistral (7B) 上，只用 UltraFeedback 資料集訓練一個 epoch，就超越了 Llama-2 Chat (13B) 甚至 Llama-2 Chat (70B) 的表現。

---

## 背景與動機

### 偏好對齊的兩階段困境

從 GPT-3 到 ChatGPT，語言模型的能力有了跳躍式成長。但這些模型在大量無監督網頁資料上訓練時，學到的不只人類想要的「有用、安全、準確」的輸出風格，也學到了各種不理想的生成模式——偏見、幻覺、無助益的回應。

為了解決這個問題，研究社群提出了**偏好對齊（preference alignment）**的核心想法：收集人類對模型輸出的偏好評估，然後根據這些偏好來調整模型的行為。

下圖展示了三種偏好對齊方法的訓練流程差異——從 RLHF 到 DPO 再到 ORPO，管線越來越簡潔：

```mermaid
flowchart TB
    subgraph RLHF["RLHF（三階段）"]
        A1[Pre-trained LM] --> B1[SFT]
        B1 --> C1[Reward Model Training]
        C1 --> D1[RLHF / PPO Fine-tuning]
        D1 --> E1[Aligned Model]
        
        R1[Reward Model] -.-> D1
        Ref1[Reference Model<br/>π_ref = π_SFT<br/>frozen] -.-> D1
    end

    subgraph DPO["DPO（兩階段）"]
        A2[Pre-trained LM] --> B2[SFT warm-up]
        B2 --> C2[DPO Training<br/>L_DPO = -log σ(β log π/π_ref)]
        C2 --> D2[Aligned Model]
        
        Ref2[Reference Model<br/>π_ref = π_SFT<br/>frozen] -.-> C2
    end

    subgraph ORPO["ORPO（單階段）"]
        A3[Pre-trained LM] --> B3[ORPO<br/>L_ORPO = L_SFT + λ·L_OR]
        B3 --> C3[Aligned Model]
    end

    style RLHF fill:#1a1a2e,stroke:#e94560,color:#fff
    style DPO fill:#16213e,stroke:#0f3460,color:#fff
    style ORPO fill:#1a1a2e,stroke:#16c79a,color:#fff
```



然而，RLHF 的訓練流程相當繁瑣，通常包含三個階段：

1. **SFT（Supervised Fine-Tuning）**：先在高品質的指令資料上做監督式微調，讓模型學會指令跟隨的基本能力
2. **Reward Modeling**：訓練一個獨立的 reward model 來評估生成品質
3. **RL Fine-Tuning**：用 PPO 等強化學習演算法，以 reward model 的分數為獎勵訊號來微調語言模型，同時用 KL 散度約束防止模型偏離太多

這個多階段流程有幾個根本問題：

- **訓練不穩定**：PPO 對超參數高度敏感，學習率、KL 係數、batch size 等都需要仔細調整
- **計算成本高**：需要同時維護 policy model、reference model、reward model 三組參數，每次更新需要多次 forward/backward pass
- **reward hacking**：policy model 可能學到「欺騙」reward model 的捷徑，而非真正的偏好理解
- **reference model 的必要性**：為了防止 mode collapse，必須保留一個凍結的 reference model 來計算 KL 散度

### DPO 的突破與遺留問題

2023 年，Rafailov 等人提出了 **DPO（Direct Preference Optimization）**，從根本上改變了偏好對齊的範式。DPO 的核心洞見是：RLHF 的 KL-constrained reward maximization 目標存在一個 closed-form 的最優策略解，而這個解可以透過變數替換轉化為一個簡單的二元分類損失函數。

DPO 的成功在於去掉了獨立訓練 reward model 和 RL 循環的步驟，大大簡化了訓練流程。但它仍然保留了兩個前提條件：

1. **需要一個 reference model**：DPO 的損失函數中，policy model 的輸出機率必須相對於一個凍結的 reference model 來計算，即 `log(π_θ(y|x) / π_ref(y|x))` 的形式
2. **需要先做 SFT**：reference model 通常是 SFT 後的模型，而且論文中建議直接用 π_SFT 作為 π_ref

換句話說，DPO 雖然去掉了 RL 階段，但仍然是兩階段流程：SFT → DPO alignment。

### ORPO 要解決的核心問題

ORPO 的作者從一個更根本的角度來看待這個問題：**SFT 階段本身為什麼不能同時做偏好對齊？**

當我們用 cross-entropy loss 做 SFT 時，模型會學會提高 chosen responses 中每一個 token 的機率。但 cross-entropy loss 對非答案 token（包含 rejected responses 中的 token）不施加任何懲罰——它只關心選中的 token 是否被正確預測。結果是，chosen 和 rejected responses 的 log probability 會隨著訓練同步上升。

這意味著傳統的 SFT 在適應領域的同時，也無差別地增強了所有回應風格（包含不理想的風格）的生成可能性。如果能在 SFT 階段引入一個機制，讓模型在適應領域的同時也學會區分偏好與非偏好的生成風格，那就不需要額外的 alignment 階段了。

這就是 ORPO 的核心動機。

---

## 核心知識點

本文圍繞以下知識點展開：

1. **SFT 在偏好對齊中的雙面性**——為什麼 cross-entropy loss 無法區分 chosen 與 rejected responses
2. **DPO 的理論基礎**——作為對比基準，DPO 如何從 RLHF 推導出 closed-form 偏好損失
3. **Odds Ratio 的定義與直覺**——為什麼 `P/(1-P)` 比 `P` 更適合做偏好對比的度量
4. **ORPO 目標函數與梯度分析**——SFT loss + odds ratio penalty 的具體形式與行為
5. **Odds Ratio vs Probability Ratio 的理論比較**——為什麼 odds ratio 在 SFT+alignment 設定中更穩定
6. **λ 超參數的角色與消融**——權重係數對 chosen/rejected 分離程度的影響

---

## 方法詳解

### 知識點 1: SFT 在偏好對齊中的雙面性

**這個知識點要回答什麼問題？為什麼我們不能只用 SFT 來對齊模型？**

SFT 的損失函數是標準的 cross-entropy（負對數似然）：

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{m} \sum_{k=1}^{m} \sum_{i=1}^{|V|} y_i^{(k)} \log(p_i^{(k)})
$$

其中 $m$ 是序列長度，$y_i$ 是第 $i$ 個 token 是否為標籤 token 的指示變數，$p_i$ 是第 $i$ 個 token 的預測機率。

這個損失函數的問題在於：**它對非答案 token 完全沒有懲罰或補償機制**。當 $y_i = 0$ 時，該 token 對 loss 沒有任何貢獻。這表示在 SFT 過程中，模型會提高 chosen response 中 token 的機率，但對於 rejected response 中的 token，它們的機率同樣會因為模型對領域的適應而上升。

**ORPO 論文的核心實驗**證明了這個現象：他們將 OPT-350M 僅在 HH-RLHF 資料集的 chosen responses 上做 SFT，並監控 rejected responses 的 log probability。結果如圖 3 所示，chosen 和 rejected 的 log probability 同時增加，甚至 rejected 有時還高於 chosen。

這可以從兩個角度理解：

- **正面**：cross-entropy loss 有效地引導模型朝向目標領域（例如對話）
- **負面**：缺乏對非理想生成的懲罰，導致 rejected responses 也獲得相當高的生成機率

#### Unlikelihood Training 的啟發

在對話生成領域，Welleck 等人提出的 unlikelihood training 已經展示了在 loss 中加入 `(1 - p_i)` 懲罰項可以有效減少重複生成等不理想的語言特徵。這個想法和 ORPO 的核心思路是一致的：**對不想要的輸出施加負面懲罰**。

ORPO 的不同之處在於：它不需要人工指定哪些 token 是「不想要的」，而是利用偏好資料對中的 rejected response 來自動定義懲罰目標。

---

### 知識點 2: DPO 的理論基礎（對比基準）

**這個知識點要回答什麼問題？DPO 如何做到不需要 RL 就能做偏好對齊？**

DPO 是理解 ORPO 最重要的前置工作。ORPO 的核心貢獻是在 DPO 的基礎上更進一步——不僅去掉 RL，還去掉 reference model 和 SFT warm-up。

#### 從 RLHF 出發

RLHF 的優化目標可以寫成：

$$
\max_{\pi} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi(y|x)} \left[ r(x, y) \right] - \beta \cdot D_{\text{KL}} \left( \pi(y|x) \| \pi_{\text{ref}}(y|x) \right)
$$

其中 $r(x, y)$ 是 reward model 給出的分數，$\pi_{\text{ref}}$ 是 reference policy（通常是 SFT 模型），$\beta$ 控制允許偏離的程度。

這個 KL-constrained 目標的最優解有 closed-form 表達：

$$
\pi_r(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left( \frac{1}{\beta} r(x, y) \right)
$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(r(x, y)/\beta)$ 是 partition function。

#### DPO 的關鍵洞見

DPO 的核心想法是**重新參數化（reparameterization）**：把 reward function 用 policy 來表示，而不需要真的去估計 partition function。

從上式可以解出 reward function：

$$
r(x, y) = \beta \log \frac{\pi_r(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

將這個 $r(x, y)$ 代入 Bradley-Terry 偏好模型：

$$
p(y_1 \succ y_2 | x) = \sigma(r(x, y_1) - r(x, y_2))
$$

partition function $Z(x)$ 在相減的過程中消掉了！最終得到 DPO 的損失函數：

$$
\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]
$$

#### DPO 的梯度行為

DPO 的梯度可以分解為：

$$
\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \mathbb{E} \left[ \underbrace{\sigma(\hat{r}(x, y_l) - \hat{r}(x, y_w))}_{\text{加權項}} \left( \underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{提升 chosen}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{壓制 rejected}} \right) \right]
$$

其中 $\hat{r}(x, y) = \beta \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$ 是隱含的 reward function。

加權項 $\sigma(\hat{r}(x, y_l) - \hat{r}(x, y_w))$ 決定了更新的幅度：當模型錯誤地把 rejected response 排得比 chosen 高時，加權項接近 1，更新幅度大；當模型已經正確排序時，加權項接近 0，更新幅度小。

#### DPO 的限制

DPO 雖然優雅，但有兩個實際限制：

1. **需要參考模型**：$\pi_{\text{ref}}$ 必須在訓練過程中凍結保存，佔用顯存
2. **需要 SFT warm-up**：$\pi_{\text{ref}}$ 就是 $\pi_{\text{SFT}}$，需要先做一輪 SFT
3. **每次更新需要兩次 forward pass**：一次對 $\pi_\theta$，一次對 $\pi_{\text{ref}}$

---

### 知識點 3: Odds Ratio 的定義與直覺

**這個知識點要回答什麼問題？為什麼要用 odds 而不是直接用 probability？**

#### Odds 的定義

在 ORPO 中，給定輸入序列 $x$，生成輸出序列 $y$ 的 odds 定義為：

$$
\text{odds}(y|x) = \frac{P(y|x)}{1 - P(y|x)}
$$

其中 $P(y|x)$ 是序列層級的機率：

$$
P(y|x) = \exp\left( \frac{1}{m} \sum_{t=1}^{m} \log P(y_t | x, y_{<t}) \right)
$$

也就是每個 token 的 log probability 的平均值的指數。

直覺上，$\text{odds}(y|x) = k$ 表示模型生成 $y$ 的機率是不生成 $y$ 的機率的 $k$ 倍。

#### Odds Ratio 的直覺

給定偏好資料對 $(x, y_w, y_l)$，chosen response $y_w$ 相對於 rejected response $y_l$ 的 odds ratio 為：

$$
\text{OR}(y_w, y_l) = \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)}
$$

這個比值告訴我們：**模型生成 chosen response 的相對傾向，比生成 rejected response 高出多少倍**。

當 $\text{OR}(y_w, y_l) > 1$ 時，模型更傾向於生成 chosen response；當 $\text{OR} < 1$ 時，模型反而更傾向 rejected response。

#### 為什麼用 odds 而不是 probability？

odds 和 probability 的關鍵差異在於**動態範圍**。當 $P$ 接近 0 時，$\text{odds} \approx P$；但當 $P$ 接近 1 時，$\text{odds} \to \infty$，動態範圍遠大於 $P$。

這在偏好對齊的語境中非常重要。訓練初期，模型尚未適應領域，chosen 和 rejected 的機率都在較低區間。如果用 probability ratio $P(y_w)/P(y_l)$，區分力度有限；如果用 odds ratio，因為分母 $1-P$ 的放大效應，即使機率值相近，也能產生更敏銳的對比訊號。

---

### 知識點 4: ORPO 目標函數與梯度分析

**這個知識點要回答什麼問題？ORPO 的損失函數長什麼樣，梯度更新如何運作？**

#### ORPO 目標函數

ORPO 的目標函數由兩個部分組成：

$$
\mathcal{L}_{\text{ORPO}} = \mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}} \left[ \mathcal{L}_{\text{SFT}} + \lambda \cdot \mathcal{L}_{\text{OR}} \right]
$$

- **SFT loss** ($\mathcal{L}_{\text{SFT}}$)：標準的因果語言模型負對數似然，最大化 chosen response 中 token 的預測機率：

  $$
  \mathcal{L}_{\text{SFT}} = -\frac{1}{m} \log P(y_w | x)
  $$

- **Odds Ratio loss** ($\mathcal{L}_{\text{OR}}$)：最大化 chosen response 相對於 rejected response 的 odds ratio：

  $$
  \mathcal{L}_{\text{OR}} = -\log \sigma \left( \log \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)} \right)
  $$

  這裡的 $\sigma$ 是 logistic sigmoid 函數，外層的 `-log` 使得最小化 $\mathcal{L}_{\text{OR}}$ 等價於最大化 odds ratio。

#### 完整的目標函數展開

將 odds 的定義代入，完整的損失函數為：

$$
\mathcal{L}_{\text{ORPO}} = -\mathbb{E} \left[ \log P(y_w|x) + \lambda \cdot \log \sigma \left( \log \frac{P(y_w|x) / (1 - P(y_w|x))}{P(y_l|x) / (1 - P(y_l|x))} \right) \right]
$$

或更簡潔地寫成：

$$
\mathcal{L}_{\text{ORPO}} = -\mathbb{E} \left[ \log P(y_w|x) + \lambda \cdot \log \sigma \left( \log \frac{P(y_w|x)}{P(y_l|x)} - \log \frac{1 - P(y_w|x)}{1 - P(y_l|x)} \right) \right]
$$

#### 梯度分析

$\mathcal{L}_{\text{OR}}$ 對參數 $\theta$ 的梯度（完整推導見論文 Appendix A）為：

$$
\nabla_\theta \mathcal{L}_{\text{OR}} = \delta(d) \cdot h(d)
$$

其中：

$$
\delta(d) = \left( 1 + \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)} \right)^{-1}
$$

$$
h(d) = \left( \frac{1}{1 - P(y_w|x)} \right) \nabla_\theta \log P(y_w|x) - \left( \frac{1}{1 - P(y_l|x)} \right) \nabla_\theta \log P(y_l|x)
$$

這兩個項各有不同的角色：

**$\delta(d)$：懲罰項**

當模型對 chosen response 的 odds 明顯高於 rejected response 時（即 $\text{odds}(y_w|x) \gg \text{odds}(y_l|x)$），$\delta(d) \to 0$，梯度趨近於零。這表示模型已經學會正確的偏好排序，不需要再大幅度更新。

但當模型對 rejected response 的 odds 高於 chosen 時（即模型判斷錯誤），$\delta(d) \to 1$，此時梯度會加速更新，快速修正模型的偏好。

**$h(d)$：加權對比項**

$h(d)$ 可以看成是 chosen 和 rejected 兩個梯度的加權對比：

- chosen 梯度的權重是 $1 / (1 - P(y_w|x))$
- rejected 梯度的權重是 $1 / (1 - P(y_l|x))$

當某個 response 的機率較低時，分母 $1-P$ 較大，對應的梯度權重也較大。這意味著模型會更積極地調整低機率 response 的表現——對於 chosen response，加速提高其機率；對於 rejected response，加速降低其機率。

#### 與 SFT 的協同作用

ORPO 的巧妙之處在於 $\mathcal{L}_{\text{SFT}}$ 和 $\mathcal{L}_{\text{OR}}$ 的搭配：

- **$\mathcal{L}_{\text{SFT}}$ 負責領域適應**：確保模型學會任務相關的領域知識（如對話格式、回答風格）
- **$\mathcal{L}_{\text{OR}}$ 負責偏好區分**：在 $\mathcal{L}_{\text{SFT}}$ 的基礎上，對 chosen 和 rejected responses 施加不同的梯度方向

兩者相加，讓模型在適應領域的同時也學會偏好排序，達成「一階段完成兩件事」的效果。

#### $\mathcal{L}_{\text{OR}}$ 的完整梯度推導（論文 Appendix A）

為了更深入理解 ORPO 的梯度行為，以下還原論文中省略的中間推導步驟：

從 $\mathcal{L}_{\text{OR}} = -\log \sigma(\log g)$ 開始，其中 $g = \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)}$：

$$
\nabla_\theta \mathcal{L}_{\text{OR}} = \nabla_\theta [-\log \sigma(\log g)]
$$

利用 sigmoid 的導數性質 $\sigma'(x) = \sigma(x)(1-\sigma(x))$：

$$
= -\frac{1}{\sigma(\log g)} \cdot \sigma(\log g)(1-\sigma(\log g)) \cdot \nabla_\theta \log g
$$

$$
= - (1 - \sigma(\log g)) \cdot \nabla_\theta \log g
$$

而 $1 - \sigma(\log g) = \sigma(-\log g) = \left(1 + \frac{\text{odds}(y_w|x)}{\text{odds}(y_l|x)}\right)^{-1}$，這正是前面定義的 $\delta(d)$。

接下來展開 $\nabla_\theta \log g$：

$$
\nabla_\theta \log g = \nabla_\theta \left[ \log \frac{P(y_w|x)}{1-P(y_w|x)} - \log \frac{P(y_l|x)}{1-P(y_l|x)} \right]
$$

分別對兩項求導。以 chosen 項為例：

$$
\nabla_\theta \log \frac{P(y_w|x)}{1-P(y_w|x)} = \nabla_\theta \log P(y_w|x) - \nabla_\theta \log (1-P(y_w|x))
$$

利用鏈式法則 $\nabla_\theta \log (1-P) = \frac{-P}{1-P} \cdot \nabla_\theta \log P$：

$$
= \nabla_\theta \log P(y_w|x) + \frac{P(y_w|x)}{1-P(y_w|x)} \cdot \nabla_\theta \log P(y_w|x)
$$

$$
= \left(1 + \frac{P(y_w|x)}{1-P(y_w|x)}\right) \cdot \nabla_\theta \log P(y_w|x)
$$

$$
= \frac{1}{1-P(y_w|x)} \cdot \nabla_\theta \log P(y_w|x)
$$

同理，對 rejected 項：

$$
\nabla_\theta \log \frac{P(y_l|x)}{1-P(y_l|x)} = \frac{1}{1-P(y_l|x)} \cdot \nabla_\theta \log P(y_l|x)
$$

將所有項組合起來：

$$
\nabla_\theta \mathcal{L}_{\text{OR}} = \delta(d) \cdot \left[ \frac{1}{1-P(y_w|x)} \nabla_\theta \log P(y_w|x) - \frac{1}{1-P(y_l|x)} \nabla_\theta \log P(y_l|x) \right]
$$

這正是論文中的 Equation 8。

**這個推導揭示了一個重要性質**：權重因子 $\frac{1}{1-P(y|x)}$ 意味著當模型對某個 response 的機率估計較低時（$P$ 小 → $1-P$ 大），梯度會被放大。這確保了訓練初期（模型尚未學好哪個 response 比較好時）能快速調整，而訓練後期也能保持有效的對比學習。

---

### 知識點 5: Odds Ratio vs Probability Ratio 的理論比較

**這個知識點要回答什麼問題？為什麼 ORPO 用 odds ratio 而不是更直觀的 probability ratio？**

Probability ratio 是另一種可行的對比度量：

$$
\text{PR}(y_w, y_l) = \frac{P(y_w|x)}{P(y_l|x)}
$$

但 ORPO 論文中透過理論分析和實驗證明，odds ratio 是更好的選擇。

#### 分佈特性

論文透過蒙地卡羅取樣比較了兩種 ratio 的分佈特性。從 Uniform(0, 1) 取出 50,000 對 $(X_1, X_2)$，計算：

- $\log \text{PR}(X_2|X_1) = \log X_1 - \log X_2$
- $\log \text{OR}(X_2|X_1) = \log \frac{X_1}{1-X_1} - \log \frac{X_2}{1-X_2}$

結果發現 $\log \text{OR}$ 的分佈範圍遠大於 $\log \text{PR}$。這意味著 odds ratio 在相同輸入機率下能產生更大的對比訊號。

#### 在 SFT+alignment 設定中的影響

這個分佈特性的差異在 SFT+alignment 設定中有實際的影響。由於 ORPO 是在 SFT 階段直接融入偏好對齊，模型尚未充分適應領域，chosen 和 rejected 的機率值可能都不高。

如果使用 probability ratio：
- 對比訊號較弱，需要更大的 margin 才能有效區分
- 為了達到足夠的區分效果，模型可能會過度壓制 rejected responses 的 logits，導致生成退化（generation degeneration）
- 如論文中 Figure 8 所示，probability ratio 訓練時 rejected 的 log probability 迅速降到 -4 以下

如果使用 odds ratio：
- 由於 $1-P$ 在分母的放大效應，在小機率區間也能產生足夠的對比
- 因此不需要過度壓制 rejected response，生成多樣性得以保留
- 論文中 Table 4 的詞彙多樣性實驗證實，ORPO 的 per-input diversity（0.8909 vs 0.8012）顯著高於 DPO

#### 計算上的考量

ORPO 只需要對 $\pi_\theta$ 做一次 forward pass（同時得到 chosen 和 rejected 的 logits），而 DPO 需要對 $\pi_\theta$ 和 $\pi_{\text{ref}}$ 各做一次（總共 4 次 forward pass）。

從計算角度來看：
- DPO：$\pi_\theta(y_w|x)$ + $\pi_\theta(y_l|x)$ + $\pi_{\text{ref}}(y_w|x)$ + $\pi_{\text{ref}}(y_l|x)$ = 4 次 forward
- ORPO：$\pi_\theta(y_w|x)$ + $\pi_\theta(y_l|x)$ = 2 次 forward

這在大型模型上（如 7B 參數）是顯著的差異。

---

### 知識點 6: λ 超參數的角色與消融

**這個知識點要回答什麼問題？λ 如何控制偏好對齊的強度？**

λ（論文中以 $\lambda$ 表示）是 ORPO 中唯一需要調整的超參數（除了標準的 learning rate 和 epoch），控制 $\mathcal{L}_{\text{OR}}$ 相對於 $\mathcal{L}_{\text{SFT}}$ 的權重。

#### λ 的三個設定對比

論文在 Mistral (7B) + UltraFeedback 上對 λ 進行了系統性的消融研究：

**λ = 0.1**：
- chosen 和 rejected 的 log probability 保持接近
- 模型主要透過提高 chosen 的機率來最小化 $\mathcal{L}_{\text{OR}}$，rejected 的機率幾乎不下降
- 適合不希望過度壓制任何生成風格的場景

**λ = 0.5**：
- chosen 的 log probability 繼續上升，同時 rejected 的 log probability 開始下降
- 偏好區分的效果開始顯現
- 在大多數任務上表現穩定

**λ = 1.0**：
- chosen 和 rejected 的 log probability 同步下降，但兩者之間的差距（margin）被放大
- 偏好區分最強烈，但在特定任務上有副作用
- **open-ended 任務**（humanities、roleplay、STEM）：表現更好，因為開放式生成更偏好多樣性
- **deterministic 任務**（math、coding、reasoning）：表現較差，因為過度適應訓練資料中的 chosen responses 會損害需要精確輸出的能力

#### λ 的選擇建議

論文的實驗結果暗示了一個 trade-off：更大的 λ 能更好地區分偏好風格，但會以犧牲需要精確性的任務為代價。對於實際應用，λ = 0.5 似乎是合理的折衷選擇——足夠的偏好區分能力，同時不至於過度偏向開放式生成。

---

## 實驗結果

### 主要實驗：單輪指令跟從（AlpacaEval）

ORPO 論文最令人印象深刻的結果來自 AlpacaEval 2.0，這是一個用 GPT-4 作為評審來評估模型生成品質的 benchmark。

| 模型 | 參數規模 | AlpacaEval 1.0 | AlpacaEval 2.0 |
|------|---------|---------------|---------------|
| Phi-2 + ORPO | 2.7B | **71.80%** | **6.35%** |
| Llama-2 Chat | 7B | 71.34% | 4.96% |
| Llama-2 Chat | 13B | 81.09% | 7.70% |
| **Llama-2 + ORPO** | **7B** | **81.26%** | **9.44%** |
| Zephyr (β) | 7B | 90.60% | 10.99% |
| **Mistral-ORPO-α** | **7B** | — | **11.33%** |
| **Mistral-ORPO-β** | **7B** | — | **12.20%** |

關鍵觀察：

- Llama-2 (7B) + ORPO 在 AlpacaEval 2.0 上達到 9.44%，超越了 Llama-2 Chat (13B) 的 7.70%，**以小勝大**
- Mistral-ORPO-β (7B) 的 12.20% 接近 Zephyr-β (7B) 的 10.99%，而 Zephyr 是經過完整 RLHF 流程訓練的
- Phi-2 (2.7B) + ORPO 從 0.11%（純 SFT）跳到 6.35%，展示了 ORPO 在小模型上的驚人效果

### 控制實驗：OPT 規模對比

為了排除模型架構和資料集的混淆變數，論文在 OPT 系列模型（125M、350M、1.3B）上進行了嚴格的控制實驗，使用 HH-RLHF 和 UltraFeedback 兩個資料集，對比 SFT、RLHF、DPO 和 ORPO 四種方法。

在 HH-RLHF 資料集上，所有方法使用 OPT-1.3B reward model 評估生成品質：

| 方法 | OPT-125M | OPT-350M | OPT-1.3B |
|------|---------|---------|---------|
| SFT | 0.414 | 0.493 | 0.564 |
| +RLHF (PPO) | 0.467 | 0.568 | 0.606 |
| +DPO | 0.336 | 0.491 | 0.580 |
| +ORPO（ours） | **0.511** | **0.598** | **0.661** |

關鍵觀察：

- **ORPO 在所有規模上都優於 SFT、RLHF 和 DPO**
- 隨著模型規模增大，ORPO 的優勢更加明顯（125M 高出 SFT 9.7%，1.3B 高出 9.7%），說明 ORPO 在大模型上能更有效地利用偏好資訊
- DPO 在 OPT-125M 上甚至低於 SFT（0.336 vs 0.414），可能因為小模型無法同時維護 policy 和 reference model 的有效分離

### 獎勵分佈分析

論文使用 OPT-1.3B reward model 對所有方法生成的 response 計算獎勵分數，並繪製分佈曲線。在 UltraFeedback 和 HH-RLHF 兩個資料集上，四種方法的獎勵分佈表現一致：

- **SFT**：分佈集中在低分區域（左偏），代表生成的品質普遍偏低
- **DPO**：分佈略有右移，但偏移幅度有限
- **RLHF（PPO）**：分佈明顯右移，但方差較大，存在品質不穩定的情況
- **ORPO**：分佈顯著右移，且具有最高的均值和中位數，方差適中

這個實驗證明了 ORPO 不僅在 benchmark 分數上勝出，在獨立 reward model 的評估中也確實產生了最高品質的生成。這排除了 ORPO 只是「過度適應某個特定 benchmark 評估方式」的可能性——它確實學會了生成更符合人類偏好的回應。

### 多輪對話能力（MT-Bench）

MT-Bench 測試模型在多輪對話中的綜合能力：

| 模型 | 總分 |
|------|-----|
| Llama-2 Chat (7B) | 6.27 |
| Llama-2 Chat (13B) | 6.65 |
| Llama-2 Chat (70B) | 6.86 |
| Mistral-ORPO-α (7B) | **7.23** |
| Mistral-ORPO-β (7B) | **7.32** |
| GPT-3.5-turbo | 7.94 |

| Mistral-ORPO-β 的 7.32 分不僅超越了所有 Llama-2 Chat 版本（包括 70B），而且接近 GPT-3.5-turbo 的 7.94 分，考慮到只用 UltraFeedback（61k 實例）訓練了一個 epoch，這個結果相當出色。

#### MT-Bench 分類別分析

論文進一步分析了 Mistral-ORPO-β 在 MT-Bench 各分類的表現：

| 類別 | Mistral-ORPO-α | Mistral-ORPO-β | Llama-2 Chat (70B) | GPT-3.5-turbo |
|------|---------------|---------------|-------------------|---------------|
| Writing | 9.38 | 9.53 | 8.94 | 9.53 |
| Humanities | 9.78 | 9.84 | 8.42 | 9.47 |
| Roleplay | 8.50 | 8.60 | 7.30 | 8.70 |
| Reasoning | 5.34 | 5.02 | 5.78 | 6.38 |
| Math | 3.95 | 4.50 | 3.80 | 5.10 |
| Coding | 4.85 | 5.35 | 5.60 | 7.55 |
| Extraction | 6.65 | 6.65 | 7.80 | 7.25 |
| STEM | 8.76 | 8.80 | 7.95 | 8.65 |

值得注意的幾點：
- **Writing 和 Humanities**：ORPO 模型甚至超過 Llama-2 Chat (70B) 和 GPT-3.5-turbo，顯示在開放式生成任務上的優勢
- **Math 和 Coding**：明顯落後 GPT-3.5-turbo，論文推測這與訓練資料 UltraFeedback 中缺乏足夠的數學/程式相關偏好對有關
- **Reasoning**：Mistral-ORPO-β (5.02) 低於 Llama-2 Chat (70B) 的 5.78，說明偏好對齊可能對邏輯推理能力造成一定程度的負面影響

### 指令層級遵循（IFEval）

IFEval 測試模型是否能準確遵循指令中的細粒度約束：

| 模型 | Prompt-Strict | Prompt-Loose | Inst-Strict | Inst-Loose |
|------|-------------|-------------|-------------|-----------|
| Mistral-ORPO-α | 0.501 | 0.508 | 0.600 | 0.616 |
| Mistral-ORPO-β | 0.529 | 0.556 | **0.636** | **0.662** |

Mistral-ORPO-β 在指令層級鬆散準確度（Inst-Loose）達到 66.19%，證明 ORPO 不僅擅長開放式對話，也能準確遵循複雜指令。

### 消融實驗：λ 對 MT-Bench 分類表現的影響

論文進一步分析了不同 λ 對 MT-Bench 各類別表現的影響：

| 類別 | λ = 0.1 | λ = 1.0 | 說明 |
|------|---------|---------|------|
| Writing | 低 | 高 | 開放式生成 |
| Roleplay | 低 | 高 | 開放式對話 |
| Humanities | 低 | 高 | 主觀性回答 |
| STEM | 低 | 高 | 半結構化知識 |
| Extraction | 高 | 低 | 需精確提取 |
| Math | 高 | 低 | 需確定性答案 |
| Coding | 高 | 低 | 需精確語法 |
| Reasoning | 高 | 低 | 需邏輯推導 |

這個結果清楚地說明了 λ 的角色：更大的 λ 讓模型更偏向訓練資料中 chosen responses 的風格（適合開放式生成），但會犧牲需要精確性和確定性輸出的任務能力。

### 詞彙多樣性

論文透過 per-input cosine similarity 來衡量生成多樣性（值越低越好）：

| 模型 | Per-Input | Across-Input |
|------|-----------|-------------|
| Phi-2 + SFT + DPO | 0.8012 | 0.6019 |
| Phi-2 + ORPO | **0.8909** | **0.5173** |
| Llama-2 + SFT + DPO | 0.8889 | 0.5658 |
| Llama-2 + ORPO | **0.9008** | **0.5091** |

ORPO 在 per-input 維度（對同一個輸入生成多個回應的多樣性）有更高的相似度，但在 across-input 維度（不同輸入之間的多樣性）有更低的相似度。這說明 ORPO 促使模型生成更「針對特定指令」的回應，而不是通用的模板式輸出。

### 實作細節與超參數設定

以下整理論文中各實驗的具體設定，供實際複現參考：

**通用設定：**
- 所有模型使用 Flash-Attention 2 加速
- OPT 系列和 Phi-2 使用 DeepSpeed ZeRO 2
- Llama-2 (7B) 和 Mistral (7B) 使用 FSDP（Fully Sharded Data Parallel）
- 7B 模型使用 4 張 NVIDIA A100，2.7B 使用 2 張，其餘使用 4 張 A6000
- 優化器：AdamW（部分使用 paged AdamW 以節省顯存）
- Learning rate scheduler：linear warmup + cosine decay
- 輸入長度：HH-RLHF 截斷為 1,024 tokens，UltraFeedback 為 2,048 tokens

**ORPO 設定：**
- Learning rate：8e-6（對所有模型）
- Training epochs：10
- Best model selection：最低 evaluation loss
- λ 預設值：文中推薦 λ = 0.5 或 λ = 1.0 視任務而定

**DPO 對比設定：**
- β = 0.1（所有實驗）
- Learning rate：5e-6
- Training epochs：3（選 validation loss 最低的 checkpoint）
- 先做 1 epoch SFT 作為 reference model

**RLHF 對比設定：**
- PPO epochs：4
- KL coefficient：0.1
- Horizon：2,000
- Learning rate：1e-5
- 使用 OPT-350M 作為 reward model（OPT 實驗）

---

## 與相關工作的對比

以下從多個維度對比三種主要的偏好對齊方法：

| 維度 | RLHF (PPO) | DPO | ORPO |
|------|-----------|-----|------|
| 訓練階段數 | 3 (SFT → RM → RL) | 2 (SFT → DPO) | **1 (ORPO)** |
| 需要 reference model | 是（KL 約束） | 是（$\pi_{\text{ref}}$） | **否** |
| 需要 reward model | 是 | **否** | **否** |
| 需要 RL | 是 (PPO) | **否** | **否** |
| Forward pass/batch | 4+ | 4 | **2** |
| 超參數數量 | 多（lr, KL coeff, PPO clip 等） | 中（$\beta$） | **少（λ）** |
| 訓練穩定性 | 低（PPO 敏感） | 高 | 高 |
| 生成多樣性 | 中（KL 約束保護） | 中 | **高** |
| 大規模驗證 | ✓ (InstructGPT 等) | ✓ (NeurIPS 2023) | 限 7B 以下 |
| 實作複雜度 | 高 | 中 | **低** |

### 對 ORPO 最有利的維度

ORPO 在計算效率（2 forward pass vs 4）、訓練簡潔度（單階段 vs 多階段）和參數效率（無需 reference model）上有明顯優勢。這使得 ORPO 成為資源受限場景下一個極具吸引力的選擇。

### DPO 仍然持有的優勢

DPO 有更完整的理論基礎（透過 Bradley-Terry 模型的嚴謹推導），且在更大規模的模型（如 Llama 13B、70B）上有更充分的事實驗證。ORPO 論文中只在 7B 以下的模型進行了測試。

---

## 我的觀察

### 關於「SFT 的重新評估」

ORPO 論文最有價值的貢獻之一，是它系統性地研究了 SFT 在偏好對齊中的角色。這是一個經常被忽略的問題：為什麼需要先做 SFT？SFT 到底在做什麼？

ORPO 的結論——SFT 主要做領域適應，而偏好對齊需要額外的懲罰機制來區分風格——雖然看起來直觀，但在 DPO 之前很少有人從這個角度深入分析過。這提醒我：**有時候最重要的發現不是在既有框架內改進方法，而是更基本地去問「我們為什麼要這麼做？」**

### 關於 odds ratio 的選擇

ORPO 選擇 odds ratio 而非 probability ratio，是一個理論與實驗互相支援的精彩案例。論文中用取樣分析展示了兩種 ratio 的分佈差異，再透過實驗驗證了 probability ratio 會導致生成退化。這種「理論預測 → 實驗驗證 → 選擇更優方案」的研究風格值得學習。

> **視覺化輔助**：[Odds Ratio vs Probability Ratio 對比示意圖](https://excalidraw.com/#json=8duo07zxDt3m_6hgJlH9q,N7sM9I7iXWx-w6rlX_Im3A)（Excalidraw）展示了兩種 ratio 的關鍵差異——odds ratio 的動態範圍更寬廣，在訓練初期也能產生有效的對比訊號。

**一個值得思考的問題是**：odds ratio 真的是最優選擇嗎？論文比較了 odds ratio 和 probability ratio，但其實還有其他可能的對比度量——比如 log-odds、logit difference、或是兩者的非線性組合（如平方、指數）。ORPO 選擇 odds ratio 有一定的直覺和理論支持，但這個空間可能還有更好的解未被發現。

### 偏好對齊的「一階段化」趨勢

從 RLHF（三階段）→ DPO（兩階段）→ ORPO（一階段），可以看出偏好對齊方法的明確趨勢：**流程越來越簡潔，依賴的外部元件越來越少**。這不僅是工程上的進步，也反映了對問題本質的更深入理解。

ORPO 之後，SimPO 進一步用 sequence-level 的平均 log probability 替代了 reference model 的比對，而 CPO（Contrastive Preference Optimization）也提出了類似的單階段方案。這些方法的共同趨勢是：
1. 去掉 reference model
2. 去掉 SFT warm-up
3. 將 alignment signal 直接嵌入 training loss

### 一個有趣的失敗案例

論文中提到，λ = 1.0 在需要確定性答案的任務（math、coding）上表現較差。這其實可以理解：當 model 過度用力地讓 chosen response 與 rejected response 區分開來時，它其實是在過度適應訓練資料中的具體答案風格，損害了精確推理所需的能力。

這也暗示了 ORPO 的一個潛在改進方向：**自適應 λ**。不同類型的問題可能需要不同的偏好對齊強度——開放式問題用大的 λ，精確推理問題用小的 λ。

### 與 DPO 的互補性

我認為 ORPO 和 DPO 不一定是競爭關係，更可能是互補的。對於資源受限的場景（單 GPU、小模型），ORPO 的計算效率和訓練簡潔度是明顯優勢。對於大規模訓練或在已經有 SFT 模型的場景，DPO 的理論成熟度和更廣泛的驗證可能更有吸引力。

未來的方向可能是在兩者之間取長補短——例如，用 ORPO 的思路來改進 DPO 中 reference model 的設計，或者把 ORPO 的 odds ratio 損失整合到其他偏好對齊框架中。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **Direct Preference Optimization: Your Language Model is Secretly a Reward Model** ([2305.18290](https://arxiv.org/abs/2305.18290))
   - 與本文關係：ORPO 的直接對比基準，DPO 從 RLHF 推導出 closed-form 偏好損失，ORPO 在 DPO 的基礎上進一步去掉 reference model 和 SFT warm-up

### 後續發展（未涵蓋，僅列出）

偏好對齊領域在 ORPO（2024-03）之後仍有許多進展：

- **[SimPO (2024)](https://arxiv.org/abs/2405.14734)**：用 sequence-level 的平均 log probability 替代 DPO 中的 reference model 比對，與 ORPO 的精神類似但使用不同度量
- **[KTO (2024)](https://arxiv.org/abs/2402.01306)**：不需要成對偏好資料的對齊方法，只需要知道一個 response 是好是壞
- **[CPO (2024)](https://arxiv.org/abs/2401.08417)**：另一種單階段偏好對齊方法，透過對比 chosen 和 rejected 的 loss
- **[GRPO (2024)](https://arxiv.org/abs/2402.03300)**：DeepSeek 提出的 group-based RL 方法，也用於推理強化

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。

---

<!--
寫完後檢查清單:
- [x] TL;DR 是否真的能三句話講完？
- [x] 知識點是「我歸納的概念」還是「論文的章節標題」？
- [x] 每個知識點都串到了種子論文與相關論文？
- [x] 論文原文引用比例 < 10%？
- [x] 公式有沒有打錯？符號是否一致？
- [x] 圖片（如有）是否放在 assets/ 並用相對路徑引用？
- [x] meta.yaml 是否同步更新？
- [x] papers.bib 是否包含所有引用的論文？
-->
