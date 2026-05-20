# Layer Normalization: 從批次走向層級的歸一化

> **種子論文**: [Layer Normalization](https://arxiv.org/abs/1607.06450) (2016-07)
> **作者**: Jimmy Lei Ba, Jamie Ryan Kiros, Geoffrey E. Hinton
> **機構**: University of Toronto & Google Inc.

---

## TL;DR

訓練深度神經網路時，每層輸入分布在訓練過程中不斷改變（internal covariate shift），導致需要低學習率和精心初始化。Layer Normalization 將 Batch Normalization 的歸一化維度從「batch」轉為「layer」——對每個 training case，獨立計算同一層所有 hidden units 的均值和方差來做歸一化。這個設計讓 LN 天然適用於 RNN、支援 batch size = 1 的線上學習，並在 RNN 為主的多個任務上顯著加速訓練，成為後續 Transformer 架構中 normalization 的標準選擇。

---

## 背景與動機

在深度神經網路中，每一層的輸入是前一層的輸出經由非線性變換後得到的。隨著訓練進行，前一層的權重不斷更新，導致下一層看到的輸入分布持續變化——這個現象被稱為 **internal covariate shift**。

Internal covariate shift 帶來的困擾：

- **梯度飽和**：當輸入落入 sigmoid / tanh 等飽和非線性的極值區域時，梯度趨近於零，學習停滯
- **學習率受限**：為了避免參數更新過大導致的訓練發散，必須使用很小的學習率
- **初始化敏感**：參數初始值的選擇對訓練能否收斂至關重要，需要精巧的初始化策略（如 Xavier、He initialization）

**Batch Normalization (Ioffe & Szegedy, 2015)** 率先提出了系統性的解法：在每一層的非線性之前插入一個 normalization 層，透過 mini-batch 的統計量標準化該層的輸入，使活化值維持在零均值、單位方差的穩定分布中。BN 在 CNN 上取得了巨大成功——14 倍訓練加速、ImageNet state-of-the-art。

但 BN 有兩個根本限制：

1. **Mini-batch 依賴性**：BN 的統計量完全依賴當前 mini-batch 的樣本。當 batch size 很小時（如大型分散式訓練），統計量 noisy；batch size = 1 時方差為零，BN 完全失效
2. **RNN 不適用**：RNN 在不同 time-step 共用同一組權重。如果對每個 time-step 獨立維護 BN 統計量，測試序列就不能比訓練序列長；如果跨 time-step 共享，又破壞 normalization 效果

Layer Normalization 正是在這樣的背景下被提出來填補 BN 的缺口。

---

## 核心知識點

本文圍繞以下知識點展開：

1. **Internal Covariate Shift 的定義與影響**——這個概念究竟是什麼、對訓練造成哪些具體困擾
2. **Batch Normalization 的核心設計**——如何用 mini-batch 統計量做 normalization，以及為何成功
3. **BN 的局限性**——為何 mini-batch 依賴性和 RNN 不友好是結構性問題
4. **Layer Normalization 的數學定義**——從 BN 到 LN 的「維度轉置」如何運作
5. **不變性與幾何分析**——LN 與 BN 在參數空間中的不變性差異，以及 Fisher Information Matrix 分析揭示的學習率調控機制
6. **實驗驗證**——兩篇論文在各自擅長領域的實證結果

---

## 方法詳解

### 知識點 1: Internal Covariate Shift

**這個問題為何重要？**

Internal covariate shift 指的是神經網路在訓練過程中，由於前層參數不斷更新，導致後層輸入分布持續變化的現象。這個概念最早由 Ioffe & Szegedy 在 BN 論文中正式提出。

假設一個子網路 $F_2$ 接收 $x = F_1(u, \Theta_1)$ 作為輸入。如果 $x$ 的分布在整個訓練過程中不斷變化，$F_2$ 的參數 $\Theta_2$ 就必須持續「追趕」不斷變化的目標——這會極大拖慢訓練速度。

**BN 怎麼處理？**

BN 的核心思路：直接在每一層的輸入上做 normalization，強制讓 $x$ 的均值和方差固定：

$$\hat{x}^{(k)} = \frac{x^{(k)} - \text{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}] + \epsilon}}$$

然後引入可學習的 $\gamma^{(k)}$ 和 $\beta^{(k)}$ 來恢復網路的表示能力：

$$y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}$$

**LN 怎麼處理？**

LN 接受 BN 提出的「internal covariate shift 需要被解決」這個前提，但選擇了不同的 normalization 維度（見知識點 4）。

---

### 知識點 2: Batch Normalization 的設計

**BN 如何運作？**

給定一個 mini-batch $B = \{x_1, ..., x_m\}$，BN 對每個 activation 獨立計算：

$$\mu_B = \frac{1}{m} \sum_{i=1}^m x_i$$

$$\sigma_B^2 = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2$$

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_i = \gamma \hat{x}_i + \beta$$

BN 的關鍵創新在於**將 normalization 嵌入網路架構本身**，使其成為一個可微分的層，參與反向傳播。這避免了早期 normalization 方法中 normalization 與 gradient descent 互相干擾的問題（如 bias 持續增長而 loss 不變的 pathological 行為）。

**BN 的理論貢獻：**
- Internal covariate shift 的正式定義
- 證明 BN 允許更高的學習率（論文展示 5–30 倍的提升）
- BN 有正則化效果，可減少或移除 Dropout
- BN 讓飽和非線性函數（sigmoid）變得可訓練

---

### 知識點 3: BN 的局限性

**為何 mini-batch 依賴性是結構性問題？**

BN 的統計量 $\mu_B$ 和 $\sigma_B^2$ 是從當前 mini-batch 中估計的。當 batch size 很小時，這些估計的方差很大。極端情況下 batch size = 1：

$$\sigma_B^2 = \frac{1}{1} \sum_{i=1}^1 (x_i - x_i)^2 = 0$$

此時 BN 退化為一個無意義的平移變換。這在大型分散式訓練（每台機器只能放 small batch）和線上學習場景中成為瓶頸。

**為何 BN 不適合 RNN？**

RNN 在同一個 time-step 內對所有序列位置使用同一組權重 $W_{hh}$。BN 有兩種應用方式，都有問題：

1. **跨 time-step 共享統計量**：$\mu$ 和 $\sigma$ 在所有 time-step 共用 → 破壞 BN 的 normalization 效果
2. **每個 time-step 獨立統計量**：需要存儲 $T$ 組統計量（$T$ 為序列長度）→ 測試序列若比訓練序列長，沒有對應的統計量可用

此外，BN 在訓練和推論時的行為不一致（使用 running averages），增加了部署複雜度。

---

### 知識點 4: Layer Normalization 的數學定義

**LN 如何「轉置」BN？**

LN 的洞察非常直接：既然 BN 在 batch 維度上做 normalization 有問題，為什麼不改到 layer 維度上做？

對第 $l$ 層的 summed inputs $a^l = [a_1^l, a_2^l, ..., a_H^l]$（$H$ 為 hidden units 數量）：

$$\mu^l = \frac{1}{H} \sum_{i=1}^H a_i^l$$

$$\sigma^l = \sqrt{\frac{1}{H} \sum_{i=1}^H (a_i^l - \mu^l)^2}$$

標準化後的輸出：

$$h = f\left(\frac{g}{\sigma^l} \odot (a^l - \mu^l) + b\right)$$

其中 $g$（gain）和 $b$（bias）是可學習的參數，$f(\cdot)$ 是非線性函數。

**BN vs. LN 的數學對比：**

$$\text{BN:} \quad \mu_i = \frac{1}{m} \sum_{j=1}^m a_{ij}, \quad \sigma_i = \sqrt{\frac{1}{m} \sum_{j=1}^m (a_{ij} - \mu_i)^2}$$

$$\text{LN:} \quad \mu_l = \frac{1}{H} \sum_{i=1}^H a_{il}, \quad \sigma_l = \sqrt{\frac{1}{H} \sum_{i=1}^H (a_{il} - \mu_l)^2}$$

BN 中，同一個 neuron 在不同 training case 的活化值共享一組統計量；LN 中，同一個 layer 內不同 neuron 的活化值共享一組統計量。

**LN 在 RNN 中的應用（LN-LSTM）：**

$$\begin{pmatrix} f_t \\ i_t \\ o_t \\ g_t \end{pmatrix} = \text{LN}(W_{hh} h_{t-1}; \gamma_1, \beta_1) + \text{LN}(W_{xh} x_t; \gamma_2, \beta_2) + b$$

$$c_t = \sigma(f_t) \odot c_{t-1} + \sigma(i_t) \odot \tanh(g_t)$$

$$h_t = \sigma(o_t) \odot \tanh(\text{LN}(c_t; \gamma_3, \beta_3))$$

注意每個 LN 都有自己獨立的 $\gamma$ 和 $\beta$ 參數，且這些參數跨 time-step 共享。

---

### 知識點 5: 不變性與幾何分析

**LN 與 BN 的不變性差異：**

LN 論文提供了深入的不變性分析，這是 BN 論文沒有的理論視角：

| 變換 | BN | Weight Norm | LN |
|------|:--:|:-----------:|:--:|
| Weight vector rescaling | 不變 | 不變 | 不（註1） |
| Weight matrix rescaling | 不 | 不 | 不變 |
| Weight re-centering | 不 | 不 | 不變（註2） |
| Dataset rescaling | 不變 | 不變 | 不變 |
| Dataset re-centering | 不變 | 不 | 不 |
| Per-case rescaling | 不 | 不 | 不變 |

> 註1: LN 對單一 weight vector 的縮放**沒有**不變性——這與 BN 不同
> 註2: LN 對整個 weight matrix 的平移具有不變性——這是 BN 和 Weight Norm 都沒有的性質

**Fisher Information Matrix 分析：**

LN 論文通過 Fisher Information Matrix 分析了 normalization 方法對學習動態的影響，這是一個重要的理論貢獻。關鍵發現：

- Normalization scalar $\sigma$ 會**隱式調控學習率**：當 weight vector 的 norm 增長時，Fisher curvature 沿 weight 方向被 $\frac{1}{\sigma^2}$ 因子縮放
- 這意味著權重越大，gradient 對它的影響越小——類似「early stopping」的效果
- 相較於標準 GLM，normalized 模型學習「權重幅度」是透過 gain 參數 $\gamma$，其 KL metric 只取決於預測誤差的大小，而非輸入的尺度——因此更穩健

---

## 實驗結果

### 主要實驗

**BN 的主要結果（ImageNet 分類）：**

| 模型 | 訓練步數（至 72.2% accuracy） | Max Accuracy |
|------|:----------------------------:|:------------:|
| Inception（baseline） | $31.0 \times 10^6$ | 72.2% |
| BN-Baseline | $13.3 \times 10^6$ | 72.7% |
| BN-x5 | $2.1 \times 10^6$ (14× 加速) | 73.0% |
| BN-x30 | $2.7 \times 10^6$ | **74.8%** |
| BN-x5-Sigmoid | — | 69.8% |
| BN Ensemble (6 nets) | — | **4.9% top-5 error** |

**LN 的主要結果（RNN 任務）：**

| 任務 | 指標 | 加速效果 |
|------|------|---------|
| Order-Embeddings（Image-Sentence Ranking） | Recall@1 | 60% 的訓練步數達最佳，Recall +1.9% |
| Attentive Reader（QA） | Validation Error | 超越 BN-LSTM，且對初始 scale 不敏感 |
| Skip-thought Vectors（5 個 downstream tasks） | 多項準確率 | 全面加速，最終結果更好 |
| DRAW（生成模型，MNIST） | Test NLL | 收斂約 2 倍快 |
| Handwriting Generation（長序列） | NLL | 長序列下顯著加速 |
| Permutation Invariant MNIST | Test Error | 小 batch (bz=4) 下優於 BN |

### 關鍵觀察

- **BN 的強項在 CNN**：在 ImageNet 這類大規模圖像分類任務上，BN 的表現無可匹敵。14 倍訓練加速、ensemble 超越人類水準
- **LN 的強項在 RNN**：在 6 個 RNN 為主的任務上，LN 都展現了加速效果和更好的泛化
- **Batch size 的關鍵差異**：在手寫生成實驗中（batch size = 8、序列長度 ~700），LN 的優勢最為明顯——這正是 BN 最弱的場景
- **LN 對初始化不敏感**：在 Attentive Reader 實驗中，LN 將 gain 初始化設為 1.0 或 0.1 影響不大，而 BN 必須設為 0.1 才有好效果

### 失敗案例與限制

- **CNN 不適用**：LN 論文的 Section 6.7 明確指出，在卷積神經網路中 BN 仍優於 LN。原因是卷積層的 hidden units 分布在圖像的不同空間位置，邊界 pixel 與中心 pixel 的活化統計差異很大，不適合用同一組 LN 統計量
- **後續研究指出**：雖然 LN 解決了 BN 在 RNN/Tranformer 的問題，但在非常深的網路中，LN 的訓練穩定性仍不如後來提出的 RMS Norm 等簡化變體

---

## 與相關工作的對比

| 維度 | Batch Normalization | Layer Normalization | Weight Normalization |
|------|:-------------------:|:-------------------:|:--------------------:|
| 訓練範式 | Mini-batch SGD | 任意（含 batch size = 1） | 任意 |
| 是否需要 running averages | 是（推論用） | 否 | 否 |
| RNN 適用性 | 困難（time-step 依賴） | 自然適用 | 可適用 |
| CNN 適用性 | **最佳** | 有限 | 中等 |
| 是否為 reparameterization | 是（可視為） | **不是** | 是 |
| 理論框架 | Internal covariate shift | Invariance + Fisher geometry | Reparameterization |

---

## 我的觀察

1. **「維度轉置」是最優雅的 insight**。LN 沒有引入任何全新的概念，只是把 BN 的統計量計算維度從 batch 轉到 layer——這個 180 度的思路轉彎，就解決了 BN 最頭痛的兩個問題。事後看來非常簡單，但在當時需要對 BN 的限制有極深刻的理解。

2. **LN 的理論貢獻常被忽略**。很多人只知道 LN 是一種「可以用在 Transformer 的 normalization」，但沒有注意到它對參數空間幾何的分析（Fisher Information Matrix 和 implicitly controlled learning rate）其實比 BN 更深入。BN 論文的核心貢獻是提供了有效的演算法和驚人的實驗結果；LN 論文則在理論分析上更細緻。

3. **BN 與 LN 是互補而非競爭**。深度學習社群的常見誤解是認為 LN「取代」了 BN。事實上，在 CNN 領域 BN 至今仍是標準；在 NLP/LLM 領域 LN 才是主流。兩者的優化目標相同（穩定訓練），但適用場景不同。

4. **轉折點的歷史偶然性**。LN 最初是為了解決 RNN 的訓練問題，從論文的實驗設計（6 個任務中有 5 個是 RNN）可以清楚看出。它後來成為 Transformer 的標準元件其實是「意外收穫」——因為 Transformer 本身也是一種序列模型，適合 LN 的設計。如果當年 Transformer 先出現，或許 normalization 的發展史會完全不同。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift** ([1502.03167](https://arxiv.org/abs/1502.03167))
   - 與本文關係：Layer Normalization 的直接前身和對比基準。LN 的設計是對 BN 的「維度轉置」，旨在解決 BN 在 RNN 和 small batch 場景的限制

### 後續發展（未涵蓋，僅列出）

- **Weight Normalization** ([Salimans & Kingma, 2016](https://arxiv.org/abs/1602.07868)) — 另一種 normalization 方法，通過重新參數化權重向量來加速訓練
- **Group Normalization** ([Wu & He, 2018](https://arxiv.org/abs/1803.08494)) — 在 batch size 很小時的替代方案，將 channel 分組後分別 normalized
- **RMS Norm** ([Zhang & Sennrich, 2019](https://arxiv.org/abs/1910.07467)) — 移除 LN 中的 mean centering，僅保留 root mean square 標準化，在 LLM 訓練中更高效
- **Pre-LN vs Post-LN** (Vaswani et al., 2017; Xiong et al., 2020) — Transformer 中 LN 放置位置的討論

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
