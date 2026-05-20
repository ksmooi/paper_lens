# Batch Normalization: 加速深層網路訓練的關鍵機制

> **種子論文**: [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) (2015-02)
> **作者**: Sergey Ioffe, Christian Szegedy
> **機構**: Google Inc.

---

## TL;DR

> 深層神經網路訓練的最大痛點之一是每層的輸入分布在訓練過程中不斷變化（Internal Covariate Shift），這迫使我們使用很小的 learning rate 與非常謹慎的初始化。Batch Normalization 把標準化直接嵌入網路架構中——每個 mini-batch 計算一次 activation 的均值與變異數做標準化，再透過可學習的 scale 與 shift 參數恢復表達能力——結果是可以用 **14 倍更少的訓練步數**達到相同準確率，還能順便拿掉 Dropout。這個方法後來幾乎成為所有深層網路的標準配備。

---

## 背景與動機

2014 年前後的深度學習正處於爆發期。Szegedy 等人的 Inception (GoogLeNet) 剛在 ILSVRC 2014 以 22 層的深度拿下冠軍，證明了「更深＝更好」這個方向。然而，訓練這些深層網路充滿了工程上的麻煩：

- **Learning rate 必須很小**，否則梯度不是爆炸就是消失
- **初始化極為敏感**——不同初始化方案對收斂速度的影響可以差好幾個數量級
- **Saturating nonlinearities（如 sigmoid、tanh）幾乎不能用**，因為深層網路很容易讓 activation 掉進飽和區，梯度接近零
- **需要使用 Dropout、L2 regularization 等技巧**來控制 overfitting

這些問題的根源是什麼？Ioffe 和 Szegedy 認為：問題出在每一層的輸入分布在訓練過程中一直在變。

假設第 $k$ 層的參數是 $\Theta_k$，前一層的參數 $\Theta_{k-1}$ 每次更新完，傳進第 $k$ 層的輸入 $x = F_{k-1}(u, \Theta_{k-1})$ 的分布就跟著變。對第 $k$ 層來說，它永遠在追一個移動的目標。這個現象被稱為 **Internal Covariate Shift**。

既有方法（如 LeCun et al., 1998 的 input whitening、Glorot & Bengio 2010 的 Xavier initialization）只處理了網路輸入層的標準化，或只處理初始階段的梯度問題，但沒有解決**訓練過程中層間輸入分布持續偏移**這個根本問題。

---

## 核心知識點

本文圍繞以下知識點展開：

1. **Internal Covariate Shift**——訓練過程中層間輸入分布不斷變化的現象，以及它為什麼讓訓練變慢
2. **Mini-Batch 標準化 (BN Transform)**——核心演算法：如何用 mini-batch 統計量做標準化，並讓它可微分
3. **學習式 Scale 與 Shift ($\gamma, \beta$)**——為什麼標準化之後需要恢復表達能力
4. **BN 的反向傳播**——梯度如何流經 BN 層
5. **訓練 vs. 推理的雙模式設計**——mini-batch 統計量 vs. population 統計量
6. **BN 的連鎖效應**——更高 learning rate、Dropout 可移除、sigmoid 可用

---

## 方法詳解

### 知識點 1: Internal Covariate Shift

**這個知識點要回答什麼問題？**

為什麼深層網路這麼難訓練？除了 vanishing/exploding gradient 外，還有一個更根本的問題。

Ioffe 與 Szegedy 將 Internal Covariate Shift 定義為：**訓練過程中，由於前一層參數改變，導致當前層輸入分布發生變化的現象**。

考慮一個網路 $\ell = F_2(F_1(u, \Theta_1), \Theta_2)$。學習 $\Theta_2$ 時，可以將 $x = F_1(u, \Theta_1)$ 視為子網路 $\ell = F_2(x, \Theta_2)$ 的輸入。如果 $x$ 的分布在訓練過程中不斷變化，$\Theta_2$ 就必須不斷「重新適應」新的輸入分布，這會顯著拖慢訓練。

更直觀的例子：當 sigmoid 的輸入 $x$ 絕對值過大時，$g'(x) \to 0$，梯度無法往下傳。由於 $x$ 受 $\Theta_1, \Theta_2, ...$ 等所有下方層參數影響，這些參數的微小變化就可能把大量 $x$ 推入飽和區。

**種子論文怎麼處理？**

BN 的核心主張是：如果每層的輸入分布能保持穩定（固定均值與變異數），那麼訓練速度就能大幅提升。這個想法延伸自 covariate shift 在 domain adaptation 中的應用——只是這次我們關心的不是整體資料分布，而是網路內部每一層的輸入分布。

**相關論文怎麼處理？**

- **Inception (Szegedy et al., 2014)**: 以架構設計（Inception module）來提升計算效率與深度，但沒有從根本上解決訓練的不穩定性。Inception 使用了輔助分類器（auxiliary classifiers）來幫助梯度傳播，這是一種工程上的 workaround。BN 則直接解決了梯度傳播的穩定性問題。

---

### 知識點 2: Mini-Batch 標準化 (BN Transform)

**這個知識點要回答什麼問題？**

理論上，對每層輸入做完整 whitening（去相關 + 標準化）是最理想的，但計算 covariance matrix 及其 inverse square root 在深層網路中完全不可行。怎麼做一個「夠好」的近似？

**種子論文怎麼處理？**

BN 做了兩個關鍵簡化：

1. **Dimension-wise 標準化，而非 joint whitening**——每個 scalar feature 獨立標準化，不做去相關。對於一個 $d$ 維輸入 $x = (x^{(1)}, ..., x^{(d)})$，每個維度 $k$ 標準化為：
   $$
   \hat{x}^{(k)} = \frac{x^{(k)} - \mathbb{E}[x^{(k)}]}{\sqrt{\text{Var}[x^{(k)}]}}
   $$
   即使不做去相關，這樣的標準化已足夠加速收斂（LeCun et al., 1998）。

2. **用 mini-batch 統計量估計整體統計量**——每個 mini-batch $B$ 計算均值與（有偏）變異數：
   $$
   \mu_B = \frac{1}{m} \sum_{i=1}^m x_i, \quad
   \sigma^2_B = \frac{1}{m} \sum_{i=1}^m (x_i - \mu_B)^2
   $$
   標準化後的 activation：
   $$
   \hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma^2_B + \epsilon}}
   $$

這個 BN Transform $\text{BN}_{\gamma,\beta}: x_{1...m} \to y_{1...m}$ 是全文的核心貢獻。

對於 convolutional layers，BN 做了進一步調整：同一 feature map 內所有 spatial locations 共享同一組 $\mu_B$ 與 $\sigma^2_B$。這意味著對於 size $m$ 的 mini-batch 與 $p \times q$ 的 feature map，實際計算統計量的樣本數是 $m' = m \cdot p \cdot q$。

**相關論文怎麼處理？**

- **Inception**: 沒有特殊的 normalization 機制。使用了 Local Response Normalization (LRN) ——一種跨 channel 的局部歸一化——但效果有限，且不是訓練穩定性的核心設計。Inception 的訓練依賴於 carefully tuned learning rate schedule、momentum、以及 auxiliary classifiers 來維持穩定。

---

### 知識點 3: 學習式 Scale 與 Shift ($\gamma, \beta$)

**這個知識點要回答什麼問題？**

單純標準化會把 activation 的分布固定為均值 0、變異數 1，但這可能不是該層最理想的輸入分布。例如，對 sigmoid 來說，若輸入永遠集中在 0 附近，就會落在 sigmoid 的線性區域，失去非線性的表達力。

**種子論文怎麼處理？**

BN 為每個標準化後的 activation $\hat{x}^{(k)}$ 引入一對可學習參數 $\gamma^{(k)}, \beta^{(k)}$：
$$
y^{(k)} = \gamma^{(k)} \hat{x}^{(k)} + \beta^{(k)}
$$

這兩個參數的關鍵在於：它們讓 BN 層**可以學到 identity transform**——只要設 $\gamma^{(k)} = \sqrt{\text{Var}[x^{(k)}]}$, $\beta^{(k)} = \mathbb{E}[x^{(k)}]$，就能完全還原原始 activation。這確保 BN 永遠不會讓網路性能變差——最多回到沒加 BN 時的狀態。

---

### 知識點 4: BN 的反向傳播

**這個知識點要回答什麼問題？**

BN Transform 依賴於 mini-batch 內**所有樣本**的統計量，這意味著在反向傳播時，loss 對某個樣本 $x_i$ 的梯度不只是通過 $\hat{x}_i$，還要通過 $\mu_B$ 和 $\sigma^2_B$ 這兩個依賴於所有樣本的中間變數。如果忽略後者，模型會 explode（論文在實作初期確實觀察到這個問題）。

**種子論文怎麼處理？**

論文給出了完整的 chain rule 推導：

$$
\frac{\partial \ell}{\partial \hat{x}_i} = \frac{\partial \ell}{\partial y_i} \cdot \gamma
$$

$$
\frac{\partial \ell}{\partial \sigma^2_B} = \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} \cdot (x_i - \mu_B) \cdot \frac{-1}{2} (\sigma^2_B + \epsilon)^{-3/2}
$$

$$
\frac{\partial \ell}{\partial \mu_B} = \left( \sum_{i=1}^m \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma^2_B + \epsilon}} \right) + \frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{\sum_{i=1}^m -2(x_i - \mu_B)}{m}
$$

$$
\frac{\partial \ell}{\partial x_i} = \frac{\partial \ell}{\partial \hat{x}_i} \cdot \frac{1}{\sqrt{\sigma^2_B + \epsilon}} + \frac{\partial \ell}{\partial \sigma^2_B} \cdot \frac{2(x_i - \mu_B)}{m} + \frac{\partial \ell}{\partial \mu_B} \cdot \frac{1}{m}
$$

以及對 $\gamma, \beta$ 的梯度：
$$
\frac{\partial \ell}{\partial \gamma} = \sum_{i=1}^m \frac{\partial \ell}{\partial y_i} \cdot \hat{x}_i, \quad
\frac{\partial \ell}{\partial \beta} = \sum_{i=1}^m \frac{\partial \ell}{\partial y_i}
$$

BN 的 fully differentiable 設計讓它可以像普通網路層一樣插入任意位置，optimizer 完全不需要知道 BN 的存在——梯度自然會流經所有 normalization 參數。

---

### 知識點 5: 訓練 vs. 推理的雙模式設計

**這個知識點要回答什麼問題？**

訓練時用 mini-batch 統計量是可行的，但在推理時——特別是在 production 環境中——我們希望模型的輸出**僅依賴於單一輸入**，而不是 batch 內的其他樣本。同時，mini-batch 因隨機取樣而帶來的 variance 也不該出現在推理結果中。

**種子論文怎麼處理？**

BN 採用雙模式設計：

- **訓練模式**: 使用當前 mini-batch 計算 $\mu_B, \sigma^2_B$，正常 forward + backward
- **推理模式**: 使用訓練過程中累積的 **population 統計量**

Population 統計量的估計方式：
$$
\mathbb{E}[x] \leftarrow \mathbb{E}_B[\mu_B], \quad
\text{Var}[x] \leftarrow \frac{m}{m-1} \mathbb{E}_B[\sigma^2_B]
$$

其中 $\frac{m}{m-1}$ 是 Bessel's correction 用於得到無偏變異數估計。在實務中更常見的做法是用 moving average 追蹤 $\mu_B$ 與 $\sigma^2_B$。

一旦統計量固定，BN 在推理時就退化為一個簡單的線性變換，甚至可以與 $\gamma, \beta$ 合併成單一仿射變換：
$$
y = \frac{\gamma}{\sqrt{\text{Var}[x] + \epsilon}} \cdot x + \left( \beta - \frac{\gamma \mathbb{E}[x]}{\sqrt{\text{Var}[x] + \epsilon}} \right)
$$

這使得 BN 在推理時的計算成本幾乎為零。

---

### 知識點 6: BN 的連鎖效應

**這個知識點要回答什麼問題？**

BN 不只是讓訓練更穩定——它打開了一連串原本做不到的工程調整。

**種子論文怎麼處理？**

論文展示了 BN 帶來的幾個重要副作用：

**更高的 learning rate**: BN 讓 gradient 不受參數 scale 影響。對於任意 scalar $a$：
$$
\text{BN}(W u) = \text{BN}((aW)u), \quad
\frac{\partial \text{BN}((aW)u)}{\partial u} = \frac{\partial \text{BN}(W u)}{\partial u}
$$
這意味著更大的權重會產生更小的梯度（因為 $\frac{\partial \text{BN}((aW)u)}{\partial (aW)} = \frac{1}{a} \cdot \frac{\partial \text{BN}(W u)}{\partial W}$），形成一種自我穩定的機制。BN-x30（將 Inception 的 learning rate 提高 30 倍）不僅沒有 divergence，反而達到了更高的最終準確率。

**Dropout 可以移除或減弱**: BN 讓每個 training example 在 mini-batch 中的上下文不斷變化（每次和其他不同樣本一起標準化），這種隨機性本身就起到了正則化作用。論文中 BN-x5 移除了 Dropout，沒有出現 overfitting。

**Sigmoid 變得可用**: 由於 BN 控制了 activation 的分布，sigmoid 不再卡在飽和區。BN-x5-Sigmoid 達到了 69.8% 的 top-1 準確率，而原本的 Inception 用 sigmoid 始終停留在 0.1%（隨機猜測水準）。

**降低 L2 weight regularization**: BN-x5 將 L2 loss 的權重降低了 5 倍，因為 BN 本身已提供正則化效果。

---

## 實驗結果

### 主要實驗 (ImageNet 分類)

論文在 Inception 架構上測試了多種 BN 變體。核心結果：

| 模型 | 達到 72.2% 準確率所需步數 | 最高 top-1 準確率 |
|------|--------------------------|-------------------|
| Inception (baseline) | $31.0 \times 10^6$ | 72.2% |
| BN-Baseline (僅加 BN) | $13.3 \times 10^6$ | 72.7% |
| BN-x5 (BN + 加速技巧) | $\mathbf{2.1 \times 10^6}$ | 73.0% |
| BN-x30 (30 倍 learning rate) | $2.7 \times 10^6$ | **74.8%** |
| BN-x5-Sigmoid (sigmoid 非線性) | — | 69.8% |

**關鍵觀察**:
- 僅加入 BN（BN-Baseline）就讓訓練速度翻了 2.3 倍，且最終準確率更高
- BN-x5 只需 **7%**（$2.1/31 \times 10^6$）的訓練步數就達到 Inception 的最佳準確率
- BN-x30 雖然前期訓練較慢（因 learning rate 過大），但最終準確率達 74.8%，顯著超越 baseline
- BN-x5-Sigmoid 證明了 BN 讓 saturating nonlinearities 變得可用

### Ensemble 結果

6 個基於 BN-x30 的網路組成的 ensemble 達到了 top-5 validation error 4.9%、test error 4.82%，超越了當時 ImageNet 的最佳公開結果，並超過了人類 raters 的估計準確率。

### 消融實驗

BN 的各項改進（Sec. 4.2.1）都是互相加成的：
- 提高 learning rate: 最主要的加速來源
- 移除 Dropout: 不影響準確率，反而加速訓練
- 降低 L2 weight regularization: 小幅提升 validation accuracy
- 加速 learning rate decay: 配合更快的訓練節奏
- 徹底的訓練資料 shuffle: 約 1% 的 validation 準確率提升（體現 BN 的正則化特性）

---

## 與相關工作的對比

| 維度 | Batch Normalization | Inception (GoogLeNet) |
|------|-------------------|----------------------|
| 核心貢獻 | 訓練演算法：層間標準化 | 網路架構：Inception module |
| 解決的問題 | 訓練不穩定、收斂慢 | 計算效率、深度與寬度的權衡 |
| 是否需要架構調整 | 否（plug-and-play） | 是（需要 redesign 整個網路） |
| 對 gradient 的影響 | 直接穩定 gradient 傳播 | 透過 auxiliary classifiers 間接幫助 |
| 推理開銷 | 幾乎為零（合併為線性層） | 與訓練時相同 |
| 與其他方法的關係 | 可與任何架構搭配 | 可與 BN 等訓練技巧搭配 |

---

## 我的觀察

BN 這篇論文的影響力遠超出它的篇幅（11 頁）。回頭來看，它的關鍵洞見不是標準化本身——標準化的想法早在 LeCun 1998 就提出了——而是把標準化**做成一個可微分的網路層**，讓它成為 deep learning 基礎設施的一部分。

這個設計哲學值得留意：不是發明全新的數學工具，而是把既有概念以工程上優雅的方式實作出來，並證明它「可以 scale」。後來 LayerNorm、GroupNorm、RMSNorm 等都是沿著這個路線的延伸。

另外，BN 在 inference 時的雙模式設計雖然直覺，但在 production 中很容易出錯——如果 training 與 inference 的統計量不一致（例如 model 被 fine-tune 但 BN 層沒被更新），會出現奇怪的預測偏差。這在後來 LayerNorm 興起的背景下成為一個驅動力：LN 不需要 batch 統計量，訓練與推理的邏輯完全一致。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **Going Deeper with Convolutions** ([1409.4842](https://arxiv.org/abs/1409.4842))
   - 與本文關係: BN 測試的 baseline 架構，代表了 BN 出現前的 SOTA 設計

### 後續發展（僅列出，未涵蓋）

- *Layer Normalization* (Ba et al., 2016) — [1607.06450](https://arxiv.org/abs/1607.06450)：BN 在 RNN 上的替代方案，沿序列維度標準化
- *Group Normalization* (Wu & He, 2018) — [1803.08494](https://arxiv.org/abs/1803.08494)：BN 在 batch size 很小時的解決方案
- *Weight Normalization* (Salimans & Kingma, 2016) — [1602.07868](https://arxiv.org/abs/1602.07868)：直接標準化權重向量而非 activation
- *Instance Normalization* (Ulyanov et al., 2016) — [1607.08022](https://arxiv.org/abs/1607.08022)：用於風格遷移的標準化方式

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
