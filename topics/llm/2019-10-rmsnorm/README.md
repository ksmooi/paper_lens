# RMSNorm: Root Mean Square Layer Normalization 論文導讀

> **種子論文**: [Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) (2019-10)
> **作者**: Biao Zhang, Rico Sennrich
> **機構**: University of Edinburgh / University of Zurich

---

## TL;DR

Layer Normalization (LayerNorm) 是現代深度網路中不可或缺的元件，但其計算包含均值與變異數兩項統計量，帶來可觀的額外開銷。RMSNorm 提出一個大膽的假設——re-centering（減均值）對於訓練成功的貢獻其實不大——並據此設計了一個僅用 Root Mean Square (RMS) 統計量的簡化歸一化方法。實驗結果顯示，RMSNorm 在各種任務與架構上能達到與 LayerNorm 相當的品質，同時減少 7%–64% 的運算時間。

---

## 背景與動機

### 歸一化為什麼重要

深度神經網路的訓練面臨一個根本問題：隨著層數增加，各層 activation 的幅度會不受控地增長或縮小。這會導致：

- **梯度爆炸/消失**——反向傳播時，梯度訊號隨深度指數級衰減或放大
- **收斂緩慢**——參數更新步長難以調控，損失函數震盪
- **對初始化敏感**——需要精心設計的初始化策略才能成功訓練

歸一化（Normalization）透過在網路結構中加入統計量標準化的步驟，從根本上緩解了這些問題。

### 既有方法與其限制

在 RMSNorm 之前，主流的歸一化方法依序發展為：

1. **Batch Normalization (BatchNorm)** (Ioffe & Szegedy, 2015) ——對每個 neuron，跨 batch 維度計算均值與變異數。有效但依賴 batch size，不適合 RNN 或線上學習。
2. **Layer Normalization (LayerNorm)** (Ba, Kiros & Hinton, 2016) ——對同一層所有 hidden units 計算統計量，不受 batch size 限制，可直接用於 RNN。成為了 Transformer 架構的標準配備。

LayerNorm 雖然解決了 BatchNorm 的序列依賴問題，但它引入的計算開銷並非微不足道。尤其是當網路規模變大變深時（如現代 LLM），每一層都要計算均值與變異數的累積成本相當可觀。

### 核心問題：可以拿掉均值嗎？

Zhang & Sennrich 在閱讀 LayerNorm 的論文時注意到一個關鍵問題：LayerNorm 的兩個核心操作——re-centering（減均值）與 re-scaling（除標準差）——對於訓練成功的貢獻是否一樣大？更具體地說：

> **如果把減均值的步驟拿掉，只保留 RMS 縮放，模型還能好好訓練嗎？**

這個問題之所以合理，是因為：
- 均值的計算引入了額外的求和運算與減法，是計算開銷的主要來源之一
- 從梯度的角度來看，均值歸一化並未減少 hidden states 或梯度的變異數
- 在一些情況下，層 activation 的均值本來就接近零（例如經過特定初始化或 activation function 之後）

---

## 核心知識點

本文圍繞以下知識點展開：

1. **歸一化的動機與必要性**——深度網路為何需要歸一化
2. **Layer Normalization 的運作原理**——均值與變異數的雙重標準化
3. **Re-centering 是否多餘？**——RMSNorm 的核心假設與理論依據
4. **RMSNorm 的數學定義**——從 LayerNorm 到 RMSNorm 的簡化
5. **不變性分析比較**——各歸一化方法的 invariance properties
6. **梯度分析與隱式學習率調節**——RMSNorm 如何影響梯度動力學
7. **pRMSNorm：部分 RMS 估計**——進一步降低計算量的極致方案
8. **實證結果**——跨任務、跨架構的效能與速度權衡

---

## 方法詳解

### 知識點 1：歸一化的動機與必要性

**深度網路訓練中，layer activation 的幅度為何需要被控制？**

在一個標準的前饋網路中，第 $i$ 個 neuron 的 summed input $a_i$ 為：

$$
a_i = \sum_{j=1}^{m} w_{ij} x_j
$$

隨著層數增加，$a_i$ 的分佈會因為前一層權重的更新而持續變化——這個現象被稱為 **internal covariate shift**。當 $a_i$ 的幅度過大時，activation function（如 sigmoid、tanh）會進入飽和區，梯度接近零；當幅度太小時，訊號會在下游被稀釋。

歸一化透過將 $a_i$ 的統計量（均值、變異數、或 RMS）固定到某個範圍，讓網路在訓練過程中 activation 的分佈保持穩定。但值得注意的是，後續研究（如 Santurkar et al., 2018）指出，歸一化的成功可能不僅僅來自於減少 internal covariate shift，更來自於它平滑化了優化 landscape。

### 知識點 2：Layer Normalization 的運作原理

**LayerNorm 是怎麼做的？**

給定同一層 $n$ 個 neuron 的 summed inputs $\mathbf{a} = [a_1, a_2, \dots, a_n]$，LayerNorm 計算：

$$
\mu = \frac{1}{n} \sum_{i=1}^{n} a_i, \quad \sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (a_i - \mu)^2
$$

標準化後的輸出為：

$$
\bar{a}_i = \frac{a_i - \mu}{\sigma} \cdot g_i, \quad y_i = f(\bar{a}_i + b_i)
$$

其中 $g_i$ 與 $b_i$ 是可學習的 gain 與 bias 參數，用於恢復網路的表達能力。

LayerNorm 的關鍵特性：
- 對同一層所有 neuron 共享 $\mu$ 與 $\sigma$，但不同訓練樣本各自計算
- 不需要跨 batch 的統計量，因此適用於 batch size = 1 的純線上學習
- 在 RNN 中，每個 time step 獨立計算統計量，不受序列長度影響

### 知識點 3：Re-centering 是否多餘？

**RMSNorm 論文的核心假設**

Zhang & Sennrich 的關鍵觀察是：

> LayerNorm 的 re-centering invariance——即對輸入或權重矩陣加入一個偏移量時，歸一化後的輸出保持不變——**其對訓練成功的貢獻可能被高估了**。

他們的論證如下：

1. **均值歸一化不減少梯度變異數**：減均值只平移了 activation 的分佈，但 variance 依然存在
2. **若 activation 本身均值接近零**：當使用 zero-mean 的初始化或特定 activation（如經過 LayerNorm 後的 tanh 輸出），re-centering 幾乎沒有實際效果
3. **計算成本不成比例**：計算均值需要額外的求和與減法，對於大型網路而言，累積的開銷相當可觀

這個假設如果成立，意味著我們可以 **安全地移除 re-centering 步驟，只保留 re-scaling**。

### 知識點 4：RMSNorm 的數學定義

**RMSNorm 如何簡化 LayerNorm？**

RMSNorm 移除了 LayerNorm 中的均值計算，僅保留 Root Mean Square 統計量：

$$
\bar{a}_i = \frac{a_i}{\text{RMS}(\mathbf{a})} \cdot g_i, \quad \text{RMS}(\mathbf{a}) = \sqrt{\frac{1}{n} \sum_{i=1}^{n} a_i^2}
$$

直觀理解：
- RMSNorm 將 summed inputs 投影到一個 $n$ 維單位球面上
- 當 summed inputs 的均值為零時，RMSNorm 與 LayerNorm 完全等價
- 由於移除了均值計算，RMSNorm 的計算量比 LayerNorm 更少
- 與歐幾里得範數 ($\|\mathbf{a}\|_2 = \sqrt{\sum a_i^2}$) 不同，RMS 除以 $\sqrt{n}$，讓歸一化對不同大小的輸入向量更穩健

**實作角度**：RMSNorm 可以直接作為 LayerNorm 的 drop-in replacement——不需要修改網路架構或訓練流程，只需要將歸一化層換掉即可。

### 知識點 5：不變性分析比較

**各歸一化方法有什麼不同？**

論文中對比了四種歸一化方法的不變性（invariance）：

| 變換 | BatchNorm | WeightNorm | LayerNorm | RMSNorm |
|------|-----------|------------|-----------|---------|
| 權重矩陣 re-scaling | ✓ | ✓ | ✓ | ✓ |
| 權重矩陣 re-centering | ✗ | ✗ | ✓ | ✗ |
| 單一權重向量 re-scaling | ✓ | ✓ | ✗ | ✗ |
| 資料集 re-scaling | ✓ | ✓ | ✓ | ✓ |
| 資料集 re-centering | ✓ | ✗ | ✗ | ✗ |
| 單一訓練樣本 re-scaling | ✗ | ✗ | ✓ | ✓ |

從表格可以清楚看到：
- **RMSNorm 保留了 re-scaling invariance**：對權重矩陣縮放、輸入縮放、資料集縮放不變——這得益於 RMS 的線性性質 $\text{RMS}(\delta \mathbf{a}) = \delta \cdot \text{RMS}(\mathbf{a})$
- **RMSNorm 失去了 re-centering invariance**：對偏移量沒有線性性質，因此不對權重或輸入的平移不變
- **與 LayerNorm 的關鍵差異**：RMSNorm 不具備 weight matrix re-centering invariance，但論文透過實驗證明這個差異不影響最終效能

### 知識點 6：梯度分析與隱式學習率調節

**RMSNorm 的梯度有什麼特性？**

論文對 RMSNorm 進行了詳細的梯度分析，發現幾個有趣的性質：

1. **gain 與 bias 的梯度對輸入縮放不變**：
   $$
   \frac{\partial \mathcal{L}}{\partial b} = \frac{\partial \mathcal{L}}{\partial \mathbf{v}}, \quad
   \frac{\partial \mathcal{L}}{\partial g} = \frac{\partial \mathcal{L}}{\partial \mathbf{v}} \odot \frac{\mathbf{Wx}}{\text{RMS}(\mathbf{a})}
   $$

2. **權重矩陣的梯度具有隱式學習率調節**：
   $$
   \frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \mathbf{x}^\top \cdot \text{diag}\left(\frac{\mathbf{g}}{\text{RMS}(\mathbf{a})}\right) \odot \frac{\partial \mathcal{L}}{\partial \mathbf{v}}
   - \frac{1}{n \cdot \text{RMS}(\mathbf{a})^2} (\mathbf{Wx})(\mathbf{Wx})^\top \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{W}}
   $$

   關鍵的發現是：梯度的第二項（含有 $-\frac{1}{n \cdot \text{RMS}(\mathbf{a})^2}$ 因子）與輸入縮放呈負相關。當權重矩陣的範數變大時，這個項會「踩剎車」，動態地降低有效學習率。這形成了 **隱式的學習率自適應機制**，有助於穩定訓練。

3. **對輸入縮放不變**：權重矩陣的梯度對輸入的縮放不變，提升了優化的穩定性。

### 知識點 7：pRMSNorm：部分 RMS 估計

**能否進一步降低計算量？**

pRMSNorm 利用了層內神經元近似獨立同分佈（i.i.d.）的特性：既然神經元共享類似的統計性質，何不只用其中一部分來估算 RMS？

pRMSNorm 的計算方式：只取前 $p\%$ 的 summed inputs 來計算 RMS：

$$
\widetilde{\text{RMS}}(\mathbf{a}) = \sqrt{\frac{1}{k} \sum_{i=1}^{k} a_i^2}, \quad k = \lfloor n \cdot p \rfloor
$$

由於 RMS 的線性性質仍然成立 $\widetilde{\text{RMS}}(\delta \mathbf{a}) = \delta \cdot \widetilde{\text{RMS}}(\mathbf{a})$，pRMSNorm 保留了與 RMSNorm 完全相同的不變性。

論文的實驗顯示，即使只使用 **6.25%** 的神經元來估算 RMS，pRMSNorm 仍能取得與完整 RMSNorm 競爭的表現。不過論文中也坦承，pRMSNorm 的理論加速未必能在實際硬體上體現，因為 `k` 過小時的梯度可能不穩定，且 GPU 的 kernel launch overhead 可能吃掉節省下來的計算時間。

### 知識點 8：實證結果

**RMSNorm 在實際任務中的表現如何？**

**機器翻譯（WMT14 En→De）**

| 模型 | Test14 (BLEU) | Test17 (BLEU) | 訓練時間 |
|------|:------------:|:------------:|:--------:|
| Baseline (無歸一化) | 21.7 | 23.4 | 399s |
| LayerNorm | 22.6 | 23.6 | 665s |
| L2-Norm (歐幾里得範數) | 20.7 | 22.0 | 482s |
| **RMSNorm** | 22.4 | 23.7 | **501s (快 24.7%)** |
| pRMSNorm (6.25%) | 22.6 | 23.1 | **493s (快 25.9%)** |

RMSNorm 在 BLEU 分數上與 LayerNorm 相當（甚至略高），但訓練速度快了 24.7%。

**影像分類（CIFAR-10）**

| 方法 | 測試錯誤率 | 每 epoch 時間 |
|------|:----------:|:------------:|
| Baseline | 8.96% | 21s |
| BatchNorm | 8.25% | 38s |
| WeightNorm | 8.28% | 23s |
| LayerNorm | 10.49% | 39s |
| **RMSNorm** | 8.83% | **31s (快 20.5%)** |

值得注意的是，在影像分類任務上，LayerNorm 的表現意外地差（錯誤率 10.49%，甚至不如不歸一化的 Baseline 8.96%）。而 RMSNorm 不僅比 LayerNorm 快了 20.5%，還取得了更好的泛化能力。這暗示了 **re-centering 在某些任務上可能有害**，因為它強制將 activation 置零，反而破壞了影像特徵中重要的偏移資訊。

**圖文檢索（MS COCO）**

在 Order-Embedding (OE) 模型上，RMSNorm 在 Recall@K 指標上與 LayerNorm 相當或略優，訓練更穩定。

**問答任務（CNN/Daily Mail）**

在閱讀理解任務上，RMSNorm 與 LayerNorm 的表現幾乎一致。

---

## 實驗結果

### 主要發現總結

| 面向 | 結論 |
|------|------|
| **任務品質** | RMSNorm 與 LayerNorm 在所有任務上表現相當，無統計顯著差異 |
| **訓練速度** | 可比 LayerNorm 快 7%–64%，視框架、硬體與架構而定 |
| **泛化能力** | 在某些任務（如 CIFAR-10）上，RMSNorm 的泛化能力優於 LayerNorm |
| **Drop-in replacement** | 可在不解釋既有程式碼的前提下直接替換 LayerNorm 層 |

### 速度提升的來源

速度提升主要來自兩個方面：

1. **計算量減少**：移除了均值計算中的求和與減法運算
2. **記憶體頻寬節省**：少了一個統計量的讀寫

實際加速幅度與以下因素相關：
- **歸一化層佔總計算的比例**：歸一化在 RNN 中佔比更高（因為矩陣乘法相對較小），速度優勢最明顯
- **框架實作效率**：在 TensorFlow、PyTorch、Theano 上速度提升不一
- **硬體特性**：GPU kernel launch overhead 會稀釋小運算的加速效果

### 消融實驗

論文的消融實驗揭示了幾個重要發現：

- **L2-Norm 不可行**：用歐幾里得範數 $\|a\|_2$ 替代 RMS（即不除以 $\sqrt{n}$）會導致訓練失敗——說明除以 $\sqrt{n}$ 這個尺度因數對跨維度的穩健性至關重要
- **pRMSNorm 的極致節省**：僅用 6.25% 的神經元估算 RMS 就能達到接近完整 RMSNorm 的表現，這驗證了層內神經元近似 i.i.d. 的假設
- **WeightNorm 競爭力不足**：在機器翻譯任務上，WeightNorm 收斂較慢且最終 BLEU 分數較低

---

## 我的觀察

### 為什麼 RMSNorm 後來成為了主流

這篇論文發表於 2019 年，但 RMSNorm 真正「出圈」是在 2023–2024 年間，隨著 LLaMA 系列的發布。LLaMA 使用 RMSNorm 代替 LayerNorm 的設計選擇，被後續大量開源 LLM 沿用（如 Mistral、Gemma、Qwen 等）。

我認為 RMSNorm 的成功有幾個關鍵因素：

1. **時間點恰到好處**：當模型規模從數億參數增長到數十億、數百億參數時，每一個歸一化層的計算開銷都被放大。RMSNorm 帶來的小比例加速在超大模型上意義重大。

2. **簡潔即力量**：RMSNorm 的公式比 LayerNorm 更簡單，在實作上更容易與 flash attention、kernel fusion 等優化技術配合。

3. **論文論證清晰**：這篇論文非常適合做為「如何做 ablation study」的範例——它提出一個清晰的假設（re-centering 可有可無），然後用理論與實驗雙重驗證。

### 一個被低估的貢獻：不變性分析

論文中 Table 1 的不變性分析雖然簡單，但在教學與研究上都很有價值。它提供了一個框架來理解不同歸一化方法的本質差異：**哪些變換不影響模型的輸出**。這個視角在後續的 normalization 研究（如 SandwichNorm、QK-Norm）中被廣泛沿用。

### 侷限與開放問題

- **為什麼 LayerNorm 在 CIFAR-10 上表現這麼差？** 論文中未深入探討。這可能與 ConvNet 中 activation 的統計特性有關——在空間維度上做 LayerNorm 的 re-centering 可能破壞了影像的區域對比度資訊。
- **pRMSNorm 的實際加速未達預期**：雖然理論上 pRMSNorm 應該更快，但作者的實測中速度差異不大。這在 GPU 上很常見——小運算的瓶頸在於 kernel launch latency，而不是算術吞吐量。
- **更大的模型？** 論文未在當時最大規模的模型上測試 RMSNorm。當時（2019 年）的 Transformer 規模遠小於今天的標準。

---

## 延伸閱讀

### Dependency Papers

1. **Layer Normalization** ([1607.06450](https://arxiv.org/abs/1607.06450))
   - 與本文關係：RMSNorm 的直接前身與對比基準

### 後續發展

- [LLaMA: Open and Efficient Foundation Language Models](https://arxiv.org/abs/2302.13971) (2023-02) ——率先在 LLM 中使用 RMSNorm 取代 LayerNorm
- [Sandwich Norm: How to Make Your Transformer More Normalized?](https://arxiv.org/abs/2310.07382) (2023-10) ——在 LayerNorm/RMSNorm 基礎上探索新的歸一化位置
- [QK-Norm: Pre-layer normalization for Query and Key in Attention](https://arxiv.org/abs/2307.11095) (2023-07) ——在 Attention 內部新增額外的歸一化層

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
