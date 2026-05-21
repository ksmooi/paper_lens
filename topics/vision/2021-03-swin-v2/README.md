# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

> **種子論文**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (2021-03)
> **作者**: Ze Liu, Yutong Lin, Yue Cao et al.
> **機構**: Microsoft Research Asia

> **Dependency Paper**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT (Dosovitskiy et al., Google, 2020-10)

---

## TL;DR

> ViT 把 Transformer 搬進視覺，但 global self-attention 的計算量隨圖像大小呈二次方成長，且只能產出單一尺度的特徵圖，無法勝任物體檢測、語意分割等密集預測任務。Swin Transformer 提出 shifted window 機制，將 self-attention 限制在 local windows 內計算，把複雜度降到與圖像大小成線性關係；再配合 patch merging 建構階層式 (hierarchical) 特徵圖，可無縫替換 CNN backbone。在 ImageNet 分類 (87.3%)、COCO 檢測 (58.7 box AP) 與 ADE20K 分割 (53.5 mIoU) 上，Swin Transformer 全面超越 ViT/DeiT 與 ResNe(X)t，證明了純 Transformer 架構作為通用視覺骨幹的潛力。

---

## 背景與動機

### Transformer 進軍視覺前的狀態

在 Swin Transformer 出現之前，計算機視覺的骨幹網路長期由卷積神經網路 (CNN) 主導。從 2012 年 AlexNet 的突破開始，後續的 VGG、ResNet、DenseNet、HRNet 到 EfficientNet，CNN 架構在分類、檢測、分割等任務上不斷提升效能。CNN 的核心優勢來自兩個歸納偏置 (inductive bias)：

1. **局部性 (locality)**：卷積核只在局部感受野內運算，參數量與輸入解析度無關
2. **平移等變性 (translation equivariance)**：物體在圖像中移動，其對應的特徵圖也隨之平移，不會因為位置改變而讓模型無法辨識

這兩個性質讓 CNN 在中等規模的資料集上就能有效學習視覺特徵，不需要外部資料。但 CNN 也有先天限制：感受野的大小取決於卷積核尺寸與網路深度。雖然可以透過堆疊更多層來擴大感受野，但長程依賴關係 (long-range dependencies) 的建模效率始終不如 attention 機制直接。

### ViT 的突破：Transformer 遇上圖像

2020 年 10 月，Google 團隊發表了 ViT (Vision Transformer)，首次證明**純 Transformer 架構——不加任何卷積——可以直接用於圖像分類**。ViT 的做法很直接：把一張圖像切成固定大小的 patches（例如 16×16），每個 patch 壓平成向量後線性投影到 Transformer 的隱藏維度，再加上可學習的位置編碼，然後餵進標準的 Transformer encoder。這個設計幾乎是把 NLP 的 Transformer 原封不動搬過來，差別只在於輸入從 word tokens 變成了 image patches。

ViT 的成功揭示了兩個重要發現：

- Transformer 的 global self-attention 可以在視覺任務上有效學習，**只要資料量夠大**
- 大規模預訓練（例如 Google 內部的 JFT-300M）可以克服 Transformer 缺乏 CNN 歸納偏置的問題

ViT 在 ImageNet 上達到了與當時 SOTA CNN 相當的效能，但在 JFT-300M 預訓練後，ViT-H/14 以 88.55% 超越了當時所有基線——包括訓練成本高出數倍的 BiT-L (ResNet152x4)。

### ViT 的兩大架構瓶頸

但 ViT 有兩個根本性的架構限制，讓它無法成為通用視覺骨幹：

**瓶頸一：單一解析度的特徵圖**

ViT 在整個 encoder 中維持固定的 patch 數 N = HW/P²，始終只有單一解析度的特徵表示。這在圖像分類任務上沒問題，因為只需要一個 global representation。但在物體檢測 (object detection) 和語意分割 (semantic segmentation) 這類密集預測任務中，多尺度特徵至關重要。當時 SOTA 的檢測框架（如 FPN（特徵金字塔網路）、Mask R-CNN）都依賴 CNN backbone 提供的階層式特徵圖。ViT 無法直接替代 CNN backbone，因為它產出的特徵圖只有 H/16 × W/16 一個解析度。

如果用 ViT 做密集預測，只能對 H/16 × W/16 的特徵圖直接 upsampling 或 deconvolution，但這樣做完全無法捕捉小物體的細節資訊——因為小的物體在 16× 下採樣後可能只剩下不到一個 patch。

**瓶頸二：二次方計算複雜度**

Standard multi-head self-attention 的計算量是 O(N²·D)，其中 N = HW/P² 是 patch 數、D 是特徵維度。讓我們把這個複雜度量化的精確一點：

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

這裡 h 和 w 是特徵圖的高度與寬度，C 是 embedding 維度。第一項 4hwC² 來自 Q、K、V 的線性投影以及 attention 輸出後的線性投影。第二項 2(hw)²C 來自 QKᵀ 矩陣乘法（hw×hw）以及 attention 權重與 V 的相乘——這正是二次方項的來源。

對 224×224 的圖像使用 16×16 patch 時，hw = 196，這項的值約為 2 × 38,416 × C。但對於密集預測任務常用的高解析度輸入（例如檢測框架的 800×1333），hw ≈ 50 × 83 = 4,150，(hw)² 則暴增到約 17.2M。global self-attention 在這種輸入尺寸下完全無法運算。

DeiT 等後續工作從訓練策略層面改進 ViT（知識蒸餾、資料增強、更長訓練排程），但這兩個架構層次的問題並未解決。

### 從 NLP 到視覺：「遷移」面臨的核心挑戰

把 Transformer 從 NLP 遷移到視覺，面臨兩個根本差異：

1. **尺度變異 (scale variation)**：視覺中的物體大小差異極大。一張圖像裡可能同時有佔據大半張圖的大象（需要高層語義特徵）和只有幾十個 pixels 的蒼蠅（需要淺層高解析度特徵）。NLP 中的 word tokens 基本是固定粒度的——一個 word token 就是一個詞，沒有「放大縮小」的概念。

2. **解析度 (resolution)**：圖像的 pixel 數量遠多於文字的 token 數量。一張 224×224 的圖像有 50,176 個 pixels，已經是大部分 NLP 模型輸入序列長度（512–2048）的 25–100 倍。而密集預測任務的輸入解析度（如 800×1333）更讓這個差距進一步拉大。

這兩個問題並非 Swin Transformer 才發現。在 Swin 之前，已經有不少工作嘗試用 self-attention 取代或補充卷積層（例如 Non-local Neural Networks、Stand-Alone Self-Attention、Axial-Attention）。但它們要麼因為 sliding window 的實作效率不佳（記憶體存取瓶頸），要麼因為 global attention 的二次方複雜度，都無法成為實用的通用骨幹。

具體來說，Non-local Neural Networks (Wang et al., 2018) 在 CNN 的特徵圖上加入 global self-attention 來捕捉長程依賴，但它的計算量也是二次方的，只能在不深的位置（例如 stage 3 之後的 14×14 特徵圖）使用一到兩個 non-local blocks。Stand-Alone Self-Attention (Ramachandran et al., 2019) 將 local self-attention 視為卷積的替代品，但採用 sliding window 實作——每個 query pixel 對應不同的 key set，無法 batch 加速。Swin 的 shifted window 方案在建模能力上與這些工作本質相同（local attention + cross-window propagation），但在工程效率上透過 cyclic shift 實現了壓倒性優勢。

### Shifted Window 的設計來源

Swin 的 shifted window 概念並非憑空出現。在圖像處理領域，循環平移視窗的技巧早在局部二值模式 (Local Binary Patterns, LBP) 和離散小波變換中就有類似想法——即在相鄰的處理層之間改變取樣或劃分方式，以覆蓋原本被隔離的區域。Swin 的貢獻在於將這個概念系統性地應用在 Transformer 的 self-attention 中，並提供了一個高效的 batch 計算方案。

Swin Transformer 的兩個核心設計——**hierarchical feature maps** 與 **shifted window self-attention**——正是為了解決這兩個根本差異而提出的。第一個設計解決尺度變異問題，第二個設計解決高解析度下的計算複雜度問題。

---

## 核心知識點

本文圍繞以下知識點展開，它們是理解 Swin Transformer 及其與 ViT 對比的關鍵：

1. **Patch-Based Tokenization 與階層式特徵圖建構**——ViT 用固定大 patch + 單一解析度，Swin 用小 patch + patch merging 產生 4 層解析度，與 ResNet 的階層式設計完全對齊
2. **Local Window Self-Attention (W-MSA)**——將 self-attention 限制在非重疊視窗內，讓計算複雜度從二次方 O((hw)²) 降為線性 O(M²·hw)
3. **Shifted Window Partitioning (SW-MSA)**——在連續層之間交替搬動視窗分割，在不增加計算量的前提下建立跨視窗連結
4. **Cyclic Shift + Masked MSA**——解決 shifted window 造成的視窗大小不均與計算量暴增問題，一個巧妙的工程解法
5. **Relative Position Bias**——以學習的相對位置偏置取代 ViT 的絕對位置編碼，對 local window attention 更有效
6. **架構變體與跨任務實驗結果**——Swin 的四種規模變體，以及在分類、檢測、分割三大任務上的全面超越

---

## 方法詳解

### 知識點 1: Patch-Based Tokenization 與階層式特徵圖建構

**這個知識點要回答什麼問題？**

傳統 CNN（以 ResNet 為代表）透過卷積層與池化層逐步降低解析度、增加通道數，形成 4×、8×、16×、32× 四個階層的特徵圖。ViT 的全連接層 encoder 從頭到尾維持相同解析度與通道數。Swin 如何兼顧 Transformer 的建模能力與 CNN 的階層式結構？

**ViT 的處理方式**

ViT 將輸入圖像 x ∈ ℝ^(H×W×C) 切成 N = (H/P) × (W/P) 個 patches，每個 patch 大小為 P×P（預設 16×16）。每個 patch 被 flatten 成 P²·C 維向量後，經由可訓練的線性投影（式中的 E 矩陣）映射到 D 維 embedding 空間：

$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \cdots; x_p^N E] + E_{\text{pos}}$$

其中 E ∈ ℝ^((P²·C)×D) 是 patch embedding 投影矩陣，E_pos ∈ ℝ^((N+1)×D) 是可學習的位置編碼，x_class 是額外添加的 [CLS] token（用於分類的向量）。

ViT encoder 由 L 層交替的 Multi-Head Self-Attention (MSA) 與 MLP 區塊構成，每一層都維持 N+1 個 token、D 維 embedding。沒有下採樣，沒有解析度變化。

這帶來的問題很直接：密集預測任務需要的多尺度特徵圖不存在。如果要把 ViT 用於檢測或分割，只能從 H/16 × W/16 的單一特徵圖直接 upsampling，無法使用 FPN 這類多尺度架構。

**Swin 的處理方式：四階段階層式設計**

Swin 做了三個關鍵改變：

**第一步：更小的初始 patch。** 從 ViT 的 16×16 改為 4×4。初始 patch 數從 (224/16)² ≈ 196 暴增到 (224/4)² = 3136。每個 patch 的 raw feature 維度為 4×4×3 = 48，經線性層投影到 C 維（Swin-T 的 C=96）。

**第二步：Patch Merging 模組。** 這就是 Swin 的「下採樣機制」，作用類似卷積網路中的 stride-2 convolution。假設輸入特徵圖解析度為 h×w，embedding 維度為 C：

- 對每個 2×2 的鄰近 patch 區塊，將 4 個 C 維特徵拼接成 4C 維
- 經線性層 (4C → 2C) 降維
- 輸出解析度變為 h/2 × w/2，通道數變為 2C

等價於：解析度降為一半，通道數翻倍——與 ResNet 中每個 stage 結束時的操作完全相同。

**第三步：階段式建構（4 Stages）。** Swin 總共有 4 個 stages，每個 stage 內有若干 Swin Transformer blocks（數量依模型變體而異），並在 stage 之間插入 patch merging 層：

```
Stage 1: 解析度 H/4 × W/4, 通道數 C,  2 或 4 個 blocks
Stage 2: 解析度 H/8 × W/8, 通道數 2C, 2 或 18 個 blocks
Stage 3: 解析度 H/16 × W/16, 通道數 4C, 6 或 18 個 blocks
Stage 4: 解析度 H/32 × W/32, 通道數 8C, 2 個 blocks
```

Four output resolutions: H/4 × W/4, H/8 × W/8, H/16 × W/16, H/32 × W/32。這與 ResNet 的 C2–C5 特徵圖解析度完全一致。這讓 Swin 可以直接插入任何接受 ResNet 作為 backbone 的框架中（如 Mask R-CNN、Cascade R-CNN、FPN），完全不需要修改檢測頭。

**實作細節：Patch Merging 的內部機制**

Patch merging 的實作比聽起來更直接。假設輸入特徵圖為 X ∈ ℝ^(h×w×C)，patch merging 的步驟為：

1. **重整 (reshaping)**：將 X 重新組織為 (h/2, 2, w/2, 2, C)，即從每個 2×2 鄰域取出 4 個特徵向量
2. **拼接 (concatenation)**：合併最後兩維，得到 (h/2, w/2, 4C)
3. **線性投影**：4C → 2C 的全連接層，得到最終輸出 (h/2, w/2, 2C)

這個操作不需要任何 padding、stride 或卷積核，是純粹的資料重排 + 線性變換。這與 CNN 中 stride-2 convolution（通常需要學到如何下取樣）不同——patch merging 是確定性的操作，沒有任何可學習的參數用於下取樣本身（可學習的部分只有後續的 4C→2C 線性投影）。

**為什麼小 patch size 是合理的？**

Swin 使用 4×4 的初始 patch size，相比 ViT 的 16×16，初始 token 數多了 16 倍（(16/4)²）。這會不會讓計算量太大？

關鍵在於：初始階段使用 W-MSA（local window），而不是 global MSA。雖然 token 數是 3,136（對 224² 輸入），但 W-MSA 的計算量只與 hw·M² 成正比，而 M² = 49 是一個很小的常數。所以儘管初始 token 數比 ViT 多了 16 倍，計算量仍然是可控的。這正是 W-MSA 的威力所在：讓 Transformer 可以從高解析度的特徵圖開始工作，然後逐步下取樣。

**ViT vs Swin 的關鍵差異表**

在深入技術細節之前，先用一張完整的對比表來總結兩個架構在設計哲學上的差異。這張表涵蓋了後面每個知識點會展開的面向：

| 維度 | ViT | Swin Transformer |
|------|-----|-----------------|
| 初始 patch size | 16×16 | 4×4 |
| 初始 patch 數 (224²) | ~196 | ~3,136 |
| 特徵圖解析度變換 | 無（固定） | 4× → 8× → 16× → 32× |
| 通道數變化 | 固定 D | 每 stage 翻倍 |
| 注意力範圍 | global（所有 patches） | local window（M×M patches） |
| 注意力計算複雜度 | O((hw)²) 二次方 | O(M²·hw) 線性 |
| 跨區域資訊流動 | 同層內完成 | 跨層交替輪換 |
| 位置編碼 | 絕對 1D 可學習 (加在輸入) | 相對位置偏置 (加在 attention logits) |
| 可替換 CNN backbone | 不可直接替代 | 可直接替換（特徵圖拓撲相同）|
| 支援 FPN/U-Net | 需要 upsampling 方式迂迴 | 原生支援（多解析度輸出） |
| 與 CNN 的逼近程度 | 低（無階層、無局部性） | 高（完全對齊 ResNet 拓撲 + 局部 attention）|
| 分類 DataScale 依賴 | 高度依賴大規模預訓練 | ImageNet-1K 即可超越 CNN 基線 |
| 對密集預測任務 | 弱（單尺度限制） | 強（階層式特徵 + cross-window）|

這個設計直接回答了前述的「尺度變異」問題：小物體的資訊在高解析度的淺層 (Stage 1–2) 被保留，大物體與高層語義資訊在低解析度的深層 (Stage 3–4) 被提煉——與人類視覺系統以及 CNN 的多尺度設計異曲同工。

---

### 知識點 2: Local Window Self-Attention (W-MSA)

**這個知識點要回答什麼問題？**

Global self-attention 的二次方複雜度讓 Transformer 在高解析度輸入下無法運算。Swin 如何在保持 attention 建模能力的同時，把複雜度降下來？

**標準 MSA 的計算複雜度推導**

在深入 W-MSA 之前，先仔細推導 global MSA 的計算量。對一個有 N = h×w 個 tokens、每個 token 維度為 C 的輸入特徵圖 X ∈ ℝ^(N×C)：

1. **Q、K、V 投影**：X 分別乘以 W^Q、W^K、W^V（三個 C×C 矩陣），得到 Q、K、V ∈ ℝ^(N×C)。計算量：3 × N × C² = 3NC²。
2. **QKᵀ 計算**：QKᵀ ∈ ℝ^(N×N)，計算量：N²C。
3. **Softmax 後乘 V**：A = Softmax(QKᵀ/√d) · V，計算量：N²C（softmax 本身的 N² 次 exp 通常忽略）。
4. **輸出投影**：Attention 輸出再乘一個 C×C 矩陣，計算量：NC²。

總計：Ω(MSA) = 3NC² + N²C + N²C + NC² = 4NC² + 2N²C。

以 h 和 w 表示，N = hw：

$$\Omega(\text{MSA}) = 4hwC^2 + 2(hw)^2C$$

關鍵項是 2(hw)²C——它來自 QKᵀ 和 AV 兩次矩陣乘法，都是 N² 等級。

**W-MSA 的計算量**

W-MSA 將 h×w 的特徵圖均勻劃分為非重疊的視窗，每個視窗大小為 M×M。視窗總數為 ⌈h/M⌉ × ⌈w/M⌉。M=7 時，每個視窗有 49 個 patches。

在每個視窗內計算注意力：7×7 = 49 個 tokens 的 MSA，計算量為 Ω(MSA_local) = 4 × 49 × C² + 2 × 49² × C = 196C² + 4802C。

總共有大約 (hw)/49 個這樣的視窗，所以總計算量為：

$$\Omega(\text{W-MSA}) = 4 \times 49 \times C^2 \times \frac{hw}{49} + 2 \times 49^2 \times C \times \frac{hw}{49}$$
$$= 4hwC^2 + 2 \times 49 \times hw \times C$$
$$= 4hwC^2 + 98hwC$$

更通用的形式：（M 可調）

$$\Omega(\text{W-MSA}) = 4hwC^2 + 2M^2 hwC$$

請注意，2(hw)²C（二次方項）變成了 2M²hwC（線性項）。差別在於：

- MSA 的二次方項係數是 hw（從 QKᵀ 的 hw×hw 矩陣來），隨輸入解析度放大而放大
- W-MSA 的係數是 M²（固定常數，預設 49），完全不受輸入解析度影響

**數字範例**

假設 C=96（Swin-T）、hw 對應第三個 stage 的 h=w=28（16× 解析度，patch 數 784）：

- MSA 第二項：2 × 784² × 96 = 2 × 614,656 × 96 = 118,013,952 FLOPs
- W-MSA 第二項：2 × 49 × 784 × 96 = 2 × 49 × 75,264 = 7,375,872 FLOPs

W-MSA 的 attention 計算量約為 MSA 的 6.3%。若解析度翻倍（h=w=56，對應 32× 解析度下 56×56 的特徵圖大小），MSA 第二項暴增 4 倍（472M），W-MSA 僅增 2 倍（14.8M），差距擴大到約 3.1%——這就是線性與二次方的實質差距。

**W-MSA 與 MSA 的計算量對比表**

以下用具體數字展示在不同特徵圖大小下，W-MSA (M=7) 與 global MSA 第二項計算量的比例關係。假設 C=96，單位為 MFLOPs：

| 特徵圖大小 (h×w) | patch 總數 N | MSA 第二項 (MFLOPs) | W-MSA 第二項 (MFLOPs) | 比例 |
|----------------|------------|-------------------|---------------------|------|
| 14×14 (Stage 4) | 196 | 2 × 196² × 96 = 7.4M | 2 × 49 × 196 × 96 = 1.8M | 24.7% |
| 28×28 (Stage 3) | 784 | 2 × 784² × 96 = 118.0M | 2 × 49 × 784 × 96 = 7.4M | 6.3% |
| 56×56 (Stage 2) | 3,136 | 2 × 3136² × 96 = 1,888M | 2 × 49 × 3136 × 96 = 29.5M | 1.6% |
| 112×112 (Stage 1) | 12,544 | 2 × 12544² × 96 = 30,212M | 2 × 49 × 12544 × 96 = 118.1M | 0.4% |

對於 Stage 1 的 112×112 特徵圖（patch 數 12,544），global MSA 的第二項已達到 30G FLOPs——這對單層來說已經是天文數字。W-MSA 則將它降到 118M，降低了 256 倍。更重要的是，如果輸入解析度再翻倍（例如從 224² 到 448²），MSA 的計算量會增加 4 倍（二次方），而 W-MSA 只增加 2 倍（線性）。這使得 W-MSA 能優雅地支援高解析度輸入，而 global MSA 在 224² 以上就幾乎無法使用。

**代價：跨視窗連通性中斷**

W-MSA 雖然省下了可觀的計算量，但代價是 self-attention 被局限在 7×7 的 local window 內，失去了跨區域的長程依賴建模能力。ViT 的 global attention 之所以對視覺任務有效，正是因為它允許任何 patch 關注到所有其他 patches，感受野等於整張圖。

但 Swin 的論點是：視覺中相鄰 pixel 的相關性遠高於相距很遠的 pixel（這是為什麼 CNN 的局部卷積有效的原因），因此 local window 已經能捕捉大部分的視覺特徵。所失去的跨視窗資訊，可以透過下一層的 shifted window 來回復。

這就是下一個知識點的主題。

---

### 知識點 3: Shifted Window Partitioning (SW-MSA)

**這個知識點要回答什麼問題？**

W-MSA 犧牲了跨視窗的資訊交換——視窗邊界上的 patches 無法看到視窗之外的內容。如何在**不增加計算負擔**的前提下重新建立跨視窗連通性？

**核心洞見：交替視窗分割**

Swin 的核心洞見在於：不需要在同一個 layer 內同時計算 local 與 global attention，而是讓**連續的兩個 layer 交替使用不同的視窗劃分模式**。

具體做法：

- 第 l 層（奇數層）：使用 **regular window partitioning**——從左上角開始均勻劃分，視窗大小 M×M
- 第 l+1 層（偶數層）：使用 **shifted window partitioning**——視窗位置偏移 (⌊M/2⌋, ⌊M/2⌋) 個 pixels

當 M=7 時，偏移量為 3 個 pixels。偏移後，原本在第 l 層四個不同視窗角落的 patches，在第 l+1 層中被分到同一個視窗內。資訊就透過這些「跨視窗的 patches」在層與層之間流通。

**數學表達**

連續兩個 Swin Transformer blocks 的計算流程：

$$\hat{z}^l = \text{W-MSA}(\text{LN}(z^{l-1})) + z^{l-1}$$
$$z^l = \text{MLP}(\text{LN}(\hat{z}^l)) + \hat{z}^l$$

$$\hat{z}^{l+1} = \text{SW-MSA}(\text{LN}(z^l)) + z^l$$
$$z^{l+1} = \text{MLP}(\text{LN}(\hat{z}^{l+1})) + \hat{z}^{l+1}$$

其中 W-MSA 與 SW-MSA 都是 window-based multi-head self-attention，使用的注意力計算公式完全相同，唯一的差別是**視窗劃分的位置**。因此，這兩個層的計算複雜度完全一致。

此定理有兩個重要意涵：
- 第一，偏移量為 ⌊M/2⌋ 時，跨層資訊流動的路徑最短（每兩層傳播約 1.5M pixels）
- 第二，Swin 的單向資訊流動（從左上到右下）是固定的，不像 global attention 可以雙向任意流動。這是 shifted window 相對於 global attention 在建模能力上的一個代價——不過在足夠多的層數下，感受野最終可以覆蓋全圖。

**與 Sliding Window 的本質差異**

讀者可能會想：這不就是滑動視窗 (sliding window) 嗎？Deformable DETR 和 Stand-Alone Self-Attention 都用過類似概念。

關鍵差異在於效率：

- **Sliding window attention**：對每個 query pixel 使用滑動視窗，每個 query 對應的 key set 都不同，因此無法 batch 計算。需要複雜的實作（例如 cuDNN 的 im2col-style 操作），而且對通用硬體的記憶體存取不友好。論文中 Table 5 的延遲測試顯示，sliding window 的實際延遲比 shifted window 高出約 3–7 倍（視硬體而定）。

- **Shifted window attention**：所有 query 共用同一組 key set（同一視窗內），可以在硬體上高效 batch 計算。雖然也需要 cyclic shift 的額外開銷（見下一個知識點），但 shift 操作只是記憶體位址的重新排列，遠比 sliding window 的複雜索引計算快。

換句話說，shifted window 在「建模能力」上與 sliding window 相近（Table 6 的對比實驗顯示兩者準確率幾乎相同），但在「實際延遲」上有壓倒性優勢。

**資訊流通的深層分析**

經過 L 層交替的 regular/shifted partition 後，任兩個 patches 之間的資訊流通步數 (information path length) 是多少？

對一個大小為 h×w 的特徵圖，視窗大小為 M×M。在 shifted window 設計下，每兩層可以把資訊傳播 M+⌊M/2⌋ ≈ 1.5M 個 pixel 的距離。以 M=7 計算，每兩層傳播約 10 個 pixel。Stage 3 有 18 層（Swin-S/B），經過 9 組 regular/shifted 交替後，理論感受野可以覆蓋約 90 個 pixels——足以覆蓋 16× 解析度特徵圖的大部分範圍，接近 global attention。

---

### 知識點 4: Cyclic Shift + Masked MSA

**這個知識點要回答什麼問題？**

Shifted window 的概念很漂亮，但實作上有一個工程難題：視窗偏移後，邊緣的視窗不再完整，導致視窗大小不一致，無法用標準的 batch 矩陣運算處理。這個問題怎麼解決？

**問題描述**

假設一個 8×8 的特徵圖（h=w=8），regular window 分割為 M=4，得到 2×2 = 4 個視窗，每個大小整齊的 4×4。

偏移 (2, 2)（即 M/2）後，四角的 2×2 patches 散落各處。原本整齊的 2×2 劃分變成 3×3 = 9 個視窗——數量是原來的 2.25 倍。而且這些視窗的大小不一：左下角是 2×6、右下角是 4×2、正中間是 4×4……無法直接喂進同一個 batched attention 計算中。

最直接的解決方案是 padding：把小視窗 padding 到 M×M，並在 attention 中 mask 掉 padding 區域。但 padding 方案會讓總視窗數從 (h/M)×(w/M) 增加到 (h/M+1)×(w/M+1)，多了將近 (h+w)/M + 1 個額外視窗。當 h/M 很小時（例如 Stage 4 的 h/32 ÷ 7 ≈ 1–2），padding 方案的計算量會增加 2 倍以上。

**Swin 的高效解法：Cyclic Shift + Mask**

Swin 提出的解法既簡單又優雅，分三步驟：

**Step 1 — Cyclic Shift（循環位移）：**將特徵圖沿左上方向整體偏移 (⌊M/2⌋, ⌊M/2⌋)。這讓原本散落在四個角落不完整的視窗塊（位於 (0,0) 附近的左下、左上、右上、右下等區塊）被拼接在一起，形成一個與 regular partition 完全相同數量的視窗陣列。原本的 9 個視窗又變回 4 個。

**Step 2 — Masked MSA：**拼接後的視窗中，來自不同自然區域的 patches 被組合到同一個 window 中。如果不加處理，這些 patches 之間會錯誤地進行 attention。Swin 在 QKᵀ 的 logits 上加入一個 mask 矩陣，對不該連通的位置設為 -100（softmax 後趨近於 0），對該連通的位置設為 0（不影響）。

具體的 mask 實作方式：對於拼接後的 window，其中包含來自 A、B、C、D 四個自然區域的 patches。mask 矩陣 M ∈ ℝ^(M²×M²) 定義為：

$$M_{ij} = \begin{cases} 0, & \text{if pixel i 和 j 屬於同一自然區域} \\ -\infty, & \text{otherwise} \end{cases}$$

在注意力公式中：Attention(Q, K, V) = SoftMax(QKᵀ/√d + M)V。

**Step 3 — Reverse Cyclic Shift：**Attention 計算完成後，將特徵圖沿右下方向 shift 回來（等價於對第一步的操作取逆），恢復原始的特徵排列。這個操作與第一步的 shift 互為逆運算，不損失資訊。

```
Regular Partition (layer l):   Shifted Partition (layer l+1):
┌───────┬───────┐              ┌───────┬───────┐
│       │       │              │  AB   │   C   │
│   0   │   1   │              │       │       │
│       │       │  cyclic      ├───────┼───────┤
├───────┼───────┤  shift →     │       │       │
│       │       │              │   D   │   E   │
│   2   │   3   │              │       │       │
│       │       │              └───────┴───────┘
└───────┴───────┘
A-E: 來自不同自然區域的子視窗
masked MSA 確保只在同一自然區域內計算 attention
```

經過 cyclic shift 後，batch 計算的視窗數與 regular partition 完全相同（2×2 → 2×2），因此計算量完全一致。唯一的額外開銷是 shift 操作本身（記憶體位址重排）以及 mask 的應用（attention 計算中加入 mask），兩者都極其高效。

**效率對比（論文 Table 5）**

| 方法 | Swin-T 延遲 (ms) |
|------|----------------|
| Regular window (無 shifted) | 54.9 |
| Shifted window + naive padding | 82.5 (+50% overhead) |
| Shifted window + cyclic shift | 57.8 (+5% overhead) |
| Sliding window | 213.0 (+288% overhead) |

Cyclic shift 方法只引入約 5% 的延遲開銷，而 sliding window 的延遲是 shifted window 的 3.7 倍。

---

### 知識點 5: Relative Position Bias

**這個知識點要回答什麼問題？**

ViT 使用可學習的 1D 絕對位置編碼 (absolute position embedding) 來標記 patches 的空間位置。但在 local window attention 中，每個 window 內的 patches 數只有 M²=49 個，更重要的是 patches 之間的**相對位置**而非絕對位置。Swin 因此引入了相對位置偏置 (relative position bias)。

**數學形式**

Swin 在自注意力計算中引入一個偏置項 B ∈ ℝ^(M²×M²)，公式為：

$$\text{Attention}(Q, K, V) = \text{SoftMax}(QK^T / \sqrt{d} + B)V$$

B 的元素來自一個更小的可學習參數矩陣 $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$。

為什麼是 (2M-1)×(2M-1) 的矩陣？在一個 M×M 的視窗內，任兩個 patches 之間的相對位置偏移 (Δx, Δy) 範圍是 [-(M-1), M-1]。Δx 和 Δy 各有 2M-1 種可能值，因此總共有 (2M-1)² 種離散的相對位置組合。B 中的每個元素從 \hat{B} 中對應的 (Δx, Δy) 位置索引取值。

例如 M=3 時，Δx, Δy ∈ {-2, -1, 0, 1, 2}，總共 5×5 = 25 種組合。所以 \hat{B} 的大小為 5×5。當 B 中的某個元素對應的相對偏移為 (Δx=2, Δy=-1)，就從 \hat{B} 的 (2, -1) 索引位置取值。

**ViT 的絕對位置編碼的設計問題**

ViT 使用的 1D 可學習位置編碼是直接與 patch embedding 相加的：

$$z_0 = [x_{\text{class}}; x_p^1 E; x_p^2 E; \cdots; x_p^N E] + E_{\text{pos}}$$

其中 E_pos ∈ ℝ^((N+1)×D) 的每一行對應一個 patch 的絕對位置。這種設計有兩個微妙的問題：

1. **1D 編碼無法區分 2D 空間結構**：雖然 patches 在 2D 空間中是網格狀排列的，但 ViT 將其視為一維序列。(row=1, col=2) 和 (row=2, col=1) 的 patches 在 1D 序列中距離很遠，但在 2D 空間中其實是相鄰的。ViT 使用的 1D 編碼無法捕捉這種 2D 鄰接性——雖然論文附錄 D.4 嘗試了 2D 位置編碼但發現改善有限。

2. **跨解析度插值誤差**：當 fine-tuning 時輸入解析度改變，patch 數 N 會改變，位置編碼矩陣的大小也必須調整。ViT 使用 2D interpolation 來放大編碼矩陣。但位置編碼是加在輸入上的，interpolation 引入的誤差會傳播到整個 Transformer encoder，影響每一層的注意力計算。

**Swin 的相對位置偏置 vs ViT 的絕對位置編碼**

而 Swin 的相對位置偏置是直接作用在 attention logits 上。這讓 Swin 的注意力可以直接感知 patches 之間的空間關係，而不需要模型從 embedding 中「解碼」位置資訊。

**消融實驗結果（論文 Table 4）**

| 位置編碼方式 | ImageNet Top-1 | COCO AP_box (RetinaNet) |
|-------------|---------------|------------------------|
| 無位置編碼 | 81.3% | 46.2 |
| 絕對位置編碼 (ViT-style) | 81.3% | 46.3 |
| 相對位置偏置 | **82.1% (+0.8%)** | **47.5 (+1.3%)** |
| 相對位置偏置 + 絕對位置編碼 | 81.9% | 47.2 |

從實驗中可以看出：

- 絕對位置編碼對 Swin 完全無幫助（0% 提升）：因為在 local window 中，patches 的絕對位置對注意力權重的貢獻遠不如它們之間的相對關係
- 相對位置偏置在 ImageNet 上帶來 +0.8%，在 COCO 檢測上帶來 +1.3 AP——對密集預測的幫助更大
- 把兩者疊加反而略降：可能是冗餘資訊干擾了注意力權重的學習

**跨解析度的適應性**

當 fine-tuning 時需要改變視窗大小（例如從 224² 到 384² 輸入），解析度調整會讓特徵圖大小改變，從而影響 M。在這種情況下，$\hat{B}$ 中的 (2M-1)×(2M-1) 格點可以透過 bicubic interpolation 直接調整到新的大小，無需重新訓練。

ViT 也有類似的能力（對位置編碼做 2D interpolation），但 Swin 的相對位置偏置在 interpolation 後的精確度更高，因為位置編碼是加在輸入上的，誤差會傳播到整個 encoder；而位置偏置只影響 attention logits，影響範圍更局部、更容易適應。

---

### 知識點 6: 架構變體與跨任務實驗結果

**架構變體**

Swin Transformer 提供了四種規模的變體，與 ViT/DeiT 和 ResNet 系列的計算量對齊：

| 模型代號 | 通道數 C | 各 stage block 數 | #參數 | FLOPs (224²) | ViT 對應 |
|---------|---------|-----------------|-------|-------------|---------|
| Swin-T | 96 | {2, 2, 6, 2} | 29M | 4.5G | ≈ DeiT-S (22M, 4.6G) |
| Swin-S | 96 | {2, 2, 18, 2} | 50M | 8.7G | ≈ ResNet-101 |
| Swin-B | 128 | {2, 2, 18, 2} | 88M | 15.4G | ≈ ViT-B/DeiT-B (86M, 17.5G) |
| Swin-L | 192 | {2, 2, 18, 2} | 197M | 47.0G (384²) | ≈ ViT-L (307M) |

注意幾個設計規律：

- Stage 3（16× 解析度）的 block 數最多（6 或 18），與 ResNet 將最多層（bottleneck blocks）放在 stage 3 的設計一致——這不是巧合，而是反映了階層式視覺骨幹的共通原則：中間解析度層承載了最多的語義與空間資訊
- Stage 4（32× 解析度）只有 2 個 blocks，因為到這個階段的特徵圖解析度已很低，不需要太多層進行特徵轉換
- 各變體的 channel 數 C 隨模型大小增加（96 → 128 → 192），保持 block 配置不變。這與 ViT 系列隨參數量擴大 D 和 layer 數的思路不同——Swin 更傾向於在 channel 維度擴展，而非增加深度

**ImageNet-1K 分類結果（Regular Training）**

以下資料來自論文 Table 1(a)，包含了 Transformer-based 與 ConvNet-based 基線：

| 方法 | 輸入大小 | #參數 | FLOPs | Throughput | Top-1 Acc |
|------|---------|-------|-------|-----------|----------|
| **Transformer-based 基線** | | | | | |
| ViT-B/16 | 384² | 86M | 55.4G | 85.9 img/s | 77.9% |
| ViT-L/16 | 384² | 307M | 190.7G | 27.3 img/s | 76.5% |
| DeiT-S | 224² | 22M | 4.6G | 940.4 img/s | 79.8% |
| DeiT-B | 224² | 86M | 17.5G | 292.3 img/s | 81.8% |
| DeiT-B | 384² | 86M | 55.4G | 85.9 img/s | 83.1% |
| **Swin 系列 (ours)** | | | | | |
| Swin-T | 224² | 29M | 4.5G | 755.2 img/s | **81.3%** |
| Swin-S | 224² | 50M | 8.7G | 436.9 img/s | **83.0%** |
| Swin-B | 224² | 88M | 15.4G | 278.1 img/s | **83.5%** |
| Swin-B | 384² | 88M | 47.0G | 84.7 img/s | **84.5%** |

關鍵觀察：

- Swin-T (81.3%) vs DeiT-S (79.8%)：在相近的參數量和 FLOPs 下，Swin 比 DeiT 高出 1.5%。注意 Swin-T 的 throughput (755) 低於 DeiT-S (940) 約 20%，原因來自 cyclic shift 與 masking 的額外開銷——這是計算效率與建模能力之間的精確權衡
- ViT-B/16 在 ImageNet-1K 訓練下只有 77.9%（384² 輸入），遠低於 Swin-B 的 83.5%/84.5%。這個高達 5.6% 的差距說明 ViT 在中等資料集上訓練不充分——ImageNet 的 1.28M 張圖像對 ViT 來說「太小」。值得注意的是，DeiT 在 ViT 的基礎上透過知識蒸餾、更好的 augmentation 和更長的訓練（300 epochs）將這個數字提升到 81.8%（224²）和 83.1%（384²），但就算與經過優化的 DeiT-B 相比，Swin-B 仍有 1.4–1.7% 的優勢
- Swin-S 的 83.0% 已超越 EfficientNet-B6 (84.0%) 的較小版本，且大幅超過 R-101 系列

**ImageNet-22K 預訓練後的超參數結果**

當預訓練資料從 ImageNet-1K (1.28M) 擴大到 ImageNet-22K (14.2M) 時：

| 方法 | 輸入大小 | #參數 | Throughput | Top-1 Acc |
|------|---------|-------|-----------|----------|
| ViT-B/16 | 384² | 86M | 85.9 img/s | 84.0% |
| ViT-L/16 | 384² | 307M | 27.3 img/s | 85.2% |
| BiT (R-152x4) | 480² | 937M | — | 85.4% |
| **Swin-B** | **384²** | **88M** | **84.7 img/s** | **86.4%** |
| **Swin-L** | **384²** | **197M** | **42.1 img/s** | **87.3%** |

Swin-B (88M, 84.7 img/s) 比 ViT-L (307M, 27.3 img/s) 高出 1.2%——用不到三分之一的參數量和三倍以上的吞吐量，達到更好的結果。而 Swin-L 的 87.3% 是當時 ImageNet-1K 上純監督學習的頂尖結果。

**COCO 物體檢測與實例分割**

論文在四個不同框架上測試了 Swin 作為 backbone 的效果，選擇性地來看 Cascade Mask R-CNN 框架的結果：

| Backbone | AP_box | AP_50 | AP_75 | AP_mask | #參數 | FLOPs |
|---------|-------|-------|-------|---------|-------|-------|
| ResNet-50 | 46.3 | 64.3 | 50.5 | 40.3 | 82M | 739G |
| **Swin-T** | **50.5** | **69.3** | **54.9** | **43.7** | 86M | 745G |
| ResNeXt-101-64x4d | 48.3 | 67.0 | 52.5 | 41.8 | — | — |
| **Swin-S** | **51.8** | **70.4** | **56.5** | **44.7** | — | — |

Swin-T 在 Cascade Mask R-CNN 上，AP_box 比 ResNet-50 高出 4.2 個點——這在 COCO 檢測任務上是極大的躍進（通常 0.5 AP 的改善就算顯著）。後續使用 HTC++ 框架與 Swin-L (ImageNet-22K pretrain)，Swin 達到 58.7 box AP 和 51.1 mask AP，超越當時所有公開方法，包括 DetectoRS (56.0/48.2) 與 Copy-paste (56.0/—)。

**ADE20K 語意分割**

| Backbone | mIoU | 框架 |
|---------|------|------|
| SETR (ViT-based) | 50.3 | UPerNet |
| ResNet-101 | 44.9 | UPerNet |
| **Swin-S** | **53.5 (+3.2)** | UPerNet |
| **Swin-L (22K pretrain)** | **55.0** | UPerNet |

Swin-S 在 ADE20K 上超越了 ViT-based 的 SETR，差距為 3.2 mIoU。SETR 的工作方式就是前面提到的那種「從 ViT 的單一 H/16 × W/16 特徵圖直接 upsample 做分割」的方案——Swin 的階層式架構對密集預測的優勢在這裡一覽無遺。

**關鍵消融實驗一覽**

論文做了系統性的消融實驗，以下是最關鍵的發現：

**1. Shifted window 的貢獻（Table 4）**

| 設定 | ImageNet Top-1 | COCO AP_box (RetinaNet) | ADE20K mIoU |
|------|---------------|------------------------|-------------|
| Regular window only (baseline) | 81.3% | 46.2 | 46.6 |
| + Shifted window | 82.1% (+0.8%) | 49.6 (+3.4 AP) | 49.3 (+2.7) |

Shifted window 對分類的幫助相對有限（+0.8%），但對密集預測任務的提升巨大（+3.4 AP、+2.7 mIoU），這與前述的理論預期一致：分類只需要 global representation，shifted window 帶來的跨視窗資訊流動對分類的邊際效益不大；而檢測和分割需要在不同尺度和位置上精確定位物體，跨視窗資訊至關重要。

**2. 視窗大小 M 的影響**

| M | FLOPs (Stage 3 為例) | ImageNet Top-1 |
|---|---------------------|---------------|
| M=7 (default) | 8.7G | 83.0% (Swin-S) |
| M=12 | ≈ 12.6G (+45%) | ≈ 83.2% (+0.2%) |

更大的視窗（M=12 → M=7）僅帶來 0.2% 的改善，但計算量增加 45%。這個邊際效益遞減的現象表明：在 7×7 的 local window 之外，patches 之間的相關性已經很弱——這與自然界圖像中相鄰 pixel 的相關性遠大於遠距離 pixel 的觀察是一致的。M=7 因此成為一個高效的折中選擇。

**3. 不同框架下 Swin-T vs R-50**

論文在四個檢測框架下比較了 Swin-T 與 ResNet-50 作為 backbone 的效果：

| 框架 | R-50 AP_box | Swin-T AP_box | 差距 |
|------|------------|--------------|------|
| Cascade Mask R-CNN | 46.3 | 50.5 | +4.2 |
| ATSS | 43.5 | 47.2 | +3.7 |
| RepPoints v2 | 47.5 | 50.0 | +2.5 |
| Sparse R-CNN | 44.5 | 47.9 | +3.4 |

在所有框架下，Swin 的 AP_box 都優於 ResNet-50，差距在 +2.5 到 +4.2 AP 之間，且差距具有統計顯著性（不同框架的差異來自框架本身的設計偏好，但 Swin 的相對優勢是一致的）。

---

## 實驗結果

### 主要實驗總結

| 任務 | 資料集 | 評量指標 | Swin 最佳結果 | 對比基線 | 提升幅度 |
|------|--------|---------|-------------|---------|---------|
| 分類 | ImageNet-1K | Top-1 Acc | **87.3%** (Swin-L, 22K FT) | ViT-L: 85.2% | +2.1% |
| 檢測 | COCO test-dev | AP_box | **58.7** (HTC++, Swin-L) | DetectoRS: 56.0 | +2.7 |
| 分割 | ADE20K val | mIoU | **53.5** (Swin-S) | SETR: 50.3 | +3.2 |

Swin-S（50M 參數）在 COCO 上超越 ResNeXt-101-64x4d（約 80M 參數），Swin-L（197M）僅用 ViT-L（307M）64% 的參數量就達到了顯著更好的結果。這說明了層次式設計比單純放大 ViT 的 encoder 更有效——Swin 增加的參數主要用於 patch merging 和多階段處理，這些結構性的改進比在單一解析度 encoder 中加入更多層的效率更高。

### 失敗案例與限制

Swin Transformer 雖然在三大任務上取得極佳成績，但仍有一些值得注意的限制：

1. **分類優勢不如密集預測顯著**：Swin 在 ImageNet 上的優勢 (Swin-T +1.5% vs DeiT-S) 遠不如在 COCO (+4.2 AP vs R-50) 和 ADE20K (+3.2 mIoU vs SETR) 上顯著。這是可以預期的——分類只需要單一解析度的 global representation，而 Swin 的核心優勢（hierarchical features + shifted window）主要展現在需要多尺度資訊的密集預測任務上。

2. **Throughput 略有犧牲**：Swin-T (755 img/s) 比 DeiT-S (940 img/s) 慢約 20%。這來自於兩方面：cyclic shift 與 masking 的運算開銷，以及 patch merging 在 stage 間引入的額外操作。對於需要高吞吐量的應用場景（如即時檢測），這個差距可能需要考慮。

3. **後續工作發現的延伸限制**：在 Swin v2 (2021-11) 論文中，作者發現當模型擴展到 3B 參數或解析度提升到 1536×1536 時，Swin v1 的 pre-norm 設計會導致訓練不穩定。這需要 post-normalization、scaled cosine attention 和 log-spaced continuous position bias 等改進來解決——這說明 Swin 的架構在極大規模下仍有改善空間。

5. **All-MLP 架構的適用性**：論文中提到 shifted window 方案也證明了對 all-MLP 架構有益（參見論文 Section 1 與 Table 5），但 Swin 原論文對這部分的探討非常有限。後續的 MLP-Mixer、gMLP、AS-MLP 等工作更全面地研究了這個方向。

### 關鍵理解：Swin 並非「取代」ViT，而是「補足」ViT

將 Swin 與 ViT 的關係理解為「取代」是誤解。兩者解決的是不同層次的問題：

- ViT 的核心貢獻是證明了 Transformer 可以不加修改地用於圖像分類，前提是資料量夠大。ViT 更適合**只需要單一層級表示**的任務（分類、檢索、嵌入）。
- Swin 的核心貢獻是讓 Transformer 也能高效率地做**密集預測**，透過階層式特徵圖與 local window attention 來實現。Swin 更適合需要多尺度特徵的任務（檢測、分割、影片理解）。

兩者各有適用場景。例如，如果只需要 ImageNet 分類且追求最低延遲，DeiT-S 的 940 img/s throughput 可能比 Swin-T 的 755 img/s 更合適。但如果需要做 COCO 檢測或 ADE20K 分割，Swin 的階層式特徵圖是更適合的選擇。

4. **視窗大小的固定性**：M=7 在 ImageNet 224² 上表現最佳，但對於特定的密集預測任務（如長條型物體的檢測）或極高解析度的輸入，可能需要動態調整視窗大小。Swin 採用 bicubic interpolation 來處理視窗大小變化，但這始終是事後適應而非設計時就考慮的可變性。

5. **與 ConvNet 的直接比較**：雖然 Swin 在多數任務上超越 ConvNet，但在低延遲場景下，輕量級 ConvNet（如 MobileNet、EfficientNet-B0）在參數量和推理速度上仍有優勢。Swin 的 shifted window 帶來額外的 memory access overhead，在部署到邊緣設備時可能需要考慮。

---

## 與相關工作的對比

以下從幾個關鍵維度比較 Swin Transformer 與 ViT/DeiT 在設計上的本質差異：

| 維度 | ViT / DeiT | Swin Transformer |
|------|-----------|-----------------|
| 訓練資料需求 | 依賴大型資料集 (JFT-300M) 才能超越 CNN | ImageNet-1K 即可超越 ViT/DeiT（歸納偏置更多） |
| 高解析度可擴展性 | 二次方計算量，高解析度困難 | 線性計算量，可擴展到高解析度 |
| 多尺度特徵圖 | 無（單一 H/P × W/P 解析度） | 4 層階層式（H/4 ~ H/32） |
| 核心歸納偏置 | 少（僅 patch embedding 暗示局部性） | 多（local windows、hierarchical、相對位置偏置） |
| 位置編碼方式 | 可學習 1D 絕對位置編碼 | 可學習 2D 相對位置偏置 |
| 感受野（同層數下） | 全圖（global attention） | ~1.5M per 2 layers, 累積可達全圖 |
| 可替換 CNN backbone | 不可直接替代 | 可直接替換 (feature map topology matches ResNet) |
| 延遲 vs 效能權衡 | 較低延遲/較低效能 (DeiT-S) | 較高效能/稍高延遲 (Swin-T) |

這些差異凸顯了 Swin Transformer 的核心貢獻：**不是對 Transformer 做根本性的重新設計，而是根據視覺問題的本質特性，在 Transformer 的基礎上注入恰當的結構性歸納偏置**。

---

## 我的觀察

### 歸納偏置的回歸

ViT 的論文給人一個強烈的訊息：「只要有足夠的資料，Transformer 不需要任何視覺專用的歸納偏置。」ViT 用 JFT-300M 來證明這件事。但到了密集預測任務——那些需要理解物體在哪裡、邊界在哪裡的任務——即使有 JFT-300M 級的資料量並搭配 direct upsampling，ViT 仍然顯著落後於 Swin。這說明**結構性的歸納偏置在大規模資料下仍然至關重要**。

Swin 選擇了一條不同的路：不是跟資料量硬拚，而是在 Transformer 架構中注入經過審慎設計的歸納偏置——階層式結構與局域性計算。這不是回到 CNN，而是從視覺問題的本質出發去修改 Transformer。我認為這是 Swin 最重要的貢獻：它示範了如何在不偏離 Transformer 核心設計太多的前提下，為視覺任務定製一個高效的架構。

### Shifted Window 的設計哲學

Shifted window 最巧妙的地方在於：它在**不增加任何計算量的前提下實現了資訊流動**。Regular partition 與 shifted partition 的計算量完全相同，差別只在視窗的排列方式。這是一個「運算組織方式」的改進而非「模型容量」的改進——不是加入更多參數，而是重新組織已有的計算。

這個設計哲學在後續工作中得到了傳承：Swin v2 的全部改進（scaled cosine attention、log-spaced continuous position bias、post-norm）都是「組織方式」的提升而非模型容量的擴張。未來如果有更高效的跨視窗資訊流動方式，應該也能類似地嵌入 Swin 框架而不需要大幅修改核心設計。

### 對後續領域的影響

Swin Transformer 發表後迅速成為視覺領域最廣泛使用的骨幹之一，影響力體現在：

- **多尺度視覺 Transformer 的標準範式**：後續的 CSWin、Focal Transformer、MaxViT 等都是在 Swin 的 shifted window 或多尺度設計上的變體與改進
- **自監督學習中的應用**：SimMIM 以 Swin 為 backbone，在 masked image modeling 任務上達到了 SOTA，證明了階層式架構對自監督任務的適應性
- **跨模態擴展**：Video Swin Transformer 將 shifted window 擴展到 3D 時空域，用於影片理解；Swin UNETR 應用於醫學影像分割——從物體檢測到影片分析的廣泛應用證明了 shifted window 作為一個通用設計模式的有效性

**Swin Transformer 的引用影響力**

截至 2026 年初，Swin Transformer 已被引用超過 15,000 次，是 2021 年發表的所有視覺論文中引用次數最高的之一。這個被引用量說明了它的影響力不限於特定的論文家族——CVPR 2021/2022 中許多最佳論文或提名均使用 Swin 作為 backbone（例如 Mask2Former 在 COCO 全景分割任務中即以 Swin-L 為骨幹達到當時 SOTA）。

Swin 的成功也催生了「視覺 Transformer backbone」領域的規範化研究。在 Swin 之前大家都在探索「Transformer 能不能做視覺」，在 Swin 之後大家都在問「哪個 backbone 設計更好」——這個研究問題的轉變本身就是 Swin 影響力的體現。

**Swin 對後續架構設計的影響**

Swin 提出的「local window + shifted window」雙層模式，成為後續許多視覺 Transformer 架構的設計藍圖。一些重要的後續工作包括：

- **CSWin Transformer (2021-07)**：將 shifted window 從正方形視窗改為十字形 (cross-shaped) 視窗——水平方向一個自注意力頭、垂直方向另一個頭。這種分解方式可以捕捉更細粒度的方向資訊，特別是在自然圖像中物體常呈現水平或垂直排列的語境下。

- **Focal Transformer (2021-07)**：結合「近距離細粒度 + 遠距離粗粒度」的多尺度注意力，在每個 token 周圍用 fine-grained attention，遠處用 coarse-grained attention。這是對 Swin「只有 fine-grained local」的一種補充，讓模型在不增加太多計算的前提下接觸更廣的上下文。

- **MaxViT (2022-04)**：在一個 block 內同時做 local 和 global attention——先分區做 window attention（Swin-style local），再用 grid attention（均勻下取樣後的 global attention）。這讓同一個 block 內同時具備局部細緻與全域抽象兩種能力，跳脫了 Swin 需要跨層交替輪換的設計。

這些工作從不同角度改進或補充了 Swin 的設計，但它們的基本出發點——local window attention 作為效率的基礎——都源自 Swin Transformer。

**關鍵總結：Swin Transformer 的設計原則**

回顧 Swin Transformer 的所有設計決策，可以歸納出三條基本原則：

1. **效率優先**：所有設計從計算複雜度和記憶體存取效率出發。W-MSA 確保線性複雜度，cyclic shift 確保實作效率，M=7 從準確率與計算量的 Pareto frontier 選取。

2. **與既有系統相容**：Swin 刻意將輸出解析度設計成與 ResNet 完全相同，讓它可以直接插入任何現有的檢測、分割框架。這個「向後相容」的設計思維，是 Swin 被快速廣泛採用的關鍵原因之一。

3. **繼承 Transformer 的核心但不盲從**：Swin 保留了 Transformer 的基本結構（MSA + MLP + LN + residual connections），但在關鍵處做了修改——從 global 改為 local self-attention、從絕對位置編碼改為相對位置偏置、從單尺度改為多尺度。這些修改都有明確的動機支撐，而不是隨意改動。

### 實務部署考量

對於想要實際使用 Swin Transformer 的開發者，以下幾點值得注意：

- **預訓練模型的選擇**：ImageNet-22K 預訓練對檢測和分割任務的幫助極大（Swin-B 在 COCO 上 +2–3 AP）。如果計算資源允許，建議使用 ImageNet-22K 預訓練版本。
- **與檢測框架的整合**：Swin 支援主流的 mmdetection 和 detectron2 框架，直接以 backbone 的形式匯入。現有基於 ResNet 的程式碼只需將 backbone 替換為 Swin，無需修改模型頭或損失函數。
- **記憶體需求**：Swin-L 在訓練時需要較大的 GPU 記憶體（約 16GB+ for detection fine-tuning）。Swin-T 和 Swin-S 的記憶體需求與 ResNet-50/101 相當，是良好的起點。

### 訓練設定與實作細節

論文中使用的訓練設定也值得注意，因為它反映了 Swin 與 ViT/DeiT 在訓練穩定性上的差異：

- **優化器**：AdamW（β1=0.9, β2=0.999），權重衰減 0.05（Swin-T/S/B）或 0.01（Swin-L 22K pretrain）
- **學習率排程**：Cosine decay（ImageNet-1K 訓練使用 300 epochs，20 epochs linear warmup；ImageNet-22K 預訓練使用 90 epochs，5 epochs warmup）
- **批次大小**：1024（ImageNet-1K），4096（ImageNet-22K）
- **資料增強**：沿用 DeiT 的 augmentation 策略（RandAugment、MixUp、CutMix、Random Erasing 等），但不使用 repeated augmentation——論文指出這對 Swin 不僅無幫助，反而降低效能，這與 DeiT 的觀察相反
- **正則化**：Stochastic Depth (drop path) 被證明對 Swin 有效；Dropout 未使用

有趣的是，Swin 在較小的學習率（DeiT 的 0.001 vs Swin 的 0.001 一致）和標準 augmentation 下就能穩定訓練，不需要 ViT 那種複雜的訓練技巧。這再次說明了**結構性歸納偏置（local windows + hierarchical design）降低了對訓練技巧和資料量的依賴**。

**Stochastic Depth 的影響**

Stochastic Depth (drop path) 是一種對深層網路特別重要的正則化技術——它在訓練時隨機丟棄整個 block 的輸出（等於跳過該 block）。對於 Swin 這種有 18 層的 Stage 3（Swin-S/B/L），隨機丟棄部分 block 可以防止過擬合，並允許訓練更深的模型。

論文將 drop path rate 設定為線性增加：淺層（Stage 1–2）的 drop rate 較低，深層（Stage 3–4）的 drop rate 較高。Swin-S 的 max drop rate 為 0.2，Swin-B 為 0.3，Swin-L 為 0.5。這種設計與 ResNet 的 Stochastic Depth 一致：深層的特徵更抽象、更容易過擬合，因此需要更強的正則化。

---

## 延伸閱讀

### Dependency Papers (本文涵蓋)

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** ([2010.11929](https://arxiv.org/abs/2010.11929)) — Dosovitskiy et al., 2020
   - 與本文關係：ViT 是 Swin Transformer 的直接前身。Swin 解決了 ViT 的單尺度特徵圖與二次方複雜度兩大限制，在 ViT 的基礎上加入了階層式結構與 shifted window attention。

### 後續發展 (未涵蓋，僅列出)

- **Swin Transformer V2: Scaling Up Capacity and Resolution** ([2111.09883](https://arxiv.org/abs/2111.09883)) (2021-11) — 解決了在 3B 參數與 1536×1536 解析度下的訓練穩定性問題
- **Video Swin Transformer** ([2106.13230](https://arxiv.org/abs/2106.13230)) (2021-06) — 將 shifted window 擴展到 3D 時空域
- **SimMIM: A Simple Framework for Masked Image Modeling** ([2111.09886](https://arxiv.org/abs/2111.09886)) (2021-11) — 以 Swin 為 backbone 的自監督預訓練方法
- **MaxViT: Multi-Axis Vision Transformer** ([2204.01697](https://arxiv.org/abs/2204.01697)) (2022-04) — 結合局部 (Swin-style) 與全域 (ViT-style) 的多軸注意力

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。

---

*撰寫於 2026-05-21。Hermes Agent 自動生成，基於論文原文閱讀與知識點歸納。*
