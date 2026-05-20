# Swin Transformer: Hierarchical Vision Transformer 解讀

> **種子論文**: [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (2021-03)
> **作者**: Ze Liu, Yutong Lin, Yue Cao, et al.
> **機構**: Microsoft Research Asia

---

## TL;DR

Swin Transformer 解決的是視覺 Transformer 缺乏層級式特徵圖與計算複雜度過高的問題。它將 self-attention 限制在非重疊的局部視窗中計算，並透過 shifted window 機制在層與層之間建立跨視窗連接，在維持線性計算複雜度的同時建構多尺度層級式特徵。它在 ImageNet 分類、COCO 物件偵測、ADE20K 語義分割上全面超越當時的 SOTA，是第一個可作為通用骨幹的視覺 Transformer。

---

## 背景與動機

2020 年 ViT（Vision Transformer）的出現證明了一件事：純 Transformer 架構不需要 CNN 的歸納偏置，只要在大規模資料上預訓練，就能在圖像分類上達到甚至超越 CNN 的表現。但 ViT 在設計上有兩個根本限制，讓它很難成為通用的視覺骨幹網路：

- **單一解析度的特徵圖**——ViT 在所有層都維持相同的 patch 數（$H/16 \times W/16$），只產生一個尺度的特徵圖。這對於需要多尺度特徵的密集預測任務（如物件偵測的 FPN、語義分割的 U-Net）來說是一個致命缺陷。

- **二次計算複雜度**——ViT 採用全局自注意力，計算量與 patch 數的平方成正比（$O(N^2)$）。當輸入解析度提高時，patch 數急劇增加，計算成本很快就變得難以承受。

這兩個問題的根源是同一個設計選擇：ViT 直接把 NLP Transformer 搬過來，將整張圖視為一個 token 序列做全局自注意力。但視覺與語言有本質差異——視覺實體的尺度變化極大，且圖像的像素數量遠多於文字序列的 token 數。

Swin Transformer 的目標很明確：**在不犧牲 Transformer 表達能力的前提下，設計一個可作為通用骨幹的視覺 Transformer，同時具有層級式特徵圖與線性計算複雜度。**

---

## 核心知識點

本文圍繞以下知識點展開。這是理解 Swin Transformer 的關鍵概念，後續章節會依序展開：

1. **從 NLP 到 CV 的 Transformer 轉移**——如何將 Transformer（原設計給 1D 序列）應用到 2D 圖像，ViT 與 Swin 分別做了哪些取捨
2. **固定解析度 vs 層級式特徵圖**——ViT 的單尺度 vs Swin 的多尺度層級結構，為什麼後者對密集預測任務至關重要
3. **二次 vs 線性計算複雜度**——從全局自注意力的 $O(N^2)$ 到窗口自注意力的 $O(N \cdot M^2)$
4. **Shifted Window 分區機制**——Swin 的核心創新：如何在局部視窗之間建立連接
5. **相對位置偏置**——為什麼在視窗內使用學習到的相對位置編碼比 ViT 的絕對位置編碼更有效
6. **全任務泛化能力**——Swin 在分類、偵測、分割上的表現與 ViT/CNN 的全面比較

---

## 方法詳解

### 知識點 1: 從 NLP 到 CV 的 Transformer 轉移

**這個知識點要回答什麼問題？**

Transformer 是為 1D 序列設計的，圖像是 2D 的。要把 Transformer 應用到圖像，首先需要解決「如何把 2D 圖像變成 1D token 序列」這個問題。

**ViT 的做法（Dosovitskiy et al., 2021）**

ViT 的做法最直接：將一張 $H \times W \times C$ 的圖像切成 $N = HW / P^2$ 個大小為 $P \times P$ 的 patch，每個 patch 攤平成一個 $P^2 \cdot C$ 維的向量，再透過線性投影映射到 Transformer 的隱藏維度 $D$。然後在序列前面加上一個 learnable 的 `[CLS]` token，其最終的隱藏狀態作為圖像的整體表示。位置資訊透過可學習的 1D position embedding 加入，與 patch embedding 相加後送入標準的 Transformer encoder。

$$
z_0 = [\mathbf{x}_{\text{class}}; \mathbf{x}_p^1 \mathbf{E}; \mathbf{x}_p^2 \mathbf{E}; \dots ; \mathbf{x}_p^N \mathbf{E}] + \mathbf{E}_{\text{pos}}, \quad \mathbf{E} \in \mathbb{R}^{(P^2 \cdot C) \times D}
$$

這個設計的關鍵特徵是「盡可能不改動原始的 Transformer 架構」——除了 patch extraction，ViT 幾乎沒有引入任何視覺領域的歸納偏置。

**Swin 的做法（Liu et al., 2021）**

Swin Transformer 採用了不同的起點。它同樣從 patch 開始——patch size 為 $4 \times 4$，遠小於 ViT 常見的 $16 \times 16$——但之後的處理就完全不同了：

1. **分階段處理**：Swin 將網路分為 4 個 stage，每個 stage 內部有若干 Swin Transformer block。這與 CNN 的設計哲學一致（如 ResNet 的 4 個 stage）。
2. **逐階段降採樣**：透過 patch merging layer 在 stage 之間減少 token 數、增加 channel 數，建構出 CNN 風格的層級式特徵圖。
3. **局部注意力**：不計算全局注意力，而是將注意力限制在非重疊的局部視窗內。

這三點加起來，使得 Swin 的設計從一開始就是「視覺優先」的——保留了 Transformer 的核心注意力機制，但在架構層面引入了視覺任務需要的層級結構與局部性。

---

### 知識點 2: 固定解析度 vs 層級式特徵圖

**這個知識點要回答什麼問題？**

為什麼 NLP 的單尺度設計在視覺上不夠用？物體偵測和語義分割這類密集預測任務為什麼需要多尺度特徵？

**ViT 的單尺度設計**

ViT 所有層都維持相同的 token 數量（$H/16 \times W/16$，即單一解析度）。這在分類任務上不是問題——只需要最後的 `[CLS]` token 做決策——但對於需要像素級預測的任務來說，這意味著：

- 無法直接接入 FPN（特徵金字塔網路），因為 FPN 需要多個解析度的特徵圖
- 無法接入 U-Net 風格的編碼器-解碼器結構
- 單一解析度的特徵圖在處理不同尺度的物體時缺乏彈性

後續的一些工作試圖透過 deconvolution 或 upsampling 將 ViT 的特徵圖放大來解決這個問題，但這只是修補而非根本解決方案。

**Swin 的層級式設計**

Swin Transformer 的 4 個 stage 分別產生以下解析度的特徵圖：

| Stage | 解析度（相對於輸入） | channel 數 |
|-------|---------------------|-----------|
| Stage 1 | $\frac{H}{4} \times \frac{W}{4}$ | $C$ |
| Stage 2 | $\frac{H}{8} \times \frac{W}{8}$ | $2C$ |
| Stage 3 | $\frac{H}{16} \times \frac{W}{16}$ | $4C$ |
| Stage 4 | $\frac{H}{32} \times \frac{W}{32}$ | $8C$ |

這恰好與 ResNet 的 4 個 stage 的輸出解析度一致（C1–C4）。這不是巧合——這使得 Swin Transformer 可以直接取代任何 CNN backbone，無需修改 FPN、U-Net 或任何依賴多尺度特徵的上層架構。

**Patch Merging 的實現**

Patch merging 是連接 stage 之間的關鍵操作。以 stage 1 到 stage 2 為例：每 $2 \times 2$ 個相鄰 patch 的特徵被拼接起來（產生 $4C$ 維的向量），再透過一個線性層投影到 $2C$ 維。這使得 token 數量降為原本的 $1/4$（2× 降採樣解析度），而 channel 數加倍。

---

### 知識點 3: 二次 vs 線性計算複雜度

**這個知識點要回答什麼問題？**

為什麼局部注意力比全局注意力更適合視覺任務？兩者的計算複雜度差距有多大？

假設輸入圖像被切成 $h \times w$ 個 patch，每個 patch 的隱藏維度為 $C$。全局 MSA（Multi-head Self-Attention）的 FLOPs 為：

$$
\Omega(\text{MSA}) = 4hw C^2 + 2(hw)^2 C
$$

其中 $4hwC^2$ 來自 Q/K/V 投影和輸出投影，$2(hw)^2 C$ 來自注意力矩陣的計算。後者對 patch 數 $hw$ 呈二次成長——當圖像解析度從 $224^2$ 提升到 $448^2$（$hw$ 增加 4 倍），這部分的計算量將增加 16 倍。

Swin Transformer 的 Window-based MSA（W-MSA）將注意力計算限制在每個大小為 $M \times M$ 的非重疊視窗內：

$$
\Omega(\text{W-MSA}) = 4hw C^2 + 2 M^2 hw C
$$

當 $M$ 固定（預設為 7）時，第二項對 $hw$ 呈**線性**成長。這使得 Swin 可以處理更高解析度的輸入而不會爆炸性增加計算成本。

論文中的實驗也證實了這一點：當輸入解析度從 $224^2$ 提升到 $384^2$ 時，Swin-B 的 FLOPs 從 15.4G 增加到 47.0G（約 3 倍），而 ViT-B 從 17.5G 增加到 55.4G（約 3.2 倍），差距隨著解析度進一步提高會更加明顯。

**相關論文（ViT）怎麼處理？**

ViT 的做法是「不解——直接做全局注意力」。它在 $16 \times 16$ 的 patch size 下，對於 $224^2$ 的輸入，序列長度為 196，全局注意力的成本還在可接受範圍內。但這也意味著 ViT 無法有效處理更高解析度的輸入——這也是為什麼 ViT 論文中的 dense prediction 實驗需要依賴 deconvolution 等後處理手段。

---

### 知識點 4: Shifted Window 分區機制

**這個知識點要回答什麼問題？**

窗口式自注意力雖然高效，但有一個明顯的問題：不同視窗之間沒有資訊交流。如果只在各自的小視窗內做注意力，模型永遠看不到跨視窗的關係。怎麼解決這個問題？

**Swin 的核心創新**

Swin 的解決方案非常優雅：**在連續的 Transformer block 之間交替使用兩種不同的 window 分區方式**。

在第 $l$ 層使用 regular window partition（從左上角開始均勻分割），在第 $l+1$ 層將 window 往右下角偏移 $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$ 個像素後再分割。

具體公式如下：

$$
\begin{aligned}
\hat{\mathbf{z}}^l &= \text{W-MSA}(\text{LN}(\mathbf{z}^{l-1})) + \mathbf{z}^{l-1}, \\
\mathbf{z}^l &= \text{MLP}(\text{LN}(\hat{\mathbf{z}}^l)) + \hat{\mathbf{z}}^l, \\
\hat{\mathbf{z}}^{l+1} &= \text{SW-MSA}(\text{LN}(\mathbf{z}^{l})) + \mathbf{z}^{l}, \\
\mathbf{z}^{l+1} &= \text{MLP}(\text{LN}(\hat{\mathbf{z}}^{l+1})) + \hat{\mathbf{z}}^{l+1}.
\end{aligned}
$$

其中 W-MSA 使用 regular partition，SW-MSA 使用 shifted partition。

**高效批次計算**

Shifted partition 會產生一個問題：邊界處的視窗會小於 $M \times M$，導致視窗數量增加（從 $\frac{h}{M} \times \frac{w}{M}$ 變成 $(\frac{h}{M}+1) \times (\frac{w}{M}+1)$），而且不同視窗大小不一，不利於批次計算。

Swin 的解法是 **cyclic-shifting**：將特徵圖往左上角循環位移，使偏移後的視窗重新對齊成規則的網格。位移後，原本分散在不同位置的子視窗會拼成一個完整的 $M \times M$ 視窗，只需在注意力計算時用 mask 遮掉不屬於同一子視窗的位置即可。

這個技巧使得 shifted configuration 的視窗數與 regular configuration 完全相同（$\frac{h}{M} \times \frac{w}{M}$），維持了批次計算的效率。

**與滑動視窗的比較**

一個直覺的替代方案是滑動視窗式自注意力，即讓每個 pixel 以自己為中心在其鄰域內計算注意力。Swin 論文中比較了這兩種方法（Table 5），發現 shifted window 方法在延遲上顯著低於滑動視窗。原因是滑動視窗中不同 query 對應不同的 key set，難以充分利用硬體的記憶體存取優化，而 shifted window 中同一個視窗內的所有 query 共享相同的 key set。

---

### 知識點 5: 相對位置偏置

**這個知識點要回答什麼問題？**

Transformer 本身是排列不變的（permutation-invariant），所以需要某種形式的 position encoding 來告訴模型 token 的位置資訊。應該用哪種方式？

**ViT 的做法**

ViT 使用標準的可學習 1D position embedding。每個位置 $i$ 都有一個可學習的向量 $\mathbf{p}_i \in \mathbb{R}^D$，與 patch embedding 相加後作為 Transformer 的輸入。雖然論文也嘗試了 2D position embedding，但沒有觀察到顯著改善。

**Swin 的做法**

Swin 在每個視窗內使用**相對位置偏置**（relative position bias）。在計算注意力分數時，不是單純的 $QK^T / \sqrt{d}$，而是加上一個偏置項 $B$：

$$
\text{Attention}(Q, K, V) = \text{SoftMax}(QK^T / \sqrt{d} + B)V
$$

其中 $B \in \mathbb{R}^{M^2 \times M^2}$ 是 $M^2$ 個 token 之間的相對位置偏置。為了減少參數量，Swin 使用一個更小的偏置矩陣 $\hat{B} \in \mathbb{R}^{(2M-1) \times (2M-1)}$——因為沿每個軸的相對位置範圍是 $[-M+1, M-1]$，共 $2M-1$ 種可能的偏移量——然後從 $\hat{B}$ 中取值填入 $B$。

消融實驗（Table 4）顯示：

- 不使用任何位置偏置：baseline
- 加上絕對位置編碼（ViT 的做法）：與 baseline 持平
- 使用相對位置偏置（Swin 的做法）：**顯著提升**
- 同時使用兩者：輕微下降

這說明在局部視窗的場景下，**相對位置關係比絕對位置更重要**——視窗內每個 token 只需要知道「它相對於 query 在水平/垂直方向偏了多少」，不需要知道它在整張圖中的絕對座標。

---

### 知識點 6: 全任務泛化能力

**這個知識點要回答什麼問題？**

Swin Transformer 是否真的能像 CNN 骨幹一樣，在分類、偵測、分割這三類核心視覺任務上都有競爭力？

**ImageNet-1K 分類結果**

| 方法 | 參數量 | FLOPs | top-1 acc |
|------|--------|-------|-----------|
| DeiT-S | 22M | 4.6G | 79.8% |
| Swin-T | 29M | 4.5G | **81.3%** |
| DeiT-B (224²) | 86M | 17.5G | 81.8% |
| Swin-B (224²) | 88M | 15.4G | **83.5%** |
| DeiT-B (384²) | 86M | 55.4G | 83.1% |
| Swin-B (384²) | 88M | 47.0G | **84.5%** |

在分類任務上，Swin 以相似的計算量持續超越 DeiT（ViT 的資料高效版本）。使用 ImageNet-22K 預訓練後，Swin-B 達到 86.4%，Swin-L 達到 **87.3%**，後者在同等推理吞吐量下比 ViT-L 高出 2.4%。

**COCO 物件偵測結果**

使用 Cascade Mask R-CNN 框架：

| Backbone | box AP | mask AP | 參數量 |
|----------|--------|---------|--------|
| ResNet-50 | 46.3 | 40.1 | 82M |
| DeiT-S | 48.0 | 41.4 | 80M |
| **Swin-T** | **50.5** | **43.7** | 86M |
| ResNeXt-101-64x4d | 48.3 | 41.7 | 140M |
| **Swin-B** | **51.9** | **45.0** | 145M |

Swin-T 以與 DeiT-S 相近的參數量，高出 +2.5 box AP 和 +2.3 mask AP。值得注意的是，Swin-T 的推論速度（15.3 FPS）遠快於 DeiT-S（10.4 FPS）——這是因為 Swin 的線性複雜度在密集預測的高解析度特徵圖上顯現了優勢。

在系統級別比較中，Swin-B 搭配 HTC++ 達到 56.4 box AP，比 Copy-paste（當時的 SOTA，55.9 box AP）更高。

**ADE20K 語義分割結果**

使用 UperNet 框架：

| Backbone | mIoU (val) |
|----------|-----------|
| DeiT-S | 44.0 |
| Swin-T | **46.1** |
| Swin-S | **49.3** |
| Swin-B | **51.6** |
| Swin-L (ImageNet-22K) | **53.5** |

Swin-L 的 53.5 mIoU 超越了 SETR（純 Transformer 分割方法）的 50.3 mIoU，高出 +3.2。

**與 ViT 的資料需求對比**

ViT 的一個重要結論是「大型 Transformer 需要大型資料集」——ViT 在 ImageNet-1K 上從頭訓練的表現不如 ResNet。但 Swin 不同：Swin-T 在 ImageNet-1K 上從頭訓練即可達到 81.3%，超越了同級 ResNet 和 DeiT。這說明**局部性設計降低了對資料量的需求**——shifted window 提供的歸納偏置讓 Swin 在中小規模資料上也能有效學習。

---

## 實驗結果

### 主要實驗

| 任務 | 資料集 | Swin 表現 | 對比 SOTA | 超越幅度 |
|------|--------|----------|-----------|---------|
| 分類 | ImageNet-1K top-1 | 87.3% (Swin-L, 22K pretrain) | 85.4% (ViT-L) | +1.9% |
| 偵測 | COCO test-dev box AP | 56.4 (Swin-B, HTC++) | 55.9 (Copy-paste) | +0.5 |
| 分割 | ADE20K val mIoU | 53.5 (Swin-L) | 50.3 (SETR) | +3.2 |

### 消融實驗

論文的 Table 4 對 shifted window 和相對位置偏置做了消融分析。關鍵發現：

- **Shifted window vs. regular window only**：在 ImageNet 上，只用 regular window（無跨視窗連接）會導致約 -1% 的 top-1 準確率。shifted window 帶來的跨視窗連接對模型表達能力至關重要。
- **Cyclic-shift vs. naive padding**：cyclic-shift 方法在延遲上遠低於 naive padding（在 2×2 視窗的場景下，因視窗數量從 4 增加到 9，計算量是 2.25 倍），但兩者的準確率相當。

### 限制

- **與高度最佳化的 CNN 相比，推理速度仍有差距**：論文中承認 Swin 的實現使用 built-in PyTorch 函數而非高度最佳化的 kernel（如 cuDNN 對 ResNet 的優化）。Swin 的 FPS 在同等 FLOPs 下仍然落後於 CNN。後續的 ConvNeXt 等工作會在某種程度上解決這個問題。
- **訓練穩定性**：Swin 雖然比 ViT 更容易訓練，但在超參數選擇（如 learning rate、weight decay）上仍比 ResNet 敏感。

---

## 與相關工作的對比

| 維度 | Swin Transformer | Vision Transformer (ViT) | ResNet (CNN) |
|------|-----------------|-------------------------|-------------|
| 特徵圖解析度 | 多尺度（4×, 8×, 16×, 32×） | 單一（16×） | 多尺度 |
| 注意力方式 | 局部窗口 + shifted window | 全局自注意力 | 卷積（無注意力） |
| 計算複雜度 | 線性（對圖像大小） | 二次 | 線性 |
| 歸納偏置 | 局部性（透過視窗） | 極少（只透過 patch extraction） | 大量（local、平移等變） |
| 所需訓練資料 | ImageNet-1K 即足夠 | 需要 JFT-300M 級別 | ImageNet-1K 即足夠 |
| 密集預測任務 | 原生支援（可直接接入 FPN） | 需後處理（upsampling） | 原生支援 |

---

## 我的觀察

Swin Transformer 在架構設計上的一個巧妙之處在於：它把 CNN 的「層級式」和 Transformer 的「注意力」結合的方式，不是將注意力作為 CNN 的補充，而是反過來——**用 Transformer 的詞彙重新實現了 CNN 的骨架**。Swim 的 4 個 stage、patch merging（類比於 pooling/strided convolution）、局部視窗（類比於卷積的局部感受野）——這些都是 CNN 的概念，但實現方式全部換成了注意力。

另一個值得注意的點是：Swin 的 shifted window 策略與分組卷積（group convolution）在精神上很像——都是將完整的運算拆成不相交的群組來降低計算量，然後透過某種方式在群組之間建立通訊。ShuffleNet 用 channel shuffle，Swin 用 shifted window partition。

論文的結果在各種任務上都極具說服力——尤其是偵測和分割任務上的優勢遠大於分類任務，這正是 Swin 相對於 ViT 的設計優勢所在。Swin Transformer 在 2021 年發表後迅速成為視覺 Transformer 的主流架構，這個地位一直維持到 ConvNeXt 等 CNN-Transformer 混合架構出現為止。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** ([2010.11929](https://arxiv.org/abs/2010.11929))
   - 與本文關係：第一個證明純 Transformer 可在圖像分類上超越 CNN 的論文。Swin Transformer 的核心動機——解決 ViT 的單尺度與二次複雜度問題——正是基於此工作。

### 後續發展（未涵蓋，僅列出）

- [ConvNeXt (2022)](https://arxiv.org/abs/2201.03545) — 受 Swin Transformer 啟發的純 CNN 設計
- [CSWin Transformer (2022)](https://arxiv.org/abs/2107.00652) — 十字形視窗注意力

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
