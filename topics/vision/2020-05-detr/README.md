# DETR: End-to-End Object Detection with Transformers —— 用 Transformer 簡化物件偵測流程

> **種子論文**: [End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872) (2020-05)
> **作者**: Nicolas Carion, Francisco Massa, Gabriel Synnaeve et al.
> **機構**: Facebook AI

---

## TL;DR

物件偵測長期依賴 anchor、proposal、NMS（非極大值抑制）等手工設計元件，流程複雜且充滿啟發式規則。DETR 將偵測重新定義為**直接的集合預測問題**：只用一個 CNN backbone + transformer encoder-decoder + 匈牙利匹配損失，就端到端地輸出所有物件的邊界框。在 COCO 資料集上，DETR 以 42.0 AP 達到與高度優化的 Faster R-CNN 相當的準確率，在大物體上更勝出 +7.8 AP，且可自然延伸至全景分割任務。

---

## 背景與動機

物件偵測在 2020 年以前的主流框架可概括為「預測相對偏差」：兩階段偵測器（Faster R-CNN）先產生 proposal 再 refine，單階段偵測器（YOLO、FCOS、RetinaNet）則在 anchor 或網格中心點上做回歸和分類。這些方法雖然有效，但引入了大量手工設計的元件：

- **Anchor / Proposal 設計**：需要人工設定 anchor 的尺寸、長寬比、數量
- **正負樣本分配**：需要 heuristic assignment rules 來匹配 anchor 與 ground truth
- **NMS 後處理**：預測會產生大量重複框，必須靠 NMS 來清理，但 NMS 的閾值選擇會直接影響最終表現
- **這些元件之間存在耦合**：改了 anchor 設計可能需要重新調 NMS 參數，修改 NMS 又會影響 recall

DETR 的出發點很簡單：**能不能把物件偵測當作一個 sequence prediction 問題，用 transformer 直接輸出一個 set of detections，完全跳過這些手工設計的元件？**

這不是一個全新的想法——過去已有使用 RNN 做端到端 set prediction 的嘗試（如 Stewart et al. 2016），但這些方法只在小型資料集上驗證，且 RNN 的自回歸推論瓶頸讓它們無法與現代偵測器競爭。DETR 的關鍵突破在於用 **transformer 的 parallel decoding** 取代 RNN 的自回歸生成，這讓 set prediction 在 COCO 等級的大規模資料上首次可行。

---

## 核心知識點

1. **集合預測 (Set Prediction) 框架**——將偵測視為直接輸出物體集合，消除 NMS、anchor、proposal
2. **Transformer Encoder-Decoder 架構**——CNN backbone + transformer encoder（全局 self-attention）+ decoder（cross-attention 與 object queries）
3. **Object Queries**——N=100 個可學習的位置編碼，每個 query 透過注意力機制專注於特定區域與尺寸
4. **Bipartite Matching Loss（匈牙利匹配）**——預測集與真實集之間找最佳一對一匹配，從根本上消除重複預測
5. **Hungarian Loss：分類 + 邊界框回歸**——匹配後對每對計算 class CE + ℓ1 + GIoU 損失
6. **全景分割擴展**——在 decoder 輸出加上 mask head，用 pixel-wise argmax 統一做 thing/stuff 預測

---

## 方法詳解

### 知識點 1: 集合預測 (Set Prediction) 框架

**這個知識點要回答什麼問題？** 傳統偵測器的 NMS 和 anchor 為什麼是繞路？DETR 如何直接預測集合？

物體偵測的輸出本質上是一個集合（set）——一組邊界框，框之間沒有順序關係。但傳統偵測器把它當作「在每個 anchor/proposal 位置上做分類和回歸」的問題來處理，這會產生兩個問題：

1. **重複預測**：多個鄰近的 anchor 可能會對同一個物體都給出高置信度預測，所以需要 NMS 來清理
2. **排列變異性**：如果模型直接輸出一個集合，同一個 ground truth 集合可以對應多種預測順序，損失函數無法直接計算

DETR 的解法是一次性輸出 $N$ 個預測（$N$ 固定為 100，大於典型圖片中的物體數量），然後用 bipartite matching 找到預測與 ground truth 之間的最佳一對一配對。這個設計的核心邏輯是：**既然偵測的輸出是無序的集合，那損失函數就應該對預測的排列不敏感（permutation-invariant）。**

**這個知識點與 Transformer 的關係**：Transformer 的 self-attention 本身也是排列不變的——如果沒有位置編碼，改變輸入順序不會改變輸出。這讓 transformer 特別適合 set prediction：它的架構不需要假設輸入序列有固定的順序，正好與 DETR 的集合預測目標一致。

### 知識點 2: Transformer Encoder-Decoder 架構

**這個知識點要回答什麼問題？** DETR 如何把 transformer 用在影像上？CNN 和 transformer 怎麼分工？

DETR 的整體架構（圖 1 和圖 2）分為三個部分：

**1. CNN Backbone**：輸入影像 $x_{\text{img}} \in \mathbb{R}^{3 \times H_0 \times W_0}$ 經過標準的 CNN（ResNet-50 或 ResNet-101）產生低解析度的 activation map $f \in \mathbb{R}^{C \times H \times W}$，其中 $C=2048$，$H = H_0/32$，$W = W_0/32$。

**2. Transformer Encoder**：先用 $1 \times 1$ 卷積將通道數從 $C$ 降到 $d=256$，得到 $z_0 \in \mathbb{R}^{d \times H \times W}$。然後將空間維度壓平成一維序列，得到 $d \times HW$ 的序列（對 $800 \times 600$ 的影像，$HW$ 約為 $25 \times 19 \approx 475$）。加上空間位置編碼後，送入標準的 transformer encoder（6 層，每層 multi-head self-attention + FFN + residual + layer norm）。

**3. Transformer Decoder**：decoder 接收 $N=100$ 個可學習的 object queries（相當於 decoder 版本的位置編碼）。與原始 transformer 的關鍵差異是：DETR 的 decoder **不是自回歸的**——它在每一層同時對所有 $N$ 個 queries 進行 self-attention 和 encoder-decoder cross-attention，並在最後一層同時輸出所有 $N$ 個預測。

$$
\text{Decoder Input: } Q = \text{ObjectQueries} \in \mathbb{R}^{N \times d}
$$
$$
\text{Self-attention: } Q' = \text{MSA}(Q) + Q
$$
$$
\text{Cross-attention: } Q'' = \text{CrossAttn}(Q', \text{EncoderOutput}) + Q'
$$
$$
\text{FFN: } Q''' = \text{FFN}(Q'') + Q''
$$

**這個知識點與 Transformer 的關係**：DETR 的 encoder 和 decoder 架構幾乎完全來自原始 transformer（Vaswani et al. 2017）。唯一的主要修改是將自回歸解碼改為平行解碼，以及用 object queries 取代原本的 target sequence 輸入。

### 知識點 3: Object Queries

**這個知識點要回答什麼問題？** Decoder 怎麼知道要關注影像的哪些區域？N=100 個 slot 各自學到什麼？

Object queries 是 DETR 最巧妙的設計之一。可以把它們理解為 $N=100$ 個可學習的「探針」——每個 query 是一個 $d$ 維向量，初始化後在訓練過程中學會關注影像中的特定區域。

實驗發現這些 queries 會自發地**分工**（圖 7 的 slot analysis）：
- 某些 queries 專注於影像左半部的較小物體
- 某些 queries 專注於中央的大物體
- 所有 queries 都有一個模式是預測「全圖範圍的框」，這對應到 COCO 中的大面積背景框

值得注意的是，queries 之間透過 decoder 的 self-attention 交換資訊，所以它們可以互相抑制重複預測。論文實驗證實：在 decoder 第一層，由於還沒有 self-attention 的交互，預測中仍有大量重複；但在後續層次，self-attention 讓每個 query 知道其他 queries 在做什麼，重複預測自然消失。這解釋了為什麼 DETR 不需要 NMS——duplicate removal 是透過注意力機制隱式學習的。

### 知識點 4: Bipartite Matching Loss（匈牙利匹配）

**這個知識點要回答什麼問題？** 預測集合與真實集合之間如何建立對應？如何確保一對一匹配？

設真實集合為 $y = \{y_i\}$，預測集合為 $\hat{y} = \{\hat{y}_i\}_{i=1}^N$。由於 $N$ 大於實際物體數，真實集合用 $\varnothing$（無物體）填補到 $N$ 個元素。

DETR 尋找一個排列 $\sigma \in \mathfrak{S}_N$ 使得配對代價最小：

$$
\hat{\sigma} = \arg\min_{\sigma \in \mathfrak{S}_N} \sum_i^N \mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})
$$

這個最佳化問題用 Hungarian algorithm 高效求解（$O(N^3)$，$N=100$ 時非常快）。

配對代價 $\mathcal{L}_{\text{match}}(y_i, \hat{y}_{\sigma(i)})$ 綜合考慮：
- **分類**：當 $c_i \neq \varnothing$ 時，使用 $-\hat{p}_{\sigma(i)}(c_i)$（預測對真實類別的機率的負值）
- **邊界框**：使用 GIoU + ℓ1 距離

與傳統偵測器的關鍵差異：傳統方法用 heuristic rules（如 IoU 閾值）為每個 anchor 分配 ground truth，可能會出現多個 anchor 匹配到同一個物體（所以需要 NMS）。DETR 的 bipartite matching 是**唯一分配**——每個 ground truth 只能匹配到一個預測，每個預測也只能匹配到一個 ground truth。這從根本上杜絕了重複預測。

### 知識點 5: Hungarian Loss

**這個知識點要回答什麼問題？** 匹配完成後，如何對匹配上的預測計算最終的損失？

匹配完成後，對 $\hat{\sigma}$ 指定的每對 $(y_i, \hat{y}_{\hat{\sigma}(i)})$ 計算損失：

$$
\mathcal{L}_{\text{Hungarian}}(y, \hat{y}) = \sum_{i=1}^N \left[ -\log \hat{p}_{\hat{\sigma}(i)}(c_i) + \mathbb{1}_{\{c_i \neq \varnothing\}} \mathcal{L}_{\text{box}}(b_i, \hat{b}_{\hat{\sigma}(i)}) \right]
$$

邊界框損失 $\mathcal{L}_{\text{box}}$ 由兩部分組成：

$$
\mathcal{L}_{\text{box}}(b_i, \hat{b}_{\sigma(i)}) = \lambda_{\text{IoU}} \mathcal{L}_{\text{IoU}}(b_i, \hat{b}_{\sigma(i)}) + \lambda_{\text{L1}} \|b_i - \hat{b}_{\sigma(i)}\|_1
$$

其中使用 $\ell_1$ 加上 GIoU 的原因是：$\ell_1$ 損失對不同尺寸的框有尺度敏感性——同樣的相對誤差在大框和小框上會有不同的 $\ell_1$ 值；GIoU 是尺度不變的，但單獨使用 GIoU 時收斂較慢。兩者結合是最好的組合。

消融實驗驗證了這個設計選擇：
- 只用 $\ell_1$ 損失 → AP 35.8（-4.8）
- 只用 GIoU 損失 → AP 39.9（-0.7）
- 兩者結合 → AP **40.6**（baseline）

另外，匹配時使用的代價函數與最終的損失函數略有不同：匹配時用 $\hat{p}$（機率值）而非 $-\log \hat{p}$（log 機率），因為這樣 box loss 和 class loss 的尺度更一致，經驗上表現更好。

### 知識點 6: 全景分割擴展

**這個知識點要回答什麼問題？** 如果 DETR 本來就是 set prediction，同一套架構能否同時處理 instance 和 semantic segmentation？

DETR 的自然延伸是全景分割。方法很直接：在訓練好的 DETR decoder 輸出之上，為每個物件加上一個 **mask head**。

Mask head 的運作方式是：將每個物件的 decoder output embedding 與 encoder output 做 multi-head attention，產生每個物件的 attention heatmap（低解析度），再透過一個 FPN 風格的 CNN decoder 將解析度提升到原圖的 1/4。

最終的全景分割只需在每個像素位置上取所有 mask 預測的 argmax，並為每個 mask 分配相對應的類別標籤。這個流程保證了 mask 之間沒有重疊——不需要傳統全景分割方法中的 heuristic alignment 步驟。

結果顯示 DETR 在全景分割上特別擅長 thing 和 stuff 的統一處理，在 stuff 類別上表現尤其突出（因為 encoder 的全局 self-attention 擅長捕捉場景級別的上下文）。

---

## 實驗結果

### 主要結果：COCO 物件偵測 vs Faster R-CNN

| 模型 | GFLOPS | FPS | 參數量 | AP | AP₅₀ | AP₇₅ | AP_S | AP_M | AP_L |
|------|--------|-----|--------|----|------|------|------|------|------|
| Faster R-CNN-FPN | 180 | 26 | 42M | 40.2 | 61.0 | 43.8 | 24.2 | 43.5 | 52.0 |
| Faster R-CNN-FPN+ | 180 | 26 | 42M | 42.0 | 62.1 | 45.5 | 26.6 | 45.4 | 53.4 |
| **DETR** | **86** | **28** | **41M** | **42.0** | **62.4** | **44.2** | 20.5 | 45.8 | **61.1** |
| DETR-DC5 | 187 | 12 | 41M | 43.3 | 63.1 | 45.9 | 22.5 | 47.3 | 61.1 |
| DETR-R101 | 152 | 20 | 60M | 43.5 | 63.8 | 46.4 | 21.9 | 48.0 | 61.8 |
| DETR-DC5-R101 | 253 | 10 | 60M | 44.9 | 64.7 | 47.7 | 23.7 | 49.5 | 62.3 |

**關鍵觀察**：

1. **整體 AP 相當**：DETR 以 42.0 AP 與加速+強化的 Faster R-CNN-FPN+（42.0 AP）持平。考慮到 DETR 是第一個版本的 transformer 偵測器，而 Faster R-CNN 已經經歷了多年的迭代優化，這個結果非常有說服力
2. **大物體大幅領先（+7.8 AP_L）**：DETR 的全局 self-attention 能捕捉大範圍的空間關係，對大物體尤其有利。Faster R-CNN 的局部卷積有限感受野，處理大物體需要特徵金字塔（FPN）的輔助
3. **小物體明顯落後（-5.5 AP_S）**：encoder 的 self-attention 是在下採樣 32 倍後的 activation map 上做的，小物體在這麼低的解析度下只剩幾個像素，資訊太少
4. **參數效率更優**：DETR 用 41M 參數（86 GFLOPS）達到與 Faster R-CNN-FPN 42M（180 GFLOPS）相當的 AP，計算效率更好

### 消融實驗

**Encoder 層數的重要性**：

| #Encoder 層 | AP | AP_L |
|------------|----|------|
| 0（無 encoder） | 36.7 | 54.2 |
| 3 層 | 40.1 | 58.6 |
| **6 層（預設）** | **40.6** | **60.2** |
| 12 層 | 41.6 | 61.9 |

完全移除 encoder 時 AP 下降 3.9 點，大物體上的下降更顯著（-6.0 AP_L）。這證明了 encoder 的全局場景推理對於區分不同物體至關重要。論文將 encoder 的注意力圖視覺化，發現 encoder 已經具備初步的 instance separation 能力。

**Decoder 層數的影響**：AP 從第一層到第六層持續提升 8.2 AP。NMS 對第一層有幫助（消除重複），但對最後一層反而有害（錯誤移除真陽性）。這證明 decoder 的 self-attention 成功地學會了自行消除重複預測。

**位置編碼的重要性**：
- 完全移除空間位置編碼 → AP 降至 32.8（-7.8）
- 只在 decoder 輸入時提供一次位置編碼 → AP 39.2（-1.4）
- **在每層 attention 中都傳入位置編碼（預設）** → AP 40.6
- 令人驚訝的是：即使 encoder 中完全不傳遞位置編碼，AP 也只下降 1.3 點（39.3）

### 全景分割結果

| 模型 | Backbone | PQ | PQ_th | PQ_st |
|------|----------|----|-------|-------|
| PanopticFPN++ | R50 | 42.4 | 49.2 | 32.3 |
| UPSNet | R50 | 42.5 | 48.6 | 33.4 |
| **DETR** | **R50** | **43.4** | 48.2 | **36.3** |
| **DETR-R101** | **R101** | **45.1** | 50.5 | 37.0 |

DETR 在全景分割的 stuff 類別上領先明顯（+4.0 PQ_st vs PanopticFPN++），再次印證了全局 self-attention 對場景理解的好處。

---

## 與相關工作的對比

| 維度 | Faster R-CNN | DETR | Transformer (Vaswani et al.) |
|------|-------------|------|-----------------------------|
| 核心架構 | CNN + RPN + RoI Pooling | CNN + Transformer Enc-Dec | Transformer Enc-Dec only |
| 是否需要 anchor | 是（RPN 產生 proposal） | 否（object queries 自學） | 不適用 |
| 是否需要 NMS | 是（post-processing） | 否（內建於 attention） | 不適用 |
| 輸出方式 | 逐步 refine proposal | 平行解碼所有 N 個預測 | 自回歸（原始） |
| 全局推理 | 需 FPN / dilated conv | 內建於 self-attention | 內建於 self-attention |
| 特定領域設計 | 大量（anchor、NMS、FPN） | 極少 | 無（通用序列模型） |

---

## 我的觀察

DETR 對我來說最重要的貢獻不是它的 AP 數字——它在大物體上的優勢明顯，但在小物體上的弱勢也很明顯，後續才有 Deformable DETR 等改進來解決這個問題。DETR 真正的價值在於它**重新定義了物件偵測的問題框架**。

傳統偵測器的演進——從 Faster R-CNN 到 Mask R-CNN，從 RetinaNet 到 EfficientDet——大多是在同一個範式（anchor/proposal + NMS）內做優化。DETR 證明了另一個可能性：一個幾乎不需要任何任務特定設計的通用架構（CNN + transformer + Hungarian loss），就可以達到與高度專用的系統競爭的結果。這和 ViT 在分類任務上的突破是同一種思路——less prior knowledge = more general solution。

另一個有趣的洞察是 object queries 的自發分工行為。DETR 沒有對 100 個 queries 施加任何空間上的先驗（沒有說「query #1 負責左上角」），但它們透過 self-attention 和資料的統計規律，自發地學會了分工。這種 emergent specialization 是 attention 機制最吸引人的特性之一。

DETR 的限制也很清楚：小物體偵測差、訓練收斂慢（300–500 epochs vs Faster R-CNN 的 36–108 epochs）、計算 self-attention 的空間複雜度與特徵圖解析度成平方關係。這些限制在後續的 Deformable DETR、DAB-DETR、DN-DETR 等工作中得到了不同程度的緩解。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **Attention Is All You Need** ([1706.03762](https://arxiv.org/abs/1706.03762))
   - Transformer 架構的原始論文。DETR 直接採用其 encoder-decoder 設計，包括 multi-head self-attention、positional encoding、FFN 等組件

### 後續發展（未涵蓋，僅列出）

- [Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159) (2020-10) —— 用 deformable attention 解決收斂慢和小物體問題
- [DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329) (2022-01) —— 將 object queries 視為動態 anchor boxes
- [DN-DETR: Accelerate DETR Training by Introducing Query DeNoising](https://arxiv.org/abs/2203.01305) (2022-03) —— 透過去噪訓練加速收斂
- [DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object Detection](https://arxiv.org/abs/2203.03605) (2022-03) —— 結合多項改進，達到 SOTA

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
