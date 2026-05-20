# Vision Transformer (ViT): An Image is Worth 16x16 Words 解讀

> **種子論文**: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) (2020-10)
> **作者**: Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov et al.
> **機構**: Google Research, Brain Team

---

## TL;DR

CNN 長期主宰電腦視覺領域，因為其 locality 與 translation equivariance 被視為是處理影像的必要歸納偏差。ViT 直接把標準 Transformer encoder 應用於從影像切出的 patch 序列上，除了最初步的 patch 切割外完全不引入影像專屬的歸納偏差，並發現只要在夠大的資料集上預訓練，這種極簡設計就能超越最先進的 CNN。在 JFT-300M 上預訓練後，ViT-H/14 在 ImageNet 達到 88.55% top-1 準確率，且預訓練計算量僅為 ResNet 基準的 1/4。

---

## 背景與動機

在 ViT 出現之前，Transformer 在 NLP 領域已是絕對主流（BERT、GPT 系列），但在電腦視覺中，CNN 仍是無可撼動的霸主。研究者嘗試過幾種將注意力機制引入視覺的方式：

- **以注意力輔助 CNN**：Non-local networks、Attention Augmented Convolution 等在 CNN 骨架中加入 attention 模組
- **以注意力完全取代 convolution**：Stand-Alone Self-Attention、Axial Attention 等嘗試，但受限於專用的 attention pattern，在現有硬體上難以高效規模化

這些方法的核心共識是：CNN 的架構先驗（locality、translation equivariance）對視覺任務至關重要，注意力只能在局部範圍內引入。ViT 挑戰了這個預設，提出一個極簡問題：**如果直接把標準的 Transformer encoder 套在影像上，會不會 work？**

---

## 核心知識點

本文圍繞以下知識點展開，從 Transformer 的原始設計出發，逐步理解 ViT 的改動與發現：

1. **Transformer 編碼器架構**——Scaled dot-product attention、multi-head attention、position-wise FFN 的設計原理
2. **影像分塊與 Patch Embedding**——如何把 2D 影像轉換為 Transformer 可以處理的 1D token 序列
3. **位置編碼的兩種風格**——Transformer 的 sinusoidal PE 與 ViT 的可學習 1D PE，以及 2D 位置資訊的處理
4. **歸納偏差之爭**——ViT 極簡的歸納偏差 vs CNN 的強烈架構先驗，以及資料規模如何改變這個平衡
5. **預訓練-微調範例與 Scaling Behavior**——為什麼 ViT 在小資料集上輸給 CNN，在大資料集上卻能反超
6. **模型變體與計算效率**——ViT-B/L/H 的設計選擇、patch size 對計算量的影響、與 ResNet 的 compute-performance trade-off

---

## 方法詳解

### 知識點 1: Transformer 編碼器架構

**這個知識點要回答什麼問題？**

ViT 說「In model design we follow the original Transformer (Vaswani et al., 2017) as closely as possible」。要理解 ViT，必須先理解它繼承了什麼。

**原始 Transformer 怎麼設計的？**

Vaswani et al. (2017) 提出的 Transformer 原本是 sequence-to-sequence 模型，包含 encoder 與 decoder 兩個 stack。ViT 只取了 **encoder** 部分，其結構為：

每個 encoder layer 包含兩個子層：
1. **Multi-Head Self-Attention (MSA)** — 讓序列中的每個位置能關注所有其他位置
2. **Position-Wise Feed-Forward Network (FFN)** — 一個兩層的 MLP，對每個位置獨立運算

每個子層外都有 residual connection，並在 residual 相加後做 layer normalization：

$$
\text{output} = \text{LayerNorm}(x + \text{Sublayer}(x))
$$

Multi-head attention 的核心是 Scaled Dot-Product Attention。給定查詢 $Q$、鍵 $K$、值 $V$：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

除以 $\sqrt{d_k}$ 的 scaling 是關鍵設計——當維度較高時，dot product 的數值會變大，softmax 的梯度會趨近於零，scaling 能緩解這個問題。

Multi-head attention 將查詢、鍵、值分別投影到 $h$ 組較低的維度，各自做 attention 後拼接：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W^O
$$

這讓模型能在不同的表示子空間中同時關注不同的位置關係。

**ViT 怎麼繼承的？**

ViT 完全保留了上述結構。唯一的差別是 ViT 只使用 encoder（沒有 decoder），且輸出取 [CLS] token 的最終隱狀態作為影像表徵。公式完全一致：

$$
\begin{aligned}
z_0 &= [x_{\text{class}}; x_p^1 E; x_p^2 E; \cdots; x_p^N E] + E_{\text{pos}} \\
z'_\ell &= \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1} \\
z_\ell &= \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell \\
y &= \text{LN}(z_L^0)
\end{aligned}
$$

---

### 知識點 2: 影像分塊與 Patch Embedding

**這個知識點要回答什麼問題？**

Transformer 的輸入是 1D token 序列。一張 $224 \times 224 \times 3$ 的影像有 150,528 個像素值，不能直接丟進 Transformer——每個位置的 attention 計算量是 $O(n^2)$，全像素級別的序列長度為 150,528，計算上完全不可行。

**ViT 怎麼做的？**

ViT 的關鍵洞察是：**不需要 pixel-level 的細粒度**。它將影像切成固定大小的 patches，例如 $16 \times 16$ 的 patch：

$$
x_p \in \mathbb{R}^{N \times (P^2 \cdot C)}
$$

其中 $N = HW/P^2$ 為 patch 數量，$P$ 為 patch 大小。對 $224 \times 224$ 的影像搭配 $16 \times 16$ 的 patch，$N = 196$——這與 NLP 中常見的序列長度（BERT 為 512 tokens）相當。

每個 patch 先 flatten 成 $P^2 \cdot C$ 維向量，再經由一個可學習的線性投影 $E \in \mathbb{R}^{(P^2 \cdot C) \times D}$ 映射到 Transformer 的隱藏維度 $D$。

與原始 Transformer 的 text embedding相比，ViT 的 patch embedding 是從像素值直接學習，而非 look-up table。

---

### 知識點 3: 位置編碼的兩種風格

**這個知識點要回答什麼問題？**

Self-attention 是 permutation-invariant 的——如果不加位置資訊，模型無法區分「patch 1 在左上角」與「patch 1 在右下角」。兩篇論文都處理了這個問題，但採用了不同的設計。

**原始 Transformer：Sinusoidal Positional Encoding**

Vaswani et al. 使用固定頻率的正弦與餘弦函數：

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin(pos / 10000^{2i/d_{\text{model}}}) \\
PE_{(pos, 2i+1)} &= \cos(pos / 10000^{2i/d_{\text{model}}})
\end{aligned}
$$

這樣的設計有兩個優點：一是不同位置的編碼可以表示為彼此的線性函數（利於模型學習相對位置）；二是在訓練中未見過的序列長度也能直接計算（外推能力）。

**ViT：可學習的 1D Position Embeddings**

ViT 使用更簡單的做法——直接讓模型從資料中學習 position embeddings $E_{\text{pos}} \in \mathbb{R}^{(N+1) \times D}$。值得注意的是，ViT **不使用 2D-aware 的位置編碼**，儘管影像有明顯的二維空間結構（上-下、左-右之間的距離關係不同）。論文的消融實驗顯示，更複雜的 2D position embeddings 並未帶來顯著提升。

這呼應了 ViT 的核心精神：**盡可能減少影像專屬的歸納偏差，讓模型自己從資料中學習空間關係**。

---

### 知識點 4: 歸納偏差之爭

**這個知識點要回答什麼問題？**

CNN 之所以在視覺領域如此成功，部分原因是它的架構天然嵌入了對視覺任務至關重要的歸納偏差：

- **Locality**：卷積核只在局部區域運算
- **Translation equivariance**：物體在影像中移動，表徵也跟著平移
- **層級式特徵抽取**：底層偵測邊緣、中層偵測形狀、高層偵測物體

ViT 的設計幾乎放棄了這些歸納偏差：

| 元件 | 是否帶有 2D 先驗 |
|------|-----------------|
| Patch 切割 | **是**—把影像切成 2D patches |
| Position embedding | 否—1D 可學習，初始化不帶 2D 資訊 |
| Self-attention | 否—全域性，每個 patch 都能關注所有 patches |
| MLP | 部分—每個位置獨立，平移等變但非局部 |

**這不是 bug，而是 feature。** ViT 的論證是：大規模預訓練可以補償歸納偏差的缺失。當資料夠多時，模型能自行學習到原本架構寫死的視覺先驗。

---

### 知識點 5: 預訓練-微調範例與 Scaling Behavior

**這個知識點要回答什麼問題？**

既然 ViT 缺乏強烈的歸納偏差，它需要在什麼樣的條件下才能真正發揮作用？

**ViT 的關鍵發現—Scaling Trumps Inductive Bias**

ViT 在三個規模的資料集上進行了實驗：

1. **ImageNet (1.3M 張)** — ViT 在此規模下不如 ResNet，差距約數個百分點
2. **ImageNet-21k (14M 張)** — ViT 開始接近 ResNet
3. **JFT-300M (3 億張)** — ViT **超越** ResNet

線性 few-shot 評估（Linear 5-shot ImageNet）揭示了更清楚的 pattern：

- ResNet 在少量資料下表現較好，但隨著資料量增加，效能提升趨於平緩
- ViT 在少量資料下表現較差，但隨著資料量增加，效能繼續提升，最終超過 ResNet

這意味著 ViT 的資料效率不如 CNN，但資料擴展性更好——給它更多資料，它的回報率更高。

**預訓練-微調流程**

ViT 遵循 NLP 中的標準範例：在大規模資料集上預訓練 → 在小規模的下游任務上微調。微調時移除預訓練的分類頭，換上一個零初始化的 $D \times K$ 線性層（$K$ 為下游任務的類別數）。微調時也常使用比預訓練更高的解析度（例如從 $224 \times 224$ 微調到 $384 \times 384$）。

---

### 知識點 6: 模型變體與計算效率

**這個知識點要回答什麼問題？**

ViT 是否真的比 CNN 更有效率？模型大小與 patch size 如何平衡效能與計算量？

**模型變體**

ViT 的模型尺寸沿用了 BERT 的命名與設定：

| 模型 | Layers | Hidden Dim $D$ | MLP Size | Heads | 參數量 |
|------|--------|---------------|----------|-------|-------|
| ViT-Base | 12 | 768 | 3072 | 12 | 86M |
| ViT-Large | 24 | 1024 | 4096 | 16 | 307M |
| ViT-Huge | 32 | 1280 | 5120 | 16 | 632M |

命名中附帶 patch size，如 ViT-L/16 表示 Large 模型 + $16 \times 16$ patch。Patch size 越小，序列長度越長，計算成本越高。

**計算效率的全面對比**

論文的 Scaling Study 比較了 7 個 ResNet（BiT）、6 個 ViT、5 個 Hybrid 模型，控制預訓練資料來源（JFT-300M），比較每種架構在相同計算預算下的傳輸表現：

- **ViT 以 2–4 倍少的計算量達到與 ResNet 相同的效能**
- **混合架構（CNN 特徵圖 + ViT）在小模型上略優於純 ViT，但大模型時差距消失**
- **ViT 在實驗範圍內未見飽和趨勢**—計算量越大，效能的提升空間仍在

TPUv3-core-days 的比較特別直觀：ViT-H/14 預訓練花費 2.5k TPU 天，而達到相近甚至更低效能的 BiT-L (ResNet152x4) 花費了 9.9k TPU 天。

---

## 實驗結果

### 主要實驗

| 資料集 | ViT-H/14 (JFT) | ViT-L/16 (JFT) | BiT-L (R152x4) | Noisy Student (EfficientNet) |
|--------|---------------|----------------|----------------|------------------------------|
| ImageNet | **88.55%** | 87.76% | 87.54% | 88.5% |
| ImageNet ReaL | **90.72%** | 90.54% | 90.54% | 90.55% |
| CIFAR-100 | **94.55%** | 93.90% | 93.51% | - |
| VTAB (19 tasks) | 77.63% | 76.28% | 76.29% | - |
| 預訓練計算量 | 2.5k TPU 天 | 0.68k TPU 天 | **9.9k** TPU 天 | **12.3k** TPU 天 |

**關鍵觀察**：

- ViT-H/14 在大多數資料集上達到或超越 SOTA，但預訓練計算量僅為 BiT-L 的 1/4
- ImageNet 上 Noisy Student 略高（88.5% vs 88.55%）但使用半監督學習與顯著的額外計算
- VTAB 的細分群體中，ViT 在 Natural 與 Specialized 群體表現突出，但在 Structured（需要幾何理解的任務）落後——這可能與缺乏 2D 歸納偏差有關

### 消融實驗

**Patch size 的影響**：更小的 patch（$14 \times 14$ vs $32 \times 32$）提升準確率但計算成本也更高。ViT-H/14 使用最小的 14 與最多的層數達成最佳結果。

**預訓練規模的影響**：ImageNet-21k 預訓練的效果明顯遜於 JFT-300M（ImageNet top-1 約 85.30% vs 88.55%），驗證了「規模化訓練是 ViT 成功的關鍵」的論點。

**自監督預訓練的初步探索**：以 masked patch prediction 預訓練的 ViT-B/16 達到 79.9%（vs 從頭訓練的 77.9%），仍落後監督式預訓練 4%。這是一個當時尚未解決的挑戰。

### 限制

- ViT 在小資料集上從頭訓練時效能明顯不如 CNN
- 缺少自監督式預訓練的有效方案（當時）
- 缺乏 2D 歸納偏差在需要幾何理解的任務（VTAB Structured 群體）上表現較弱
- 注意力的二次複雜度仍然是長序列時的限制

---

## 與相關工作的對比

| 維度 | Transformer (Vaswani et al.) | ViT (Dosovitskiy et al.) |
|------|------------------------------|--------------------------|
| 領域 | NLP（機器翻譯） | 電腦視覺（分類） |
| 架構 | Encoder-Decoder | 僅 Encoder |
| 輸入 | Word/token embeddings | Image patches |
| 位置編碼 | Sinusoidal（固定） | Learned 1D（可學習） |
| 歸納偏差 | 無序列位置先驗 | 極少（僅 patch 切割） |
| 預訓練 | 無（從頭訓練） | 在大規模資料上預訓練 |
| 核心貢獻 | 提出純 attention 架構 | 將 Transformer 直接應用於視覺 |

---

## 延伸閱讀

### Dependency Papers (本文涵蓋)

1. **Attention Is All You Need** ([1706.03762](https://arxiv.org/abs/1706.03762))
   - 與本文關係：ViT 的架構基石，ViT 直接沿用其 Transformer encoder 設計，包含 multi-head self-attention、position-wise FFN、layer normalization、residual connections

### 後續發展 (未涵蓋，僅列出)

- [DeiT: Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) (2020-12)—透過知識蒸餾讓 ViT 能在 ImageNet 從頭訓練
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) (2021-03)—引入階層式架構與移位視窗，適用於偵測與分割
- [MAE: Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) (2021-11)—有效的大規模自監督 ViT 預訓練方法
- [CLIP: Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021-03)—ViT 在 multi-modal 領域的經典應用

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
