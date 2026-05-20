# DeiT: Data-efficient Image Transformers —— 讓 ViT 不再需要超大資料集

> **種子論文**: [Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) (2020-12)
> **作者**: Hugo Touvron, Matthieu Cord, Matthijs Douze et al.
> **機構**: Facebook AI / Sorbonne University

---

## TL;DR

Vision Transformer (ViT) 雖然在分類任務上表現卓越，但需要 JFT-300M 這類數億張影像的超大規模預訓練資料才能超越 CNN。DeiT 提出了**一套資料高效的訓練配方**，搭配一個**專為 transformer 設計的知識蒸餾機制（蒸餾令牌）**，讓 ViT 僅用 ImageNet-1k 與單機三天訓練就能達到與最先進 CNN 競爭的表現（83.1% top-1），並在結合 convnet 教師輔助後達到 85.2%。

---

## 背景與動機

2020 年，Vision Transformer (ViT) 證明了一件事：純 transformer 架構不需要卷積神經網路固有的歸納偏置（inductive bias），只要資料量夠大，就能在影像分類任務上超越 CNN。但這個「夠大」的條件非常嚴格——ViT 在 ImageNet-1k（130 萬張）上的表現反而不如 ResNet，必須先在 ImageNet-21k（1400 萬張）或 JFT-300M（3 億張）上預訓練，再 fine-tune 到下游任務，才有競爭力。

這個「資料需求門檻」帶來了三個實際問題：

- **計算資源門檻高**：JFT-300M 是 Google 內部資料集，外部研究人員無法使用
- **複現困難**：論文的訓練需要大量 TPU/GPU 叢集，資源不足的研究者難以驗證
- **生態割裂**：transformer 在 NLP 的優勢是可從頭訓練，但在 CV 領域卻需要特殊的大型預訓練才能發揮

DeiT 要回答的問題很直接：**ViT 能否在不使用外部資料、僅用 ImageNet-1k 的前提下，透過更好的訓練策略和蒸餾技術來達到與 CNN 競爭的程度？**

---

## 核心知識點

本文圍繞以下知識點展開。這是理解 DeiT 的關鍵概念，後續章節會依序展開：

1. **Vision Transformer (ViT) 架構**——ViT 如何將影像視為 patch 序列來處理
2. **ViT 的資料需求瓶頸**——為什麼 transformer 在小型資料集上不如 CNN
3. **DeiT 的資料高效訓練配方**——哪些 augmentation 和 regularization 是關鍵
4. **蒸餾令牌 (Distillation Token)**——專為 transformer 設計的知識蒸餾機制
5. **Hard-label Distillation**——一種更簡單有效的蒸餾損失函數
6. **ConvNet Teacher 優於 Transformer Teacher**——為什麼 inductive bias 可以透過蒸餾傳遞

---

## 方法詳解

### 知識點 1: Vision Transformer (ViT) 架構

**這個知識點要回答什麼問題？** ViT 如何用一套從 NLP 搬來的 transformer 架構來處理影像？

ViT 的設計極簡：將固定大小的 RGB 影像 $x \in \mathbb{R}^{H \times W \times C}$ 切割成 $N$ 個 $16 \times 16$ 的 patch，其中 $N = HW / 16^2$。每個 patch 透過線性投影（可訓練的矩陣 $E \in \mathbb{R}^{(16^2 \cdot C) \times D}$）映射為 $D$ 維的 patch embedding：

$$z_0 = [x_{\text{class}}; x^1_p E; x^2_p E; \dots; x^N_p E] + E_{\text{pos}}$$

其中 $x_{\text{class}}$ 是 class token（從 BERT 繼承的設計），$E_{\text{pos}}$ 是位置編碼。

這串 $(N+1)$ 個 token 依序送入標準的 transformer encoder block：

$$z'_\ell = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}$$
$$z_\ell = \text{MLP}(\text{LN}(z'_\ell)) + z'_\ell$$

最終只取 class token 對應的輸出 $y = \text{LN}(z^0_L)$ 接上線性分類器進行預測。

**這個架構的關鍵特徵**：self-attention 是全局的——每個 patch token 可以關注所有其他 patch token，不依賴 CNN 那種局部感受野的設計。這既是優點（可以建模長程依賴），也是缺點（缺少歸納偏置，需要更多資料才能學會有用的空間關係）。

### 知識點 2: ViT 的資料需求瓶頸

**這個知識點要回答什麼問題？** 為什麼同樣是 transformer，在 NLP 可以從頭訓練，在 CV 就需要超大資料集？

ViT 論文本身已經指出這個問題：「*Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.*」

具體來說：

- **CNN 的歸納偏置**：卷積運算假設「空間局部性」和「平移等變性」，這等於內建了一個很強的結構化先驗——讓網路在一開始就知道相鄰像素是相關的、物體位置平移後分類結果不變。這讓 CNN 在資料量少時仍然能學到有用的表徵。
- **Transformer 的全局注意力**：ViT 的 self-attention 在初始化時對所有 patch 一視同仁，patch 之間的位置關係全靠位置編碼去學。沒有 CNN 那種「局部→全局」的層級結構，因此在資料不足時搜尋空間過大。

DeiT 的策略不是改變架構來加入歸納偏置，而是**用資料增廣和正則化來補足 transformer 缺少的歸納偏置**，讓 ViT 在有限資料下也能收斂。

### 知識點 3: DeiT 的資料高效訓練配方

**這個知識點要回答什麼問題？** 除了 ViT 原有的架構，DeiT 換了哪些訓練設定讓它在 ImageNet-1k 就能訓練好？

DeiT 對 ViT-B（86M 參數）調整了整個訓練流程，其預設配置摘要如下：

| 設定 | ViT-B | DeiT-B |
|------|-------|--------|
| Epochs | 300 | 300 |
| Batch size | 4096 | 1024 |
| Optimizer | AdamW | AdamW |
| Learning rate | 0.003 | 0.0005 × batchsize/512 |
| 排程 | cosine | cosine |
| Weight decay | 0.3 | 0.05 |
| Label smoothing | 無 | 0.1 |
| Dropout | 0.1 | **不使用** |
| Stochastic Depth | 無 | 0.1 |
| Repeated Aug | 無 | 有（3 次重複） |
| Rand Augment | 無 | 9/0.5 |
| Mixup | 無 | 0.8 |
| CutMix | 無 | 1.0 |

幾個值得注意的觀察：

**Dropout 有害**。這與直覺相反——transformer 本來就容易 overfit，為什麼關掉 dropout 反而更好？DeiT 的消融實驗顯示，當 Stochastic Depth 和其他 data augmentation 同時啟用時，dropout 反而限制了模型容量。Stochastic Depth 已經提供了有效的正則化，疊加 dropout 會過度正則化。

**Repeated Augmentation 是關鍵**。DeiT 將每張訓練圖片做 3 次不同的 augmentation 後全部餵進一次 epoch 中，而不是只選一個 augmentation。這讓模型在同樣的訓練時長內看到更多樣的資料變體。論文標註這是最重要的單一成分之一。

**Batch size 從 4096 降到 1024**。較小的 batch size 配合 learning rate 按 `lr_scaled = lr_base × batchsize/512` 縮放，讓訓練更穩定。這與 ViT 使用大 batch size 的設定形成對比。

**AdamW weight decay 大幅降低**。ViT 使用 0.3 的 weight decay，DeiT 降到 0.05。論文指出 ViT 的 weight decay 在 DeiT 的訓練設定下阻礙收斂。

### 知識點 4: 蒸餾令牌 (Distillation Token)

**這個知識點要回答什麼問題？** 如何讓 transformer 在同樣參數量下表現更好？DeiT 的答案是「向更強的模型學習」，但蒸餾的方式是為 transformer 量身訂做的。

傳統的知識蒸餾是在損失函數層面進行的——在 classification loss 之外加上 teacher 的 soft label 作為輔助監督。DeiT 提出了一個架構層面的創新：在 token 序列中加入一個**蒸餾令牌 (distillation token)**。

蒸餾令牌的設計邏輯：

- 它與 class token 平行存在，初始時是一個可訓練的向量
- 它同樣經過所有 transformer block，與 patch tokens 和其他 tokens 透過 self-attention 互動
- 在最後一層，它的輸出經過另一個線性分類器，目的是**預測 teacher 的預測結果**（而非真實標籤）

視覺上，這相當於在 transformer 中嵌入了一個「模仿教師」的子路徑。class token 專注於真實標籤，distillation token 專注於模仿教師，兩條路徑透過 self-attention 互相交換資訊。

一個重要的實驗證據：class token 和 distillation token 在底層的 cosine similarity 只有 0.06（幾乎不相關），但在頂層提升到 0.93（高度相關但不等於 1）。作為對照，如果在此架構中加入第二個 class token（兩者都預測真實標籤），兩個 token 在訓練後會收斂到幾乎完全相同（cos=0.999），對效能完全沒有幫助。這證明 distillation token 的有效性來自於**它學到了與 class token 互補但不同的表徵**。

測試時，DeiT 可以有三種預測模式：
1. 只用 class embedding（與一般 ViT 相同）
2. 只用 distillation embedding
3. **兩者融合**（論文預設的推薦模式，將兩個分類器的 softmax 輸出相加）

### 知識點 5: Hard-label Distillation

**這個知識點要回答什麼問題？** 蒸餾的損失函數應該怎麼設計才是最適合 transformer 的？

傳統的知識蒸餾（soft distillation）使用 KL divergence：

$$\mathcal{L}_{\text{global}} = (1-\lambda)\mathcal{L}_{\text{CE}}(\psi(Z_s), y) + \lambda \tau^2 \text{KL}(\psi(Z_s / \tau), \psi(Z_t / \tau))$$

其中需要調整溫度參數 $\tau$ 和平衡權重 $\lambda$。

DeiT 提出了一種更簡潔的變體——**hard-label distillation**。想法很直接：直接將 teacher 的硬決策 $y_t = \arg\max_c Z_t(c)$ 視為 pseudo-label，然後同時最小化對真實標籤和 teacher 標籤的 cross-entropy：

$$\mathcal{L}_{\text{hardDistill}}^{\text{global}} = \frac{1}{2}\mathcal{L}_{\text{CE}}(\psi(Z_s), y) + \frac{1}{2}\mathcal{L}_{\text{CE}}(\psi(Z_s), y_t)$$

這個設計有幾個優點：

1. **無需調參**——不需要 $\tau$ 和 $\lambda$ 這類超參數，直接使用標準 cross-entropy
2. **概念簡單**——teacher 的角色就像一個「第二標註者」，與真實標籤同等看待
3. **表現更好**——論文實驗證實 hard-label distillation 優於 soft distillation

注意一個細節：teacher 的預測 $y_t$ 會隨 augmentation 改變（同一張圖 crop 不同位置可能得到不同的 teacher 預測）。DeiT 認為這不是問題，甚至可能是一種有益的隨機性。

### 知識點 6: ConvNet Teacher 優於 Transformer Teacher

**這個知識點要回答什麼問題？** 什麼樣的 teacher 對 transformer student 最有幫助？

一個反直覺的發現：用 convnet（RegNetY-16GF）當 teacher，比用另一個同樣強的 transformer 當 teacher**更好**。

DeiT 分析了多個模型之間的決策不一致率（disagreement rate）：

| 模型對 | 不一致率 |
|--------|---------|
| ConvNet vs DeiT (無蒸餾) | 13.3% |
| ConvNet vs DeiT⚗ (class token) | 11.2% |
| ConvNet vs DeiT⚗ (distillation token) | **10.0%** |
| DeiT⚗ class vs distillation token | 5.0% |

從表中可以清楚看到：
- 蒸餾後（DeiT⚗），distillation token 的預測最接近 convnet teacher（10.0% 不一致）
- class token 雖然也接近 teacher（11.2%），但保留了更多自己的特性
- 兩個 token 的預測高度一致（5.0%），說明兩者學到的表徵有大量重疊但不完全相同

這個結果的解釋來自 Abnar et al. (2020) 的觀點：**知識蒸餾可以傳遞歸納偏置**。ConvNet 內建的空間局部性和平移等變性等歸納偏置，可以透過蒸餾軟性地「轉移」給 transformer student——student 不用直接整合卷積運算，但透過模仿 convnet 的決策模式，學會了類似於擁有了那些歸納偏置時才有的決策行為。這是 DeiT 中最具洞察力的發現之一。

---

## 實驗結果

### 主要結果：ImageNet top-1 準確率 vs Throughput

DeiT 的核心比較是「僅在 ImageNet-1k 上訓練」的設定，對比 EfficientNet 系列（當時最先進的 CNN）和 ViT。

| 模型 | 參數量 | 解析度 | ImageNet top-1 | ImageNet Real |
|------|--------|--------|---------------|---------------|
| EfficientNet-B5 | 30M | 456² | 83.6% | 88.3% |
| **DeiT-B** (無蒸餾) | **86M** | 224² | 81.8% | 86.7% |
| **DeiT-B↑384** (無蒸餾) | **86M** | 384² | 83.1% | 87.7% |
| **DeiT-B⚗** (有蒸餾) | **87M** | 224² | 83.4% | 88.3% |
| **DeiT-B⚗↑384** (蒸餾+高解析) | **87M** | 384² | 84.5% | 89.0% |
| **DeiT-B⚗↑384 / 1000 epochs** | **87M** | 384² | **85.2%** | **89.3%** |
| ViT-B/16 (JFT-300M pre-train) | 86M | 384² | 84.15% | — |

**關鍵觀察**：

1. DeiT-B 無蒸餾已達到 83.1%（384²），大幅超越 ViT 在 ImageNet-1k 上的 77.9%
2. DeiT-B⚗↑384 的 85.2%（1000 epochs）甚至超過了 ViT-B 使用 JFT-300M 預訓練的 84.15%
3. 加入蒸餾後，同樣參數量下準確率提升約 1.6 個百分點
4. 更長的訓練時間（300→1000 epochs）對蒸餾模型有持續的增益，但對無蒸餾模型幫助有限

### 遷移學習表現

| 模型 | CIFAR-10 | CIFAR-100 | Flowers | Cars | iNat-18 | iNat-19 |
|------|---------|-----------|---------|------|---------|---------|
| ViT-B/16 | 98.1% | 87.1% | 89.5% | — | — | — |
| **DeiT-B** | **99.1%** | **90.8%** | **98.4%** | **92.1%** | 73.2% | 77.7% |
| **DeiT-B⚗↑384** | **99.2%** | **91.4%** | **98.9%** | **93.9%** | **80.1%** | **83.0%** |

DeiT 在遷移學習上的表現也優於 ViT，特別是在細粒度分類（Flowers、Cars）上優勢明顯。

### 消融實驗摘要

DeiT 對訓練配方的每個元件進行了系統性消融（從 224² 開始訓練後 fine-tune 到 384²）：

| 移除項目 | 224² 準確率變化 | 384² 準確率變化 |
|---------|----------------|----------------|
| 全部元件（SGD 優化器） | -7.3% | -5.8% |
| 移除所有 data augmentation | -2.2% | -2.7% |
| 僅使用 AutoAugment（無 RandAugment） | -0.6% | -1.2% |
| 僅使用 Mixup（無 CutMix） | -1.8% | -2.5% |
| 僅使用 CutMix（無 Mixup） | -3.1% | -2.5% |
| 移除 Stochastic Depth | +0.1% | 0% |
| 移除 Repeated Aug | -0.5% | 0% |
| 啟用 Dropout | -4.9% | +0.1% |

最有意思的消融結果是 dropout：在 224² 階段啟用 dropout 會導致 4.9% 的嚴重下降，但在 384² fine-tune 階段影響很小。這說明 dropout 對從頭訓練有害，但對 fine-tune 影響不大。

---

## 與相關工作的對比

| 維度 | ViT (Dosovitskiy et al.) | DeiT (Touvron et al.) |
|------|--------------------------|----------------------|
| 模型架構 | 純 transformer | 與 ViT 相同 |
| 訓練資料 | JFT-300M / ImageNet-21k | ImageNet-1k only |
| 訓練資源 | 大量 TPU | 4–8 GPU, 2–3 天 |
| Data augmentation | 基本（無 RandAugment/Mixup） | 多種（RandAug, Mixup, CutMix, etc.） |
| 知識蒸餾 | 無 | 蒸餾令牌 + hard-label distillation |
| 外部資料依賴 | 必要 | 非必要 |
| ImageNet top-1 (僅 1k) | 77.9% (ViT-B/16) | **83.1% (DeiT-B↑384)** |

---

## 我的觀察

DeiT 的貢獻其實分為兩個層次，一個是工程層面，一個是方法層面：

**工程層面的貢獻**：證明了 transformer 在視覺領域的「資料效率瓶頸」可以透過更好的訓練配方來突破，不需要改變架構。這個發現的重要性在於它降低了 vision transformer 的研究門檻——不再需要 Google 級別的資料和算力才能產出有意義的結果。

**方法層面的貢獻**：蒸餾令牌的設計雖然簡單，但它體現了一個重要的設計原則——**在 transformer 中加入一個專用 token 來承載特定的監督訊號**。這個思路後來在多模態、物件偵測等領域持續被使用。class token 和 distillation token 學到不同表徵的現象也說明了 transformer 的 token 機制比單純的「特徵向量拼接」更靈活。

不過 DeiT 也有一些限制：蒸餾需要一個已經訓練好的 teacher 模型，這在某些場景（如全新任務）可能不可行。此外，論文的主要實驗集中在 ImageNet 分類上，對於其他視覺任務（如偵測、分割）的資料效率問題沒有深入探討。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale** ([2010.11929](https://arxiv.org/abs/2010.11929))
   - ViT 是 DeiT 的基礎架構，DeiT 的目的就是讓 ViT 可以不用大規模預訓練。不理解 ViT 的架構與資料需求瓶頸，就無法理解 DeiT 的設計動機。

### 後續發展（未涵蓋，僅列出）

- [LeViT: a Vision Transformer in ConvNet's Clothing for Faster Inference](https://arxiv.org/abs/2104.01136) (2021-04) —— DeiT 團隊後續工作，結合 CNN 與 transformer 的混合架構
- [DeiT III: Revenge of the ViT](https://arxiv.org/abs/2204.07118) (2022-04) —— 進一步改進訓練配方，達到更強的性能

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
