# CLIP: Contrastive Language-Image Pre-training 論文導讀

> **種子論文**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) (2021-02)
> **作者**: Alec Radford, Jong Wook Kim, Chris Hallacy, et al. (OpenAI)
> **依賴論文**: [VirTex: Learning Visual Representations from Textual Annotations](https://arxiv.org/abs/2006.06666) — Karan Desai, Justin Johnson (CVPR 2021)

---

## TL;DR

CLIP 解決了傳統視覺模型只能辨識固定類別標籤的問題，讓模型從自然語言中學習視覺概念。它的核心方法是用對比式目標函數（contrastive loss）同時訓練一個影像編碼器和一個文字編碼器，以最大化正確圖文配對的餘弦相似度——不需要預測精確的文字，只需要判斷哪個文字配哪張圖片。在 400M 圖文對上訓練後，CLIP 能以零樣本方式在 ImageNet 上達到 ResNet-50 的全監督水準，並在 30+ 個資料集上展現出優異的泛化能力與分布偏移魯棒性。

---

## 背景與動機

### 固定類別分類的侷限

在 CLIP 之前，電腦視覺的主流做法是在 ImageNet 上預訓練後再遷移到下游任務。ImageNet 雖然有 1,000 個類別和 128 萬張標註圖片，但這種固定類別的分類頭有兩個根本限制：

- **封閉詞彙**：模型只能辨識預定義的類別，新增一個類別就得重新標註資料
- **語義稀疏**：一張圖片只對應一個標籤，遺失了物件屬性、空間關係、動作等豐富訊息

### 自然語言作為監督訊號的潛力

自然語言提供了完全不同的方向。網路上的圖文配對數量極大（部落格、新聞、社群媒體），且不需要人工標註固定格式。如果模型能從這些自然語言描述中學習視覺概念，就能：

1. **避開標註瓶頸**——文字是網路原生存在的，不需要大規模 crowd-sourcing
2. **學到開放詞彙**——自然語言可以表達任何視覺概念，不限於 1,000 個類別
3. **自然支援零樣本**——學到的視覺表徵天然與語言對齊，可用文字查詢視覺內容

### CLIP 之前的前置工作

這條路線並不新。早在 2016 年，Joulin et al. 就嘗試在 YFCC100M 上用 image tags 做多標籤分類。到了 2020 年，幾個工作進一步展現了自然語言監督的潛力：

- **VirTex** (Desai & Johnson, 2020)：以生成式方法（image captioning）訓練，從 COCO Captions（118K 圖）學習視覺表徵
- **ConVIRT** (Zhang et al., 2020)：在醫學影像上使用對比式目標
- **ICMLM** (Sariyildiz et al., 2020)：使用 masked language modeling

但這些前期工作的規模都太小——訓練在加速器天數而非年數的量級。CLIP 的關鍵洞察是：**自然語言監督的潛力需要足夠大的規模才能釋放**。

---

## 核心知識點

本文圍繞以下 6 個知識點展開：

1. **自然語言監督的動機與優勢**——為什麼要拋棄固定類別分類
2. **生成式 vs 對比式：兩種預訓練範式**——VirTex 與 CLIP 的架構對比與效率差異
3. **對比式目標函數（InfoNCE/N-pair loss）與雙編碼器架構**——CLIP 的核心演算法
4. **資料規模與預訓練資料建構**——從 118K 到 400M 圖文對的飛躍
5. **零樣本遷移：以自然語言作為彈性分類器介面**——如何用 prompt 實現跨資料集分類
6. **評估架構與分布偏移魯棒性**——30+ 資料集的系統性評估

---

## 方法詳解

### 知識點 1：自然語言監督的動機與優勢

**為什麼要轉向自然語言？**

傳統的 supervised pretraining 在 ImageNet 上達到巔峰後，要繼續提升就需要更大的標註資料集。但標註成本高昂（ImageNet 1,000 類就耗費大量人力），且類別之間的不平衡、長尾分佈等問題難以迴避。

自然語言監督的關鍵優勢在於：

1. **可擴展性（scalability）**：文字是網路原生的副產品，不需要人工轉換成 1-of-N 格式
2. **語義豐富度**：caption 可以描述物件的屬性（orange and white cat）、空間關係（cat near a plate）、動作（looking at apples），遠比一個類別標籤豐富
3. **連動語言**：學到的視覺表徵天然與文字對齊，可直接用於 zero-shot transfer 和 retrieval，不需要額外的對齊步驟

CLIP 論文中特別強調：雖然這些方法被各自稱為 unsupervised、self-supervised、weakly supervised，但它們的共同本質都是 **learning from natural language supervision**。

### 知識點 2：生成式 vs 對比式——兩種預訓練範式

這是本文最關鍵的對比，也是選擇 VirTex 作為 dependency paper 的原因。

**VirTex 的生成式方法**

VirTex 採用 image captioning 作為預訓練任務。模型由兩部分組成：

- **Visual backbone**：ResNet-50（影像編碼器）
- **Textual head**：雙向 Transformer decoder（Forward + Backward），以自迴歸方式預測 caption 的每一個 token

訓練目標是最大化 caption token 的對數似然：

$$
\mathcal{L}(\theta, \phi) = \sum_{t=1}^{T+1} \log p(c_t | c_{0:t-1}, I; \phi_f, \theta) + \sum_{t=0}^{T} \log p(c_t | c_{t+1:T+1}, I; \phi_b, \theta)
$$

預訓練後，丟棄 textual head，只保留 visual backbone 遷移到下游任務。

VirTex 的關鍵侷限在於它使用的是 **predictive objective**——必須預測 caption 的精確用詞。同一張圖片可以有不同的描述方式（"a brown dog" vs "a puppy with brown fur"），強迫模型預測精確詞彙不僅困難，也浪費了監督訊號。

**CLIP 的對比式方法**

CLIP 放棄預測精確詞彙，改為解決一個更簡單的代理任務：**判斷哪個文字配哪張圖片**。給定 batch 內 $N$ 對（image, text），CLIP 要從 $N \times N$ 種可能的配對組合中，找出正確的 $N$ 對。

CLIP 論文中用圖 2 的實驗清楚展示了兩者的效率差異：

| 方法 | 相對效率 |
|------|---------|
| Transformer Language Model（生成式） | 1×（基線） |
| Bag-of-Words Prediction | 3×（比生成式快） |
| Contrastive (CLIP) | 12×（比生成式快，比 BoW 快 4×） |

對比式目標在零樣本 ImageNet 上的學習效率是生成式的 **12 倍**。這說明了為什麼 CLIP 選擇對比式路徑——在自然語言監督的規模下，訓練效率是決定成敗的關鍵。

### 知識點 3：對比式目標函數（InfoNCE/N-pair loss）與雙編碼器架構

CLIP 的核心演算法非常簡潔，論文中給出了完整的 pseudocode：

```
# image_encoder - ResNet or Vision Transformer
# text_encoder  - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l]       - minibatch of aligned texts

# extract feature representations
I_f = image_encoder(I)  # [n, d_i]
T_f = text_encoder(T)   # [n, d_t]

# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)

# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)

# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss   = (loss_i + loss_t) / 2
```

關鍵設計選擇：

1. **對稱式交叉熵（symmetric cross entropy）**：同時對 image→text 和 text→image 兩個方向計算 loss，然後取平均。這讓兩個編碼器都學會好的 embedding。

2. **線性投影（linear projection）**：CLIP 不使用非線性 projection head（不同於 SimCLR 等自監督方法）。論文提到線性與非線性在訓練效率上沒有顯著差異，推測非線性 projection 可能與當前的自監督學習方法有特定的共適應關係。

3. **可學習的溫度參數 $t$**：控制 logits 的 scale，在訓練初期被初始化然後隨訓練調整。

4. **Large batch size（32,768）**：因為對比式學習依賴 batch 內負樣本的數量，CLIP 使用極大的 batch size 來提供足夠的 negatives。

**架構選擇**

| 元件 | 細節 |
|------|------|
| Image encoder | ResNet 系列（RN50 到 RN50x64）或 ViT（B/32, B/16, L/14, L/14-336px） |
| Text encoder | Transformer（63M params, 12 layers, 512-wide, 8 heads），與 GPT-2 架構相同 |
| 文字 tokenization | BPE（vocab size 49,152），最大序列長度 76 |
| 投影維度 | 從 512 到 1024 不等，依模型而定 |

**與 VirTex 的架構對比**

| 維度 | VirTex | CLIP |
|------|--------|------|
| 預訓練目標 | 生成式（caption prediction） | 對比式（pairing prediction） |
| 輸出 | 詞彙表上的機率分佈 | 多模態 embedding space |
| 是否預測精確用詞 | 是 | 否 |
| Text encoder | Transformer decoder（自迴歸） | Transformer encoder（雙向） |
| 從頭訓練 | 是 | 是 |

### 知識點 4：資料規模與預訓練資料建構

**VirTex 的資料**

VirTex 使用 COCO Captions 資料集：
- 118K 訓練圖片
- 每張圖片 5 個人工撰寫的 caption
- 總計約 600K 圖文對

這是高品質但小規模的資料。VirTex 論文的貢獻在於證明：**在高品質的 caption 資料上，即使資料量只有 ImageNet 的 1/10，也能學到可比較或更好的視覺表徵**。但 VirTex 的方法在資料量增大時無法有效擴展——generative objective 的訓練效率隨資料量成長而下降。

**CLIP 的 WIT 資料庫**

CLIP 為了解決資料規模問題，自行建立了 WebImageText (WIT)：
- 400M 圖文對
- 從網路上各種公開來源收集
- 查詢詞表基於 Wikipedia 高頻詞、bigram PMI、WordNet 同義詞集，共 500,000 個查詢
- 每個查詢最多 20,000 張圖片以保持類別平衡
- 總詞彙量與 GPT-2 的 WebText 相當

論文同時做了一個重要的消融實驗：將 CLIP 分別在 WIT 和 YFCC100M（過濾後約 1,500 萬張）上訓練，兩者的平均表現非常接近。這說明 CLIP 的關鍵不是特定資料集，而是**足夠大的規模**。

**資料規模對比**

| 維度 | VirTex | CLIP |
|------|--------|------|
| 資料集 | COCO Captions | WIT (WebImageText) |
| 圖文對數量 | ~590K（118K 圖 × 5 captions） | 400M |
| 規模倍數 | 1× | ~680× |
| 資料來源 | 人工標註 | 網路收集 |
| 每張圖片 caption 數 | 5（固定） | 1（稀疏） |

### 知識點 5：零樣本遷移——以自然語言作為彈性分類器介面

CLIP 最引人注目的能力是 zero-shot transfer。作法非常直觀：

1. 給定目標資料集的類別名稱（如 ImageNet 的 1,000 個類別），用 prompt template（如 "a photo of a {class}"）將每個類別轉換為文字描述
2. 用 text encoder 將這些文字描述編碼成 embedding
3. 對測試圖片用 image encoder 編碼，計算與所有類別 embedding 的餘弦相似度
4. 選相似度最高的類別作為預測結果

**這為什麼重要？**

傳統的 supervised 模型需要一個固定大小的分類頭（如 1,000 維的 softmax），改變類別就得重新訓練。CLIP 完全不需要——你只需要改變輸入的文字描述，就能即時切換到任何分類任務。

例如，要把一個 ImageNet 分類器變成 OCR 手寫數字辨識，傳統做法需要重新訓練分類頭；而 CLIP 只需要把類別文字從 "dog, cat, car..." 改成 "0, 1, 2, 3..."。

**Prompt engineering 的影響**

CLIP 論文中發現 prompt template 的選擇對 zero-shot 效能有顯著影響。簡單的 "a photo of a {class}" 在某些資料集上可行，但對於細粒度分類或非照片類型的資料集（如衛星影像），需要更特定的 prompt（如 "satellite imagery of {class}"）。CLIP 的最終評估使用多個 template 並取平均。

### 知識點 6：評估架構與分布偏移魯棒性

CLIP 在超過 30 個資料集上進行了系統性評估，覆蓋以下任務類別：

| 任務類別 | 代表性資料集 |
|---------|-------------|
| 一般分類 | ImageNet, CIFAR-10/100, STL-10 |
| 細粒度分類 | Stanford Cars, FGVC Aircraft, Oxford Pets, Flowers102 |
| OCR 相關 | MNIST, SVHN, IIIT5K, Hateful Memes |
| 動作辨識（影片） | UCF-101, Kinetics-700 |
| 地理定位 | Country211, IM2GPS |
| 分布偏移 | ImageNet-A/R/Sketch/Vid/ObjectNet |
| 場景/衛星 | SUN397, EuroSAT, RESISC45 |

**主要結果**

1. **Zero-shot ImageNet**：CLIP 的 ViT-L/14-336px 達到 76.2% top-1 accuracy，與原始 ResNet-50（全監督）的 76.5% 相當——**完全不需要使用 ImageNet 的 128 萬張訓練圖片**。

2. **分布偏移魯棒性**：這是最令人驚喜的發現。CLIP 在 ImageNet 的各種分布偏移變體（ImageNet-A, ImageNet-R, ObjectNet, ImageNet-Sketch, ImageNet-Vid, Youtube-BB）上表現大幅優於傳統 supervised 模型。例如：
   - ImageNet-A（對抗性範例）：CLIP zero-shot 77.2% vs 最佳監督模型 84.9%（但監督模型在原始 ImageNet 上 88.3%，CLIP 僅 76.2%，說明 CLIP 的相對魯棒性更好）
   - ImageNet-R（渲染/藝術圖）：CLIP zero-shot 88.9% vs 最佳監督模型 80.0%

3. **動作辨識**：CLIP 在 UCF-101 上以 linear probe 方式達到與 SOTA 相當的 92.0%，在 RareAct（罕見動作）上超越先前最佳結果 10 個百分點。

4. **檢索任務**：CLIP 在 Flickr30k 的文字檢索 R@1 達到 88.0%，接近 SOTA fine-tuned 結果。

**VirTex 的評估結果（作為對比）**

VirTex 在較小規模上的表現：

- VOC07 mAP：VirTex-100%（118K 圖）88.7 vs ImageNet-sup-100%（1.28M 圖）87.6——用 1/10 的資料超越全監督
- ImageNet linear probe：VirTex-100% 53.8 vs ImageNet-sup-10%（128K 圖）53.6——與對等資料量的監督方法相當
- COCO 物體偵測：VirTex 40.9 AP vs ImageNet-sup 41.1——差距在 0.2 AP 以內
- LVIS 分割：VirTex 23.0 AP vs ImageNet-sup 22.6——甚至略優

VirTex 證明了自然語言監督在小規模上的資料效率優勢，而 CLIP 證明了這種方法在大規模上的可擴展性。

---

## 實驗結果

### CLIP 在 27 個資料集上的 Zero-shot 表現（精選）

| 資料集 | CLIP RN50 | CLIP RN50x64 | CLIP ViT-L/14-336px | 最佳監督基線 |
|-------|-----------|--------------|--------------------|-------------|
| ImageNet | 59.6 | 73.6 | **76.2** | 88.4 (NS ENet-L2) |
| CIFAR10 | 75.6 | 86.8 | **95.7** | — |
| CIFAR100 | 41.6 | 61.3 | **77.5** | — |
| Oxford Pets | 85.4 | 93.4 | **93.5** | — |
| Stanford Cars | 55.8 | 76.0 | 78.8 | — |
| Food101 | 81.1 | 91.8 | **93.8** | — |
| SUN397 | 59.6 | 66.9 | 68.4 | — |
| Caltech101 | 82.1 | 90.6 | **92.8** | — |
| DTD | 41.7 | 53.4 | **55.7** | — |
| EuroSAT | 41.1 | 59.4 | **59.6** | — |
| Country211 | 16.1 | 29.6 | **34.9** | — |
| UCF101 | 63.6 | 74.1 | **76.9** | — |

**關鍵觀察**：

- ViT 架構普遍優於 ResNet 架構（特別是在 CIFAR 等資料集上差距明顯）
- 模型越大、解析度越高（336px vs 224px），表現越好
- 某些資料集（如 Country211、EuroSAT）仍有較大成長空間

### 分布偏移下的 Robustness

| 資料集 | NS ENet-L2 | CLIP ZS (ViT-L/14-336px) |
|-------|-----------|--------------------------|
| ImageNet | **88.3** | 76.2 |
| ImageNet-A | **84.9** | 77.2 |
| ImageNet-R | 80.0 | **88.9** |
| ObjectNet | 68.5 | **72.3** |
| ImageNet-Sketch | 47.6 | **60.2** |
| ImageNet-Vid (PM0) | 88.0 | **95.3** |
| Youtube-BB (PM0) | 67.7 | **95.2** |

CLIP 在 5/7 的分布偏移資料集上超越了最佳監督模型。這說明 zero-shot evaluation 不僅是一個方便的評估方式，更是衡量模型真正泛化能力的更好指標。

---

## 與相關工作的對比

| 維度 | CLIP | VirTex | 傳統 ImageNet 預訓練 |
|------|------|--------|--------------------|
| 監督來源 | 自然語言（400M 圖文對） | 自然語言（118K 圖文對） | 人工標註（1.28M 圖片） |
| 預訓練目標 | 對比式 | 生成式 | 分類式 |
| Zero-shot 能力 | ✅ 原生支援 | ❌（需訓練分類器） | ❌ |
| 訓練效率 | 高（12× 對比生成式） | 低 | 中 |
| 資料效率 | 依賴大規模 | ✅ 資料效率最高 | 中等 |
| 架構彈性 | 任意影像編碼器 + 文字編碼器 | ResNet + Transformer | 固定分類頭 |
| 代表性表現（ImageNet） | 76.2% zero-shot | 53.8% linear probe | 76.5% 全監督 |

---

## 我的觀察

CLIP 的影響深遠，它不只是某一篇論文，而是開創了整個 vision-language pre-training 的典範轉移。我認為有幾個值得注意的點：

1. **簡單就是力量**：CLIP 的方法在概念上極為簡單——contrastive learning + dual encoder + large batch。這種 simplicity 正是其影響力最大的原因之一，也解釋了為什麼後續工作（ALIGN、Florence、SigLIP 等）都沿著同一條路線前進。

2. **從「學到好的表徵」到「學到可查詢的表徵」**：傳統視覺預訓練的目標是讓下游任務有個好的初始化點；CLIP 產出的表徵天然與語言對齊，從根本上改變了「什麼是好的視覺表徵」的定義。

3. **魯棒性的驚喜**：CLIP 在分布偏移下的優異表現是當時的意外發現。這可能來自於自然語言監督天然的多樣性——訓練資料涵蓋了藝術創作、雜誌掃描、新聞圖片等，讓模型學到了更通用的視覺概念。

4. **規模不是萬能**：VirTex 用 1/10 的資料量就達到了可比較的結果，說明了資料品質的重要性。CLIP 的成功部分來自於規模，但更來自於正確的架構選擇（對比式勝過生成式）。

---

## 延伸閱讀

### Dependency Papers（本文涵蓋）

1. **VirTex: Learning Visual Representations from Textual Annotations** ([2006.06666](https://arxiv.org/abs/2006.06666))
   - 與本文關係：生成式預訓練的代表作，CLIP 的初始靈感來源與效率對比基準。VirTex 以 image captioning 從 COCO Captions（118K 張圖）學習視覺表徵，證明在小規模上 caption-based pretraining 的資料效率優於 ImageNet 監督式預訓練。

### 後續發展（未涵蓋，僅列出）

- **ALIGN** (Jia et al., 2021, [2102.05918](https://arxiv.org/abs/2102.05918))：Google 的對比式圖文預訓練，使用 1.8B 嘈雜圖文對，進一步驗證了 CLIP 路線的可擴展性
- **OpenCLIP** (Ilharco et al., 2021)：開源復現 CLIP 訓練流程，提供了完整可追溯的 open-source 模型
- **SigLIP** (Zhai et al., 2023, [2303.15343](https://arxiv.org/abs/2303.15343))：用 sigmoid loss 取代 softmax，擺脫對大 batch size 的依賴
- **BLIP/BLIP-2** (Li et al., 2022/2023)：將 CLIP 式的對比學習與生成式 captioning 結合，形成統一的 vision-language 理解與生成框架

---

## 引用

完整 BibTeX 見 [`papers.bib`](./papers.bib)。
