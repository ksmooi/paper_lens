# RoPE（Rotary Position Embedding）論文導讀

> **種子論文**: Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
> **依賴論文**: Shaw et al. (2018). Self-Attention with Relative Position Representations. NAACL 2018. arXiv:1803.02155.

**閱讀時間**: 繁體中文，全文約 700 行。適合對 Transformer 有基本了解的讀者，建議預留 30–45 分鐘專注閱讀。

---

## TL;DR

- **核心問題**：Transformer 的 self-attention 是 position-agnostic 的，沒有位置資訊就無法區分 token 的順序。傳統解法將位置編碼「加」到 token embedding 或 attention logits 上，但這種加法模式限制了相對位置的建模能力，也無法與線性注意力（linear attention）相容。
- **RoPE 的解法**：改用「乘法」——將 query 與 key 向量在 2D 子空間中旋轉，使注意力內積自然依賴於相對位置。這不僅數學優美，還帶來了序列長度彈性、長距離衰減、以及與線性注意力的相容性。
- **效果**：WMT14 英德翻譯 BLEU 27.5、GLUE 3/6 任務勝過 BERT、中文長文本分類準確率 69.79%（vs WoBERT 68.10%），且與線性注意力（Performer）結合後收斂更快。後續 LLaMA、Mistral、Gemma 等主流開源模型均採用 RoPE，使其成為事實上的標準位置編碼方案。

---

## 為什麼要讀這篇論文？

RoPE 不是一篇典型的「屠榜」論文——它在機器翻譯上的 BLEU 提升僅 0.2 點，GLUE 上也不是全面壓制 BERT。但它的影響力遠遠超越了這些基準測試分數。

如果你正在閱讀這篇文章，可能是因為你是一位深度學習研究人員或工程師，在日常工作中遇到了與位置編碼相關的問題。也許你在調整 LLaMA 的 RoPE 頻率時一頭霧水，也許你在尋找一個能與 FlashAttention 或線性注意力相容的位置編碼方案，又或者你只是單純好奇：「為什麼現在所有模型都在用 RoPE？」

無論你是哪種情況，這篇文章的目標都是讓你在讀完後，能夠清楚地回答以下三個問題：

1. **RoPE 是什麼？** ——旋轉位置編碼的數學原理與直覺
2. **為什麼是 RoPE？** ——相比 Shaw、Transformer-XL、T5 等加法式方法，RoPE 的乘法式設計帶來了哪些具體優勢
3. **RoPE 的成功說明了什麼？** ——從理論驅動的架構設計角度，RoPE 給我們帶來了哪些啟發

首先，**RoPE 解決了一個深層的架構設計問題**：位置編碼到底是加法還是乘法？加法（Shaw、Transformer-XL、T5）是一種工程啟發，乘法（RoPE）則源於數學約束的推導。這種從「我們試試加個偏置」到「讓內積自然依賴於相對位置」的轉變，展示了理論驅動的研究如何帶來更優雅的解決方案。

其次，**RoPE 的影響是架構級的**。它不是一個只能在特定任務上湊效的技巧，而是一個可以嵌入任何 Transformer 變體的通用組件。從 LLaMA 到 Mistral 再到 Gemma，幾乎所有後續的開源 LLM 都選擇了 RoPE 作為標準位置編碼——這在快速變動的 AI 領域是極高的認可。

最後，**RoPE 與線性注意力的相容性是具有前瞻性的設計**。隨著序列長度從 512 增長到 128K、2M，O(N²) 的標準注意力越來越難以承受。RoPE 是目前少數能與 O(N) 線性注意力無縫結合的相對位置編碼方法——這個來自 2021 年的設計選擇，在長上下文時代被證明具有遠見。

---

## 背景與動機

### 序列建模中的位置編碼問題

在自然語言處理中，詞序（word order）承載了大量語法與語義資訊。「狗追貓」與「貓追狗」包含完全相同的詞彙，但意義截然不同。任何有效的語言模型都必須能夠編碼元素的順序。

傳統的序列模型使用不同的機制來編碼位置：

- **遞迴神經網路（RNN）**：透過隱藏狀態的循環更新，天然地沿時間維度編碼位置資訊。h_t = f(h_{t-1}, x_t) 中的 h_t 包含了從 t=1 到 t 的所有歷史位置資訊。但 RNN 有梯度消失/爆炸問題，且無法平行化計算。
- **卷積神經網路（CNN）**：在卷積核大小內可以捕捉局部相對位置，但超出 kernel size 的長距離依賴需要多層堆疊。CNN 理論上是 position-agnostic 的（參數共享），但 Islam et al.（2020）發現 padding 操作可以隱式傳遞位置資訊。
- **Transformer**：完全依賴 self-attention 機制，沒有循環或卷積結構。這帶來了極佳的平行化能力和長距離建模能力，但也帶來了根本性的「**位置困境**」。

### Transformer 的位置困境

Transformer（Vaswani et al., 2017）的核心——self-attention——有一個廣為討論的特性：它是 **position-agnostic** 的。這意味著如果不做任何處理，模型看到「我愛你」和「你愛我」的內部表示是完全無法區分的。Yun et al.（2020）從理論上證明了這一點：self-attention 層本身是 permutation-invariant 的（對序列順序不變）。

這個問題的根源在於 self-attention 的運算方式。給定 token 序列 {x₁, x₂, ..., x_N}（x_i ∈ R^d 是詞嵌入向量），標準的 scaled dot-product attention 為：

$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V $$

其中 Q = XW^Q, K = XW^K, V = XW^V。這裡 X ∈ R^{N×d} 是詞嵌入矩陣，**不包含任何位置資訊**。假設我們交換 X 中第 i 行和第 j 行——相當於交換序列中第 i 和第 j 個 token——那麼 Q、K、V 的對應行也會交換，最終輸出矩陣只是行序交換，每個位置的輸出向量本身不受影響。這就是「position-agnostic」的數學本質。

為了解決這個問題，必須在 attention 計算的某個環節注入位置資訊。形式化地說，需要設計三個函數 f_q、f_k、f_v，使得：

$$ q_m = f_q(x_m, m), \quad k_n = f_k(x_n, n), \quad v_n = f_v(x_n, n) $$

不同位置編碼方法的核心差異，就在於這三個函數的選擇。

### 第一條路：絕對位置編碼（Absolute Position Encoding）

這是最直觀的做法。給每個位置 i 分配一個位置向量 p_i，然後加到詞嵌入上：

$$ f_{t:\{q,k,v\}}(x_i, i) := W_{t:\{q,k,v\}}(x_i + p_i) $$

p_i 的選擇有兩種主流做法：

**正弦函數編碼（Sinusoidal, Vaswani 2017）**：

$$ p_{i,2t} = \sin(i / 10000^{2t/d}), \quad p_{i,2t+1} = \cos(i / 10000^{2t/d}) $$

Vaswani 的原始論點是：這種預定義的 sinusoidal 模式可以讓模型透過線性變換學到任意位置的編碼（因為 sin(α+β) 是 sin α 與 cos α 的線性組合），從而隱式捕捉相對位置。

**學習式編碼（Learned, BERT/Devlin 2019）**：

BERT 讓位置編碼成爲可學習的參數矩陣 P ∈ R^{L×d}，其中 L 是最大序列長度（BERT-base 為 512）。在預訓練過程中，P 透過反向傳播與模型其他參數一同更新。

**絕對位置編碼的固有問題**：
1. **長度限制**：訓練時的最大長度 L 限制了推理時能處理的序列長度。超過 L 的位置沒有對應的編碼
2. **外推困難**：對於學習式編碼，超過 L 的位置完全無法處理；對於 sinusoidal 編碼，雖然可以計算任意位置的編碼，但模型在訓練中從未見過這些取值範圍，實際效果不佳
3. **位置與內容耦合**：位置向量加在內容表示上後，在經過線性投影和 attention 內積時，位置項與內容項會交叉相乘，難以獨立控制

### 第二條路：相對位置編碼（Relative Position Encoding）

Shaw et al.（2018）開創了另一個方向：與其告訴模型「token i 在第 i 個位置」，不如告訴它「token i 和 token j 相距 j-i 個位置」。

這個思路更符合直覺——在自然語言中，真正影響語義的是詞與詞之間的相對距離（例如「這個」與「名詞」通常相隔 0-2 個位置），而不是某個詞恰好出現在第 17 個位置。

**相對位置編碼的數學共性**：
所有加法式相對位置編碼方法都可以歸結為將 attention 的 q·k 內積分解為四項：

$$ q_m \cdot k_n = \underbrace{x_m W^q W^k x_n}_{\text{content-content}} + \underbrace{x_m W^q W^k p_n}_{\text{content-position}} + \underbrace{p_m W^q W^k x_n}_{\text{position-content}} + \underbrace{p_m W^q W^k p_n}_{\text{position-position}} $$

不同方法的核心差異在於如何處理 p_m、p_n。Shaw（2018）用可學習的相對位置向量取代 p_n 並裁剪距離；Transformer-XL（Dai et al., 2019）區分內容與位置的投影矩陣 W^k 與 \tilde{W}^k；T5（Raffel et al., 2020）簡化為純標量偏置 b_{i,j}；DeBERTa（He et al., 2020）將位置項拆分為兩項分別建模。

這些方法雖然有效，但它們都共享一個根本限制：**位置資訊是透過加法注入的**，因此在注意力內積的公式中，內容項與位置項可以被分離但無法深度融合。更重要的是，這種加法模式在面對線性注意力（需要點積的 kernel 分解）時完全失效。

RoPE 正是在這個脈絡下提出了一個本質性的轉向。

---

## 核心知識點

### 知識點 1：Shaw et al.（2018）——加法式相對位置編碼的奠基

Peter Shaw、Jakob Uszkoreit、Ashish Vaswani 在 NAACL 2018 上發表了本篇論文，首次系統性地將相對位置資訊引入 Transformer 的 self-attention。

**核心想法**：不把位置資訊加到輸入 embedding 上，而是為每個注意力 head 學習「當 query 位置 i 與 key 位置 j 的相對距離為 j-i 時，應該在相容性函數上添加多少偏置」。

具體來說，Shaw 修改了標準 self-attention 的兩個關鍵公式。

相容性函數（決定 attention 權重 α_{ij}）：

$$ e_{ij} = \frac{x_i W^Q (x_j W^K + a^K_{ij})^T}{\sqrt{d_z}} $$

值聚合（決定輸出 z_i）：

$$ z_i = \sum_{j=1}^n \alpha_{ij} (x_j W^V + a^V_{ij}) $$

其中 a^K_{ij}, a^V_{ij} ∈ R^{d_a} 是對應於相對距離 j-i 的可學習向量：

$$ a^K_{ij} = w^K_{\text{clip}(j-i,k)}, \quad a^V_{ij} = w^V_{\text{clip}(j-i,k)} $$

這裡的 clip(x,k) = max(-k, min(k,x)) 是一個關鍵設計。對於 k=16 的情況，總共會學習 2k+1 = 33 個不同的向量（表示從「左邊 16 個位置」到「右邊 16 個位置」）。每個向量與注意力頭數無關——所有頭共享同一組相對位置表示，顯著減少了參數量。

**為什麼要裁剪距離（clipping）？**Shaw 的假設是：精確的相對位置資訊在超過一定距離後就不再有用。他們的消融實驗（下表）證實了這個假設：

| k (裁剪距離) | EN-DE BLEU |
|-------------|-----------|
| 0 | 12.5 |
| 1 | 25.5 |
| 2 | 25.8 |
| 4 | 25.9 |
| 16 | 25.8 |
| 64 | 25.9 |
| 256 | 25.8 |

k=0（無位置資訊）的 BLEU 僅 12.5，這與 Transformer 在沒有位置編碼時的表現一致，說明位置資訊對於序列建模至關重要。但令人驚訝的是，k=2 到 k=256 的 BLEU 幾乎完全相同（25.8-25.9）。這意味著兩層級的精確位置資訊（左鄰、右鄰）就足以讓深度網路透過多層編碼器傳播更遠的相對位置資訊——線性層和殘差連接可以組合低層的局部資訊來隱式編碼更遠的關係。

**高效率實現**：Shaw 的一個重要貢獻是展示了相對位置編碼可以透過矩陣運算高效實現。關鍵是將 e_{ij} 拆分為兩項：

$$ e_{ij} = \underbrace{x_i W^Q W^K x_j^T}_{\text{Term 1: standard attention}} + \underbrace{x_i W^Q (a^K_{ij})^T}_{\text{Term 2: position bias}} $$

Term 1 可用標準的批次矩陣乘法一次計算所有位置對（O(bh n² d_z)）。Term 2 可透過 tensor reshaping 實現：對每個 query 位置 i 獨立計算它與所有相對距離的相容性，然後根據 j-i 對應到正確的 key 位置。最終僅增加約 7% 的計算開銷。

**消融實驗**：

| a^V_{ij} | a^K_{ij} | EN-DE BLEU |
|---------|---------|-----------|
| Yes | Yes | 25.8 |
| No | Yes | 25.8 |
| Yes | No | 25.3 |
| No | No | 12.5 |

從消融結果來看，key 側的相對位置表示（a^K_{ij}）是主要的性能來源——沒有它但保留 value 側（a^V_{ij}）時 BLEU 從 25.8 降到 25.3；只有 key 側沒有 value 側時 BLEU 保持不變。這暗示對多數任務來說，在決定「應該關注哪裡」時需要位置資訊，但在聚合內容時位置資訊的用處較小。

**實驗結果（完整表格）**：

| 模型 | 位置資訊 | EN-DE BLEU | EN-FR BLEU |
|------|---------|-----------|-----------|
| Transformer (base) | Absolute (sinusoidal) | 26.5 | 38.2 |
| Transformer (base) | Relative (Shaw) | 26.8 | 38.7 |
| Transformer (big) | Absolute (sinusoidal) | 27.9 | 41.2 |
| Transformer (big) | Relative (Shaw) | **29.2** | **41.5** |

Shaw 方法的出色之處在於它的**簡潔與高效**。僅需學習 2×(2k+1)×d_a ≈ 2×33×64 = 4224 個額外參數（對每個 attention head 獨立），就能在 big 模型設定下將 EN-DE BLEU 從 27.9 提升到 29.2——1.3 個點的提升在機器翻譯領域是顯著的，遠超過許多複雜的模型修改方法。

更重要的是，Shaw 的方法為後續所有相對位置編碼研究建立了標準框架：位置資訊應該作用於 attention 的相容性函數本身，而非僅作為輸入的添加項。這個框架影響了 Transformer-XL、T5、DeBERTa，乃至 RoPE 的設計思路。

**Shaw 方法的限制**：
1. **計算限制**：雖然實現高效，但空間複雜度從 O(bhnd_z) 增至 O(bhnd_z + n²d_a)，對於超長序列（n > 4096）依然是個負擔
2. **不對稱資訊流**：位置資訊只在 key 和 value 側編碼，query 側完全沒有位置資訊（f_q(x_m) := W^q x_m 不含位置項）
3. **線性注意力不相容**：位置偏置是加在 attention logits 上的標量（或向量點積），而線性注意力透過 φ(q)·Σ[φ(k)v] 避免計算完整的 n×n 注意力矩陣——加法式偏置無法融入這種分解
4. **缺乏理論統一性**：方法本質上是啟發式的——「我們試著把位置偏置加在這裡，結果有效」。沒有從基本原則推導出為什麼這是合理的

### 知識點 2：RoPE 的問題設定——相對位置的閉式條件

RoPE 的出發點是一個優雅的數學條件。Su et al.（2021）從一個抽象問題開始：「我們能不能設計 f_q 和 f_k，使得 query q_m 與 key k_n 的內積**只取決於**詞嵌入 x_m、x_n 和它們的**相對位置** m-n？」

$$ \langle f_q(x_m, m), f_k(x_n, n) \rangle = g(x_m, x_n, m - n) $$

這個條件（論文中式 11）是整個 RoPE 方法的核心形式化約束。它包含幾個重要意涵：

1. **平移不變性**：整個序列向右移動一個位置，任意兩個 token 的相對位置保持不變，attention 權重也應保持不變。這與自然語言的直覺一致——「the cat sits」中 the 與 cat 的關係，不應因為句子是「A the cat sits」還是「the cat sits on」而改變
2. **初始條件**：當 m = n = 0 時──即不考慮位置資訊──應退化為標準的線性變換 f_q(x_m, 0) = W^q x_m, f_k(x_n, 0) = W^k x_n
3. **沒有自由參數**：條件本身對 f_q、f_k 的形式施加了限制——不是任何函數都可以滿足

這個形式化的力量在於：它把位置編碼問題從「設計一個看起來合理的函數」轉變為「在給定約束下求解一個方程」。這在深度學習研究中是少見的「由第一性原理推導」的方法。

### 知識點 3：RoPE 在 2D 情形的完整推導

為了解決上述條件，RoPE 從 d=2 的最簡單情形開始。利用複數的幾何性質，將二維向量映射到複數平面。令 x_m = (x^{(1)}_m, x^{(2)}_m) ∈ R²，其複數表示為 z_m = x^{(1)}_m + ix^{(2)}_m。

將 f_q、f_k 用複數的極坐標形式表示：

$$ f_q(x_m, m) = R_q(x_m, m) e^{i\theta_q(x_m, m)} $$
$$ f_k(x_n, n) = R_k(x_n, n) e^{i\theta_k(x_n, n)} $$

在複數空間中，二維向量的內積對應於 Re(z₁ · \bar{z₂})。代入條件式：

$$ \langle f_q(x_m, m), f_k(x_n, n) \rangle = \text{Re}[R_q R_k e^{i(\theta_k - \theta_q)}] = R_g(x_m, x_n, n-m) e^{i\theta_g(x_m, x_n, n-m)} $$

這給出了關於徑向分量和角分量的兩個獨立條件：

$$ R_q(x_m, m) R_k(x_n, n) = R_g(x_m, x_n, n-m) \tag{A} $$
$$ \theta_k(x_n, n) - \theta_q(x_m, m) = \theta_g(x_m, x_n, n-m) \tag{B} $$

**步驟 1：求解徑向分量**。設 m = n，並利用初始條件：

$$ R_q(x_m, m) R_k(x_m, m) = R_g(x_m, x_m, 0) = R_q(x_m, 0) R_k(x_m, 0) = \|W^q x_m\| \cdot \|W^k x_m\| $$

這意味著徑向函數與位置 m 無關：

$$ R_q(x_m, m) = R_q(x_m, 0), \quad R_k(x_n, n) = R_k(x_n, 0) $$

徑向分量等於線性變換後的向量範數——它編碼的是內容資訊，而非位置資訊。

**步驟 2：求解角分量**。從條件 (B) 並設 m = n：

$$ \theta_k(x_m, m) - \theta_q(x_m, m) = \theta_g(x_m, x_m, 0) = \theta_k(x_m, 0) - \theta_q(x_m, 0) $$

這表示 θ_k(x_m, m) - θ_q(x_m, m) 與 m 無關。定義 Δ(x) = θ_k(x, m) - θ_q(x, m)，則 Δ(x) 只取決於 x 本身。

由於 f_q 和 f_k 是對稱的（query 側和 key 側的角色在注意力中是可交換的），一個自然的選擇是讓兩者相差一個與位置無關的常數（可吸收進初始相位中）。設：

$$ \theta_q(x_m, m) = \phi(x_m) + \theta(m), \quad \theta_k(x_n, n) = \phi(x_n) + \theta(n) $$

**步驟 3：證明角度是等差數列**。從條件 (B)：

$$ \theta_k(x_n, n) - \theta_q(x_m, m) = [\phi(x_n) + \theta(n)] - [\phi(x_m) + \theta(m)] $$

要讓這個值只取決於 n-m 而與 x_m, x_n 獨立，唯一的方式是 φ(x) = 常數（可設為 0，因為線性變換 W^q, W^k 已經包含了學習 x 表示的足夠自由度）。於是：

$$ \theta(n) - \theta(m) = \theta_g(n-m) $$

設 n = m+1：

$$ \theta(m+1) - \theta(m) = \theta_g(1) $$

由於右側是與 m 無關的常數，θ(m) 必須是等差數列：

$$ \theta(m) = m\theta + \theta_0 $$

設初始相位 θ_0 = 0（可吸收進 W^q, W^k 的旋轉中），得最優美解：

$$ f_q(x_m, m) = (W^q x_m) e^{i m\theta}, \quad f_k(x_n, n) = (W^k x_n) e^{i n\theta} $$

**步驟 4：驗證**。回到原始條件：

$$ \langle q_m, k_n \rangle = \text{Re}[(W^q x_m) e^{i m\theta} \cdot \overline{(W^k x_n)} e^{-i n\theta}] = \text{Re}[(W^q x_m) \overline{(W^k x_n)} e^{i (m-n)\theta}] $$

確實只取決於相對位置 (m-n)！證畢。

將複數形式轉回向量形式：

$$ f_{q,k}(x_m, m) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} W^{q,k} x_m $$

這就是「旋轉位置編碼」名稱的由來——對 query/key 向量施加一個角度為 mθ 的旋轉變換。

### 知識點 4：RoPE 的一般形式與高效計算

對於一般情形（x_i ∈ R^d, d 為偶數），RoPE 將 d 維空間分成 d/2 個 2D 子空間，每個子空間使用不同的旋轉頻率：

$$ f_{q,k}(x_m, m) = R_{\Theta,m} W^{q,k} x_m $$

其中 R_{\Theta,m} ∈ R^{d×d} 是 block-diagonal 旋轉矩陣：

$$ R_{\Theta,m} = \begin{pmatrix}
\cos m\theta_1 & -\sin m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
\sin m\theta_1 & \cos m\theta_1 & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos m\theta_2 & -\sin m\theta_2 & \cdots & 0 & 0 \\
0 & 0 & \sin m\theta_2 & \cos m\theta_2 & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos m\theta_{d/2} & -\sin m\theta_{d/2} \\
0 & 0 & 0 & 0 & \cdots & \sin m\theta_{d/2} & \cos m\theta_{d/2}
\end{pmatrix} $$

頻率設定沿用 sinusoidal PE 的多頻率方案（Vaswani 2017），確保不同維度編碼不同粒度（scope）的位置資訊：

$$ \theta_i = 10000^{-2(i-1)/d}, \quad i \in [1, 2, ..., d/2] $$

**為什麼使用多頻率？**低維度（i 小、θ_i 大）的旋轉速度快，編碼細粒度的短距離位置關係；高維度（i 大、θ_i 小）的旋轉速度慢，編碼粗粒度的長距離位置關係。這種分層設計讓模型可以同時捕捉從鄰近詞到跨段落的多尺度位置依賴。

**高效計算的實現技巧**：

由於 R_{\Theta,m} 是稀疏矩陣（只有 2d 個非零元素），直接做 d×d 矩陣乘法是浪費的。論文給出了一種更高效的計算方式——對每個 2D 子空間獨立做旋轉：

$$ R_{\Theta,m} x = \begin{pmatrix}
x_1 \cos m\theta_1 - x_2 \sin m\theta_1 \\
x_2 \cos m\theta_1 + x_1 \sin m\theta_1 \\
x_3 \cos m\theta_2 - x_4 \sin m\theta_2 \\
x_4 \cos m\theta_2 + x_3 \sin m\theta_2 \\
\vdots \\
x_{d-1} \cos m\theta_{d/2} - x_d \sin m\theta_{d/2} \\
x_d \cos m\theta_{d/2} + x_{d-1} \sin m\theta_{d/2}
\end{pmatrix} $$

這只需要 O(d) 的乘加運算，與原本的線性投影 W^{q,k}x_m 的 O(d²) 相比可忽略不計。由於 W^{q,k}x_m 的計算依然保留，RoPE 的實際計算開銷幾乎可以忽略（約 1-3% 的額外 overhead）。

最終 attention 內積為：

$$ q_m \cdot k_n = (R_{\Theta,m} W^q x_m) \cdot (R_{\Theta,n} W^k x_n) = x_m^T W^q R_{\Theta,n-m} W^k x_n $$

關鍵在於 R_{\Theta,n-m} = R_{\Theta,m}^T R_{\Theta,n}──旋轉矩陣的相乘等於角度相加，因此相鄰位置的旋轉效果會相互抵消，只剩下相對角度的差值。這正是「絕對位置編碼 + 自然呈現相對位置依賴」的具體數學含義。

```mermaid
---
config:
  theme: neutral
---
block-beta
  columns 4
  block:q["Query"]:2
    columns 2
    q1["q₁"] q2["q₂"]
  end
  block:k["Key"]:2
    columns 2
    k1["k₁"] k2["k₂"]
  end
  space
  block:mq["旋轉後 Query"]:2
    columns 2
    mq1["q₁cos(mθ)-q₂sin(mθ)"] mq2["q₂cos(mθ)+q₁sin(mθ)"]
  end
  block:mk["旋轉後 Key"]:2
    columns 2
    mk1["k₁cos(nθ)-k₂sin(nθ)"] mk2["k₂cos(nθ)+k₁sin(nθ)"]
  end
  space
  block:dot["內積結果"]:4
    dot1["q·k = (q₁k₁+q₂k₂)cos((m-n)θ) + (q₂k₁-q₁k₂)sin((m-n)θ)"]
  end
  q --> mq
  k --> mk
  mq --> dot
  mk --> dot
end
```

### 知識點 5：長距離衰減（Long-term Decay）

使用 Vaswani 的頻率調度 θ_i = 10000^{-2i/d} 並非任意選擇。論文利用 Abel transformation（分部求和的離散版本）證明了：在此調度下，內積的長期平均值隨相對距離增加而衰減。

數學上，將 2D 子空間的貢獻寫成複數形式：

$$ \langle q_m, k_n \rangle = \text{Re}\left[\sum_{i=0}^{d/2-1} h_i e^{i(m-n)\theta_i}\right] $$

其中 h_i = q_{[2i:2i+1]} · k_{[2i:2i+1]} 是第 i 個子空間中內容向量的未旋轉點積（實數）。使用 Abel transformation：

$$ \sum_{i=0}^{d/2-1} h_i S_i = -\sum_{i=0}^{d/2-2} S_{i+1}(h_{i+1} - h_i) $$

其中 S_i = \sum_{j=0}^{i-1} e^{i(m-n)\theta_j} 是部分和。取絕對值後：

$$ \left|\sum_{i=0}^{d/2-1} h_i S_i\right| \leq \sum_{i=0}^{d/2-2} |S_{i+1}| \cdot |h_{i+1} - h_i| \leq \max_i |h_{i+1} - h_i| \cdot \sum_{i=0}^{d/2-2} |S_{i+1}| $$

關鍵觀察：由於 θ_i = 10000^{-2i/d} 隨 i 迅速遞減（對典型 d=768，θ_1 ≈ 1, θ_{384} ≈ 0.0001），|S_i| 的平均值確實隨 (m-n) 增大而衰減。高頻子空間（小 i、大 θ_i）的 S_i 增長慢但波動大；低頻子空間（大 i、小 θ_i）的 S_i 增長快但波動小。整體效果近似於在 (m-n) 上的指數或 1/(m-n) 衰減。

這與自然語言的直覺一致：相隔很遠的兩個詞通常沒有直接的語法或語義依賴關係，attention 權重理應較低。

```mermaid
---
config:
  theme: neutral
---
xyChart
    x-axis "相對距離" [0, 50, 100, 150, 200, 250]
    y-axis "衰減上界" 0 --> 20
    line "上界值" [20, 15, 12, 10, 8, 7]
```

### 知識點 6：與線性注意力的相容性——與加法式方法的關鍵分歧

這可能是 RoPE 最具實用價值的性質，也是它相對於所有加法式方法（Shaw、Transformer-XL、T5、DeBERTa）的根本性優勢。

**線性注意力的動機**。標準 softmax attention 的計算複雜度為 O(N²)（N 為序列長度），這對於長序列（如文件級別的 4096+ tokens）是主要瓶頸。線性注意力（Katharopoulos et al., 2020）透過以下變換將複雜度降至 O(N)：

$$ \text{Attention}(Q,K,V) = \frac{\sum_{n=1}^N \phi(q_m) (\phi(k_n)^T v_n)}{\sum_{n=1}^N \phi(q_m) \phi(k_n)^T} $$

其中 φ 是非負函數（如 ELU(x)+1）。關鍵在於計算順序：先計算 Σ_n φ(k_n) v_n^T（O(Nd²)），再對每個 query 計算 φ(q_m) · (這個聚合)（O(Nd²)），避免了計算完整的 n×n 注意力矩陣。

**為什麼加法式相對編碼無法融入**。Shaw 的相容性函數是：

$$ e_{ij} = \underbrace{x_i W^Q W^K x_j^T}_{\text{content}} + \underbrace{x_i W^Q a^K_{ij}}_{\text{position}} $$

第二項 x_i W^Q a^K_{ij} 對每個 (i,j) 對都是不同的（a^K_{ij} 取決於 j-i）。在線性注意力框架中，attention 權重需要分解為 φ(q_i)·φ(k_j) 的形式，而加法式偏置無法被分解為 query-only 和 key-only 的函數乘積。

**為什麼 RoPE 可以**。RoPE 的注意力內積是：

$$ q_m \cdot k_n = (R_{\Theta,m} W^q x_m) \cdot (R_{\Theta,n} W^k x_n) $$

這自然可以分解為 (R_{\Theta,m} W^q x_m) · (R_{\Theta,n} W^k x_n) = f̃_q(x_m, m) · f̃_k(x_n, n) 的形式——剛好滿足線性注意力要求的 factorized form。將 RoPE 與線性注意力結合：

$$ \text{Attention}_m = \frac{\sum_{n=1}^N [R_{\Theta,m} \phi(W^q x_m)] \cdot [R_{\Theta,n} \phi(W^k x_n)] \cdot (W^v x_n)}{\sum_{n=1}^N \phi(W^q x_m) \cdot \phi(W^k x_n)} $$

注意分母中沒有旋轉矩陣（因為分母是歸一化常數，不需要位置資訊），這保持了數值穩定性（避免除以零的風險）。

**實驗驗證**：論文讓 Performer（一種線性注意力變體）在有/無 RoPE 的情況下在 Enwik8 上預訓練。結果顯示 RoPE 使 Performer 的 loss 下降顯著加快——這不僅驗證了相容性，還證明 RoPE 對線性注意力的位置建模有實質性幫助。

### 知識點 7：RoPE 的實驗結果全貌

#### 4.1 機器翻譯

RoFormer 在 WMT14 英德翻譯上以標準 Transformer base 為 baseline（27.3 BLEU），僅替換位置編碼層為 RoPE，取得 27.5 BLEU。0.2 BLEU 的提升雖然不大，但考慮到 RoPE 沒有引入任何額外參數（連 sinusoidal PE 向量都不用學），這個結果是有意義的——它說明了單靠旋轉編碼的建模能力就能超過精心設計的 sinusoidal 絕對編碼。

#### 4.2 預訓練語言建模

在 BookCorpus + Wikipedia 上預訓練 BERT 與 RoFormer（均 100k steps, batch size 64, max seq len 512），RoFormer 的 MLM loss 從第一步開始就低於 BERT，且差距持續到訓練結束。這說明了 RoPE 提供了更有效的位置監督信號，幫助模型更快地學習詞序依賴。

#### 4.3 GLUE 下游任務

| 模型 | MRPC (F1) | SST-2 (Acc) | QNLI (Acc) | STS-B (Spearman) | QQP (F1) | MNLI-m (Acc) | MNLI-mm (Acc) |
|------|-----------|-------------|------------|-------------------|----------|--------------|---------------|
| BERT (Devlin 2019) | 88.9 | 93.5 | 90.5 | 90.7 | 88.0 | 84.6 | 83.4 |
| RoFormer | **89.5** | 93.5 | **91.2** | 90.4 | 87.0 | 80.2 | 79.8 |

RoFormer 在 MRPC（句子配對，F1 89.5）、QNLI（問答自然語言推論，Acc 91.2）上勝過 BERT；在 STS-B（語義相似度）上相當。但在 QQP（問題配對，F1 87.0 vs 88.0）和 MNLI（自然語言推論，80.2/79.8 vs 84.6/83.4）上表現較差。這種不一致可能來自於兩個因素：(1) RoFormer 只預訓練了 100k steps（BERT 是 ~1M steps）；(2) GLUE 的資料集大小差異大——RoPE 在中小型資料集上的優勢更明顯。

#### 4.4 中文長文本（CAIL2019-SCM）

這是最具說服力的實驗之一。CAIL2019-SCM 是一個法律案件相似度比對任務，包含 8964 個三元組（A, B, C）——每個案例的描述經常超過 512 字。由於 RoPE 不需要學習固定長度的位置嵌入，它可以自然地處理比訓練時更長的輸入：

| 模型 | Validation Acc | Test Acc |
|------|---------------|----------|
| BERT-512 | 64.13% | 67.77% |
| WoBERT-512 | 64.07% | 68.10% |
| RoFormer-512 | 64.13% | 68.29% |
| RoFormer-1024 | **66.07%** | **69.79%** |

在長度 512 時，三模型表現接近（67.77-68.29%）。但當 RoFormer 的輸入長度增加到 1024 時（**不需要重新訓練**，只改變推理時的序列長度），準確率提升了約 1.5%（絕對值）。這直接驗證了 RoPE 的序列長度彈性——學習式絕對位置編碼（BERT、WoBERT）無法做到這點，因為它們的 P ∈ R^{512×d} 在推理時無法處理位置 513 以後的編碼。

#### 4.5 多階段預訓練策略

RoFormer 在中文資料集上使用了多階段預訓練策略：

| Stage | Max seq len | Batch size | Steps | Loss | Accuracy |
|-------|-------------|------------|-------|------|----------|
| 1 | 512 | 256 | 200k | 1.73 | 65.0% |
| 2 | 1536 | 256 | 12.5k | 1.61 | 66.8% |
| 3 | 256 | 256 | 120k | 1.75 | 64.6% |
| 4 | 128 | 512 | 80k | 1.83 | 63.4% |
| 5 | 1536 | 256 | 10k | 1.58 | 67.4% |
| 6 | 512 | 512 | 30k | 1.66 | 66.2% |

這個策略有趣的點在於：Stage 2 和 Stage 5（長序列）的準確率最高（66.8%、67.4%），而 Stage 4（最短序列 128）的準確率最低（63.4%）。Stage 5 在 Stage 4 之後僅用 10k 步就恢復到最高準確率，說明模型在長序列上學到的位置表示具有強可遷移性——這是因為 RoPE 的旋轉矩陣是位置無關的，長序列訓練只是讓模型有機會在更長的 context window 中練習位置依賴。

從訓練策略的角度來看，RoFormer 的實驗設計體現了一個重要的實務洞察：在預訓練過程中**交替使用不同序列長度**，可以幫助模型在保持泛化能力的同時針對性地訓練長上下文處理能力。Stage 2 直接用 1536 長度訓練（遠超過 BERT 的 512 上限），Stage 3 又回到 256 聚焦基礎語言能力，Stage 4 用 128 強化短文本密集訊息編碼——這種「先長後短」的模式與人類學習語言時「先聽長句培養語感，再精練短句」的直覺是一致的。

### 4.6 RoPE 的 PyTorch 實作範例

在實際應用中，RoPE 的實現非常簡潔。以下是一個標準的 PyTorch 實現：

```python
import torch
import torch.nn as nn

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        # 計算每個 2D 子空間的頻率 θ_i
        # dim 必須為偶數
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # 預計算所有位置的 cos 與 sin 值
        # shape: (max_seq_len, dim/2)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)  # (max_seq_len, dim/2)
        # 對每個頻率儲存 (cos, sin) 各一份
        # shape: (max_seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, seq_len: int):
        # x: (batch, seq_len, num_heads, dim)
        cos = self.cos_cached[:seq_len].view(1, seq_len, 1, -1)
        sin = self.sin_cached[:seq_len].view(1, seq_len, 1, -1)
        return self._apply_rotate(x, cos, sin)

    @staticmethod
    def _apply_rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        # 將 x 視為 (x_even, x_odd) 交替排列
        # 按偶/奇索引拆分
        x1 = x[..., ::2]    # 偶數維度
        x2 = x[..., 1::2]   # 奇數維度
        # 旋轉公式: (x1*cos - x2*sin, x2*cos + x1*sin)
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x1 * sin
        # 交錯合併還原
        return torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2)
```

這段程式碼展現了 RoPE 的幾個實作要點：

1. **頻率預計算**：inv_freq 在初始化時就計算好，作為 buffer 儲存，避免每次 forward 重複計算
2. **位置編碼快取**：cos_cached 與 sin_cached 預先計算所有位置的旋轉角度值，forward 時只需按實際序列長度切片
3. **高效的旋轉運算**：將 d 維張量按偶/奇索引拆成兩個 shape (..., d/2) 的張量，用 element-wise 乘法取代稀疏矩陣乘法
4. **無訓練參數**：RoPE 本身不引入任何可學習參數，僅依賴預定義的頻率常數

### 4.7 NTK-aware 頻率縮放與 YaRN

RoPE 的原始實現受限於 base=10000 的頻率調度。當需要將上下文長度從訓練時的 2048 擴展到推理時的 32768 或更長時，高維度子空間的旋轉角度（mθ_i = m × 10000^{-2i/d}）會變得過大，導致位置表示失效。

**NTK-aware scaling**（Neural Tangent Kernel 感知縮放）的直覺是：不改變頻率值的**相對關係**，而是將所有頻率整體放緩。具體來說：

$$ \theta_i' = \text{base}'^{-2i/d}, \quad \text{base}' = \text{base} \times s^{d/(d-2)} $$

其中 s = L' / L 是目標長度與訓練長度的比值（scaling factor）。這樣做的效果是：原本在位置 L 的旋轉角度等於縮放後在位置 L × s 的角度，從而在不重新訓練的情況下讓模型「以為」長序列仍在訓練長度範圍內。

**YaRN**（Yet another RoPE extensioN, Peng et al., 2023）在此基礎上做出了兩項改進：

1. **逐維度頻率調整**：不同頻段的子空間使用不同的縮放因子——高頻（短距離）幾乎保持不變，低頻（長距離）大幅放緩。這避免了 NTK 方法對所有維度使用統一縮放因子的粗糙處理

2. **溫度縮放**：在調整頻率的同時，對注意力 logits 進行溫度縮放（乘以一個係數），補償因頻率變化導致的注意力分佈過度平滑或過度尖銳

YaRN 實現了令人矚目的成果：在 LLaMA 7B 上將上下文從 2048 擴展到 128K tokens（64×），在 LLaMA 33B 上擴展到 2M tokens（1000×）而不需要任何微調——這充分展示了 RoPE 作為預定義旋轉編碼的極佳外推能力。

### 4.8 RoPE 在實際模型中的具體配置

不同模型在使用 RoPE 時有一些微調，反映了對位置編碼頻率和維度的不同設計取捨：

| 模型 | 維度 d | base | 訓練長度 | 外推技術 |
|------|--------|------|----------|---------|
| RoFormer (原論文) | 768 | 10000 | 512 | 無 |
| LLaMA 1/2 | 4096 | 10000 | 2048/4096 | NTK-aware scaling |
| Mistral | 4096 | 10000 | 8192 | 滑動視窗 + RoPE |
| Gemma | 2048/3072 | 10000 | 8192 | 內部優化 |
| GPT-NeoX-20B | 6144 | 10000 | 2048 | 線性尺度縮放 |
| Code Llama | 4096 | 500000 | 16384 | base 擴大到 500k |
| Qwen 2.5 | 4096 | 1000000 | 32768 | base 擴大到 1M |

從表中可以看出，後來的工作幾乎都將 base 從 10000 提高到更大的值（500k-1M），以適應更長的上下文。這不是一個隨意的選擇——提高 base 從根本上減緩了所有頻率的旋轉速度，讓更高維度的子空間在更長序列中仍能有效區分位置。

值得注意的是 Code Llama 將 base 提高到 500,000 的設計決定：程式碼的 token 序列往往包含大量重複的結構模式（如縮排、大括號嵌套），這使得序列的有效長度遠大於一般文本。透過大幅提高 base 值，Code Llama 確保了即使在 16K tokens 的上下文視窗內，不同位置的 token 仍然能被旋轉編碼有效區分。類似地，Qwen 2.5 將 base 提高到 1,000,000 並搭配 32K 的訓練長度，實現了極佳的長上下文外推性能。

---

## 實驗結果總結

| 任務 | 資料集 | 指標 | RoFormer | Baseline | 提升 |
|------|--------|------|----------|----------|------|
| 機器翻譯 | WMT14 EN-DE | BLEU | 27.5 | 27.3 (Transformer) | +0.2 |
| 預訓練收斂 | BookCorpus+Wiki | MLM Loss | 更快下降 | BERT | 顯著 |
| 句子相似度 | MRPC (GLUE) | F1 | **89.5** | 88.9 (BERT) | +0.6 |
| 問答 NLI | QNLI (GLUE) | Acc | **91.2** | 90.5 (BERT) | +0.7 |
| 中文長文本 | CAIL2019-SCM | Test Acc | **69.79%** | 68.10% (WoBERT) | +1.69% |
| 線性注意力 | Enwik8 | LM Loss | 更快下降 | Performer | 顯著 |

---

## 限制與後續批評

### 論文承認的限制

1. **收斂加速的理論解釋不足**。論文在 Limitations 中坦承：「儘管我們有數學理論基礎和實驗驗證，但對於為什麼 RoPE 比基線方法收斂更快，仍缺乏透徹的解釋。」這可能是因為旋轉編碼在優化景觀中創造了更平滑的梯度信號，但尚未有正式證明。

2. **長距離衰減與性能的因果關係不明**。RoPE 的 long-term decay 性質是與 sinusoidal PE 共享的，但 RoPE 的性能卻更好。論文承認「沒有找到令人信服的解釋」。

3. **硬體資源需求**。作為基於 Transformer 的方法，RoPE 需要 GPU 資源進行預訓練，這限制了小型研究團隊的使用。

### 後續研究的批評與擴展

- **NTK-aware 頻率縮放**（bloc97, 2023; Peng et al., 2023）：RoPE 的原始頻率調度在極長序列（如 32K/128K tokens）上的外推效果仍有限。LLaMA 社群提出在推理時調整 θ_i = base^{-2i/d} 中的 base（從 10000 提升到 ~500000），使旋轉速度變慢，讓更高維度的子空間也能有效編碼長距離位置。這種「Neural Tangent Kernel (NTK) 感知」的縮放方法不需要額外訓練。

- **YaRN**（Peng et al., 2023）：進一步提出了「Yet another RoPE extensioN」方法，在 NTK 縮放的基礎上對頻率進行逐維度調整，並配合注意力 logits 的溫度縮放，實現了 2M tokens 的上下文長度外推。

- **隨機化頻率**：Sun et al.（2022）在「Randomized Position Encoding」中質疑了固定頻率調度的最優性，提出了訓練時隨機化 θ_i 的方案。他們認為固定的多頻率設定可能導致模型過度依賴特定頻段的訊號，從而降低泛化能力。

- **RoPE 對位置編碼效能的影響評估**：Kazemnejad et al.（2023）系統性研究了 RoPE 在語言模型中的實際作用，發現 RoPE 的主要貢獻並非來自其理論上的「相對位置編碼」能力，而是來自其保留的「絕對位置資訊」——這與原始論文的說法形成了一定的張力。

- **ALiBi 與 RoPE 的比較**：Press et al.（2022）提出的 ALiBi（Attention with Linear Biases）是另一個完全不學習任何位置參數的方法。與 RoPE 的旋轉方案不同，ALiBi 直接在 attention 分數上施加一個與距離線性相關的負偏置。兩者都不引入參數，但 RoPE 允許在 attention 加權時「看到」相對位置（因為 q/k 被旋轉），而 ALiBi 只是被動地壓制長距離的注意力權重。實驗顯示 RoPE 在外推性能和與線性注意力的相容性方面優於 ALiBi。

- **對 Transformer-XL 風格的替代**：Transformer-XL 的相對位置編碼需要同時維護內容與位置的投影矩陣 W^k 和 \tilde{W}^k，增加了實作複雜度。RoPE 透過統一的旋轉矩陣簡化了這一設計，這也是它被後續模型廣泛採用的實務原因之一。

### 開放問題與未來方向

1. **最優頻率調度是否存在？** 雖然 10000^{-2i/d} 是繼承自 Vaswani 2017 的調度，但後續 LLaMA、Qwen、Code Llama 等模型各自使用了不同的 base 值（10000 ~ 1e6）。目前缺乏對「什麼場景該用什麼頻率」的系統性理論指導。

2. **旋轉編碼的資訊瓶頸**。RoPE 將 d 維空間分成 d/2 個獨立的 2D 子空間，每個子空間的旋轉頻率固定。這是否意味著某些維度的位置編碼能力在高維空間中被浪費了？多頭注意力中不同 head 可以專注於不同頻段的資訊，但這尚未被系統性研究。

3. **超越旋轉：其他乘法式編碼的可能性**。RoPE 的成功證明了乘法式位置編碼的可行性。後續是否有更一般的乘法族（如除了 SO(2) 旋轉以外的正交變換）能帶來更好的位置編碼？這是一個尚待探索的設計空間。

4. **多語言與跨模態的影響**。RoPE 的頻率調度是針對自然語言文本設計的。當應用到程式碼（Code Llama）、數學（Minerva）、或多模態（視覺語言模型）時，是否需要不同的頻率配置？目前的做法是繼續沿用 10000 base 或透過實驗調整，缺乏一般化的指導原則。

5. **與其他位置編碼的混合可能性**。是否有場景需要同時使用 RoPE 與另一種位置編碼（例如在編碼器中使用 RoPE，解碼器中使用 ALiBi）？不同層或不同頭使用不同位置編碼策略的混合架構尚未被充分探索。

---

## 延伸閱讀

### 技術演化脈絡

1. **2017: Vaswani et al. — Attention Is All You Need**
   - 正弦絕對位置編碼
   - 預定義頻率，無需學習
   - 開創了 Transformer 架構

2. **2018: Shaw et al. — Self-Attention with Relative Position Representations**
   - 首個相對位置編碼方案
   - 學習式位置偏置，距離裁剪
   - +1.3 BLEU (EN-DE big)

3. **2019: Dai et al. — Transformer-XL**
   - 相對位置編碼 + segment-level recurrence
   - 區分內容/位置的投影矩陣
   - 解決固定長度上下文的限制

4. **2020: Raffel et al. — Exploring the Limits of Transfer Learning with T5**
   - 簡化為純標量相對偏置 b_{i,j}
   - 大幅減少參數量
   - 證明簡單方法也有效

5. **2020: He et al. — DeBERTa**
   - Disentangled attention：將內容和位置投影到不同空間
   - 位置項拆分為兩項獨立建模
   - 在多個 NLU 任務上超越 BERT/RoBERTa

6. **2021: Su et al. — RoFormer / RoPE ★**（本文核心）
   - 乘法式位置編碼（旋轉矩陣）
   - 從第一性原理推導
   - 相容線性注意力
   - 成為 LLaMA、Mistral、Gemma 等現代 LLM 的標準

7. **2022: Press et al. — ALiBi**
   - 線性偏置注意力（Attention with Linear Biases）
   - 不引入任何參數，與距離線性相關的負偏置
   - 在長序列外推上表現出色，但無法與線性注意力結合

8. **2023: Peng et al. — YaRN**
   - Yet another RoPE extensioN
   - 逐維度頻率調整 + 溫度縮放
   - 在 LLaMA 33B 上達成 2M tokens 上下文

9. **2023: LLaMA (Touvron et al.)**
   - 首個大規模採用 RoPE 的開源 LLM
   - 驗證了 RoPE 在大模型規模下的有效性
   - 引發了 RoPE 在後續模型中的廣泛採用

### 推薦閱讀順序

如果想深入理解 RoPE 與位置編碼的完整脈絡，建議依以下順序閱讀：

1. 先讀 **Vaswani 2017** 第 3.5 節（positional encoding）——理解位置編碼的原始問題
2. 接著讀 **Shaw 2018**——掌握相對位置編碼的基本框架
3. 然後讀 **Su 2021 (RoPE)**——見證從加法到乘法的轉變
4. 最後讀 **bloc97/Reddit 2023** 關於 NTK-aware scaling 的文章——了解 RoPE 在現代 LLM 中的實戰應用

### 為什麼 RoPE 成為現代 LLM 的標準選擇？

後 LLaMA 時代，RoPE 幾乎成為開源 LLM 的事實標準位置編碼方案。主要原因有三：

1. **零引進參數**：其他方法（Shaw、Transformer-XL、T5）都需要學習位置相關的參數，而 RoPE 的旋轉頻率是預定義的，不需要學習
2. **推論長度可擴展**：透過 NTK-aware scaling 或 YaRN 等技術，RoPE 可以輕鬆擴展到 128K 甚至 2M tokens 的上下文長度，而學習式方法需要重新訓練或微調
3. **與現有架構的相容性**：RoPE 只需要修改 attention 的 q/k 計算方式，不影響 FFN、layer norm、殘差連接等任何其他組件，可以作為 drop-in replacement 直接替換現有模型的位置編碼

4. **理論優雅性**：從數學條件出發推導而非啟發式設計，這讓 RoPE 的行為更容易理解和預測。對於 LLM 研究社群來說，一個有嚴謹理論基礎的方法比「我們試了，結果有效」的方法更值得信賴

5. **已被大規模驗證**：LLaMA（Meta, 2023）的開源與廣泛使用，讓 RoPE 成爲社群預設的選擇。一旦社群圍繞 RoPE 建立了完善的工具鏈（HuggingFace 整合、NTK-aware 腳本、長上下文擴展工具），切換到其他方案的轉換成本變得很高

### 與其他位置編碼方案的對比總結

| 方法 | 引入參數 | 相對位置 | 外推能力 | 線性注意力 | 理論基礎 |
|------|---------|---------|---------|-----------|---------|
| Sinusoidal (Vaswani 2017) | 0 | ❌ | 有限 | ❌ | 頻率分析 |
| Learned (BERT 2019) | L×d | ❌ | ❌ | ❌ | 無 |
| Shaw (2018) | 2×(2k+1)×d_a | ✅ | ✅ | ❌ | 無（啟發式） |
| Transformer-XL (2019) | 2×d×d（額外） | ✅ | ✅ | ❌ | 無（啟發式） |
| T5 bias (2020) | n×1（標量） | ✅ | ✅ | ❌ | 無（啟發式） |
| ALiBi (2022) | 0 | ✅ | ✅✅ | ❌ | 線性衰減分析 |
| **RoPE** (2021) | **0** | ✅ | ✅✅ | **✅** | **旋轉群理論** |

從上表可以看出，RoPE 是唯一在「零參數」、「相對位置」、「外推能力」和「線性注意力相容性」四個維度上都得到支援的方法。這解釋了為什麼它在近年來的 LLM 浪潮中脫穎而出。

---

## 參考資料

本篇文章涵蓋的論文：

- **種子論文**: Su, J., Lu, Y., Pan, S., Murtadha, A., Wen, B., & Liu, Y. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding. arXiv:2104.09864.
- Shaw, P., Uszkoreit, J., & Vaswani, A. (2018). Self-Attention with Relative Position Representations. NAACL 2018. arXiv:1803.02155.
- Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS 2017.
- Dai, Z., et al. (2019). Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. ACL 2019.
- Raffel, C., et al. (2020). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer. JMLR.
- He, P., et al. (2020). DeBERTa: Decoding-enhanced BERT with Disentangled Attention. ICLR 2021.
- Katharopoulos, A., et al. (2020). Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention. ICML 2020.
- Choromanski, K., et al. (2020). Rethinking Attention with Performers. ICLR 2021.
- Peng, B., et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models. arXiv:2309.00071.
- Sun, Y., et al. (2022). Randomized Position Encoding. arXiv:2205.09123.
- Kazemnejad, A., et al. (2023). The Impact of Position Encoding on Length Generalization in Transformers. arXiv:2305.19466.
- Press, O., et al. (2022). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation. ICLR 2022.
|- Touvron, H., et al. (2023). LLaMA: Open and Efficient Foundation Language Models. arXiv:2302.13971.
|- Jiang, A.Q., et al. (2023). Mistral 7B. arXiv:2310.06825.
|- Su, J. (2021). RoFormer 原始程式碼實作. GitHub: ZhuiyiTechnology/roformer.
|- bloc97. (2023). NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation. Reddit/r/LocalLLaMA.
