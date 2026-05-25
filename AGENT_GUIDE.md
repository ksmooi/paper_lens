# Agent Guide

這份文件給 **Hermes Agent**(或任何其他自動化 agent)閱讀。
人類讀者請看 [`README.md`](./README.md)。

本文件規範 agent 如何向本 repo 新增文章、命名檔案、發 Pull Request,以及在哪些情境下可以自行決策。

---

## 0. 一個原則

> **這份指南未明確規範的部分,agent 自行判斷並繼續執行,不中斷等待確認。判斷依據寫進 PR 描述。**

明確規範的部分必須照做。模糊地帶才用判斷力。

---

## 1. 寫入位置

新文章必須放在:

```
topics/{topic-slug}/{YYYY-MM}-{paper-slug}/
```

### Topic slug 對照表

| Topic Slug | 涵蓋範圍 |
|------------|---------|
| `llm` | 大型語言模型:架構、預訓練、Scaling Law。微調與對齊見 `peft` / `rl` |
| `normalization` | 正規化方法:BatchNorm、LayerNorm、RMSNorm、GroupNorm、AdaIN |
| `vision` | 電腦視覺:骨幹網路、物件偵測、語意分割、GAN、自監督視覺學習、3D 視覺 |
| `vlm` | 視覺語言模型:多模態理解與生成、CLIP、LLaVA、Flamingo |
| `attention` | Attention 機制本身的改進:位置編碼、效率化、線性 Attention、KV Cache 優化 |
| `diffusion` | 擴散模型:DDPM、LDM、DiT、加速採樣、流匹配、條件式生成 |
| `efficiency` | 推理與訓練效率:量化、剪枝、蒸餾、推理加速、分散式訓練基礎設施 |
| `embedding` | 嵌入與表示學習:詞向量、句向量、對比學習、多模態嵌入、向量檢索 |
| `moe` | Mixture of Experts:路由機制、負載均衡、MoE LLM |
| `optimization` | 優化器與訓練策略:Adam 系列、學習率排程、混合精度、損失函數設計 |
| `peft` | 參數高效微調與指令微調:LoRA、QLoRA、Adapter、Prefix-Tuning、FLAN、Alpaca |
| `rag` | 檢索增強生成:稠密檢索、自適應 RAG、圖譜增強、RAG 評估 |
| `rl` | 強化學習與對齊:PPO、RLHF、DPO、過程監督、推理強化。Agent 框架見 `agentic` |
| `agentic` | LLM Agent：規劃、反思、工具使用、程式碼 Agent、Multi-Agent。Agent 外部控制層見 `harness` |
| `harness` | Harness Engineering：Agent 外部控制層、Scaffolding、Context Engineering、Runtime Orchestration、Tool Governance、Verification Gate、Evaluation Harness |

### 主題分類規則

- 如果論文明確屬於上表某個 topic,直接放進去
- 跨主題的論文(例如 ControlNet 同時屬於 `diffusion` 與 `peft`),選**最主要的貢獻領域**作為 topic,在 `meta.yaml` 的 `tags` 欄位補上其他相關 tag
- 如果論文不屬於任何現有 topic,**新建一個 topic 目錄**,slug 用 kebab-case,並在 PR 描述中說明新增理由

### 常見跨主題判斷參考

| 論文類型 | 優先歸屬 | 理由 |
|----------|----------|------|
| RLHF、InstructGPT | `rl` | 核心貢獻是 RL 訓練框架 |
| LoRA、QLoRA、指令微調 | `peft` | 核心貢獻是參數高效適應方法 |
| ControlNet、LoRA for Diffusion | `peft` | 核心貢獻是微調方法，非擴散架構 |
| ReAct、Toolformer | `agentic` | 核心貢獻是讓 LLM 具備推理、行動與工具使用能力 |
| AgentBench、SWE-Bench、WebArena | `agentic` | 核心貢獻是評估 Agent 在不同環境中的任務能力 |
| Natural-Language Agent Harnesses、AutoHarness | `harness` | 核心貢獻是設計 Agent 外部控制層、執行規格與 harness runtime |
| UTBoost、SWE-Bench-CL | `harness` | 核心貢獻是強化 Agent 評估流程、verification gate、測試可靠性或 continual evaluation |
| Building AI Coding Agents for the Terminal、Scalable Agent Scaffolding | `harness` | 核心貢獻是 coding agent 的 scaffolding、context engineering、runtime orchestration 與可控執行流程 |
| DeepSeek-R1、GRPO | `rl` | 核心貢獻是用 RL 強化推理能力 |
| FlashAttention | `attention` | 核心貢獻是 Attention 計算方式重設計 |
| CLIP（整體架構） | `vlm` | 視覺語言對齊的完整系統 |
| Sentence-BERT、SimCSE | `embedding` | 核心貢獻是句向量訓練方法 |

### 日期規則

- `YYYY-MM` 使用**論文 arxiv 第一版發表年月**(不是 v2、v3,也不是文章撰寫時間)
- 如果論文有期刊版本與 arxiv 版本,以 arxiv v1 為準
- 例如 DPO 論文 arxiv ID 為 2305.18290,目錄為 `2023-05-dpo/`

### Paper slug 規則

- 用論文的**通用簡稱**(community 慣用名),不是論文標題
- kebab-case,全小寫,只能有英數與連字號
- 例如:`dpo`、`grpo`、`react`、`self-rag`、`clip`、`lora`、`qlora`
- 如果論文沒有公認簡稱,用「方法名 + 關鍵字」組合,例如 `chain-of-thought`
- Slug 長度建議在 30 字元內

---

## 2. 必要檔案

每個文章資料夾**必須**包含以下檔案:

```
{YYYY-MM}-{paper-slug}/
├── README.md       # 文章主體(必須)
├── meta.yaml       # 結構化 metadata(必須)
├── papers.bib      # 涵蓋的論文 BibTeX(必須)
└── assets/         # 圖片、示意圖(選用,有圖才建)
```

### README.md(文章主體)

使用 [`_templates/article-template.md`](./_templates/article-template.md) 作為骨架。必須包含以下章節(順序可調整,但不可省略):

- **TL;DR**——三句話總結
- **背景與動機**
- **核心知識點**——條列形式,本篇主題的關鍵概念
- **方法詳解**——依知識點逐一展開
- **實驗結果**——數據、對比、消融
- **延伸閱讀**——dependency papers 與後續發展

撰寫要求:

- 語言:**繁體中文**(專有名詞與論文名稱保留英文)
- 篇幅:**700–1000 行**(以 Markdown 原始檔 `wc -l` 計,含 frontmatter、空行、code block、圖檔引用)
  - 未達 700 行:補強數學推導、消融分析解讀、實作細節、限制與批評
  - 超過 1000 行:壓縮重複論述、合併相鄰小節
  - 不允許用廢話湊行數,也不允許刪核心知識點削減行數
- 文體:解讀、敘事為主,避免直譯論文章節
- 引用:引用論文原文時必須用 markdown blockquote 與行內引用標註,引用比例不可超過全文 10%
- 數學公式:使用 LaTeX,行內用 `$...$`,獨立公式用 `$$...$$`
- 圖片:放在 `assets/`,markdown 引用用相對路徑;Mermaid 圖直接內嵌為 code block

### meta.yaml(結構化 metadata)

使用 [`_templates/meta-template.yaml`](./_templates/meta-template.yaml) 作為骨架。必填欄位:

```yaml
title: "DPO: Direct Preference Optimization 解讀"
topic: rl
paper_date: 2023-05         # 論文發表年月(對應目錄名)
article_date: 2026-05-20    # 文章撰寫日期
seed_paper:
  title: "Direct Preference Optimization: Your Language Model is Secretly a Reward Model"
  arxiv_id: "2305.18290"
  url: "https://arxiv.org/abs/2305.18290"
dependency_papers:
  - title: "Training language models to follow instructions with human feedback"
    arxiv_id: "2203.02155"
    url: "https://arxiv.org/abs/2203.02155"
    relation: "DPO 的前身方法 RLHF/InstructGPT"
knowledge_points:
  - "從 RLHF 到 DPO 的動機:為何要去掉 reward model"
  - "DPO 損失函數的推導:從 Bradley-Terry 模型出發"
  - "與 PPO 的對比:訓練穩定性與計算成本"
tags: [alignment, preference-learning, rlhf]
agent_generated: true
agent_run_id: "hermes-2026-05-20-001"   # agent 執行批次的識別碼,方便追溯
```

`dependency_papers` 至多 3 筆。`knowledge_points` 建議 3–6 點。

### papers.bib(BibTeX)

包含種子論文 + 所有 dependency papers 的 BibTeX entry。從 arxiv 或 Google Scholar 直接抓即可。

---

## 3. Pull Request 規範

### Branch 策略

- 從 `main` 開新 branch,**禁止直接 push 到 main**
- Branch 命名:`add-{paper-slug}-{YYYYMMDD}`
  例如:`add-dpo-20260520`、`add-lora-20260521`
- 如果同一個 paper-slug 在同一天重新生成(改寫、修正),在日期後加 `-v2`、`-v3`,例如 `add-dpo-20260520-v2`
- 如果同一個 paper-slug 已存在於 remote 的任何 branch(不論日期),paper-slug 後綴加 `-v2`、`-v3` 遞增,直到不衝突為止

### Commit 規範

- 每個 PR 一個 commit(或邏輯清晰的少數幾個 commit)
- Commit message 格式:
  ```
  feat: add {paper-slug} article (Hermes Agent validation)
  ```
  例如:`feat: add dpo article (Hermes Agent validation)`

### PR 標題

格式:`新增 {RESEARCH_TOPIC} 主題文章`

例如:`新增 DPO (Direct Preference Optimization) 主題文章`

### PR 描述模板

使用 [`.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md)。必填區塊:

```markdown
## 涵蓋論文
- [{種子論文標題} ({作者前三位} {年份})](https://arxiv.org/abs/{arxiv_id})
- [{dependency paper 標題}](https://arxiv.org/abs/{arxiv_id})

## Dependency Paper 選擇理由
- 選擇理由:(2–3 句說明,引用種子論文中的證據)
- 考慮過但未選的候選:(如有,說明原因)

## 核心知識點
- {知識點 1}
- {知識點 2}
- ...

## 分類與路徑
- Topic: {topic-slug}
- 文章路徑: topics/{topic-slug}/{YYYY-MM}-{paper-slug}/
- 文章行數: {wc -l 結果}(目標 700–1000 行)
- 圖檔數量: {N 張,含 Mermaid 內嵌 X 張、獨立檔案 Y 張}

## 視覺資產使用情況
- 圖 1: {用途} | 工具: {實際用的工具} | 檔案: {檔名或「內嵌 Mermaid」}
- ...

## 篇幅調整紀錄
- 初稿行數: {數字}
- 最終行數: {數字}
- 調整輪次: {次數}
- 主要調整內容: {擴充了什麼 / 削減了什麼}

## Agent 決策記錄
- Topic 分類與理由(含跨主題取捨):
- Paper slug 命名與理由:
- nano-pdf 使用情況:
- llm-wiki 使用情況(查了哪些術語):
- 工具替換(若有):
- 未明確規範的部分:
- 其他自行判斷的決策:

## Git 身份驗證
- Commit author: ksmooi <kaishianmooi@gmail.com>
- 驗證指令輸出:(貼上 `git log -1 --format='%an <%ae>'` 的結果)

## 驗證
- [ ] `meta.yaml` 通過 `scripts/validate_article.py` 檢查
- [ ] 所有 dependency papers 都有對應 BibTeX entry
- [ ] 圖片(如有)放在 `assets/` 並以相對路徑引用
- [ ] 文章未引用超過 10% 的論文原文
- [ ] 文章行數在 700–1000 行範圍內(`wc -l` 驗證)

## 驗證版本說明
此 PR 為 Hermes Agent 論文講解功能,主題:{RESEARCH_TOPIC}。
```

---

## 4. 常見 Fallback 規則

以下是已知會發生的幾種情境,**先按規則處理,並在 PR 描述記錄**:

### 4.1 種子論文有多個候選

如果某主題的「種子論文」不只一篇(例如 contrastive learning 的 SimCLR vs MoCo),選擇:

1. **時間最早**的那篇作為種子
2. 另一篇放進 dependency papers
3. 在 PR 描述說明取捨理由

### 4.2 Dependency paper 找不到 arxiv 版本

- 優先使用 arxiv,如果沒有,使用論文官方來源(會議 proceedings、期刊頁面)
- `meta.yaml` 的 `arxiv_id` 留空,`url` 填官方來源連結
- 在 PR 描述註明該論文無 arxiv 版本

### 4.3 該主題已有同名文章

- 如果新文章是「修訂」既有文章,branch 命名在 paper-slug 後加 `-v2`,例如 `add-dpo-20260521-v2`,在 PR 描述說明修訂原因
- 如果是「不同切入角度」的新文章,paper-slug 後加區分後綴,例如 `dpo-derivation`、`dpo-empirical`
- 不要直接覆蓋既有文章

### 4.4 知識點數量超出建議範圍

- 建議 3–6 點是參考值,不是硬上限
- 如果論文確實需要 7、8 個知識點才講得清楚,寫進去並在 PR 描述說明
- 如果只能想出 1、2 個知識點,可能是對論文理解不足,重讀後再寫

### 4.5 文章行數達不到 700 行

按以下優先順序補充,直到達標:

1. 補強數學推導(補出論文省略的中間步驟)
2. 補強消融實驗(ablation study)解讀
3. 補強實作細節與工程考量
4. 補強與後續工作的連結、被引用情況
5. 補強限制(limitations)與批評觀點

不允許用無意義的廢話或重複內容湊行數。

### 4.6 文章行數超過 1000 行

按以下優先順序削減:

1. 壓縮重複論述的段落
2. 把次要數學推導改為「請參閱原論文 §X.Y」
3. 合併相鄰小節
4. 削減舉例的數量

不允許刪掉核心知識點或視覺資產。

### 4.7 視覺 skill 產出失敗

- 降級為 Mermaid code block 純文字嵌入
- 若 Mermaid 也無法表達,允許省略該張圖
- 在 PR 描述「視覺資產使用情況」說明降級原因

### 4.8 論文涉及敏感或爭議議題

- 照常解讀,聚焦技術內容
- 如果論文有明顯的倫理或社會影響討論,在「延伸閱讀」章節中性陳述,不下價值判斷
- 不確定時,跳過該段並在 PR 描述註明

### 4.9 其他未列舉的情境

自行判斷,在 PR 描述「Agent 決策記錄」區塊記錄判斷依據。**寧可繼續執行也不要中斷等待確認**。

---

## 5. 禁止事項

- ❌ 直接 push 到 `main` branch
- ❌ 修改其他文章的內容(除非該 PR 的明確目的就是修訂)
- ❌ 修改 `_index/` 下的檔案(由 CI 自動生成)
- ❌ 修改 `_templates/`、`AGENT_GUIDE.md`、`README.md`(這些變更需人工審閱)
- ❌ 文章中超過 10% 比例的論文原文直接引用
- ❌ 知識點清單抄論文章節標題(必須是 agent 歸納後的概念,不是論文目錄)
- ❌ 在沒有實際讀過論文的情況下憑印象寫(每篇 dependency paper 都必須實際讀過再寫)
- ❌ 修改 git remote URL(尤其是嵌入 token 到 URL)
- ❌ 修改 git credential 設定(credential.helper、cmdkey、.git-credentials 等)
- ❌ 用 curl 帶 token 直接呼叫 GitHub API 繞過 gh CLI

---

## 6. 驗證流程

PR 開啟後,GitHub Actions 會自動執行:

1. `lint-markdown.yml`——檢查 markdown 格式
2. `check-frontmatter.yml`——驗證 `meta.yaml` 必填欄位齊全
3. `scripts/validate_article.py`——檢查目錄結構、檔案命名、引用比例

CI 失敗時,agent 應修正後重新 push 到同一個 branch。連續 3 次 CI 失敗,agent 應在 PR 中留言說明遇到的具體問題,然後停止重試,等待人工介入。

---

## 7. 風格細節

- 「論文」一詞統一用「論文」,不用「文章」(文章指 repo 中的解讀文章)
- 第一人稱用「我」(代表 repo 主人視角),不用「我們」「筆者」
- 不寫客套話、不寫「希望本文能...」這類結語
- 數字與單位之間留半形空格:`5 GB`、`128 tokens`(中文與英數之間也是)
- 條列項目之間不空行(除非條目本身是多段落)

---

最後一次更新:2026-05-21
