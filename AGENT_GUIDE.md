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
| `llm` | 大型語言模型:架構、預訓練、微調、對齊、推理 |
| `vlm` | 視覺-語言模型、多模態理解與生成 |
| `contrastive-learning` | 對比學習、自監督表徵學習 |
| `rl` | 強化學習(general),包含 RLHF 的 RL 部分 |
| `agentic-ai` | Agent 系統、工具使用、規劃、多 agent 協作 |
| `rag` | 檢索增強生成、向量檢索與 LLM 結合 |

### 主題分類規則

- 如果論文明確屬於上表某個 topic,直接放進去
- 跨主題的論文(例如 RLHF 兼具 LLM 與 RL),選**最主要的貢獻領域**作為 topic,在 `meta.yaml` 的 `tags` 欄位補上其他相關 tag
- 如果論文不屬於任何現有 topic,**新建一個 topic 目錄**,slug 用 kebab-case,並在 PR 描述中說明新增理由與建議放在索引何處

### 日期規則

- `YYYY-MM` 使用**論文 arxiv 第一版發表年月**(不是 v2、v3,也不是文章撰寫時間)
- 如果論文有期刊版本與 arxiv 版本,以 arxiv v1 為準
- 例如 DPO 論文 arxiv ID 為 2305.18290,目錄為 `2023-05-dpo/`

### Paper slug 規則

- 用論文的**通用簡稱**(community 慣用名),不是論文標題
- kebab-case,全小寫,只能有英數與連字號
- 例如:`dpo`、`grpo`、`react`、`self-rag`、`clip`
- 如果論文沒有公認簡稱,用「方法名 + 關鍵字」組合,例如 `chain-of-thought` 而非 `cot-prompting-elicits-reasoning`
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
- 文體:解讀、敘事為主,避免直譯論文章節
- 引用:引用論文原文時必須用 markdown blockquote 與行內引用標註,引用比例不可超過全文 10%
- 數學公式:使用 LaTeX,行內用 `$...$`,獨立公式用 `$$...$$`
- 圖片:放在 `assets/`,markdown 引用用相對路徑

### meta.yaml(結構化 metadata)

使用 [`_templates/meta-template.yaml`](./_templates/meta-template.yaml) 作為骨架。必填欄位:

```yaml
title: "DPO: Direct Preference Optimization 解讀"
topic: llm
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

`dependency_papers` 至多 3 筆。`knowledge_points` 建議 3~6 點。

### papers.bib(BibTeX)

包含種子論文 + 所有 dependency papers 的 BibTeX entry。從 arxiv 或 Google Scholar 直接抓即可。

---

## 3. Pull Request 規範

### Branch 策略

- 從 `main` 開新 branch,**禁止直接 push 到 main**
- Branch 命名:`agent/{topic-slug}/{paper-slug}`,例如 `agent/llm/dpo`
- 如果同一篇論文重新生成(改寫、修正),在後面加 `-rev2`、`-rev3`,例如 `agent/llm/dpo-rev2`

### Commit 規範

- 每個 PR 一個 commit(或邏輯清晰的少數幾個 commit)
- Commit message 格式:
  ```
  [{topic-slug}] add: {paper-slug} 解讀文章
  ```
  例如:`[llm] add: dpo 解讀文章`

### PR 標題

格式:`[{topic-slug}] {paper 通用簡稱}: 解讀文章`

例如:`[llm] DPO: 解讀文章`、`[rag] Self-RAG: 解讀文章`

### PR 描述模板

使用 [`.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md)。必填區塊:

```markdown
## 新增主題
- Topic: {topic-slug}
- 種子論文: {paper title} ({arxiv_id})

## 涵蓋論文清單
1. (種子) {paper title} - arxiv:{id}
2. {dependency paper 1} - arxiv:{id}
3. ...

## 核心知識點
- {知識點 1}
- {知識點 2}
- ...

## Agent 決策記錄
(這裡記錄執行過程中所有「規格未明確、agent 自行判斷」的決策,例如:
- 為何選 X 作為 dependency paper 而非 Y
- 為何把這篇論文歸到 topic A 而非 topic B
- 是否新建了 topic、為何
- 是否遇到 fallback 規則中描述的情境、如何處理
)

## 驗證
- [ ] `meta.yaml` 通過 `scripts/validate_article.py` 檢查
- [ ] 所有 dependency papers 都有對應 BibTeX entry
- [ ] 圖片(如有)放在 `assets/` 並以相對路徑引用
- [ ] 文章未引用超過 10% 的論文原文
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

- 如果新文章是「修訂」既有文章,使用 `agent/{topic}/{paper}-rev2` branch,在 PR 描述說明修訂原因
- 如果是「不同切入角度」的新文章,paper-slug 後加區分後綴,例如 `dpo-derivation`、`dpo-empirical`
- 不要直接覆蓋既有文章

### 4.4 知識點數量超出建議範圍

- 建議 3~6 點是參考值,不是硬上限
- 如果論文確實需要 8、9 個知識點才講得清楚,寫進去並在 PR 描述說明
- 如果只能想出 1、2 個知識點,可能是對論文理解不足,重讀後再寫

### 4.5 論文涉及敏感或爭議議題

- 照常解讀,聚焦技術內容
- 如果論文有明顯的倫理或社會影響討論,在「延伸閱讀」章節中性陳述,不下價值判斷
- 不確定時,跳過該段並在 PR 描述註明

### 4.6 其他未列舉的情境

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

最後一次更新:2026-05-20
