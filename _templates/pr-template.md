<!--
這是 Paper Lens 的 PR 描述範本(供 Hermes Agent 使用)。
這份檔案會被同步到 .github/PULL_REQUEST_TEMPLATE.md,GitHub 開新 PR 時會自動套用。

PR 標題格式:[{topic-slug}] {paper 通用簡稱}: 解讀文章
例如:[llm] DPO: 解讀文章
-->

## 新增主題

- **Topic**: `{topic-slug}`
- **是否新增 topic**: 否 / 是(說明:...)
- **種子論文**: [{Paper Title}](https://arxiv.org/abs/{arxiv_id}) ({YYYY-MM})
- **文章路徑**: `topics/{topic-slug}/{YYYY-MM}-{paper-slug}/`

---

## 涵蓋論文清單

| # | 角色 | 論文 | arxiv ID | 與本文關係 |
|---|------|------|----------|------------|
| 1 | 種子 | {Paper Title} | {arxiv_id} | — |
| 2 | Dep. | {Paper Title} | {arxiv_id} | 一句話說明 |
| 3 | Dep. | {Paper Title} | {arxiv_id} | 一句話說明 |
| 4 | Dep. | {Paper Title} | {arxiv_id} | 一句話說明 |

> Dependency papers 至多 3 篇。連同種子論文總共最多 4 篇。

---

## 核心知識點

本文圍繞以下知識點展開:

1. {知識點 1}
2. {知識點 2}
3. {知識點 3}
4. {知識點 4}(選填)

---

## Agent 決策記錄

> 這個區塊記錄執行過程中所有**規格未明確、agent 自行判斷**的決策,以便事後審閱。
> 即使本次執行沒有特殊判斷,也請保留此區塊並寫「無特殊決策」。

### Dependency papers 選擇

**為何選 X 而非 Y?**

- (例) 在 PPO 與 TRPO 之間選擇 PPO 作為 RL 基準論文,因為 InstructGPT 實際使用 PPO,且 PPO 在現代 LLM 對齊文獻中更常被引用作為對照
- (例) 未納入 SLiC-HF,因為 dependency 上限為 3 篇,優先保留形成清晰方法對照的論文

### Topic 歸類

**為何把這篇歸到 topic A 而非 topic B?**

- (例) DPO 雖然涉及 preference learning 與 RL 概念,但實際應用場景與評估都是 LLM 對齊,主要貢獻屬於 LLM 訓練方法,故歸入 `llm` topic,在 tags 中補上 `rlhf`、`preference-learning`
- 如果新建了 topic,在此說明新建理由與該 topic 的範圍定義

### 知識點歸納

**為何選這幾個知識點?**

- (例) 從種子論文的 Introduction 與 Method 章節歸納出 4 個知識點,涵蓋「動機 / 數學推導 / 對比 / 失敗模式」四個面向
- 如果知識點數量超出建議範圍(3~6),在此說明

### 其他決策

- (例) 論文中圖 3 的部分數字與正文不一致,本文採用正文數字並在文章內以註腳標記
- (例) 某段論文原文翻譯時意譯而非直譯,因直譯後在中文語境下難以理解

### 遇到的 Fallback 情境

- [ ] 種子論文有多個候選 → 處理方式: ...
- [ ] Dependency paper 找不到 arxiv 版本 → 處理方式: ...
- [ ] 該主題已有同名文章 → 處理方式: ...
- [ ] 其他: ...

(沒遇到的情境不用勾選)

---

## 驗證清單

執行 PR 前 agent 自我檢查:

- [ ] `meta.yaml` 必填欄位齊全
- [ ] `papers.bib` 包含所有引用論文的 BibTeX entry
- [ ] 文章符合 [`_templates/article-template.md`](../_templates/article-template.md) 的章節結構
- [ ] 語言為繁體中文,專有名詞保留英文
- [ ] 論文原文直接引用比例 < 10%
- [ ] 知識點是「歸納的概念」而非「論文章節標題」
- [ ] 圖片(如有)放在 `assets/` 並以相對路徑引用
- [ ] Branch 名稱為 `agent/{topic-slug}/{paper-slug}` 格式
- [ ] Commit message 為 `[{topic-slug}] add: {paper-slug} 解讀文章` 格式

---

## CI 狀態

- [ ] `lint-markdown` 通過
- [ ] `check-frontmatter` 通過
- [ ] `validate_article` 通過

> 如果 CI 失敗超過 3 次,agent 會在此 PR 留言說明遇到的問題並停止重試,等待人工介入。

---

## 給審閱者

- 本文由 Hermes Agent 自動生成
- Agent run ID: `{hermes-YYYY-MM-DD-NNN}`
- 如需重新生成本文,請在 `meta.yaml` 中將 `status` 改為 `needs-revision` 並 close 此 PR
