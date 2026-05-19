# Templates

這個目錄存放 Hermes Agent 寫文章時使用的範本。

## 檔案說明

| 檔案 | 用途 | 複製到哪裡 |
|------|------|------------|
| [`article-template.md`](./article-template.md) | 文章主體骨架 | `topics/{topic}/{YYYY-MM}-{paper}/README.md` |
| [`meta-template.yaml`](./meta-template.yaml) | 結構化 metadata | `topics/{topic}/{YYYY-MM}-{paper}/meta.yaml` |
| [`pr-template.md`](./pr-template.md) | PR 描述範本 | `.github/PULL_REQUEST_TEMPLATE.md`(GitHub 自動套用) |

## 修改規則

- **這些範本由人類維護**,agent 不應在 PR 中修改本目錄下的檔案
- 範本變更會影響後續所有新文章,務必同步更新 [`../AGENT_GUIDE.md`](../AGENT_GUIDE.md) 對應規範
- 既有文章不需要回頭套用新範本(歷史版本保留原樣)

## 範本之間的關係

```
article-template.md ←→ AGENT_GUIDE.md §2 README.md 章節要求
meta-template.yaml  ←→ AGENT_GUIDE.md §2 meta.yaml 必填欄位
pr-template.md      ←→ AGENT_GUIDE.md §3 PR 描述模板
```

修改任一範本時,記得檢查 `AGENT_GUIDE.md` 相關章節是否需要同步更新。
