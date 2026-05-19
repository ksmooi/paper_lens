# Index

這個目錄存放**自動生成的索引檔案**,方便從不同維度查找文章。

> ⚠️ **這些檔案由 `scripts/generate_index.py` 自動生成,請勿手動編輯。**
> 任何手動修改都會在下次 CI 執行時被覆蓋。

---

## 檔案說明

| 檔案 | 排序維度 | 用途 |
|------|---------|------|
| [`by-date.md`](./by-date.md) | 論文發表時間 | 看 AI 領域的時間線 |
| [`by-topic.md`](./by-topic.md) | 主題分類 | 看每個主題下有哪些文章 |
| [`by-paper.md`](./by-paper.md) | 論文反查 | 查某篇論文被哪些文章引用 |
| [`reading-map.md`](./reading-map.md) | 論文依賴 | 看論文間的引用關係(知識圖譜) |

---

## 生成機制

每次 PR merge 到 `main` 後,GitHub Actions 會自動執行:

```bash
python scripts/generate_index.py
```

該腳本會掃描 `topics/**/meta.yaml`,依照各個維度重建本目錄下的索引檔案。

## 本地預覽

如果想在發 PR 前本地預覽索引長相:

```bash
python scripts/generate_index.py --dry-run    # 印到 stdout 不寫入檔案
python scripts/generate_index.py              # 實際寫入 _index/
```

## Schema 變更

如果新增了索引維度(例如 `by-author.md`),需要:

1. 修改 `scripts/generate_index.py` 加入新的生成函式
2. 更新本檔案的「檔案說明」表格
3. 在 PR 描述中說明新索引的用途
