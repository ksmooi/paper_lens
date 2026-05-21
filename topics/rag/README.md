# RAG — Retrieval-Augmented Generation

讓語言模型在生成時能夠動態查詢外部知識，解決知識截止日期與幻覺問題。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 基礎 RAG | 檢索 + 生成的基本框架 | RAG (Lewis et al., 2020) |
| 稠密檢索 | 以嵌入向量做語意檢索 | DPR、ColBERT |
| 生成式檢索 | 模型直接生成文件 ID | DSI、GENRE |
| 長文本 RAG | 處理長段落或多文件的 RAG | RAPTOR、LongRAG |
| 圖譜增強 | 結合知識圖譜的 RAG | GraphRAG |
| 自適應 RAG | 動態決定是否需要檢索 | Self-RAG、Adaptive RAG |
| RAG 評估 | RAG 系統的評估框架 | RAGAS、RGB |

## 不收錄於此

- 嵌入模型本身（Sentence-BERT 等）→ [`embedding/`](../embedding/)
- 純語言模型（不含檢索）→ [`llm/`](../llm/)
- 以 RAG 為工具的完整 Agent 系統 → [`agentic/`](../agentic/)

## 目錄結構

```
rag/
├── README.md                   # 本文件
├── 2020-05-rag/                # Retrieval-Augmented Generation (Lewis)
├── 2020-04-dpr/                # Dense Passage Retrieval
├── 2023-10-self-rag/           # Self-RAG
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
BM25（傳統稀疏檢索）
  └─ DPR (2020)               ← 稠密向量檢索
       └─ RAG (2020)          ← 整合進生成模型
            └─ FiD (2020)
            └─ RETRO (2021)
  └─ ColBERT (2020)           ← 遲後互動式檢索

Self-RAG (2023)               ← 自適應檢索
GraphRAG (2024)               ← 圖譜增強
RAPTOR (2024)                 ← 層次化長文本
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
