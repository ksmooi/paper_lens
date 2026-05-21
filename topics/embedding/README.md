# Embedding — 嵌入與表示學習

如何將原始資料（文字、圖像、程式碼）壓縮成密集向量表示，使語意相近的輸入在空間中鄰近。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 詞嵌入 | 詞彙層級的靜態表示 | Word2Vec、GloVe、FastText |
| 上下文嵌入 | 依上下文動態生成的表示 | ELMo、BERT Embeddings |
| 句子/段落嵌入 | 整段文字的單一向量表示 | Sentence-BERT、SimCSE |
| 對比學習 | 用正負樣本對訓練表示空間 | MoCo、SimCLR |
| 程式碼嵌入 | 程式碼的語意表示 | CodeBERT、StarCoder Embeddings |
| 多模態嵌入 | 統一不同模態的向量空間 | ImageBind、CLAP |
| 向量檢索 | 高效近似最近鄰搜尋 | FAISS、ScaNN、HNSW |

## 不收錄於此

- 以對比學習為基礎的完整視覺語言模型（CLIP 整體）→ [`vlm/`](../vlm/)
- 整合檢索的生成系統 → [`rag/`](../rag/)
- 位置嵌入（RoPE、ALiBi）→ [`attention/`](../attention/)

## 目錄結構

```
embedding/
├── README.md                   # 本文件
├── 2013-01-word2vec/           # Efficient Estimation of Word Representations
├── 2018-02-elmo/               # Deep Contextualized Word Representations
├── 2019-08-sentence-bert/      # Sentence-BERT
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
Word2Vec (2013) → GloVe (2014)
  └─ ELMo (2018)              ← 上下文嵌入
       └─ BERT (2018)
            └─ Sentence-BERT (2019)
                 └─ SimCSE (2021)

MoCo (2019) → SimCLR (2020)  ← 自監督視覺對比
  └─ CLIP (2021)              ← 跨模態對比
       └─ ImageBind (2023)    ← 多模態統一嵌入
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
