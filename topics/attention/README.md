# Attention — Attention 機制

Attention 機制從提出到現代的演進，涵蓋效率優化、位置編碼、長距離建模等核心子問題。

此類別專注於 **Attention 機制本身的設計與改進**，而非使用 Attention 的下游應用模型。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 基礎 Attention | Scaled Dot-Product、Multi-Head Attention | Attention Is All You Need (2017) |
| 位置編碼 | 如何賦予序列位置資訊 | Sinusoidal、RoPE、ALiBi |
| 效率化 Attention | 降低 O(n²) 複雜度 | Longformer、FlashAttention |
| 線性 Attention | 用核函數近似 Attention | Linear Transformer、Performer |
| 稀疏 Attention | 只計算部分 token 對 | Sparse Transformer、Routing Transformer |
| KV Cache 優化 | 推理加速的 KV 快取策略 | GQA、MQA、PagedAttention |
| Cross-Attention | 跨模態或跨序列的 Attention | Encoder-Decoder Attention |

## 不收錄於此

- 採用 Attention 的完整語言模型架構 → [`llm/`](../llm/)
- 採用 Attention 的視覺模型 → [`vision/`](../vision/)
- KV Cache 以外的推理加速（量化、蒸餾）→ [`efficiency/`](../efficiency/)

> **判斷原則**：主要貢獻是「Attention 機制本身的改進」→ 收錄於此。
> 主要貢獻是「用 Attention 解決某個應用任務」→ 收錄於對應的應用類別。

## 目錄結構

```
attention/
├── README.md                   # 本文件
├── 2017-06-transformer/        # Attention Is All You Need
├── 2022-05-flash-attention/    # FlashAttention
├── 2021-04-rope/               # RoFormer: RoPE
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
Scaled Dot-Product Attention (2017)
  └─ 效率化:
       Longformer (2020) → FlashAttention (2022) → FlashAttention-2 (2023)
  └─ 位置編碼:
       Sinusoidal (2017) → RoPE (2021) → ALiBi (2021)
  └─ KV Cache:
       MQA (2019) → GQA (2023) → PagedAttention (2023)
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
