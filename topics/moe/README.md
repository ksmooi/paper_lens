# MoE — Mixture of Experts

稀疏激活的混合專家架構：以遠低於稠密模型的計算量，達到相當甚至更優的模型能力。

MoE 的核心洞見是「**並非所有參數都需要為每個 token 啟動**」。透過路由機制動態選擇少數專家子網路，實現參數量與計算量的解耦。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 基礎 MoE | 最早期的 MoE 理論與架構 | Jacobs et al. (1991)、Shazeer (2017) |
| Transformer MoE | 將 MoE 嵌入 Transformer FFN 層 | GShard、Switch Transformer |
| 路由機制 | 如何決定 token 送往哪個 Expert | Top-k Routing、Expert Choice |
| 負載均衡 | 避免 Expert 使用率不均 | Auxiliary Loss、Z-Loss |
| MoE LLM | 以 MoE 為架構的完整 LLM | Mixtral、DeepSeek-MoE |
| 細粒度 MoE | 更多但更小的 Expert | DeepSeekMoE、Fine-Grained MoE |

## 不收錄於此

- 純稠密 Transformer 架構 → [`llm/`](../llm/)
- MoE 模型的量化/蒸餾 → [`efficiency/`](../efficiency/)
- MoE 模型的 PEFT 微調 → [`peft/`](../peft/)

## 目錄結構

```
moe/
├── README.md                       # 本文件
├── 2017-01-outrageously-large/     # Outrageously Large Neural Networks
├── 2021-01-switch-transformer/     # Switch Transformer
├── 2024-01-deepseek-moe/           # DeepSeekMoE
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
Adaptive Mixtures (1991)          ← 理論起源
  └─ Sparsely-Gated MoE (2017)   ← 現代 MoE 起點
       └─ GShard (2020)
       └─ Switch Transformer (2021)
            └─ GLaM (2021)
            └─ Mixtral 8x7B (2023)
            └─ DeepSeekMoE (2024)
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
