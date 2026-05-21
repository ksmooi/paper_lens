# Optimization — 優化器與訓練策略

神經網路訓練過程中的梯度優化方法、學習率排程、訓練穩定性與損失設計。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 一階優化器 | 基於梯度的優化方法 | SGD、Adam、AdamW、Adafactor |
| 二階優化器 | 利用曲率資訊的優化方法 | K-FAC、Shampoo |
| 學習率排程 | 動態調整學習率的策略 | Warmup、Cosine Decay、OneCycleLR |
| 訓練穩定性 | 解決梯度爆炸/消失的技術 | Gradient Clipping、Loss Scaling |
| 損失函數設計 | 目標函數的設計與改進 | Label Smoothing、Focal Loss |
| 資料效率 | 用更少資料達到相同效果 | Curriculum Learning、Data Mixing |
| 混合精度訓練 | FP16/BF16 訓練策略 | Mixed Precision (Micikevicius) |

## 不收錄於此

- 參數高效微調（LoRA 等）→ [`peft/`](../peft/)
- 量化推理（GPTQ 等，非訓練時）→ [`efficiency/`](../efficiency/)
- Batch/Layer Normalization 等正規化層 → [`normalization/`](../normalization/)
- 強化學習中的策略優化（PPO 等）→ [`rl/`](../rl/)

## 目錄結構

```
optimization/
├── README.md                   # 本文件
├── 2014-12-adam/               # Adam: A Method for Stochastic Optimization
├── 2017-11-adamw/              # Decoupled Weight Decay Regularization
├── 2018-09-adafactor/          # Adafactor: Adaptive Learning Rates
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
SGD + Momentum
  └─ RMSProp (2012)
  └─ Adam (2014)              ← 現代 LLM 預設優化器
       └─ AdamW (2017)        ← 修正 Weight Decay
       └─ Adafactor (2018)    ← 記憶體高效替代
       └─ Lion (2023)         ← 符號梯度優化器

混合精度訓練 (2018)
  └─ BF16 訓練（LLM 標配）
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
