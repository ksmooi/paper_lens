# Normalization — 正規化方法

神經網路訓練中各種正規化（Normalization）技術的演進與比較。

正規化方法解決的核心問題是**內部共變量偏移（Internal Covariate Shift）**與**訓練穩定性**，是現代深度學習得以擴展至極深網路的關鍵基礎技術之一。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| Batch-level | 對 mini-batch 統計量正規化 | Batch Normalization (2015) |
| Layer-level | 對單一樣本所有神經元正規化 | Layer Normalization (2016) |
| Instance-level | 對單一樣本單一 channel 正規化 | Instance Normalization (2017) |
| Group-level | 介於 Layer 與 Instance 之間的分組方式 | Group Normalization (2018) |
| 無 re-centering | 捨棄均值統計以降低計算成本 | RMSNorm (2019) |
| 動態/條件式 | 根據外部條件動態調整縮放參數 | AdaIN、FiLM |

## 不收錄於此

- 優化器層面的梯度正規化（Gradient Clipping 等）→ [`optimization/`](../optimization/)
- Dropout、Weight Decay 等**正則化**（Regularization）→ 請注意與正規化（Normalization）區分

## 目錄結構

```
normalization/
├── README.md                   # 本文件
├── 2015-02-batchnorm/          # Batch Normalization
├── 2016-07-layernorm/          # Layer Normalization
├── 2019-10-rmsnorm/            # RMSNorm
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 方法對照表

| 方法 | 統計維度 | 適用場景 | 含 mean 計算 |
|---|---|---|---|
| BatchNorm | (N, H, W) | CNN、大 batch | ✓ |
| LayerNorm | (C, H, W) | Transformer、NLP | ✓ |
| InstanceNorm | (H, W) | 風格遷移 | ✓ |
| GroupNorm | group(C), H, W | 小 batch CV | ✓ |
| RMSNorm | (C, H, W) | LLM（省略 mean） | ✗ |

## 閱讀路徑建議

```
BatchNorm (2015)
  └─ LayerNorm (2016)    ← Transformer 採用
       └─ RMSNorm (2019) ← LLaMA、Mistral 採用
  └─ GroupNorm (2018)    ← 小 batch CV 場景
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
