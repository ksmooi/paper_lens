# Efficiency — 推理與訓練效率優化

在不顯著犧牲模型能力的前提下，降低計算成本、記憶體佔用或延遲的技術。

此類別聚焦於**對已訓練模型的壓縮與加速**，以及**訓練基礎設施的規模化**。參數高效微調（LoRA、Adapter 等）請見 [`peft/`](../peft/)。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 量化（Quantization） | 降低權重與激活值的數值精度 | GPTQ、AWQ、LLM.int8() |
| 剪枝（Pruning） | 移除不重要的權重或神經元 | SparseGPT、Wanda |
| 知識蒸餾（Distillation） | 用小模型模仿大模型 | DistilBERT、TinyLLaMA |
| 推理加速 | 縮短首 token 延遲與提升吞吐量 | Speculative Decoding、vLLM |
| 記憶體優化 | 降低訓練/推理的 GPU 記憶體 | Gradient Checkpointing、ZeRO |
| 硬體感知設計 | 針對 GPU/TPU 架構優化的演算法 | Triton Kernel、CUDA Graph |
| 大規模訓練基礎設施 | 數千 GPU 的分散式訓練框架 | Megatron-LM、DeepSpeed |

## 不收錄於此

- 參數高效微調（LoRA、QLoRA、Adapter 等）→ [`peft/`](../peft/)
- FlashAttention（核心是 Attention 機制重新設計）→ [`attention/`](../attention/)
- MoE 稀疏架構（稀疏性是架構設計本身）→ [`moe/`](../moe/)
- 優化器本身（Adam、Adafactor 等）→ [`optimization/`](../optimization/)

> **判斷原則**：論文的主要貢獻是「讓現有模型跑得更快/更省」→ 收錄於此。
> 「讓模型用更少參數適應新任務」→ 收錄於 [`peft/`](../peft/)。

## 目錄結構

```
efficiency/
├── README.md                   # 本文件
├── 2022-10-gptq/               # GPTQ: Accurate Post-Training Quantization
├── 2023-06-awq/                # AWQ: Activation-aware Weight Quantization
├── 2022-11-speculative/        # Speculative Decoding
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
量化:
  LLM.int8() (2022) → GPTQ (2022) → AWQ (2023) → GGUF (2023)

剪枝:
  SparseGPT (2023) → Wanda (2023)

推理加速:
  Speculative Decoding (2022) → vLLM / PagedAttention (2023)

分散式訓練:
  ZeRO (2020) → ZeRO-2/3 → ZeRO-Infinity (2021)
  Megatron-LM → Tensor Parallelism → Pipeline Parallelism
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
