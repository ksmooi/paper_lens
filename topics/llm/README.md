# LLM — Large Language Models

大型語言模型的核心架構、預訓練策略與能力延伸。

此類別收錄以**語言建模為核心目標**的論文，涵蓋從基礎 Transformer 架構到當代 LLM 的演進脈絡。微調與對齊方法請見 [`peft/`](../peft/) 與 [`rl/`](../rl/)。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 基礎架構 | Transformer 及其衍生架構設計 | Attention Is All You Need (2017) |
| 預訓練策略 | 語言模型預訓練目標與資料配方 | GPT 系列、BERT、T5 |
| 長上下文 | 擴展 context window 的技術 | LongFormer、YaRN |
| 推理能力 | 鏈式思考、推理增強 | Chain-of-Thought、OpenAI o1 |
| 多語言 | 跨語言泛化能力 | mBERT、XLM-R |
| 模型擴展定律 | Scaling Law 與資料配方 | Chinchilla、Scaling Laws (Kaplan) |

## 不收錄於此

- 指令微調、PEFT（LoRA 等）→ [`peft/`](../peft/)
- 強化學習對齊（RLHF、DPO）→ [`rl/`](../rl/)
- 視覺語言多模態模型 → [`vlm/`](../vlm/)
- 推理/記憶體效率優化 → [`efficiency/`](../efficiency/)
- Mixture of Experts 架構 → [`moe/`](../moe/)
- 檢索增強生成 → [`rag/`](../rag/)
- LLM Agent 框架 → [`agentic/`](../agentic/)

## 目錄結構

```
llm/
├── README.md               # 本文件
├── 2017-06-transformer/    # Attention Is All You Need
├── 2018-10-bert/           # BERT
├── 2020-05-gpt3/           # GPT-3
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
Transformer (2017)
  └─ BERT (2018) ──── T5 (2019)
  └─ GPT-2 (2019)
       └─ GPT-3 (2020)          → 指令微調見 peft/
            └─ Chinchilla (2022) ← Scaling Law 修正
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
