# RL — Reinforcement Learning & Alignment

強化學習在深度學習中的應用，以及讓語言模型對齊人類偏好的訓練方法。

此類別聚焦於**訓練時的 RL 方法與對齊技術**。以 LLM 作為決策核心的 Agent 框架與 Harness 請見 [`agentic/`](../agentic/)。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 深度強化學習基礎 | 結合深度學習與 RL 的核心演算法 | DQN、PPO、SAC |
| RLHF | 從人類反饋學習獎勵模型 | InstructGPT、Constitutional AI |
| 獎勵模型 | 訓練評估模型輸出品質的模型 | Reward Model、PRM、ORM |
| 直接偏好優化 | 無需獎勵模型的對齊方法 | DPO、SimPO、ORPO |
| 過程監督 | 對推理步驟給予逐步獎勵 | Let's Verify Step by Step |
| 推理強化 | 用 RL 增強模型的推理能力 | DeepSeek-R1、GRPO |

## 不收錄於此

- 純 LLM 指令微調（無 RL 成分）→ [`peft/`](../peft/)
- LLM Agent 框架、工具使用、Multi-Agent → [`agentic/`](../agentic/)
- RAG 與工具使用（不涉及獎勵訓練）→ [`rag/`](../rag/)

> **判斷原則**：論文的核心是「用 RL 訓練模型」→ 收錄於此。
> 「用訓練好的 LLM 做 Agent 決策」→ 收錄於 [`agentic/`](../agentic/)。

## 目錄結構

```
rl/
├── README.md                   # 本文件
├── 2015-02-dqn/                # Human-level control through deep RL
├── 2017-07-ppo/                # Proximal Policy Optimization
├── 2022-03-instructgpt/        # Training LMs to follow instructions (RLHF)
├── 2023-05-dpo/                # Direct Preference Optimization
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
DQN (2015)
  └─ A3C (2016) → PPO (2017)    ← RLHF 訓練的標準 RL 演算法

RLHF:
  InstructGPT (2022)
    └─ Constitutional AI (2022)
    └─ DPO (2023)               ← 移除 RL、直接優化偏好
         └─ SimPO / ORPO (2024)

推理強化:
  Process Reward Model (2023)
    └─ DeepSeek-R1 (2025)
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
