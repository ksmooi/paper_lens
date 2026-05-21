# Agentic — LLM Agent 與 Agentic 框架

以 LLM 作為決策核心，透過規劃、工具使用、記憶與多 Agent 協作，完成複雜多步驟任務的系統與框架。

Agentic AI 的核心轉變是：LLM 不再只是「回答問題」，而是「**在環境中自主行動**」——觀察狀態、規劃行動、呼叫工具、反思結果、迭代修正。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 基礎 Agent 框架 | 賦予 LLM 工具使用與多步驟推理能力 | ReAct (2022)、Toolformer |
| 規劃與推理 | Agent 的任務分解與規劃策略 | CoT、Tree-of-Thought、LLM+P |
| 反思與自我修正 | Agent 透過反饋改善輸出 | Reflexion、Self-Refine |
| 記憶機制 | 短期/長期記憶的儲存與檢索 | MemGPT、Memory-Augmented LLM |
| 工具使用 | 呼叫外部 API、程式碼執行、瀏覽器 | Toolformer、HuggingGPT、WebGPT |
| 程式碼 Agent | 以撰寫與執行程式碼完成任務 | CodeAct、SWE-Agent、Devin |
| Multi-Agent | 多個 Agent 協作分工的框架 | AutoGen、CrewAI、MetaGPT |
| Agent Harness | 評估與測試 Agent 能力的框架 | AgentBench、SWE-Bench、WebArena |
| Computer Use | Agent 直接操控 GUI/桌面的能力 | Anthropic Computer Use、AppAgent |

## 不收錄於此

- 用 RL 訓練 Agent 的對齊方法（PPO、DPO）→ [`rl/`](../rl/)
- RAG 作為知識檢索元件本身 → [`rag/`](../rag/)
  （RAG 整合進 Agent 工具鏈的設計 → 收錄於此）
- LLM 推理能力本身的訓練方法 → [`llm/`](../llm/) 或 [`rl/`](../rl/)

> **判斷原則**：論文的核心是「如何讓 LLM 在環境中自主行動、使用工具、協作」→ 收錄於此。
> 「如何訓練 LLM 具備更強的推理能力」→ 收錄於 [`rl/`](../rl/) 或 [`llm/`](../llm/)。

## 目錄結構

```
agentic/
├── README.md                   # 本文件
├── 2022-10-react/              # ReAct: Synergizing Reasoning and Acting in LLMs
├── 2022-12-toolformer/         # Toolformer: Language Models Can Teach Themselves
├── 2023-03-reflexion/          # Reflexion: Language Agents with Verbal Reinforcement
├── 2023-05-hugginggpt/         # HuggingGPT / JARVIS
├── 2023-08-autogen/            # AutoGen: Enabling Next-Gen LLM Applications
├── 2023-10-swe-bench/          # SWE-bench: Can LLMs Resolve GitHub Issues?
├── 2024-01-codeact/            # CodeAct: Executable Code Actions Elicit Better
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
基礎能力:
  Chain-of-Thought (2022)       ← 推理能力基礎（見 llm/）
    └─ ReAct (2022)             ← 推理 + 行動的統一框架  ★ 入口
         └─ Reflexion (2023)    ← 加入自我反思
         └─ Tree-of-Thought (2023)

工具使用:
  Toolformer (2022)             ← 讓 LLM 學會呼叫工具
    └─ HuggingGPT (2023)        ← 以 ChatGPT 協調 AI 模型
    └─ WebGPT (2021)            ← 瀏覽器工具使用

程式碼 Agent:
  CodeAct (2024)                ← 以程式碼為行動空間
    └─ SWE-Agent (2024)
    └─ Devin / OpenDevin (2024)

Multi-Agent:
  AutoGen (2023)                ← 多 Agent 對話框架
  MetaGPT (2023)                ← 軟體開發 Multi-Agent
  CrewAI (2024)

評估 Harness:
  AgentBench (2023)
  SWE-Bench (2023)              ← 程式碼 Agent 基準
  WebArena (2023)               ← 網頁操作基準
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
