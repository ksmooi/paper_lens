# Harness — Agent 外部控制層與執行框架

Harness Engineering 關注的是：如何在 LLM Agent 外部設計一層可控、可觀測、可驗證的執行框架，讓 Agent 不只是「會回答問題」，而是能在真實任務中穩定完成工作。

如果說 Agentic AI 的核心是讓 LLM 能夠「**在環境中自主行動**」，那麼 Harness Engineering 的核心就是讓這個行動過程變得「**可控、可驗證、可維護、可擴充**」。

Harness 不只是 prompt，也不只是工具清單，而是包在模型外部的一整層工程系統：

```text
Harness =
  instructions
  + context assembly
  + tool routing
  + state management
  + memory
  + task workflow
  + verification gates
  + runtime control
  + human approval
  + evaluation framework
```

---

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| Harness Engineering 基礎 | 將 Agent 外部控制層視為獨立工程設計對象 | Natural-Language Agent Harnesses、Harness Engineering for Language Agents |
| Scaffolding | 為 Agent 任務建立流程支架、步驟限制與執行規則 | AutoHarness、Scalable Agent Scaffolding for Real-World Codebases |
| Context Engineering | 控制 Agent 在每一步看到什麼資訊、如何組裝 context | Building AI Coding Agents for the Terminal |
| Runtime Orchestration | 控制 Agent 的 observe-plan-act loop、tool call、retry、rollback | Natural-Language Agent Harnesses、OpenDevin / OpenHands |
| Tool Governance | 管理工具權限、工具選擇、工具輸入輸出與安全邊界 | Toolformer、HuggingGPT、MCP-based Agent Systems |
| Memory / State Management | 管理短期狀態、長期記憶、任務進度與 artifacts | MemGPT、SWE-Bench-CL |
| Verification Gates | 在 Agent 完成任務前加入測試、review、eval、lint、security check | UTBoost、SWE-Bench、AgentBench |
| Evaluation Harness | 建立可重複評估 Agent 能力的測試環境與 benchmark | SWE-Bench、WebArena、AgentBench、OSWorld |
| Coding Agent Harness | 針對程式碼 Agent 的 repo 理解、修改、測試、PR review 流程設計 | SWE-agent、OpenHands、Building AI Coding Agents for the Terminal |
| Human-in-the-loop | 在高風險步驟加入人工確認、審核與 approval gate | HumanEval-style Review、Agent Safety Evaluation |

---

## 不收錄於此

- Agent 的基本行為模式，例如 ReAct、Reflexion、AutoGen → [`agentic/`](../agentic/)
- RAG 作為知識檢索元件本身 → [`rag/`](../rag/)
  （RAG 如何被放進 Agent Harness 作為工具或 context source → 可收錄於此）
- 用 RL 訓練模型的對齊方法，例如 PPO、DPO、GRPO → [`rl/`](../rl/)
- LLM 模型架構與推理能力本身，例如 Transformer、CoT、MoE → [`llm/`](../llm/)
- 單純的 MLOps / Model Serving / Inference Optimization → [`mlops/`](../mlops/) 或 [`inference/`](../inference/`)

> **判斷原則**：論文的核心是「如何設計 Agent 外部控制層、執行流程、工具治理、context 管理、驗證機制」→ 收錄於此。  
> 「如何讓 LLM 具備 Agent 行為」→ 收錄於 [`agentic/`](../agentic/)。  
> 「如何訓練 LLM 具備更強推理能力」→ 收錄於 [`llm/`](../llm/) 或 [`rl/`](../rl/)。

---

## 目錄結構

```text
harness/
├── README.md                                      # 本文件
├── 2025-06-utboost/                              # UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench
├── 2025-07-swe-bench-cl/                         # SWE-Bench-CL: Continual Learning for Coding Agents
├── 2025-08-reliable-weak-to-strong-monitoring/   # Reliable Weak-to-Strong Monitoring of LLM Agents
├── 2025-12-scalable-agent-scaffolding/           # Scalable Agent Scaffolding for Real-World Codebases
├── 2026-03-autoharness/                          # AutoHarness: Automatically Synthesizing a Code Harness
├── 2026-03-natural-language-agent-harnesses/     # Natural-Language Agent Harnesses
├── 2026-03-building-ai-coding-agents-terminal/   # Building AI Coding Agents for the Terminal
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

---

## 閱讀路徑建議

```text
基礎概念:
  Natural-Language Agent Harnesses (2026)       ← Harness Engineering 入口
    └─ Harness Engineering for Language Agents  ← Control / Agency / Runtime 概念
    └─ AutoHarness (2026)                       ← 自動合成 code harness

Coding Agent Harness:
  SWE-Bench (2023)                              ← 程式碼 Agent 評估基準（見 agentic/）
    └─ SWE-agent (2024)                         ← Agent-Computer Interface（見 agentic/）
    └─ Building AI Coding Agents for the Terminal (2026)
    └─ Scalable Agent Scaffolding for Real-World Codebases (2025)

Evaluation / Verification:
  AgentBench (2023)                             ← 通用 Agent benchmark（見 agentic/）
    └─ SWE-Bench (2023)
    └─ UTBoost (2025)                           ← 強化 coding agent 測試可靠性
    └─ WebArena / OSWorld                       ← Web / Computer Use 評估環境

Runtime / Monitoring:
  ReAct (2022)                                  ← observe-reason-act 基礎（見 agentic/）
    └─ Runtime Orchestration
    └─ Reliable Weak-to-Strong Monitoring (2025)
    └─ Human-in-the-loop Approval Gates

Memory / Continual Learning:
  MemGPT                                        ← Agent 記憶機制（見 agentic/）
    └─ SWE-Bench-CL (2025)                      ← coding agent continual learning
```

---

## Harness Engineering 的核心問題

Harness Engineering 主要回答以下問題：

```text
1. Agent 在每一步應該看到哪些 context？
2. Agent 可以使用哪些工具？
3. 工具呼叫前後要不要檢查？
4. 任務是否需要先 plan 再 execute？
5. 長任務中如何保存 state？
6. 發生錯誤時如何 retry、debug、rollback？
7. 什麼情況下需要 human approval？
8. 如何驗證 Agent 的輸出是正確的？
9. 如何記錄 artifacts 讓結果可追蹤？
10. 如何讓 Agent 行為可以被測試、比較與改進？
```

---

## Harness 與 Agentic AI 的差異

| 面向 | Agentic AI | Harness Engineering |
|---|---|---|
| 核心問題 | 如何讓 LLM 自主行動 | 如何控制與驗證 LLM Agent 的行動 |
| 關注對象 | Agent 行為模式 | Agent 外部控制層 |
| 典型能力 | 規劃、反思、工具使用、多 Agent 協作 | context 管理、runtime、tool governance、verification gates |
| 代表論文 | ReAct、Reflexion、AutoGen、SWE-agent | Natural-Language Agent Harnesses、AutoHarness、UTBoost |
| 工程重點 | 讓 Agent 能做事 | 讓 Agent 穩定、可靠、安全地做事 |
| 類比 | Agent 是工作者 | Harness 是工作流程、規則、監控與驗證系統 |

簡單來說：

```text
Agentic AI 關心：
  Agent 如何思考與行動？

Harness Engineering 關心：
  如何讓 Agent 的思考與行動被正確約束、執行、觀測與驗證？
```

---

## 典型 Harness 架構

```text
User Request
  ↓
Task Intake
  ↓
Requirement Clarification
  ↓
Context Assembly
  ↓
Planning Gate
  ↓
Execution Loop
  ├─ Observe State
  ├─ Select Tool
  ├─ Call Tool
  ├─ Inspect Result
  ├─ Update State
  └─ Retry / Debug if needed
  ↓
Verification Gate
  ├─ Run Tests
  ├─ Run Eval
  ├─ Code Review
  ├─ Security Check
  └─ Acceptance Criteria Check
  ↓
Human Approval Gate
  ↓
Final Artifact / Report
```

這也是 Claude Code、Codex、Cursor、OpenHands、Devin 這類 coding agent 背後非常重要的工程觀念。

---

## Practical Applications

Harness Engineering 特別適合應用在以下場景：

```text
- AI Coding Assistant
- Repo Understanding Agent
- PR Review Agent
- Bug Triage Agent
- Migration Agent
- RAG Agent
- Deep Research Agent
- Data Analysis Agent
- DevOps Automation Agent
- Customer Support Agent
- Enterprise Workflow Agent
```

在企業環境中，Harness Engineering 的價值不只是提升 Agent 成功率，也包括：

```text
- 降低幻覺與錯誤工具使用
- 提升任務可重現性
- 強化安全與權限控管
- 建立 review / approval 流程
- 讓 Agent 輸出可被測試
- 讓 Agent 行為可觀測
- 讓 Agent 系統可持續改進
```

---

## 與實務工具的對應

| Harness 元件 | 實務工具 / 框架 |
|---|---|
| Workflow Orchestration | LangGraph、AutoGen、CrewAI |
| Coding Agent Runtime | Claude Code、Codex、OpenHands、Devin |
| Context Engineering | Claude Code CLAUDE.md、Codex instructions、Cursor rules |
| Tool Interface | MCP、Function Calling、OpenAPI Tools |
| Memory / State | LangGraph Checkpointer、Vector DB、Knowledge Graph |
| Verification | pytest、Promptfoo、DeepEval、Ragas、SWE-Bench |
| Code Review Gate | Codex Review、Claude Code Review、GitHub PR Review |
| Observability | LangSmith、OpenTelemetry、structured logs |
| Human Approval | HITL workflow、manual approval gate、policy check |

---

## 核心閱讀清單

| 論文 | 年份 | 重點 |
|---|---:|---|
| Natural-Language Agent Harnesses | 2026 | 將 Agent Harness 外部化為自然語言規格 |
| AutoHarness | 2026 | 自動合成 code harness，提高 agent 表現 |
| Building AI Coding Agents for the Terminal | 2026 | terminal coding agent 的 scaffolding、harness、context engineering |
| Harness Engineering for Language Agents | 2026 | 從 control、agency、runtime 角度定義 harness layer |
| Scalable Agent Scaffolding for Real-World Codebases | 2025 | 大型真實 codebase 的 agent scaffolding |
| UTBoost | 2025 | 強化 SWE-Bench 測試與 verification reliability |
| Reliable Weak-to-Strong Monitoring of LLM Agents | 2025 | Agent runtime monitoring 與 monitor scaffolding |
| SWE-Bench-CL | 2025 | Coding agent continual learning 與 memory evaluation |

---

## Learning Goals

讀完本目錄後，應該能夠回答：

```text
1. Harness Engineering 與 Agentic AI 有什麼差異？
2. Agent Harness 包含哪些核心元件？
3. Context Engineering 在 Agent 系統中扮演什麼角色？
4. Scaffolding 如何提升 Agent 的穩定性？
5. Runtime Orchestration 如何控制 Agent 的執行流程？
6. Tool Governance 如何降低工具誤用風險？
7. Verification Gate 如何避免 Agent 產生錯誤結果？
8. Coding Agent 為什麼需要 SWE-Bench、UTBoost 這類 evaluation harness？
9. Human-in-the-loop 在企業 Agent 系統中如何設計？
10. 如何把 Harness Engineering 應用到 Claude Code、Codex、LangGraph 或 MCP 系統？
```

---

## 建議研究方向

```text
1. Agent Harness as Runtime
   - 研究 Agent 執行流程、state、tool call、retry、rollback

2. Context Engineering
   - 研究如何組裝 task context、repo context、memory context、tool context

3. Evaluation Harness
   - 研究如何設計 benchmark、eval cases、test suites、acceptance criteria

4. Coding Agent Harness
   - 研究 repo reading、code editing、test execution、PR review、branch workflow

5. Human-in-the-loop Harness
   - 研究 approval gates、policy checks、high-risk tool confirmation

6. Multi-Agent Harness
   - 研究 controller / worker / reviewer / critic 的協作流程

7. Enterprise Agent Governance
   - 研究權限控管、審計紀錄、安全邊界、可觀測性與合規需求
```

---