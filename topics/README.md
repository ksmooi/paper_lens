# Topics — 文章分類目錄

這裡是 [Paper Lens](../README.md) 所有論文解讀文章的存放處,依研究主題分類。

每個類別目錄下,每篇文章是一個獨立資料夾,命名格式為 `{YYYY-MM}-{paper-slug}/`。年月對應**論文 arxiv 第一版發表時間**,不是文章撰寫時間。

```
topics/
└── {topic-slug}/
    └── {YYYY-MM}-{paper-slug}/
        ├── README.md     # 文章主體(繁體中文,700–1000 行)
        ├── meta.yaml     # 結構化 metadata(種子論文、dependency papers、知識點)
        ├── papers.bib    # 涵蓋的論文 BibTeX
        └── assets/       # 圖片、示意圖(選用)
```

---

## 14 個主題類別

### 模型架構

| 類別 | 說明 | 典型論文 |
|------|------|----------|
| [llm](./llm/) | 大型語言模型的架構、預訓練策略與 Scaling Law | Transformer、BERT、GPT-3、Chinchilla |
| [vision](./vision/) | 電腦視覺:骨幹網路、物件偵測、語意分割、自監督學習 | ResNet、ViT、DETR、MAE |
| [vlm](./vlm/) | 視覺語言多模態:圖文對齊、多模態理解與生成 | CLIP、Flamingo、LLaVA |
| [diffusion](./diffusion/) | 擴散生成模型:去噪、潛在擴散、流匹配 | DDPM、Stable Diffusion、DiT |
| [moe](./moe/) | Mixture of Experts:稀疏激活、路由機制、負載均衡 | Switch Transformer、Mixtral、DeepSeekMoE |

### 核心機制

| 類別 | 說明 | 典型論文 |
|------|------|----------|
| [attention](./attention/) | Attention 機制本身的設計與改進:效率化、位置編碼、KV Cache | FlashAttention、RoPE、GQA |
| [normalization](./normalization/) | 正規化方法:Batch、Layer、Instance、Group、RMS | BatchNorm、LayerNorm、RMSNorm |
| [embedding](./embedding/) | 嵌入與表示學習:詞向量、句向量、對比學習、多模態嵌入 | Word2Vec、Sentence-BERT、SimCSE |
| [optimization](./optimization/) | 優化器與訓練策略:梯度方法、學習率排程、混合精度 | Adam、AdamW、Adafactor |

### 效率與適應

| 類別 | 說明 | 典型論文 |
|------|------|----------|
| [efficiency](./efficiency/) | 推理與訓練效率:量化、剪枝、蒸餾、推理加速、分散式訓練 | GPTQ、AWQ、vLLM、ZeRO |
| [peft](./peft/) | 參數高效微調與指令微調:低秩分解、Adapter、Prompt Tuning | LoRA、QLoRA、Alpaca、FLAN |

### 知識與決策

| 類別 | 說明 | 典型論文 |
|------|------|----------|
| [rag](./rag/) | 檢索增強生成:稠密檢索、自適應 RAG、圖譜增強 | DPR、Self-RAG、GraphRAG |
| [rl](./rl/) | 強化學習與對齊:策略優化、RLHF、直接偏好優化、推理強化 | PPO、InstructGPT、DPO、DeepSeek-R1 |
| [agentic](./agentic/) | LLM Agent：推理與行動、工具使用、反思、自我改進、Multi-Agent 協作 | ReAct、Reflexion、AutoGen、SWE-agent |
| [harness](./harness/) | Harness Engineering：Agent 外部控制層、Scaffolding、Context Engineering、Runtime、Evaluation Gate | Natural-Language Agent Harnesses、AutoHarness、Building AI Coding Agents for the Terminal |

---

## 類別邊界速查

同一篇論文有時橫跨多個主題,下表列出常見的歸屬決策:

| 如果論文的核心貢獻是… | 歸屬類別 |
|-----------------------|----------|
| LLM 預訓練架構或 Scaling Law | `llm` |
| LoRA、QLoRA、指令微調等微調方法 | `peft` |
| RLHF、DPO 等對齊訓練方法 | `rl` |
| ReAct、工具使用、Multi-Agent 框架 | `agentic` |
| Attention 機制本身的改進 | `attention` |
| 量化、剪枝、推理加速 | `efficiency` |
| 對比學習的嵌入訓練方法 | `embedding` |
| 完整的視覺語言多模態系統 | `vlm` |
| ControlNet、Diffusion LoRA 等微調方法 | `peft` |

跨主題論文的完整決策規則見 [`AGENT_GUIDE.md`](../AGENT_GUIDE.md#常見跨主題判斷參考)。

---

## 推薦的閱讀起點

不知道從哪裡開始?幾條有主題脈絡的路徑:

**從 Transformer 出發,往 LLM 走**
`attention/` → `normalization/` → `llm/` → `peft/` → `rl/`

**從視覺模型出發,往多模態走**
`vision/` → `embedding/` → `vlm/` → `diffusion/`

**從效率問題出發**
`efficiency/` → `peft/` → `moe/`

**從 Agent 出發**
`llm/` → `rag/` → `agentic/`

完整的跨主題知識圖譜見 [`_index/reading-map.md`](../_index/reading-map.md)。

---

> 本目錄由 Hermes Agent 協作維護。新增規則見 [`AGENT_GUIDE.md`](../AGENT_GUIDE.md)。
