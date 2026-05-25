# Paper Lens 📖🔍

用一篇文章的長度，把一篇 AI 論文講清楚。

這是我的個人 AI 論文閱讀筆記庫，涵蓋 LLM、VLM、Attention、擴散模型、正規化、PEFT、Agentic AI、RAG 等領域。每篇文章圍繞一篇**種子論文**展開，並串接最相關的 dependency papers，試圖把該主題的**核心知識點**講透。

文章以繁體中文撰寫，主要是寫給未來的自己看，但歡迎任何路過的人閱讀、討論、糾錯。

取名 Paper Lens，是因為閱讀論文這件事很像用放大鏡看東西——不是看得更廣，而是看得更深、更清楚。每篇文章試圖把一個主題的核心概念放大到足夠清楚，讓人真正「看見」它在做什麼，而不只是知道它存在。

---

## 為什麼有這個 repo

讀論文最大的痛點不是讀不懂單篇,而是讀完之後**留不下東西**。

這個 repo 試圖解決三件事:

1. **強迫輸出**——讀完不寫,等於沒讀
2. **建立關聯**——一篇論文很少獨立存在,得把上下游串起來
3. **沉澱知識點**——比起記得「這篇論文做了什麼」,更重要的是記得「這個主題有哪些核心問題」

每篇文章背後都有一個固定的工作流(見下方),由我和一個叫 Hermes 的 agent 協作完成。

---

## 目錄結構

```
paper-lens/
├── topics/                    # 文章主體,依主題分類
│   ├── llm/                   # 大型語言模型架構與預訓練
│   ├── normalization/         # 正規化方法(BN、LN、RMSNorm 等)
│   ├── vision/                # 電腦視覺模型(ResNet、ViT、DETR 等)
│   ├── vlm/                   # 視覺語言多模態模型
│   ├── attention/             # Attention 機制與位置編碼
│   ├── diffusion/             # 擴散模型(DDPM、LDM、DiT 等)
│   ├── efficiency/            # 推理效率、量化、剪枝、分散式訓練
│   ├── embedding/             # 嵌入與表示學習
│   ├── moe/                   # Mixture of Experts
│   ├── optimization/          # 優化器與訓練策略
│   ├── peft/                  # 參數高效微調與指令微調
│   ├── rag/                   # 檢索增強生成
│   ├── rl/                    # 強化學習與對齊(RLHF、DPO 等)
│   ├── agentic/               # LLM Agent、工具使用、程式碼 Agent、Multi-Agent
│   └── harness/               # Agent 外部控制層、Scaffolding、Runtime、Evaluation Harness
├── _templates/                # 文章與 metadata 範本
├── _index/                    # 自動生成的索引(依日期/主題/論文)
├── scripts/                   # 索引生成、文章驗證腳本
├── AGENT_GUIDE.md             # 給 Hermes Agent 看的寫入規則
├── CONTRIBUTING.md            # 給人類貢獻者的指南
└── README.md                  # 你正在看的這份
```

每個主題下,每篇文章是一個獨立資料夾,命名格式為 `YYYY-MM-{paper-slug}/`,其中年月對應**論文發表時間**(不是文章撰寫時間)。例如 DPO 論文是 2023 年 5 月發表的:

```
topics/llm/2023-05-dpo/
├── README.md       # 文章主體
├── meta.yaml       # 結構化 metadata
├── papers.bib      # 涵蓋的論文(BibTeX)
└── assets/         # 圖片、示意圖
```

---

## 已收錄的主題

> 索引由 `scripts/generate_index.py` 自動產生,完整列表見 [`_index/by-topic.md`](./_index/by-topic.md)。

| 主題 | 說明 |
|------|------|
| [LLM](./topics/llm/) | 大型語言模型:架構、預訓練、Scaling Law |
| [Normalization](./topics/normalization/) | 正規化方法:BatchNorm、LayerNorm、RMSNorm |
| [Vision](./topics/vision/) | 電腦視覺:骨幹網路、物件偵測、分割、自監督 |
| [VLM](./topics/vlm/) | 視覺語言模型:多模態理解與生成 |
| [Attention](./topics/attention/) | Attention 機制:效率化、位置編碼、KV Cache |
| [Diffusion](./topics/diffusion/) | 擴散模型:DDPM、LDM、DiT、流匹配 |
| [Efficiency](./topics/efficiency/) | 推理效率:量化、剪枝、蒸餾、分散式訓練 |
| [Embedding](./topics/embedding/) | 嵌入與表示學習:詞向量、句向量、對比學習 |
| [MoE](./topics/moe/) | Mixture of Experts:路由機制、負載均衡 |
| [Optimization](./topics/optimization/) | 優化器與訓練策略:Adam、混合精度 |
| [PEFT](./topics/peft/) | 參數高效微調:LoRA、QLoRA、Adapter、指令微調 |
| [RAG](./topics/rag/) | 檢索增強生成:稠密檢索、自適應 RAG |
| [RL](./topics/rl/) | 強化學習與對齊:PPO、RLHF、DPO、推理強化 |
| [Agentic](./topics/agentic/) | LLM Agent：規劃、反思、工具使用、程式碼 Agent、Multi-Agent |
| [Harness](./topics/harness/) | Harness Engineering：Agent 外部控制層、Scaffolding、Context Engineering、Runtime Orchestration、Tool Governance、Verification Gate、Evaluation Harness |

新主題會隨閱讀範圍擴張而新增。

---

## 文章的寫作流程

每篇文章都遵循同一套流程,由 Hermes Agent 自動執行:

1. **種子論文識別**——從 arxiv 找出該主題的原始論文
2. **Dependency papers 蒐集**——從引用網路中挑出最相關的至多 3 篇
3. **知識點歸納**——從種子論文中抽取該主題的核心知識點
4. **逐篇精讀**——依知識點清單,擷取每篇論文的原理、方法、實驗結果
5. **文章撰寫**——整合成一篇繁體中文文章(700–1000 行),涵蓋每篇論文在各知識點上的貢獻
6. **發布到 GitHub**——建立新 branch、開 PR、不直接 push 到 main

Agent 在執行過程中遇到模糊情境時會自行判斷並繼續,判斷依據會記錄在 PR 描述中以便事後審閱。詳細規則見 [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)。

---

## 文章的結構

每篇文章大致遵循這個骨架(範本見 [`_templates/article-template.md`](./_templates/article-template.md)):

- **TL;DR**——三句話講完
- **背景與動機**——這個方法在解決什麼問題
- **核心知識點**——本篇主題的關鍵概念清單
- **方法詳解**——依知識點逐一展開,串接所有相關論文的貢獻
- **實驗結果**——數據、對比、消融
- **延伸閱讀**——dependency papers 與後續發展

---

## 如何閱讀

幾種建議的進入方式:

- **按主題挖**:點進 `topics/{主題}/` 看該領域的所有文章
- **按時間追**:看 [`_index/by-date.md`](./_index/by-date.md) 了解 AI 領域的時間線
- **按論文反查**:看 [`_index/by-paper.md`](./_index/by-paper.md) 找出某篇論文被哪些文章引用
- **看知識圖譜**:看 [`_index/reading-map.md`](./_index/reading-map.md) 了解論文間的依賴關係

---

## 貢獻與回饋

這是個人知識庫,內容由我主導,但歡迎:

- **發 Issue 指出錯誤**——任何理解錯誤、實驗數據引用錯誤都歡迎指正
- **發 Issue 推薦論文**——某個主題覺得我漏了重要論文,請告訴我
- **發 PR 改進文字**——錯字、語句不通、補充說明都歡迎

如果是大幅改寫或新增整篇文章的 PR,請先開 Issue 討論。詳細規則見 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

---

## 一些自我提醒

- 文章是寫給**未來的自己**看的,所以該寫的細節不要省
- 抄論文等於沒讀過——所有解讀必須是自己的理解,引用論文原文要少且精
- 知識點優先於論文——文章的結構是知識點驅動,不是論文章節驅動
- 對齊原文——不確定的地方寧可標記「TODO」,也不要瞎掰

---

## License

文章內容採 [CC BY 4.0](./LICENSE),程式碼採 MIT License。引用論文遵循原論文授權。
