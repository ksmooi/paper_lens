# VLM — Vision-Language Models

視覺與語言的多模態融合模型，讓語言模型能夠「看見」並理解圖像。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 對比式學習 | 對齊圖文表示空間 | CLIP (2021)、ALIGN |
| 圖像描述 | 從圖像生成文字描述 | Show and Tell、BLIP |
| 視覺問答（VQA） | 依圖像回答自然語言問題 | VQA、Flamingo |
| 多模態 LLM | 讓 LLM 接受圖像輸入 | LLaVA、GPT-4V、Qwen-VL |
| 文生圖 | 依文字描述生成圖像 | DALL-E、Stable Diffusion XL |
| 文件理解 | 處理含圖文混排的文件 | DocVQA、Donut |

## 不收錄於此

- 純視覺骨幹網路（不含語言）→ [`vision/`](../vision/)
- 純語言模型（不含視覺輸入）→ [`llm/`](../llm/)
- 擴散式圖像生成架構原理 → [`diffusion/`](../diffusion/)
- VLM 的 PEFT 微調方法 → [`peft/`](../peft/)

## 目錄結構

```
vlm/
├── README.md               # 本文件
├── 2021-02-clip/           # Learning Transferable Visual Models
├── 2022-04-flamingo/       # Flamingo: a Visual Language Model
├── 2023-04-llava/          # LLaVA
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
CLIP (2021)              ← 對比學習奠基
  └─ BLIP (2022) → BLIP-2 (2023)
  └─ DALL-E 2 (2022)    ← 圖像生成

Flamingo (2022)          ← Few-shot VLM
  └─ LLaVA (2023)
       └─ LLaVA-1.5 / LLaVA-NeXT (2024)
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
