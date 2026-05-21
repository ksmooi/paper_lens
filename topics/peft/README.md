# PEFT — Parameter-Efficient Fine-Tuning & Instruction Tuning

讓預訓練模型以最低的參數更新成本，適應新任務或遵循指令的訓練技術。

PEFT 解決的核心問題是：**全量微調一個大型預訓練模型成本極高**（需要儲存完整梯度與優化器狀態）。PEFT 方法只更新極少數參數（0.1%–1%），卻能達到接近全量微調的效果。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 低秩分解（LoRA） | 用低秩矩陣近似權重更新 | LoRA (2021)、LoRA+ |
| 量化 + PEFT | 在量化模型上做 PEFT | QLoRA (2023) |
| Adapter | 在層間插入小型可訓練模組 | Adapter (2019)、AdapterFusion |
| Prefix Tuning | 在輸入前綴加可訓練 soft token | Prefix-Tuning (2021)、P-Tuning v2 |
| Prompt Tuning | 僅學習輸入端的連續提示向量 | Prompt Tuning (2021) |
| 指令微調（IT） | 用指令-回應對資料全量或部分微調 | FLAN、Alpaca、WizardLM |
| 多工 PEFT | 單一 PEFT 模組適應多個任務 | AdapterFusion、MAM Adapter |
| VLM PEFT | 針對視覺語言模型的參數高效微調 | LLaVA PEFT、CLIP Adapter |
| Diffusion PEFT | 針對擴散模型的條件化微調 | ControlNet、LoRA for Diffusion |

## 不收錄於此

- 強化學習對齊方法（RLHF、DPO）→ [`rl/`](../rl/)
  （RLHF 雖包含微調步驟，但核心貢獻是 RL 訓練框架，不是參數效率）
- 量化推理（GPTQ、AWQ，非訓練時）→ [`efficiency/`](../efficiency/)
- 預訓練模型本身的架構設計 → [`llm/`](../llm/)

> **判斷原則**：論文的主要貢獻是「如何讓預訓練模型更有效率地適應新任務」→ 收錄於此，不論是全量微調的改進版還是 PEFT。

## 目錄結構

```
peft/
├── README.md                   # 本文件
├── 2019-02-adapter/            # Parameter-Efficient Transfer Learning (Adapter)
├── 2021-01-prefix-tuning/      # Prefix-Tuning
├── 2021-06-lora/               # LoRA: Low-Rank Adaptation
├── 2021-09-flan/               # Finetuned Language Models Are Zero-Shot Learners
├── 2022-03-alpaca/             # Self-Instruct / Alpaca
├── 2023-05-qlora/              # QLoRA: Efficient Finetuning of Quantized LLMs
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 方法比較

| 方法 | 可訓練參數佔比 | 修改位置 | 推理額外成本 |
|---|---|---|---|
| Full Fine-tuning | 100% | 全部權重 | 無 |
| Adapter | ~1–3% | 層間插入模組 | 有（串行） |
| Prefix-Tuning | ~0.1% | 輸入前綴 token | 有（KV cache 增大） |
| Prompt Tuning | ~0.01% | 輸入 embedding | 有（少量） |
| LoRA | ~0.1–1% | Q/V 權重低秩矩陣 | 無（可合併） |
| QLoRA | ~0.1–1% | 同 LoRA，但底層量化 | 量化推理成本 |

## 閱讀路徑建議

```
Full Fine-tuning（基準）
  └─ Adapter (2019)              ← PEFT 起點
  └─ Prefix-Tuning (2021)
  └─ LoRA (2021)                 ← 目前最廣泛使用的 PEFT
       └─ QLoRA (2023)           ← 量化 + LoRA
       └─ LoRA+ / VeRA (2024)    ← LoRA 改進

指令微調 (IT):
  FLAN (2021)                    ← 大規模指令資料
    └─ Self-Instruct (2022)
    └─ Alpaca (2023)             ← LLaMA + 指令微調
    └─ WizardLM (2023)
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
