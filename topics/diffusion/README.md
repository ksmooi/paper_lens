# Diffusion — 擴散模型

以擴散過程為核心的生成模型，是當前圖像、影片、音訊生成的主流範式。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 基礎擴散模型 | DDPM 及其理論基礎 | DDPM (2020)、Score Matching |
| 加速採樣 | 減少去噪步驟的方法 | DDIM (2020)、DPM-Solver |
| 條件式生成 | 依條件（文字、類別）引導生成 | Classifier Guidance、CFG |
| 潛在擴散 | 在低維潛在空間做擴散 | Latent Diffusion、Stable Diffusion |
| 影片生成 | 時序一致的影片擴散模型 | Video Diffusion、Sora |
| 擴散 Transformer | 以 Transformer 取代 U-Net | DiT (2022) |
| 流匹配 | 與擴散相關的 ODE 流方法 | Flow Matching、Rectified Flow |

## 不收錄於此

- GAN 系列生成模型 → [`vision/`](../vision/)
- 文生圖的 CLIP 文字理解部分 → [`vlm/`](../vlm/)
- 擴散模型的 PEFT 微調（ControlNet、LoRA for Diffusion）→ [`peft/`](../peft/)

## 目錄結構

```
diffusion/
├── README.md                   # 本文件
├── 2020-06-ddpm/               # Denoising Diffusion Probabilistic Models
├── 2020-10-ddim/               # Denoising Diffusion Implicit Models
├── 2021-12-latent-diffusion/   # High-Resolution Image Synthesis with LDMs
├── 2022-12-dit/                # Scalable Diffusion Models with Transformers
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
Score Matching (2019)
  └─ DDPM (2020)              ← 擴散模型起點
       └─ DDIM (2020)         ← 快速採樣
       └─ CFG (2022)          ← 無分類器引導
       └─ LDM (2021)          ← Stable Diffusion 基礎
            └─ SDXL (2023)

DiT (2022)                    ← U-Net → Transformer
  └─ Sora (2024)
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
