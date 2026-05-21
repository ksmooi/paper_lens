# Vision — 電腦視覺模型

電腦視覺領域的核心模型架構，涵蓋從 CNN 時代到 Transformer 時代的演進。

## 收錄範圍

| 子主題 | 說明 | 典型論文 |
|---|---|---|
| 骨幹網路（Backbone） | 用於特徵萃取的通用視覺架構 | ResNet、ViT、Swin Transformer |
| 物件偵測 | 定位並分類圖像中的物件 | YOLO 系列、DETR、Faster R-CNN |
| 語意分割 | 逐像素的類別預測 | U-Net、DeepLab、Mask2Former |
| 影像生成（GAN） | 對抗式生成模型 | GAN、StyleGAN、BigGAN |
| 自監督視覺學習 | 不依賴標籤的視覺表示學習 | MAE、DINO、MoCo |
| 3D 視覺 | 點雲、深度估計、NeRF | PointNet、NeRF |

## 不收錄於此

- 視覺語言多模態模型（CLIP、LLaVA 等）→ [`vlm/`](../vlm/)
- 影像生成的擴散模型（Stable Diffusion 等）→ [`diffusion/`](../diffusion/)
- 視覺 Transformer 的 Attention 機制分析 → [`attention/`](../attention/)

## 目錄結構

```
vision/
├── README.md               # 本文件
├── 2015-12-resnet/         # Deep Residual Learning
├── 2020-10-vit/            # An Image is Worth 16x16 Words
├── 2020-05-detr/           # End-to-End Object Detection with Transformers
└── ...
```

目錄命名格式：`{YYYY-MM}-{topic-slug}/`，日期取論文 arxiv 首次提交年月。

## 閱讀路徑建議

```
AlexNet (2012)
  └─ ResNet (2015)        ← 殘差連接里程碑
       └─ EfficientNet (2019)
  └─ Faster R-CNN (2015)  ← 兩階段偵測
       └─ DETR (2020)     ← Transformer 進入偵測

ViT (2020)                ← Transformer 進入視覺
  └─ Swin Transformer (2021)
  └─ MAE (2021)           ← 自監督 ViT
```

---

> 本目錄由 [paper_lens](../../README.md) Hermes Agent 自動維護。
