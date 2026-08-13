# neural-sketch-diffusion
A guided image-generation pipeline built with PyTorch, Stable Diffusion v1.5, and ControlNet (Canny edge detection). This project transforms hand-drawn sketches or low-fidelity input photos into high-resolution, photorealistic, or stylized digital artwork using structural edge conditioning.
# 🎨 Neural Sketch-to-Art Diffusion Engine

An edge-conditioned Generative AI pipeline that transforms rough hand-drawn line sketches into photorealistic artwork using **ControlNet**, **Stable Diffusion v1.5**, and **PyTorch**. The entire system runs locally on free GPU hardware (Google Colab T4) with zero reliance on paid third-party APIs.

---

## 🎯 Project Motive

Standard text-to-image diffusion models often lack spatial control, making it difficult to generate images with exact subject poses, architectural bounds, or strict silhouettes. 

This project bridges **Computer Vision (OpenCV)** and **Generative Deep Learning (Diffusion Models)**. By conditioning the diffusion process with a real-time structural Canny edge map, the model retains the exact geometry of the original line sketch while synthesizing realistic textures, lighting, and depth from text prompts.

---

## ⚙️ System Workflow

```text
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────────┐     ┌─────────────────────┐
│ Hand-Drawn      │ ──► │ OpenCV Canny     │ ──► │ ControlNet             │ ──► │ Stable Diffusion    │
│ Line Sketch /   │     │ Edge Extraction  │     │ Spatial Guidance       │     │ UNet Denoising      │
│ Input Image     │     │ (Binary Matrix)  │     │ Conditioning Adapter   │     │ (20 UniPC Steps)    │
└─────────────────┘     └──────────────────┘     └────────────────────────┘     └─────────────────────┘
                                                                                           │
                                                                                           ▼
                                                                                ┌─────────────────────┐
                                                                                │ Rendered 8K         │
                                                                                │ Photorealistic Art  │
                                                                                └─────────────────────┘
