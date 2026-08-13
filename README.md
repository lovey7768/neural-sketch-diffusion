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



## 🛠️ Tool & Technology Stack

| Tool / Framework | Component Type | Function & Purpose |
| :--- | :--- | :--- |
| **PyTorch (`torch`)** | Deep Learning Framework | Handles CUDA tensor execution and model weight loading on the GPU. |
| **Hugging Face `diffusers`** | Model Orchestration | Manages the Stable Diffusion pipeline execution loop and memory slicing. |
| **`sd-controlnet-canny`** | Neural Network Adapter | Adds spatial conditioning layers to Stable Diffusion without retraining the base model. |
| **Stable Diffusion v1.5** | Latent Diffusion Model | Generates photorealistic RGB textures, materials, and lighting effects. |
| **OpenCV (`cv2`)** | Computer Vision | Performs low-level pixel manipulation and threshold edge detection. |
| **`UniPCMultistepScheduler`** | ODE Fast-Sampling Engine | Reduces required inference steps from 50 to 20, cutting generation time by 60%. |
| **Gradio** | Web UI Engine | Serves a shareable, real-time interactive browser dashboard directly from Google Colab. |
