# 🎯 Multi-View Depth Consistent Image Generation

**Architectural Design Automation from Shoebox Models using Generative AI**  
📄 [IEEE Paper (2024)](https://ieeexplore.ieee.org/document/10418885)  
🎬 [Demo Video](https://www.youtube.com/watch?v=0H19NyXDRJE)

---

## 🔍 Overview

This project presents a **three-stage generation framework** that converts simplified **shoebox models** into multi-view, photorealistic architectural renderings using **generative AI**.

We enhance **ControlNet** to support multi-view inputs, apply **monocular depth estimation**, and fuse RGB–depth information using a novel **depth-aware consistency mechanism**. The framework targets early-stage university building design and aims to improve both **design visualization** and **geometric coherence**.

<div align="center">
  <img src="https://user-images.githubusercontent.com/demo-image-framework.jpg" width="600" alt="Framework Overview">
  <p><em>Three-stage pipeline for architectural image generation (replace with actual image)</em></p>
</div>

---

## 🏗️ Core Contributions

- ⚙️ A **multi-view diffusion architecture** based on ControlNet for generating coherent architectural images.
- 🎨 An **image-space consistency loss** combining style, structure, and viewpoint alignment.
- 🧠 A **depth-aware fusion module** using MiDaS and MVD-Fusion to enhance multi-view 3D realism.
- 🏫 A custom **university building dataset**: 12,600 shoebox views paired with 12,600 photorealistic designs.

---

## 🧪 Method Summary

### Stage 1: Multi-View Generation
> Shoebox inputs → Multi-view ControlNet → Architectural images with color, texture, structure.

### Stage 2: Depth Estimation
> Generated views → MiDaS → View-specific depth maps.

### Stage 3: Fusion & Consistency Refinement
> Images + Depth → MVD-Fusion → Style-aligned & structurally coherent renderings.

---

## 📦 Dataset

- **Shoebox models**: 210 models representing simplified university buildings.
- **Multi-view images**: Rendered with Blender (60 angles/model).
- **Total images**: 25,200 paired images (RGB and ground truth).

---

## 📊 Results

| Evaluation Metric      | Reconstruction | Generation |
| ---------------------- | -------------- | ---------- |
| Structural Integrity   | 3.57           | 3.55       |
| Structural Consistency | 3.53           | 3.43       |
| Detail Integrity       | 3.56           | 3.35       |
| Detail Consistency     | 3.39           | 3.28       |
| Visual Aesthetics      | 3.17           | 3.03       |
| Practicality           | 3.42           | 3.37       |

---

## 📚 Citation
If you use this work, please cite:

@inproceedings{du2024multiview,
  title={Multi-View Depth Consistent Image Generation Using Generative AI Models: Application on Architectural Design of University Buildings},
  author={Du, Xusheng and Gui, Ruihan and Wang, Zhengyang and Zhang, Ye and Xie, Haoran},
  booktitle={IEEE Conference on Computer Vision},
  year={2024}
}

