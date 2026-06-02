# SGCR

> **WARNING:** This repository contains evaluation protocols for concept recovery, which may involve safety-critical target concepts (e.g., explicit content). Following responsible release guidelines, no explicit reference images are included.

### Introduction
This repository contains the PyTorch implementation and evaluation protocol for the paper "Geometry-Guided Concept Recovery in Text-to-Image Diffusion Models: A Robustness Study of Structural Control after Concept Erasure".

<p align="center">
  <a href="./assets/framework.pdf">
    <img src="./assets/framework.png" alt="SGCR Framework" width="95%">
  </a>
</p>
<p align="center">
  <em>(Click the image to view the high-resolution PDF)</em>
</p>

### Abstract
Concept erasure has become an important safeguard for reducing unsafe, copyrighted or otherwise unwanted concepts in text-to-image diffusion models. Most existing defences, however, are developed and evaluated primarily under text-only prompting, where the target concept is assumed to be activated through an explicit or implicit textual trigger. This paper investigates whether such an assumption remains valid when erased models are used in modern controllable generation pipelines. We show that concepts suppressed from the text pathway can still be recovered through structural conditioning. Specifically, we study a geometry-guided concept recovery setting in which a benign prompt is combined with a structural signal, such as a Canny edge map, depth map or segmentation map, extracted from a reference image. The resulting inference-time procedure does not require retraining, gradient-based optimisation or modification of the defended backbone, but can redirect the denoising trajectory towards target-consistent visual content through image-side feature interactions. We evaluate the phenomenon across multiple concept categories, including explicit content, copyrighted artistic styles and generic objects, as well as representative concept-erasure defences covering parameter editing, inference-time guidance and feature decomposition. Experimental results indicate that structural guidance can substantially increase attack success rates even when text-only generation appears effectively suppressed. Further analysis across structural modalities and guidance strengths suggests that the vulnerability is not limited to a single controller or condition type. These findings reveal a practical robustness gap in current concept erasure methods and suggest that future safety evaluation for diffusion models should move beyond text-only prompts towards system-level assessment of multimodal, controller-augmented generation.

### Content
```text
├── README.md
├── assets
├── data: dual-track prompt matrices
├── examples: sanitized benchmark examples & extracted geometric scaffolds
├── run_benchmark.py
├── scripts
│   ├── extract_conditions.py
│   ├── evaluate_nudity.py
│   ├── evaluate_style.py
│   └── evaluate_objects.py
└── requirements.txt