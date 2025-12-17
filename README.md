# NeuralGuard: Automated Mechanical Component Quantification

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![License](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Research_Prototype-green?style=for-the-badge)

**NeuralGuard** is a deep learning system designed to solve a fine-grained computer vision problem: the simultaneous counting of multiple mechanical parts (Bolts, Locating Pins, Nuts, and Washers) within a single image.

Unlike standard object detection pipelines (e.g., YOLO), this project utilizes a **Global Multi-Output Regression** approach to achieve high-speed inference and "Exact Match" accuracy without the need for bounding box annotations.

---

## 📌 Problem Statement

The objective is to predict the count of four distinct components in synthetic images. 

**The Challenge:** The evaluation metric is **Exact Match Accuracy**. A prediction is considered correct *only* if the counts for **all four** categories match the ground truth exactly. 
* *Prediction:* `[Bolt: 4, Pin: 0, Nut: 2, Washer: 1]`
* *Truth:* `[Bolt: 4, Pin: 0, Nut: 2, Washer: 2]` 
* *Result:* **FAIL** (0.0 Accuracy)

This zero-tolerance constraint necessitates a model architecture robust to outliers and capable of precise integer regression.

---

## 🏗️ Architecture & Methodology

### 1. The Backbone: EfficientNet-B3 (Noisy Student)
We utilize **EfficientNet-B3** pre-trained on ImageNet with "Noisy Student" weights.
* **Why:** B3 offers an optimal trade-off between parameter efficiency and feature extraction depth.
* **Benefit:** The "Noisy Student" pre-training provides superior robustness to visual noise and occlusions compared to standard supervised pre-training.

### 2. Custom Regression Head
The standard classification head was replaced with a density-estimation block:
* **Pooling:** Adaptive Average Pooling.
* **Hidden Layer:** Linear (1536 $\to$ 512) with **Mish Activation**.
* **Regularization:** Dropout (p=0.2) to prevent overfitting on synthetic textures.
* **Output:** Linear layer predicting 4 continuous floating-point values.

### 3. Optimization Strategy
* **Loss Function: Huber Loss.** chosen over MSE to prevent gradient explosion from outliers (e.g., confusing a texture for a massive cluster of nuts).
* **Optimizer:** AdamW with Cosine Annealing scheduler.
* **Mixed Precision:** Implemented `torch.cuda.amp` (FP16) for accelerated training.

---

## 📊 Dataset & Preprocessing

The dataset consists of synthetic images containing randomized clusters of parts.

| Component | Class Index |
| :--- | :--- |
| **Bolt** | 0 |
| **Locating Pin** | 1 |
| **Nut** | 2 |
| **Washer** | 3 |

**Pipeline:**
* **Input Resolution:** 512x512 pixels.
* **Normalization:** ImageNet mean/std statistics.
* **Augmentation (Albumentations):**
    * `RandomRotate90`, `Flip`, `ShiftScaleRotate` to force the model to learn geometric invariants (shape/texture) rather than pixel positions.

---

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/neuralguard.git](https://github.com/yourusername/neuralguard.git)
   cd neuralguard
