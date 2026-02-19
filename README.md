# NeuralGuard: Automated Parts Counting Model

NeuralGuard is a state-of-the-art computer vision system developed for the **SolidWorks AI Hackathon** to automate the counting of mechanical components in cluttered industrial scenes. By leveraging a multi-label regression approach, the model provides high-precision quantification across four hardware categories: **Bolts, Locating Pins, Nuts, and Washers**.

---

## 🚀 Key Performance Results
* **Exact Match Accuracy**: **99.70%** (Validation set)
* **Training Loss**: **0.0031** (Huber Loss)
* **Inference Capacity**: Processed 2,000 test images with automated submission generation.

---

## 🏗️ Model Architecture
The system utilizes a custom regression architecture optimized for feature extraction and numerical precision:
* **Backbone**: `tf_efficientnet_b3_ns` (Noisy Student) pretrained backbone.
* **Custom Head**: 
    * Dropout layers (0.2) for robust regularization.
    * Dense layers (512 units) for high-dimensional feature mapping.
    * **Mish Activation** for improved gradient flow during backpropagation.
    * Final output layer with 4 units for multi-category regression.

---

## 🛠️ Technical Implementation

### Data Pipeline & Augmentation
NeuralGuard uses the `albumentations` library to ensure spatial invariance and model robustness.
* **Input Resolution**: 512 x 512 pixels.
* **Augmentations**: Horizontal Flip, Vertical Flip, and Random 90° Rotations.
* **Normalization**: Standard ImageNet statistics (Mean: [0.485, 0.456, 0.406]).

### Optimization Strategy
* **Loss Function**: **Huber Loss**, selected for its resilience to outliers compared to standard MSE.
* **Optimizer**: **AdamW** with a learning rate of `1e-4`.
* **Mixed Precision**: Leverages `torch.cuda.amp` to accelerate training and reduce GPU memory overhead.
* **Evaluation Metric**: **Exact Match Accuracy** — a strict validator requiring perfect rounded predictions for all four classes simultaneously.

---

## 📁 Repository Structure
* `neuralguard-1.ipynb`: The complete end-to-end pipeline (EDA, Training, Validation, Inference).
* `best_model.pth`: Serialized weights for the top-performing model epoch.
* `submission.csv`: Final predicted counts for the competition test set.

---

## 🚀 Getting Started

### Installation
```bash
pip install torch torchvision timm albumentations opencv-python pandas
