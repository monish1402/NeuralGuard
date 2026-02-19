# NeuralGuard: Automated Mechanical Component Quantification

This repository contains a deep learning solution for automated parts counting, developed for the SolidWorks AI Hackathon. The project utilizes an EfficientNet-B3 backbone to perform multi-label regression, predicting the quantity of four specific mechanical components in an image: bolts, locating pins, nuts, and washers.

Project Overview
The core objective is to accurately count individual parts within cluttered scenes. The model architecture and training pipeline are optimized for high-precision counting, achieving over 99% exact match accuracy on the validation set.

Key Features
Backbone: tf_efficientnet_b3_ns (Noisy Student pretrained).

Custom Head: Features multiple dropout layers and a Mish activation function for robust feature processing.

Loss Function: Huber Loss, chosen for its robustness to outliers compared to MSE.

Augmentation: Extensive spatial augmentations using the Albumentations library, including random rotations (90°), horizontal flips, and vertical flips.

Optimization: AdamW optimizer with Mixed Precision training (AMP) for faster convergence and reduced memory footprint.


Technical Implementation
Data Processing
The PartsDataset class handles image loading via OpenCV and converts them to RGB. Images are resized to 512x512 pixels. During training, labels are processed as float tensors for regression tasks.

Model Architecture
Python
class CountModel(nn.Module):
    def __init__(self, model_name):
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.backbone.num_features, 512),
            nn.Mish(),
            nn.Dropout(0.2),
            nn.Linear(512, 4)
        )
Training Pipeline
The pipeline runs for 15 epochs with a learning rate of 1e-4. It employs an "Exact Match" metric for validation, ensuring that a prediction is only counted as accurate if the counts for all four categories are perfectly correct after rounding.

Getting Started
Environment: Optimized for GPU acceleration (specifically tested on Kaggle with CUDA support).

Dependencies:

torch & torchvision

timm (PyTorch Image Models)

albumentations

opencv-python

pandas & numpy

Inference: The notebook includes a dedicated inference loop that loads the best_model.pth, processes the test set, and generates a submission.csv.

Performance
During the final training run, the model achieved the following metrics:

Training Loss: ~0.0031

Validation Exact Match Accuracy: 99.70%
