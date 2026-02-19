#NeuralGuard: Automated Parts Counting Model
NeuralGuard is a deep learning-based computer vision solution designed to accurately count mechanical parts in cluttered scenes. Developed for the SolidWorks AI Hackathon, the model leverages a state-of-the-art EfficientNet-B3 backbone to perform multi-label regression for part quantification.

#🎯 Objective
The primary goal is to predict the quantity of four specific mechanical components within an image:

Bolts

Locating Pins

Nuts

Washers

#🏗️ Model Architecture
The project employs a custom regression head built on top of a pre-trained backbone:

Backbone: tf_efficientnet_b3_ns (Noisy Student) from the timm library.

Custom Head:

Dropout layer (0.2) for regularization.

Linear layer (512 units).

Mish activation function.

Secondary dropout layer (0.2).

Final output layer (4 units) for multi-label regression.

#🧪 Technical Strategy
Data Processing & Augmentation
Input Size: 512x512 pixels.

Augmentations: To ensure spatial robustness, the pipeline uses the albumentations library for:

Horizontal & Vertical Flips.

90-degree Random Rotations.

Normalization: Standard ImageNet mean and standard deviation.

Training Configuration
Loss Function: Huber Loss, used for its robustness to outliers in regression tasks.

Optimizer: AdamW with a learning rate of 1e-4.

Training Specs: 15 epochs with a batch size of 16.

Precision: Mixed Precision training (AMP) via GradScaler for optimized performance on modern GPUs.

Evaluation Metric
The model is evaluated using Exact Match Accuracy. A prediction is only considered correct if the rounded counts for all four categories match the ground truth perfectly.

#📈 Performance
In its final training run, the model achieved exceptional results on the validation set:

Final Training Loss: 0.0031

Validation Exact Match Accuracy: 99.70%

#📁 Repository Structure
neuralguard-1.ipynb: The complete development pipeline including EDA, training, and inference.

best_model.pth: Saved weights for the highest-performing validation epoch.

submission.csv: Generated predictions for the competition test set.

#🚀 Getting Started
Dependencies
Bash
pip install torch torchvision timm albumentations opencv-python pandas numpy tqdm
Dataset Requirement
The pipeline expects the following structure (optimized for Kaggle environments):

#Plaintext
/kaggle/input/solidworks-ai-hackathon/
├── train/          # Training images
├── test/           # Test images
└── train_labels.csv # Labels for training
#🛠️ Usage
Training: Run the first 7 cells of the notebook to initialize the dataset, build the model, and begin the training loop.

Inference: The final cells load best_model.pth, process the test/ directory, and generate a submission.csv containing predicted counts for the 2,000 test images.
