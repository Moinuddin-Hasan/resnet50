# ResNet-50 ImageNet Training Pipeline on AWS EC2

## Project Overview

This repository contains the code and documentation for training a ResNet-50 model from scratch on the full ImageNet-1k dataset. The primary objective is to achieve a **top-1 validation accuracy of at least 75%** by leveraging AWS EC2 for GPU-accelerated training.

The project will culminate in a publicly accessible web application hosted on Hugging Face Spaces, allowing users to test the final trained model.

---

## Status for Initial Submission

This submission serves as a request for AWS training credits. The code in this repository represents a **complete, end-to-end trainable pipeline** that is ready for large-scale execution on an AWS EC2 instance.

To ensure the robustness and correctness of the pipeline, it has been successfully tested on **Imagenette**, a smaller 10-class subset of ImageNet. This preliminary run on Google Colab (see `notebooks/` directory) has validated the following components:
- Data loading and augmentation.
- Model instantiation (from scratch).
- The training and validation loops.
- Loss calculation and optimization steps.
- Learning rate scheduling.
- Logging and checkpointing logic.

With this verified pipeline, we are prepared to proceed with the full-scale training on the ImageNet dataset as soon as the AWS credits are allocated.

---

## Key Project Details

*   **Model**: ResNet-50 (trained from scratch, no pre-trained weights).
*   **Dataset**: ImageNet-1k (ILSVRC2012).
*   **Target Accuracy**: ≥ 75% top-1 validation accuracy.
*   **Stretch Goal**: ≥ 78% top-1 validation accuracy for additional points.
*   **Training Platform**: AWS EC2 GPU Instance (e.g., `p3.2xlarge`).
*   **Deployment Target**: Hugging Face Spaces.
*   **Framework**: PyTorch.

---

## Proposed Training Recipe

The full-scale training on EC2 will follow established best practices for this task.

*   **Optimizer**: SGD with Momentum.
    *   `momentum`: 0.9
    *   `weight_decay`: 1e-4
*   **Learning Rate Schedule**: Step Decay or Cosine Annealing.
    *   Initial `learning_rate`: 0.1 (to be scaled linearly with batch size).
    *   For Step Decay, the learning rate will be reduced by a factor of 10 at epochs 30 and 60.
*   **Batch Size**: 256 or 512 (to be finalized based on the selected EC2 instance's VRAM).
*   **Total Epochs**: 90.
*   **Data Augmentation (Training)**:
    *   Random Resized Crop to 224x224.
    *   Random Horizontal Flip.
    *   Standard Color Jitter.
    *   Normalization using ImageNet's mean and standard deviation.
*   **Data Preprocessing (Validation)**:
    *   Resize to 256x256.
    *   Center crop to 224x224.
    *   Normalization using ImageNet's mean and standard deviation.

---

## Repository Structure Overview

```
.
├── README.md                # Project overview and documentation
├── notebooks/
│   └── 01_pipeline_test.ipynb # Colab notebook for testing the pipeline on a subset
├── src/
│   ├── config.py              # Centralized hyperparameters
│   ├── dataset.py             # Data loading and augmentation logic
│   ├── model.py               # Model definition
│   └── train.py               # Main script for training
└── requirements.txt         # Project dependencies
```

---

## Team Members

*   **Moinuddin Hasan**: [moinuddin.hasan.raichur@gmail.com]
*   **Sharan Raghu Venkatachalam**
