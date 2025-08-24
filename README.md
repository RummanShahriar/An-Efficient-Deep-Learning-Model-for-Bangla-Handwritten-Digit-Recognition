# An Efficient Deep Learning Model for Bangla Handwritten Digit Recognition

EfficientCNN is a lightweight deep learning model for Bangla handwritten digit recognition using the BHaND dataset. Optimized with Global Average Pooling, it achieves 99.78% accuracy while reducing parameters, offering superior efficiency over models like LeNet-5, AlexNet, and MobileNetV2.

## Overview
This repository implements **EfficientCNN**, a custom CNN tailored for **32×32** grayscale Bangla digit images (BHaND). The design replaces large fully connected layers with **Global Average Pooling (GAP)** to cut parameters and latency without sacrificing accuracy. The model targets deployment on resource-constrained devices while maintaining state-of-the-art performance.

## Key Features
- High test accuracy: **99.78%** on BHaND
- **GAP instead of Flatten** → drastically fewer parameters
- Optimized for **32×32** inputs, low latency inference
- Benchmarked against **LeNet-5**, **AlexNet**, **MobileNetV2-0.5**
- Reproducible training, evaluation, and inference scripts

## Dataset
- **BHaND** (Bangla Handwritten Digits), 10 classes (0–9), 32×32 grayscale.
- Place images under: `data/BHAND/` with `train/` and `test/` subfolders (class-wise directories: `0/ ... 9/`).
- If your copy differs, update paths in the config.

## Model
- Convolutional blocks with BatchNorm and ReLU
- Strided convs for downsampling (no heavy pooling stacks)
- **Global Average Pooling** → Dense(10) with softmax
- Optional dropout for regularization

## Results (BHaND, 32×32)
- **EfficientCNN**: 99.78% test accuracy
- Fewer parameters and faster inference vs baselines
- Confusion matrix and per-class metrics available via `--report`

## Project Details (Tools Used)
- **Language:** Python 3.9+
- **DL Framework:** TensorFlow/Keras (or PyTorch alternative; update scripts accordingly)
- **Data/Utils:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib
