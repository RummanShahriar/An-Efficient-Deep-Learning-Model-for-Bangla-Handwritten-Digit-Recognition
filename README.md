# EfficientCNN: A High-Performance Lightweight Model for Bangla Handwritten Digit Recognition

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Accuracy](https://img.shields.io/badge/accuracy-99.78%25-brightgreen)]()

**EfficientCNN** is a custom deep learning architecture for recognizing handwritten Bangla digits. Designed for the **BHaND dataset** (32×32 grayscale images), it achieves **99.78% test accuracy** while being significantly more parameter‑efficient and faster than classic models like AlexNet, LeNet‑5, and MobileNetV2. The key innovation is the use of **Global Average Pooling (GAP)** instead of a flatten layer, which drastically reduces the number of parameters without sacrificing feature learning.

> **Associated paper:** *"An Efficient Deep Learning Model for Bangla Handwritten Digit Recognition"* (included in this repository as a PDF).

---

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Results](#results)
  - [Performance Metrics](#performance-metrics)
  - [Benchmark Comparison](#benchmark-comparison)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
  - [Training](#training)
  - [Evaluation](#evaluation)
  - [Inference](#inference)
- [Project Structure](#project-structure)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Overview

Handwritten digit recognition is a core component of OCR systems, especially for languages with complex scripts like Bengali (Bangla), spoken by over 265 million people. However, existing models either lack accuracy or are too computationally heavy for real‑world, resource‑constrained devices (e.g., mobile phones, embedded systems).

**EfficientCNN** bridges this gap by delivering:
- **State‑of‑the‑art accuracy** (99.78%) on the BHaND benchmark.
- **Low parameter count** (1.44 million) → small model size (~16.6 MB).
- **Fast inference** (0.724 ms per 1024 images) → suitable for real‑time applications.

The architecture is specifically tailored for 32×32 grayscale images and outperforms both heavyweight networks (AlexNet) and efficient general‑purpose networks (MobileNetV2‑0.5).

---

## Key Features

- **High Accuracy** – 99.78% test accuracy on 10,000 unseen BHaND images.
- **Parameter Efficiency** – Global Average Pooling replaces fully connected layers, reducing parameters by ~75% compared to AlexNet.
- **Low Latency** – Optimized for fast inference (0.724 ms per batch of 1024 images).
- **Robust Generalization** – Percentage‑point gap (∆pp) of only −0.07, indicating no overfitting.
- **Comprehensive Benchmarking** – Compared against LeNet‑5, AlexNet, and MobileNetV2‑0.5.
- **Reproducible Pipeline** – Includes training, evaluation, and inference scripts with data augmentation and callbacks.

---

## Dataset

**BHaND (Bengali Handwritten Numerals Dataset)**  
- **Images:** 70,000 grayscale images of digits 0–9  
- **Split:** 50,000 training, 10,000 validation, 10,000 test  
- **Size:** 32×32 pixels, pixel values inverted (digit stroke = 1, background = 0)  
- **Format:** GZIP‑compressed pickle file (flattened 1024‑dim vectors, reshaped to (32,32,1) for CNNs)  

The dataset is designed to mirror the MNIST structure but for Bangla digits, capturing the unique calligraphic variability of the script.

> **Source:** The BHaND dataset is publicly available. If you use it, please cite the original creators (see [citation](#citation)).

---

## Model Architecture

EfficientCNN is a **custom convolutional neural network** with the following layers:

| Block | Layer Type               | Filters/Units | Kernel | Activation | Regularization      |
|-------|--------------------------|---------------|--------|------------|---------------------|
| 1     | Conv2D + Conv2D          | 32            | 3×3    | ReLU       | BatchNorm, MaxPool2D, Dropout(0.25) |
| 2     | Conv2D + Conv2D          | 64            | 3×3    | ReLU       | BatchNorm, MaxPool2D, Dropout(0.25) |
| 3     | Conv2D + Conv2D          | 128           | 3×3    | ReLU       | BatchNorm, MaxPool2D, Dropout(0.25) |
| 4     | Conv2D + Conv2D          | 256           | 3×3    | ReLU       | BatchNorm, MaxPool2D, Dropout(0.25) |
| -     | **Global Average Pooling** | -           | -      | -          | Reduces parameters drastically |
| 5     | Dense + BatchNorm        | 512           | -      | ReLU       | -                   |
| 6     | Dense + BatchNorm + Dropout | 256        | -      | ReLU       | Dropout(0.5)        |
| 7     | Dense (output)           | 10            | -      | Softmax    | -                   |

**Total parameters:** 1,442,154 (1,439,658 trainable)  
**Input shape:** (32, 32, 1)  
**Output:** Probability distribution over 10 digit classes

> **Key innovation:** The **Global Average Pooling** layer replaces the traditional `Flatten` + large dense layers, dramatically cutting the parameter count while preserving spatial information.

---

## Results

### Performance Metrics

| Metric                | Value      |
|-----------------------|------------|
| Test Accuracy         | **99.78%** |
| Test Loss             | 0.0114     |
| Precision (macro avg) | ≥0.996     |
| Recall (macro avg)    | ≥0.996     |
| F1-score (macro avg)  | ≥0.996     |

The confusion matrix shows an almost perfect diagonal, with extremely few misclassifications (e.g., digits “2” and “3” achieved 100% recall).

### Benchmark Comparison

| Model              | Accuracy | Parameters | Inference Latency (ms/1024 images) | Model Size (MB) |
|--------------------|----------|------------|-------------------------------------|-----------------|
| **EfficientCNN**   | **99.78%** | 1,442,154  | 0.724                               | 16.63           |
| AlexNet            | 99.25%   | 5,686,858  | 0.613                               | 65.16           |
| LeNet‑5            | 98.96%   | 82,826     | 0.350                               | 0.99            |
| MobileNetV2‑0.5    | 98.24%   | 719,040    | 5.033                               | 8.59            |

**Key observations:**
- **Higher accuracy** than all competitors, including AlexNet (+0.53%).
- **75% smaller** than AlexNet (16.6 MB vs. 65.2 MB) with 1/4 the parameters.
- **7× faster inference** than MobileNetV2‑0.5 (0.724 ms vs. 5.033 ms), while being significantly more accurate.
- LeNet‑5 is extremely lightweight but accuracy lags by ~0.8 percentage points.

---

## Installation & Requirements

### Prerequisites
- Python 3.9 or higher
- TensorFlow 2.x (with Keras)
- NumPy, Pandas, scikit‑learn
- Matplotlib, Seaborn (for visualizations)
- OpenCV (optional, for custom image preprocessing)

### Setup

```bash
# Clone the repository
git clone https://github.com/RummanShahriar/An-Efficient-Deep-Learning-Model-for-Bangla-Handwritten-Digit-Recognition.git
cd An-Efficient-Deep-Learning-Model-for-Bangla-Handwritten-Digit-Recognition

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

If no `requirements.txt` is provided, manually install:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn opencv-python
```

---

## Usage

### 1. Data Preparation

Place the BHaND dataset in the following structure:

```
data/
└── BHAND/
    ├── train/
    │   ├── 0/
    │   ├── 1/
    │   └── ... (up to 9)
    └── test/
        ├── 0/
        ├── 1/
        └── ...
```

If your dataset is a single pickle file, update the data loader in the notebook/script accordingly.

### 2. Training the Model

Run the provided Jupyter notebook:

```bash
jupyter notebook "Code-An Efficient Deep Learning Model for Bangla.ipynb"
```

Or execute as a Python script (if converted). The training pipeline includes:
- Real‑time data augmentation (rotation, shifts, shear, zoom)
- Callbacks: `ModelCheckpoint`, `ReduceLROnPlateau`, `EarlyStopping`, `ExponentialLR`
- 50 epochs maximum (early stopping typically at epoch 40)

### 3. Evaluation

After training, the model will be evaluated on the 10,000 test images. Metrics reported:
- Test accuracy and loss
- Classification report (precision, recall, F1 per class)
- Confusion matrix (visualized)

### 4. Inference on a Single Image

```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('efficientcnn_best.h5')

# Load and preprocess a custom 32x32 grayscale image
img = cv2.imread('digit.png', cv2.IMREAD_GRAYSCALE)
img = cv2.resize(img, (32, 32))
img = img.astype('float32') / 255.0
img = 1 - img   # Invert if needed (stroke=1, background=0)
img = np.expand_dims(img, axis=(0, -1))  # shape (1,32,32,1)

# Predict
pred = model.predict(img)
digit = np.argmax(pred)
print(f'Predicted digit: {digit}')
```

---

## Project Structure

```
.
├── README.md                          # This file
├── An Efficient Deep Learning Model for Bangla.pdf  # Full paper
├── Code-An Efficient Deep Learning Model for Bangla.ipynb  # Main notebook
├── data/                              # (Place dataset here)
│   └── BHAND/
├── models/                            # Saved model weights
├── outputs/                           # Plots (accuracy/loss curves, confusion matrix)
└── requirements.txt                   # Dependencies (optional)
```

---

## Citation

If you use this code or the EfficientCNN architecture in your research, please cite the associated paper:

```bibtex
@article{kabir2025efficientcnn,
  title={An Efficient Deep Learning Model for Bangla Handwritten Digit Recognition},
  author={Kabir, A.K.M. Nihalul and Shamim, Md. Shahadat Hossain and Shahriar, Md. Rumman and Biplob, Md. Asif Hasan},
  journal={arXiv preprint},
  year={2025},
  note={BRAC University, Dhaka, Bangladesh}
}
```

Also cite the BHaND dataset:

```bibtex
@misc{chowdhury2020bhand,
  author = {Chowdhury, Saadat},
  title = {BHaND – Bengali Handwritten Numerals Dataset},
  year = {2020},
  publisher = {GitHub},
  url = {https://github.com/SaadatChowdhury/BHaND}
}
```

---

## Acknowledgments

- **Dr. Md. Ashraful Alam** (Associate Professor, BRAC University) for invaluable guidance and support.
- **Shammo Biswas** (Research Assistant, BRAC University) for technical assistance.
- The authors of the BHaND dataset for making it publicly available.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.  
You are free to use, modify, and distribute this code for academic and commercial purposes, provided you give appropriate credit.

---

## Contact

**Md. Rumman Shahriar**  
[GitHub: RummanShahriar](https://github.com/RummanShahriar)  

For issues or questions, please open an issue on this repository.
