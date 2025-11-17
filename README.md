
# **Brain MRI & Tumor Classification using DenseNet201**

A deep learning project for classifying **brain MRI scans** into **8 categories**, covering dementia severity and tumor types, using **DenseNet201 transfer learning** and a two-phase training strategy.

---

## **Table of Contents**

* [Overview](#overview)
* [Dataset](#dataset)
* [Model Architecture](#model-architecture)
* [Training Strategy](#training-strategy)
* [Data Augmentation](#data-augmentation)
* [Performance](#performance)
* [Installation](#installation)
* [Results](#results)
* [Technologies Used](#technologies-used)
* [Model Optimization](#model-optimization)
* [License](#license)

---

## **Overview**

This project implements a high-accuracy **multi-class deep learning classifier** for brain MRI images, capable of identifying:

### **Dementia Classes**

* NonDemented
* VeryMildDemented
* MildDemented
* ModerateDemented

### **Tumor Classes**

* Glioma
* Meningioma
* No Tumor
* Pituitary

The model achieves **99.75% test accuracy**, made possible by DenseNet201 transfer learning, aggressive augmentation, class-weight balancing, and two-phase fine-tuning.

### **Key Features**

* 8-class MRI classification
* DenseNet201 pretrained on ImageNet
* Fine-tuned classification head
* Two-phase training (feature extraction → fine-tuning)
* Class imbalance correction
* Extensive data augmentation

---

## **Dataset**

Total images: **57,865**

| Split      | Images | Classes |
| ---------- | ------ | ------- |
| Training   | 41,264 | 8       |
| Validation | 5,708  | 8       |
| Testing    | 10,893 | 8       |

### **Training Set Class Distribution**

* NonDemented — 8,960
* VeryMildDemented — 7,839
* MildDemented — 7,000
* ModerateDemented — 7,000
* Glioma — 3,121
* Meningioma — 2,880
* No Tumor — 2,696
* Pituitary — 1,768

---

## **Model Architecture**

### **Base Model**

* **DenseNet201**
* Pretrained on ImageNet
* Frozen initially, later unfrozen for fine-tuning

### **Custom Classification Layers**

```
DenseNet201  
 ↓
GlobalAveragePooling2D  
 ↓
Dense(1024, activation='silu') + BatchNorm + Dropout(0.3)  
Dense(512,  activation='silu') + BatchNorm + Dropout(0.25)  
Dense(256,  activation='silu') + BatchNorm + Dropout(0.2)  
Dense(128,  activation='silu') + BatchNorm + Dropout(0.15)  
Dense(64,   activation='silu') + BatchNorm + Dropout(0.1)  
Dense(8, activation='softmax')
```

---

## **Training Strategy**

### **Phase 1 — Feature Extraction**

* Base model frozen
* Adam Optimizer
* Learning rate: **1e-4**
* Epochs: **50** (EarlyStopping enabled)

### **Phase 2 — Fine-Tuning**

* Base model unfrozen
* Adam Optimizer
* Learning rate: **1e-5**
* Epochs: **25** (EarlyStopping enabled)

---

## **Data Augmentation**

Applied using `ImageDataGenerator`:

* Rescaling
* Horizontal flip
* Width/height shift (±20%)
* Zoom (±20%)
* Shear
* Rotation (±30°)

---

## **Performance**

### **Validation Performance**

| Metric         | Score      |
| -------------- | ---------- |
| Accuracy       | **99.88%** |
| Precision      | 99.88%     |
| Recall         | 99.88%     |
| F1-Score       | 99.88%     |
| AUC-ROC        | 99.99%     |
| Loss           | 0.0055     |
| Top-5 Accuracy | 100%       |

### **Test Performance**

| Metric         | Score      |
| -------------- | ---------- |
| Accuracy       | **99.75%** |
| Precision      | 99.75%     |
| Recall         | 99.75%     |
| F1-Score       | 99.75%     |
| AUC-ROC        | 99.99%     |
| Loss           | 0.0079     |
| Top-5 Accuracy | 100%       |

### **Per-Class Performance**

All classes show **0.98–1.00** precision, recall, and F1-score.

---

## **Installation**

### **Requirements**

* Python **3.8+**
* TensorFlow **2.16**
* GPU with CUDA (recommended)


## **Results**

* **Validation Accuracy:** 99.88%
* **Test Accuracy:** 99.75%
* **Macro ROC-AUC:** 1.0000

Training curves show smooth convergence with minimal overfitting, thanks to dropout, normalization, and staged training.

---

## **Technologies Used**

* TensorFlow / Keras
* DenseNet201
* NumPy
* Scikit-learn
* Matplotlib

---

## **Model Optimization**
## ⚖️ Class Imbalance Handling

This project addresses class imbalance using **automatically computed class weights** via the `scikit-learn` package.

### 🔍 Class Weight Summary
- **Minimum weight:** ~0.58 (most frequent class)
- **Maximum weight:** ~2.92 (least frequent class)

These weights were computed using:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
class_weights=compute_class_weight(class_weight='balanced',classes=np.unique(train_data.classes),y=train_data.classes)
```
This ensures that underrepresented classes are given higher importance during model training, helping to improve overall performance and fairness.

### **Callbacks**
* `EarlyStopping(patience=5, restore_best_weights=True)`
---

## **License**

This project is licensed under the **MIT License**.

---

<div align="center">

**Developed with ❤️ for advancing medical AI diagnostics with help of Dhruv Kumar Dubey,Rakshitha Dahiya,Ankush Mitra**
</div>
