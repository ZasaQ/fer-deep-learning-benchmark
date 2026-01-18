# FER Deep Learning Benchmark

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-2.13+-red.svg)](https://keras.io/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-active-success.svg)]()

> Comprehensive benchmark of deep learning architectures and transfer learning strategies for Facial Expression Recognition (FER).

## Table of Contents

- [Overview](#overview)
- [Datasets](#datasets)
- [Architectures](#architectures)
- [Transfer Learning Strategies](#transfer-learning-strategies)
- [Data Augmentation Summary](#data-augmentation-summary)
- [Complete Training Strategy Matrix](#complete-training-strategy-matrix)
- [Key Features](#key-features)

---

## Overview

This repository contains a systematic comparison of **4 deep learning architectures** across **3 facial expression datasets** using **3 transfer learning strategies**, resulting in **36 comprehensive experiments**. The research aims to:

- Establish benchmark results for modern CNN architectures on FER tasks
- Compare the impact of dataset quality
- Evaluate transfer learning strategies (TL, PFT, FFT)
- Provide production-ready models for real-time emotion recognition
- TFLite conversion with different optimalization variants
- Analyze generalization across controlled (CK+) and in-the-wild (RAF-DB) conditions

---

## Datasets

| Dataset | Images | Classes | Resolution | Type | Label Quality | Year |
|---------|--------|---------|------------|------|---------------|------|
| **FER2013** | 35,887 | 7 | 48×48 | Grayscale | Noisy (crowd-sourced) | 2013 |
| **CK+** | 981 | 7 | Various | Grayscale | Clean (controlled lab) | 2010 |
| **RAF-DB** | 15,339 | 7 | Various | RGB | In-the-wild | 2017 |

### Emotion Classes

All datasets use the same 7 emotion categories:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

---

## Architectures

| Model | Parameters | Depth | Pre-trained | Best Use Case |
|-------|-----------|-------|-------------|---------------|
| **SimpleCNN** | 1.2M | 8 layers | No | Baseline comparison |
| **VGG16** | 14.9M | 16 layers | ImageNet | Small datasets (CK+) |
| **ResNet50** | 24.1M | 50 layers | ImageNet | **Best overall performance** |
| **MobileNetV2** | 3.5M | 53 layers | ImageNet | Real-time deployment |

---

## Transfer Learning Strategies

| Strategy | Trainable Layers | Learning Rate | Epochs | Use Case |
|----------|------------------|---------------|--------|----------|
| **TL** (Transfer Learning) | Head only | 1e-3 | 50-60 | Fast baseline, small datasets |
| **PFT** (Partial Fine-Tuning) | Last 2-3 blocks + head | 1e-4 | 80-100 | **Best accuracy/time** |
| **FFT** (Full Fine-Tuning) | All layers | 1e-5 | 100-120 | Maximum accuracy, large datasets |

---

## Data Augmentation Summary

Data augmentation is a key component in training FER models; however, **the intensity of augmentation must be adapted to the quality and size of the dataset**. Overly aggressive augmentation on clean data may degrade performance, while overly conservative augmentation on noisy data may fail to fully exploit the model’s potential.

### Types

#### 1. **Conservative**
**Use case:** Small, controlled datasets with clean labels

| Transformation | Value | Rationale |
|---------------|-------|-----------|
| `rotation_range` | **10°** | Minimal rotation – faces are already aligned |
| `width_shift_range` | **0.05** (5%) | Small shifts – centered faces |
| `height_shift_range` | **0.05** (5%) | Small shifts – centered faces |
| `zoom_range` | **0.1** (±10%) | Mild zoom in/out |
| `brightness_range` | **[0.9, 1.1]** | Minimal brightness variation – controlled lighting |
| `horizontal_flip` | **True** | The only aggressive transform – faces are naturally symmetric |
| `shear_range` | **0.0** | Facial geometry is not distorted |

**Why conservative?**
- Small dataset (CK+: ~1000 images) – risk of overfitting to augmentation  
- Controlled laboratory conditions – no need to simulate real-world variability  
- High-quality labels – avoiding the introduction of artificial noise  

#### 2. **Medium**
**Use case:** Large datasets with clean (re-labeled) annotations

| Transformation | Value | Rationale |
|---------------|-------|-----------|
| `rotation_range` | **10°** | Moderate rotation – natural head pose variations |
| `width_shift_range` | **0.08** (8%) | Moderate shifts – off-center simulation |
| `height_shift_range` | **0.08** (8%) | Moderate shifts – off-center simulation |
| `zoom_range` | **0.1** (±10%) | Different camera distances |
| `brightness_range` | **[0.85, 1.15]** | Moderate lighting variation |
| `horizontal_flip` | **True** | Facial symmetry |
| `shear_range` | **0.0** | Facial geometry is not distorted |

**Why medium?**
- Balanced approach: leverages dataset size without degrading label quality  

#### 3. **Aggressive**
**Use case:** Large datasets with noisy labels

| Transformation | Value | Rationale |
|---------------|-------|-----------|
| `rotation_range` | **15°** | Larger rotations – compensates for label noise |
| `width_shift_range` | **0.1** (10%) | Stronger shifts |
| `height_shift_range` | **0.1** (10%) | Stronger shifts |
| `zoom_range` | **0.15** (±15%) | Greater scale variance |
| `brightness_range` | **[0.8, 1.2]** | Wide lighting range |
| `horizontal_flip` | **True** | Standard |
| `shear_range` | **0.0** | Facial geometry is preserved even with noise |

**Why aggressive?**
- Noisy (crowd-sourced) labels – augmentation acts as regularization  
- Large dataset – can tolerate stronger augmentation  
- Increased diversity helps the model generalize despite label errors  

#### 4. **Very Aggressive**
**Use case:** In-the-wild datasets with highly variable conditions

| Transformation | Value | Rationale |
|---------------|-------|-----------|
| `rotation_range` | **20°** | **Maximum** – real-world head pose angles |
| `width_shift_range` | **0.15** (15%) | Strong shifts – diverse framing |
| `height_shift_range` | **0.15** (15%) | Strong shifts – diverse framing |
| `zoom_range` | **0.2** (±20%) | Very different distances |
| `brightness_range` | **[0.7, 1.3]** | **Wide range** – varying lighting (day/night/indoor/outdoor) |
| `shear_range` | **0.15** | **Additional transform** – perspective and camera angles |
| `horizontal_flip` | **True** | Standard |

**Why very aggressive?**
- **In-the-wild conditions** – extremely high real-world variance  
- Diverse lighting (sunlight, artificial light, shadows)  
- Different camera angles and distances  
- The model must be robust to all conditions  

### Comparison of All Levels

| Transformation | Conservative | Medium | Aggressive | Very Aggressive |
|---------------|-------------|--------|------------|-----------------|
| **Dataset** | CK+ | FERPlus | FER2013 | RAF-DB |
| **Rotation** | 10° | 10° | 15° | 20° |
| **Width Shift** | 0.05 | 0.08 | 0.1 | 0.15 |
| **Height Shift** | 0.05 | 0.08 | 0.1 | 0.15 |
| **Zoom** | 0.1 | 0.1 | 0.15 | 0.2 |
| **Brightness** | [0.9, 1.1] | [0.85, 1.15] | [0.8, 1.2] | [0.7, 1.3] |
| **Shear** | 0.0 | 0.0 | 0.0 | 0.15 |
| **H-Flip** | True | True | True | True |

### Summary

| Dataset | Level | Key Characteristic | Main Reason |
|--------|-------|--------------------|-------------|
| **CK+** | Conservative | Small transformations | Small dataset + controlled conditions |
| **FER2013** | Aggressive | Strong transformations | Noisy labels require regularization |
| **RAF-DB** | Very Aggressive | Maximum diversity | In-the-wild data = maximum variance |

---

## Complete Training Strategy Matrix

| Model           | Dataset | Strategy | LR   | Batch | Epochs | Dropout (conv) | Dropout (dense) | Dense Units | Weight Decay | ES/LR Patience | Augmentation    |
| --------------- | ------- | -------- | ---- | ----- | ------ | -------------- | --------------- | ----------- | ------------ | -------------- | --------------- |
| **SimpleCNN**   | FER2013 | Baseline | 1e-3 | 64    | 100    | 0.25           | 0.5             | 256         | 1e-4         | 20/7           | Aggressive      |
| **SimpleCNN**   | CK+     | Baseline | 5e-4 | 32    | 150    | 0.3            | 0.6             | 128         | 1e-3         | 25/8           | Conservative    |
| **SimpleCNN**   | RAF-DB  | Baseline | 1e-3 | 64    | 100    | 0.25           | 0.5             | 256         | 1e-4         | 20/7           | Very aggressive |
| **VGG16**       | FER2013 | TL       | 1e-3 | 32    | 50     | –              | 0.5             | 256         | 1e-4         | 10/3           | Medium          |
| **VGG16**       | FER2013 | PFT      | 1e-4 | 32    | 80     | –              | 0.5             | 256         | 1e-4         | 15/5           | Aggressive      |
| **VGG16**       | FER2013 | FFT      | 1e-5 | 32    | 100    | –              | 0.5             | 256         | 1e-4         | 20/7           | Aggressive      |
| **VGG16**       | CK+     | TL       | 5e-4 | 16    | 80     | –              | 0.6             | 128         | 5e-4         | 15/5           | Conservative    |
| **VGG16**       | CK+     | PFT      | 5e-5 | 16    | 100    | –              | 0.6             | 128         | 5e-4         | 20/7           | Medium          |
| **VGG16**       | RAF-DB  | TL       | 1e-3 | 32    | 50     | –              | 0.5             | 256         | 1e-4         | 10/3           | Aggressive      |
| **VGG16**       | RAF-DB  | PFT      | 1e-4 | 32    | 80     | –              | 0.5             | 256         | 1e-4         | 15/5           | Very aggressive |
| **VGG16**       | RAF-DB  | FFT      | 1e-5 | 32    | 100    | –              | 0.5             | 256         | 1e-4         | 20/7           | Very aggressive |
| **ResNet50**    | FER2013 | TL       | 1e-3 | 64    | 50     | –              | 0.5             | 256         | 1e-4         | 10/3           | Medium          |
| **ResNet50**    | FER2013 | PFT      | 1e-4 | 64    | 80     | –              | 0.5             | 256         | 1e-4         | 15/5           | Aggressive      |
| **ResNet50**    | FER2013 | FFT      | 1e-5 | 32    | 100    | –              | 0.5             | 256         | 1e-4         | 20/7           | Aggressive      |
| **ResNet50**    | CK+     | TL       | 5e-4 | 16    | 80     | –              | 0.6             | 128         | 5e-4         | 15/5           | Conservative    |
| **ResNet50**    | CK+     | PFT      | 5e-5 | 16    | 100    | –              | 0.6             | 128         | 5e-4         | 20/7           | Medium          |
| **ResNet50**    | RAF-DB  | TL       | 1e-3 | 64    | 50     | –              | 0.5             | 256         | 1e-4         | 10/3           | Aggressive      |
| **ResNet50**    | RAF-DB  | PFT      | 1e-4 | 64    | 80     | –              | 0.5             | 256         | 1e-4         | 15/5           | Very aggressive |
| **ResNet50**    | RAF-DB  | FFT      | 1e-5 | 32    | 100    | –              | 0.5             | 256         | 1e-4         | 20/7           | Very aggressive |
| **MobileNetV2** | FER2013 | TL       | 1e-3 | 64    | 50     | –              | 0.4             | 128         | 1e-4         | 10/3           | Medium          |
| **MobileNetV2** | FER2013 | PFT      | 1e-4 | 64    | 80     | –              | 0.4             | 128         | 1e-4         | 15/5           | Aggressive      |
| **MobileNetV2** | FER2013 | FFT      | 1e-5 | 64    | 100    | –              | 0.4             | 128         | 1e-4         | 20/7           | Aggressive      |
| **MobileNetV2** | CK+     | TL       | 5e-4 | 32    | 80     | –              | 0.5             | 64          | 5e-4         | 15/5           | Conservative    |
| **MobileNetV2** | CK+     | PFT      | 5e-5 | 32    | 100    | –              | 0.5             | 64          | 5e-4         | 20/7           | Medium          |
| **MobileNetV2** | CK+     | FFT      | 5e-6 | 16    | 100    | –              | 0.6             | 64          | 1e-3         | 20/7           | Very aggressive |
| **MobileNetV2** | RAF-DB  | TL       | 1e-3 | 64    | 50     | –              | 0.4             | 128         | 1e-4         | 10/3           | Aggressive      |
| **MobileNetV2** | RAF-DB  | PFT      | 1e-4 | 64    | 80     | –              | 0.4             | 128         | 1e-4         | 15/5           | Very aggressive |
| **MobileNetV2** | RAF-DB  | FFT      | 1e-5 | 64    | 100    | –              | 0.4             | 128         | 1e-4         | 20/7           | Very aggressive |


## Key Features

### Comprehensive Analysis
- **36+ training configurations** with optimal hyperparameters
- Automated dataset distribution analysis and visualization
- Per-class performance metrics (precision, recall, F1-score)
- Confusion matrices (absolute counts + normalized percentages)
- Training history visualization (accuracy, loss, learning rate)

### Production-Ready Models
- **TFLite conversion** with multiple optimization strategies:
  - `default`: Weight quantization (~4x compression)
  - `quantized`: Full INT8 quantization (~15x compression)
- **Real-time inference**: 250-800 FPS (CPU)
- Keras vs TFLite accuracy comparison
- Deployment-ready model artifacts

### Advanced Visualizations
- Dataset sample grids and class montages
- Training curves with overfitting detection
- Precision-Recall trade-off analysis
- F1-score ranking per emotion
- Support distribution analysis
- t-SNE embeddings for feature visualization

### Interactive Configuration
- **GUI widget** for hyperparameter tuning
- Preset configurations for each dataset/model combination
- Real-time parameter validation
- One-click "Apply Recommended Settings"

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Datasets:**
  - FER2013: [Kaggle]([https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge](https://www.kaggle.com/datasets/msambare/fer2013))
  - CK+: [Kaggle]([http://www.consortium.ri.cmu.edu/ckagree/](https://www.kaggle.com/datasets/davilsena/ckdataset))
  - RAF-DB: [Kaggle]([http://www.whdeng.cn/RAF/model1.html](https://www.kaggle.com/datasets/shuvoalok/raf-db-dataset/code))

- **Pre-trained Models:** ImageNet weights from Keras Applications

---

## Star History

If you find this project helpful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=ZasaQ/fer-deep-learning-benchmark&type=Date)](https://star-history.com/#yourusername/fer-deep-learning-benchmark&Date)

---

**Last Updated:** January 2026
