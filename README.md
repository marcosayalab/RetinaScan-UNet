# Retinal Vessel Segmentation using Keras 3 and U-Net

An advanced deep learning pipeline developed for the **Inteligencia Artificial** course (*Artificial Intelligence for Software Engineering*) at the **University of Seville** (Academic Year 2025/2026).

This project implements a fully customized convolutional neural network based on the **U-Net** architecture to segment blood vessels in retinal fundus images using the benchmark **DRIVE 2004** dataset.

---

# 🌟 Project Overview & Clinical Context

Automated segmentation of retinal blood vessels is a critical task in computer-aided diagnosis and vision-based healthcare.

Delimiting these vascular structures with pixel-level precision provides essential quantitative biomarkers used by clinical experts to monitor and diagnose major systemic and ophthalmologic pathologies:

* **Diabetic Retinopathy:** Detecting microaneurysms and abnormal neovascularization.
* **Hypertensive Retinopathy:** Measuring arteriolar narrowing and arteriovenous (AV) nicking.
* **Glaucoma & Macular Degeneration:** Assessing changes in optic disc topography and overall vascular health.

This repository addresses the problem as a binary pixel classification challenge. Since medical datasets are notoriously small due to the high cost of expert annotations, this solution leverages:

* Data augmentation
* Patching strategies
* Symmetric skip connections

to achieve high-fidelity segmentations under tight data constraints without requiring high-end GPU clusters.

---

# 📊 Dataset Specification: DRIVE 2004

The **Digital Retinal Images for Vessel Extraction (DRIVE 2004)** dataset is an industry benchmark composed of:

* **Total Samples:** 40 high-resolution digital fundus photographs (`584 × 565` pixels).
* **Training Set (20 Images):**

  * Includes a Field of View (FoV) mask.
  * Includes one definitive manual segmentation by an ophthalmic expert.
* **Test Set (20 Images):**

  * Includes a Field of View (FoV) mask.
  * Includes **two independent manual segmentations** produced by different specialists to measure inter-expert variability.

The manual segmentations act as the absolute **Ground Truth** for training, optimization, and evaluation.

---

# ⚙️ Methodology & Architecture Pipeline

To enable efficient training on standard personal computers without dedicated GPUs, the pipeline shifts computational effort toward smart preprocessing and real-time generation.

## 1. Advanced Preprocessing & Spatial Adjustments

* **Matrix Patching:**
  High-resolution fundus images are divided into smaller sub-images (patches such as `128 × 128` or `256 × 256` pixels). This significantly increases the number of samples while reducing RAM/VRAM usage.

* **Zero Padding:**
  Dynamic padding is applied before patch extraction to ensure compatibility with encoder-decoder downsampling operations. Padding is removed during reconstruction.

* **Intensity Scaling:**
  Pixel matrices are normalized to the `[0.0, 1.0]` range by dividing values by `255`.

---

## 2. On-The-Fly Custom Data Generation

A specialized `DataGenerator` class inheriting from Keras sequence utilities was implemented.

Features include:

* Real-time patch loading
* Normalization
* Geometric data augmentation:

  * Horizontal/vertical flips
  * Random rotations
  * Contrast scaling

This approach prevents memory saturation during training.

---

## 3. Symmetric U-Net Architecture

The model is implemented using the **Keras Functional API**.

### Contracting Path (Encoder)

Repeated blocks composed of:

* Two `3×3 Conv2D` layers with ReLU activation
* One `2×2 MaxPooling2D` layer

This path extracts semantic information while reducing spatial dimensions.

### Expanding Path (Decoder)

Uses:

* `UpSampling2D` or `Conv2DTranspose`
* Followed by `3×3` convolutional layers

This path restores spatial resolution and refines localization.

### Skip Connections

`Concatenate` layers transfer high-resolution features directly from encoder blocks to decoder blocks, preserving:

* Thin vessel structures
* Edge information
* Fine vascular boundaries

---

## 4. Robust 5-Fold Cross-Validation Training

To maximize generalization from the limited dataset:

* A **5-Fold Cross-Validation** strategy is used.
* Five independent model configurations are trained.
* Models are saved as native `.keras` artifacts.

Training configuration:

* **Optimizer:** Adam
* **Loss Function:** Binary Cross-Entropy
* **Callback:** EarlyStopping

Early stopping monitors validation performance to reduce overfitting.

---

# 📐 Performance Evaluation: Dice Coefficient

Traditional metrics such as `Accuracy` are heavily biased in medical segmentation due to severe class imbalance.

Therefore, this project uses the **Sørensen–Dice Coefficient (DICE Score)**:

```math
Dice Score = \frac{2 \times |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}
```

Where:

* `Y` → Manual ground truth mask
* `Ŷ` → Predicted segmentation mask

## Performance Targets

The pipeline must achieve the following average DICE scores:

* **Vs. Expert Clinician 1:** `≥ 0.78`
* **Vs. Expert Clinician 2:** `≥ 0.74`
* **Combined Mean Target:** `≥ 0.75`

---

# 📂 Repository Layout

```text
├── data/
│   ├── training/          # DRIVE training images and masks
│   └── test/              # DRIVE test images and expert masks
│
├── src/
│   ├── generator.py       # DataGenerator (patching, augmentation, padding)
│   ├── model.py           # U-Net model definition
│   ├── train.py           # 5-Fold Cross-Validation training
│   └── evaluate.py        # DICE evaluation and mask reconstruction
│
├── models/
│   ├── unet_fold_1.keras
│   └── ...
│
├── output/
│   ├── result_01.png
│   └── ...
│
├── docs/                  # Scientific report / LaTeX files
│
└── README.md
```

---

# 🛠️ Setup & Execution Guide

## Installation

Ensure you are using **Python 3.10+**.

Install dependencies using:

```bash
pip install tensorflow keras numpy scikit-learn matplotlib pillow
```

---

## Execution Steps

### 1. Run Training

Executes the full 5-fold cross-validation training pipeline and stores trained models in the `models/` directory.

```bash
python src/train.py
```

---

### 2. Run Evaluation & Inference

Evaluates saved models against the test dataset, computes final DICE scores, and generates segmentation masks as `.png` files.

```bash
python src/evaluate.py
```

---

# 🎓 Academic Attribution

* **Institution:**
  Universidad de Sevilla — Departamento de Ciencias de la Computación e Inteligencia Artificial (CCIA)

* **Course:**
  Inteligencia Artificial (Ingeniería del Software)

* **Project Specifications:**
  Manuel Soriano Trigueros

* **Academic Year:**
  2025/2026

