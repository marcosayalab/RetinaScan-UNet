# Retinal Vessel Segmentation using Keras 3 and U-Net

An advanced deep learning pipeline for automated retinal vessel segmentation, developed for the **Inteligencia Artificial** course (*Artificial Intelligence for Software Engineering*) at the **University of Seville** (Academic Year 2025/2026).

This project implements a custom **U-Net** architecture trained through **5-Fold Cross-Validation** to segment retinal blood vessels in fundus images using the benchmark **DRIVE 2004** dataset. The final inference pipeline incorporates **Test-Time Augmentation (TTA)**, **Adaptive Otsu Thresholding**, and **Morphological Post-Processing** to maximize segmentation quality and robustness.

---

# 🌟 Project Overview & Clinical Context

Automated retinal vessel segmentation is a fundamental task in computer-aided diagnosis and vision-based healthcare.

Accurate extraction of vascular structures provides quantitative biomarkers that ophthalmologists use to monitor and diagnose several diseases:

- **Diabetic Retinopathy:** Detection of microaneurysms and abnormal neovascularization.
- **Hypertensive Retinopathy:** Analysis of arteriolar narrowing and arteriovenous (AV) nicking.
- **Glaucoma:** Evaluation of vascular changes related to optic nerve damage.
- **Age-Related Macular Degeneration (AMD):** Assessment of retinal vascular integrity.

This repository addresses the problem as a binary semantic segmentation task where each pixel is classified as either vessel or background.

Because medical imaging datasets are typically small and expensive to annotate, the project combines:

- Data augmentation
- Patch-based training
- Skip-connected encoder-decoder architectures
- Cross-validation
- Advanced inference post-processing

to achieve high-quality vessel extraction while remaining computationally affordable.

---

# 📊 Dataset Specification: DRIVE 2004

The **Digital Retinal Images for Vessel Extraction (DRIVE 2004)** dataset is one of the most widely used benchmarks for retinal vessel segmentation.

## Dataset Composition

- **Total Images:** 40 retinal fundus photographs
- **Resolution:** `584 × 565` pixels
- **Training Set:** 20 images
- **Testing Set:** 20 images

## Available Annotations

### Training Images

Each image includes:

- Original RGB fundus image
- Field of View (FoV) mask
- Expert manual vessel segmentation

### Test Images

Each image includes:

- Original RGB fundus image
- Field of View (FoV) mask
- Two independent expert segmentations

The second annotation allows measuring **inter-observer variability**, providing a realistic upper-bound reference for algorithm performance.

---

# ⚙️ Methodology & Architecture Pipeline

The pipeline is designed to run efficiently on standard consumer hardware while maintaining competitive segmentation performance.

---

## 1. Advanced Preprocessing & Spatial Adjustments

### Matrix Patching

High-resolution retinal images are divided into smaller patches (e.g. `128×128` or `256×256` pixels).

Benefits:

- Increases effective dataset size
- Reduces memory requirements
- Enables larger batch sizes
- Improves training stability

### Zero Padding

Images are dynamically padded before patch extraction to ensure dimensions remain compatible with encoder-decoder downsampling operations.

Padding is removed during final reconstruction.

### Intensity Normalization

Input pixels are scaled to:

```text
[0.0, 1.0]
```

through division by `255`.

---

## 2. On-The-Fly Data Generation & Augmentation

A custom `DataGenerator` derived from Keras sequence utilities performs:

- Dynamic patch loading
- Real-time normalization
- Batch generation

### Applied Augmentations

- Horizontal flips
- Vertical flips
- Random rotations
- Contrast perturbations

This strategy increases dataset diversity while keeping RAM usage low.

---

## 3. Symmetric U-Net Architecture

The network is implemented using the **Keras Functional API**.

### Contracting Path (Encoder)

Each encoder block contains:

- Two `3×3 Conv2D` layers with ReLU activation
- He initialization
- One `2×2 MaxPooling2D` layer

Purpose:

- Extract semantic information
- Increase receptive field
- Reduce spatial dimensions

### Expanding Path (Decoder)

The decoder uses:

- `UpSampling2D` or `Conv2DTranspose`
- Skip-connections
- Additional convolutional refinement

Purpose:

- Recover spatial resolution
- Improve localization accuracy

### Skip Connections

Feature maps from encoder layers are directly concatenated with decoder layers using:

```python
Concatenate()
```

Benefits:

- Preservation of thin vessels
- Sharper boundaries
- Better reconstruction of capillary structures

---

## 4. Robust 5-Fold Cross-Validation Training

Given the small dataset size, model robustness is increased through:

### Cross-Validation Strategy

- 5 independent folds
- Training/validation rotation
- Reduced sampling bias

### Training Configuration

- **Optimizer:** Adam
- **Loss Function:** Binary Cross-Entropy
- **Framework:** Keras 3 / TensorFlow
- **Callback:** EarlyStopping

### Output

Five trained models are generated:

```text
fold_1.keras
fold_2.keras
fold_3.keras
fold_4.keras
fold_5.keras
```

These models can later be evaluated individually or combined as an ensemble.

---

# 🔬 Advanced Inference Pipeline

Unlike a standard U-Net implementation, the final prediction workflow integrates several post-processing techniques.

## Test-Time Augmentation (TTA)

For each image, predictions are computed on:

- Original image
- Horizontally flipped image
- Vertically flipped image

Predictions are transformed back and averaged.

Benefits:

- Reduces prediction variance
- Improves robustness
- Produces smoother probability maps

---

## Adaptive Otsu Thresholding

Instead of using a fixed threshold (e.g. 0.5), the system computes an adaptive threshold using Otsu's method.

Advantages:

- Adapts to illumination differences
- Handles varying vessel contrast
- Improves vessel/background separation

A sensitivity offset is applied to better preserve thin capillaries.

---

## Morphological Post-Processing

After binarization, morphological operations are applied.

### Elliptical Closing

Used to:

- Reconnect fragmented vessels
- Fill tiny discontinuities
- Preserve anatomical vessel shapes

without introducing excessive artifacts.

---

# 📐 Performance Evaluation: Dice Coefficient

Medical image segmentation datasets are highly imbalanced because vessel pixels represent only a small fraction of the image.

For this reason, traditional metrics such as Accuracy are often misleading.

The project evaluates segmentation quality using the **Sørensen–Dice Coefficient**:

```math
Dice = \frac{2 \times |Y \cap \hat{Y}|}{|Y| + |\hat{Y}|}
```

Where:

- `Y` = Ground Truth segmentation
- `Ŷ` = Predicted segmentation

A Dice score of:

- **1.0** → Perfect overlap
- **0.0** → No overlap

---

## Target Performance

| Evaluation | Target Dice Score |
|------------|------------------|
| Expert 1 | ≥ 0.78 |
| Expert 2 | ≥ 0.74 |
| Combined Mean | ≥ 0.75 |

These values align with the academic objectives established for the project.

---

# 📂 Repository Structure

```text
RetinaScan-UNet/
│
├── .gitignore
├── LICENSE
├── README.md
│
├── data/
│   ├── training/
│   │   ├── images/         # Training fundus images
│   │   ├── 1st_manual/     # Expert vessel annotations
│   │   └── mask/           # Field of View masks
│   │
│   └── test/
│       ├── images/         # Test fundus images
│       ├── 1st_manual/     # Expert 1 annotations
│       ├── 2nd_manual/     # Expert 2 annotations
│       └── mask/           # Field of View masks
│
├── src/
│   ├── model.py            # U-Net architecture
│   ├── generator.py        # Data loading and augmentation
│   ├── metrics.py          # Dice coefficient implementation
│   ├── train.py            # Training pipeline
│   ├── evaluate.py         # Evaluation and inference
│   │
│   └── test/
│       ├── test_metrics.py
│       └── test_integracion.py
│
├── models/
│   ├── fold_1.keras
│   ├── fold_2.keras
│   ├── fold_3.keras
│   ├── fold_4.keras
│   └── fold_5.keras
│
├── predictions/
│   ├── *_pred.png
│   └── *_comparacion.png
│
├── notebooks/
│   ├── exploracion.ipynb
│   ├── evaluation_test.ipynb
│   ├── demo_defensa.ipynb
│   └── Segmentacion_Pipeline_Completo.ipynb
│
├── docs/
│
└── LICENSE
```

---

## Directory Description

| Directory | Description |
|------------|------------|
| `data/` | Complete DRIVE dataset with images, masks and annotations. |
| `src/` | Core implementation of the segmentation pipeline. |
| `src/test/` | Unit and integration tests. |
| `models/` | Trained models obtained through 5-Fold Cross-Validation. |
| `predictions/` | Generated segmentation masks and visual comparisons. |
| `notebooks/` | Interactive notebooks for experimentation and demonstrations. |
| `docs/` | Reports, documentation and supplementary material. |

---

# 🛠️ Setup & Execution Guide

## Requirements

- Python 3.10+
- TensorFlow / Keras 3

Install dependencies:

```bash
pip install tensorflow keras numpy scikit-learn matplotlib pillow opencv-python pytest
```

---

## Training

Run the complete 5-Fold Cross-Validation pipeline:

```bash
python src/train.py
```

Generated models will be stored inside:

```text
models/
```

---

## Evaluation & Inference

Execute the full evaluation pipeline:

```bash
python src/evaluate.py
```

This stage performs:

1. Model loading
2. Ensemble prediction
3. Test-Time Augmentation
4. Adaptive Otsu Thresholding
5. Morphological Cleaning
6. Dice computation
7. PNG mask generation

Generated outputs are saved inside:

```text
predictions/
```

---

## ✅ Testing

Available test suites:

- `test_metrics.py` → Validation of Dice coefficient calculations.
- `test_integracion.py` → End-to-end integration testing.

---

# 🎓 Academic Notebooks

Two main interactive notebooks are included for reproducibility and presentations.

## demo_defensa.ipynb

Presentation-oriented notebook.

Features:

- Interactive image selection
- Qualitative result comparison
- Visualization of TTA effects
- Otsu thresholding demonstrations

Ideal for project defense and live demonstrations.

---

## Segmentacion_Pipeline_Completo.ipynb

Complete end-to-end implementation.

Includes:

```text
Data Loading
     ↓
Preprocessing
     ↓
Training
     ↓
Cross-Validation
     ↓
Inference
     ↓
TTA
     ↓
Adaptive Otsu
     ↓
Post-Processing
     ↓
Metrics
```

Suitable for:

- Google Colab
- Jupyter Notebook
- JupyterLab

---

# 📚 References & Methodology

### U-Net

Ronneberger, O., Fischer, P., & Brox, T. (2015).

> U-Net: Convolutional Networks for Biomedical Image Segmentation.

Presented at MICCAI.

---

### Otsu Thresholding

Otsu, N. (1979).

> A Threshold Selection Method from Gray-Level Histograms.

IEEE Transactions on Systems, Man, and Cybernetics.

---

### Test-Time Augmentation

Wang, G., et al. (2019).

> Aleatoric Uncertainty Estimation with Test-Time Augmentation for Medical Image Segmentation.

Neurocomputing.

---

# 🛡️ Academic Attribution

- **Institution:** Universidad de Sevilla — Departamento de Ciencias de la Computación e Inteligencia Artificial (CCIA)
- **Course:** Inteligencia Artificial (Ingeniería del Software)
- **Project Specifications:** Manuel Soriano Trigueros
- **Academic Year:** 2025/2026

---

## 📄 License

This repository is intended exclusively for educational and academic purposes within the Artificial Intelligence course at the University of Seville.

The DRIVE dataset remains subject to its original licensing and usage conditions.

The source code is distributed under the terms specified in the accompanying `LICENSE` file.