# 🐾 Footprint Classification Project

A deep learning project that classifies **animal** and **dinosaur footprints** using Convolutional Neural Networks (CNNs) and Transfer Learning with TensorFlow/Keras.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Datasets](#datasets)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Acknowledgements & Data Sources](#acknowledgements--data-sources)
- [License](#license)

---

## Overview

This project explores image classification techniques on two distinct footprint datasets:

1. **Animal Footprint Classification** — Classifying footprints of domestic cats, domestic dogs, and European badgers using both a custom CNN and MobileNetV2 transfer learning.
2. **Dinosaur Footprint Classification** — Classifying dinosaur track silhouettes into groups (Theropoda, Ornithopoda, Stegosauria) using data from the DinoTracker research project.

---

## Project Structure

```
Footprint_Classification_Project/
│
├── data/                              # Animal footprint images
│   ├── train/                         # Training set (163 images)
│   ├── valid/                         # Validation set (31 images)
│   └── test/                          # Test set (61 images)
│       ├── domestic_cat/
│       ├── domestic_dog/
│       └── european_badger/
│
├── notebooks/
│   ├── CNN/                           # Custom CNN approach
│   │   ├── 01_setup_and_explore.ipynb
│   │   ├── 02_data_loading.ipynb
│   │   ├── 03_build_model.ipynb
│   │   ├── 04_train_model.ipynb
│   │   ├── 05_evaluate_model.ipynb
│   │   └── complete_pipeline.ipynb    # End-to-end CNN pipeline
│   └── TransferLearning/             # MobileNetV2 approach
│       ├── 01_understanding_transfer_learning.ipynb
│       ├── 02_data_setup.ipynb
│       ├── 03_build_model.ipynb
│       ├── 04_train_model.ipynb
│       ├── 05_evaluate_model.ipynb
│       └── complete_pipeline.ipynb    # End-to-end TL pipeline
│
├── models/                            # Saved models & visualizations
│
├── Dino Footprint/                    # Dinosaur footprint sub-project
│   ├── data/                          # DinoTracker dataset files
│   │   ├── images_compressed.npz
│   │   ├── names.npy
│   │   └── tracks.xlsx
│   ├── models/                        # Dino classifier outputs
│   ├── dino_pipeline.ipynb            # Complete dino classifier
│   └── inspect_data.py               # Data inspection utility
│
├── requirements.txt
└── README.md
```

---

## Datasets

### Animal Footprints

- **Classes**: Domestic Cat, Domestic Dog, European Badger
- **Total images**: 255 (163 train / 31 validation / 61 test)
- **Format**: RGB photographs organized in class-based folders
- **Source**: Derived from the [AnimalClue YOLO Detection](https://huggingface.co/spaces/risashinoda/animalclue_yolo_det) space on Hugging Face by [risashinoda](https://huggingface.co/risashinoda)

### Dinosaur Footprints

- **Classes**: Theropoda (967), Ornithopoda (661), Stegosauria (52)
- **Total images**: 1,680 matched samples (from 1,976 track entries)
- **Format**: Binary silhouette images stored as compressed NumPy arrays (`.npz`) with metadata in Excel (`.xlsx`)
- **Source**: [DinoTracker](https://github.com/gregh83/DinoTracker) by Gregor Hartmann et al.

---

## Models

| Model | Architecture | Dataset | Test Accuracy | Model Size |
|---|---|---|---|---|
| Custom CNN | Conv2D from scratch | Animal Footprints | 49.0% | ~128 MB |
| Transfer Learning | MobileNetV2 (fine-tuned) | Animal Footprints | 60.7% | ~20 MB |
| Dino CNN | Conv2D from scratch | Dinosaur Footprints | ~69% (train) | ~58 MB |

Each notebook pipeline is structured in 5 progressive steps:
1. **Setup & Exploration** — Environment config, data overview
2. **Data Loading** — Image generators, augmentation, preprocessing
3. **Model Building** — Architecture definition
4. **Training** — Training loop with callbacks (early stopping, checkpointing)
5. **Evaluation** — Confusion matrix, sample predictions, metrics

---

## Installation

### Prerequisites

- Python 3.10+
- pip

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd Footprint_Classification_Project

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Linux/macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Animal Footprint Classification

Run the complete pipelines using Jupyter:

```bash
jupyter notebook
```

Then open either:
- `notebooks/CNN/complete_pipeline.ipynb` — Custom CNN approach
- `notebooks/TransferLearning/complete_pipeline.ipynb` — MobileNetV2 approach

Or follow the step-by-step notebooks (`01` through `05`) for a detailed walkthrough.

### Dinosaur Footprint Classification

```bash
# Inspect the raw dino data
python "Dino Footprint/inspect_data.py"

# Run the full dino pipeline
jupyter notebook "Dino Footprint/dino_pipeline.ipynb"
```

---

## Results

### Animal Footprints — CNN vs Transfer Learning

| Metric | Custom CNN | Transfer Learning (MobileNetV2) |
|---|---|---|
| Test Accuracy | 49.0% | **60.7%** |
| Training Epochs | 14 (early stop) | 10 + 5 (two-phase) |
| Best Validation Acc | ~55% | ~68% |

### Dinosaur Footprints

| Metric | Value |
|---|---|
| Best Validation Accuracy | ~77% (epoch 1) |
| Final Training Accuracy | ~63% |
| Classes Predicted | 2 of 3 (Stegosauria never predicted) |

> **Note**: Performance is limited by the small and imbalanced datasets. Future work could improve results through larger datasets, class weighting, advanced augmentation, and stronger backbone architectures.

---

## Acknowledgements & Data Sources

### 🐾 Animal Footprint Data — AnimalClue

The animal footprint images used in this project are derived from the **AnimalClue** dataset, a large-scale benchmark for recognizing animals by their indirect traces (ICCV 2025 Highlight).

- **Project Page**: [AnimalClue: Recognizing Animals by their Traces](https://dahlian00.github.io/AnimalCluePage/)
- **Paper**: [arXiv:2507.20240](https://arxiv.org/abs/2507.20240)
- **Demo**: [AnimalClue YOLO Detection](https://huggingface.co/spaces/risashinoda/animalclue_yolo_det) on Hugging Face
- **Dataset**: [risashinoda/footprint_yolo](https://huggingface.co/datasets/risashinoda/footprint_yolo) on Hugging Face
- **Code**: [dahlian00/AnimalClue](https://github.com/dahlian00/AnimalClue) on GitHub
- **Authors**: Risa Shinoda, Nakamasa Inoue, Iro Laina, Christian Rupprecht, Hirokatsu Kataoka
- **Affiliations**: University of Osaka, Kyoto University, Tokyo Institute of Technology, AIST, University of Oxford (VGG)

As requested by the authors, we cite the following paper:

```bibtex
@article{shinoda2025animalcluerecognizinganimalstraces,
  title={AnimalClue: Recognizing Animals by their Traces},
  author={Risa Shinoda and Nakamasa Inoue and Iro Laina and Christian Rupprecht and Hirokatsu Kataoka},
  year={2025},
  eprint={2507.20240},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2507.20240},
}
```

We used footprint images from 3 species (domestic cat, domestic dog, European badger) from their footprint dataset. We thank the AnimalClue team for making their dataset and research openly available.

---

### 🦕 Dinosaur Footprint Data — DinoTracker

The dinosaur footprint data used in this project comes from the **DinoTracker** project. As requested by the authors, we cite the following paper:

> G. Hartmann, T. Blakesley, P.E. dePolo, & S.L. Brusatte, *Identifying variation in dinosaur footprints and classifying problematic specimens via unbiased unsupervised machine learning*, Proc. Natl. Acad. Sci. U.S.A. **123** (5) e2527222122, [https://doi.org/10.1073/pnas.2527222122](https://doi.org/10.1073/pnas.2527222122) (2026).

- **Source**: [DinoTracker — GitHub Repository](https://github.com/gregh83/DinoTracker)
- **Authors**: Gregor Hartmann, T. Blakesley, P.E. dePolo, & S.L. Brusatte
- **License**: GPL-3.0
- **Description**: An app for dinosaur footprint analysis via disentangled variational autoencoder. The project has been covered by international media including The Guardian, BBC Newsround, Reuters, and The Conversation.
- **Contact**: 📧 [gregor.hartmann@helmholtz-berlin.de](mailto:gregor.hartmann@helmholtz-berlin.de)

We are grateful to the DinoTracker team for making their dataset and research openly available.

---

## Technologies Used

- [TensorFlow](https://www.tensorflow.org/) / [Keras](https://keras.io/) — Deep learning framework
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Pre-trained backbone for transfer learning
- [NumPy](https://numpy.org/) — Numerical computing
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — Visualization
- [scikit-learn](https://scikit-learn.org/) — Evaluation metrics
- [openpyxl](https://openpyxl.readthedocs.io/) — Excel file parsing
- [Jupyter Notebook](https://jupyter.org/) — Interactive development

---

## License

This project is provided for educational and research purposes. Please note:
- The **AnimalClue** dataset images are linked to individual observation IDs, and usage of each image must comply with the license of the corresponding observation. The project website is licensed under [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).
- The **DinoTracker** data and code are distributed under the [GPL-3.0 License](https://github.com/gregh83/DinoTracker/blob/main/LICENSE).
- Please refer to the individual data source licenses for any restrictions on redistribution or commercial use.
