# Histology Batch Correction for Valve Tissue Analysis

## Overview
This project implements a deep learning method for batch correction in valve histology images. It specifically addresses the challenge of comparing diseased valve tissue with healthy valve tissue by correcting batch effects that arise from differences in sample collection, processing, and imaging protocols.

## Problem Statement
Healthy heart valve tissue is difficult to collect, leading to challenges in comparative studies between healthy and diseased valve tissue. Batch effects caused by different collection times, preservation methods, staining protocols, and imaging conditions can introduce technical variability that confounds biological differences. This project provides a computational solution to adjust for these batch effects, enabling more accurate comparisons.

## Methods
We implement several state-of-the-art deep learning approaches for batch correction:

1. **CycleGAN with Self-Attention**: Unsupervised image-to-image translation with attention mechanisms
2. **Contrastive Learning Framework**: Preserves biological features while removing batch-specific information
3. **Domain Adversarial Neural Networks**: Learns batch-invariant features
4. **Diffusion Models**: For high-fidelity image transformation
5. **Feature Disentanglement**: Separates biological features from technical batch effects

## Project Structure
```
histology_batch_correction/
├── data/                      # Data storage
│   ├── raw/                   # Original images
│   └── processed/             # Preprocessed and batch-corrected images
├── src/                       # Source code
│   ├── data/                  # Data loading and preprocessing
│   ├── models/                # Model implementations
│   ├── preprocessing/         # Image preprocessing pipeline
│   └── utils/                 # Utility functions
├── testing/                   # Testing framework
│   ├── benchmarks/            # Benchmark implementations
│   └── data/                  # Test data
├── notebooks/                 # Jupyter notebooks for exploration and visualization
├── results/                   # Results and outputs
│   ├── figures/               # Generated figures for manuscript
│   └── models/                # Trained model weights
└── docs/                      # Documentation
```

## Installation
```bash
# Clone the repository
git clone https://github.com/menyawino/BatchGAN.git

# Install dependencies
pip install -r requirements.txt
```

## Usage
The main workflow consists of:

1. **Data Preprocessing**: 
   ```bash
   python -m src.preprocessing.preprocess --data_dir path/to/data
   ```

2. **Model Training**:
   ```bash
   python -m src.models.train --config configs/cyclegan.yaml
   ```

3. **Batch Correction**:
   ```bash
   python -m src.models.predict --model_path results/models/best_model.pth --input_dir data/raw --output_dir data/processed
   ```

4. **Evaluation**:
   ```bash
   python -m testing.benchmarks.evaluate --corrected_dir data/processed --metrics structural,biological
   ```
