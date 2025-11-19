# Deep Representation Learning for Optimizing Treatment Decisions with Electroencephalogram Biomarkers

This is a minimal, publication-ready version of the MEHE code for heterogeneous treatment effect estimation. This version uses **fully synthetic data** to demonstrate model usage without requiring access to real patient data.

## Overview

The MEHE (Multi-head EEG Variational Autoencoder) is a deep learning model for estimating heterogeneous treatment effects (HTE) and deriving individualized treatment rules (ITRs) using EEG biomarkers and clinical features. The model consists of:

- **Encoder Network (M1)**: EEGNet-based encoder that extracts spatiotemporal features from EEG signals
- **HTE Prediction Module (M2)**: Separate prediction heads for each treatment arm
- **Decoder Network (M3)**: Transposed EEGNet decoder for reconstruction

[View MEHE Architecture Figure](MEHE.pdf)

## Repository Structure

```
publication_minimal/
├── models/
│   ├── __init__.py
│   ├── eegnet.py              # EEGNet encoder and decoder
│   ├── eegvae.py              # Basic EEGVAE for reconstruction
│   └── mehe.py                 # MEHE model for treatment effect estimation
├── data/
│   ├── __init__.py
│   └── generate_synthetic_data.py  # Synthetic data generation
├── demo.py                     # Demonstration script
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

## Installation

1. Clone or download this repository

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the demonstration script to see the model in action:

```bash
python demo.py
```

This will:
1. Generate synthetic EEG and clinical data
2. Train the EEGVAE model for signal reconstruction
3. Train the MEHE model for treatment effect estimation
4. Evaluate both models on test data
5. Demonstrate how to use the models for new predictions

## Usage

### Basic Usage - EEGVAE

```python
import torch
from models import EEGVAE, train_eegvae
from data import create_synthetic_dataset
from torch.utils.data import DataLoader

# Generate synthetic data
train_dataset, val_dataset, test_dataset = create_synthetic_dataset(
    n_train=500,
    n_val=100,
    n_test=200
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create EEGVAE model
model = EEGVAE(
    n_channels=54,
    n_samples=256,
    latent_dim=8,
    beta=1.0
)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
history, best_model_path = train_eegvae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    device=device
)
```

### Basic Usage - MEHE

```python
import torch
from models import MeheEEGVAE, train_mehe
from data import create_synthetic_dataset
from torch.utils.data import DataLoader

# Generate synthetic data
train_dataset, val_dataset, test_dataset = create_synthetic_dataset(
    n_train=500,
    n_val=100,
    n_test=200
)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create MEHE model
model = MeheEEGVAE(
    n_channels=54,
    n_samples=256,
    latent_dim=8,
    n_clinical=47,
    beta_1=10.0,
    beta_2=5.0
)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
history, best_model_path = train_mehe(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=100,
    device=device
)
```


