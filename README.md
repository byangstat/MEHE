# MEHE: Minimal Publication Version

This is a minimal, publication-ready version of the MEHE code for heterogeneous treatment effect estimation. This version uses **fully synthetic data** to demonstrate model usage without requiring access to real patient data.

## Overview

The MEHE (Multi-head EEG Variational Autoencoder) is a deep learning model for estimating heterogeneous treatment effects (HTE) and deriving individualized treatment rules (ITRs) using EEG biomarkers and clinical features. The model consists of:

- **Encoder Network (M1)**: EEGNet-based encoder that extracts spatiotemporal features from EEG signals
- **HTE Prediction Module (M2)**: Separate prediction heads for each treatment arm
- **Decoder Network (M3)**: Transposed EEGNet decoder for reconstruction

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

# Reconstruct EEG signals
model.eval()
with torch.no_grad():
    eeg = ...  # EEG data (batch, channels, samples, 1)
    x_recon, mu, logvar, z = model(eeg)
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

### Predicting Treatment Effects

```python
# Load trained model
checkpoint = torch.load('checkpoints/best_model.pt', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Predict for new patients
eeg = ...  # EEG data (batch, channels, samples, 1)
clinical = ...  # Clinical features (batch, n_clinical)

# Estimate CATE (Conditional Average Treatment Effect)
cate = model.estimate_cate(eeg, clinical)

# Derive ITR (Individualized Treatment Rule)
optimal_treatment = model.derive_itr(eeg, clinical)
```

## Model Architecture

This repository contains two models: **EEGVAE** (basic reconstruction) and **MEHE** (treatment effect estimation).

### EEGVAE Architecture

The **EEGVAE** is a basic Variational Autoencoder for EEG signal reconstruction:

```
Input EEG (batch, channels, samples, 1)
    ↓
┌─────────────────────────────────────┐
│  Encoder (EEGNet-based)             │
│  - Temporal Convolution (F1 filters) │
│  - Depthwise Spatial Convolution     │
│  - Separable Convolution             │
│  - FC: flatten_size → 128            │
│  - FC: 128 → latent_dim (μ)          │
│  - FC: 128 → latent_dim (log σ²)     │
└─────────────────────────────────────┘
    ↓
Latent Space: z ~ N(μ, σ²)
    ↓
┌─────────────────────────────────────┐
│  Decoder (Transposed EEGNet)        │
│  - FC: latent_dim → 128             │
│  - FC: 128 → flatten_size           │
│  - Transposed Separable Convolution  │
│  - Transposed Depthwise Convolution  │
│  - Transposed Temporal Convolution   │
└─────────────────────────────────────┘
    ↓
Reconstructed EEG (batch, channels, samples, 1)
```

**Loss Function:**
```
L_EEGVAE = L_recon + β * L_KL
         = MSE(x, x_recon) + β * KL(N(μ,σ²) || N(0,1))
```

**Parameters:**
- Default: ~27,872 parameters
- Encoder: EEGNet with F1=8 temporal filters
- Decoder: Transposed EEGNet
- Latent dimension: 8 (default)

### MEHE Architecture

The **MEHE** (Multi-head EEG Variational Autoencoder) extends EEGVAE with treatment effect prediction:

```
Input: EEG (batch, channels, samples, 1) + Clinical (batch, n_clinical) + Treatment (batch,)
    ↓
┌─────────────────────────────────────┐
│  (M1) Encoder Network               │
│  - Same as EEGVAE encoder           │
│  - Output: μ, log σ²                │
└─────────────────────────────────────┘
    ↓
Latent Space: z ~ N(μ, σ²)
    ↓
    ├─────────────────────────────────┐
    │                                 │
    ↓                                 ↓
┌──────────────────┐        ┌──────────────────┐
│  (M3) Decoder     │        │  (M2) Prediction  │
│  - Reconstructs  │        │  - Head 0 (t=0)  │
│    EEG signal     │        │  - Head 1 (t=1)  │
└──────────────────┘        └──────────────────┘
    │                                 │
    │                    ┌──────────────────────────┐
    │                    │  Input: [μ, σ, clinical] │
    │                    │  - Linear(→64) + ReLU   │
    │                    │  - Dropout(0.3)         │
    │                    │  - Linear(→32) + ReLU   │
    │                    │  - Dropout(0.3)         │
    │                    │  - Linear(→1) + Sigmoid │
    │                    └──────────────────────────┘
    │                                 │
    ↓                                 ↓
Reconstructed EEG          Predicted Outcome (batch, 1)
```

**Components:**

1. **Encoder (M1)**: EEGNet-based encoder
   - Input: EEG data (batch, channels, samples, 1)
   - Architecture:
     - Temporal convolution: 1×kern_length (default: 1×128)
     - Depthwise spatial convolution: n_channels×1
     - Separable convolution: 1×16
     - Fully connected layers: flatten_size → 128 → latent_dim
   - Output: Latent mean μ and log-variance log σ² (batch, latent_dim)

2. **HTE Prediction Module (M2)**: Separate prediction heads
   - Input: Concatenated [μ, σ, clinical_features] (batch, 2×latent_dim + n_clinical)
   - Architecture (each head):
     - Linear(→64) + ReLU + Dropout(0.3)
     - Linear(→32) + ReLU + Dropout(0.3)
     - Linear(→1) + Sigmoid
   - Output: Predicted outcome probability for each treatment arm (batch, 1)

3. **Decoder (M3)**: Transposed EEGNet decoder
   - Input: Latent representation z (batch, latent_dim)
   - Architecture (inverse of encoder):
     - Fully connected: latent_dim → 128 → flatten_size
     - Transposed separable convolution
     - Transposed depthwise spatial convolution
     - Transposed temporal convolution
   - Output: Reconstructed EEG signal (batch, channels, samples, 1)

**Loss Function:**
```
L_MEHE = L_recon + β₁ * L_KL + β₂ * L_pred
       = MSE(x, x_recon) + β₁ * KL(N(μ,σ²) || N(0,1)) + β₂ * BCE(y, y_pred)
```

**Parameters:**
- Default: ~40,290 parameters
- Encoder: Same as EEGVAE (~13,936 params)
- Decoder: Same as EEGVAE (~13,936 params)
- Prediction heads: 2 × ~6,209 params each
- Default hyperparameters: β₁=10.0, β₂=5.0

## Loss Functions

### EEGVAE Loss

```
L_EEGVAE = L_recon + β * L_KL
         = MSE(x, x_recon) + β * KL(N(μ,σ²) || N(0,1))
```

where:
- `L_recon`: Reconstruction loss (MSE between original and reconstructed EEG)
- `L_KL`: KL divergence between latent distribution and standard normal
- `β`: Weight hyperparameter (default: 1.0)

### MEHE Loss

```
L_MEHE = L_recon + β₁ * L_KL + β₂ * L_pred
       = MSE(x, x_recon) + β₁ * KL(N(μ,σ²) || N(0,1)) + β₂ * BCE(y, y_pred)
```

where:
- `L_recon`: Reconstruction loss (MSE)
- `L_KL`: KL divergence loss
- `L_pred`: Prediction loss (binary cross-entropy for outcome prediction)
- `β₁`, `β₂`: Weight hyperparameters (default: 10.0, 5.0)

## Synthetic Data

The synthetic data generator creates simple random data for demonstration purposes.

**Note**: This synthetic data is for demonstration purposes only and does not represent real patient data.

## Training Configuration

Default training hyperparameters:

**EEGVAE:**
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32 (adjustable)
- Epochs: 100 (adjustable)
- Beta (KL weight): 1.0

**MEHE:**
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 32 (adjustable)
- Epochs: 100 (adjustable)
- Beta_1 (KL weight): 10.0
- Beta_2 (prediction weight): 5.0

## Citation

If you use this code, please cite:

```
Yang, Kim, & Wang (2025). Deep Representation Learning for Optimizing 
Individualized Treatment Decisions with EEG Biomarkers.
```

## License

[Add your license information here]

## Contact

[Add contact information here]

## Notes

- This is a minimal version for publication purposes
- All data is synthetic - no real patient data is included
- The model architecture and training code are identical to the full version
- For questions or issues, please refer to the main repository or contact the authors

