"""
Generate Synthetic EEG Data for Demonstration

This module generates simple synthetic EEG data using random numbers
to demonstrate the model functionality. This allows for publication without
data sharing restrictions.
"""

import numpy as np
import torch
from torch.utils.data import Dataset


def generate_synthetic_eeg(
    n_samples,
    n_channels=54,
    n_timepoints=256,
    sampling_rate=128,
    seed=42
):
    """
    Generate synthetic EEG data using random numbers.
    
    Args:
        n_samples: Number of samples to generate
        n_channels: Number of EEG channels (default: 54)
        n_timepoints: Number of time points per sample (default: 256)
        sampling_rate: Sampling rate in Hz (default: 128)
        seed: Random seed for reproducibility
    
    Returns:
        eeg_data: Synthetic EEG data (n_samples, n_channels, n_timepoints)
    """
    np.random.seed(seed)
    
    # Generate random EEG data
    eeg_data = np.random.randn(n_samples, n_channels, n_timepoints)
    
    # Normalize to have zero mean and unit variance per channel
    for i in range(n_samples):
        for ch in range(n_channels):
            eeg_data[i, ch, :] = (eeg_data[i, ch, :] - np.mean(eeg_data[i, ch, :])) / (np.std(eeg_data[i, ch, :]) + 1e-8)
    
    return eeg_data


def generate_synthetic_clinical(
    n_samples,
    n_features=47,
    seed=42
):
    """
    Generate synthetic clinical features using random numbers.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of clinical features (default: 47)
        seed: Random seed for reproducibility
    
    Returns:
        clinical_data: Synthetic clinical data (n_samples, n_features)
    """
    np.random.seed(seed)
    
    # Generate random clinical features
    clinical_data = np.random.randn(n_samples, n_features)
    
    return clinical_data


def generate_synthetic_outcomes(
    eeg_data,
    clinical_data,
    treatment,
    effect_strength=1.0,
    seed=42
):
    """
    Generate synthetic outcomes using random numbers.
    
    Args:
        eeg_data: EEG data (n_samples, n_channels, n_timepoints)
        clinical_data: Clinical features (n_samples, n_features)
        treatment: Treatment assignment (n_samples,) with values 0 or 1
        effect_strength: Strength of treatment effect (default: 1.0)
        seed: Random seed for reproducibility
    
    Returns:
        outcomes: Binary outcomes (n_samples,)
    """
    np.random.seed(seed)
    n_samples = len(eeg_data)
    
    # Simple random outcome generation with treatment effect
    base_prob = 0.3 + 0.2 * np.random.rand(n_samples)  # Base probability between 0.3-0.5
    treatment_effect = effect_strength * 0.1 * np.random.rand(n_samples)  # Small treatment effect
    
    # Generate outcomes
    outcomes = np.zeros(n_samples)
    for i in range(n_samples):
        if treatment[i] == 0:
            prob = np.clip(base_prob[i], 0.01, 0.99)
        else:
            prob = np.clip(base_prob[i] + treatment_effect[i], 0.01, 0.99)
        outcomes[i] = np.random.binomial(1, prob)
    
    return outcomes


class SyntheticEEGDataset(Dataset):
    """
    PyTorch Dataset for synthetic EEG data.
    """
    
    def __init__(self, eeg, clinical, treatment, outcomes):
        self.eeg = torch.FloatTensor(eeg).unsqueeze(-1)  # Add channel dimension
        self.clinical = torch.FloatTensor(clinical)
        self.treatment = torch.LongTensor(treatment)
        self.outcomes = torch.FloatTensor(outcomes).reshape(-1, 1)
    
    def __len__(self):
        return len(self.eeg)
    
    def __getitem__(self, idx):
        return {
            'eeg': self.eeg[idx],
            'clinical': self.clinical[idx],
            'treatment': self.treatment[idx],
            'y': self.outcomes[idx]
        }


def create_synthetic_dataset(
    n_train=500,
    n_val=100,
    n_test=200,
    n_channels=54,
    n_timepoints=256,
    n_clinical=47,
    effect_strength=1.0,
    seed=42
):
    """
    Create a complete synthetic dataset for training, validation, and testing.
    
    Args:
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples
        n_channels: Number of EEG channels
        n_timepoints: Number of time points per sample
        n_clinical: Number of clinical features
        effect_strength: Strength of treatment effect
        seed: Random seed for reproducibility
    
    Returns:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
    """
    # Generate training data
    train_eeg = generate_synthetic_eeg(
        n_train, n_channels, n_timepoints, seed=seed
    )
    train_clinical = generate_synthetic_clinical(
        n_train, n_clinical, seed=seed
    )
    train_treatment = np.random.binomial(1, 0.5, n_train)
    train_outcomes = generate_synthetic_outcomes(
        train_eeg, train_clinical, train_treatment,
        effect_strength=effect_strength, seed=seed
    )
    
    # Generate validation data
    val_eeg = generate_synthetic_eeg(
        n_val, n_channels, n_timepoints, seed=seed + 1
    )
    val_clinical = generate_synthetic_clinical(
        n_val, n_clinical, seed=seed + 1
    )
    val_treatment = np.random.binomial(1, 0.5, n_val)
    val_outcomes = generate_synthetic_outcomes(
        val_eeg, val_clinical, val_treatment,
        effect_strength=effect_strength, seed=seed + 1
    )
    
    # Generate test data
    test_eeg = generate_synthetic_eeg(
        n_test, n_channels, n_timepoints, seed=seed + 2
    )
    test_clinical = generate_synthetic_clinical(
        n_test, n_clinical, seed=seed + 2
    )
    test_treatment = np.random.binomial(1, 0.5, n_test)
    test_outcomes = generate_synthetic_outcomes(
        test_eeg, test_clinical, test_treatment,
        effect_strength=effect_strength, seed=seed + 2
    )
    
    # Create datasets
    train_dataset = SyntheticEEGDataset(
        train_eeg, train_clinical, train_treatment, train_outcomes
    )
    val_dataset = SyntheticEEGDataset(
        val_eeg, val_clinical, val_treatment, val_outcomes
    )
    test_dataset = SyntheticEEGDataset(
        test_eeg, test_clinical, test_treatment, test_outcomes
    )
    
    return train_dataset, val_dataset, test_dataset

