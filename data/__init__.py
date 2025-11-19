"""
Data Generation Package

This package contains utilities for generating synthetic EEG data.
"""

from .generate_synthetic_data import (
    generate_synthetic_eeg,
    generate_synthetic_clinical,
    generate_synthetic_outcomes,
    SyntheticEEGDataset,
    create_synthetic_dataset
)

__all__ = [
    'generate_synthetic_eeg',
    'generate_synthetic_clinical',
    'generate_synthetic_outcomes',
    'SyntheticEEGDataset',
    'create_synthetic_dataset'
]

