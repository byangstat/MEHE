"""
MEHE Model Package

This package contains the model architecture for the MEHE
for heterogeneous treatment effect estimation.
"""

from .eegnet import EEGNetEncoder, EEGNetDecoder
from .eegvae import EEGVAE, train_eegvae
from .mehe import MeheEEGVAE, train_mehe

__all__ = [
    'EEGNetEncoder',
    'EEGNetDecoder',
    'EEGVAE',
    'train_eegvae',
    'MeheEEGVAE',
    'train_mehe'
]

