"""
MEHE Model Package

This package contains the model architecture for the MEHE
for heterogeneous treatment effect estimation.
"""

from .eegnet import EEGNetEncoder, EEGNetDecoder
from .eegvae import EEGVAE, train_eegvae
from .mehe import MEHE, train_mehe

__all__ = [
    'EEGNetEncoder',
    'EEGNetDecoder',
    'EEGVAE',
    'train_eegvae',
    'MEHE',
    'train_mehe'
]

