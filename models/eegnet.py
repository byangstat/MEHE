"""
EEGNet Encoder and Decoder for Multi-head EEGVAE

Based on:
Lawhern et al. (2018). EEGNet: a compact convolutional neural network for 
EEG-based brain-computer interfaces. Journal of Neural Engineering.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNetEncoder(nn.Module):
    """
    EEGNet encoder for extracting spatiotemporal features from EEG.
    
    Args:
        n_channels: Number of EEG channels (default: 54)
        n_samples: Number of time samples per epoch (default: 256)
        latent_dim: Dimension of latent space (default: 8)
        F1: Number of temporal filters (default: 8)
        D: Depth multiplier for spatial filters (default: 1)
        kern_length: Temporal filter kernel size (default: 128)
        dropout: Dropout rate (default: 0.25)
    """
    
    def __init__(
        self,
        n_channels=54,
        n_samples=256,
        latent_dim=8,
        F1=8,
        D=1,
        kern_length=128,
        dropout=0.25
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.F1 = F1
        self.F2 = F1 * D
        
        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        
        # Block 2: Depthwise spatial convolution
        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1),
            groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropout)
        
        # Block 3: Separable convolution
        self.separable = nn.Conv2d(
            F1 * D, self.F2, (1, 16),
            padding='same', bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropout)
        
        # Calculate flattened size
        self.flatten_size = self._get_flatten_size()
        
        # Latent space projection
        self.fc = nn.Linear(self.flatten_size, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
    
    def _get_flatten_size(self):
        """Calculate size after convolutions"""
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels, self.n_samples)
            x = self.pool1(self.depthwise(self.conv1(x)))
            x = self.pool2(self.separable(x))
            return x.numel()
    
    def forward(self, x):
        """
        Args:
            x: EEG data (batch, channels, samples, 1) or (batch, 1, channels, samples)
        
        Returns:
            mu: Mean of latent distribution (batch, latent_dim)
            logvar: Log variance of latent distribution (batch, latent_dim)
        """
        # Handle input shape
        if x.dim() == 4 and x.shape[-1] == 1:
            x = x.squeeze(-1).unsqueeze(1)  # (batch, 1, channels, samples)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        
        # Block 2
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Block 3
        x = self.separable(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        
        # Flatten and project to latent space
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc(x))
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        return mu, logvar


class EEGNetDecoder(nn.Module):
    """
    Transpose of EEGNet encoder for reconstructing EEG from latent space.
    
    Args:
        latent_dim: Dimension of latent space
        n_channels: Number of EEG channels
        n_samples: Number of time samples
        F1: Number of temporal filters
        D: Depth multiplier
        kern_length: Temporal filter kernel size
    """
    
    def __init__(
        self,
        latent_dim=8,
        n_channels=54,
        n_samples=256,
        F1=8,
        D=1,
        kern_length=128
    ):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.F1 = F1
        self.F2 = F1 * D
        
        # Calculate intermediate sizes
        self.h_after_pool = n_samples // (4 * 8)  # After two pooling layers
        self.w_after_pool = 1
        
        # Project from latent to feature space
        self.fc = nn.Linear(latent_dim, 128)
        self.fc_out = nn.Linear(128, self.F2 * self.w_after_pool * self.h_after_pool)
        
        # Transpose Block 3
        self.bn3 = nn.BatchNorm2d(self.F2)
        self.upsample2 = nn.Upsample(scale_factor=(1, 8), mode='nearest')
        self.deconv_sep = nn.ConvTranspose2d(
            self.F2, F1 * D, (1, 16),
            padding=(0, 7), bias=False
        )
        
        # Transpose Block 2
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.upsample1 = nn.Upsample(scale_factor=(1, 4), mode='nearest')
        self.deconv_depth = nn.ConvTranspose2d(
            F1 * D, F1, (n_channels, 1),
            bias=False
        )
        
        # Transpose Block 1
        self.bn1 = nn.BatchNorm2d(F1)
        pad = (kern_length - 1) // 2
        self.deconv1 = nn.ConvTranspose2d(
            F1, 1, (1, kern_length),
            padding=(0, pad), bias=False
        )
    
    def forward(self, z):
        """
        Args:
            z: Latent representation (batch, latent_dim)
        
        Returns:
            x_recon: Reconstructed EEG (batch, channels, samples, 1)
        """
        # Project to feature space
        x = F.relu(self.fc(z))
        x = self.fc_out(x)
        x = x.view(-1, self.F2, self.w_after_pool, self.h_after_pool)
        
        # Transpose Block 3
        x = self.bn3(x)
        x = F.elu(x)
        x = self.upsample2(x)
        x = self.deconv_sep(x)
        
        # Transpose Block 2
        x = self.bn2(x)
        x = F.elu(x)
        x = self.upsample1(x)
        x = self.deconv_depth(x)
        
        # Transpose Block 1
        x = self.bn1(x)
        x = self.deconv1(x)
        
        # Ensure correct output size and reshape
        if x.shape[3] > self.n_samples:
            x = x[:, :, :, :self.n_samples]
        elif x.shape[3] < self.n_samples:
            pad_size = self.n_samples - x.shape[3]
            x = F.pad(x, (0, pad_size, 0, 0))
        
        # Reshape to (batch, channels, samples, 1)
        x = x.squeeze(1).unsqueeze(-1)
        
        return x

