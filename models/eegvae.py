"""
Basic EEGVAE for EEG Signal Reconstruction

A simple Variational Autoencoder for EEG signals using EEGNet encoder/decoder.
This is the base model without treatment effect prediction.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .eegnet import EEGNetEncoder, EEGNetDecoder


class EEGVAE(nn.Module):
    """
    Basic EEG Variational Autoencoder for signal reconstruction.
    
    Architecture:
    - Encoder Network: EEGNet-based encoder
    - Decoder Network: Transposed EEGNet decoder
    
    Args:
        n_channels: Number of EEG channels (default: 54)
        n_samples: Number of time samples (default: 256)
        latent_dim: Dimension of latent space (default: 8)
        F1: Number of temporal filters (default: 8)
        kern_length: Temporal kernel size (default: 128)
        beta: Weight for KL divergence (default: 1.0)
    """
    
    def __init__(
        self,
        n_channels=54,
        n_samples=256,
        latent_dim=8,
        F1=8,
        D=1,
        kern_length=128,
        beta=1.0
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder Network
        self.encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_samples=n_samples,
            latent_dim=latent_dim,
            F1=F1,
            D=D,
            kern_length=kern_length
        )
        
        # Decoder Network
        self.decoder = EEGNetDecoder(
            latent_dim=latent_dim,
            n_channels=n_channels,
            n_samples=n_samples,
            F1=F1,
            D=D,
            kern_length=kern_length
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, eeg):
        """
        Forward pass through EEGVAE.
        
        Args:
            eeg: EEG data (batch, channels, samples, 1)
        
        Returns:
            x_recon: Reconstructed EEG (batch, channels, samples, 1)
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
            z: Sampled latent vector (batch, latent_dim)
        """
        # Encode to latent space
        mu, logvar = self.encoder(eeg)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        x_recon = self.decoder(z)
        
        return x_recon, mu, logvar, z
    
    def loss_function(self, x, x_recon, mu, logvar):
        """
        Compute EEGVAE loss.
        
        L_total = L_recon + β * L_KL
        
        Args:
            x: Original EEG (batch, channels, samples, 1)
            x_recon: Reconstructed EEG (batch, channels, samples, 1)
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
        
        Returns:
            total_loss: Total loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss


def train_eegvae(
    model,
    train_loader,
    val_loader=None,
    epochs=100,
    lr=0.001,
    device='cpu',
    verbose=True,
    checkpoint_dir=None,
    save_best=True
):
    """
    Train EEGVAE model.
    
    Args:
        model: EEGVAE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        epochs: Number of training epochs
        lr: Learning rate
        device: Device to train on
        verbose: Print training progress
        checkpoint_dir: Directory to save checkpoints (default: None, don't save)
        save_best: If True, save best model based on validation loss
    
    Returns:
        history: Dictionary with training history
        best_model_path: Path to best model checkpoint (if saved)
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_recon': [],
        'train_kl': [],
        'val_loss': []
    }
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # Create checkpoint directory if specified
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics = {'loss': 0, 'recon': 0, 'kl': 0}
        
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            
            # Forward
            x_recon, mu, logvar, _ = model(eeg)
            total_loss, recon_loss, kl_loss = model.loss_function(
                eeg, x_recon, mu, logvar
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record
            train_metrics['loss'] += total_loss.item()
            train_metrics['recon'] += recon_loss.item()
            train_metrics['kl'] += kl_loss.item()
        
        # Average over batches
        n_batches = len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= n_batches
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)
                    
                    x_recon, mu, logvar, _ = model(eeg)
                    total_loss, _, _ = model.loss_function(
                        eeg, x_recon, mu, logvar
                    )
                    
                    val_loss += total_loss.item()
            
            val_loss /= len(val_loader)
            history['val_loss'].append(val_loss)
            
            # Save best model
            if save_best and checkpoint_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(checkpoint_dir, 'best_eegvae_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': history
                }, best_model_path)
                if verbose:
                    print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Recon: {train_metrics['recon']:.4f}, "
                  f"KL: {train_metrics['kl']:.4f})")
            if val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}")
    
    return history, best_model_path

