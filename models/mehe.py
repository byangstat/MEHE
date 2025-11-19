"""
MEHE for Heterogeneous Treatment Effect Estimation

Main contribution of the paper:
Yang, Kim, & Wang (2025). Deep Representation Learning for Optimizing 
Individualized Treatment Decisions with EEG Biomarkers.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .eegnet import EEGNetEncoder, EEGNetDecoder


class MEHE(nn.Module):
    """
    MEHE (Multi-head EEG Variational Autoencoder) for HTE estimation.
    
    Architecture:
    - (M1) Encoder Network: EEGNet-based encoder
    - (M2) HTE Prediction Module: Separate heads for each treatment arm
    - (M3) Decoder Network: Transposed EEGNet decoder
    
    Args:
        n_channels: Number of EEG channels (default: 54)
        n_samples: Number of time samples (default: 256)
        latent_dim: Dimension of latent space (default: 8)
        n_clinical: Number of clinical features (default: 47)
        F1: Number of temporal filters (default: 8)
        kern_length: Temporal kernel size (default: 128)
        beta_1: Weight for KL divergence (default: 10.0)
        beta_2: Weight for prediction loss (default: 5.0)
    """
    
    def __init__(
        self,
        n_channels=54,
        n_samples=256,
        latent_dim=8,
        n_clinical=47,
        F1=8,
        D=1,
        kern_length=128,
        beta_1=10.0,
        beta_2=5.0
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.latent_dim = latent_dim
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        
        # (M1) Encoder Network - shared across treatment arms
        self.encoder = EEGNetEncoder(
            n_channels=n_channels,
            n_samples=n_samples,
            latent_dim=latent_dim,
            F1=F1,
            D=D,
            kern_length=kern_length
        )
        
        # (M3) Decoder Network
        self.decoder = EEGNetDecoder(
            latent_dim=latent_dim,
            n_channels=n_channels,
            n_samples=n_samples,
            F1=F1,
            D=D,
            kern_length=kern_length
        )
        
        # (M2) HTE Prediction Module - separate heads for each arm
        # Input: latent_dim (mu) + latent_dim (std) + n_clinical
        input_dim = 2 * latent_dim + n_clinical
        
        # Head for control (t=0)
        self.head_0 = self._build_prediction_head(input_dim)
        
        # Head for treatment (t=1)
        self.head_1 = self._build_prediction_head(input_dim)
    
    def _build_prediction_head(self, input_dim):
        """Build MLP for outcome prediction"""
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, eeg, clinical, treatment):
        """
        Forward pass through MEHE.
        
        Args:
            eeg: EEG data (batch, channels, samples, 1)
            clinical: Clinical features (batch, n_clinical)
            treatment: Treatment indicator (batch,) with values 0 or 1
        
        Returns:
            x_recon: Reconstructed EEG (batch, channels, samples, 1)
            y_pred: Predicted outcome for assigned treatment (batch, 1)
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
        """
        # (M1) Encode to latent space
        mu, logvar = self.encoder(eeg)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # (M3) Decode
        x_recon = self.decoder(z)
        
        # (M2) Predict outcome using appropriate head
        # Concatenate mu, std, and clinical features
        std = torch.exp(0.5 * logvar)
        combined = torch.cat([mu, std, clinical], dim=1)
        
        # Select appropriate head based on treatment
        y_pred = torch.zeros(eeg.size(0), 1, device=eeg.device)
        
        # For control group (t=0)
        mask_0 = (treatment == 0)
        if mask_0.any():
            y_pred[mask_0] = self.head_0(combined[mask_0])
        
        # For treatment group (t=1)
        mask_1 = (treatment == 1)
        if mask_1.any():
            y_pred[mask_1] = self.head_1(combined[mask_1])
        
        return x_recon, y_pred, mu, logvar
    
    def predict_both_arms(self, eeg, clinical):
        """
        Predict potential outcomes under both treatment arms.
        
        This is used for estimating HTE and deriving optimal ITR.
        
        Args:
            eeg: EEG data (batch, channels, samples, 1)
            clinical: Clinical features (batch, n_clinical)
        
        Returns:
            y_pred_0: Predicted outcome under control (batch, 1)
            y_pred_1: Predicted outcome under treatment (batch, 1)
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
        """
        # Encode
        mu, logvar = self.encoder(eeg)
        
        # Prepare input for prediction heads
        std = torch.exp(0.5 * logvar)
        combined = torch.cat([mu, std, clinical], dim=1)
        
        # Predict for both arms
        y_pred_0 = self.head_0(combined)
        y_pred_1 = self.head_1(combined)
        
        return y_pred_0, y_pred_1, mu, logvar
    
    def estimate_cate(self, eeg, clinical):
        """
        Estimate Conditional Average Treatment Effect (CATE).
        
        CATE(x) = E[Y(1) - Y(0) | X = x]
                = μ_1(x) - μ_0(x)
        
        Args:
            eeg: EEG data (batch, channels, samples, 1)
            clinical: Clinical features (batch, n_clinical)
        
        Returns:
            cate: Estimated CATE (batch, 1)
        """
        y_pred_0, y_pred_1, _, _ = self.predict_both_arms(eeg, clinical)
        return y_pred_1 - y_pred_0
    
    def derive_itr(self, eeg, clinical):
        """
        Derive optimal Individualized Treatment Rule (ITR).
        
        For outcomes where higher is better (e.g., response):
        d*(x) = I[CATE(x) > 0] = I[μ_1(x) > μ_0(x)]
        
        Args:
            eeg: EEG data (batch, channels, samples, 1)
            clinical: Clinical features (batch, n_clinical)
        
        Returns:
            optimal_treatment: Optimal treatment (batch,) with values 0 or 1
        """
        y_pred_0, y_pred_1, _, _ = self.predict_both_arms(eeg, clinical)
        optimal_treatment = (y_pred_1 > y_pred_0).long().squeeze()
        return optimal_treatment
    
    def loss_function(self, x, x_recon, y, y_pred, mu, logvar):
        """
        Compute MEHE loss (Equation 4 in paper).
        
        L_total = L_VAE + β_2 * L_pred
                = L_recon + β_1 * L_KL + β_2 * L_pred
        
        Args:
            x: Original EEG (batch, channels, samples, 1)
            x_recon: Reconstructed EEG (batch, channels, samples, 1)
            y: True outcomes (batch, 1)
            y_pred: Predicted outcomes (batch, 1)
            mu: Latent mean (batch, latent_dim)
            logvar: Latent log variance (batch, latent_dim)
        
        Returns:
            total_loss: Total loss
            recon_loss: Reconstruction loss
            kl_loss: KL divergence
            pred_loss: Prediction loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction='mean')
        
        # KL divergence (Equation 3 in paper)
        kl_loss = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        
        # Prediction loss (binary cross-entropy, Equation 3)
        pred_loss = F.binary_cross_entropy(y_pred, y, reduction='mean')
        
        # Total loss (Equation 4)
        total_loss = recon_loss + self.beta_1 * kl_loss + self.beta_2 * pred_loss
        
        return total_loss, recon_loss, kl_loss, pred_loss


def train_mehe(
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
    Train MEHE model (Algorithm 1 in paper).
    
    Args:
        model: MEHE model
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
        'train_pred': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_model_path = None
    
    # Create checkpoint directory if specified
    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_metrics = {'loss': 0, 'recon': 0, 'kl': 0, 'pred': 0}
        
        for batch in train_loader:
            eeg = batch['eeg'].to(device)
            clinical = batch['clinical'].to(device)
            treatment = batch['treatment'].to(device)
            y = batch['y'].to(device)
            
            # Forward
            x_recon, y_pred, mu, logvar = model(eeg, clinical, treatment)
            total_loss, recon_loss, kl_loss, pred_loss = model.loss_function(
                eeg, x_recon, y, y_pred, mu, logvar
            )
            
            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            # Record
            train_metrics['loss'] += total_loss.item()
            train_metrics['recon'] += recon_loss.item()
            train_metrics['kl'] += kl_loss.item()
            train_metrics['pred'] += pred_loss.item()
        
        # Average over batches
        n_batches = len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= n_batches
        
        history['train_loss'].append(train_metrics['loss'])
        history['train_recon'].append(train_metrics['recon'])
        history['train_kl'].append(train_metrics['kl'])
        history['train_pred'].append(train_metrics['pred'])
        
        # Validation
        if val_loader is not None:
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)
                    clinical = batch['clinical'].to(device)
                    treatment = batch['treatment'].to(device)
                    y = batch['y'].to(device)
                    
                    x_recon, y_pred, mu, logvar = model(eeg, clinical, treatment)
                    total_loss, _, _, _ = model.loss_function(
                        eeg, x_recon, y, y_pred, mu, logvar
                    )
                    
                    val_loss += total_loss.item()
                    val_correct += ((y_pred > 0.5).float() == y).sum().item()
                    val_total += y.size(0)
            
            val_loss /= len(val_loader)
            val_acc = val_correct / val_total
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            # Save best model
            if save_best and checkpoint_dir is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'val_acc': val_acc,
                    'history': history
                }, best_model_path)
                if verbose:
                    print(f"  ✓ Saved best model (val_loss={val_loss:.4f})")
        
        # Print progress
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} "
                  f"(Recon: {train_metrics['recon']:.4f}, "
                  f"KL: {train_metrics['kl']:.4f}, "
                  f"Pred: {train_metrics['pred']:.4f})")
            if val_loader is not None:
                print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    return history, best_model_path

