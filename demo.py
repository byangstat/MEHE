"""
Demonstration Script for EEGVAE and MEHE

This script demonstrates how to:
1. Generate synthetic EEG data
2. Train the basic EEGVAE model for reconstruction
3. Train the MEHE model for treatment effect estimation
4. Use the models for predictions

Usage:
    python demo.py
"""

import torch
from torch.utils.data import DataLoader
import numpy as np

from models import EEGVAE, train_eegvae, MEHE, train_mehe
from data import create_synthetic_dataset


def demo_eegvae(train_loader, val_loader, test_loader, device):
    """Demonstrate basic EEGVAE for reconstruction"""
    print("\n" + "=" * 70)
    print("EEGVAE Demonstration - Signal Reconstruction")
    print("=" * 70)
    
    # Create EEGVAE model
    print("\nCreating EEGVAE Model...")
    eegvae_model = EEGVAE(
        n_channels=54,
        n_samples=256,
        latent_dim=8,
        F1=8,
        beta=1.0
    )
    
    total_params = sum(p.numel() for p in eegvae_model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Train model
    print("\nTraining EEGVAE...")
    history, best_model_path = train_eegvae(
        model=eegvae_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=30,  # Reduced for demo
        lr=0.001,
        device=device,
        verbose=True,
        checkpoint_dir='checkpoints',
        save_best=True
    )
    
    print(f"\nTraining complete!")
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")
    
    # Evaluate reconstruction
    print("\nEvaluating Reconstruction Quality...")
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        eegvae_model.load_state_dict(checkpoint['model_state_dict'])
    
    eegvae_model.eval()
    total_recon_error = 0
    total_samples = 0
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            x_recon, mu, logvar, z = eegvae_model(eeg)
            
            recon_error = torch.mean((x_recon - eeg) ** 2).item()
            total_recon_error += recon_error * eeg.size(0)
            total_samples += eeg.size(0)
    
    avg_recon_error = total_recon_error / total_samples
    print(f"Average Reconstruction MSE: {avg_recon_error:.6f}")
    
    # Show example reconstruction
    print("\nExample Reconstruction:")
    test_batch = next(iter(test_loader))
    eeg_example = test_batch['eeg'][:3].to(device)
    with torch.no_grad():
        x_recon_example, mu_example, logvar_example, z_example = eegvae_model(eeg_example)
    
    for i in range(3):
        mse = torch.mean((x_recon_example[i] - eeg_example[i]) ** 2).item()
        print(f"  Sample {i+1}: MSE = {mse:.6f}, Latent dim = {z_example[i].shape[0]}")


def demo_mehe(train_loader, val_loader, test_loader, device):
    """Demonstrate MEHE for treatment effect estimation"""
    print("\n" + "=" * 70)
    print("MEHE Demonstration - Treatment Effect Estimation")
    print("=" * 70)
    
    # Create MEHE model
    print("\nCreating MEHE Model...")
    model = MEHE(
        n_channels=54,
        n_samples=256,
        latent_dim=8,
        n_clinical=47,
        F1=8,
        beta_1=10.0,
        beta_2=5.0
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Train model
    print("\nTraining MEHE...")
    history, best_model_path = train_mehe(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=50,  # Reduced for demo
        lr=0.001,
        device=device,
        verbose=True,
        checkpoint_dir='checkpoints',
        save_best=True
    )
    
    print(f"\nTraining complete!")
    if best_model_path:
        print(f"Best model saved to: {best_model_path}")
    
    # Evaluate model
    print("\nEvaluating MEHE...")
    if best_model_path:
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded best model from epoch {checkpoint['epoch']}")
    
    model.eval()
    
    # Evaluate on test set
    test_correct = 0
    test_total = 0
    all_cates = []
    all_itr = []
    
    with torch.no_grad():
        for batch in test_loader:
            eeg = batch['eeg'].to(device)
            clinical = batch['clinical'].to(device)
            treatment = batch['treatment'].to(device)
            y = batch['y'].to(device)
            
            # Predict outcomes
            x_recon, y_pred, mu, logvar = model(eeg, clinical, treatment)
            
            # Accuracy
            test_correct += ((y_pred > 0.5).float() == y).sum().item()
            test_total += y.size(0)
            
            # Estimate CATE and ITR
            cate = model.estimate_cate(eeg, clinical)
            itr = model.derive_itr(eeg, clinical)
            
            all_cates.append(cate.cpu().numpy())
            all_itr.append(itr.cpu().numpy())
    
    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # CATE statistics
    all_cates = np.concatenate(all_cates)
    print(f"\nCATE Statistics:")
    print(f"  Mean: {np.mean(all_cates):.4f}")
    print(f"  Std: {np.std(all_cates):.4f}")
    print(f"  Min: {np.min(all_cates):.4f}")
    print(f"  Max: {np.max(all_cates):.4f}")
    
    # ITR statistics
    all_itr = np.concatenate(all_itr)
    treatment_rate = np.mean(all_itr)
    print(f"\nITR Statistics:")
    print(f"  Recommended treatment rate: {treatment_rate:.4f}")
    print(f"  Recommended control rate: {1 - treatment_rate:.4f}")
    
    # Example usage
    print("\nExample Usage - Predicting for New Patients:")
    test_batch = next(iter(test_loader))
    eeg_example = test_batch['eeg'][:5].to(device)
    clinical_example = test_batch['clinical'][:5].to(device)
    
    # Predict outcomes for both treatment arms
    y_pred_0, y_pred_1, mu, logvar = model.predict_both_arms(
        eeg_example, clinical_example
    )
    
    # Estimate CATE
    cate = model.estimate_cate(eeg_example, clinical_example)
    
    # Derive ITR
    optimal_treatment = model.derive_itr(eeg_example, clinical_example)
    
    print("\nExample predictions for 5 patients:")
    for i in range(5):
        print(f"\nPatient {i+1}:")
        print(f"  Predicted outcome (control): {y_pred_0[i].item():.4f}")
        print(f"  Predicted outcome (treatment): {y_pred_1[i].item():.4f}")
        print(f"  Estimated CATE: {cate[i].item():.4f}")
        print(f"  Recommended treatment: {'Treatment' if optimal_treatment[i].item() == 1 else 'Control'}")


def main():
    print("=" * 70)
    print("EEGVAE and MEHE Demonstration")
    print("=" * 70)
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    
    # ========================================================================
    # Step 1: Generate Synthetic Data
    # ========================================================================
    print("\n" + "=" * 70)
    print("Step 1: Generating Synthetic Data")
    print("=" * 70)
    
    train_dataset, val_dataset, test_dataset = create_synthetic_dataset(
        n_train=500,
        n_val=100,
        n_test=200,
        n_channels=54,
        n_timepoints=256,
        n_clinical=47,
        effect_strength=1.0,
        seed=42
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # ========================================================================
    # Part 1: EEGVAE Demonstration
    # ========================================================================
    demo_eegvae(train_loader, val_loader, test_loader, device)
    
    # ========================================================================
    # Part 2: MEHE Demonstration
    # ========================================================================
    demo_mehe(train_loader, val_loader, test_loader, device)
    
    print("\n" + "=" * 70)
    print("Demonstration Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()

