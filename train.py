
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os
from pathlib import Path

import config
from model import CrossModalContrastiveModel
from dataset import prepare_dataloader


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, initial_temperature: float, min_temp: float, max_temp: float, 
                 label_smoothing: float):
        super().__init__()
        self.label_smoothing = label_smoothing  # ε for label smoothing
        
        # Store min/max in log-space for efficient clamping
        self.min_temp_log = np.log(min_temp)
        self.max_temp_log = np.log(max_temp)
        
        # Create a learnable parameter for temperature (logit_scale)
        # We store log(T) and exponentiate it
        self.logit_scale = nn.Parameter(
            torch.ones([]) * np.log(initial_temperature)
        )
        
    def get_temperature(self) -> float:
        # Clamp logit_scale
        logit_scale_clamped = torch.clamp(
            self.logit_scale, self.min_temp_log, self.max_temp_log
        )
        # T = exp(log(T))
        return logit_scale_clamped.exp().item()
        
        
        
    def forward(self, embeddings: torch.Tensor, class_labels: torch.Tensor, 
                modalities: torch.Tensor) -> dict:

        batch_size = embeddings.size(0)
        device = embeddings.device
        
        # Class labels to indices
        class_indices = torch.argmax(class_labels, dim=1)  # (B,)
        
        # Compute similarity matrix (cosine similarity, range [-1, 1])
        sim_matrix = embeddings @ embeddings.T  # (B, B)
        
        logit_scale_clamped = torch.clamp(
            self.logit_scale, self.min_temp_log, self.max_temp_log
        )
        temperature = logit_scale_clamped.exp()
        
        # Apply temperature scaling
        logits = sim_matrix / temperature  # (B, B)
        
        # Masks for Supervised Contrastive Loss
        class_match = class_indices.unsqueeze(0) == class_indices.unsqueeze(1)  # (B, B)
        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        
        # Positive: same class, but NOT self
        positive_mask = class_match & ~self_mask
        
        # Negative: different class (self-comparison is already false)
        negative_mask = ~class_match
        
        
        # Check if we have valid pairs
        if not positive_mask.any():
            return {
                'loss': torch.tensor(0.0, device=device, requires_grad=True),
                'pos_sim': 0.0,
                'neg_sim': 0.0,
                'num_pairs': 0
            }
        
        # Compute InfoNCE loss (vectorized)
        LARGE_NUM = 1e9
        logits_pos = torch.where(positive_mask, logits, torch.tensor(-LARGE_NUM, device=device))
        logits_neg = torch.where(negative_mask, logits, torch.tensor(-LARGE_NUM, device=device))
        
        # For numerical stability, subtract max
        logits_max = torch.max(
            torch.max(logits_pos, dim=1, keepdim=True)[0],
            torch.max(logits_neg, dim=1, keepdim=True)[0]
        )
        logits_pos = logits_pos - logits_max
        logits_neg = logits_neg - logits_max
        
        # Compute exp and sum
        exp_pos = torch.exp(logits_pos) * positive_mask.float()  # (B, B)
        exp_neg = torch.exp(logits_neg) * negative_mask.float()  # (B, B)
        
        sum_exp_pos = exp_pos.sum(dim=1)  # (B,)
        sum_exp_neg = exp_neg.sum(dim=1)  # (B,)
        
        # Standard InfoNCE loss
        has_positive = positive_mask.any(dim=1)  # (B,)
        
        if has_positive.any():
            # Compute log probability
            log_prob = torch.log(sum_exp_pos[has_positive] + 1e-8) - \
                       torch.log(sum_exp_pos[has_positive] + sum_exp_neg[has_positive] + 1e-8)
            
            # Apply label smoothing
            if self.label_smoothing > 0:
                loss = -(1.0 - self.label_smoothing) * log_prob.mean()
            else:
                loss = -log_prob.mean()
        else:
            loss = torch.tensor(0.0, device=device, requires_grad=True)
        
        # Compute metrics (on raw similarities, not scaled)
        with torch.no_grad():
            if positive_mask.any():
                pos_sim = (sim_matrix * positive_mask.float()).sum() / positive_mask.sum()
                pos_sim = pos_sim.item()
            else:
                pos_sim = 0.0
            
            if negative_mask.any():
                neg_sim = (sim_matrix * negative_mask.float()).sum() / negative_mask.sum()
                neg_sim = neg_sim.item()
            else:
                neg_sim = 0.0
            
            num_pairs = positive_mask.sum().item()
        
        return {
            'loss': loss,
            'pos_sim': pos_sim,
            'neg_sim': neg_sim,
            'num_pairs': int(num_pairs)
        }


def train_epoch(model: nn.Module, train_loader: DataLoader, 
                criterion: SupervisedContrastiveLoss, optimizer: torch.optim.Optimizer,
                device: torch.device, epoch: int) -> dict:
    model.train()
    criterion.train()
    
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (spectra, labels_and_mod) in enumerate(pbar):
        spectra = spectra.to(device)
        labels_and_mod = labels_and_mod.to(device)
        
        # Split labels and modality
        class_labels = labels_and_mod[:, :-1]  # (B, num_classes)
        modalities = labels_and_mod[:, -1]     # (B,)
        
        # Forward pass
        embeddings = model(spectra, modalities)
        
        # Compute loss
        loss_dict = criterion(
            embeddings, 
            class_labels, 
            modalities
        )
        
        loss = loss_dict['loss']
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping (stability)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
        torch.nn.utils.clip_grad_norm_(criterion.parameters(), max_norm=config.GRADIENT_CLIP_NORM)
        
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        total_pos_sim += loss_dict['pos_sim']
        total_neg_sim += loss_dict['neg_sim']
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'pos': f'{loss_dict["pos_sim"]:.3f}',
            'neg': f'{loss_dict["neg_sim"]:.3f}',
            'temp': f'{criterion.get_temperature():.3f}',
            'pairs': loss_dict['num_pairs']
        })
    
    return {
        'loss': total_loss / num_batches,
        'pos_sim': total_pos_sim / num_batches,
        'neg_sim': total_neg_sim / num_batches
    }


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, 
             criterion: SupervisedContrastiveLoss, device: torch.device) -> dict:
    model.eval()
    criterion.eval() 
    
    total_loss = 0.0
    total_pos_sim = 0.0
    total_neg_sim = 0.0
    num_batches = 0
    
    for spectra, labels_and_mod in tqdm(val_loader, desc='Validation'):
        spectra = spectra.to(device)
        labels_and_mod = labels_and_mod.to(device)
        
        class_labels = labels_and_mod[:, :-1]
        modalities = labels_and_mod[:, -1]
        
        embeddings = model(spectra, modalities)
        
        loss_dict = criterion(
            embeddings, 
            class_labels, 
            modalities
        )
        
        total_loss += loss_dict['loss'].item()
        total_pos_sim += loss_dict['pos_sim']
        total_neg_sim += loss_dict['neg_sim']
        num_batches += 1
    
    return {
        'loss': total_loss / num_batches,
        'pos_sim': total_pos_sim / num_batches,
        'neg_sim': total_neg_sim / num_batches
    }


def train():
    print("="*80)
    print("Cross-Modal Contrastive Learning for Raman-GC Spectroscopy")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  - Classes: {config.CLASSES}")
    print(f"  - Embedding dimension: {config.EMBEDDING_DIM}")
    print(f"  - Hidden dimension: {config.HIDDEN_DIM}")
    print(f"  - Batch size: {config.BATCH_SIZE}")
    print(f"  - Epochs: {config.EPOCHS}")
    print(f"  - Learning rate: {config.LEARNING_RATE}")
    print(f"  - Weight decay: {config.WEIGHT_DECAY}")
    print(f"  - Dropout: 0.3")
    print(f"  - Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print(f"\nHyperparameters (from config.py):")
    print(f"  - Initial Temperature: {config.INITIAL_TEMPERATURE}")
    print(f"  - Temp Range: ({config.MIN_TEMP}, {config.MAX_TEMP})")
    print(f"  - Label Smoothing: {config.LABEL_SMOOTHING}")
    print(f"  - Gradient Clip Norm: {config.GRADIENT_CLIP_NORM}")
    print()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data loaders
    print("Preparing data loaders...")
    train_loader, val_loader, class_names = prepare_dataloader(
        use_lazy_loading=config.USE_LAZY_LOADING,
        use_parallel=config.USE_PARALLEL,
        max_workers=config.MAX_WORKERS,
        num_workers=config.NUM_WORKERS
    )
    
    # Model
    print("\nInitializing model...")
    model = CrossModalContrastiveModel(
        hidden_dim=config.HIDDEN_DIM,
        embed_dim=config.EMBEDDING_DIM
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    # Loss with regularization
    criterion = SupervisedContrastiveLoss(
        initial_temperature=config.INITIAL_TEMPERATURE,
        min_temp=config.MIN_TEMP,
        max_temp=config.MAX_TEMP,
        label_smoothing=config.LABEL_SMOOTHING
    ).to(device)
    
    print(f"Total parameters (model): {total_params:,}")
    print(f"Trainable parameters (model): {trainable_params:,}")
    print(f"Trainable parameters (criterion): 1 (logit_scale)")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(criterion.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler (cosine annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.EPOCHS,
        eta_min=config.LEARNING_RATE * 0.01
    )
    
    # Training loop
    best_val_loss = float('inf')
    
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80 + "\n")
    
    for epoch in range(1, config.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{config.EPOCHS}")
        print("-" * 60)
        
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f"Train - Loss: {train_metrics['loss']:.4f}, "
              f"Pos Sim: {train_metrics['pos_sim']:.3f}, "
              f"Neg Sim: {train_metrics['neg_sim']:.3f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, "
              f"Pos Sim: {val_metrics['pos_sim']:.3f}, "
              f"Neg Sim: {val_metrics['neg_sim']:.3f}")
        
        print(f"Temperature: {criterion.get_temperature():.4f}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Scheduler step
        scheduler.step()
        
        # Save best model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            # Create directory if needed
            model_dir = Path(config.MODEL_DIR).parent
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(), # Save criterion state (for logit_scale)
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': {
                    'embedding_dim': config.EMBEDDING_DIM,
                    'hidden_dim': config.HIDDEN_DIM,
                    'classes': config.CLASSES
                }
            }, config.MODEL_DIR)
            
            print(f"✓ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = Path(config.MODEL_DIR).parent / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, checkpoint_path)
            print(f"✓ Saved checkpoint at epoch {epoch}")
    
    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*80)


if __name__ == '__main__':
    train()