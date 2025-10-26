"""Check if data preprocessing is working correctly"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import prepare_dataloader
import config

def check_data():
    print("Loading data...")
    train_loader, _, _ = prepare_dataloader(
        use_lazy_loading=config.USE_LAZY_LOADING,
        use_parallel=config.USE_PARALLEL,
        max_workers=config.MAX_WORKERS,
        num_workers=0  # Single worker for debugging
    )
    
    print("\nFetching one batch...")
    spectra, labels = next(iter(train_loader))
    
    print(f"\nBatch Info:")
    print(f"  Spectra shape: {spectra.shape}")  # Should be [batch_size, 4096]
    print(f"  Labels shape: {labels.shape}")    # Should be [batch_size, 5] (4 classes + 1 modality)
    
    # Check class distribution in batch
    class_labels = labels[:, :-1]
    modality = labels[:, -1]
    
    class_indices = torch.argmax(class_labels, dim=1)
    print(f"\nClass distribution in batch:")
    for i, cls in enumerate(config.CLASSES):
        count = (class_indices == i).sum().item()
        print(f"  {cls}: {count}")
    
    # Check modality distribution
    raman_count = (modality == 0).sum().item()
    gc_count = (modality == 1).sum().item()
    print(f"\nModality distribution:")
    print(f"  Raman: {raman_count}")
    print(f"  GC: {gc_count}")
    
    # Check spectrum statistics
    print(f"\nSpectrum statistics:")
    print(f"  Min: {spectra.min().item():.6f}")
    print(f"  Max: {spectra.max().item():.6f}")
    print(f"  Mean: {spectra.mean().item():.6f}")
    print(f"  Std: {spectra.std().item():.6f}")
    
    # Check for zero padding in Raman
    raman_mask = modality == 0
    if raman_mask.any():
        raman_spectra = spectra[raman_mask]
        print(f"\nRaman spectra check:")
        print(f"  Shape: {raman_spectra.shape}")
        print(f"  Zeros percentage: {(raman_spectra == 0).float().mean().item()*100:.2f}%")
        
        # Plot first Raman spectrum
        plt.figure(figsize=(12, 4))
        plt.plot(raman_spectra[0].numpy())
        plt.title("Sample Raman Spectrum (after preprocessing)")
        plt.xlabel("Index (0-4095)")
        plt.ylabel("Normalized Intensity")
        plt.grid(True, alpha=0.3)
        plt.savefig("sample_raman.png", dpi=150, bbox_inches='tight')
        print("  Saved plot: sample_raman.png")
    
    # Check GC
    gc_mask = modality == 1
    if gc_mask.any():
        gc_spectra = spectra[gc_mask]
        print(f"\nGC spectra check:")
        print(f"  Shape: {gc_spectra.shape}")
        print(f"  Zeros percentage: {(gc_spectra == 0).float().mean().item()*100:.2f}%")
        
        # Plot first GC spectrum
        plt.figure(figsize=(12, 4))
        plt.plot(gc_spectra[0].numpy())
        plt.title("Sample GC Spectrum (after preprocessing)")
        plt.xlabel("Index (0-4095)")
        plt.ylabel("Normalized Intensity")
        plt.grid(True, alpha=0.3)
        plt.savefig("sample_gc.png", dpi=150, bbox_inches='tight')
        print("  Saved plot: sample_gc.png")
    
    # Check if same class samples are actually similar
    print(f"\n" + "="*60)
    print("Checking similarity between same-class samples:")
    
    for cls_idx, cls_name in enumerate(config.CLASSES):
        mask = class_indices == cls_idx
        if mask.sum() < 2:
            continue
        
        cls_spectra = spectra[mask][:10]  # First 10 samples
        
        # Compute pairwise cosine similarity
        cls_spectra_norm = cls_spectra / (cls_spectra.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(cls_spectra_norm, cls_spectra_norm.t())
        
        # Get upper triangle (excluding diagonal)
        mask_upper = torch.triu(torch.ones_like(similarity), diagonal=1).bool()
        similarities = similarity[mask_upper]
        
        print(f"  {cls_name}:")
        print(f"    Mean similarity: {similarities.mean().item():.4f}")
        print(f"    Std similarity: {similarities.std().item():.4f}")
        print(f"    Min similarity: {similarities.min().item():.4f}")
        print(f"    Max similarity: {similarities.max().item():.4f}")

if __name__ == "__main__":
    check_data()
