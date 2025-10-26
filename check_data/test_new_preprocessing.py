"""Test new preprocessing with unified length (4096) for both Raman and GC"""
import numpy as np
import torch
from dataset import load_spectrum, preprocess_spectrum
import config

print("=" * 80)
print("Testing Unified Preprocessing Strategy (Both → 4096)")
print("=" * 80)

# Test 1: Raman spectrum with partial coverage
print("\n[Test 1] Raman spectrum (partial coverage: 500-1500 cm^-1)")
# Simulate a Raman file with 500 points covering 500-1500 cm^-1
x_raman = np.linspace(500, 1500, 500, dtype=np.float32)
y_raman = np.random.rand(500).astype(np.float32) * 100
y_raman[100:150] += 200  # Add a peak at ~600-700 cm^-1
y_raman[300:350] += 150  # Add a peak at ~1100-1200 cm^-1

y_processed = preprocess_spectrum(x_raman, y_raman, 'raman')
print(f"  Input: x range {x_raman.min():.1f}-{x_raman.max():.1f}, {len(x_raman)} points")
print(f"  Output: {len(y_processed)} points (should be {config.COMMON_LENGTH})")
print(f"  Output range: {y_processed.min():.4f}-{y_processed.max():.4f}")

# Check zero padding
common_x = np.linspace(config.RAMAN_X_MIN, config.RAMAN_X_MAX, config.COMMON_LENGTH)
idx_500 = np.argmin(np.abs(common_x - 500))
idx_1500 = np.argmin(np.abs(common_x - 1500))
print(f"  Before data (x<500): mean = {y_processed[:idx_500].mean():.4f} (should be ~0)")
print(f"  After data (x>1500): mean = {y_processed[idx_1500:].mean():.4f} (should be ~0)")
print(f"  Data region (500-1500): mean = {y_processed[idx_500:idx_1500].mean():.4f} (should be >0)")

# Test 2: GC spectrum (full coverage, downsample from 5347 to 4096)
print("\n[Test 2] GC spectrum (full coverage: 0.1-17.0 min, 5347→4096)")
x_gc = np.linspace(0.1, 17.0, 5347, dtype=np.float32)
y_gc = np.random.rand(5347).astype(np.float32) * 100

y_processed_gc = preprocess_spectrum(x_gc, y_gc, 'gc')
print(f"  Input: x range {x_gc.min():.1f}-{x_gc.max():.1f}, {len(x_gc)} points")
print(f"  Output: {len(y_processed_gc)} points (should be {config.COMMON_LENGTH})")
print(f"  Output range: {y_processed_gc.min():.4f}-{y_processed_gc.max():.4f}")
print(f"  Resolution: {5347} → {config.COMMON_LENGTH} ({config.COMMON_LENGTH/5347*100:.1f}% of original)")

# Test 3: Model forward pass
print("\n[Test 3] Model compatibility check")
from model_mobilenet import RamanGCCLIP

model = RamanGCCLIP(
    input_dim=config.COMMON_LENGTH,
    embedding_dim=config.EMBEDDING_DIM,
    width_mult=config.WIDTH_MULT
)

# Create mixed batch (now all same length!)
raman_samples = torch.randn(2, config.COMMON_LENGTH)
gc_samples = torch.randn(2, config.COMMON_LENGTH)
mixed_batch = torch.cat([raman_samples, gc_samples], dim=0)
modality_mask = torch.tensor([0, 0, 1, 1])  # 0=raman, 1=gc

print(f"  Mixed batch shape: {mixed_batch.shape}")
print(f"  Modality mask: {modality_mask.tolist()}")

try:
    with torch.no_grad():
        embeddings = model(mixed_batch, modality_mask)
    print(f"  ✓ Forward pass successful!")
    print(f"  Embeddings shape: {embeddings.shape} (should be (4, {config.EMBEDDING_DIM}))")
    print(f"  Embeddings normalized: {torch.norm(embeddings, dim=1)}")
except Exception as e:
    print(f"  ✗ Forward pass failed: {e}")

print("\n" + "=" * 80)
print("Model parameter count:")
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")
print(f"  Raman encoder: {sum(p.numel() for p in model.raman_encoder.parameters()):,}")
print(f"  GC encoder: {sum(p.numel() for p in model.gc_encoder.parameters()):,}")
print(f"\n  Input dimension: {config.COMMON_LENGTH}")
print(f"  Embedding dimension: {config.EMBEDDING_DIM}")
print("=" * 80)

