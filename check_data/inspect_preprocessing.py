"""Detailed inspection of preprocessing pipeline"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import config

def inspect_single_file(filepath, modality):
    """Inspect preprocessing step-by-step for one file"""
    print(f"\n{'='*60}")
    print(f"File: {filepath.name}")
    print(f"Modality: {modality.upper()}")
    print(f"{'='*60}")
    
    # Step 1: Load raw data
    try:
        if modality == 'raman':
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            if data.ndim == 2:
                x = data[:, 0]
                y = data[:, 1]
            else:
                y = data
                x = np.arange(len(y), dtype=np.float32)
        else:  # gc
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
            if data.ndim == 2 and data.shape[1] >= 3:
                x = data[:, 1]  # Time column
                y = data[:, 2]  # Intensity column
            else:
                print("❌ Invalid GC data format")
                return
    except Exception as e:
        print(f"❌ Error loading: {e}")
        return
    
    print(f"\n1. RAW DATA:")
    print(f"   X shape: {x.shape}")
    print(f"   Y shape: {y.shape}")
    print(f"   X range: [{x.min():.4f}, {x.max():.4f}]")
    print(f"   Y range: [{y.min():.4f}, {y.max():.4f}]")
    print(f"   Y non-zero: {(y != 0).sum()} / {len(y)} ({100*(y!=0).sum()/len(y):.1f}%)")
    
    # Step 2: Create common x-axis
    if modality == 'raman':
        common_x = np.linspace(config.RAMAN_X_MIN, config.RAMAN_X_MAX, config.COMMON_LENGTH)
        print(f"\n2. TARGET X-AXIS (Raman):")
        print(f"   Range: [{config.RAMAN_X_MIN}, {config.RAMAN_X_MAX}]")
    else:
        common_x = np.linspace(config.GC_X_MIN, config.GC_X_MAX, config.COMMON_LENGTH)
        print(f"\n2. TARGET X-AXIS (GC):")
        print(f"   Range: [{config.GC_X_MIN}, {config.GC_X_MAX}]")
    print(f"   Points: {config.COMMON_LENGTH}")
    
    # Step 3: Interpolate
    if modality == 'raman':
        y_aligned = np.interp(common_x, x, y, left=0.0, right=0.0)
    else:
        y_aligned = np.interp(common_x, x, y)
    
    print(f"\n3. AFTER INTERPOLATION:")
    print(f"   Shape: {y_aligned.shape}")
    print(f"   Range: [{y_aligned.min():.4f}, {y_aligned.max():.4f}]")
    print(f"   Zeros: {(y_aligned == 0).sum()} / {len(y_aligned)} ({100*(y_aligned==0).sum()/len(y_aligned):.1f}%)")
    print(f"   Non-zeros: {(y_aligned != 0).sum()}")
    
    # Check overlap
    data_x_min, data_x_max = x.min(), x.max()
    target_x_min, target_x_max = common_x.min(), common_x.max()
    
    overlap_min = max(data_x_min, target_x_min)
    overlap_max = min(data_x_max, target_x_max)
    overlap = max(0, overlap_max - overlap_min)
    target_range = target_x_max - target_x_min
    overlap_pct = 100 * overlap / target_range
    
    print(f"\n4. X-AXIS OVERLAP:")
    print(f"   Data X: [{data_x_min:.4f}, {data_x_max:.4f}]")
    print(f"   Target X: [{target_x_min:.4f}, {target_x_max:.4f}]")
    print(f"   Overlap: {overlap:.4f} / {target_range:.4f} ({overlap_pct:.1f}%)")
    
    if overlap_pct < 50:
        print(f"   ⚠️  WARNING: Less than 50% overlap!")
    
    # Step 4: Normalize
    y_min, y_max = y_aligned.min(), y_aligned.max()
    y_normalized = (y_aligned - y_min) / (y_max - y_min + 1e-10)
    
    print(f"\n5. AFTER NORMALIZATION:")
    print(f"   Range: [{y_normalized.min():.6f}, {y_normalized.max():.6f}]")
    print(f"   Mean: {y_normalized.mean():.6f}")
    print(f"   Std: {y_normalized.std():.6f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Raw data
    axes[0, 0].plot(x, y, 'b-', linewidth=0.5)
    axes[0, 0].set_title('Step 1: Raw Data')
    axes[0, 0].set_xlabel('Original X')
    axes[0, 0].set_ylabel('Intensity')
    axes[0, 0].grid(True, alpha=0.3)
    
    # After interpolation (before normalization)
    axes[0, 1].plot(common_x, y_aligned, 'g-', linewidth=0.5)
    axes[0, 1].set_title(f'Step 3: After Interpolation to {config.COMMON_LENGTH} points')
    axes[0, 1].set_xlabel('Target X')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.3, label='Zero line')
    axes[0, 1].legend()
    
    # After normalization
    axes[1, 0].plot(y_normalized, 'r-', linewidth=0.5)
    axes[1, 0].set_title('Step 5: After Normalization [0, 1]')
    axes[1, 0].set_xlabel('Index (0-4095)')
    axes[1, 0].set_ylabel('Normalized Intensity')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Histogram of normalized values
    axes[1, 1].hist(y_normalized, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('Distribution of Normalized Values')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].axvline(x=0, color='r', linestyle='--', alpha=0.5, label='Zero')
    axes[1, 1].legend()
    
    plt.tight_layout()
    output_name = f'preprocessing_{modality}_{filepath.stem}.png'
    plt.savefig(output_name, dpi=150, bbox_inches='tight')
    print(f"\n📊 Saved visualization: {output_name}")
    plt.close()
    
    return y_normalized

def main():
    print("="*60)
    print("DETAILED PREPROCESSING INSPECTION")
    print("="*60)
    
    # Check 2 files from each modality/class
    test_cases = [
        ('raman', '2-CEES'),
        ('gc', '2-CEES'),
        ('raman', 'DMMP'),
        ('gc', 'DMMP'),
    ]
    
    for modality, class_name in test_cases:
        if modality == 'raman':
            data_dir = Path(config.BASE_DATA_DIR) / config.RAMAN_DIRS[class_name]
        else:
            data_dir = Path(config.BASE_DATA_DIR) / config.GC_DIRS[class_name]
        
        files = list(data_dir.glob('*.csv'))[:2]  # First 2 files
        
        for filepath in files:
            result = inspect_single_file(filepath, modality)
    
    print("\n" + "="*60)
    print("SUMMARY OF CONFIG SETTINGS:")
    print("="*60)
    print(f"COMMON_LENGTH: {config.COMMON_LENGTH}")
    print(f"RAMAN_X_MIN: {config.RAMAN_X_MIN}")
    print(f"RAMAN_X_MAX: {config.RAMAN_X_MAX}")
    print(f"GC_X_MIN: {config.GC_X_MIN}")
    print(f"GC_X_MAX: {config.GC_X_MAX}")
    print("="*60)

if __name__ == "__main__":
    main()
