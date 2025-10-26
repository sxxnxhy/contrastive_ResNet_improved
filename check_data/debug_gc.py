"""Debug GC data loading"""
import numpy as np
from pathlib import Path
import config

# Find first GC file
gc_dir = Path(config.BASE_DATA_DIR) / config.GC_DIRS['2-CEES']
gc_files = list(gc_dir.glob('*.csv'))

if len(gc_files) == 0:
    print("No GC files found!")
else:
    print(f"Found {len(gc_files)} GC files")
    print(f"First file: {gc_files[0]}")
    
    # Try to load it
    print("\nTrying to load first GC file...")
    try:
        data = np.loadtxt(gc_files[0], delimiter=',', dtype=np.float32, comments='#')
        print(f"  Shape: {data.shape}")
        print(f"  Data preview (first 5 rows):")
        print(data[:5])
        
        if data.ndim == 1:
            print("\n  ⚠️  Data is 1D - only intensity values!")
        else:
            print(f"\n  Columns: {data.shape[1]}")
            if data.shape[1] >= 3:
                x = data[:, 1]  # Time
                y = data[:, 2]  # Intensity
                print(f"\n  X (time) range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"  Y (intensity) range: [{y.min():.4f}, {y.max():.4f}]")
                print(f"\n  Expected X range: [{config.GC_X_MIN}, {config.GC_X_MAX}]")
                
                # Check if x is actually in range
                if x.min() < config.GC_X_MIN or x.max() > config.GC_X_MAX:
                    print(f"\n  ⚠️  X range mismatch!")
                    print(f"     Actual: [{x.min():.4f}, {x.max():.4f}]")
                    print(f"     Expected: [{config.GC_X_MIN}, {config.GC_X_MAX}]")
                
                # Test interpolation
                print("\n  Testing interpolation to 4096 points...")
                common_x = np.linspace(config.GC_X_MIN, config.GC_X_MAX, config.COMMON_LENGTH)
                y_aligned = np.interp(common_x, x, y)
                
                print(f"    Result shape: {y_aligned.shape}")
                print(f"    Zeros: {(y_aligned == 0).sum()} / {len(y_aligned)} ({100*(y_aligned == 0).sum()/len(y_aligned):.1f}%)")
                print(f"    Non-zeros: {(y_aligned != 0).sum()}")
                print(f"    Min: {y_aligned.min():.6f}, Max: {y_aligned.max():.6f}")
                
                if (y_aligned == 0).sum() > len(y_aligned) * 0.9:
                    print("\n  🚨 PROBLEM: Over 90% zeros after interpolation!")
                    print("     This means x-values don't overlap with target range")
            else:
                print(f"  ⚠️  Expected 3 columns, got {data.shape[1]}")
                
    except Exception as e:
        print(f"  ❌ Error loading file: {e}")
        print("\n  Trying to read first few lines as text...")
        with open(gc_files[0], 'r') as f:
            for i, line in enumerate(f):
                if i < 10:
                    print(f"    Line {i}: {line.strip()}")
                else:
                    break
