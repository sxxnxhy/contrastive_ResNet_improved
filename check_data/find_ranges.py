"""Find actual X ranges for all GC and Raman files"""
import numpy as np
from pathlib import Path
import config
from tqdm import tqdm

def find_gc_range():
    """Find min/max X values across all GC files"""
    all_min = []
    all_max = []
    
    for class_name, rel_dir in config.GC_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        gc_files = list(full_dir.glob('*.csv'))[:500]  # Sample 500 files per class
        
        print(f"Checking {class_name}: {len(gc_files)} files...")
        for filepath in tqdm(gc_files, desc=class_name):
            try:
                data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
                if data.ndim == 2 and data.shape[1] >= 3:
                    x = data[:, 1]  # Time column
                    all_min.append(x.min())
                    all_max.append(x.max())
            except:
                pass
    
    if all_min:
        print(f"\n✅ GC X-range:")
        print(f"   Min: {min(all_min):.4f}")
        print(f"   Max: {max(all_max):.4f}")
        print(f"   Recommended: X_MIN={min(all_min):.4f}, X_MAX={max(all_max):.4f}")
    return min(all_min), max(all_max)

def find_raman_range():
    """Find min/max X values across all Raman files"""
    all_min = []
    all_max = []
    
    for class_name, rel_dir in config.RAMAN_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        raman_files = list(full_dir.glob('*.csv'))[:500]  # Sample 500 files per class
        
        print(f"Checking {class_name}: {len(raman_files)} files...")
        for filepath in tqdm(raman_files, desc=class_name):
            try:
                data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
                if data.ndim == 2:
                    x = data[:, 0]  # Wavenumber column
                    all_min.append(x.min())
                    all_max.append(x.max())
            except:
                pass
    
    if all_min:
        print(f"\n✅ Raman X-range:")
        print(f"   Min: {min(all_min):.4f}")
        print(f"   Max: {max(all_max):.4f}")
        print(f"   Recommended: X_MIN={min(all_min):.4f}, X_MAX={max(all_max):.4f}")
    return min(all_min), max(all_max)

if __name__ == "__main__":
    print("="*60)
    print("Finding GC X-range...")
    print("="*60)
    gc_min, gc_max = find_gc_range()
    
    print("\n" + "="*60)
    print("Finding Raman X-range...")
    print("="*60)
    raman_min, raman_max = find_raman_range()
    
    print("\n" + "="*60)
    print("SUMMARY - Update config.py with these values:")
    print("="*60)
    print(f"GC_X_MIN = {gc_min:.4f}")
    print(f"GC_X_MAX = {gc_max:.4f}")
    print(f"RAMAN_X_MIN = {raman_min:.4f}")
    print(f"RAMAN_X_MAX = {raman_max:.4f}")
