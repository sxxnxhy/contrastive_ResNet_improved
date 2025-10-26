"""Analyze X-range per class for Raman data"""
import numpy as np
from pathlib import Path
import config
from tqdm import tqdm

def analyze_raman_ranges_by_class():
    """Check if different classes have different X ranges"""
    
    class_ranges = {}
    
    for class_name, rel_dir in config.RAMAN_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        raman_files = list(full_dir.glob('*.csv'))[:200]  # Sample 200 files per class
        
        all_min = []
        all_max = []
        all_lengths = []
        
        print(f"\n{'='*60}")
        print(f"Analyzing {class_name}: {len(raman_files)} files")
        print(f"{'='*60}")
        
        for filepath in tqdm(raman_files, desc=class_name):
            try:
                data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
                if data.ndim == 2:
                    x = data[:, 0]
                    all_min.append(x.min())
                    all_max.append(x.max())
                    all_lengths.append(len(x))
            except:
                pass
        
        if all_min:
            min_x = min(all_min)
            max_x = max(all_max)
            avg_min = np.mean(all_min)
            avg_max = np.mean(all_max)
            avg_len = np.mean(all_lengths)
            
            class_ranges[class_name] = {
                'min': min_x,
                'max': max_x,
                'avg_min': avg_min,
                'avg_max': avg_max,
                'avg_length': avg_len
            }
            
            print(f"\n  X-range:")
            print(f"    Min: {min_x:.4f} (avg: {avg_min:.4f})")
            print(f"    Max: {max_x:.4f} (avg: {avg_max:.4f})")
            print(f"    Span: {max_x - min_x:.4f}")
            print(f"    Avg length: {avg_len:.1f} points")
            
            # Check how much overlaps with config range
            config_min = config.RAMAN_X_MIN
            config_max = config.RAMAN_X_MAX
            
            overlap_min = max(min_x, config_min)
            overlap_max = min(max_x, config_max)
            
            if overlap_max > overlap_min:
                overlap = overlap_max - overlap_min
                data_span = max_x - min_x
                config_span = config_max - config_min
                overlap_pct = 100 * overlap / config_span
                
                print(f"\n  Overlap with config [{config_min:.4f}, {config_max:.4f}]:")
                print(f"    Overlap: [{overlap_min:.4f}, {overlap_max:.4f}]")
                print(f"    Overlap: {overlap:.4f} / {config_span:.4f} ({overlap_pct:.1f}%)")
                
                # Check if data extends beyond config
                if min_x < config_min:
                    print(f"    ⚠️  Data starts before config ({min_x:.4f} < {config_min:.4f})")
                if max_x > config_max:
                    print(f"    ⚠️  Data ends after config ({max_x:.4f} > {config_max:.4f})")
    
    print("\n" + "="*60)
    print("SUMMARY - X-RANGE BY CLASS:")
    print("="*60)
    for class_name, ranges in class_ranges.items():
        print(f"\n{class_name}:")
        print(f"  Range: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
        print(f"  Avg: [{ranges['avg_min']:.4f}, {ranges['avg_max']:.4f}]")
        print(f"  Avg length: {ranges['avg_length']:.1f}")
    
    # Check if we need class-specific ranges
    print("\n" + "="*60)
    print("RECOMMENDATION:")
    print("="*60)
    
    all_mins = [r['min'] for r in class_ranges.values()]
    all_maxs = [r['max'] for r in class_ranges.values()]
    
    global_min = min(all_mins)
    global_max = max(all_maxs)
    
    print(f"\nGlobal range (covers all classes):")
    print(f"  RAMAN_X_MIN = {global_min:.4f}")
    print(f"  RAMAN_X_MAX = {global_max:.4f}")
    
    # Check variability
    min_std = np.std([r['avg_min'] for r in class_ranges.values()])
    max_std = np.std([r['avg_max'] for r in class_ranges.values()])
    
    print(f"\nVariability across classes:")
    print(f"  Min X std: {min_std:.4f}")
    print(f"  Max X std: {max_std:.4f}")
    
    if min_std > 10 or max_std > 10:
        print(f"\n  ⚠️  HIGH VARIABILITY! Consider class-specific ranges.")
        print(f"     Different classes have significantly different X-ranges.")
    else:
        print(f"\n  ✅  Low variability. Single global range is fine.")

if __name__ == "__main__":
    print("="*60)
    print("RAMAN X-RANGE ANALYSIS BY CLASS")
    print("="*60)
    analyze_raman_ranges_by_class()
