"""Verify if ALL files in each class have the EXACT same X-range"""
import numpy as np
from pathlib import Path
import config
from tqdm import tqdm
from collections import Counter

def verify_class_ranges():
    """Check if all files within a class have identical X-ranges"""
    
    expected_ranges = {
        "2-CEES": (401.425, 1499.68),
        "2-CEPS": (300.146, 3199.99),
        "DMMP": (400.428, 1499.68),
        "4-NP": (300.117, 3199.99)
    }
    
    for class_name, rel_dir in config.RAMAN_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        raman_files = list(full_dir.glob('*.csv'))
        
        print(f"\n{'='*60}")
        print(f"Class: {class_name}")
        print(f"Expected range: {expected_ranges[class_name]}")
        print(f"Total files: {len(raman_files)}")
        print(f"{'='*60}")
        
        # Collect all unique (min, max) pairs
        range_counts = Counter()
        range_examples = {}  # Store example files for each range
        all_lengths = []
        
        for filepath in tqdm(raman_files, desc=f"Checking {class_name}"):
            try:
                data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
                if data.ndim == 2:
                    x = data[:, 0]
                    x_min = round(x.min(), 3)  # Round to 3 decimals
                    x_max = round(x.max(), 3)
                    x_len = len(x)
                    
                    range_key = (x_min, x_max)
                    range_counts[range_key] += 1
                    all_lengths.append(x_len)
                    
                    # Store first example for each range
                    if range_key not in range_examples:
                        range_examples[range_key] = filepath.name
            except Exception as e:
                print(f"  Error reading {filepath.name}: {e}")
        
        # Report results
        print(f"\n  Found {len(range_counts)} unique X-ranges:")
        for (x_min, x_max), count in sorted(range_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = 100 * count / len(raman_files)
            example = range_examples.get((x_min, x_max), "N/A")
            
            # Check if matches expected
            exp_min, exp_max = expected_ranges[class_name]
            matches = (abs(x_min - exp_min) < 0.01 and abs(x_max - exp_max) < 0.01)
            status = "✅" if matches else "⚠️"
            
            print(f"\n  {status} Range [{x_min:.3f}, {x_max:.3f}]:")
            print(f"     Count: {count:,} / {len(raman_files):,} ({percentage:.1f}%)")
            print(f"     Example: {example}")
            
            if not matches:
                print(f"     ⚠️  MISMATCH! Expected: [{exp_min:.3f}, {exp_max:.3f}]")
        
        # Length statistics
        if all_lengths:
            unique_lengths = set(all_lengths)
            length_counts = Counter(all_lengths)
            print(f"\n  📊 Data Point Distribution:")
            print(f"     Total unique lengths: {len(unique_lengths)}")
            print(f"     Min points: {min(all_lengths)}")
            print(f"     Max points: {max(all_lengths)}")
            print(f"\n     Detailed breakdown:")
            for length, count in sorted(length_counts.items(), key=lambda x: x[1], reverse=True):
                percentage = 100 * count / len(all_lengths)
                bar_length = int(percentage / 2)  # Scale to 50 chars max
                bar = "█" * bar_length
                print(f"       {length:5d} points: {count:6,d} files ({percentage:5.1f}%) {bar}")
        
        # Final verdict
        if len(range_counts) == 1:
            print(f"\n  ✅ UNIFORM: All files have the same X-range!")
        else:
            print(f"\n  ⚠️  VARIABLE: Multiple X-ranges found!")
            print(f"     This will cause problems with unified preprocessing.")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print("\nExpected ranges per class:")
    for class_name, (x_min, x_max) in expected_ranges.items():
        print(f"  {class_name}: [{x_min:.3f}, {x_max:.3f}]")

if __name__ == "__main__":
    print("="*60)
    print("VERIFYING RAMAN X-RANGES ARE CONSISTENT WITHIN EACH CLASS")
    print("="*60)
    verify_class_ranges()
