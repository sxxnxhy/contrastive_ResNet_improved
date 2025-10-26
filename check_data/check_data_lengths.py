"""
Check spectrum data lengths by class and modality
"""
import numpy as np
from pathlib import Path
from collections import defaultdict
import config

def check_file_length(filepath, modality):
    """Check the length and x-axis range of data in a file"""
    try:
        if modality == 'raman':
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
            x, y = data[:, 0], data[:, 1]
        else:  # gc
            data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
            x, y = data[:, 1], data[:, 2]
        
        return {
            'length': len(y),
            'x_min': float(x.min()),
            'x_max': float(x.max()),
            'x_range': float(x.max() - x.min())
        }
    except Exception as e:
        return None

def main():
    print("="*70)
    print("CHECKING SPECTRUM DATA LENGTHS")
    print("="*70)
    
    stats = {
        'raman': defaultdict(lambda: {'lengths': [], 'x_mins': [], 'x_maxs': [], 'x_ranges': []}),
        'gc': defaultdict(lambda: {'lengths': [], 'x_mins': [], 'x_maxs': [], 'x_ranges': []})
    }
    
    # Check Raman files
    print("\n📊 Checking Raman files...")
    for class_label, rel_dir in config.RAMAN_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        if not full_dir.exists():
            print(f"  ⚠️  Directory not found: {full_dir}")
            continue
        
        file_count = 0
        for filepath in full_dir.glob('*.csv'):
            info = check_file_length(str(filepath), 'raman')
            if info is not None:
                stats['raman'][class_label]['lengths'].append(info['length'])
                stats['raman'][class_label]['x_mins'].append(info['x_min'])
                stats['raman'][class_label]['x_maxs'].append(info['x_max'])
                stats['raman'][class_label]['x_ranges'].append(info['x_range'])
                file_count += 1
                
                # Show progress every 1000 files
                if file_count % 1000 == 0:
                    print(f"  {class_label}: {file_count} files checked...", end='\r')
        
        print(f"  ✓ {class_label}: {file_count} files checked        ")
    
    # Check GC files
    print("\n📊 Checking GC files...")
    for class_label, rel_dir in config.GC_DIRS.items():
        full_dir = Path(config.BASE_DATA_DIR) / rel_dir
        if not full_dir.exists():
            print(f"  ⚠️  Directory not found: {full_dir}")
            continue
        
        file_count = 0
        for filepath in full_dir.glob('*.csv'):
            info = check_file_length(str(filepath), 'gc')
            if info is not None:
                stats['gc'][class_label]['lengths'].append(info['length'])
                stats['gc'][class_label]['x_mins'].append(info['x_min'])
                stats['gc'][class_label]['x_maxs'].append(info['x_max'])
                stats['gc'][class_label]['x_ranges'].append(info['x_range'])
                file_count += 1
                
                if file_count % 1000 == 0:
                    print(f"  {class_label}: {file_count} files checked...", end='\r')
        
        print(f"  ✓ {class_label}: {file_count} files checked        ")
    
    # Print summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n🔬 RAMAN Spectra:")
    print("-" * 90)
    print(f"{'Class':<15} {'Count':>8} {'Len Min':>8} {'Len Max':>8} {'X Min':>10} {'X Max':>10} {'X Range':>12}")
    print("-" * 90)
    
    for class_label in sorted(stats['raman'].keys()):
        data = stats['raman'][class_label]
        if data['lengths']:
            print(f"{class_label:<15} {len(data['lengths']):>8} "
                  f"{min(data['lengths']):>8} {max(data['lengths']):>8} "
                  f"{min(data['x_mins']):>10.1f} {max(data['x_maxs']):>10.1f} "
                  f"{np.mean(data['x_ranges']):>12.1f}")
    
    print("\n🧪 GC Spectra:")
    print("-" * 90)
    print(f"{'Class':<15} {'Count':>8} {'Len Min':>8} {'Len Max':>8} {'X Min':>10} {'X Max':>10} {'X Range':>12}")
    print("-" * 90)
    
    for class_label in sorted(stats['gc'].keys()):
        data = stats['gc'][class_label]
        if data['lengths']:
            print(f"{class_label:<15} {len(data['lengths']):>8} "
                  f"{min(data['lengths']):>8} {max(data['lengths']):>8} "
                  f"{min(data['x_mins']):>10.1f} {max(data['x_maxs']):>10.1f} "
                  f"{np.mean(data['x_ranges']):>12.1f}")
    
    # Overall statistics
    print("\n" + "="*90)
    print("OVERALL X-AXIS RANGE STATISTICS")
    print("="*90)
    
    all_raman_lengths = [l for data in stats['raman'].values() for l in data['lengths']]
    all_raman_x_mins = [x for data in stats['raman'].values() for x in data['x_mins']]
    all_raman_x_maxs = [x for data in stats['raman'].values() for x in data['x_maxs']]
    
    all_gc_lengths = [l for data in stats['gc'].values() for l in data['lengths']]
    all_gc_x_mins = [x for data in stats['gc'].values() for x in data['x_mins']]
    all_gc_x_maxs = [x for data in stats['gc'].values() for x in data['x_maxs']]
    
    if all_raman_lengths:
        print(f"\n🔬 All Raman files:")
        print(f"   Total files: {len(all_raman_lengths)}")
        print(f"   Y-axis length range: {min(all_raman_lengths)} - {max(all_raman_lengths)}")
        print(f"   X-axis range: {min(all_raman_x_mins):.1f} - {max(all_raman_x_maxs):.1f}")
        print(f"   X min range: {min(all_raman_x_mins):.1f} - {max(all_raman_x_mins):.1f}")
        print(f"   X max range: {min(all_raman_x_maxs):.1f} - {max(all_raman_x_maxs):.1f}")
        
        # Check if x-axis is consistent
        x_min_std = np.std(all_raman_x_mins)
        x_max_std = np.std(all_raman_x_maxs)
        
        if x_min_std < 0.1 and x_max_std < 0.1:
            print(f"   ✅ X-axis range is CONSISTENT across all files")
            print(f"      Suggested range: [{np.mean(all_raman_x_mins):.1f}, {np.mean(all_raman_x_maxs):.1f}]")
        else:
            print(f"   ⚠️  X-axis range VARIES across files!")
            print(f"      X min std: {x_min_std:.1f}, X max std: {x_max_std:.1f}")
            print(f"   Suggested common range: [{min(all_raman_x_mins):.1f}, {max(all_raman_x_maxs):.1f}]")
    
    if all_gc_lengths:
        print(f"\n🧪 All GC files:")
        print(f"   Total files: {len(all_gc_lengths)}")
        print(f"   Y-axis length range: {min(all_gc_lengths)} - {max(all_gc_lengths)}")
        print(f"   X-axis range: {min(all_gc_x_mins):.1f} - {max(all_gc_x_maxs):.1f}")
        print(f"   X min range: {min(all_gc_x_mins):.1f} - {max(all_gc_x_mins):.1f}")
        print(f"   X max range: {min(all_gc_x_maxs):.1f} - {max(all_gc_x_maxs):.1f}")
        
        x_min_std = np.std(all_gc_x_mins)
        x_max_std = np.std(all_gc_x_maxs)
        
        if x_min_std < 0.1 and x_max_std < 0.1:
            print(f"   ✅ X-axis range is CONSISTENT across all files")
            print(f"      Suggested range: [{np.mean(all_gc_x_mins):.1f}, {np.mean(all_gc_x_maxs):.1f}]")
        else:
            print(f"   ⚠️  X-axis range VARIES across files!")
            print(f"      X min std: {x_min_std:.1f}, X max std: {x_max_std:.1f}")
            print(f"   Suggested common range: [{min(all_gc_x_mins):.1f}, {max(all_gc_x_maxs):.1f}]")
    
    print("\n" + "="*90)
    print(f"✓ Target length for interpolation: {config.COMMON_AXIS_POINTS}")
    print("="*90)

if __name__ == "__main__":
    main()
