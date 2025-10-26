"""
Analyze GC peak patterns by interferent type
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from collections import defaultdict

sys.path.insert(0, '..')
import config

def extract_interferent(filename):
    """Extract interferent name from filename"""
    # Common interferents: EtOH, MC, MeOH, THF
    interferents = ['EtOH', 'MC', 'MeOH', 'THF']
    
    for interf in interferents:
        if interf in filename:
            return interf
    
    return 'Unknown'

def load_and_group_gc_data():
    """Load GC data and group by class and interferent"""
    data_by_class = {}
    
    # Use absolute path
    base_dir = '../data'

    gc_dirs = config.GC_DIRS

    for cls in config.CLASSES:
        gc_dir = os.path.join(base_dir, gc_dirs[cls])
        
        if not os.path.exists(gc_dir):
            print(f"Warning: {gc_dir} not found")
            continue
        
        # Group by interferent
        by_interferent = defaultdict(list)
        
        all_files = []
        for root, dirs, files in os.walk(gc_dir):
            for f in files:
                if f.endswith('.csv'):  # GC files are CSV
                    fpath = os.path.join(root, f)
                    all_files.append(fpath)
        
        print(f"\n{cls}: Found {len(all_files)} files")
        
        # Sample files and group by interferent
        for fpath in all_files[:100]:  # Sample 100 files
            filename = os.path.basename(fpath)
            interferent = extract_interferent(filename)
            
            try:
                data = np.loadtxt(fpath, delimiter=',', skiprows=3)  # Skip 3 header lines
                if data.shape[0] > 0 and data.shape[1] >= 3:
                    # Format: Point, X(Minutes), Y(Counts)
                    x, y = data[:, 1], data[:, 2]
                    
                    by_interferent[interferent].append((x, y, filename))
            except Exception as e:
                pass
        
        # Print statistics
        for interf, samples in by_interferent.items():
            print(f"  {interf}: {len(samples)} samples")
        
        data_by_class[cls] = by_interferent
    
    return data_by_class

def plot_gc_by_interferent(data_by_class, output_dir='check_data'):
    """Plot GC chromatograms grouped by interferent"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    for cls, by_interferent in data_by_class.items():
        if not by_interferent:
            continue
        
        # Create figure with subplots for each interferent
        interferents = sorted(by_interferent.keys())
        n_interferents = len(interferents)
        
        fig, axes = plt.subplots(n_interferents, 1, figsize=(14, 4*n_interferents))
        if n_interferents == 1:
            axes = [axes]
        
        fig.suptitle(f'GC Chromatograms: {cls} by Interferent', 
                     fontsize=16, fontweight='bold')
        
        for idx, interferent in enumerate(interferents):
            ax = axes[idx]
            samples = by_interferent[interferent]
            
            # Plot first 5 samples
            colors = plt.cm.tab10(np.linspace(0, 1, 5))
            for i, (x, y, fname) in enumerate(samples[:5]):
                ax.plot(x, y, alpha=0.7, linewidth=1.5, 
                       label=f'Sample {i+1}', color=colors[i])
            
            ax.set_title(f'{interferent} (n={len(samples)} samples)', 
                        fontsize=13, fontweight='bold')
            ax.set_xlabel('Retention Time (min)', fontsize=11)
            ax.set_ylabel('Intensity', fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9, loc='upper right')
            ax.set_xlim(0, 18)  # Fix X-axis to 0-18 minutes
            
            # Find and mark major peaks from first sample
            if len(samples) > 0:
                x, y, _ = samples[0]
                # Find peaks (> 30% of max)
                peak_threshold = 0.3 * y.max()
                
                # Simple peak detection
                peaks = []
                for i in range(1, len(y)-1):
                    if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > peak_threshold:
                        # Check if this is a new peak (not too close to existing ones)
                        is_new = True
                        for px, _ in peaks:
                            if abs(x[i] - px) < 0.5:  # Within 0.5 minutes
                                is_new = False
                                break
                        if is_new:
                            peaks.append((x[i], y[i]))
                
                # Mark peaks
                for px, py in peaks:
                    ax.axvline(px, color='red', linestyle='--', alpha=0.4, linewidth=1.5)
                    ax.text(px, ax.get_ylim()[1]*0.95, f'{px:.1f}', 
                           ha='center', fontsize=10, color='red', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        plt.tight_layout()
        
        output_path = os.path.join(output_dir, f'gc_interferents_{cls}.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Saved: {output_path}")
        plt.close()

def analyze_peak_variation(data_by_class):
    """Analyze how much peaks vary by interferent"""
    
    print("\n" + "="*80)
    print("PEAK VARIATION ANALYSIS")
    print("="*80)
    
    for cls, by_interferent in data_by_class.items():
        print(f"\n{cls}:")
        
        all_peak_positions = []
        
        for interferent, samples in by_interferent.items():
            if len(samples) == 0:
                continue
            
            # Find major peaks in each sample
            peak_positions = []
            
            for x, y, _ in samples[:10]:  # First 10 samples
                peak_threshold = 0.3 * y.max()
                
                # Find local maxima
                for i in range(1, len(y)-1):
                    if y[i] > y[i-1] and y[i] > y[i+1] and y[i] > peak_threshold:
                        peak_positions.append(x[i])
            
            if peak_positions:
                peak_positions = np.array(peak_positions)
                all_peak_positions.extend(peak_positions)
                
                print(f"  {interferent}:")
                print(f"    Peak retention times: {np.unique(np.round(peak_positions, 1))}")
                print(f"    Mean: {peak_positions.mean():.2f} ± {peak_positions.std():.2f} min")
        
        if all_peak_positions:
            all_peaks = np.array(all_peak_positions)
            print(f"  Overall peak variation (std): {all_peaks.std():.2f} min")
            
            if all_peaks.std() > 2.0:
                print(f"  ⚠️  HIGH VARIATION! Interferents cause large peak shifts!")

if __name__ == '__main__':
    print("="*80)
    print("GC INTERFERENT ANALYSIS")
    print("="*80)
    print("\nLoading GC data and grouping by interferent...")
    
    data_by_class = load_and_group_gc_data()
    
    print("\n" + "="*80)
    print("Generating plots...")
    print("="*80)
    
    plot_gc_by_interferent(data_by_class)
    
    analyze_peak_variation(data_by_class)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("Check the 'check_data/' folder for output plots")
    print("="*80)
