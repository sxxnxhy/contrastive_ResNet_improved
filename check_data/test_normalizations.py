"""
Test different normalization strategies for GC data
Compare: baseline, log, global, hybrid (log+global)
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, '..')
import config

def load_sample_gc_data(n_samples=20):
    """Load sample GC data from each class"""
    samples = {}
    
    base_dir = '../data'
    
    for cls in config.CLASSES:
        gc_dir = os.path.join(base_dir, config.GC_DIRS[cls])
        
        all_files = []
        for root, dirs, files in os.walk(gc_dir):
            for f in files:
                if f.endswith('.csv'):
                    all_files.append(os.path.join(root, f))
        
        # Sample random files
        selected = np.random.choice(all_files, min(n_samples, len(all_files)), replace=False)
        
        samples[cls] = []
        for fpath in selected:
            try:
                data = np.loadtxt(fpath, delimiter=',', skiprows=3)
                x, y = data[:, 1], data[:, 2]
                samples[cls].append((x, y))
            except:
                pass
    
    return samples

def compute_global_statistics(samples):
    """Compute global 99th percentile across all samples"""
    all_intensities = []
    
    for cls, data_list in samples.items():
        for x, y in data_list:
            all_intensities.extend(y)
    
    all_intensities = np.array(all_intensities)
    
    p99 = np.percentile(all_intensities, 99)
    p99_log = np.percentile(np.log1p(all_intensities), 99)
    
    return p99, p99_log

def normalize_baseline(y):
    """Baseline: per-sample min-max (current method)"""
    y_min, y_max = y.min(), y.max()
    return (y - y_min) / (y_max - y_min + 1e-10)

def normalize_log(y):
    """Log transform + per-sample normalization"""
    y_log = np.log1p(y)
    return y_log / (y_log.max() + 1e-10)

def normalize_global(y, global_p99):
    """Global statistics normalization"""
    return np.clip(y / global_p99, 0, 1)

def normalize_hybrid(y, global_p99_log):
    """Hybrid: log + global"""
    y_log = np.log1p(y)
    return np.clip(y_log / global_p99_log, 0, 1)

def visualize_comparison(samples, global_p99, global_p99_log):
    """Visualize all 4 normalization methods"""
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle('GC Normalization Comparison: 4 Methods × 4 Classes', 
                 fontsize=18, fontweight='bold')
    
    methods = [
        ('Baseline (Per-sample Min-Max)', normalize_baseline, None),
        ('Log Transform', normalize_log, None),
        ('Global P99', normalize_global, global_p99),
        ('Hybrid (Log + Global)', normalize_hybrid, global_p99_log)
    ]
    
    for row_idx, (method_name, norm_func, param) in enumerate(methods):
        for col_idx, cls in enumerate(config.CLASSES):
            ax = axes[row_idx, col_idx]
            
            # Plot 3 samples
            data_list = samples[cls][:3]
            colors = plt.cm.Set1(np.linspace(0, 1, 3))
            
            for i, (x, y) in enumerate(data_list):
                if param is not None:
                    y_norm = norm_func(y, param)
                else:
                    y_norm = norm_func(y)
                
                ax.plot(x, y_norm, alpha=0.7, linewidth=1.5, 
                       color=colors[i], label=f'S{i+1}')
            
            # Styling
            if row_idx == 0:
                ax.set_title(cls, fontsize=13, fontweight='bold')
            
            if col_idx == 0:
                ax.set_ylabel(method_name, fontsize=11, fontweight='bold')
            
            if row_idx == 3:
                ax.set_xlabel('Retention Time (min)', fontsize=10)
            
            ax.set_xlim(0, 18)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('normalization_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: normalization_comparison.png")

def analyze_within_class_variance(samples, global_p99, global_p99_log):
    """Analyze how much variance remains after normalization"""
    
    print("\n" + "="*80)
    print("WITHIN-CLASS VARIANCE ANALYSIS")
    print("="*80)
    print("Lower variance = more consistent preprocessing")
    print("(Good for learning class-specific patterns)\n")
    
    methods = [
        ('Baseline', normalize_baseline, None),
        ('Log', normalize_log, None),
        ('Global', normalize_global, global_p99),
        ('Hybrid', normalize_hybrid, global_p99_log)
    ]
    
    for method_name, norm_func, param in methods:
        print(f"\n{method_name}:")
        
        for cls in config.CLASSES:
            # Normalize all samples
            normalized = []
            for x, y in samples[cls]:
                if param is not None:
                    y_norm = norm_func(y, param)
                else:
                    y_norm = norm_func(y)
                
                # Interpolate to common grid
                common_x = np.linspace(0, 17, 4096)
                y_aligned = np.interp(common_x, x, y_norm)
                normalized.append(y_aligned)
            
            # Compute variance
            normalized = np.array(normalized)
            variance = normalized.var(axis=0).mean()
            
            print(f"  {cls}: mean variance = {variance:.6f}")

if __name__ == '__main__':
    print("="*80)
    print("GC NORMALIZATION STRATEGY COMPARISON")
    print("="*80)
    
    print("\nLoading sample data...")
    samples = load_sample_gc_data(n_samples=20)
    
    print("\nComputing global statistics...")
    global_p99, global_p99_log = compute_global_statistics(samples)
    print(f"  Global 99th percentile (raw): {global_p99:.2f}")
    print(f"  Global 99th percentile (log): {global_p99_log:.4f}")
    
    print("\nGenerating comparison plots...")
    visualize_comparison(samples, global_p99, global_p99_log)
    
    analyze_within_class_variance(samples, global_p99, global_p99_log)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80)
