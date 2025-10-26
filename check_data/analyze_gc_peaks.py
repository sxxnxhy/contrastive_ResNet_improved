"""
Analyze GC data to find consistent retention time peaks for each class.
Goal: Verify that each class (2-CEES, 2-CEPS, DMMP, 4-NP) has a characteristic 
retention time where peaks consistently appear across all files.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from collections import defaultdict
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_gc_file(filepath):
    """Load GC spectrum from CSV file."""
    try:
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
        if data.ndim == 1:
            return None, None
        x = data[:, 1]  # Retention time
        y = data[:, 2]  # Intensity
        return x, y
    except:
        return None, None

def find_significant_peaks(x, y, height_percentile=90):
    """
    Find significant peaks in GC data.
    Returns retention times of peaks above the height_percentile.
    """
    if len(y) == 0:
        return []
    
    # Find peaks
    threshold = np.percentile(y[y > 0], height_percentile) if np.any(y > 0) else 0
    peaks, properties = find_peaks(y, height=threshold, distance=10)
    
    # Return retention times of peaks
    return x[peaks]

print("=" * 80)
print("ANALYZING GC RETENTION TIME PATTERNS BY CLASS")
print("=" * 80)

# Store peak retention times for each class
class_peaks = defaultdict(list)

# Analyze each class
for class_name in config.CLASSES:
    print(f"\n{'='*80}")
    print(f"Class: {class_name}")
    print(f"{'='*80}")
    
    class_folder = os.path.join(config.BASE_DATA_DIR, config.GC_DIRS[class_name])
    
    if not os.path.exists(class_folder):
        print(f"  ⚠️  Folder not found: {class_folder}")
        continue
    
    gc_files = [f for f in os.listdir(class_folder) if f.endswith('.csv')]
    
    # Sample files for analysis
    sample_size = min(500, len(gc_files))  # Analyze 500 files per class
    sample_files = np.random.choice(gc_files, sample_size, replace=False)
    
    print(f"  Total files: {len(gc_files):,}")
    print(f"  Analyzing: {sample_size} files")
    
    all_peaks = []
    file_count = 0
    
    for filename in tqdm(sample_files, desc=f"  Processing {class_name}"):
        filepath = os.path.join(class_folder, filename)
        x, y = load_gc_file(filepath)
        
        if x is None or len(x) == 0:
            continue
        
        # Find peaks in this file
        peak_times = find_significant_peaks(x, y, height_percentile=85)
        all_peaks.extend(peak_times)
        file_count += 1
    
    class_peaks[class_name] = all_peaks
    
    print(f"\n  Files processed: {file_count}")
    print(f"  Total peaks found: {len(all_peaks)}")
    print(f"  Peaks per file (avg): {len(all_peaks)/file_count:.1f}")

# Analyze retention time distributions
print(f"\n{'='*80}")
print("RETENTION TIME ANALYSIS")
print(f"{'='*80}")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('GC Retention Time Distribution by Class\n(Looking for Characteristic Peaks)', 
             fontsize=16, fontweight='bold')

for idx, class_name in enumerate(config.CLASSES):
    ax = axes.flat[idx]
    peaks = class_peaks[class_name]
    
    if len(peaks) == 0:
        print(f"\n{class_name}: No peaks found")
        continue
    
    # Create histogram of retention times
    bins = np.linspace(config.GC_X_MIN, config.GC_X_MAX, 200)
    counts, edges = np.histogram(peaks, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Find the most common retention times (characteristic peaks)
    peak_indices = find_peaks(counts, height=np.percentile(counts, 90), distance=5)[0]
    characteristic_peaks = centers[peak_indices]
    
    print(f"\n{class_name}:")
    print(f"  Total peaks detected: {len(peaks)}")
    print(f"  Retention time range: [{min(peaks):.3f}, {max(peaks):.3f}] minutes")
    print(f"  Characteristic retention times (peaks appearing in many files):")
    
    for i, rt in enumerate(sorted(characteristic_peaks)):
        # Count how many files have peaks near this retention time
        peak_count = counts[peak_indices[np.argmin(np.abs(centers[peak_indices] - rt))]]
        percentage = (peak_count / len(class_peaks[class_name])) * 100
        print(f"    {i+1}. RT = {rt:.3f} min (appears in ~{percentage:.0f}% of observations)")
    
    # Plot
    ax.hist(peaks, bins=200, alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Retention Time (minutes)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{class_name}\n{len(peaks)} peaks from {len(class_peaks[class_name])//len(peaks) if len(peaks) > 0 else 0} files', 
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(config.GC_X_MIN, config.GC_X_MAX)
    
    # Mark characteristic peaks
    for rt in characteristic_peaks:
        ax.axvline(rt, color='red', linestyle='--', linewidth=2, alpha=0.7)
    
    # Add legend
    if len(characteristic_peaks) > 0:
        ax.text(0.98, 0.98, f'{len(characteristic_peaks)} characteristic peaks', 
                transform=ax.transAxes, fontsize=10, verticalalignment='top', 
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('check_data/gc_retention_time_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print("✅ Plot saved: check_data/gc_retention_time_analysis.png")
print("="*80)

# Compare characteristic peaks between classes
print(f"\n{'='*80}")
print("CLASS SEPARATION ANALYSIS")
print(f"{'='*80}")

print("""
Expected Result:
  - Each class should have 1-3 characteristic retention times
  - These retention times should be DIFFERENT between classes
  - This indicates the model can learn to distinguish classes based on retention time

If classes share the same retention times:
  - This is EXPECTED for interferents/solvents (common peaks)
  - But each class should have at least ONE unique peak for its target compound
  - Model will learn to focus on the discriminative peaks
""")

# Create a comparison plot
fig2, ax2 = plt.subplots(figsize=(16, 8))

colors = ['red', 'blue', 'green', 'orange']
y_positions = [4, 3, 2, 1]

for idx, class_name in enumerate(config.CLASSES):
    peaks = class_peaks[class_name]
    if len(peaks) == 0:
        continue
    
    # Create histogram
    bins = np.linspace(config.GC_X_MIN, config.GC_X_MAX, 200)
    counts, edges = np.histogram(peaks, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2
    
    # Normalize counts for visualization
    counts_norm = counts / counts.max() if counts.max() > 0 else counts
    
    # Plot as offset curves
    ax2.fill_between(centers, y_positions[idx] - 0.4, y_positions[idx] - 0.4 + counts_norm * 0.8, 
                     alpha=0.6, color=colors[idx], label=class_name)
    ax2.plot(centers, y_positions[idx] - 0.4 + counts_norm * 0.8, 
             color=colors[idx], linewidth=2, alpha=0.8)

ax2.set_xlabel('Retention Time (minutes)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Class', fontsize=14, fontweight='bold')
ax2.set_yticks(y_positions)
ax2.set_yticklabels(config.CLASSES, fontsize=12)
ax2.set_title('GC Retention Time Comparison Across Classes\n(Do classes have distinct characteristic peaks?)', 
              fontsize=16, fontweight='bold')
ax2.legend(loc='upper right', fontsize=12)
ax2.grid(True, alpha=0.3, axis='x')
ax2.set_xlim(config.GC_X_MIN, config.GC_X_MAX)

plt.tight_layout()
plt.savefig('check_data/gc_class_comparison.png', dpi=150, bbox_inches='tight')
print(f"✅ Comparison plot saved: check_data/gc_class_comparison.png")

print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print("""
✅ Check the generated plots to verify:

1. Each class has characteristic retention time peaks (red dashed lines)
2. These peaks appear consistently across many files of the same class
3. Different classes have different characteristic peaks (class separation)
4. Some common peaks may appear in all classes (solvents/interferents) - this is OK

If classes have distinct characteristic peaks:
  → Model can learn to distinguish classes based on retention time ✅
  → Cross-modal learning will work (GC peaks + Raman spectra both identify same compound) ✅

If all classes have identical peaks:
  → Problem with data labeling or GC collection ❌
  → Model cannot distinguish classes from GC data alone ❌
""")
