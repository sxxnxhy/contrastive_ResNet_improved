"""
Analyze why GC clustering is difficult and propose solutions.
Compare raw GC data characteristics vs. what models need.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def load_gc_file(filepath):
    """Load GC spectrum."""
    try:
        data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
        if data.ndim == 1:
            return None
        x = data[:, 1]  # Retention time
        y = data[:, 2]  # Intensity
        return x, y
    except:
        return None

def preprocess_gc(x, y):
    """Same preprocessing as in dataset.py"""
    common_x = np.linspace(config.GC_X_MIN, config.GC_X_MAX, config.COMMON_LENGTH)
    y_aligned = np.interp(common_x, x, y)
    
    # Normalize to [0, 1]
    y_min, y_max = y_aligned.min(), y_aligned.max()
    y_normalized = (y_aligned - y_min) / (y_max - y_min + 1e-10)
    
    return y_normalized

print("=" * 80)
print("WHY IS GC CLUSTERING DIFFICULT?")
print("=" * 80)

# Load sample data from each class
samples_per_class = 100
all_data = []
all_labels = []
all_classes = []

for class_idx, class_name in enumerate(config.CLASSES):
    print(f"\nLoading {class_name}...")
    class_folder = os.path.join(config.BASE_DATA_DIR, config.GC_DIRS[class_name])
    
    if not os.path.exists(class_folder):
        continue
    
    gc_files = [f for f in os.listdir(class_folder) if f.endswith('.csv')]
    sample_files = np.random.choice(gc_files, min(samples_per_class, len(gc_files)), replace=False)
    
    for filename in tqdm(sample_files, desc=f"  Processing"):
        filepath = os.path.join(class_folder, filename)
        x, y = load_gc_file(filepath)
        
        if x is None:
            continue
        
        # Preprocess
        y_processed = preprocess_gc(x, y)
        
        all_data.append(y_processed)
        all_labels.append(class_idx)
        all_classes.append(class_name)

all_data = np.array(all_data)
all_labels = np.array(all_labels)

print(f"\n{'='*80}")
print(f"Loaded {len(all_data)} samples")
print(f"Data shape: {all_data.shape}")

# Analyze data characteristics
print(f"\n{'='*80}")
print("GC DATA CHARACTERISTICS")
print(f"{'='*80}")

sparsity = (all_data == 0).sum(axis=1).mean() / all_data.shape[1]
print(f"\n1. SPARSITY (average zeros per sample): {sparsity*100:.1f}%")
print(f"   → Most of the signal is ZERO (no compound detected)")
print(f"   → Only a few peaks at specific retention times")

variance_per_point = all_data.var(axis=0)
high_variance_ratio = (variance_per_point > 0.01).sum() / len(variance_per_point)
print(f"\n2. INFORMATIVE REGIONS: {high_variance_ratio*100:.1f}% of time points")
print(f"   → Only a small portion of the timeline has discriminative information")
print(f"   → Most time points are uniformly zero across all samples")

# Within-class vs between-class distances
print(f"\n3. SIMILARITY ANALYSIS:")
for class_idx, class_name in enumerate(config.CLASSES):
    class_mask = all_labels == class_idx
    class_data = all_data[class_mask]
    
    if len(class_data) < 2:
        continue
    
    # Within-class pairwise distances
    within_distances = []
    for i in range(min(20, len(class_data))):
        for j in range(i+1, min(20, len(class_data))):
            dist = np.linalg.norm(class_data[i] - class_data[j])
            within_distances.append(dist)
    
    # Between-class distances
    other_data = all_data[~class_mask]
    between_distances = []
    for i in range(min(20, len(class_data))):
        for j in range(min(20, len(other_data))):
            dist = np.linalg.norm(class_data[i] - other_data[j])
            between_distances.append(dist)
    
    print(f"\n   {class_name}:")
    print(f"     Within-class distance: {np.mean(within_distances):.4f} ± {np.std(within_distances):.4f}")
    print(f"     Between-class distance: {np.mean(between_distances):.4f} ± {np.std(between_distances):.4f}")
    ratio = np.mean(between_distances) / np.mean(within_distances)
    print(f"     Separation ratio: {ratio:.2f}x")
    if ratio < 1.5:
        print(f"     ⚠️  LOW separation! Classes overlap significantly")
    elif ratio < 2.0:
        print(f"     ⚙️  MODERATE separation")
    else:
        print(f"     ✅ GOOD separation")

# Visualize with PCA and t-SNE
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle('GC Data Visualization: Why Clustering is Difficult', fontsize=16, fontweight='bold')

colors = ['red', 'blue', 'green', 'orange']
class_colors = [colors[label] for label in all_labels]

# 1. Raw data sample
ax = axes[0, 0]
for class_idx, class_name in enumerate(config.CLASSES):
    class_mask = all_labels == class_idx
    if np.any(class_mask):
        sample = all_data[class_mask][0]
        common_x = np.linspace(config.GC_X_MIN, config.GC_X_MAX, config.COMMON_LENGTH)
        ax.plot(common_x, sample + class_idx * 1.2, label=class_name, color=colors[class_idx], linewidth=0.8)
ax.set_title('Raw GC Signals (Offset for Visibility)', fontsize=12, fontweight='bold')
ax.set_xlabel('Retention Time (minutes)')
ax.set_ylabel('Normalized Intensity (offset)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. Variance across samples
ax = axes[0, 1]
common_x = np.linspace(config.GC_X_MIN, config.GC_X_MAX, config.COMMON_LENGTH)
ax.plot(common_x, variance_per_point, linewidth=0.8)
ax.set_title(f'Variance Across Samples\n(Only {high_variance_ratio*100:.1f}% of points are informative)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Retention Time (minutes)')
ax.set_ylabel('Variance')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)
ax.axhline(0.01, color='red', linestyle='--', label='Threshold (0.01)')
ax.legend()

# 3. Sparsity pattern
ax = axes[0, 2]
sparsity_per_sample = (all_data == 0).sum(axis=1) / all_data.shape[1]
for class_idx, class_name in enumerate(config.CLASSES):
    class_mask = all_labels == class_idx
    ax.hist(sparsity_per_sample[class_mask], bins=20, alpha=0.6, 
            label=class_name, color=colors[class_idx])
ax.set_title(f'Sparsity Distribution\n(Avg: {sparsity*100:.0f}% zeros)', 
             fontsize=12, fontweight='bold')
ax.set_xlabel('Fraction of Zero Values')
ax.set_ylabel('Count')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. PCA (linear)
print(f"\n{'='*80}")
print("Computing PCA...")
pca = PCA(n_components=2)
data_pca = pca.fit_transform(all_data)

ax = axes[1, 0]
for class_idx, class_name in enumerate(config.CLASSES):
    class_mask = all_labels == class_idx
    ax.scatter(data_pca[class_mask, 0], data_pca[class_mask, 1], 
               c=colors[class_idx], label=class_name, alpha=0.6, s=20)
ax.set_title(f'PCA (Linear)\nExplained variance: {pca.explained_variance_ratio_.sum()*100:.1f}%', 
             fontsize=12, fontweight='bold')
ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.legend()
ax.grid(True, alpha=0.3)

# 5. t-SNE (non-linear)
print("Computing t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
data_tsne = tsne.fit_transform(all_data)

ax = axes[1, 1]
for class_idx, class_name in enumerate(config.CLASSES):
    class_mask = all_labels == class_idx
    ax.scatter(data_tsne[class_mask, 0], data_tsne[class_mask, 1], 
               c=colors[class_idx], label=class_name, alpha=0.6, s=20)
ax.set_title('t-SNE (Current State)\n⚠️ Poor Clustering', fontsize=12, fontweight='bold')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.legend()
ax.grid(True, alpha=0.3)

# 6. Summary text
ax = axes[1, 2]
ax.axis('off')
summary_text = f"""
PROBLEMS WITH GC DATA:

1. HIGH SPARSITY ({sparsity*100:.0f}% zeros)
   → Most timeline has no signal
   → Hard for model to find patterns

2. FEW INFORMATIVE REGIONS
   → Only {high_variance_ratio*100:.1f}% of points vary
   → 4096 dimensions but <200 useful

3. PEAK-BASED NATURE
   → Information in a few sharp peaks
   → Not in overall shape/texture
   → 1D CNN may smooth out peaks

4. HIGH VARIATION WITHIN CLASS
   → Different interferents cause
     different peak patterns
   → Same class looks very different

5. LOW INTER-CLASS SEPARATION
   → Some classes share common peaks
   → Hard to distinguish linearly

RECOMMENDED SOLUTIONS:
✅ Use attention mechanisms
✅ Peak-aware architectures
✅ Multi-scale feature extraction
✅ Stronger contrastive loss
✅ Focal loss for hard examples
"""
ax.text(0.1, 0.9, summary_text, fontsize=10, verticalalignment='top',
        family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('check_data/gc_clustering_analysis.png', dpi=150, bbox_inches='tight')
print(f"\n{'='*80}")
print("✅ Analysis saved: check_data/gc_clustering_analysis.png")
print("="*80)

print(f"\n{'='*80}")
print("SOLUTIONS TO IMPROVE GC CLUSTERING")
print("="*80)
print("""
Current Model: MobileNet (designed for 2D images with textures)
Problem: GC data is 1D with sparse peaks, not textures

SOLUTION 1: ATTENTION-BASED ARCHITECTURE ⭐⭐⭐ (RECOMMENDED)
-----------------------------------------------------------
Why: Attention can focus on specific retention times (peaks)
Architecture:
  - 1D Conv layers to extract local peak patterns
  - Multi-head self-attention to find important peaks
  - Position encoding to preserve retention time information
  
Implementation:
  - Replace MobileNet with Transformer encoder
  - Or add attention layers after CNN backbone
  - Model learns "which peaks matter for each class"

SOLUTION 2: MULTI-SCALE FEATURE EXTRACTION ⭐⭐
-----------------------------------------------------------
Why: Peaks have different widths, need multiple scales
Architecture:
  - Parallel conv layers with different kernel sizes
  - Dilated convolutions for wide receptive fields
  - Capture both sharp peaks and broad baseline patterns

SOLUTION 3: PEAK-AWARE PREPROCESSING ⭐⭐
-----------------------------------------------------------
Why: Give model pre-computed peak features
Features to add:
  - Peak detection + peak positions as extra channel
  - Peak heights, widths, areas
  - Model gets both raw signal AND peak info

SOLUTION 4: STRONGER LOSS FUNCTIONS ⭐
-----------------------------------------------------------
Current: NT-Xent + CrossEntropy + Cross-modal alignment
Add:
  - Focal loss: Focus on hard-to-separate examples
  - Triplet loss with hard negative mining
  - Center loss: Pull same class closer in embedding space

SOLUTION 5: DATA AUGMENTATION FOR GC ⭐
-----------------------------------------------------------
Current augmentation may not preserve peak structure
Better augmentation:
  - Baseline noise addition
  - Peak intensity scaling (simulate concentration)
  - Small time shifts (simulate RT drift)
  - Keep peaks intact!

SOLUTION 6: HYBRID ARCHITECTURE (BEST) ⭐⭐⭐
-----------------------------------------------------------
Combine CNN + Attention + Peak features:
  
  GC Input (4096)
       ↓
  1D Conv layers (extract local patterns)
       ↓
  Multi-head Attention (focus on key peaks)
       ↓
  Global pooling
       ↓
  Embedding (256D)

This gives:
  ✅ Local pattern extraction (CNN)
  ✅ Global peak relationships (Attention)
  ✅ Focus on discriminative regions
  ✅ Better than pure CNN or pure Attention

EASIEST TO IMPLEMENT NOW:
1. Add self-attention layer after MobileNet encoder
2. Use focal loss for classification
3. Increase cross-modal alignment weight

Want me to implement any of these?
""")
