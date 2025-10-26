"""Check raw data similarity before preprocessing"""
import numpy as np
from pathlib import Path
import config
import matplotlib.pyplot as plt

def check_raw_similarity(modality='raman', class_name='2-CEES', num_samples=20):
    """Load raw files and check similarity"""
    
    if modality == 'raman':
        data_dir = Path(config.BASE_DATA_DIR) / config.RAMAN_DIRS[class_name]
    else:
        data_dir = Path(config.BASE_DATA_DIR) / config.GC_DIRS[class_name]
    
    files = list(data_dir.glob('*.csv'))[:num_samples]
    print(f"Loading {len(files)} {modality} files for {class_name}...")
    
    raw_data = []
    for filepath in files:
        try:
            if modality == 'raman':
                data = np.loadtxt(filepath, delimiter=',', dtype=np.float32)
                if data.ndim == 2:
                    y = data[:, 1]
                else:
                    y = data
            else:  # gc
                data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, comments='#')
                if data.ndim == 2 and data.shape[1] >= 3:
                    y = data[:, 2]
                else:
                    continue
            
            # Simple normalization
            y_norm = (y - y.min()) / (y.max() - y.min() + 1e-10)
            raw_data.append(y_norm)
        except:
            pass
    
    if len(raw_data) < 2:
        print(f"Not enough data loaded for {class_name} {modality}")
        return
    
    print(f"Loaded {len(raw_data)} files")
    
    # Find common length (take min)
    min_len = min(len(d) for d in raw_data)
    raw_data = [d[:min_len] for d in raw_data]
    raw_data = np.array(raw_data)
    
    print(f"Shape after trimming: {raw_data.shape}")
    
    # Compute pairwise similarity
    from scipy.spatial.distance import cosine
    similarities = []
    for i in range(len(raw_data)):
        for j in range(i+1, len(raw_data)):
            sim = 1 - cosine(raw_data[i], raw_data[j])
            similarities.append(sim)
    
    similarities = np.array(similarities)
    
    print(f"\nRaw data similarity ({class_name} {modality}):")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std: {similarities.std():.4f}")
    print(f"  Min: {similarities.min():.4f}")
    print(f"  Max: {similarities.max():.4f}")
    
    # Plot first 5 samples
    fig, ax = plt.subplots(figsize=(12, 6))
    for i in range(min(5, len(raw_data))):
        ax.plot(raw_data[i], alpha=0.7, label=f"Sample {i+1}")
    ax.set_title(f"Raw {modality.upper()} spectra - {class_name} (normalized)")
    ax.set_xlabel("Data points")
    ax.set_ylabel("Normalized intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"raw_{modality}_{class_name}.png", dpi=150)
    print(f"Saved plot: raw_{modality}_{class_name}.png")
    
    return similarities.mean()

if __name__ == "__main__":
    print("="*60)
    print("Checking RAW data similarity (before interpolation)")
    print("="*60)
    
    results = {}
    
    for class_name in config.CLASSES:
        print(f"\n{'='*60}")
        print(f"Class: {class_name}")
        print(f"{'='*60}")
        
        # Check Raman
        raman_sim = check_raw_similarity('raman', class_name, num_samples=20)
        
        print()
        
        # Check GC
        gc_sim = check_raw_similarity('gc', class_name, num_samples=20)
        
        results[class_name] = {'raman': raman_sim, 'gc': gc_sim}
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for class_name, sims in results.items():
        print(f"{class_name}:")
        if sims['raman'] is not None:
            print(f"  Raman similarity: {sims['raman']:.4f}")
        if sims['gc'] is not None:
            print(f"  GC similarity: {sims['gc']:.4f}")
