"""
Visualization script for cross-modal embeddings

임베딩 공간을 시각화하여:
1. 클래스별 클러스터링 확인
2. Cross-modal alignment 확인 (Raman과 GC가 같은 클래스에서 섞여있는지)
3. Inter-class separation 확인

시각화 방법:
- t-SNE: Local structure preservation (가까운 이웃 관계)
- UMAP: Global + local structure, faster than t-SNE
- PCA: Linear projection (전체적인 분산 방향)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import config
from model import CrossModalContrastiveModel
from dataset import prepare_dataloader


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=5000, 
                      stratified=True, samples_per_class_modality=None):
    """
    Extract embeddings from the model
    
    Args:
        max_samples: Maximum total samples (used if stratified=False)
        stratified: If True, sample equally from each class-modality pair
        samples_per_class_modality: Samples per (class, modality) pair. 
                                   If None, auto-calculated from max_samples
    """
    model.eval()
    
    if stratified and samples_per_class_modality is None:
        # 4 classes × 2 modalities = 8 groups
        # Default: max_samples / 8, but ensure reasonable number
        samples_per_class_modality = max(100, max_samples // 8)
        print(f"Using stratified sampling: {samples_per_class_modality} samples per (class, modality) pair")
    
    if stratified:
        # Stratified sampling: collect per class-modality group
        from collections import defaultdict
        group_samples = defaultdict(lambda: {'embeddings': [], 'labels': [], 'modalities': []})
        
        for spectra, labels_and_mod in tqdm(dataloader, desc='Extracting embeddings (stratified)'):
            spectra = spectra.to(device)
            labels_and_mod = labels_and_mod.to(device)
            
            class_labels = labels_and_mod[:, :-1]
            modalities = labels_and_mod[:, -1]
            
            embeddings = model(spectra, modalities)
            
            # Group by (class, modality)
            class_indices = torch.argmax(class_labels, dim=1).cpu().numpy()
            modality_values = modalities.cpu().numpy()
            embedding_values = embeddings.cpu().numpy()
            
            for i in range(len(class_indices)):
                class_idx = int(class_indices[i])
                modality = int(modality_values[i])
                key = (class_idx, modality)
                
                # Check if this group needs more samples
                if len(group_samples[key]['embeddings']) < samples_per_class_modality:
                    group_samples[key]['embeddings'].append(embedding_values[i])
                    group_samples[key]['labels'].append(class_idx)
                    group_samples[key]['modalities'].append(modality)
            
            # Check if all groups are full
            all_full = all(len(group['embeddings']) >= samples_per_class_modality 
                          for group in group_samples.values())
            if all_full and len(group_samples) >= 8:  # 4 classes × 2 modalities
                break
        
        # Combine all groups
        all_embeddings = []
        all_class_labels = []
        all_modalities = []
        
        for key in sorted(group_samples.keys()):
            group = group_samples[key]
            all_embeddings.extend(group['embeddings'])
            all_class_labels.extend(group['labels'])
            all_modalities.extend(group['modalities'])
        
        embeddings = np.array(all_embeddings)
        class_labels = np.array(all_class_labels)
        modalities = np.array(all_modalities)
        
        print(f"Extracted {len(embeddings)} samples total")
        print(f"Samples per group: {[(k, len(group_samples[k]['embeddings'])) for k in sorted(group_samples.keys())]}")
        
    else:
        # Original method: just take first max_samples
        all_embeddings = []
        all_class_labels = []
        all_modalities = []
        
        total_samples = 0
        
        for spectra, labels_and_mod in tqdm(dataloader, desc='Extracting embeddings'):
            if total_samples >= max_samples:
                break
            
            spectra = spectra.to(device)
            labels_and_mod = labels_and_mod.to(device)
            
            class_labels = labels_and_mod[:, :-1]
            modalities = labels_and_mod[:, -1]
            
            embeddings = model(spectra, modalities)
            
            all_embeddings.append(embeddings.cpu().numpy())
            all_class_labels.append(torch.argmax(class_labels, dim=1).cpu().numpy())
            all_modalities.append(modalities.cpu().numpy())
            
            total_samples += spectra.size(0)
        
        embeddings = np.concatenate(all_embeddings, axis=0)[:max_samples]
        class_labels = np.concatenate(all_class_labels, axis=0)[:max_samples]
        modalities = np.concatenate(all_modalities, axis=0)[:max_samples]
    
    return embeddings, class_labels, modalities


def visualize_embeddings(embeddings, class_labels, modalities, class_names, 
                        method='tsne', save_path='visualization.png'):
    """
    Visualize embeddings using different dimensionality reduction methods
    
    Args:
        method: 'tsne', 'umap', or 'pca'
    """
    print(f"Running {method.upper()}...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        embeddings_2d = reducer.fit_transform(embeddings)
        title_suffix = 't-SNE'
        xlabel, ylabel = 't-SNE 1', 't-SNE 2'
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_2d = reducer.fit_transform(embeddings)
        title_suffix = 'UMAP'
        xlabel, ylabel = 'UMAP 1', 'UMAP 2'
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
        title_suffix = 'PCA'
        xlabel, ylabel = 'PC 1', 'PC 2'
        print(f"  Explained variance: {reducer.explained_variance_ratio_[0]:.3f}, {reducer.explained_variance_ratio_[1]:.3f}")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Color palette
    colors = sns.color_palette('husl', n_colors=len(class_names))
    
    # Plot 1: Color by class
    ax = axes[0]
    for class_idx, class_name in enumerate(class_names):
        mask = class_labels == class_idx
        
        # Raman samples
        raman_mask = mask & (modalities < 0.5)
        ax.scatter(embeddings_2d[raman_mask, 0], embeddings_2d[raman_mask, 1],
                  c=[colors[class_idx]], marker='o', s=50, alpha=0.6,
                  label=f'{class_name} (Raman)', edgecolors='black', linewidths=0.5)
        
        # GC samples
        gc_mask = mask & (modalities > 0.5)
        ax.scatter(embeddings_2d[gc_mask, 0], embeddings_2d[gc_mask, 1],
                  c=[colors[class_idx]], marker='s', s=50, alpha=0.6,
                  label=f'{class_name} (GC)', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Embedding Space ({title_suffix}): Colored by Class (○=Raman, □=GC)', 
                fontsize=14, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by modality
    ax = axes[1]
    raman_mask = modalities < 0.5
    gc_mask = modalities > 0.5
    
    ax.scatter(embeddings_2d[raman_mask, 0], embeddings_2d[raman_mask, 1],
              c='blue', marker='o', s=50, alpha=0.5, label='Raman', edgecolors='black', linewidths=0.5)
    ax.scatter(embeddings_2d[gc_mask, 0], embeddings_2d[gc_mask, 1],
              c='red', marker='s', s=50, alpha=0.5, label='GC', edgecolors='black', linewidths=0.5)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'Embedding Space ({title_suffix}): Colored by Modality', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved {method.upper()} visualization to {save_path}")
    plt.close()


def visualize_tsne(embeddings, class_labels, modalities, class_names, save_path='tsne_visualization.png'):
    """t-SNE visualization (backward compatibility)"""
    visualize_embeddings(embeddings, class_labels, modalities, class_names, 
                        method='tsne', save_path=save_path)


def compute_metrics(embeddings, class_labels, modalities, class_names):
    """Compute clustering and alignment metrics"""
    print("\n" + "="*60)
    print("Embedding Space Metrics")
    print("="*60)
    
    # 1. Intra-class similarity (같은 클래스 내 유사도)
    print("\n1. Intra-class Similarity (higher is better):")
    for class_idx, class_name in enumerate(class_names):
        mask = class_labels == class_idx
        class_embeddings = embeddings[mask]
        
        if len(class_embeddings) > 1:
            # Pairwise cosine similarity
            sim_matrix = class_embeddings @ class_embeddings.T
            # Exclude diagonal
            n = len(class_embeddings)
            intra_sim = (sim_matrix.sum() - n) / (n * (n - 1))
            print(f"   {class_name}: {intra_sim:.4f}")
    
    # 2. Cross-modal alignment (같은 클래스 내 Raman-GC 유사도)
    print("\n2. Cross-Modal Alignment (higher is better):")
    for class_idx, class_name in enumerate(class_names):
        class_mask = class_labels == class_idx
        raman_mask = class_mask & (modalities < 0.5)
        gc_mask = class_mask & (modalities > 0.5)
        
        raman_emb = embeddings[raman_mask]
        gc_emb = embeddings[gc_mask]
        
        if len(raman_emb) > 0 and len(gc_emb) > 0:
            # Average similarity between Raman and GC
            cross_modal_sim = (raman_emb @ gc_emb.T).mean()
            print(f"   {class_name}: {cross_modal_sim:.4f}")
    
    # 3. Inter-class separation (다른 클래스 간 거리)
    print("\n3. Inter-Class Similarity (lower is better):")
    for i, class_i in enumerate(class_names):
        for j, class_j in enumerate(class_names):
            if i < j:
                mask_i = class_labels == i
                mask_j = class_labels == j
                
                emb_i = embeddings[mask_i]
                emb_j = embeddings[mask_j]
                
                if len(emb_i) > 0 and len(emb_j) > 0:
                    inter_sim = (emb_i @ emb_j.T).mean()
                    print(f"   {class_i} ↔ {class_j}: {inter_sim:.4f}")
    
    print("="*60 + "\n")


def main(max_samples=10000, stratified=True, samples_per_class_modality=None):
    """
    Main visualization function
    
    Args:
        max_samples: Maximum total samples (if stratified=False)
        stratified: Use stratified sampling (recommended for balanced view)
        samples_per_class_modality: Samples per (class, modality) pair
                                   Default: None (auto-calculate)
    
    Recommendations:
        - For quick preview: max_samples=5000, stratified=True
        - For detailed view: max_samples=20000, stratified=True
        - For ALL data: stratified=False, max_samples=999999 (but very slow!)
    """
    print("="*80)
    print("Cross-Modal Embedding Visualization")
    print("="*80)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading model...")
    model = CrossModalContrastiveModel(
        hidden_dim=config.HIDDEN_DIM,
        embed_dim=config.EMBEDDING_DIM
    ).to(device)
    
    if Path(config.MODEL_DIR).exists():
        checkpoint = torch.load(config.MODEL_DIR, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {config.MODEL_DIR}")
        print(f"Trained for {checkpoint['epoch']} epochs")
    else:
        print(f"Warning: Model file {config.MODEL_DIR} not found!")
        print("Using randomly initialized model (visualization will be meaningless)")
    
    # Load data
    print("\nPreparing data...")
    _, val_loader, class_names = prepare_dataloader(
        use_lazy_loading=config.USE_LAZY_LOADING,
        use_parallel=False,  # Don't need parallel for visualization
        max_workers=1,
        num_workers=2
    )
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    print(f"Settings: max_samples={max_samples}, stratified={stratified}")
    embeddings, class_labels, modalities = extract_embeddings(
        model, val_loader, device, 
        max_samples=max_samples,
        stratified=stratified,
        samples_per_class_modality=samples_per_class_modality
    )
    
    print(f"\nExtracted {len(embeddings)} samples total")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Print sample distribution
    print("\nSample distribution:")
    for class_idx, class_name in enumerate(class_names):
        raman_count = ((class_labels == class_idx) & (modalities < 0.5)).sum()
        gc_count = ((class_labels == class_idx) & (modalities > 0.5)).sum()
        print(f"  {class_name}: {raman_count} Raman, {gc_count} GC")
    
    # Compute metrics
    compute_metrics(embeddings, class_labels, modalities, class_names)
    
    # Output directory
    output_dir = Path(config.MODEL_DIR).parent
    
    # Visualize with multiple methods
    print("\n" + "="*80)
    print("Creating visualizations...")
    print("="*80)
    
    # 1. t-SNE (best for local structure)
    print("\n1. t-SNE visualization...")
    visualize_embeddings(embeddings, class_labels, modalities, class_names,
                        method='tsne', save_path=output_dir / 'embedding_tsne.png')
    
    # 2. PCA (fastest, shows global variance)
    print("\n2. PCA visualization...")
    visualize_embeddings(embeddings, class_labels, modalities, class_names,
                        method='pca', save_path=output_dir / 'embedding_pca.png')
    
    # 3. UMAP (balance of local and global, faster than t-SNE)
    print("\n3. UMAP visualization...")
    try:
        visualize_embeddings(embeddings, class_labels, modalities, class_names,
                            method='umap', save_path=output_dir / 'embedding_umap.png')
    except Exception as e:
        print(f"   Warning: UMAP failed ({str(e)})")
        print("   Install UMAP with: pip install umap-learn")
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print(f"Saved to: {output_dir}")
    print("  - embedding_tsne.png (t-SNE: local structure)")
    print("  - embedding_pca.png (PCA: global variance)")
    print("  - embedding_umap.png (UMAP: balanced view)")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize cross-modal embeddings')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum total samples (default: 10000)')
    parser.add_argument('--stratified', action='store_true', default=True,
                       help='Use stratified sampling (default: True)')
    parser.add_argument('--no-stratified', dest='stratified', action='store_false',
                       help='Disable stratified sampling')
    parser.add_argument('--samples-per-group', type=int, default=None,
                       help='Samples per (class, modality) pair. Default: auto-calculate')
    parser.add_argument('--all', action='store_true',
                       help='Visualize ALL validation data (warning: very slow!)')
    
    args = parser.parse_args()
    
    if args.all:
        print("Warning: Visualizing ALL data. This will take a LONG time!")
        print("Consider using --max-samples 20000 instead for a detailed view.")
        max_samples = 999999
        stratified = False
    else:
        max_samples = args.max_samples
        stratified = args.stratified
    
    main(max_samples=max_samples, 
         stratified=stratified, 
         samples_per_class_modality=args.samples_per_group)
