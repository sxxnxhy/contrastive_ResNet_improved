"""
Visualization script for cross-modal embeddings (PCA 2D)

(MODIFIED: Plot order changed to draw circles (Raman) on top of squares (GC))

임베딩 공간을 시각화하여:
1. 클래스별 클러스터링 확인
2. Cross-modal alignment 확인 (Raman과 GC가 같은 클래스에서 섞여있는지)
3. Inter-class separation 확인
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # t-SNE -> PCA로 변경
import seaborn as sns
from pathlib import Path
from tqdm import tqdm

import config
from model import CrossModalContrastiveModel
from dataset import prepare_dataloader


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=5000):
    """Extract embeddings from the model"""
    model.eval()
    
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


def visualize_pca(embeddings, class_labels, modalities, class_names, save_path='pca_visualization_2d.png'):
    """PCA visualization (t-SNE에서 변경)"""
    print("Running PCA for 2D...")
    
    # --- [MODIFIED] t-SNE -> PCA ---
    pca = PCA(n_components=2, random_state=42)
    embeddings_2d = pca.fit_transform(embeddings)
    variance = pca.explained_variance_ratio_
    total_variance = variance.sum()
    print(f"  Explained variance: PC1={variance[0]:.3f}, PC2={variance[1]:.3f}, Total={total_variance:.3f}")
    # --- [END MODIFIED] ---

    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    
    # Color palette
    colors = sns.color_palette('husl', n_colors=len(class_names))
    
    # Plot 1: Color by class
    ax = axes[0]
    for class_idx, class_name in enumerate(class_names):
        mask = class_labels == class_idx
        
        # --- [MODIFIED] Plot GC (squares) FIRST ---
        gc_mask = mask & (modalities > 0.5)
        ax.scatter(embeddings_2d[gc_mask, 0], embeddings_2d[gc_mask, 1],
                  c=[colors[class_idx]], marker='s', s=50, alpha=0.6,
                  label=f'{class_name} (GC)', edgecolors='black', linewidths=0.5)
        # --- [END MODIFIED] ---

        # --- [MODIFIED] Plot Raman (circles) SECOND (on top) ---
        raman_mask = mask & (modalities < 0.5)
        ax.scatter(embeddings_2d[raman_mask, 0], embeddings_2d[raman_mask, 1],
                  c=[colors[class_idx]], marker='o', s=50, alpha=0.6,
                  label=f'{class_name} (Raman)', edgecolors='black', linewidths=0.5)
        # --- [END MODIFIED] ---
    
    # --- [MODIFIED] Labels and Title ---
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title(f'PCA Embedding Space (Total Variance: {total_variance:.1%})\nColored by Class (○=Raman, □=GC)', fontsize=14, fontweight='bold')
    # --- [END MODIFIED] ---
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Color by modality
    ax = axes[1]
    raman_mask = modalities < 0.5
    gc_mask = modalities > 0.5
    
    # --- [MODIFIED] Plot GC (red squares) FIRST ---
    ax.scatter(embeddings_2d[gc_mask, 0], embeddings_2d[gc_mask, 1],
              c='red', marker='s', s=50, alpha=0.5, label='GC', edgecolors='black', linewidths=0.5)
    # --- [END MODIFIED] ---
    
    # --- [MODIFIED] Plot Raman (blue circles) SECOND (on top) ---
    ax.scatter(embeddings_2d[raman_mask, 0], embeddings_2d[raman_mask, 1],
              c='blue', marker='o', s=50, alpha=0.5, label='Raman', edgecolors='black', linewidths=0.5)
    # --- [END MODIFIED] ---
    
    # --- [MODIFIED] Labels and Title ---
    ax.set_xlabel('Principal Component 1', fontsize=12)
    ax.set_ylabel('Principal Component 2', fontsize=12)
    ax.set_title(f'PCA Embedding Space (Total Variance: {total_variance:.1%})\nColored by Modality', fontsize=14, fontweight='bold')
    # --- [END MODIFIED] ---
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
    plt.close()


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


def main():
    """Main visualization function"""
    print("="*80)
    print("Cross-Modal Embedding Visualization (PCA 2D)")
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
    
    if Path(config.MODEL_PLOT_DIR).exists():
        checkpoint = torch.load(config.MODEL_PLOT_DIR, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from {config.MODEL_PLOT_DIR}")
        print(f"Trained for {checkpoint['epoch']} epochs")
    else:
        print(f"Warning: Model file {config.MODEL_PLOT_DIR} not found!")
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
    embeddings, class_labels, modalities = extract_embeddings(
        model, val_loader, device, max_samples=5000
    )
    
    print(f"Extracted {len(embeddings)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Compute metrics
    compute_metrics(embeddings, class_labels, modalities, class_names)
    
    # --- [MODIFIED] ---
    # Visualize
    print("Creating PCA 2D visualization...")
    output_dir = Path(config.MODEL_PLOT_DIR).parent
    save_path = output_dir / 'embedding_pca_2d.png'
    visualize_pca(embeddings, class_labels, modalities, class_names, save_path)
    # --- [END MODIFIED] ---
    
    print("\n" + "="*80)
    print("Visualization complete!")
    print("="*80)


if __name__ == '__main__':
    main()