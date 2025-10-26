"""
3D Interactive Visualization for Cross-Modal Embeddings

(MODIFIED: Added jitter to GC samples for better visibility)

3D 공간에서 임베딩을 시각화하고 마우스로 회전/확대 가능:
- t-SNE 3D
- UMAP 3D  
- PCA 3D
- HTML 파일로 저장되어 브라우저에서 인터랙티브하게 확인 가능
"""

import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
try:
    import umap.umap_ as umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Install with: pip install umap-learn")

from pathlib import Path
from tqdm import tqdm
import seaborn as sns

import config
from model import CrossModalContrastiveModel
from dataset import prepare_dataloader


@torch.no_grad()
def extract_embeddings(model, dataloader, device, max_samples=10000, 
                      stratified=True, samples_per_class_modality=None):
    """
    Extract embeddings with stratified sampling
    """
    model.eval()
    
    if stratified and samples_per_class_modality is None:
        samples_per_class_modality = max(100, max_samples // 8)
        print(f"Using stratified sampling: {samples_per_class_modality} samples per (class, modality) pair")
    
    if stratified:
        from collections import defaultdict
        group_samples = defaultdict(lambda: {'embeddings': [], 'labels': [], 'modalities': []})
        
        for spectra, labels_and_mod in tqdm(dataloader, desc='Extracting embeddings (stratified)'):
            spectra = spectra.to(device)
            labels_and_mod = labels_and_mod.to(device)
            
            class_labels = labels_and_mod[:, :-1]
            modalities = labels_and_mod[:, -1]
            
            embeddings = model(spectra, modalities)
            
            class_indices = torch.argmax(class_labels, dim=1).cpu().numpy()
            modality_values = modalities.cpu().numpy()
            embedding_values = embeddings.cpu().numpy()
            
            for i in range(len(class_indices)):
                class_idx = int(class_indices[i])
                modality = int(modality_values[i])
                key = (class_idx, modality)
                
                if len(group_samples[key]['embeddings']) < samples_per_class_modality:
                    group_samples[key]['embeddings'].append(embedding_values[i])
                    group_samples[key]['labels'].append(class_idx)
                    group_samples[key]['modalities'].append(modality)
            
            all_full = all(len(group['embeddings']) >= samples_per_class_modality 
                          for group in group_samples.values())
            if all_full and len(group_samples) >= 8:
                break
        
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
        
    else:
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


def create_3d_visualization(embeddings_3d, class_labels, modalities, class_names, 
                           method='t-SNE', title='3D Embedding Visualization'):
    """
    Create interactive 3D visualization with Plotly
    (MODIFIED: Added jitter to GC samples)
    """
    # Color palette
    colors_rgb = sns.color_palette('husl', n_colors=len(class_names))
    colors_hex = [f'rgb({int(r*255)}, {int(g*255)}, {int(b*255)})' for r, g, b in colors_rgb]
    
    # --- [NEW JITTER LOGIC] ---
    # 겹침 문제를 해결하기 위해 GC 샘플에 미세한 노이즈(Jitter) 추가
    data_std = np.std(embeddings_3d, axis=0)
    jitter_strength = 0.02 # 데이터 표준편차의 2%만큼 Jitter
    
    raman_mask = modalities < 0.5
    gc_mask = modalities > 0.5
    
    # 시각화용 임베딩 배열 복사
    embeddings_3d_viz = np.copy(embeddings_3d)
    
    # GC 샘플에만 Jitter 적용
    n_gc = gc_mask.sum()
    if n_gc > 0:
        # (n_gc, 3) 크기의 0~1 사이 랜덤값 생성, -0.5를 곱해 (-0.5~0.5)로 변경
        jitter = (np.random.rand(n_gc, 3) - 0.5) * data_std * jitter_strength
        embeddings_3d_viz[gc_mask] += jitter
    # --- [END NEW JITTER LOGIC] ---
    
    
    # Create subplots: [Class view, Modality view]
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'{method}: Colored by Class', f'{method}: Colored by Modality'),
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        horizontal_spacing=0.1
    )
    
    # Plot 1: Color by class (with modality as marker)
    for class_idx, class_name in enumerate(class_names):
        mask = class_labels == class_idx
        
        # Raman samples (circles)
        # [MODIFIED] Use raman_mask (pre-calculated)
        raman_plot_mask = mask & raman_mask
        if raman_plot_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=embeddings_3d_viz[raman_plot_mask, 0], # Jitter가 적용되지 않은 원본
                    y=embeddings_3d_viz[raman_plot_mask, 1],
                    z=embeddings_3d_viz[raman_plot_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors_hex[class_idx],
                        symbol='circle',
                        line=dict(color='black', width=0.5),
                        opacity=0.7
                    ),
                    name=f'{class_name} (Raman)',
                    legendgroup=f'class_{class_idx}',
                    hovertemplate=f'<b>{class_name} (Raman)</b><br>' +
                                 'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # GC samples (squares)
        # [MODIFIED] Use gc_mask (pre-calculated)
        gc_plot_mask = mask & gc_mask
        if gc_plot_mask.any():
            fig.add_trace(
                go.Scatter3d(
                    x=embeddings_3d_viz[gc_plot_mask, 0], # Jitter가 적용된 GC
                    y=embeddings_3d_viz[gc_plot_mask, 1],
                    z=embeddings_3d_viz[gc_plot_mask, 2],
                    mode='markers',
                    marker=dict(
                        size=4,
                        color=colors_hex[class_idx],
                        symbol='square',
                        line=dict(color='black', width=0.5),
                        opacity=0.7
                    ),
                    name=f'{class_name} (GC)',
                    legendgroup=f'class_{class_idx}',
                    hovertemplate=f'<b>{class_name} (GC - Jittered)</b><br>' +
                                 'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
                ),
                row=1, col=1
            )
    
    # Plot 2: Color by modality
    # raman_mask and gc_mask are already defined
    
    if raman_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=embeddings_3d_viz[raman_mask, 0], # Jitter가 적용되지 않은 원본
                y=embeddings_3d_viz[raman_mask, 1],
                z=embeddings_3d_viz[raman_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='blue',
                    symbol='circle',
                    line=dict(color='black', width=0.5),
                    opacity=0.6
                ),
                name='Raman',
                hovertemplate='<b>Raman</b><br>' +
                             'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    if gc_mask.any():
        fig.add_trace(
            go.Scatter3d(
                x=embeddings_3d_viz[gc_mask, 0], # Jitter가 적용된 GC
                y=embeddings_3d_viz[gc_mask, 1],
                z=embeddings_3d_viz[gc_mask, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color='red',
                    symbol='square',
                    line=dict(color='black', width=0.5),
                    opacity=0.6
                ),
                name='GC (Jittered)',
                hovertemplate='<b>GC (Jittered)</b><br>' +
                             'X: %{x:.2f}<br>Y: %{y:.2f}<br>Z: %{z:.2f}<extra></extra>'
            ),
            row=1, col=2
        )
    
    # Update layout
    # --- [MODIFIED] Title updated to mention jitter ---
    fig.update_layout(
        title=dict(
            text=f'<b>{title}</b><br><sup>Drag to rotate, Scroll to zoom, Double-click to reset</sup>' +
                 '<br><sup>(GC samples are slightly jittered for visibility)</sup>',
            x=0.5,
            xanchor='center'
        ),
    # --- [END MODIFIED] ---
        width=1600,
        height=800,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="Black",
            borderwidth=1
        ),
        scene=dict(
            xaxis_title=f'{method} 1',
            yaxis_title=f'{method} 2',
            zaxis_title=f'{method} 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        scene2=dict(
            xaxis_title=f'{method} 1',
            yaxis_title=f'{method} 2',
            zaxis_title=f'{method} 3',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )
    
    return fig


def visualize_3d(embeddings, class_labels, modalities, class_names, 
                method='tsne', save_path='visualization_3d.html'):
    """
    Reduce to 3D and create interactive visualization
    
    Args:
        method: 'tsne', 'umap', or 'pca'
    """
    print(f"\nRunning {method.upper()} for 3D...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=3, random_state=42, perplexity=30, n_iter=1000)
        embeddings_3d = reducer.fit_transform(embeddings)
        title = '3D t-SNE Visualization'
    elif method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not available. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.1)
        embeddings_3d = reducer.fit_transform(embeddings)
        title = '3D UMAP Visualization'
    elif method == 'pca':
        reducer = PCA(n_components=3, random_state=42)
        embeddings_3d = reducer.fit_transform(embeddings)
        variance = reducer.explained_variance_ratio_
        print(f"  Explained variance: PC1={variance[0]:.3f}, PC2={variance[1]:.3f}, PC3={variance[2]:.3f}")
        title = f'3D PCA Visualization (Total variance: {variance.sum():.1%})'
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create interactive plot
    print(f"Creating interactive 3D plot...")
    fig = create_3d_visualization(embeddings_3d, class_labels, modalities, 
                                  class_names, method.upper(), title)
    
    # Save to HTML
    fig.write_html(save_path)
    print(f"Saved interactive 3D visualization to {save_path}")
    print(f"Open {save_path} in your web browser to interact with the plot!")
    
    return fig


def main(max_samples=10000, stratified=True):
    """
    Main function for 3D visualization
    """
    print("="*80)
    print("3D Interactive Cross-Modal Embedding Visualization")
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
        use_parallel=False,
        max_workers=1,
        num_workers=2
    )
    
    # Extract embeddings
    print(f"\nExtracting embeddings (max_samples={max_samples}, stratified={stratified})...")
    embeddings, class_labels, modalities = extract_embeddings(
        model, val_loader, device, 
        max_samples=max_samples,
        stratified=stratified
    )
    
    print(f"\nExtracted {len(embeddings)} samples")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Print distribution
    print("\nSample distribution:")
    for class_idx, class_name in enumerate(class_names):
        raman_count = ((class_labels == class_idx) & (modalities < 0.5)).sum()
        gc_count = ((class_labels == class_idx) & (modalities > 0.5)).sum()
        print(f"  {class_name}: {raman_count} Raman, {gc_count} GC")
    
    # Output directory
    output_dir = Path(config.MODEL_PLOT_DIR).parent
    
    # Create 3D visualizations
    print("\n" + "="*80)
    print("Creating 3D visualizations...")
    print("="*80)
    
    # 1. PCA 3D (fastest)
    print("\n1. PCA 3D...")
    visualize_3d(embeddings, class_labels, modalities, class_names,
                method='pca', save_path=output_dir / 'embedding_3d_pca.html')
    
    # 2. UMAP 3D (balanced)
    print("\n2. UMAP 3D...")
    try:
        visualize_3d(embeddings, class_labels, modalities, class_names,
                    method='umap', save_path=output_dir / 'embedding_3d_umap.html')
    except Exception as e:
        print(f"   Warning: UMAP failed ({str(e)})")
        print("   Install UMAP with: pip install umap-learn")
    
    # 3. t-SNE 3D (local structure, slowest)
    print("\n3. t-SNE 3D (this may take a while)...")
    visualize_3d(embeddings, class_labels, modalities, class_names,
                method='tsne', save_path=output_dir / 'embedding_3d_tsne.html')
    
    print("\n" + "="*80)
    print("3D Visualization complete!")
    print(f"Saved to: {output_dir}")
    print("  - embedding_3d_pca.html (PCA: global variance)")
    print("  - embedding_3d_umap.html (UMAP: balanced view)")
    print("  - embedding_3d_tsne.html (t-SNE: local structure)")
    print("\nOpen these HTML files in your web browser to interact!")
    print("  - Drag to rotate")
    print("  - Scroll to zoom")
    print("  - Double-click to reset view")
    print("="*80)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Create 3D interactive visualizations')
    parser.add_argument('--max-samples', type=int, default=10000,
                       help='Maximum total samples (default: 10000)')
    parser.add_argument('--stratified', action='store_true', default=True,
                       help='Use stratified sampling (default: True)')
    parser.add_argument('--no-stratified', dest='stratified', action='store_false',
                       help='Disable stratified sampling')
    
    args = parser.parse_args()
    
    main(max_samples=args.max_samples, stratified=args.stratified)