#!/usr/bin/env python3
"""
Create an improved spermatogenesis overview plot with bigger legend fonts and cluster numbers.

This script recreates the spermatogenesis_overview.png with enhanced typography
for better readability.
"""

import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

# Set scanpy settings for better plot quality
sc.settings.set_figure_params(dpi=300, facecolor='white')

def create_improved_overview_plot(adata, output_path='spermatogenesis_overview_improved.png'):
    """
    Create the spermatogenesis overview plot with improved typography.
    
    Args:
        adata: AnnData object with processed spermatogenesis data
        output_path: Path for the output PNG file
    """
    
    print("Creating improved spermatogenesis overview plot...")
    
    # Create the main overview plot with enhanced font sizes
    fig, axes = plt.subplots(2, 3, figsize=(24, 16))  # Slightly larger figure
    
    # 1. Clusters with bigger numbers on data
    sc.pl.umap(adata, color='leiden_0.3', 
              legend_loc='on data', 
              legend_fontsize=16,  # Increased from 10
              legend_fontweight='bold',
              ax=axes[0,0], 
              show=False, 
              frameon=False,
              size=40)  # Larger point size
    axes[0,0].set_title('Leiden Clusters (0.3)\nannotated according to cluster_summary.csv', 
                       fontsize=18, fontweight='bold', pad=20)
    
    # Enhance cluster number visibility on the plot
    # Get cluster centers for annotation
    if 'X_umap' in adata.obsm:
        umap_coords = adata.obsm['X_umap']
        clusters = adata.obs['leiden_0.3'].astype('category')
        
        # Calculate cluster centers
        for cluster in clusters.cat.categories:
            cluster_mask = clusters == cluster
            if cluster_mask.sum() > 0:
                cluster_coords = umap_coords[cluster_mask]
                center_x = np.median(cluster_coords[:, 0])
                center_y = np.median(cluster_coords[:, 1])
                
                # Add large, bold cluster numbers
                axes[0,0].text(center_x, center_y, str(cluster), 
                              fontsize=20, fontweight='bold', 
                              ha='center', va='center',
                              bbox=dict(boxstyle='round,pad=0.3', 
                                       facecolor='white', 
                                       edgecolor='black',
                                       alpha=0.8))
    
    # 2. Annotations with bigger legend
    sc.pl.umap(adata, color='cluster_annotation', 
              legend_loc='right margin',
              legend_fontsize=14,  # Increased from default
              ax=axes[0,1], 
              show=False, 
              frameon=False,
              size=40)
    axes[0,1].set_title('Cell Type Annotations', fontsize=18, fontweight='bold', pad=20)
    
    # 3. Pseudotime
    if 'dpt_pseudotime' in adata.obs.columns:
        sc.pl.umap(adata, color='dpt_pseudotime', 
                  color_map='viridis',
                  ax=axes[0,2], 
                  show=False, 
                  frameon=False,
                  size=40)
        axes[0,2].set_title('Diffusion Pseudotime', fontsize=18, fontweight='bold', pad=20)
    else:
        axes[0,2].text(0.5, 0.5, 'Pseudotime\nNot Available', 
                      ha='center', va='center',
                      transform=axes[0,2].transAxes, 
                      fontsize=16, fontweight='bold')
        axes[0,2].set_title('Pseudotime', fontsize=18, fontweight='bold', pad=20)
    
    # 4. Progression score
    if 'progression_score' in adata.obs.columns:
        sc.pl.umap(adata, color='progression_score', 
                  color_map='RdYlBu_r',
                  ax=axes[1,0], 
                  show=False, 
                  frameon=False,
                  size=40)
        axes[1,0].set_title('Spermatogenesis Progression Score', 
                           fontsize=18, fontweight='bold', pad=20)
    else:
        axes[1,0].text(0.5, 0.5, 'Progression Score\nNot Available', 
                      ha='center', va='center',
                      transform=axes[1,0].transAxes, 
                      fontsize=16, fontweight='bold')
        axes[1,0].set_title('Spermatogenesis Progression Score', 
                           fontsize=18, fontweight='bold', pad=20)
    
    # 5. Total UMI with enhanced colorbar
    sc.pl.umap(adata, color='total_counts', 
              color_map='plasma',
              ax=axes[1,1], 
              show=False, 
              frameon=False,
              size=40)
    axes[1,1].set_title('Total UMI Counts', fontsize=18, fontweight='bold', pad=20)
    
    # 6. Number of genes
    sc.pl.umap(adata, color='n_genes_by_counts', 
              color_map='viridis',
              ax=axes[1,2], 
              show=False, 
              frameon=False,
              size=40)
    axes[1,2].set_title('Number of Genes', fontsize=18, fontweight='bold', pad=20)
    
    # Enhance all axis labels
    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=12)
            ax.set_xlabel('UMAP 1', fontsize=14, fontweight='bold')
            ax.set_ylabel('UMAP 2', fontsize=14, fontweight='bold')
    
    # Adjust layout with more padding
    plt.tight_layout(pad=3.0)
    
    # Save with high quality
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Improved overview plot saved to: {output_path}")
    
    # Also save a copy in the current directory for easy access
    current_dir_path = Path.cwd() / 'spermatogenesis_overview_improved.png'
    plt.savefig(current_dir_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    return fig

def add_enhanced_gene_markers(adata, output_path='key_marker_genes_improved.png'):
    """Create enhanced marker gene plots with better typography."""
    
    # Define key markers for spermatogenesis
    key_markers = ['PRM1', 'TNP1', 'SMC1B', 'SYCP3', 'DDX4', 'ACRV1']
    
    # Check which markers are available
    if hasattr(adata, 'raw') and adata.raw is not None:
        available_markers = [m for m in key_markers if m in adata.raw.var_names]
        use_raw = True
    else:
        available_markers = [m for m in key_markers if m in adata.var_names]
        use_raw = False
    
    print(f"Found {len(available_markers)} markers: {available_markers}")
    
    if len(available_markers) >= 4:
        # Create a 2x3 layout for better visibility
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, marker in enumerate(available_markers[:6]):
            sc.pl.umap(adata, color=marker, 
                      use_raw=use_raw, 
                      color_map='Reds',
                      ax=axes[i], 
                      show=False, 
                      frameon=False,
                      size=30)
            axes[i].set_title(f'{marker} Expression', 
                             fontsize=16, fontweight='bold', pad=15)
            axes[i].set_xlabel('UMAP 1', fontsize=12, fontweight='bold')
            axes[i].set_ylabel('UMAP 2', fontsize=12, fontweight='bold')
            axes[i].tick_params(labelsize=10)
        
        # Hide unused axes
        for i in range(len(available_markers), 6):
            axes[i].set_visible(False)
        
        plt.tight_layout(pad=2.0)
        plt.savefig(output_path, dpi=300, bbox_inches='tight',
                    facecolor='white', edgecolor='none')
        
        print(f"Enhanced marker gene plot saved to: {output_path}")
        plt.show()
        
        return fig
    else:
        print("Not enough marker genes available for enhanced plot")
        return None

def main():
    """Main function to create improved plots."""
    
    # Path to the processed data
    data_path = Path('/Users/oskarhaupt/Documents/DE/2024_FU-Bachelor/WS-24-25/Charité/05_sorted/GSEA_new/improved_spermatogenesis_trajectory.h5ad')
    
    if not data_path.exists():
        print(f"Data file not found: {data_path}")
        print("Please make sure the improved_spermatogenesis_trajectory.h5ad file exists.")
        return 1
    
    print(f"Loading data from: {data_path}")
    adata = ad.read_h5ad(data_path)
    
    print(f"Loaded data: {adata.n_obs} cells × {adata.n_vars} genes")
    print(f"Available columns: {list(adata.obs.columns)}")
    
    # Create the improved overview plot
    fig1 = create_improved_overview_plot(adata)
    
    # Create enhanced marker gene plots
    fig2 = add_enhanced_gene_markers(adata)
    
    print("\n✅ Enhanced plots created successfully!")
    print("📁 Files saved:")
    print("   - spermatogenesis_overview_improved.png")
    print("   - key_marker_genes_improved.png")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
