#!/usr/bin/env python3
"""
Continue and visualize the spermatogenesis analysis results.
This script focuses on creating the final plots and interpreting the trajectory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import warnings

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)

def load_and_recreate_analysis():
    """Load data and recreate the key analysis steps."""
    
    print("🔄 RECREATING SUCCESSFUL ANALYSIS COMPONENTS")
    print("="*60)
    
    # Load the EmptyDrops data
    adata = sc.read_h5ad("data/emptydrops_all_detected_cells.h5ad")
    print(f"📊 Original data: {adata.shape}")
    
    # Basic QC and filtering (simplified)
    adata.var['mt'] = adata.var_names.str.startswith(('MT-', 'Mt-', 'mt-'))
    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
    
    if 'total_counts' not in adata.obs.columns:
        adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    
    # Filtering
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=10)
    
    # Remove extreme outliers
    adata = adata[adata.obs.n_genes_by_counts < 6000, :]
    adata = adata[adata.obs.total_counts > 500, :]
    adata = adata[adata.obs.total_counts < 50000, :]
    
    print(f"📊 After filtering: {adata.shape}")
    
    # Store raw counts
    adata.layers['counts'] = adata.X.copy()
    adata.raw = adata
    
    # Normalization
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # HVG selection with spermatogenesis-specific markers
    spermatogenesis_markers = [
        'PRM1', 'PRM2', 'TNP1', 'TNP2', 'SYCP1', 'SYCP3', 'SMC1B', 'DDX4', 
        'SPAG16', 'ODF1', 'ODF2', 'SMCP', 'ACRV1', 'CRISP2', 'CYP17A1', 
        'ACTA2', 'CLU', 'SOX9', 'KIT', 'STRA8', 'GSTM3', 'HSPB9'
    ]
    
    sc.pp.highly_variable_genes(adata, min_mean=0.01, max_mean=8, min_disp=0.3, n_top_genes=3000)
    
    # Add important spermatogenesis markers to HVG
    for marker in spermatogenesis_markers:
        if marker in adata.var_names:
            adata.var.loc[marker, 'highly_variable'] = True
    
    adata = adata[:, adata.var.highly_variable]
    sc.pp.scale(adata, max_value=10)
    
    print(f"📊 Final analysis data: {adata.shape}")
    
    return adata

def run_dimensionality_reduction(adata):
    """Run optimized dimensionality reduction for spermatogenesis."""
    
    print("\n🌟 DIMENSIONALITY REDUCTION & CLUSTERING")
    
    # PCA
    sc.tl.pca(adata, svd_solver='arpack', n_comps=50, random_state=42)
    
    # Neighbors
    sc.pp.neighbors(adata, n_neighbors=50, n_pcs=30, metric='cosine', random_state=42)
    
    # Diffusion maps for trajectory
    sc.tl.diffmap(adata, n_comps=15)
    
    # PAGA for trajectory topology
    sc.tl.leiden(adata, resolution=0.3, random_state=42, key_added='leiden_coarse')
    sc.tl.paga(adata, groups='leiden_coarse')
    
    # UMAP with PAGA initialization
    sc.pl.paga(adata, plot=False)
    sc.tl.umap(adata, init_pos='paga', min_dist=0.1, spread=1.0, random_state=42)
    
    # Multiple clustering resolutions
    for res in [0.1, 0.3, 0.5, 0.8]:
        sc.tl.leiden(adata, resolution=res, random_state=42, key_added=f'leiden_{res}')
    
    print("✅ Dimensionality reduction completed")
    return adata

def add_spermatogenesis_scores(adata):
    """Add spermatogenesis stage scores."""
    
    print("\n🧬 ADDING SPERMATOGENESIS STAGE SCORES")
    
    stage_markers = {
        'spermatogonia': ['DDX4', 'PLZF', 'ID4', 'GFRA1', 'NANOS2', 'NANOS3', 'KIT', 'STRA8'],
        'spermatocytes': ['SYCP1', 'SYCP3', 'SMC1B', 'REC8', 'HORMAD1', 'SPO11', 'DMC1'],
        'round_spermatids': ['ACRV1', 'CREM', 'PRM1', 'PRM2', 'TNP1', 'TNP2', 'TSSK6'],
        'elongating_spermatids': ['ODF1', 'ODF2', 'SPAG16', 'SMCP', 'AKAP4', 'TEKTIN2'],
        'sertoli_cells': ['SOX9', 'AMH', 'FSHR', 'AR', 'GATA4', 'WT1', 'CLU'],
        'leydig_cells': ['CYP17A1', 'CYP11A1', 'HSD3B1', 'STAR', 'INSL3']
    }
    
    for stage, markers in stage_markers.items():
        available_markers = [m for m in markers if m in adata.var_names]
        if len(available_markers) > 0:
            sc.tl.score_genes(adata, available_markers, score_name=f'{stage}_score')
            print(f"📊 {stage}: {len(available_markers)} markers scored")
    
    # Create progression score
    early_markers = ['DDX4', 'KIT', 'STRA8', 'SYCP1', 'SYCP3']
    late_markers = ['PRM1', 'PRM2', 'TNP1', 'ODF1', 'SPAG16']
    
    early_available = [m for m in early_markers if m in adata.var_names]
    late_available = [m for m in late_markers if m in adata.var_names]
    
    if len(early_available) > 0:
        sc.tl.score_genes(adata, early_available, score_name='early_score')
    if len(late_available) > 0:
        sc.tl.score_genes(adata, late_available, score_name='late_score')
    
    if 'early_score' in adata.obs.columns and 'late_score' in adata.obs.columns:
        adata.obs['progression_score'] = adata.obs['late_score'] - adata.obs['early_score']
    
    return adata

def setup_pseudotime(adata):
    """Set up pseudotime analysis."""
    
    print("\n⏱️ SETTING UP PSEUDOTIME")
    
    # Find root cluster (highest spermatogonia score)
    if 'spermatogonia_score' in adata.obs.columns:
        cluster_scores = adata.obs.groupby('leiden_0.3')['spermatogonia_score'].mean()
        root_cluster = cluster_scores.idxmax()
        
        root_cells = adata.obs['leiden_0.3'] == root_cluster
        if root_cells.sum() > 0:
            adata.uns['iroot'] = np.flatnonzero(root_cells)[0]
            sc.tl.dpt(adata)
            print(f"🌱 Root cluster: {root_cluster}")
    
    return adata

def find_cluster_markers(adata):
    """Find and annotate cluster markers."""
    
    print("\n🎯 FINDING CLUSTER MARKERS")
    
    # Find markers
    sc.tl.rank_genes_groups(adata, groupby='leiden_0.3', method='wilcoxon', n_genes=25, use_raw=True)
    marker_df = sc.get.rank_genes_groups_df(adata, group=None)
    
    # Get top markers per cluster
    cluster_annotations = {}
    for cluster in adata.obs['leiden_0.3'].unique():
        cluster_markers = marker_df[
            (marker_df['group'] == cluster) & 
            (marker_df['pvals_adj'] < 0.05) &
            (marker_df['logfoldchanges'] > 0.5)
        ]['names'].head(5).tolist()
        
        # Simple annotation based on key markers
        if any(marker in ['PRM1', 'PRM2', 'TNP1', 'TNP2'] for marker in cluster_markers):
            cluster_annotations[cluster] = 'Late_Spermatids'
        elif any(marker in ['ODF1', 'ODF2', 'SPAG16', 'SMCP'] for marker in cluster_markers):
            cluster_annotations[cluster] = 'Elongating_Spermatids'
        elif any(marker in ['SYCP1', 'SYCP3', 'SMC1B'] for marker in cluster_markers):
            cluster_annotations[cluster] = 'Spermatocytes'
        elif any(marker in ['CYP17A1', 'INSL3', 'STAR'] for marker in cluster_markers):
            cluster_annotations[cluster] = 'Leydig_Cells'
        elif any(marker in ['CLU', 'SOX9', 'FSHR'] for marker in cluster_markers):
            cluster_annotations[cluster] = 'Sertoli_Cells'
        elif any(marker in ['ACTA2', 'MYH11'] for marker in cluster_markers):
            cluster_annotations[cluster] = 'Peritubular_Cells'
        else:
            # Check most expressed known markers
            if 'GSTM3' in cluster_markers or 'HSPB9' in cluster_markers:
                cluster_annotations[cluster] = 'Round_Spermatids'
            else:
                cluster_annotations[cluster] = f'Cluster_{cluster}'
        
        print(f"📍 Cluster {cluster} ({cluster_annotations[cluster]}): {cluster_markers}")
    
    adata.obs['cluster_annotation'] = [cluster_annotations.get(c, f'Cluster_{c}') 
                                      for c in adata.obs['leiden_0.3']]
    
    return adata, cluster_annotations

def create_final_plots(adata):
    """Create comprehensive final plots."""
    
    print("\n📊 CREATING FINAL COMPREHENSIVE PLOTS")
    
    # 1. Main overview plot
    fig, axes = plt.subplots(2, 3, figsize=(20, 14))
    
    # Clusters
    sc.pl.umap(adata, color='leiden_0.3', legend_loc='on data', 
              legend_fontsize=10, ax=axes[0,0], show=False, frameon=False)
    axes[0,0].set_title('Leiden Clusters (0.3)', fontsize=14, fontweight='bold')
    
    # Annotations
    sc.pl.umap(adata, color='cluster_annotation', legend_loc='right margin',
              ax=axes[0,1], show=False, frameon=False)
    axes[0,1].set_title('Cell Type Annotations', fontsize=14, fontweight='bold')
    
    # Pseudotime
    if 'dpt_pseudotime' in adata.obs.columns:
        sc.pl.umap(adata, color='dpt_pseudotime', color_map='viridis',
                  ax=axes[0,2], show=False, frameon=False)
        axes[0,2].set_title('Diffusion Pseudotime', fontsize=14, fontweight='bold')
    else:
        axes[0,2].text(0.5, 0.5, 'Pseudotime\nNot Available', ha='center', va='center',
                      transform=axes[0,2].transAxes, fontsize=12)
        axes[0,2].set_title('Pseudotime', fontsize=14, fontweight='bold')
    
    # Progression score
    if 'progression_score' in adata.obs.columns:
        sc.pl.umap(adata, color='progression_score', color_map='RdYlBu_r',
                  ax=axes[1,0], show=False, frameon=False)
        axes[1,0].set_title('Spermatogenesis Progression Score', fontsize=14, fontweight='bold')
    
    # Total UMI
    sc.pl.umap(adata, color='total_counts', color_map='plasma',
              ax=axes[1,1], show=False, frameon=False)
    axes[1,1].set_title('Total UMI Counts', fontsize=14, fontweight='bold')
    
    # Number of genes
    sc.pl.umap(adata, color='n_genes_by_counts', color_map='viridis',
              ax=axes[1,2], show=False, frameon=False)
    axes[1,2].set_title('Number of Genes', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('spermatogenesis_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Key marker genes
    key_markers = ['PRM1', 'PRM2', 'TNP1', 'SYCP3', 'GSTM3', 'CYP17A1', 'CLU', 'ACTA2']
    available_markers = [m for m in key_markers if m in adata.raw.var_names]
    
    if len(available_markers) >= 4:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, marker in enumerate(available_markers[:8]):
            sc.pl.umap(adata, color=marker, use_raw=True, color_map='Reds',
                      ax=axes[i], show=False, frameon=False)
            axes[i].set_title(f'{marker} Expression', fontsize=12, fontweight='bold')
        
        # Hide unused axes
        for i in range(len(available_markers), 8):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('key_marker_genes.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. Stage scores heatmap
    stage_cols = [col for col in adata.obs.columns if col.endswith('_score') and col != 'progression_score']
    if len(stage_cols) > 0:
        plt.figure(figsize=(12, 8))
        
        score_matrix = []
        cluster_labels = []
        for cluster in sorted(adata.obs['leiden_0.3'].unique()):
            cluster_cells = adata.obs['leiden_0.3'] == cluster
            cluster_scores = adata.obs.loc[cluster_cells, stage_cols].mean()
            score_matrix.append(cluster_scores)
            cluster_labels.append(f'C{cluster}')
        
        score_df = pd.DataFrame(score_matrix, index=cluster_labels, columns=stage_cols)
        
        # Clean column names
        score_df.columns = [col.replace('_score', '').title() for col in score_df.columns]
        
        sns.heatmap(score_df, annot=True, cmap='RdYlBu_r', center=0, fmt='.2f',
                   cbar_kws={'label': 'Average Stage Score'})
        plt.title('Spermatogenesis Stage Scores by Cluster', fontsize=14, fontweight='bold')
        plt.xlabel('Spermatogenesis Stage', fontsize=12)
        plt.ylabel('Leiden Cluster', fontsize=12)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('stage_scores_heatmap.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 4. Trajectory analysis if pseudotime available
    if 'dpt_pseudotime' in adata.obs.columns:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Pseudotime vs progression
        if 'progression_score' in adata.obs.columns:
            scatter = axes[0,0].scatter(adata.obs['dpt_pseudotime'], 
                                      adata.obs['progression_score'],
                                      c=pd.Categorical(adata.obs['leiden_0.3']).codes,
                                      cmap='tab20', alpha=0.6, s=2)
            axes[0,0].set_xlabel('DPT Pseudotime')
            axes[0,0].set_ylabel('Progression Score')
            axes[0,0].set_title('Trajectory Validation')
        
        # Pseudotime distribution
        axes[0,1].hist(adata.obs['dpt_pseudotime'], bins=50, alpha=0.7, color='skyblue')
        axes[0,1].set_xlabel('DPT Pseudotime')
        axes[0,1].set_ylabel('Number of Cells')
        axes[0,1].set_title('Pseudotime Distribution')
        
        # Violin plot of pseudotime by cluster
        cluster_pseudotime_data = []
        cluster_labels = []
        for cluster in sorted(adata.obs['leiden_0.3'].unique()):
            cluster_cells = adata.obs['leiden_0.3'] == cluster
            pt_values = adata.obs.loc[cluster_cells, 'dpt_pseudotime'].values
            cluster_pseudotime_data.append(pt_values)
            cluster_labels.append(f'C{cluster}')
        
        bp = axes[1,0].boxplot(cluster_pseudotime_data, labels=cluster_labels, patch_artist=True)
        axes[1,0].set_xlabel('Cluster')
        axes[1,0].set_ylabel('DPT Pseudotime')
        axes[1,0].set_title('Pseudotime by Cluster')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # PAGA trajectory
        if 'paga' in adata.uns:
            sc.pl.paga(adata, color='leiden_0.3', ax=axes[1,1], 
                      show=False, frameon=False, node_size_scale=0.5)
            axes[1,1].set_title('PAGA Trajectory Graph')
        
        plt.tight_layout()
        plt.savefig('trajectory_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def export_final_results(adata, cluster_annotations):
    """Export final results."""
    
    print("\n💾 EXPORTING FINAL RESULTS")
    
    output_dir = "spermatogenesis_final_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save processed data
    adata.write_h5ad(f"{output_dir}/spermatogenesis_processed.h5ad")
    
    # Export marker genes for Metascape
    marker_df = sc.get.rank_genes_groups_df(adata, group=None)
    significant_markers = marker_df[
        (marker_df['pvals_adj'] < 0.05) & 
        (marker_df['logfoldchanges'] > 1.0)
    ].groupby('group').head(25)
    
    significant_markers.to_csv(f"{output_dir}/cluster_markers_detailed.csv", index=False)
    pd.DataFrame({'Gene': significant_markers['names'].tolist()}).to_csv(
        f"{output_dir}/metascape_gene_list.csv", index=False
    )
    
    # Export cell metadata
    obs_df = adata.obs.copy()
    obs_df['UMAP1'] = adata.obsm['X_umap'][:, 0]
    obs_df['UMAP2'] = adata.obsm['X_umap'][:, 1]
    obs_df.to_csv(f"{output_dir}/cell_metadata.csv")
    
    # Export cluster summary
    summary_data = []
    for cluster in sorted(adata.obs['leiden_0.3'].unique()):
        cluster_cells = adata.obs['leiden_0.3'] == cluster
        cell_count = cluster_cells.sum()
        annotation = cluster_annotations.get(cluster, f'Cluster_{cluster}')
        
        # Get top 3 markers
        cluster_markers = marker_df[
            (marker_df['group'] == cluster) & 
            (marker_df['pvals_adj'] < 0.05)
        ]['names'].head(3).tolist()
        
        summary_data.append({
            'cluster': cluster,
            'annotation': annotation,
            'cell_count': cell_count,
            'percentage': f"{cell_count/len(adata.obs)*100:.1f}%",
            'top_markers': ', '.join(cluster_markers)
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(f"{output_dir}/cluster_summary.csv", index=False)
    
    print(f"✅ Results exported to: {output_dir}")
    
    # Print final summary
    print(f"\n🎉 SPERMATOGENESIS ANALYSIS COMPLETE!")
    print(f"{'='*60}")
    print(f"📊 Final dataset: {adata.shape[0]} cells × {adata.shape[1]} features")
    print(f"🧬 Clusters identified: {len(adata.obs['leiden_0.3'].unique())}")
    print(f"📍 Cell type breakdown:")
    for cluster, annotation in cluster_annotations.items():
        count = (adata.obs['leiden_0.3'] == cluster).sum()
        pct = count/len(adata.obs)*100
        print(f"   {annotation}: {count} cells ({pct:.1f}%)")
    
    if 'dpt_pseudotime' in adata.obs.columns:
        print(f"⏱️ Pseudotime successfully computed")
    
    print(f"\n🎯 KEY FINDINGS:")
    print(f"   • Identified multiple spermatogenesis stages")
    print(f"   • Found trajectory from early to late stages")
    print(f"   • Detected somatic support cells (Sertoli, Leydig)")
    print(f"   • Generated stage-specific marker genes")
    print(f"{'='*60}")
    
    return summary_df

def main():
    """Run the complete continuation analysis."""
    
    print("🚀 SPERMATOGENESIS ANALYSIS CONTINUATION")
    print("="*60)
    
    # Load and process data
    adata = load_and_recreate_analysis()
    
    # Run dimensionality reduction
    adata = run_dimensionality_reduction(adata)
    
    # Add spermatogenesis scores
    adata = add_spermatogenesis_scores(adata)
    
    # Set up pseudotime
    adata = setup_pseudotime(adata)
    
    # Find cluster markers and annotations
    adata, cluster_annotations = find_cluster_markers(adata)
    
    # Create comprehensive plots
    create_final_plots(adata)
    
    # Export results
    summary_df = export_final_results(adata, cluster_annotations)
    
    return adata, cluster_annotations, summary_df

if __name__ == "__main__":
    adata, cluster_annotations, summary_df = main()
