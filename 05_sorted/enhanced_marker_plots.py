#!/usr/bin/env python3
"""
Enhanced Marker Gene Visualization

Creates detailed kernel density plots and comparison matrices for genes showing
strong agreement between EmptyDrops and CellRanger methods.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def create_enhanced_marker_plots():
    """Create enhanced plots for top marker genes."""
    
    print("🧬 Creating Enhanced Marker Gene Plots")
    print("=" * 50)
    
    # Load data
    print("Loading data...")
    raw_adata = sc.read_10x_h5("data/raw_feature_bc_matrix.h5")
    filtered_adata = sc.read_10x_h5("data/filtered_feature_bc_matrix.h5")
    emptydrops_results = pd.read_csv("empty_drops_visualizations/empty_drops_results.csv", index_col=0)
    
    # Load analysis results
    summary_table = pd.read_csv("marker_gene_analysis/tables/marker_gene_summary_table.csv")
    
    print(f"Raw data: {raw_adata.shape}")
    print(f"Filtered data: {filtered_adata.shape}")
    print(f"EmptyDrops results: {emptydrops_results.shape}")
    
    # Categorize cells
    print("Categorizing cells...")
    cellranger_cells = set(filtered_adata.obs_names)
    emptydrops_cells = set(emptydrops_results[emptydrops_results['FDR'] < 0.05].index)
    
    cell_categories = pd.Series(index=raw_adata.obs_names, dtype='str')
    for barcode in raw_adata.obs_names:
        if barcode in cellranger_cells:
            if barcode in emptydrops_cells:
                cell_categories[barcode] = 'Called by Both Methods'
            else:
                cell_categories[barcode] = 'CellRanger Only'
        else:
            if barcode in emptydrops_cells:
                cell_categories[barcode] = 'EmptyDrops Only'
            else:
                cell_categories[barcode] = 'Called by Neither'
    
    category_counts = cell_categories.value_counts()
    print("Cell categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count:,}")
    
    # Define fixed gene positions for consistent plotting
    # Always include target genes SYCP3 and ACRV1 regardless of ranking
    priority_genes = ['PRM1', 'TNP1', 'SYCP3', 'ACRV1']  # Must-include genes
    
    # Get top genes from summary table (excluding priority genes to avoid duplicates)
    top_from_table = []
    for _, row in summary_table.iterrows():
        gene = row['Gene']
        if gene not in priority_genes and len(top_from_table) < 12:  # Get 12 more to make 16 total
            top_from_table.append(gene)
    
    # Combine in fixed order: priority genes first, then top performers
    top_agreement_genes = priority_genes + top_from_table[:12]
    
    # Filter genes that exist in the data
    available_genes = [gene for gene in top_agreement_genes if gene in raw_adata.var_names]
    print(f"\nAnalyzing {len(available_genes)} genes: {available_genes}")
    
    # Create enhanced kernel density matrix
    create_kernel_density_matrix(raw_adata, cell_categories, available_genes)
    
    # Create detailed comparison plots
    create_detailed_comparison_plots(raw_adata, cell_categories, available_genes[:8])
    
    # Create agreement analysis plot
    create_agreement_analysis_plot(summary_table)
    
    print("✅ Enhanced marker plots completed!")

def create_kernel_density_matrix(raw_adata, cell_categories, genes, figsize=(24, 20)):
    """Create an enhanced kernel density matrix."""
    
    print(f"Creating enhanced kernel density matrix for {len(genes)} genes...")
    
    n_genes = len(genes)
    n_cols = 4
    n_rows = int(np.ceil(n_genes / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Define colors and order for consistency
    colors = {
        'Called by Both Methods': '#2E8B57',      # Sea Green
        'EmptyDrops Only': '#4169E1',             # Royal Blue  
        'CellRanger Only': '#FF8C00',             # Dark Orange
        'Called by Neither': '#DC143C'            # Crimson
    }
    
    category_order = ['Called by Both Methods', 'EmptyDrops Only', 'CellRanger Only', 'Called by Neither']
    
    # Get total UMI counts once
    total_counts = np.array(raw_adata.X.sum(axis=1)).flatten()
    
    for i, gene in enumerate(genes):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        try:
            plot_gene_kde_enhanced(raw_adata, cell_categories, gene, total_counts, ax, colors, category_order)
        except Exception as e:
            print(f"  Warning: Could not plot {gene}: {e}")
            ax.text(0.5, 0.5, f'Error plotting\n{gene}', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gene, fontsize=14, weight='bold')
    
    # Hide empty subplots
    for i in range(n_genes, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # Add overall title with gene count and note about target genes
    n_priority = len([g for g in genes if g in ['PRM1', 'TNP1', 'SYCP3', 'ACRV1']])
    n_top = len(genes) - n_priority
    
    title = f'Marker Gene Expression by Cell Calling Method\n'
    title += f'Top {n_top} Genes + {n_priority} Target Genes (PRM1, TNP1, SYCP3, ACRV1)\n'
    title += f'Note: SYCP3 and ACRV1 included for validation regardless of ranking'
    
    fig.suptitle(title, fontsize=18, weight='bold', y=0.96)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save plot
    output_dir = "marker_gene_analysis/plots"
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'enhanced_marker_gene_kde_matrix.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
    plt.close()
    
    print(f"  Saved enhanced KDE matrix to: {plot_path}")
    return plot_path

def plot_gene_kde_enhanced(raw_adata, cell_categories, gene, total_counts, ax, colors, category_order):
    """Plot enhanced kernel density for a single gene."""
    
    if gene not in raw_adata.var_names:
        ax.text(0.5, 0.5, f'Gene {gene}\nnot found', 
               ha='center', va='center', transform=ax.transAxes, fontsize=12)
        ax.set_title(gene, fontsize=14, weight='bold')
        return
    
    gene_idx = list(raw_adata.var_names).index(gene)
    
    # Plot for each category in order
    legend_entries = []
    max_density = 0
    
    for category in category_order:
        if category not in cell_categories.values:
            continue
            
        category_mask = cell_categories == category
        category_cells = cell_categories[category_mask].index
        
        if len(category_cells) == 0:
            continue
        
        # Get gene expression
        gene_counts = np.array(raw_adata[category_cells, gene_idx].X.todense()).flatten()
        cat_total_counts = total_counts[category_mask]
        
        # Calculate normalized expression
        normalized_expr = gene_counts / (cat_total_counts + 1e-10)
        
        # Only plot cells with non-zero expression
        nonzero_expr = normalized_expr[normalized_expr > 0]
        
        if len(nonzero_expr) > 10:  # Need enough points for KDE
            try:
                # Create KDE plot
                sns.kdeplot(
                    data=nonzero_expr,
                    color=colors[category],
                    log_scale=True,
                    linewidth=3,
                    alpha=0.8,
                    ax=ax
                )
                
                # Track max density for consistent y-axis
                y_data = ax.lines[-1].get_ydata()
                max_density = max(max_density, np.max(y_data))
                
                # Calculate percentage in target range (1-6%)
                target_range = np.sum((nonzero_expr >= 0.01) & (nonzero_expr <= 0.06))
                target_pct = target_range / len(category_cells) * 100
                
                legend_entries.append(f'{category}\n(n={len(nonzero_expr)}, target={target_pct:.1f}%)')
                
            except Exception as e:
                print(f"    KDE failed for {gene} {category}: {e}")
                # Fallback to histogram
                ax.hist(nonzero_expr, bins=30, alpha=0.3, color=colors[category], 
                       density=True, label=f'{category} (n={len(nonzero_expr)})')
    
    # Formatting
    ax.set_xlabel('Normalized Expression (log scale)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(gene, fontsize=14, weight='bold', pad=10)
    ax.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Add legend with custom entries
    if legend_entries:
        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [Line2D([0], [0], color=colors[cat.split('\n')[0]], lw=3, label=entry) 
                          for cat, entry in zip(category_order, legend_entries) if entry]
        ax.legend(handles=legend_elements, fontsize=9, loc='upper right', framealpha=0.9)
    
    # Set consistent y-axis limit
    if max_density > 0:
        ax.set_ylim(0, max_density * 1.1)
    
    # Add vertical lines for target range
    ax.axvspan(0.01, 0.06, alpha=0.1, color='red', zorder=0)
    ax.text(0.02, ax.get_ylim()[1] * 0.9, 'Target\nRange', fontsize=8, color='red', 
           ha='left', va='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

def create_detailed_comparison_plots(raw_adata, cell_categories, genes):
    """Create detailed comparison plots for top genes."""
    
    print(f"Creating detailed comparison plots for {len(genes)} genes...")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = {
        'Called by Both Methods': '#2E8B57',
        'EmptyDrops Only': '#4169E1', 
        'CellRanger Only': '#FF8C00',
        'Called by Neither': '#DC143C'
    }
    
    total_counts = np.array(raw_adata.X.sum(axis=1)).flatten()
    
    for i, gene in enumerate(genes):
        if i >= 8:  # Only plot first 8
            break
            
        ax = axes[i]
        
        try:
            gene_idx = list(raw_adata.var_names).index(gene)
            
            # Create box plots showing distribution by category
            box_data = []
            labels = []
            
            for category in ['Called by Both Methods', 'EmptyDrops Only', 'CellRanger Only']:
                if category not in cell_categories.values:
                    continue
                    
                category_mask = cell_categories == category
                category_cells = cell_categories[category_mask].index
                
                if len(category_cells) == 0:
                    continue
                
                gene_counts = np.array(raw_adata[category_cells, gene_idx].X.todense()).flatten()
                cat_total_counts = total_counts[category_mask]
                normalized_expr = gene_counts / (cat_total_counts + 1e-10)
                nonzero_expr = normalized_expr[normalized_expr > 0]
                
                if len(nonzero_expr) > 5:
                    box_data.append(nonzero_expr)
                    labels.append(f'{category}\n(n={len(nonzero_expr)})')
            
            if box_data:
                bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
                
                # Color the boxes
                category_names = ['Called by Both Methods', 'EmptyDrops Only', 'CellRanger Only']
                for patch, cat in zip(bp['boxes'], category_names[:len(bp['boxes'])]):
                    patch.set_facecolor(colors[cat])
                    patch.set_alpha(0.7)
            
            ax.set_yscale('log')
            ax.set_ylabel('Normalized Expression')
            ax.set_title(gene, fontsize=12, weight='bold')
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add target range
            ax.axhspan(0.01, 0.06, alpha=0.2, color='red', zorder=0)
            
        except Exception as e:
            print(f"  Warning: Could not create box plot for {gene}: {e}")
    
    # Hide unused subplots
    for i in range(len(genes), 8):
        axes[i].set_visible(False)
    
    # Add title with gene count info
    n_priority = len([g for g in genes[:8] if g in ['PRM1', 'TNP1', 'SYCP3', 'ACRV1']])
    n_top = min(8, len(genes)) - n_priority
    
    title = f'Top {n_top} Genes + {n_priority} Target Genes: Expression Distribution\n'
    title += f'Box Plots with Target Range (SYCP3 and ACRV1 included for validation)'
    
    plt.suptitle(title, fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_path = "marker_gene_analysis/plots/detailed_gene_comparison_boxplots.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved detailed comparison plots to: {output_path}")

def create_agreement_analysis_plot(summary_table):
    """Create analysis plot showing method agreement."""
    
    print("Creating method agreement analysis plot...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Detection rate comparison
    ax1 = axes[0, 0]
    scatter = ax1.scatter(summary_table['CellRanger Detection'], 
                         summary_table['EmptyDrops Detection'],
                         c=summary_table['Target Range %'], 
                         s=100, alpha=0.7, cmap='viridis')
    ax1.plot([0, 1], [0, 1], 'r--', alpha=0.5)
    ax1.set_xlabel('CellRanger Detection Rate')
    ax1.set_ylabel('EmptyDrops Detection Rate')
    ax1.set_title('Method Agreement in Gene Detection')
    plt.colorbar(scatter, ax=ax1, label='Target Range %')
    
    # Add gene labels for top genes
    top_genes = summary_table.head(10)
    for _, row in top_genes.iterrows():
        ax1.annotate(row['Gene'], 
                    (row['CellRanger Detection'], row['EmptyDrops Detection']),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # Plot 2: Target range vs overall detection
    ax2 = axes[0, 1]
    target_genes = summary_table[summary_table['Target Gene'] == True]
    other_genes = summary_table[summary_table['Target Gene'] == False]
    
    ax2.scatter(other_genes['Overall Detection Rate'], other_genes['Target Range %'], 
               alpha=0.6, label='Other Genes', color='lightblue')
    ax2.scatter(target_genes['Overall Detection Rate'], target_genes['Target Range %'], 
               alpha=0.8, label='Target Genes', color='red', s=100)
    
    for _, row in target_genes.iterrows():
        ax2.annotate(row['Gene'], 
                    (row['Overall Detection Rate'], row['Target Range %']),
                    xytext=(5, 5), textcoords='offset points', fontsize=9, weight='bold')
    
    ax2.set_xlabel('Overall Detection Rate')
    ax2.set_ylabel('Target Range %')
    ax2.set_title('Target Gene Performance')
    ax2.legend()
    
    # Plot 3: Method-specific detection rates
    ax3 = axes[1, 0]
    top_20 = summary_table.head(20)
    x = np.arange(len(top_20))
    width = 0.35
    
    ax3.bar(x - width/2, top_20['CellRanger Detection'], width, 
           label='CellRanger', alpha=0.8, color='orange')
    ax3.bar(x + width/2, top_20['EmptyDrops Detection'], width, 
           label='EmptyDrops', alpha=0.8, color='blue')
    
    ax3.set_xlabel('Genes (Top 20)')
    ax3.set_ylabel('Detection Rate')
    ax3.set_title('Detection Rate Comparison: Top 20 Genes')
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_20['Gene'], rotation=45, ha='right')
    ax3.legend()
    
    # Plot 4: Target range distribution
    ax4 = axes[1, 1]
    ax4.hist(summary_table['Target Range %'], bins=30, alpha=0.7, color='green')
    ax4.axvline(summary_table['Target Range %'].mean(), color='red', linestyle='--', 
               label=f'Mean: {summary_table["Target Range %"].mean():.1f}%')
    ax4.set_xlabel('Target Range %')
    ax4.set_ylabel('Number of Genes')
    ax4.set_title('Distribution of Target Range Percentages')
    ax4.legend()
    
    plt.suptitle('Method Agreement and Gene Performance Analysis', fontsize=16, weight='bold')
    plt.tight_layout()
    
    output_path = "marker_gene_analysis/plots/method_agreement_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved agreement analysis to: {output_path}")

if __name__ == "__main__":
    create_enhanced_marker_plots() 