"""
Analyze PRM1 gene distribution across different cell classifications.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import issparse

def create_confusion_matrix(cell_categories, fdr_threshold):
    """Create and plot confusion matrix."""
    # Count occurrences of each category
    both_methods = np.sum(cell_categories == 'True Positive')  # Called by both methods
    only_emptydrops = np.sum(cell_categories == 'False Positive')  # Called only by EmptyDrops
    only_cellranger = np.sum(cell_categories == 'False Negative')  # Called only by CellRanger
    neither_method = np.sum(cell_categories == 'True Negative')  # Called by neither
    
    # Create confusion matrix with same orientation as EmptyDrops visualizer
    conf_matrix = np.array([[neither_method, only_emptydrops],   # [Not Called, Called] by EmptyDrops
                           [only_cellranger, both_methods]])      # [Not Called, Called] by CellRanger
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, 
                annot=True, 
                fmt=',d',
                cmap='Blues',
                xticklabels=['Not Called', 'Called'],
                yticklabels=['Not Called', 'Called'])
    
    plt.suptitle('Method Comparison: CellRanger vs EmptyDrops', y=1.02, fontsize=14)
    plt.title(f'EmptyDrops FDR < {fdr_threshold}', fontsize=12, pad=10)
    plt.ylabel('CellRanger')
    plt.xlabel('EmptyDrops')
    
    # Calculate agreement metrics
    total = conf_matrix.sum()
    agreement = (both_methods + neither_method) / total
    cellranger_rate = (both_methods + only_cellranger) / total
    emptydrops_rate = (both_methods + only_emptydrops) / total
    overlap_rate = both_methods / (both_methods + only_cellranger + only_emptydrops)
    
    # Add metrics as text
    plt.figtext(0.02, 0.02, 
                f'Overall Agreement: {agreement:.3f}\n'
                f'CellRanger Call Rate: {cellranger_rate:.3f}\n'
                f'EmptyDrops Call Rate: {emptydrops_rate:.3f}\n'
                f'Method Overlap: {overlap_rate:.3f}\n'
                f'Expected False Discoveries: {int(emptydrops_rate * total * fdr_threshold):,}',
                fontsize=10,
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Save plot
    plt.savefig(f'empty_drops_visualizations/method_comparison_matrix_fdr_{fdr_threshold}.png',
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()

def load_and_process_data(fdr_threshold=0.001):
    """Load matrices and process PRM1 data."""
    print("Loading matrices...")
    
    # Load matrices
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    filtered_adata = sc.read_10x_h5('filtered_feature_bc_matrix.h5')
    
    # Load EmptyDrops results
    results = pd.read_csv('empty_drops_results.csv', index_col=0)
    
    # Find PRM1 gene index
    try:
        prm1_idx = np.where(raw_adata.var_names == 'PRM1')[0][0]
    except IndexError:
        raise ValueError("PRM1 gene not found in the dataset!")
    
    print(f"Found PRM1 at index {prm1_idx}")
    
    # Get total UMI counts per cell
    total_counts = np.array(raw_adata.X.sum(axis=1)).flatten()
    
    # Get PRM1 counts
    prm1_counts = np.array(raw_adata.X[:, prm1_idx].todense()).flatten()
    
    print(f"Number of cells expressing PRM1 (>0): {np.sum(prm1_counts > 0)}")
    print(f"Percentage of cells expressing PRM1: {(np.sum(prm1_counts > 0) / len(prm1_counts) * 100):.2f}%")
    
    # Compare different FDR thresholds
    print(f"\nAnalyzing with FDR threshold < {fdr_threshold}:")
    print("-" * 80)
    cells_at_fdr = results[results['FDR'] < fdr_threshold].index
    n_cells = len(cells_at_fdr)
    prm1_expressing = np.sum(prm1_counts[results.index.isin(cells_at_fdr)] > 0)
    print(f"  • Number of cells called: {n_cells:,}")
    print(f"  • Number expressing PRM1: {prm1_expressing:,}")
    print(f"  • Percentage expressing PRM1: {(prm1_expressing/n_cells*100):.1f}%")
    print(f"  • Expected number of false positives: {int(n_cells * fdr_threshold):,}")
    print("-" * 80)
    
    # Add small constant to avoid division by zero
    normalized_prm1 = prm1_counts / (total_counts + 1e-10)
    
    # Create classification categories
    cell_categories = pd.Series(index=raw_adata.obs_names, dtype='str')
    
    # Get Cell Ranger calls (filtered matrix)
    cellranger_cells = set(filtered_adata.obs_names)
    
    # Get EmptyDrops calls using specified FDR threshold
    emptydrops_cells = set(results[results['FDR'] < fdr_threshold].index)
    
    # Classify each cell into confusion matrix quadrants
    for barcode in raw_adata.obs_names:
        if barcode in cellranger_cells:
            if barcode in emptydrops_cells:
                cell_categories[barcode] = 'True Positive'  # Both called it a cell
            else:
                cell_categories[barcode] = 'False Negative'  # Only CellRanger called it a cell
        else:
            if barcode in emptydrops_cells:
                cell_categories[barcode] = 'False Positive'  # Only EmptyDrops called it a cell
            else:
                cell_categories[barcode] = 'True Negative'  # Both called it empty
    
    return normalized_prm1, cell_categories, prm1_counts, total_counts

def plot_prm1_distribution(normalized_prm1, cell_categories, prm1_counts, total_counts, fdr_threshold):
    """Create kernel density plot for PRM1 distribution."""
    # Create figure with larger size for better visibility
    plt.figure(figsize=(15, 10))
    
    # Update category names to reflect method comparison rather than true/false terminology
    category_mapping = {
        'True Positive': 'Called by Both Methods',
        'False Positive': 'EmptyDrops Only',
        'False Negative': 'CellRanger Only',
        'True Negative': 'Called by Neither'
    }
    
    # Plot density for each category
    categories = ['True Positive', 'False Positive', 'False Negative', 'True Negative']
    colors = ['green', 'blue', 'orange', 'red']
    
    for cat, color in zip(categories, colors):
        # Create mask for current category
        mask = cell_categories == cat
        if mask.any():  # Only plot if category has data
            # Get normalized PRM1 expression for this category
            expr = normalized_prm1[mask]
            # Only include cells with non-zero expression for the density plot
            # This prevents the large spike at zero from dominating the plot
            nonzero_expr = expr[expr > 0]
            
            if len(nonzero_expr) > 0:  # Only plot if there are non-zero expressions
                # Create kernel density estimate
                # - data: the normalized PRM1 expression values
                # - log_scale=True: x-axis is logarithmic to better show the range of expressions
                # - label: shows category name and counts (total cells and expressing cells)
                sns.kdeplot(
                    data=nonzero_expr,
                    label=f'{category_mapping[cat]} (n={mask.sum()}, expr={len(nonzero_expr)})',
                    color=color,
                    log_scale=True,  # Use log scale for better visualization of wide range
                    linewidth=2
                )
    
    plt.title(f'PRM1 Gene Expression Distribution by Cell Calling Method (FDR < {fdr_threshold})\nNormalized by Total UMI Count', 
              fontsize=14, 
              pad=20)
    plt.xlabel('Normalized PRM1 Expression (log scale)', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    
    # Customize the grid for better readability
    plt.grid(True, which='both', linestyle=':', alpha=0.3)
    
    # Move legend outside plot to prevent overlap with curves
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Adjust layout to prevent text cutoff
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'empty_drops_visualizations/prm1_distribution_fdr_{fdr_threshold}.png', 
                dpi=300, 
                bbox_inches='tight',
                pad_inches=0.5)
    plt.close()
    
    # Open file to save statistics
    with open(f'empty_drops_visualizations/prm1_statistics_fdr_{fdr_threshold}.txt', 'w') as f:
        # Print and save detailed statistics
        header = f"DETAILED PRM1 EXPRESSION ANALYSIS BY CATEGORY (FDR < {fdr_threshold})"
        separator = "="*80
        
        # Function to both print and write to file
        def print_and_write(*args, **kwargs):
            print(*args, **kwargs)
            print(*args, **kwargs, file=f)
        
        print_and_write("\n" + separator)
        print_and_write(header)
        print_and_write(separator)
        
        for cat in categories:
            mask = cell_categories == cat
            if mask.any():
                cat_prm1 = prm1_counts[mask]
                cat_total = total_counts[mask]
                expr = normalized_prm1[mask]
                
                # Calculate statistics
                n_cells = mask.sum()
                n_expressing = (cat_prm1 > 0).sum()
                percent_expressing = (n_expressing/n_cells*100)
                mean_prm1 = cat_prm1.mean()
                mean_total = cat_total.mean()
                mean_norm = expr.mean()
                
                print_and_write(f"\n{category_mapping[cat].upper()}")
                print_and_write("-"*80)
                print_and_write(f"Cell Counts:")
                print_and_write(f"  • Total cells in category: {n_cells:,}")
                print_and_write(f"  • Cells expressing PRM1: {n_expressing:,} ({percent_expressing:.1f}%)")
                
                print_and_write(f"\nPRM1 Expression:")
                print_and_write(f"  • Mean PRM1 UMI counts per cell: {mean_prm1:.2f}")
                if n_expressing > 0:
                    print_and_write(f"  • Mean PRM1 UMI counts in expressing cells: {cat_prm1[cat_prm1 > 0].mean():.2f}")
                
                print_and_write(f"\nTotal UMI Counts:")
                print_and_write(f"  • Mean total UMIs per cell: {mean_total:.2f}")
                
                print_and_write(f"\nNormalized Expression (PRM1 counts / total counts):")
                print_and_write(f"  • Mean: {mean_norm:.2%}")
                print_and_write(f"  • Median: {np.median(expr):.2%}")
                if n_expressing > 0:
                    print_and_write(f"  • Range in expressing cells: {np.min(expr[expr > 0]):.2%} to {np.max(expr):.2%}")
                
                print_and_write(f"\nKey Observations:")
                if cat == "True Positive":
                    print_and_write(f"  • High PRM1 expression rate ({percent_expressing:.1f}%)")
                    print_and_write(f"  • Moderate PRM1 counts ({mean_prm1:.1f} UMIs/cell)")
                    print_and_write(f"  • High total UMI counts ({mean_total:.1f} UMIs/cell)")
                elif cat == "False Positive":
                    print_and_write(f"  • Very high PRM1 expression rate ({percent_expressing:.1f}%)")
                    print_and_write(f"  • High PRM1 counts ({mean_prm1:.1f} UMIs/cell)")
                    print_and_write(f"  • Similar total UMI counts to cells called by both methods ({mean_total:.1f} UMIs/cell)")
                elif cat == "False Negative":
                    print_and_write(f"  • High PRM1 expression rate ({percent_expressing:.1f}%)")
                    print_and_write(f"  • Lower PRM1 counts ({mean_prm1:.1f} UMIs/cell)")
                    print_and_write(f"  • Much lower total UMI counts ({mean_total:.1f} UMIs/cell)")
                else:  # True Negative
                    print_and_write(f"  • Low PRM1 expression rate ({percent_expressing:.1f}%)")
                    print_and_write(f"  • Very low PRM1 counts ({mean_prm1:.1f} UMIs/cell)")
                    print_and_write(f"  • Very low total UMI counts ({mean_total:.1f} UMIs/cell)")
        
        print_and_write("\n" + separator)
        print_and_write(f"Analysis complete! Check 'empty_drops_visualizations/prm1_distribution_fdr_{fdr_threshold}.png' for the plot.")
        print_and_write(separator + "\n")
        
        print(f"\nStatistics have been saved to 'empty_drops_visualizations/prm1_statistics_fdr_{fdr_threshold}.txt'")

def main():
    """Main function to run the analysis."""
    # Create visualization directory if it doesn't exist
    import os
    os.makedirs('empty_drops_visualizations', exist_ok=True)
    
    # Set FDR threshold
    fdr_threshold = 0.05  # 5% FDR
    
    # Load and process data
    normalized_prm1, cell_categories, prm1_counts, total_counts = load_and_process_data(fdr_threshold)
    
    # Create confusion matrix
    create_confusion_matrix(cell_categories, fdr_threshold)
    
    # Create visualization
    plot_prm1_distribution(normalized_prm1, cell_categories, prm1_counts, total_counts, fdr_threshold)

if __name__ == "__main__":
    main() 