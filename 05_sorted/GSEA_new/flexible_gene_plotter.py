#!/usr/bin/env python3
"""
Flexible Gene Expression Plotter for GSEA Analysis
Usage: python3 flexible_gene_plotter.py GENE1 GENE2 GENE3 [--output filename.png]
"""

import argparse
import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_genes_on_umap(genes, data_file="spermatogenesis_final_results/spermatogenesis_processed.h5ad", 
                      output_file=None, color_map='Reds', vmax=6):
    """
    Plot specified genes on UMAP using the processed spermatogenesis data.
    
    Parameters:
    - genes: list of gene names to plot
    - data_file: path to h5ad file
    - output_file: output filename (auto-generated if None)
    - color_map: matplotlib colormap name
    - vmax: maximum value for color scale (default: 6)
    """
    
    print(f"🔬 Loading data from {data_file}...")
    try:
        adata = sc.read_h5ad(data_file)
        print(f"📊 Loaded {adata.shape[0]} cells × {adata.shape[1]} features")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Check gene availability
    available_genes = []
    missing_genes = []
    alternative_genes = {}
    
    for gene in genes:
        if gene in adata.var_names:
            available_genes.append(gene)
        else:
            missing_genes.append(gene)
            
            # Look for alternative names (case-insensitive partial matching)
            alternatives = [g for g in adata.var_names 
                          if gene.upper() in g.upper() or g.upper() in gene.upper()]
            if alternatives:
                alternative_genes[gene] = alternatives[:3]  # Show max 3 alternatives
    
    # Report findings
    print(f"\n📋 Gene Analysis:")
    print(f"✅ Available: {available_genes}")
    if missing_genes:
        print(f"❌ Missing: {missing_genes}")
    if alternative_genes:
        print(f"🔍 Suggested alternatives:")
        for original, alternatives in alternative_genes.items():
            print(f"   {original} → {alternatives}")
    
    if not available_genes:
        print("\n❌ No genes available for plotting!")
        print("\n💡 Try searching for genes in the dataset:")
        
        # Show some example genes
        example_genes = list(adata.var_names[:20])
        print(f"Example available genes: {example_genes}")
        return
    
    # Set up plot dimensions
    n_genes = len(available_genes)
    
    if n_genes <= 2:
        ncols = n_genes
        nrows = 1
        figsize = (10 * ncols, 10)
    elif n_genes <= 4:
        ncols = 2
        nrows = 2
        figsize = (20, 20)
    elif n_genes <= 6:
        ncols = 3
        nrows = 2
        figsize = (30, 20)
    else:
        ncols = 4
        nrows = (n_genes + 3) // 4  # Ceiling division
        figsize = (40, 10 * nrows)
    
    print(f"\n📊 Creating plot with {nrows}×{ncols} layout...")
    print(f"🎨 Using color scale: 0 to {vmax} (consistent across all genes)")
    
    # Create the plot
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    
    # Handle single gene case
    if n_genes == 1:
        axes = [axes]
    elif nrows == 1:
        # For single row, axes might not be a 2D array
        if ncols == 1:
            axes = [axes]
    else:
        # Flatten axes for easy indexing
        axes = axes.flatten()
    
    # Plot each gene
    for i, gene in enumerate(available_genes):
        print(f"   📈 Plotting {gene}...")
        
        sc.pl.umap(adata, color=gene,
                  ax=axes[i],
                  show=False,
                  color_map=color_map,
                  title=f'{gene} Expression',
                  vmin=0,
                  vmax=vmax,
                  frameon=False)
        
        # Style the subplot
        axes[i].set_title(f'{gene} Expression', fontsize=16, fontweight='bold', pad=20)
    
    # Hide empty subplots
    if n_genes < len(axes):
        for i in range(n_genes, len(axes)):
            axes[i].set_visible(False)
    
    # Add overall title
    gene_list_str = ', '.join(available_genes)
    fig.suptitle(f'Gene Expression Analysis: {gene_list_str}', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    # Generate output filename if not provided
    if output_file is None:
        gene_suffix = '_'.join(available_genes[:3])  # Use first 3 genes for filename
        if len(available_genes) > 3:
            gene_suffix += f'_plus{len(available_genes)-3}more'
        output_file = f'gene_expression_{gene_suffix}.png'
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✅ Saved plot as: {output_file}")
    
    # Show plot
    plt.show()
    
    return output_file

def search_genes(search_terms, data_file="spermatogenesis_final_results/spermatogenesis_processed.h5ad"):
    """Search for genes matching given terms."""
    
    print(f"🔍 Searching for genes in {data_file}...")
    try:
        adata = sc.read_h5ad(data_file)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print(f"\n🔍 Search Results:")
    for term in search_terms:
        matches = [gene for gene in adata.var_names 
                  if term.upper() in gene.upper()]
        print(f"\n'{term}' matches:")
        if matches:
            for match in matches[:10]:  # Show max 10 matches
                print(f"   {match}")
            if len(matches) > 10:
                print(f"   ... and {len(matches)-10} more")
        else:
            print("   No matches found")

def main():
    parser = argparse.ArgumentParser(
        description="Plot gene expression on UMAP from spermatogenesis analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 flexible_gene_plotter.py PRM1 TNP1 ACRV1
  python3 flexible_gene_plotter.py SYCP3 SMC1B --output meiotic_genes.png
  python3 flexible_gene_plotter.py --search PRM TNP ACRO
  python3 flexible_gene_plotter.py --colormap viridis --vmax 7 PRM1 TNP1
        """
    )
    
    parser.add_argument('genes', nargs='*', 
                      help='Gene names to plot (e.g., PRM1 TNP1 ACRV1)')
    parser.add_argument('--output', '-o', 
                      help='Output filename (auto-generated if not specified)')
    parser.add_argument('--colormap', '-c', default='Reds',
                      help='Matplotlib colormap (default: Reds)')
    parser.add_argument('--data', '-d', 
                      default="spermatogenesis_final_results/spermatogenesis_processed.h5ad",
                      help='Path to h5ad data file')
    parser.add_argument('--search', '-s', action='store_true',
                      help='Search for genes instead of plotting')
    parser.add_argument('--vmax', '-v', type=float, default=6.0,
                      help='Maximum value for color scale (default: 6.0)')
    
    args = parser.parse_args()
    
    if not args.genes:
        print("❌ Please provide gene names to plot or search")
        print("\n💡 Examples:")
        print("   python3 flexible_gene_plotter.py PRM1 TNP1")
        print("   python3 flexible_gene_plotter.py --search PRM")
        parser.print_help()
        return
    
    if args.search:
        search_genes(args.genes, args.data)
    else:
        plot_genes_on_umap(args.genes, args.data, args.output, args.colormap, args.vmax)

if __name__ == "__main__":
    main()
