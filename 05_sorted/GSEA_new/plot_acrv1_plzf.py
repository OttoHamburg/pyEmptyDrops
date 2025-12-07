#!/usr/bin/env python3
"""
Plot ACRV1 and PLZF markers on UMAP - matching original key_marker_genes.png style
"""

import scanpy as sc
import matplotlib.pyplot as plt
import numpy as np

# Load data
adata = sc.read_h5ad("spermatogenesis_final_results/spermatogenesis_processed.h5ad")

# Check if genes are available
genes_to_plot = ['ACRV1', 'PLZF']
available_genes = []
missing_genes = []

for gene in genes_to_plot:
    if gene in adata.var_names:
        available_genes.append(gene)
    else:
        missing_genes.append(gene)

print(f"Available genes: {available_genes}")
print(f"Missing genes: {missing_genes}")

# Check for alternative gene names if missing
if 'PLZF' not in adata.var_names:
    # PLZF is also known as ZBTB16
    if 'ZBTB16' in adata.var_names:
        print("Found PLZF as ZBTB16")
        genes_to_plot = ['ACRV1', 'ZBTB16'] if 'ACRV1' in adata.var_names else ['ZBTB16']
        available_genes = [g for g in genes_to_plot if g in adata.var_names]

# Create the plot matching original style
if len(available_genes) > 0:
    # Set figure size to match original proportions (approximately 20x10 inches for 2 genes)
    fig, axes = plt.subplots(1, len(available_genes), figsize=(10*len(available_genes), 10))
    
    # If only one gene, make axes a list for consistency
    if len(available_genes) == 1:
        axes = [axes]
    
    # Plot each available gene
    for i, gene in enumerate(available_genes):
        sc.pl.umap(adata, color=gene, 
                  ax=axes[i], 
                  show=False,
                  color_map='Reds',  # Standard scanpy color scheme
                  title=f'{gene} Expression',
                  vmin=0,  # Start colormap at 0
                  frameon=False)
        
        # Style the plot
        axes[i].set_title(f'{gene} Expression', fontsize=16, fontweight='bold', pad=20)
    
    # Add overall title
    gene_list_str = ' and '.join(available_genes)
    fig.suptitle(f'Key Spermatogenesis Markers: {gene_list_str}', 
                 fontsize=20, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.savefig('key_marker_genes2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Created key_marker_genes2.png with {len(available_genes)} genes")
    
else:
    print("❌ No genes available for plotting")
    
    # Show available genes that might be related
    related_genes = []
    for gene in adata.var_names:
        if any(search_term in gene.upper() for search_term in ['ACRV', 'PLZF', 'ZBTB']):
            related_genes.append(gene)
    
    if related_genes:
        print(f"Related genes found: {related_genes[:10]}")  # Show first 10
