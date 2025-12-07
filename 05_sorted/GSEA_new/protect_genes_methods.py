#!/usr/bin/env python3
"""
Methods to protect specific genes during highly variable gene selection
"""

import scanpy as sc
import pandas as pd
import numpy as np

def method_1_manual_addition(adata, must_keep_genes, n_top_genes=2000):
    """
    Method 1: Manual addition after HVG selection
    This is the most straightforward approach.
    """
    print("🔧 METHOD 1: Manual Addition After HVG Selection")
    
    # Step 1: Find highly variable genes normally
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, inplace=False)
    
    # Step 2: Mark must-keep genes as highly variable
    for gene in must_keep_genes:
        if gene in adata.var_names:
            adata.var.loc[gene, 'highly_variable'] = True
            print(f"   ✅ Protected: {gene}")
        else:
            print(f"   ❌ Not found: {gene}")
    
    # Step 3: Count final HVGs
    n_hvg = adata.var['highly_variable'].sum()
    print(f"   📊 Total HVGs: {n_hvg} (including {len(must_keep_genes)} protected)")
    
    return adata

def method_2_iterative_selection(adata, must_keep_genes, n_top_genes=2000):
    """
    Method 2: Iterative selection (exclude must-keep from HVG, then add back)
    More controlled approach.
    """
    print("🔧 METHOD 2: Iterative Selection")
    
    # Step 1: Temporarily remove must-keep genes
    temp_mask = ~adata.var_names.isin(must_keep_genes)
    adata_temp = adata[:, temp_mask].copy()
    
    # Step 2: Find HVGs in remaining genes
    n_remaining = n_top_genes - len(must_keep_genes)
    sc.pp.highly_variable_genes(adata_temp, n_top_genes=n_remaining, inplace=True)
    
    # Step 3: Create full HVG mask
    adata.var['highly_variable'] = False
    
    # Add back HVGs from temp selection
    hvg_genes_temp = adata_temp.var_names[adata_temp.var['highly_variable']]
    adata.var.loc[hvg_genes_temp, 'highly_variable'] = True
    
    # Add must-keep genes
    for gene in must_keep_genes:
        if gene in adata.var_names:
            adata.var.loc[gene, 'highly_variable'] = True
            print(f"   ✅ Protected: {gene}")
    
    n_hvg = adata.var['highly_variable'].sum()
    print(f"   📊 Total HVGs: {n_hvg}")
    
    return adata

def method_3_custom_hvg_function(adata, must_keep_genes, n_top_genes=2000):
    """
    Method 3: Custom HVG function with protection built-in
    Most flexible approach.
    """
    print("🔧 METHOD 3: Custom HVG Function")
    
    # Calculate HVG metrics without subsetting
    sc.pp.highly_variable_genes(adata, n_top_genes=None, inplace=True)
    
    # Get dispersion/variance scores
    if 'dispersions_norm' in adata.var.columns:
        score_col = 'dispersions_norm'
    elif 'variances_norm' in adata.var.columns:
        score_col = 'variances_norm'
    else:
        raise ValueError("No suitable variability metric found")
    
    # Sort by variability score
    adata.var['hvg_rank'] = adata.var[score_col].rank(ascending=False)
    
    # Initialize HVG selection
    adata.var['highly_variable'] = False
    
    # First, mark must-keep genes
    protected_count = 0
    for gene in must_keep_genes:
        if gene in adata.var_names:
            adata.var.loc[gene, 'highly_variable'] = True
            protected_count += 1
            print(f"   ✅ Protected: {gene} (rank: {adata.var.loc[gene, 'hvg_rank']:.0f})")
    
    # Then select top remaining genes
    remaining_spots = n_top_genes - protected_count
    
    # Get genes not already selected, sorted by rank
    unselected = adata.var[~adata.var['highly_variable']].sort_values('hvg_rank')
    top_remaining = unselected.head(remaining_spots).index
    
    adata.var.loc[top_remaining, 'highly_variable'] = True
    
    n_hvg = adata.var['highly_variable'].sum()
    print(f"   📊 Total HVGs: {n_hvg} ({protected_count} protected + {remaining_spots} by variability)")
    
    return adata

def demonstrate_protection_methods():
    """
    Demonstrate how to protect ZBTB16 specifically
    """
    print("🧬 DEMONSTRATION: Protecting ZBTB16 During HVG Selection")
    print("=" * 60)
    
    # Load original emptydrops data
    print("📊 Loading emptydrops data...")
    adata = sc.read_h5ad('../data/emptydrops_all_detected_cells.h5ad')
    print(f"   Shape: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Verify ZBTB16 is present
    if 'ZBTB16' not in adata.var_names:
        print("❌ ZBTB16 not found in data!")
        return
    
    print("✅ ZBTB16 found in original data")
    
    # Define genes to protect
    must_keep_genes = ['ZBTB16', 'PRM1', 'TNP1', 'ACRV1', 'SMC1B', 'DDX4']
    available_genes = [g for g in must_keep_genes if g in adata.var_names]
    
    print(f"\n🎯 Genes to protect: {available_genes}")
    
    # Test Method 1
    print(f"\n" + "="*60)
    adata_test1 = adata.copy()
    adata_test1 = method_1_manual_addition(adata_test1, available_genes, n_top_genes=3000)
    
    # Check results
    zbtb16_protected = adata_test1.var.loc['ZBTB16', 'highly_variable']
    print(f"   🎯 ZBTB16 in final HVG list: {zbtb16_protected}")
    
    return adata_test1

if __name__ == "__main__":
    # Run demonstration
    result = demonstrate_protection_methods()
    
    print(f"\n💡 RECOMMENDED APPROACH:")
    print(f"   Use Method 1 (manual addition) - simplest and most reliable")
    print(f"   Add this code after your HVG selection:")
    print(f"   ")
    print(f"   # Protect important genes")
    print(f"   must_keep = ['ZBTB16', 'PRM1', 'TNP1', 'ACRV1']")
    print(f"   for gene in must_keep:")
    print(f"       if gene in adata.var_names:")
    print(f"           adata.var.loc[gene, 'highly_variable'] = True")


