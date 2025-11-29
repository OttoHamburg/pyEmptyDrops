#!/usr/bin/env python3
"""
UMAP Difference Diagnostic Script

This script helps identify why UMAPs look different between analyses
by comparing key preprocessing and parameter choices.
"""

import scanpy as sc
import pandas as pd
import numpy as np

def diagnose_umap_differences(adata1, adata2, name1="Analysis 1", name2="Analysis 2"):
    """
    Compare two AnnData objects to identify sources of UMAP differences.
    """
    
    print("🔍 UMAP DIFFERENCE DIAGNOSTIC")
    print("=" * 60)
    
    # 1. Basic data comparison
    print(f"\n📊 DATA SHAPES:")
    print(f"  {name1}: {adata1.shape}")
    print(f"  {name2}: {adata2.shape}")
    
    # 2. Gene filtering differences
    if adata1.shape != adata2.shape:
        print(f"\n⚠️  DIFFERENT DATA SHAPES - This is a major source of differences!")
        
        # Check which genes are different
        genes1 = set(adata1.var_names)
        genes2 = set(adata2.var_names)
        
        only_in_1 = genes1 - genes2
        only_in_2 = genes2 - genes1
        
        if only_in_1:
            print(f"  Genes only in {name1}: {len(only_in_1)} genes")
            if len(only_in_1) < 10:
                print(f"    {list(only_in_1)[:10]}")
        
        if only_in_2:
            print(f"  Genes only in {name2}: {len(only_in_2)} genes")
            if len(only_in_2) < 10:
                print(f"    {list(only_in_2)[:10]}")
    
    # 3. Preprocessing differences
    print(f"\n🧬 PREPROCESSING STATUS:")
    
    # Check normalization
    print(f"  Log normalization:")
    print(f"    {name1}: {'✅' if 'log1p' in adata1.uns else '❌'}")
    print(f"    {name2}: {'✅' if 'log1p' in adata2.uns else '❌'}")
    
    # Check scaling
    print(f"  Scaling:")
    print(f"    {name1}: {'✅' if 'scale' in adata1.uns else '❌'}")
    print(f"    {name2}: {'✅' if 'scale' in adata2.uns else '❌'}")
    
    # Check HVG selection
    if 'highly_variable' in adata1.var.columns:
        hvg1 = adata1.var['highly_variable'].sum()
        print(f"  HVGs in {name1}: {hvg1}")
    else:
        print(f"  HVGs in {name1}: Not computed")
        
    if 'highly_variable' in adata2.var.columns:
        hvg2 = adata2.var['highly_variable'].sum()
        print(f"  HVGs in {name2}: {hvg2}")
    else:
        print(f"  HVGs in {name2}: Not computed")
    
    # 4. PCA differences
    print(f"\n🧮 PCA STATUS:")
    if 'X_pca' in adata1.obsm:
        print(f"  {name1} PCA shape: {adata1.obsm['X_pca'].shape}")
    else:
        print(f"  {name1}: No PCA computed")
        
    if 'X_pca' in adata2.obsm:
        print(f"  {name2} PCA shape: {adata2.obsm['X_pca'].shape}")
    else:
        print(f"  {name2}: No PCA computed")
    
    # 5. Neighbors parameters
    print(f"\n🕸️ NEIGHBORS PARAMETERS:")
    
    for i, (adata, name) in enumerate([(adata1, name1), (adata2, name2)]):
        if 'neighbors' in adata.uns:
            neighbors_params = adata.uns['neighbors']['params']
            print(f"  {name}:")
            print(f"    n_neighbors: {neighbors_params.get('n_neighbors', 'Unknown')}")
            print(f"    n_pcs: {neighbors_params.get('n_pcs', 'Unknown')}")
            print(f"    metric: {neighbors_params.get('metric', 'Unknown')}")
            print(f"    use_rep: {neighbors_params.get('use_rep', 'X_pca')}")
        else:
            print(f"  {name}: No neighbors computed")
    
    # 6. UMAP parameters
    print(f"\n🗺️ UMAP PARAMETERS:")
    
    for i, (adata, name) in enumerate([(adata1, name1), (adata2, name2)]):
        if 'umap' in adata.uns:
            umap_params = adata.uns['umap']['params']
            print(f"  {name}:")
            print(f"    min_dist: {umap_params.get('min_dist', 'Unknown')}")
            print(f"    spread: {umap_params.get('spread', 'Unknown')}")
            print(f"    n_epochs: {umap_params.get('n_epochs', 'Unknown')}")
            print(f"    init_pos: {umap_params.get('init_pos', 'Unknown')}")
            print(f"    random_state: {umap_params.get('random_state', 'Unknown')}")
        else:
            print(f"  {name}: No UMAP computed")
    
    # 7. Random state differences
    print(f"\n🎲 POTENTIAL RANDOM STATE ISSUES:")
    print("  Different random states can cause different UMAPs even with same parameters")
    print("  Check if both analyses used the same random_state values")
    
    # 8. Data version differences
    print(f"\n📁 DATA SOURCE COMPARISON:")
    print("  Are you using the exact same input data?")
    print("  - Same h5ad file?")
    print("  - Same preprocessing steps applied in same order?")
    print("  - Same gene filtering criteria?")
    
    return True

def quick_umap_comparison():
    """
    Quick function to load and compare UMAPs from different sources
    """
    
    print("🔍 QUICK UMAP COMPARISON")
    print("=" * 40)
    
    # Try to load data from different sources
    possible_files = [
        "../data/emptydrops_all_detected_cells.h5ad",
        "spermatogenesis_final_results/spermatogenesis_processed.h5ad",
        "improved_spermatogenesis_trajectory.h5ad"
    ]
    
    loaded_data = {}
    
    for file_path in possible_files:
        try:
            adata = sc.read_h5ad(file_path)
            loaded_data[file_path] = adata
            print(f"✅ Loaded: {file_path} - Shape: {adata.shape}")
        except:
            print(f"❌ Could not load: {file_path}")
    
    if len(loaded_data) >= 2:
        files = list(loaded_data.keys())
        adata1, adata2 = loaded_data[files[0]], loaded_data[files[1]]
        diagnose_umap_differences(adata1, adata2, files[0], files[1])
    
    return loaded_data

if __name__ == "__main__":
    # Run quick comparison
    loaded_data = quick_umap_comparison()
