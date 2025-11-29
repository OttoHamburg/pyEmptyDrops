#!/usr/bin/env python3
"""
Filter raw single-cell data using EmptyDrops results and save as H5 matrix.
This script removes cells that were not detected by EmptyDrops, keeping only the validated cells.
"""

import pandas as pd
import scanpy as sc
import numpy as np
import os
import argparse
from pathlib import Path

def load_emptydrops_results(results_path: str) -> pd.DataFrame:
    """Load EmptyDrops results from CSV file."""
    print(f"Loading EmptyDrops results from: {results_path}")
    results = pd.read_csv(results_path, index_col=0)
    print(f"Loaded {len(results)} total droplets")
    return results

def filter_cells_by_emptydrops(raw_adata, emptydrops_results, fdr_threshold=0.05):
    """
    Filter raw data to keep only cells detected by EmptyDrops.
    
    Parameters:
    -----------
    raw_adata : AnnData
        Raw single-cell data
    emptydrops_results : pd.DataFrame
        EmptyDrops results with FDR values
    fdr_threshold : float
        FDR threshold for cell calling
        
    Returns:
    --------
    AnnData : Filtered data containing only EmptyDrops-detected cells
    """
    print(f"Original raw data shape: {raw_adata.shape}")
    
    # Find cells detected by EmptyDrops
    if 'FDR' in emptydrops_results.columns:
        # Use FDR column
        detected_cells = emptydrops_results[
            (emptydrops_results['FDR'].notna()) & 
            (emptydrops_results['FDR'] < fdr_threshold)
        ].index
    elif 'FDR < 0.05' in emptydrops_results.columns:
        # Use boolean column
        detected_cells = emptydrops_results[
            emptydrops_results['FDR < 0.05'] == True
        ].index
    else:
        raise ValueError("No FDR column found in EmptyDrops results")
    
    print(f"EmptyDrops detected {len(detected_cells)} cells with FDR < {fdr_threshold}")
    
    # Filter raw data to keep only detected cells
    detected_cell_mask = raw_adata.obs_names.isin(detected_cells)
    filtered_adata = raw_adata[detected_cell_mask].copy()
    
    print(f"Filtered data shape: {filtered_adata.shape}")
    print(f"Kept {filtered_adata.shape[0]} cells ({filtered_adata.shape[0]/raw_adata.shape[0]*100:.2f}% of original)")
    
    return filtered_adata

def save_h5_matrix(adata, output_path: str):
    """Save AnnData object as H5 file."""
    print(f"Saving filtered matrix to: {output_path}")
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Save as H5 file
    adata.write_h5ad(output_path)
    print(f"✅ Successfully saved filtered matrix with {adata.shape[0]} cells and {adata.shape[1]} genes")

def main():
    parser = argparse.ArgumentParser(description='Filter raw matrix using EmptyDrops results')
    parser.add_argument('--raw-h5', default='data/raw_feature_bc_matrix.h5',
                       help='Path to raw H5 matrix file')
    parser.add_argument('--emptydrops-results', 
                       default='empty_drops_visualizations/empty_drops_results.csv',
                       help='Path to EmptyDrops results CSV file')
    parser.add_argument('--output', default='data/emptydrops_filtered_matrix.h5ad',
                       help='Output path for filtered H5 matrix')
    parser.add_argument('--fdr-threshold', type=float, default=0.05,
                       help='FDR threshold for cell calling (default: 0.05)')
    parser.add_argument('--use-optimized', action='store_true',
                       help='Use optimized EmptyDrops results instead of main results')
    
    args = parser.parse_args()
    
    # Use optimized results if requested
    if args.use_optimized:
        args.emptydrops_results = 'optuna_optimization/results/full_dataset_optimization_final_called_cells.csv'
        print("Using optimized EmptyDrops results")
    
    # Check if input files exist
    if not os.path.exists(args.raw_h5):
        print(f"❌ Error: Raw H5 file not found: {args.raw_h5}")
        return
    
    if not os.path.exists(args.emptydrops_results):
        print(f"❌ Error: EmptyDrops results file not found: {args.emptydrops_results}")
        return
    
    # Load raw data
    print(f"Loading raw data from: {args.raw_h5}")
    raw_adata = sc.read_10x_h5(args.raw_h5)
    raw_adata.var_names_make_unique()
    
    # Load EmptyDrops results
    emptydrops_results = load_emptydrops_results(args.emptydrops_results)
    
    # Filter cells
    filtered_adata = filter_cells_by_emptydrops(
        raw_adata, 
        emptydrops_results, 
        fdr_threshold=args.fdr_threshold
    )
    
    # Add metadata
    filtered_adata.obs['emptydrops_detected'] = True
    filtered_adata.obs['fdr_threshold'] = args.fdr_threshold
    filtered_adata.obs['total_umi'] = np.array(filtered_adata.X.sum(axis=1)).flatten()
    
    # Add info to uns
    filtered_adata.uns['filtering_info'] = {
        'method': 'EmptyDrops',
        'fdr_threshold': args.fdr_threshold,
        'original_cells': raw_adata.shape[0],
        'filtered_cells': filtered_adata.shape[0],
        'genes': filtered_adata.shape[1],
        'emptydrops_results_file': args.emptydrops_results
    }
    
    # Save filtered matrix
    save_h5_matrix(filtered_adata, args.output)
    
    # Print summary
    print("\n" + "="*60)
    print("FILTERING SUMMARY")
    print("="*60)
    print(f"Original droplets: {raw_adata.shape[0]:,}")
    print(f"EmptyDrops detected cells: {filtered_adata.shape[0]:,}")
    print(f"Genes: {filtered_adata.shape[1]:,}")
    print(f"FDR threshold: {args.fdr_threshold}")
    print(f"Retention rate: {filtered_adata.shape[0]/raw_adata.shape[0]*100:.2f}%")
    print(f"Output file: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main() 