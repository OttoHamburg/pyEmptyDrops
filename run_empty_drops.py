#!/usr/bin/env python3
"""
Production-ready EmptyDrops runner script.

This script can be used as a command-line tool or imported as a module.
It provides plotting functionality and saves results as CSV and H5AD files.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt

from empty_drops import empty_drops


def plot_barcode_ranks(
    data_csr,
    lower: int,
    retain: int,
    output_path: str,
    title: str = "Barcode Rank Plot"
):
    """
    Create a log-log barcode rank plot with threshold lines.
    
    Parameters
    ----------
    data_csr : scipy.sparse.csr_matrix
        Sparse matrix of cell counts
    lower : int
        Lower threshold (minimum UMI count to test)
    retain : int
        Calculated retain threshold (knee point)
    output_path : str
        Path to save the PNG plot
    title : str
        Plot title
    """
    # Calculate totals and ranks
    totals = np.asarray(data_csr.sum(axis=1)).flatten().astype(int)
    
    # Sort by total count (descending)
    o = np.argsort(-totals)
    ordered_totals = totals[o]
    
    # Create rank array (1-based)
    ranks = np.arange(1, len(ordered_totals) + 1)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    plt.loglog(ranks, ordered_totals, 'b.', alpha=0.6, markersize=1, label='Barcodes')
    
    # Add threshold lines
    if lower is not None and np.isfinite(lower):
        plt.axhline(y=lower, color='orange', linestyle=':', linewidth=2, 
                   label=f'Lower threshold ({lower})')
    
    if retain is not None and np.isfinite(retain):
        plt.axhline(y=retain, color='red', linestyle='--', linewidth=2, 
                   label=f'Retain threshold ({retain})')
    
    plt.xlabel('Rank (log10)', fontsize=12)
    plt.ylabel('Total UMI Counts (log10)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Barcode rank plot saved to: {output_path}")


def save_results(
    results_df: pd.DataFrame,
    metadata: dict,
    output_prefix: str,
    save_csv: bool = True,
    save_h5ad: bool = True,
    original_data: Optional[sc.AnnData] = None
):
    """
    Save EmptyDrops results as CSV and/or H5AD files.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        EmptyDrops results DataFrame
    metadata : dict
        Metadata dictionary
    output_prefix : str
        Prefix for output files (without extension)
    save_csv : bool
        Whether to save CSV file
    save_h5ad : bool
        Whether to save H5AD file
    original_data : sc.AnnData, optional
        Original AnnData object (needed for H5AD output)
    """
    if save_csv:
        csv_path = f"{output_prefix}_results.csv"
        results_df.to_csv(csv_path, index=True)
        print(f"Results saved to CSV: {csv_path}")
    
    if save_h5ad and original_data is not None:
        # Filter cells with FDR <= 0.05
        if 'FDR' in results_df.columns:
            fdr_mask = results_df['FDR'] <= 0.05
            n_cells_filtered = fdr_mask.sum()
            
            # Create filtered AnnData object
            filtered_data = original_data[fdr_mask].copy()
            
            # Add results to filtered AnnData object
            for col in results_df.columns:
                filtered_data.obs[col] = results_df.loc[fdr_mask, col].values
            
            # Store metadata in AnnData uns
            filtered_data.uns['empty_drops'] = metadata
            
            h5ad_path = f"{output_prefix}_results.h5ad"
            filtered_data.write(h5ad_path)
            print(f"Results saved to H5AD: {h5ad_path} ({n_cells_filtered:,} cells with FDR <= 0.05)")
        else:
            # If no FDR column, save all cells
            for col in results_df.columns:
                original_data.obs[col] = results_df[col].values
            
            original_data.uns['empty_drops'] = metadata
            
            h5ad_path = f"{output_prefix}_results.h5ad"
            original_data.write(h5ad_path)
            print(f"Results saved to H5AD: {h5ad_path}")
    
    # Save metadata as JSON if CSV is saved
    if save_csv:
        import json
        json_path = f"{output_prefix}_metadata.json"
        with open(json_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {json_path}")


def run_empty_drops(
    input_file: str,
    output_dir: str = ".",
    lower: int = 100,
    niters: int = 10000,
    retain: Optional[int] = None,
    max_batches: int = 100,
    plot: bool = True,
    gex_only: bool = True,
    output_prefix: Optional[str] = None
):
    """
    Run EmptyDrops analysis on a 10x H5 file.
    
    Parameters
    ----------
    input_file : str
        Path to input 10x H5 file
    output_dir : str
        Directory to save output files
    lower : int
        Lower threshold for testing
    niters : int
        Number of Monte Carlo iterations
    retain : int, optional
        Retain threshold (if None, auto-calculated)
    max_batches : int
        Maximum number of batches for optimization
    plot : bool
        Whether to create barcode rank plot
    gex_only : bool
        Whether to use only Gene Expression data
    output_prefix : str, optional
        Prefix for output files (default: input filename without extension)
    
    Returns
    -------
    tuple
        (results_df, metadata, original_data)
    """
    # Setup output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine output prefix
    if output_prefix is None:
        output_prefix = Path(input_file).stem
    
    output_prefix = output_dir / output_prefix
    
    # Load data
    print(f"Loading data from: {input_file}")
    adata = sc.read_10x_h5(input_file, gex_only=gex_only)
    adata.var_names_make_unique()
    print(f"Loaded data: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Run EmptyDrops
    results_df, metadata = empty_drops(
        adata.copy(),
        lower=lower,
        niters=niters,
        retain=retain,
        max_batches=max_batches,
        return_metadata=True
    )
    
    # Save results
    save_results(
        results_df,
        metadata,
        str(output_prefix),
        save_csv=True,
        save_h5ad=True,
        original_data=adata
    )
    
    # Create plot if requested
    if plot:
        retain_value = metadata.get('calculated_retain', retain)
        # Handle np.inf and None cases
        if retain_value is None or (isinstance(retain_value, float) and np.isinf(retain_value)):
            retain_value = None
        
        plot_path = f"{output_prefix}_barcode_ranks.png"
        plot_barcode_ranks(
            adata.X.tocsr(),
            lower=lower,
            retain=retain_value,
            output_path=str(plot_path),
            title=f"Barcode Rank Plot - {Path(input_file).stem}"
        )
    
    return results_df, metadata, adata


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Run EmptyDrops analysis on 10x Genomics H5 files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python empty_drops.py input.h5
  
  # With custom output directory
  python empty_drops.py input.h5 -o results/
  
  # With custom parameters
  python empty_drops.py input.h5 --lower 200 --niters 5000
  
  # Without plotting
  python empty_drops.py input.h5 --no-plot
  
  # Import as module
  from empty_drops import run_empty_drops
  results_df, metadata, adata = run_empty_drops('input.h5')
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input 10x H5 file'
    )
    
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        default='.',
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--output-prefix',
        type=str,
        default=None,
        help='Prefix for output files (default: input filename without extension)'
    )
    
    parser.add_argument(
        '--lower',
        type=int,
        default=100,
        help='Lower threshold for testing (default: 100)'
    )
    
    parser.add_argument(
        '--niters',
        type=int,
        default=10000,
        help='Number of Monte Carlo iterations (default: 10000)'
    )
    
    parser.add_argument(
        '--retain',
        type=int,
        default=None,
        help='Retain threshold (default: auto-calculate)'
    )
    
    parser.add_argument(
        '--max-batches',
        type=int,
        default=100,
        help='Maximum number of batches for optimization (default: 100)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable barcode rank plot generation'
    )
    
    parser.add_argument(
        '--gex-only',
        action='store_true',
        default=True,
        help='Use only Gene Expression data (default: True)'
    )
    
    parser.add_argument(
        '--all-features',
        action='store_true',
        help='Use all features (overrides --gex-only)'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)
    
    # Determine GEX-only setting
    gex_only = args.gex_only and not args.all_features
    
    # Run analysis
    try:
        results_df, metadata, adata = run_empty_drops(
            input_file=args.input_file,
            output_dir=args.output_dir,
            lower=args.lower,
            niters=args.niters,
            retain=args.retain,
            max_batches=args.max_batches,
            plot=not args.no_plot,
            gex_only=gex_only,
            output_prefix=args.output_prefix
        )
        
        print("\n✓ EmptyDrops analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Error during EmptyDrops analysis: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
