#!/usr/bin/env python3
"""
Production-ready EmptyDrops runner script.
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

from empty_drops_v5_batched import empty_drops_v5_batched


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
        # Add results to AnnData object
        for col in results_df.columns:
            original_data.obs[col] = results_df[col].values
        
        # Store metadata in AnnData uns
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
