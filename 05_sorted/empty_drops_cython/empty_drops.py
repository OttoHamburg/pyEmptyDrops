"""
EmptyDrops - Cython-optimized implementation of the EmptyDrops algorithm.
"""

import numpy as np
import scanpy as sc
from scipy.sparse import issparse
from statsmodels.stats.multitest import multipletests
import warnings
from typing import Optional, Tuple, Dict
import pandas as pd
from tqdm import tqdm
import time
from datetime import timedelta

# Import Cython functions
from empty_drops_core import (
    parallel_multinomial_prob_cy,
    batch_monte_carlo_cy,
    good_turing_freqs_cy
)

def format_time(seconds: float) -> str:
    """Format time in seconds to a human-readable string."""
    return str(timedelta(seconds=int(seconds)))

def empty_drops_cy(
    data: sc.AnnData,
    lower: int = 100,
    retain: Optional[int] = None,
    niters: int = 1000,
    alpha: Optional[float] = np.inf,
    batch_size: int = 100,
    progress: bool = True
) -> pd.DataFrame:
    """
    Cython-optimized implementation of EmptyDrops.
    
    Parameters
    ----------
    data : AnnData
        Input data matrix
    lower : int
        Lower bound for empty droplets
    retain : int, optional
        Threshold for certain non-empty droplets
    niters : int
        Number of Monte Carlo iterations
    alpha : float, optional
        Dirichlet concentration parameter
    batch_size : int
        Size of batches for Monte Carlo simulation
    progress : bool
        Whether to show progress bars
        
    Returns
    -------
    pd.DataFrame
        Results including p-values and FDR
    """
    start_time = time.time()
    
    if progress:
        print("Starting Cython-optimized EmptyDrops analysis...")
    
    # Filter genes with zero counts
    if progress:
        print("Filtering genes...")
    sc.pp.filter_genes(data, min_counts=1)
    
    # Get total counts per barcode
    if progress:
        print("Calculating total counts...")
    totals = data.X.sum(axis=1).A1 if issparse(data.X) else data.X.sum(axis=1)
    
    # Identify empty droplets
    empty_mask = totals <= lower
    if progress:
        n_empty = empty_mask.sum()
        print(f"Identified {n_empty} empty droplets out of {len(totals)} total droplets")
    
    # Get ambient profile
    if progress:
        print("Calculating ambient profile...")
    ambient_data = data[empty_mask]
    ambient_counts = ambient_data.X.sum(axis=0).A1 if issparse(ambient_data.X) else ambient_data.X.sum(axis=0)
    
    # Calculate ambient proportions using Good-Turing estimation
    freq_counts, gt_estimates = good_turing_freqs_cy(ambient_counts.astype(np.int64))
    ambient_props = np.zeros_like(ambient_counts, dtype=np.float64)
    
    for i, count in enumerate(ambient_counts):
        if count < len(gt_estimates):
            ambient_props[i] = gt_estimates[int(count)]
    
    # Normalize proportions
    ambient_props = ambient_props / ambient_props.sum()
    
    # Process non-empty droplets
    if progress:
        print("Processing non-empty droplets...")
    non_empty_mask = ~empty_mask
    obs_data = data[non_empty_mask]
    obs_totals = totals[non_empty_mask]
    
    # Convert sparse matrix to dense for Cython processing
    obs_matrix = obs_data.X.toarray() if issparse(obs_data.X) else obs_data.X
    
    # Calculate multinomial probabilities
    if progress:
        print("Calculating multinomial probabilities...")
    obs_probs = parallel_multinomial_prob_cy(
        obs_matrix.astype(np.float64),
        ambient_props.astype(np.float64),
        float(alpha)
    )
    
    # Perform Monte Carlo testing
    if progress:
        print("Performing Monte Carlo testing...")
    sim_results = batch_monte_carlo_cy(
        obs_totals.astype(np.int64),
        ambient_props,
        niters,
        batch_size
    )
    
    # Calculate p-values
    if progress:
        print("Calculating p-values and FDR...")
    n_above = (sim_results >= obs_probs[:, np.newaxis]).sum(axis=1)
    pvals = (n_above + 1) / (niters + 1)
    limited = n_above == 0
    
    # Create results DataFrame
    results = pd.DataFrame(index=data.obs_names)
    results['Total'] = totals
    results['LogProb'] = np.nan
    results['PValue'] = np.nan
    results['Limited'] = np.nan
    results['FDR'] = np.nan
    
    # Fill in results for tested cells
    non_empty_barcodes = data[non_empty_mask].obs_names
    results.loc[non_empty_barcodes, 'LogProb'] = obs_probs
    results.loc[non_empty_barcodes, 'PValue'] = pvals
    results.loc[non_empty_barcodes, 'Limited'] = limited
    
    # Handle retain threshold
    if retain is not None:
        retain_mask = totals >= retain
        results.loc[data[retain_mask].obs_names, 'PValue'] = 0
    
    # Apply FDR correction
    tested_mask = ~results['PValue'].isna()
    if tested_mask.any():
        fdr = multipletests(results.loc[tested_mask, 'PValue'], method='fdr_bh')[1]
        results.loc[tested_mask, 'FDR'] = fdr
    
    if progress:
        total_time = time.time() - start_time
        print(f"EmptyDrops analysis complete! Total time: {format_time(total_time)}")
    
    return results 