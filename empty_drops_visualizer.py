"""
EmptyDrops Visualizer - A wrapper for the EmptyDrops algorithm that adds visualization capabilities.

This module provides visualization tools for the EmptyDrops algorithm without modifying
the original implementation. It imports the empty_drops function and adds visualization
for each step of the algorithm.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Any, Tuple
import scanpy as sc
from empty_drops import empty_drops
import multiprocessing as mp
from numba import jit, prange
from scipy.sparse import issparse

# Create directory for visualizations
VISUALIZATION_DIR = "empty_drops_visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def visualize_empty_drops(
    data: sc.AnnData,
    lower: int = 100,
    retain: Optional[int] = None,
    barcode_args: Optional[dict] = None,
    test_ambient: bool = False,
    niters: int = 1000,
    ignore: Optional[int] = None,
    alpha: Optional[float] = np.inf,
    round: bool = True,
    by_rank: Optional[int] = None,
    known_empty: Optional[np.ndarray] = None,
    progress: bool = True,
    adaptive: bool = True,
    min_iters: int = 100,
    max_iters: int = 1000,
    early_stopping: bool = True,
    batch_size: int = 100,
    use_cache: bool = True,
    save_intermediate_results: bool = True,
    n_processes: int = mp.cpu_count(),
    chunk_size: int = 100
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run EmptyDrops with visualization for each step.
    
    This function wraps the empty_drops function and adds visualization
    for each step of the algorithm without modifying the original function.
    
    Parameters
    ----------
    data : AnnData
        The input data matrix where rows are genes and columns are barcodes
    lower : int, optional (default: 100)
        Lower bound on total UMI count below which barcodes are assumed empty
    retain : int, optional (default: None)
        Threshold above which all barcodes are assumed to contain cells
    barcode_args : dict, optional (default: None)
        Additional arguments for barcode ranking
    test_ambient : bool, optional (default: False)
        Whether to test barcodes with totals <= lower
    niters : int, optional (default: 1000)
        Maximum number of iterations for Monte Carlo p-value calculations
    ignore : int, optional (default: None)
        Lower bound on total UMI count below which barcodes are ignored
    alpha : float, optional (default: np.inf)
        Scaling parameter for Dirichlet-multinomial sampling
    round : bool, optional (default: True)
        Whether to round non-integer values
    by_rank : int, optional (default: None)
        Alternative method for identifying empty droplets
    known_empty : array-like, optional (default: None)
        Indices of barcodes assumed to be empty
    progress : bool, optional (default: True)
        Whether to show progress bars
    adaptive : bool, optional (default: True)
        Whether to use adaptive stopping for Monte Carlo testing
    min_iters : int, optional (default: 100)
        Minimum number of iterations before adaptive stopping
    max_iters : int, optional (default: 1000)
        Maximum number of iterations
    early_stopping : bool, optional (default: True)
        Whether to use early stopping with confidence intervals
    batch_size : int, optional (default: 100)
        Size of batches for processing
    use_cache : bool, optional (default: True)
        Whether to use cached results if available
    save_intermediate_results : bool, optional (default: True)
        Whether to save intermediate results for visualization
    n_processes : int, optional (default: mp.cpu_count())
        Number of processes to use for parallel processing
    chunk_size : int, optional (default: 100)
        Size of chunks for parallel processing
        
    Returns
    -------
    tuple
        (results, intermediate_data)
        - results: DataFrame with EmptyDrops results
        - intermediate_data: Dictionary with intermediate results for visualization
    """
    start_time = time.time()
    
    if progress:
        print("Starting EmptyDrops analysis with visualization...")
    
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Dictionary to store intermediate results
    intermediate_data = {}
    
    # Step 1: Filter genes with zero counts
    if progress:
        print("Step 1: Filtering genes...")
    print(f"{(data_copy.X.sum(axis=0).A1 == 0).sum()} genes filtered out since sum(counts) over the gene was 0.")
    sc.pp.filter_genes(data_copy, min_counts=1)
    
    # Step 2: Calculate total counts per barcode
    if progress:
        print("Step 2: Calculating total counts per barcode...")
    totals = data_copy.X.sum(axis=1).A1
    intermediate_data['totals'] = totals
    
    # Visualize total counts distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(totals, bins=100, log_scale=True)
    plt.title('Distribution of Total UMI Counts')
    plt.xlabel('Total UMI Counts (log scale)')
    plt.ylabel('Count')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'total_counts_distribution.png'))
    plt.close()
    print(f"Total counts distribution plot saved to '{os.path.join(VISUALIZATION_DIR, 'total_counts_distribution.png')}'")
    
    # Step 3: Identify putative empty droplets
    if progress:
        print("Step 3: Identifying empty droplets...")
    if by_rank is not None:
        # Implement rank-based empty droplet identification
        pass
    else:
        empty_mask = totals <= lower
        intermediate_data['empty_mask'] = empty_mask
    
    # Print statistics about empty droplets
    n_empty = np.sum(empty_mask)
    n_total = len(totals)
    print(f"Identified {n_empty} empty droplets out of {n_total} total droplets ({n_empty/n_total*100:.2f}%)")
    
    # Visualize empty droplet threshold
    plt.figure(figsize=(10, 6))
    sns.histplot(totals, bins=100, log_scale=True)
    plt.axvline(x=lower, color='r', linestyle='--', label=f'Lower threshold ({lower})')
    plt.title('Empty Droplet Threshold')
    plt.xlabel('Total UMI Counts (log scale)')
    plt.ylabel('Count')
    plt.legend()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'empty_droplet_threshold.png'))
    plt.close()
    print(f"Empty droplet threshold plot saved to '{os.path.join(VISUALIZATION_DIR, 'empty_droplet_threshold.png')}'")
    
    # Step 4: Get ambient profile from empty droplets
    if progress:
        print("Step 4: Calculating ambient profile...")
    ambient_data = data_copy[empty_mask]
    ambient_totals = totals[empty_mask]
    intermediate_data['ambient_data'] = ambient_data
    intermediate_data['ambient_totals'] = ambient_totals
    
    # Import the good_turing_ambient_pool function from empty_drops
    from empty_drops import good_turing_ambient_pool
    
    # Calculate ambient proportions using Good-Turing estimation
    gene_names, ambient_proportions, gene_expectations = good_turing_ambient_pool(
        data_copy, ambient_data.X.sum(axis=0).A1
    )
    intermediate_data['gene_names'] = gene_names
    intermediate_data['ambient_proportions'] = ambient_proportions
    intermediate_data['gene_expectations'] = gene_expectations
    
    # Visualize ambient proportions
    plt.figure(figsize=(10, 6))
    top_genes = np.argsort(ambient_proportions)[-20:]  # Top 20 genes
    sns.barplot(x=ambient_proportions[top_genes], y=gene_names[top_genes])
    plt.title('Top 20 Genes in Ambient Profile')
    plt.xlabel('Proportion')
    plt.ylabel('Gene')
    plt.tight_layout()
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'ambient_profile.png'))
    plt.close()
    print(f"Ambient profile plot saved to '{os.path.join(VISUALIZATION_DIR, 'ambient_profile.png')}'")
    
    # Step 5: Estimate alpha if not specified
    if alpha is None:
        if progress:
            print("Step 5: Estimating alpha parameter...")
        # Import the estimate_alpha function from empty_drops
        from empty_drops import estimate_alpha
        
        alpha = estimate_alpha(
            ambient_data.X, 
            ambient_proportions,
            ambient_totals,
            progress=progress
        )
        intermediate_data['alpha'] = alpha
        print(f"Estimated alpha parameter: {alpha:.4f}")
    else:
        intermediate_data['alpha'] = alpha
    
    # Step 6: Calculate probabilities for non-empty droplets
    if progress:
        print("Step 6: Calculating probabilities for non-empty droplets...")
    non_empty_mask = ~empty_mask
    if ignore is not None:
        non_empty_mask &= totals > ignore
    intermediate_data['non_empty_mask'] = non_empty_mask
        
    obs_data = data_copy[non_empty_mask]
    obs_totals = totals[non_empty_mask]
    intermediate_data['obs_data'] = obs_data
    intermediate_data['obs_totals'] = obs_totals
    
    # Import the compute_multinom_prob function from empty_drops
    from empty_drops import compute_multinom_prob
    
    # Calculate multinomial probabilities
    obs_probs = compute_multinom_prob(
        obs_data.X,
        ambient_proportions,
        alpha,
        progress=progress
    )
    intermediate_data['obs_probs'] = obs_probs
    
    # Visualize log probabilities
    plt.figure(figsize=(10, 6))
    sns.histplot(obs_probs, bins=50)
    plt.title('Distribution of Log Probabilities')
    plt.xlabel('Log Probability')
    plt.ylabel('Count')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'log_probabilities.png'))
    plt.close()
    print(f"Log probabilities plot saved to '{os.path.join(VISUALIZATION_DIR, 'log_probabilities.png')}'")
    
    # Step 7: Perform Monte Carlo testing
    if progress:
        print("Step 7: Performing Monte Carlo testing...")
    # Import the permute_counter function from empty_drops
    from empty_drops import permute_counter
    
    n_above = permute_counter(
        obs_totals,
        obs_probs,
        ambient_proportions,
        niters,
        alpha,
        progress=progress,
        adaptive=adaptive,
        min_iters=min_iters,
        max_iters=max_iters,
        early_stopping=early_stopping,
        batch_size=batch_size
    )
    intermediate_data['n_above'] = n_above
    
    # Step 8: Calculate p-values
    if progress:
        print("Step 8: Calculating p-values and FDR...")
    pvals = (n_above + 1) / (niters + 1)
    limited = n_above == 0
    intermediate_data['pvals'] = pvals
    intermediate_data['limited'] = limited
    
    # Visualize p-value distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(pvals, bins=50)
    plt.title('Distribution of P-values')
    plt.xlabel('P-value')
    plt.ylabel('Count')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'p_values.png'))
    plt.close()
    print(f"P-value distribution plot saved to '{os.path.join(VISUALIZATION_DIR, 'p_values.png')}'")
    
    # Apply FDR correction
    if retain is not None:
        high_count_mask = totals >= retain
        pvals[high_count_mask] = 0
    
    # Use statsmodels for multiple testing correction
    from statsmodels.stats.multitest import multipletests
    fdr = multipletests(
        pvals,
        method='fdr_bh'
    )[1]
    intermediate_data['fdr'] = fdr
    
    # Visualize FDR distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(fdr, bins=50)
    plt.title('Distribution of FDR')
    plt.xlabel('FDR')
    plt.ylabel('Count')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'fdr.png'))
    plt.close()
    print(f"FDR distribution plot saved to '{os.path.join(VISUALIZATION_DIR, 'fdr.png')}'")
    
    # Step 9: Run the original empty_drops function
    if progress:
        print("Step 9: Running EmptyDrops algorithm...")
    results = empty_drops(
        data,
        lower=lower,
        retain=retain,
        barcode_args=barcode_args,
        test_ambient=test_ambient,
        niters=niters,
        ignore=ignore,
        alpha=alpha,
        round=round,
        by_rank=by_rank,
        known_empty=known_empty,
        progress=progress,
        adaptive=adaptive,
        min_iters=min_iters,
        max_iters=max_iters,
        early_stopping=early_stopping,
        batch_size=batch_size,
        use_cache=use_cache,
        n_processes=n_processes,
        chunk_size=chunk_size
    )
    
    # Visualize final results
    plt.figure(figsize=(10, 6))
    # Create a column for FDR threshold
    results['FDR < 0.05'] = results['FDR'] < 0.05
    sns.scatterplot(data=results, x='Total', y='FDR', hue='FDR < 0.05', 
                   palette=['red', 'blue'], alpha=0.5)
    plt.title('FDR vs Total UMI Counts')
    plt.xlabel('Total UMI Counts')
    plt.ylabel('FDR')
    plt.xscale('log')
    plt.savefig(os.path.join(VISUALIZATION_DIR, 'final_results.png'))
    plt.close()
    print(f"Final results plot saved to '{os.path.join(VISUALIZATION_DIR, 'final_results.png')}'")
    
    # Save intermediate results if requested
    if save_intermediate_results:
        import pickle
        with open(os.path.join(VISUALIZATION_DIR, 'intermediate_results.pkl'), 'wb') as f:
            pickle.dump(intermediate_data, f)
        print(f"Intermediate results saved to '{os.path.join(VISUALIZATION_DIR, 'intermediate_results.pkl')}'")
    
    if progress:
        total_time = time.time() - start_time
        print(f"EmptyDrops analysis with visualization complete! Total time: {format_time(total_time)}")
    
    return results, intermediate_data

def plot_umi_vs_genes(data, results, output_file=None):
    """
    Plot UMI counts vs. number of genes for each droplet.
    
    Parameters
    ----------
    data : AnnData
        The input data matrix
    results : DataFrame
        The results from EmptyDrops
    output_file : str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    # Calculate number of genes per droplet
    n_genes = (data.X > 0).sum(axis=1).A1
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Total': results['Total'],
        'n_genes': n_genes,
        'FDR < 0.05': results['FDR'] < 0.05
    })
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_data, x='Total', y='n_genes', hue='FDR < 0.05', 
                   palette=['red', 'blue'], alpha=0.5)
    plt.title('UMI Counts vs. Number of Genes')
    plt.xlabel('Total UMI Counts')
    plt.ylabel('Number of Genes')
    plt.xscale('log')
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved to '{output_file}'")
    else:
        plt.show()

def plot_alpha_sensitivity(data, lower=100, alpha_values=None, output_file=None):
    """
    Plot the sensitivity of EmptyDrops to different alpha values.
    
    Parameters
    ----------
    data : AnnData
        The input data matrix
    lower : int, optional
        Lower bound on total UMI count
    alpha_values : list, optional
        List of alpha values to test. If None, default values are used.
    output_file : str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    if alpha_values is None:
        alpha_values = [0.1, 1, 10, 100, 1000, np.inf]
    
    # Run EmptyDrops with different alpha values
    results = []
    for alpha in alpha_values:
        print(f"Running EmptyDrops with alpha={alpha}...")
        result = empty_drops(data, lower=lower, alpha=alpha, progress=False)
        n_cells = (result['FDR'] < 0.05).sum()
        results.append({'alpha': alpha, 'n_cells': n_cells})
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(results)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='alpha', y='n_cells', marker='o')
    plt.title('Sensitivity to Alpha Parameter')
    plt.xlabel('Alpha')
    plt.ylabel('Number of Cells Identified')
    plt.xscale('log')
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved to '{output_file}'")
    else:
        plt.show()

def plot_lower_threshold_sensitivity(data, lower_values=None, output_file=None):
    """
    Plot the sensitivity of EmptyDrops to different lower thresholds.
    
    Parameters
    ----------
    data : AnnData
        The input data matrix
    lower_values : list, optional
        List of lower threshold values to test. If None, default values are used.
    output_file : str, optional
        Path to save the plot. If None, the plot is displayed.
    """
    if lower_values is None:
        lower_values = [10, 50, 100, 200, 500, 1000]
    
    # Run EmptyDrops with different lower thresholds
    results = []
    for lower in lower_values:
        print(f"Running EmptyDrops with lower={lower}...")
        result = empty_drops(data, lower=lower, progress=False)
        n_cells = (result['FDR'] < 0.05).sum()
        results.append({'lower': lower, 'n_cells': n_cells})
    
    # Create a DataFrame for plotting
    plot_data = pd.DataFrame(results)
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=plot_data, x='lower', y='n_cells', marker='o')
    plt.title('Sensitivity to Lower Threshold')
    plt.xlabel('Lower Threshold')
    plt.ylabel('Number of Cells Identified')
    plt.xscale('log')
    
    if output_file:
        plt.savefig(output_file)
        plt.close()
        print(f"Plot saved to '{output_file}'")
    else:
        plt.show()

@jit(nopython=True, parallel=True)
def compute_multinom_prob_batch(X, ambient_prop, alpha):
    """Vectorized computation of multinomial probabilities for a batch of cells"""
    n_cells = X.shape[0]
    log_probs = np.zeros(n_cells)
    
    for i in prange(n_cells):  # Parallel loop
        x = X[i]
        log_probs[i] = compute_multinom_prob_single(x, ambient_prop, alpha)
    
    return log_probs

def monte_carlo_test(obs_totals, obs_probs, ambient_prop, niters, alpha, batch_size=1000):
    """Memory-efficient Monte Carlo testing with batched simulations"""
    n_cells = len(obs_totals)
    n_above = np.zeros(n_cells, dtype=np.int32)
    
    for i in range(0, niters, batch_size):
        batch_iters = min(batch_size, niters - i)
        # Simulate only batch_size iterations at a time
        sim_probs = simulate_batch(obs_totals, ambient_prop, alpha, batch_iters)
        n_above += (sim_probs >= obs_probs[:, None]).sum(axis=1)
        
    return n_above

try:
    import cupy as cp
    
    def cuda_simulate_batch(totals, ambient_prop, alpha, n_iter):
        """GPU-accelerated simulation of multinomial samples"""
        d_totals = cp.asarray(totals)
        d_ambient = cp.asarray(ambient_prop)
        d_results = cp.zeros((len(totals), n_iter))
        
        # Launch CUDA kernel for parallel simulation
        threadsperblock = 256
        blockspergrid = (len(totals) + (threadsperblock - 1)) // threadsperblock
        cuda_simulate_kernel[blockspergrid, threadsperblock](
            d_totals, d_ambient, alpha, d_results)
        
        return cp.asnumpy(d_results)
except ImportError:
    # Fall back to CPU implementation if CUDA is not available
    pass

def optimize_sparse_operations(X):
    """Optimize sparse matrix operations"""
    if issparse(X):
        # Convert to CSR format for efficient row operations
        X = X.tocsr()
        # Pre-compute indices for frequently accessed elements
        X.sort_indices()
        return X
    return X

def improved_caching(data_hash, params):
    """Enhanced caching with partial results"""
    cache_key = _get_cache_key(data_hash, params)
    
    # Try to load partial results
    partial_results = load_partial_results(cache_key)
    if partial_results is not None:
        # Resume from last checkpoint
        return resume_from_checkpoint(partial_results)
    
    # Initialize new cache
    cache = {
        'ambient_profile': None,
        'monte_carlo_results': None,
        'final_results': None
    }
    
    # Save results at checkpoints
    def save_checkpoint(stage, data):
        cache[stage] = data
        save_partial_results(cache_key, cache)

def profile_guided_optimization():
    """Enable profile-guided optimization for frequently executed code paths"""
    import cProfile
    import pstats
    
    profiler = cProfile.Profile()
    stats_file = "empty_drops_stats.prof"
    
    def profile_decorator(func):
        def wrapper(*args, **kwargs):
            profiler.enable()
            result = func(*args, **kwargs)
            profiler.disable()
            
            # Save profiling stats
            stats = pstats.Stats(profiler)
            stats.dump_stats(stats_file)
            
            return result
        return wrapper
    
    return profile_decorator

if __name__ == "__main__":
    # Example usage
    print("EmptyDrops Visualizer")
    print("This module provides visualization tools for the EmptyDrops algorithm.")
    print("Import and use the functions in this module to visualize EmptyDrops results.") 