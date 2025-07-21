"""
EmptyDrops - Python implementation of the EmptyDrops algorithm for droplet-based single-cell RNA sequencing.
Modified for Apple Silicon (M-series) compatibility with enhanced performance optimizations.

This module provides functionality to distinguish between droplets containing cells 
and ambient RNA in droplet-based single-cell RNA sequencing experiments.

Based on:
Lun A, Riesenfeld S, Andrews T, et al. (2019).
Distinguishing cells from empty droplets in droplet-based single-cell RNA sequencing data.
Genome Biol. 20, 63.
"""

import scanpy as sc
import numpy as np
from numba import typeof, jit, prange, cuda
from scipy.sparse import csr_matrix, issparse, spmatrix
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma, gammaln, factorial, loggamma
from scipy.interpolate import UnivariateSpline
import scipy.stats as ss
from scipy import stats
from statsmodels.stats.multitest import multipletests
from collections import Counter
from tqdm import tqdm
from typing import Tuple, Dict, Union, Optional, List
import pandas as pd
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial
from tqdm.auto import tqdm
import time
from datetime import datetime, timedelta
import os
import pickle
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gc
from empty_drops_cython.empty_drops_core import permute_counter, parallel_multinomial_prob_cy

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
# some pkg_resources warnings probably coming from scanpy, anndata or statsmodels. 

try:
    import cupy as cp
    HAS_GPU = True
except ImportError:
    HAS_GPU = False

# Performance note: from 100 (13:00min) changed to 300 (5:00min), to 1000 (3:19min), to 2000 (2:49min), to 4000 (2:15min), to 20000 (0:23min)
# imporved performance by 34x by removing chunking.
# Chunking removed as processing all data at once is faster.
# R func took 895:35min to run, which is about 3810x slower than the Python function.

# Optimized constants for M-series chips
MAX_WORKERS = mp.cpu_count()  # Use all available cores
CACHE_DIR = "empty_drops_cache"
N_PROCESSES = mp.cpu_count()  # Use all cores
MAX_MEMORY_PERCENT = 100  # changed from 80 to 100
COOLING_PERIOD = 1  # Reduced cooling period
MAX_CPU_PERCENT = 100  # Increased CPU usage threshold
SAVE_FREQUENCY = 5  # More frequent checkpoints

# Create cache directory if it doesn't exist
os.makedirs(CACHE_DIR, exist_ok=True)

def get_system_stats():
    """Get current system resource usage."""
    process = psutil.Process(os.getpid())
    return {
        'memory_percent': process.memory_percent(),
        'cpu_percent': process.cpu_percent(),
        'num_threads': process.num_threads()
    }

def check_system_resources():
    """Check if system resources are too high."""
    stats = get_system_stats()
    if stats['memory_percent'] > MAX_MEMORY_PERCENT or stats['cpu_percent'] > MAX_CPU_PERCENT:
        print(f"\nHigh resource usage detected - Memory: {stats['memory_percent']:.1f}%, CPU: {stats['cpu_percent']:.1f}%")
        print("Cooling down...")
        gc.collect()
        time.sleep(COOLING_PERIOD)
        return True
    return False

def save_intermediate_results(key: str, data: dict):
    """Save intermediate results to prevent data loss."""
    temp_file = os.path.join(CACHE_DIR, f"intermediate_{key}.pkl")
    with open(temp_file, 'wb') as f:
        pickle.dump(data, f)

def get_memory_usage():
    """Get current memory usage as a percentage."""
    process = psutil.Process(os.getpid())
    return process.memory_percent()

def check_memory_usage():
    """Check if memory usage is too high."""
    usage = get_memory_usage()
    if usage > MAX_MEMORY_PERCENT:
        gc.collect()  # Force garbage collection
        return True
    return False

def _get_cache_key(data_hash: str, params: dict) -> str:
    """Generate a unique cache key for the given data and parameters."""
    # Sort parameters to ensure consistent keys
    sorted_params = sorted(params.items())
    param_str = str(sorted_params)
    
    # Combine data hash and parameters
    key_str = f"{data_hash}_{param_str}"
    
    # Generate a hash of the combined string
    return hashlib.md5(key_str.encode()).hexdigest()

def _save_to_cache(key: str, data: dict):
    """Save data to cache."""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)

def _load_from_cache(key: str) -> Optional[dict]:
    """Load data from cache if it exists."""
    cache_file = os.path.join(CACHE_DIR, f"{key}.pkl")
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return None

# Numba-compatible gammaln function for scalar inputs
@jit(nopython=True)
def _numba_gammaln_scalar(x):
    """Numba-compatible version of gammaln for scalar inputs."""
    # Lanczos approximation for log gamma
    c = np.array([
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ])
    
    y = x
    tmp = x + 5.5
    tmp = (x + 0.5) * np.log(tmp) - tmp
    ser = 1.000000000190015
    for j in range(6):
        y = y + 1
        ser = ser + c[j] / y
    return tmp + np.log(2.5066282746310005 * ser / x)

# Vectorized version for array inputs
@jit(nopython=True)
def _numba_gammaln_array(x):
    """Numba-compatible version of gammaln for array inputs."""
    result = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        result.flat[i] = _numba_gammaln_scalar(x.flat[i])
    return result

class TimeEstimator:
    """Helper class to estimate remaining time for long-running operations."""
    def __init__(self, total, desc, position=0):
        self.pbar = tqdm(
            total=total, 
            desc=desc, 
            position=position, 
            leave=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        self.start_time = time.time()
        self.current = 0
        self.total = total
        
    def update(self, n=1):
        self.current += n
        self.pbar.update(n)
        
    def close(self):
        self.pbar.close()
        
    def get_elapsed_time(self):
        return time.time() - self.start_time
        
    def reset(self):
        """Reset the progress bar to the beginning."""
        self.pbar.reset()
        self.current = 0
        self.start_time = time.time()

def format_time(seconds):
    """Format time in seconds to a human-readable string."""
    return str(timedelta(seconds=int(seconds)))

def create_output_directory(data_name: str = None, base_dir: str = "emptydrops_runs") -> str:
    """
    Create a timestamped output directory for EmptyDrops results.
    
    Parameters
    ----------
    data_name : str, optional
        Name of the dataset. If None, will use 'dataset'
    base_dir : str
        Base directory name for all EmptyDrops runs
        
    Returns
    -------
    str
        Path to the created output directory
    """
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean data name for filesystem
    if data_name is None:
        data_name = "dataset"
    else:
        # Remove problematic characters for filesystem
        import re
        data_name = re.sub(r'[<>:"/\\|?*]', '_', str(data_name))
        data_name = data_name.strip('.')  # Remove leading/trailing dots
    
    # Create directory structure: base_dir/data_name/timestamp_run/
    output_dir = os.path.join(base_dir, data_name, f"{timestamp}_run")
    
    # Create all necessary subdirectories
    subdirs = [
        "",  # Main directory
        "visualizations",
        "results", 
        "metadata",
        "logs"
    ]
    
    for subdir in subdirs:
        full_path = os.path.join(output_dir, subdir)
        os.makedirs(full_path, exist_ok=True)
    
    return output_dir

def get_dataset_name(data: sc.AnnData) -> str:
    """
    Extract a meaningful dataset name from AnnData object.
    
    Parameters
    ----------
    data : AnnData
        Input data object
        
    Returns
    -------
    str
        Dataset name for directory creation
    """
    # Try to get name from various sources
    if hasattr(data, 'uns') and 'dataset_name' in data.uns:
        return str(data.uns['dataset_name'])
    elif hasattr(data, 'uns') and 'sample_id' in data.uns:
        return str(data.uns['sample_id'])
    elif hasattr(data, 'obs') and 'sample' in data.obs.columns:
        # Use first unique sample name if available
        unique_samples = data.obs['sample'].unique()
        if len(unique_samples) == 1:
            return str(unique_samples[0])
        elif len(unique_samples) > 1:
            return f"multi_sample_{len(unique_samples)}"
    
    # Fallback: use data dimensions
    return f"data_{data.n_obs}cells_{data.n_vars}genes"

@jit(nopython=True)
def _compute_good_turing_freqs(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Good-Turing frequency estimates for observed counts.
    
    Parameters
    ----------
    counts : array-like
        Observed counts
        
    Returns
    -------
    tuple
        (Smoothed frequencies of frequencies, Good-Turing estimates)
    """
    # Count frequencies of frequencies
    max_count = int(counts.max())
    freq_counts = np.zeros(max_count + 1)
    
    # Sequential counting of frequencies
    for i in range(len(counts)):
        c = int(counts[i])
        if c <= max_count:
            freq_counts[c] += 1
            
    # Smooth frequencies using linear interpolation in log-space
    log_counts = np.log1p(np.arange(max_count + 1))
    log_freqs = np.log1p(freq_counts)
    
    # Replace zeros with interpolated values
    for i in range(1, max_count + 1):
        if freq_counts[i] == 0:
            # Find next non-zero frequency
            next_nonzero = i + 1
            while next_nonzero <= max_count and freq_counts[next_nonzero] == 0:
                next_nonzero += 1
                
            if next_nonzero <= max_count:
                # Linear interpolation in log space
                prev_nonzero = i - 1
                while prev_nonzero >= 0 and freq_counts[prev_nonzero] == 0:
                    prev_nonzero -= 1
                    
                if prev_nonzero >= 0:
                    slope = (log_freqs[next_nonzero] - log_freqs[prev_nonzero]) / (log_counts[next_nonzero] - log_counts[prev_nonzero])
                    log_freqs[i] = log_freqs[prev_nonzero] + slope * (log_counts[i] - log_counts[prev_nonzero])
                    freq_counts[i] = np.exp(log_freqs[i]) - 1
    
    # Calculate Good-Turing estimates
    gt_estimates = np.zeros_like(freq_counts)
    for i in range(max_count):
        if freq_counts[i] > 0:
            gt_estimates[i] = (i + 1) * freq_counts[i + 1] / freq_counts[i]
            
    return freq_counts, gt_estimates

def good_turing_ambient_pool(
    data: sc.AnnData,
    low_count_gene_sums: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate the ambient RNA profile using Good-Turing estimation.
    
    This implementation follows the method described in Lun et al. (2019) for estimating
    the ambient RNA profile from empty droplets using Good-Turing frequency estimation.
    
    Parameters
    ----------
    data : AnnData
        The input data matrix where rows are cells and columns are genes
    low_count_gene_sums : array-like
        Sum of counts for each gene in low-count barcodes
        
    Returns
    -------
    tuple
        (gene_names, ambient_proportions, gene_expectations)
        - gene_names: names of genes
        - ambient_proportions: Good-Turing estimates of gene proportions in ambient RNA
        - gene_expectations: Expected counts for each gene
    """
    gene_names = data.var_names.values
    
    # Convert to dense array for processing
    if issparse(low_count_gene_sums):
        counts = low_count_gene_sums.A1
    else:
        counts = low_count_gene_sums
    
    # Remove genes with zero counts
    nonzero_mask = counts > 0
    nonzero_counts = counts[nonzero_mask]
    nonzero_genes = gene_names[nonzero_mask]
    
    if len(nonzero_counts) == 0:
        warnings.warn("No non-zero counts found in ambient set")
        return gene_names, np.zeros_like(counts), np.zeros_like(counts)
    
    # Compute Good-Turing estimates
    freq_counts, gt_estimates = _compute_good_turing_freqs(nonzero_counts)
    
    # Calculate proportions using Good-Turing estimates
    total_counts = np.sum(nonzero_counts)
    ambient_props = np.zeros_like(counts)
    
    for i, count in enumerate(nonzero_counts):
        if count < len(gt_estimates):
            ambient_props[nonzero_mask][i] = gt_estimates[int(count)] / total_counts
        else:
            # For high counts, use the observed proportion
            ambient_props[nonzero_mask][i] = count / total_counts
            
    # Normalize proportions
    sum_props = np.sum(ambient_props)
    if sum_props > 0:
        ambient_props = ambient_props / sum_props
    else:
        warnings.warn("Sum of ambient proportions is zero, using uniform distribution")
        ambient_props = np.ones_like(ambient_props) / len(ambient_props)
    
    # Calculate expectations
    gene_expectations = ambient_props * total_counts
    
    return gene_names, ambient_props, gene_expectations

@jit(nopython=True)
def _compute_multinom_prob_chunk(data_chunk: np.ndarray, prop: np.ndarray, alpha: float) -> np.ndarray:
    """Compute multinomial probabilities for a chunk of data.
    
    Args:
        data_chunk: Array of counts for the chunk
        prop: Array of proportions
        alpha: Concentration parameter
        
    Returns:
        Array of log probabilities
    """
    # Ensure consistent dtype
    data_chunk = data_chunk.astype(np.float64)
    prop = prop.astype(np.float64)
    
    # Initialize array for log probabilities
    log_probs = np.zeros(data_chunk.shape[0], dtype=np.float64)
    
    # Compute log probabilities for each row using multinomial formula:
    # log P(x|n,p) = lgamma(n+1) - sum(lgamma(x_i+1)) + sum(x_i * log(p_i))
    for i in range(data_chunk.shape[0]):
        row_sum = 0.0
        sum_log_factorials = 0.0
        
        # Calculate total count and sum of log factorials
        for j in range(data_chunk.shape[1]):
            if data_chunk[i, j] > 0:
                sum_log_factorials += _numba_gammaln_scalar(data_chunk[i, j] + 1)
                row_sum += data_chunk[i, j]
        
        # Multinomial formula: lgamma(n+1) - sum(lgamma(x_i+1)) + sum(x_i * log(p_i))
        if row_sum > 0:
            log_probs[i] = _numba_gammaln_scalar(row_sum + 1) - sum_log_factorials
        
        # Add log probability of proportions
        if alpha > 0:
            log_probs[i] += np.log(prop).dot(data_chunk[i])
        else:
            log_probs[i] += np.log(prop + 1e-10).dot(data_chunk[i])
    
    return log_probs

def compute_multinom_prob(
    data: Union[np.ndarray, spmatrix],
    prop: np.ndarray,
    alpha: float = np.inf,
    progress: bool = True
) -> np.ndarray:
    """Calculate multinomial probabilities directly without chunking."""
    if issparse(data):
        data = data.tocsr().toarray()
    
    if progress:
        print("\nComputing multinomial probabilities...")
        print(f"Data size: {data.nbytes / (1024 * 1024):.2f} MB")
        print(f"Number of cells: {len(data)}\n")
    
    # Process all data at once (faster than chunking)
    return _compute_multinom_prob_chunk(data, prop, alpha)

@jit(nopython=True, parallel=True)
def _simulate_batch(totals: np.ndarray, ambient: np.ndarray, n_iter: int) -> np.ndarray:
    """Simulate multiple count vectors in parallel using multinomial distribution."""
    n_cells = len(totals)
    results = np.zeros((n_cells, n_iter))
    
    for i in prange(n_cells):
        if totals[i] <= 0:
            continue
        for j in range(n_iter):
            results[i, j] = np.random.multinomial(totals[i], ambient).sum()
    
    return results

if HAS_GPU:
    @cuda.jit
    def _cuda_simulate_kernel(totals, ambient, results):
        """CUDA kernel for parallel simulation on GPU."""
        idx = cuda.grid(1)
        if idx < len(totals):
            if totals[idx] > 0:
                # Use shared memory for ambient profile
                shared_ambient = cuda.shared.array(shape=(ambient.shape[0],), dtype=np.float64)
                if cuda.threadIdx.x == 0:
                    for i in range(ambient.shape[0]):
                        shared_ambient[i] = ambient[i]
                cuda.syncthreads()
                
                # Simulate multinomial using cuRAND
                for j in range(results.shape[1]):
                    results[idx, j] = cuda.random.multinomial(totals[idx], shared_ambient).sum()

def optimize_sparse_matrix(X):
    """Optimize sparse matrix operations."""
    if issparse(X):
        X = X.tocsr()
        X.sort_indices()
        X.sum_duplicates()
        return X
    return X

def calculate_knee_point(totals: np.ndarray, 
                        lower: int = 100, 
                        fit_bounds: Optional[Tuple[float, float]] = None, 
                        output_dir: Optional[str] = None) -> Tuple[float, dict]:
    """
    Calculate knee points using the same algorithm as described in the EmptyDrops paper.
    
    This implements the sophisticated knee detection algorithm from DropletUtils,
    which finds the point that represents the knee point of the curvature.
    
    Parameters
    ----------
    totals : array-like
        Total UMI counts per barcode
    lower : int
        Lower bound on total counts for curve fitting
    fit_bounds : tuple, optional
        (lower, upper) bounds for curve fitting region
    output_dir : str, optional
        Output directory for saving the plot
    Returns
    -------
    tuple
        (knee_count, knee_parameters) - total UMI counts at knee and dict with knee parameters
    """
    tb = totals[totals > lower]

    if len(tb) < 3:
        warnings.warn("Insufficient unique points for computing knee/inflection points")
        # Return reasonable fallbacks
        high_count_threshold = np.percentile(totals[totals > lower], 90) if np.any(totals > lower) else lower * 10
        return float(high_count_threshold), {}

    # Rank all barcodes in order of decreasing total UMI count
    tb = np.sort(np.asarray(tb))[::-1]

    # Calculate ranks (1-based)
    ranks = np.arange(1, len(tb) + 1)

    # Calculate log-transformed values
    x = np.log10(ranks)
    y = np.log10(tb + 1e-8)  # avoid log(0)

    spline = UnivariateSpline(x, y, k=3, s=1e5) # s = smoothing factor
    f1 = spline.derivative(n=1)(x) #erste ableitung
    f2 = spline.derivative(n=2)(x) # zwote  
    curvature = f2 / (1 + f1**2)**1.5 #signed curvature
    # Minimum (most negative curvature)
    knee_idx = np.argmin(curvature) - 1 # da 0 basiert
    knee_x = x[knee_idx]
    knee_y = y[knee_idx]
    knee_count = tb[knee_idx]#10 ** knee_y
    plt.figure(figsize=(6,4))
    plt.plot(x, y, label='log10(total counts)')
    plt.scatter(knee_x, knee_y, color='red', label=f'Py Knee ≈ {knee_count:.0f}')
    plt.plot(x, curvature, color='red', label='Py curvature f2/1+f1^2^1.5')
    plt.xlabel("log10(Rank)")
    plt.ylabel("log10(Total UMI count)")
    plt.title("Knee point via curvature minimization")
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'visualizations', 'knee_point.png'))
    plt.close()
    print(f"Knee point plot saved to '{os.path.join(output_dir, 'visualizations', 'knee_point.png')}'")
    
    # Create knee parameters dictionary to return
    knee_parameters = {
        "knee_idx": int(knee_idx),
        "knee_x": float(knee_x),
        "knee_y": float(knee_y),
        "knee_count": float(knee_count)
    }
    
    return float(knee_count), knee_parameters

def _run_monte_carlo_chunk(args):
    """
    Helper function to run a chunk of Monte Carlo iterations in parallel.
    
    KEY INSIGHT: We import permute_counter HERE (in the worker process), 
    not pass it as an argument. This way each worker gets its own Cython function!
    """
    totals, obs_probs, ambient_props, alpha, n_iterations, seed, show_progress = args
    
    # Import Cython function in the worker process (this works!)
    from empty_drops_cython.empty_drops_core import permute_counter
    
    # Set random seed for this process
    np.random.seed(seed)
    
    # Use the FAST Cython implementation!
    n_above = permute_counter(
        totals,
        obs_probs,
        ambient_props,
        n_iterations,
        alpha,
        progress=show_progress,  # Can show progress for this worker
        adaptive=False,          # Disable adaptive in parallel mode
        min_iters=n_iterations,  # Run exactly n_iterations
        max_iters=n_iterations,
        early_stopping=False,    # Disable early stopping in parallel
        batch_size=200           # Reasonable batch size
    )
    
    return n_above

def empty_drops(
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
    batch_size: int = 200,
    use_cache: bool = True,
    visualize: bool = True,
    confidence_level: float = 0.95,
    n_processes: int = N_PROCESSES,
    use_gpu: bool = HAS_GPU,
    output_dir: Optional[str] = None,
    dataset_name: Optional[str] = None
) -> pd.DataFrame:
    """
    Enhanced EmptyDrops implementation with performance optimizations.
    
    This function identifies cell-containing droplets from ambient RNA in single-cell
    RNA sequencing data. It automatically creates organized output directories with
    timestamped folders to save all results and visualizations.
    
    Parameters
    ----------
    data : AnnData
        Single-cell data object containing count matrix
    lower : int, optional (default: 100)
        Lower bound on total UMI count for testing droplets
    retain : int, optional (default: None)
        Total UMI count above which all barcodes are considered cells.
        If None, will be calculated using knee point detection
    barcode_args : dict, optional
        Additional arguments for barcode ranking (knee point detection)
    test_ambient : bool, optional (default: False)
        Whether to test ambient RNA profile
    niters : int, optional (default: 1000)
        Number of iterations for Monte Carlo p-value calculation
    ignore : int, optional (default: None)
        Total UMI count below which barcodes are ignored
    alpha : float, optional (default: np.inf)
        Dirichlet-multinomial alpha parameter. If np.inf, uses multinomial
    round : bool, optional (default: True)
        Whether to round expected counts to integers
    by_rank : int, optional (default: None)
        Number of top-ranked barcodes to use for ambient profile
    known_empty : array-like, optional (default: None)
        Boolean mask of known empty droplets
    progress : bool, optional (default: True)
        Whether to show progress bars and messages
    adaptive : bool, optional (default: True)
        Whether to use adaptive iteration stopping
    min_iters : int, optional (default: 100)
        Minimum number of iterations for adaptive stopping
    max_iters : int, optional (default: 1000)
        Maximum number of iterations for adaptive stopping
    early_stopping : bool, optional (default: True)
        Whether to enable early stopping based on p-value convergence
    batch_size : int, optional (default: 20000)
        Batch size for Monte Carlo iterations (critical for performance)
    use_cache : bool, optional (default: True)
        Whether to use caching for repeated analyses
    visualize : bool, optional (default: True)
        Whether to create visualization plots
    confidence_level : float, optional (default: 0.95)
        Confidence level for statistical tests
    n_processes : int, optional (default: N_PROCESSES)
        Number of CPU processes for parallel Monte Carlo computation.
        0 = auto-detect (uses all CPUs - 1), 1 = disable parallelization
    use_gpu : bool, optional (default: HAS_GPU)
        Whether to use GPU acceleration if available
    output_dir : str, optional (default: None)
        Custom output directory path. If None, creates timestamped directory
    dataset_name : str, optional (default: None)
        Name for the dataset. If None, infers from data object
        
    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        - Total: Total UMI count per barcode
        - LogProb: Log probability under ambient profile
        - PValue: Monte Carlo p-value
        - Limited: Whether testing was limited by iterations
        - FDR: False discovery rate (Benjamini-Hochberg corrected)
        
    Notes
    -----
    The function creates an organized directory structure:
    - emptydrops_runs/dataset_name/timestamp_run/
      - visualizations/: All plots and figures
      - results/: CSV files with results and ambient profile
      - metadata/: JSON metadata with parameters and summary
      - logs/: Future log files
    """
    start_time = time.time()
    
    # Set up output directory
    if output_dir is None:
        if dataset_name is None:
            dataset_name = get_dataset_name(data)
        output_dir = create_output_directory(dataset_name)
    else:
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        for subdir in ["visualizations", "results", "metadata", "logs"]:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)
    
    if progress:
        print("Starting EmptyDrops analysis...")
        print(f"Output directory: {output_dir}")
    
    # Generate a hash of the data
    data_hash = hashlib.md5(data.X.data.tobytes()).hexdigest()
    
    # Create a dictionary of parameters
    params = {
        'lower': lower,
        'retain': retain,
        'test_ambient': test_ambient,
        'niters': niters,
        'ignore': ignore,
        'alpha': alpha,
        'round': round,
        'by_rank': by_rank,
        'adaptive': adaptive,
        'min_iters': min_iters,
        'max_iters': max_iters,
        'early_stopping': early_stopping,
        'batch_size': batch_size,
        'confidence_level': confidence_level,
        'n_processes': n_processes,
        'use_gpu': use_gpu
    }
    
    # Generate cache key
    cache_key = _get_cache_key(data_hash, params)
    
    # Check if cached results exist
    if use_cache:
        cached_results = _load_from_cache(cache_key)
        if cached_results is not None:
            if progress:
                print("Loading cached results...")
            return cached_results['results']
    
    # Visualization directory is already created in output_dir setup
    
    # Filter genes with zero counts
    # Note: This doesn't affect cell totals since we're only removing genes with 0 counts across ALL cells
    if progress:
        print("Filtering genes...")
    n_zero_genes = (data.X.sum(axis=0).A1 == 0).sum()
    print(f"{n_zero_genes} genes filtered out since sum(counts) over the gene was 0.")
    sc.pp.filter_genes(data, min_counts=1)
    
    # Get total counts per barcode
    if progress:
        print("Calculating total counts per barcode...")
        print(f"Data shape: {data.shape} (n_obs={data.n_obs}, n_vars={data.n_vars})")
    totals = data.X.sum(axis=1).A1
    
    # Visualize total counts distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(totals, bins=100, log_scale=True)
        plt.title('Distribution of Total UMI Counts')
        plt.xlabel('Total UMI Counts (log scale)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'visualizations', 'total_counts_distribution.png'))
        plt.close()
        print(f"Total counts distribution plot saved to '{os.path.join(output_dir, 'visualizations', 'total_counts_distribution.png')}'")
    
    # Handle retain threshold - calculate BEFORE testing
    knee_parameters = {}  # Initialize for later use in metadata
    if retain is None:
        # Calculate knee point using barcodeRanks equivalent
        if progress:
            print("Calculating knee point for retain threshold...")
        
        try:
            knee, knee_parameters = calculate_knee_point(totals, lower, barcode_args.get('fit_bounds') if barcode_args else None, output_dir)
            retain = knee
            if progress:
                print(f"Calculated retain threshold (knee point): {retain:.1f}")
        except Exception as e:
            if progress:
                print(f"Knee point calculation failed: {e}")
                print("Using fallback retain threshold...")
            # Fallback to 95th percentile of high-count droplets
            high_count_droplets = totals[totals > lower]
            if len(high_count_droplets) > 0:
                retain = np.percentile(high_count_droplets, 95)
            else:
                retain = lower * 10  # Conservative fallback
            if progress:
                print(f"Fallback retain threshold: {retain:.1f}")
    
    # Identify putative empty droplets
    if progress:
        print("Identifying empty droplets...")
    if by_rank is not None:
        # Implement rank-based empty droplet identification
        pass
    else:
        empty_mask = totals <= lower
    
    # Print statistics about empty droplets
    n_empty = np.sum(empty_mask)
    n_total = len(totals)
    print(f"Identified {n_empty} empty droplets out of {n_total} total droplets ({n_empty/n_total*100:.2f}%)")
    
    # Visualize empty droplet threshold
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(totals, bins=100, log_scale=True)
        plt.axvline(x=lower, color='r', linestyle='--', label=f'Lower threshold ({lower})')
        if retain is not None:
            plt.axvline(x=retain, color='g', linestyle='--', label=f'Retain threshold ({retain})')
        plt.title('Empty Droplet Threshold')
        plt.xlabel('Total UMI Counts (log scale)')
        plt.ylabel('Count')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'visualizations', 'empty_droplet_threshold.png'))
        plt.close()
        print(f"Empty droplet threshold plot saved to '{os.path.join(output_dir, 'visualizations', 'empty_droplet_threshold.png')}'")
    
    # Get ambient profile from empty droplets
    if progress:
        print("Calculating ambient profile...")
    ambient_data = data[empty_mask]
    ambient_totals = totals[empty_mask]
    
    # Calculate ambient proportions using Good-Turing estimation
    gene_names, ambient_proportions, gene_expectations = good_turing_ambient_pool(
        data, ambient_data.X.sum(axis=0).A1
    )
    
    # Visualize ambient proportions
    if visualize:
        plt.figure(figsize=(10, 6))
        top_genes = np.argsort(ambient_proportions)[-20:]  # Top 20 genes
        sns.barplot(x=ambient_proportions[top_genes], y=gene_names[top_genes])
        plt.title('Top 20 Genes in Ambient Profile')
        plt.xlabel('Proportion')
        plt.ylabel('Gene')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'visualizations', 'ambient_profile.png'))
        plt.close()
        print(f"Ambient profile plot saved to '{os.path.join(output_dir, 'visualizations', 'ambient_profile.png')}'")
    
    # Estimate alpha if not specified
    if alpha is None:
        if progress:
            print("Estimating alpha parameter...")
        alpha = estimate_alpha(
            ambient_data.X, 
            ambient_proportions,
            ambient_totals,
            progress=progress
        )
        print(f"Estimated alpha parameter: {alpha:.4f}")
    
    # Calculate probabilities for droplets that need testing
    # R tests ALL cells with Total > lower, not just those between lower and retain
    if progress:
        print(f"Identifying droplets to test (all cells with > {lower} UMI counts)...")
    
    # Cells to test: all cells above lower threshold
    # The retain threshold is only used later for FDR correction
    test_mask = totals > lower
    
    if ignore is not None:
        test_mask &= totals > ignore
    
    n_to_test = np.sum(test_mask)
    print(f"Testing {n_to_test} droplets above lower threshold ({lower})")
    print(f"Note: {np.sum(totals >= retain)} droplets >= retain threshold ({retain:.1f}) will be auto-retained in FDR correction")
        
    obs_data = data[test_mask]
    obs_totals = totals[test_mask]
    
    # Convert ambient_proportions to float64 BEFORE calling Cython
    ambient_proportions = ambient_proportions.astype(np.float64)
    
    # Calculate multinomial probabilities using Cython (same as Monte Carlo)
    if progress:
        print("\nComputing multinomial probabilities...")
        obs_data_dense = obs_data.X.toarray() if issparse(obs_data.X) else obs_data.X
        # Ensure float64 for Cython
        obs_data_dense = obs_data_dense.astype(np.float64)
        print(f"Data size: {obs_data_dense.nbytes / (1024 * 1024):.2f} MB")
        print(f"Number of cells: {len(obs_data_dense)}")
        print(f"[DEBUG] obs_data_dense.shape: {obs_data_dense.shape}")
        print(f"[DEBUG] ambient_proportions.shape: {ambient_proportions.shape}")
        if obs_data_dense.shape[1] != len(ambient_proportions):
            print(f"❌ ERROR: Gene dimension mismatch! Data has {obs_data_dense.shape[1]} genes but ambient has {len(ambient_proportions)}")
        print("[DEBUG] Using Cython parallel_multinomial_prob_cy for observed probabilities\n")
        obs_probs = parallel_multinomial_prob_cy(obs_data_dense, ambient_proportions, alpha)
    else:
        obs_data_dense = obs_data.X.toarray() if issparse(obs_data.X) else obs_data.X
        obs_data_dense = obs_data_dense.astype(np.float64)
        obs_probs = parallel_multinomial_prob_cy(obs_data_dense, ambient_proportions, alpha)
    
    # Visualize log probabilities
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(obs_probs, bins=50)
        plt.title('Distribution of Log Probabilities')
        plt.xlabel('Log Probability')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'visualizations', 'log_probabilities.png'))
        plt.close()
        print(f"Log probabilities plot saved to '{os.path.join(output_dir, 'visualizations', 'log_probabilities.png')}'")
    
    # Convert arrays to correct types for Cython function
    print("\nConverting arrays to correct types...")
    print(f"obs_totals dtype before: {obs_totals.dtype}")
    obs_totals = obs_totals.astype(np.int64)
    print(f"obs_totals dtype after: {obs_totals.dtype}")
    
    print(f"obs_probs dtype before: {obs_probs.dtype}")
    obs_probs = obs_probs.astype(np.float64)
    print(f"obs_probs dtype after: {obs_probs.dtype}")
    
    # ambient_proportions already converted to float64 above
    print(f"ambient_proportions dtype: {ambient_proportions.dtype}")
    print(f"ambient_proportions shape: {ambient_proportions.shape}")
    print(f"ambient_proportions sum: {ambient_proportions.sum()}")
    
    # === SANITY CHECKS (Fast Validation) ===
    if progress:
        print("\n=== Running Sanity Checks ===")
        
        # Check 1: Observed probabilities should be reasonable
        print(f"Observed log-prob range: [{obs_probs.min():.2f}, {obs_probs.max():.2f}]")
        if obs_probs.max() > 0:
            print("⚠️  WARNING: Some log-probabilities are positive (should be negative)!")
        
        # Check 2: Test a single simulation to verify comparison direction
        test_sim = np.zeros((len(obs_totals), len(ambient_proportions)), dtype=np.float64)
        for i in range(min(10, len(obs_totals))):  # Test first 10 cells
            if obs_totals[i] > 0:
                test_sim[i] = np.random.multinomial(int(obs_totals[i]), ambient_proportions)
        
        print(f"[DEBUG] test_sim.shape: {test_sim[:10].shape}, ambient_proportions.shape: {ambient_proportions.shape}")
        test_probs = parallel_multinomial_prob_cy(test_sim[:10], ambient_proportions, alpha)
        
        print(f"Sample observed probs (first 3): {obs_probs[:3]}")
        print(f"Sample simulated probs (first 3): {test_probs[:3]}")
        print(f"Difference (sim - obs, should vary): {test_probs[:3] - obs_probs[:3]}")
        print(f"  → Positive means simulation is MORE likely (less negative)")
        print(f"  → For real cells: expect positive (sim > obs)")
        print(f"  → For empty droplets: expect ~0 (sim ≈ obs)")
        
        n_lower = np.sum(test_probs <= obs_probs[:10])
        print(f"Quick test (10 cells, 1 simulation): {n_lower}/10 simulations had LOWER probability than observed")
        print(f"  → Expected: ~5/10 for ambient, 0-2/10 for real cells")
        
        if n_lower == 0:
            print("⚠️  Quick test: No simulations more extreme (expected for real cells)")
            print("   Note: This is fine if most tested cells are real cells, not empty droplets")
            print("   Will be validated with full Monte Carlo run...")
        elif n_lower == 10:
            print("⚠️  WARNING: All simulations were more extreme. This suggests wrong comparison!")
        else:
            print("✅ Comparison direction looks correct!")
        
        print("=== Sanity Checks Complete ===\n")
    
    # Perform Monte Carlo testing
    if progress:
        print("Performing Monte Carlo testing...")
    
    # Use multiprocessing if requested and beneficial
    if n_processes == 0:
        n_processes = max(1, mp.cpu_count()-1)  # Auto-detect, leave one core free
    
    if n_processes > 1 and niters >= n_processes * 10:
        # Parallel execution using multiprocessing
        iters_per_process = niters // n_processes
        remainder = niters % n_processes
        
        # Create tasks for each process with unique seeds
        tasks = []
        base_seed = np.random.randint(0, 2**31)
        for proc_idx in range(n_processes):
            n_iters = iters_per_process + (1 if proc_idx < remainder else 0)
            seed = base_seed + proc_idx
            # show_progress=False to avoid progress bar clutter (we track workers instead)
            tasks.append((obs_totals, obs_probs, ambient_proportions, alpha, n_iters, seed, False))
        
        # Run in parallel with progress bar
        if progress:
            from tqdm import tqdm
            print(f"Using {n_processes} CPU cores for parallel Monte Carlo simulation...")
            with mp.Pool(processes=n_processes) as pool:
                results = list(tqdm(
                    pool.imap(_run_monte_carlo_chunk, tasks),
                    total=len(tasks),
                    desc=f"Monte Carlo (using {n_processes} CPUs)",
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} workers [{elapsed}<{remaining}]'
                ))
        else:
            with mp.Pool(processes=n_processes) as pool:
                results = pool.map(_run_monte_carlo_chunk, tasks)
        
        # Aggregate results from all processes
        n_above = np.zeros(len(obs_totals), dtype=np.int64)
        for result in results:
            n_above += result
    else:
        # Single-process execution (fallback for small jobs or n_processes=1)
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
    
    # Calculate p-values
    if progress:
        print("Calculating p-values...")
        print(f"  n_above statistics: min={n_above.min()}, max={n_above.max()}, mean={n_above.mean():.2f}")
        
        # Validation: Check if results make sense
        pct_all_extreme = 100 * np.sum(n_above == niters) / len(n_above)
        pct_none_extreme = 100 * np.sum(n_above == 0) / len(n_above)
        print(f"  Cells where ALL {niters} simulations were more extreme: {pct_all_extreme:.1f}%")
        print(f"  Cells where NO simulations were more extreme: {pct_none_extreme:.1f}%")
        
        if pct_all_extreme > 80:
            print("  ⚠️  WARNING: >80% cells have all simulations more extreme - likely a bug!")
        elif pct_none_extreme > 20:
            print("  ⚠️  WARNING: >20% cells have no extreme simulations - likely very different from ambient (good!)")
        else:
            print("  ✅ Distribution looks reasonable!")
        print(f"  Number of cells with n_above=0: {(n_above == 0).sum()}")
        print(f"  Number of cells with n_above={niters}: {(n_above == niters).sum()}")
    
    pvals = (n_above + 1) / (niters + 1)
    limited = n_above == 0
    
    if progress:
        print(f"  P-value statistics: min={pvals.min():.6f}, max={pvals.max():.6f}, mean={pvals.mean():.6f}")
        print(f"  P-values = {1/(niters+1):.6f} (minimum possible): {(pvals == 1/(niters+1)).sum()}")
    
    # Create a full DataFrame with all barcodes
    full_results = pd.DataFrame(index=data.obs_names)
    full_results['Total'] = totals
    full_results['IsEmpty'] = empty_mask
    
    # Initialize columns with NaN
    full_results['LogProb'] = np.nan
    full_results['PValue'] = np.nan
    full_results['FDR'] = np.nan
    full_results['Limited'] = np.nan
    
    # Fill in results for tested cells
    test_barcodes = data[test_mask].obs_names
    full_results.loc[test_barcodes, 'LogProb'] = obs_probs
    full_results.loc[test_barcodes, 'PValue'] = pvals
    full_results.loc[test_barcodes, 'Limited'] = limited
    
    # Apply retain threshold BEFORE FDR correction (following R implementation)
    # This ensures large cells are always retained, even if they look similar to ambient
    if retain is not None:
        retain_mask = totals >= retain
        # Set p-values to 0 for high-count droplets (following R code line 405)
        # This happens BEFORE FDR correction so these don't affect other cells' FDR
        full_results.loc[data[retain_mask].obs_names, 'PValue'] = 0
        if progress:
            n_retained = retain_mask.sum()
            print(f"Set PValue=0 for {n_retained} high-count droplets (>= {retain:.1f}) before FDR correction")
    
    # Visualize barcode ranks and knee point
    if visualize and retain is not None:
        try:
            # Calculate ranks directly (no need for barcode_ranks function)
            sorted_indices = np.argsort(totals)[::-1]  # Descending order
            ranks = np.empty_like(totals, dtype=float)
            ranks[sorted_indices] = np.arange(1, len(totals) + 1)  # 1-based ranking
            
            # Use the knee point we already calculated
            knee_point = retain
            
            plt.figure(figsize=(12, 8))
            
            # Create rank plot
            plt.subplot(2, 2, 1)
            plt.loglog(ranks, totals, 'b.', alpha=0.6, markersize=1)
            plt.axhline(y=knee_point, color='red', linestyle='--', label=f'Knee ({knee_point:.0f})')
            plt.axhline(y=lower, color='orange', linestyle=':', label=f'Lower ({lower})')
            plt.xlabel('Rank')
            plt.ylabel('Total UMI Counts')
            plt.title('Barcode Rank Plot')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Create histogram with thresholds
            plt.subplot(2, 2, 2)
            plt.hist(totals, bins=100, alpha=0.7, log=True)
            plt.axvline(x=lower, color='orange', linestyle=':', label=f'Lower ({lower})')
            plt.axvline(x=knee_point, color='red', linestyle='--', label=f'Knee/Retain ({knee_point:.0f})')
            plt.xlabel('Total UMI Counts')
            plt.ylabel('Frequency')
            plt.title('UMI Count Distribution')
            plt.legend()
            plt.yscale('log')
            
            # Zoom in on the knee region
            plt.subplot(2, 2, 3)
            knee_region = (totals >= lower) & (totals <= knee_point * 10)
            if np.any(knee_region):
                plt.loglog(ranks[knee_region], totals[knee_region], 'b.', alpha=0.8, markersize=2)
                plt.axhline(y=knee_point, color='red', linestyle='--', label=f'Knee ({knee_point:.0f})')
                plt.xlabel('Rank')
                plt.ylabel('Total UMI Counts')
                plt.title('Knee Region (Zoomed)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Show cumulative distribution
            plt.subplot(2, 2, 4)
            sorted_totals = np.sort(totals)[::-1]
            cumulative = np.arange(1, len(sorted_totals) + 1)
            plt.loglog(cumulative, sorted_totals, 'b-', alpha=0.8)
            plt.axhline(y=knee_point, color='red', linestyle='--', label=f'Knee ({knee_point:.0f})')
            plt.xlabel('Cumulative Barcodes')
            plt.ylabel('Total UMI Counts')
            plt.title('Cumulative Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'visualizations', 'barcode_ranks_knee_detection.png'), dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Barcode ranks and knee detection plot saved to '{os.path.join(output_dir, 'visualizations', 'barcode_ranks_knee_detection.png')}'")
            
        except Exception as e:
            if progress:
                print(f"Could not create barcode ranks visualization: {e}")
    
    # Apply FDR correction only to cells that were tested
    tested_mask = ~full_results['PValue'].isna()
    if tested_mask.any():
        fdr = multipletests(full_results.loc[tested_mask, 'PValue'], method='fdr_bh')[1]
        full_results.loc[tested_mask, 'FDR'] = fdr
    
    # Add significance indicator columns at common thresholds
    full_results['FDR_0.001'] = full_results['FDR'] <= 0.001
    full_results['FDR_0.01'] = full_results['FDR'] <= 0.01
    full_results['FDR_0.05'] = full_results['FDR'] <= 0.05
    
    # Print summary statistics
    if progress:
        print("\n=== Significance Summary ===")
        print(f"Cells with FDR <= 0.001: {full_results['FDR_0.001'].sum()}")
        print(f"Cells with FDR <= 0.01:  {full_results['FDR_0.01'].sum()}")
        print(f"Cells with FDR <= 0.05:  {full_results['FDR_0.05'].sum()}")
        print(f"Cells with FDR > 0.05:   {(full_results['FDR'] > 0.05).sum()}")
        
        # Check p-value distribution for diagnostic
        valid_pvals = full_results.loc[tested_mask, 'PValue']
        print(f"\n=== P-value Distribution ===")
        print(f"Min p-value: {valid_pvals.min():.6f}")
        print(f"Max p-value: {valid_pvals.max():.6f}")
        print(f"Mean p-value: {valid_pvals.mean():.6f}")
        print(f"Median p-value: {valid_pvals.median():.6f}")
        print(f"P-values == 0: {(valid_pvals == 0).sum()}")
        print(f"P-values < 0.001: {(valid_pvals < 0.001).sum()}")
        print(f"P-values < 0.01: {(valid_pvals < 0.01).sum()}")
        print(f"P-values < 0.05: {(valid_pvals < 0.05).sum()}")
    
    # Visualize p-value distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(full_results.loc[tested_mask, 'PValue'], bins=50)
        plt.title('Distribution of P-values')
        plt.xlabel('P-value')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'visualizations', 'p_values.png'))
        plt.close()
        print(f"P-value distribution plot saved to '{os.path.join(output_dir, 'visualizations', 'p_values.png')}'")
    
    # Visualize FDR distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(full_results.loc[tested_mask, 'FDR'], bins=50)
        plt.title('Distribution of FDR')
        plt.xlabel('FDR')
        plt.ylabel('Count')
        plt.savefig(os.path.join(output_dir, 'visualizations', 'fdr.png'))
        plt.close()
        print(f"FDR distribution plot saved to '{os.path.join(output_dir, 'visualizations', 'fdr.png')}'")
    
    # Create scatter plot of FDR vs Total UMI Counts
    full_results['FDR < 0.05'] = full_results['FDR'] < 0.05
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_results[tested_mask], x='Total', y='FDR', 
                   hue='FDR < 0.05', palette=['red', 'blue'], alpha=0.5)
    plt.title('FDR vs Total UMI Counts')
    plt.xlabel('Total UMI Counts')
    plt.ylabel('False Discovery Rate')
    plt.savefig(os.path.join(output_dir, 'visualizations', 'final_results.png'))
    plt.close()
    
    # Save results to organized directory
    try:
        # Save main results
        results_path = os.path.join(output_dir, 'results', 'emptydrops_results.csv')
        full_results.to_csv(results_path, index=True)
        print(f"Results saved to: {results_path}")
        
        # Save metadata (convert numpy types to native Python types for JSON serialization)
        metadata = {
            'dataset_name': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': {
                'lower': int(lower),
                'retain': int(retain) if retain is not None else None,
                'niters': int(niters),
                'alpha': float(alpha) if alpha != np.inf else 'inf',
                'test_ambient': bool(test_ambient),
                'batch_size': int(batch_size),
                'n_processes': int(n_processes),
                'use_gpu': bool(use_gpu),
                'knee_point_parameters': knee_parameters  # Add knee point parameters
            },
            'data_info': {
                'n_obs': int(data.n_obs),
                'n_vars': int(data.n_vars),
                'data_hash': str(data_hash)
            },
            'results_summary': {
                'total_barcodes': int(len(full_results)),
                'tested_barcodes': int(tested_mask.sum()),
                'cells_above_lower': int((totals > lower).sum()),
                'cells_above_retain': int((totals >= retain).sum()),
                'significant_cells_fdr_0.001': int((full_results['FDR'] <= 0.001).sum()),
                'significant_cells_fdr_0.01': int((full_results['FDR'] <= 0.01).sum()),
                'significant_cells_fdr_0.05': int((full_results['FDR'] <= 0.05).sum()),
                'runtime_seconds': float(time.time() - start_time)
            }
        }
        
        metadata_path = os.path.join(output_dir, 'metadata', 'run_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to: {metadata_path}")
        
        # Save ambient profile if available
        if 'ambient_proportions' in locals():
            ambient_path = os.path.join(output_dir, 'results', 'ambient_profile.csv')
            ambient_df = pd.DataFrame({
                'gene': data.var_names,
                'proportion': ambient_proportions
            })
            ambient_df.to_csv(ambient_path, index=False)
            print(f"Ambient profile saved to: {ambient_path}")
            
    except Exception as e:
        print(f"Warning: Could not save results to directory: {e}")
    
    # Cache the results
    if use_cache:
        _save_to_cache(cache_key, {'results': full_results})
    
    if progress:
        total_time = time.time() - start_time
        print(f"EmptyDrops analysis complete! Total time: {format_time(total_time)}")
        print(f"All outputs saved to: {output_dir}")
    
    return full_results

def estimate_alpha(
    mat: Union[np.ndarray, spmatrix],
    prop: np.ndarray,
    totals: np.ndarray,
    interval: Tuple[float, float] = (0.01, 10000),
    progress: bool = True
) -> float:
    """
    Estimate the Dirichlet-multinomial alpha parameter.
    
    Parameters
    ----------
    mat : array-like or sparse matrix
        The count data
    prop : array-like
        The proportion vector
    totals : array-like
        The total counts per barcode
    interval : tuple, optional (default: (0.01, 10000))
        The interval to search for alpha
    progress : bool, optional (default: True)
        Whether to show progress bars
        
    Returns
    -------
    float
        The estimated alpha parameter
    """
    if issparse(mat):
        mat = mat.tocsr()
    
    def loglik(alpha):
        prop_alpha = prop * alpha
        return (gammaln(alpha) * len(totals) -
                np.sum(gammaln(totals + alpha)) +
                np.sum(gammaln(mat.data + prop_alpha)) -
                np.sum(gammaln(prop_alpha)))
    
    # Use multiple starting points for better optimization
    best_alpha = None
    best_loglik = -np.inf
    starting_points = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    if progress:
        pbar = TimeEstimator(len(starting_points), "Optimizing alpha")
    
    for i, start in enumerate(starting_points):
        result = minimize_scalar(
            lambda x: -loglik(x),
            bounds=interval,
            method='bounded',
            x0=start
        )
        
        if -result.fun > best_loglik:
            best_loglik = -result.fun
            best_alpha = result.x
            
        if progress:
            pbar.update(1)
            
            # Print estimated completion time every 2 starting points
            if (i + 1) % 2 == 0:
                elapsed = pbar.get_elapsed_time()
                rate = (i + 1) / elapsed
                remaining = (len(starting_points) - (i + 1)) / rate
                print(f"\nEstimated time remaining: {format_time(remaining)}")
    
    if progress:
        pbar.close()
    
    return best_alpha 