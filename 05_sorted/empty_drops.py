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
from datetime import timedelta
import os
import pickle
import hashlib
import matplotlib.pyplot as plt
import seaborn as sns
import psutil
import gc
# Only import as fallback if new implementation fails
try:
    from empty_drops_cython.empty_drops_core import permute_counter as cython_permute_counter
    HAS_CYTHON = True
except ImportError:
    HAS_CYTHON = False

# GPU availability check
try:
    from numba import cuda
    HAS_GPU = cuda.is_available()
except ImportError:
    HAS_GPU = False

# Optimized constants for M-series chips
CHUNK_SIZE = 100  # Increased for better vectorization
MAX_WORKERS = mp.cpu_count()  # Use all available cores
N_PROCESSES = mp.cpu_count()  # Use all cores
MAX_MEMORY_PERCENT = 80  # Slightly increased but still safe
COOLING_PERIOD = 1  # Reduced cooling period
MAX_CPU_PERCENT = 90  # Increased CPU usage threshold
SAVE_FREQUENCY = 5  # More frequent checkpoints

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
    temp_file = f"intermediate_{key}.pkl"
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
    
    # Compute log probabilities for each row
    for i in range(data_chunk.shape[0]):
        row_sum = 0.0
        for j in range(data_chunk.shape[1]):
            if data_chunk[i, j] > 0:
                log_probs[i] += _numba_gammaln_scalar(data_chunk[i, j] + 1)
                row_sum += data_chunk[i, j]
        
        # Add log probability of proportions
        if alpha > 0:
            log_probs[i] += np.log(prop).dot(data_chunk[i])
        else:
            log_probs[i] += np.log(prop + 1e-10).dot(data_chunk[i])
    
    return log_probs

def _process_chunk_mp(args):
    """Process a chunk of data using multiprocessing."""
    chunk, prop, alpha = args
    return _compute_multinom_prob_chunk(chunk, prop, alpha)

class ProcessingRateMonitor:
    """Monitor processing rate and predict remaining time."""
    def __init__(self, total_size_mb):
        self.start_time = time.time()
        self.total_size_mb = total_size_mb
        self.processed_mb = 0
        self.last_update = self.start_time
        self.rate_mb_s = 0
        
    def update(self, size_mb):
        current_time = time.time()
        elapsed = current_time - self.last_update
        self.processed_mb += size_mb
        
        # Update processing rate (MB/s)
        if elapsed > 0:
            self.rate_mb_s = size_mb / elapsed
        
        self.last_update = current_time
        
    def get_stats(self):
        if self.rate_mb_s > 0:
            remaining_mb = self.total_size_mb - self.processed_mb
            remaining_seconds = remaining_mb / self.rate_mb_s
            return {
                'rate_mb_s': self.rate_mb_s,
                'processed_mb': self.processed_mb,
                'total_mb': self.total_size_mb,
                'remaining_seconds': remaining_seconds,
                'percent_complete': (self.processed_mb / self.total_size_mb) * 100
            }
        return None

def debug_mp_info():
    """Print debug information about multiprocessing setup."""
    print("\nMultiprocessing Debug Info:")
    print(f"Number of CPU cores available: {mp.cpu_count()}")
    print(f"Number of processes to be used: {N_PROCESSES}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print("Process Pool starting...\n")

def compute_multinom_prob(
    data: Union[np.ndarray, spmatrix],
    prop: np.ndarray,
    alpha: float = np.inf,
    progress: bool = True
) -> np.ndarray:
    """Calculate multinomial probabilities using multiprocessing."""
    if issparse(data):
        data = data.tocsr().toarray()
    
    if progress:
        debug_mp_info()
    
    # Calculate total data size in MB
    data_size_mb = data.nbytes / (1024 * 1024)
    rate_monitor = ProcessingRateMonitor(data_size_mb)
    
    # Split data into chunks
    n_chunks = max(1, len(data) // CHUNK_SIZE)
    chunks = np.array_split(data, n_chunks)
    chunk_size_mb = data_size_mb / n_chunks
    
    if progress:
        print(f"Total data size: {data_size_mb:.2f} MB")
        print(f"Number of chunks: {n_chunks}")
        print(f"Chunk size: {chunk_size_mb:.2f} MB\n")
    
    # Prepare arguments for multiprocessing
    chunk_args = [(chunk, prop, alpha) for chunk in chunks]
    
    if progress:
        pbar = tqdm(total=n_chunks, desc="Computing probabilities", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    # Process chunks using multiprocessing
    with mp.Pool(processes=N_PROCESSES) as pool:
        if progress:
            print(f"Process pool created with {N_PROCESSES} workers")
        
        results = []
        for i, result in enumerate(pool.imap(_process_chunk_mp, chunk_args)):
            results.append(result)
            if progress:
                rate_monitor.update(chunk_size_mb)
                stats = rate_monitor.get_stats()
                if stats:
                    pbar.set_postfix({
                        'Rate': f"{stats['rate_mb_s']:.1f} MB/s",
                        'Remaining': f"{stats['remaining_seconds']/60:.1f}min",
                        'Active_Processes': N_PROCESSES
                    })
                pbar.update(1)
                
                # Print periodic process status
                if i % 10 == 0:
                    print(f"\nProcessed {i}/{n_chunks} chunks using {N_PROCESSES} processes")
    
    if progress:
        pbar.close()
        print("\nMultiprocessing completed")
    
    return np.concatenate(results)

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

# Optimized multiprocessing Monte Carlo implementation
@jit(nopython=True, parallel=False)  # Disable parallel for multiprocessing
def _simulate_multinomial_batch_numba(totals, ambient_props, n_iter, seed_offset):
    """Numba-optimized multinomial simulation for a batch of cells."""
    np.random.seed(seed_offset)
    n_cells = len(totals)
    results = np.zeros((n_cells, n_iter), dtype=np.float64)
    
    for i in range(n_cells):
        if totals[i] <= 0:
            continue
        for j in range(n_iter):
            # Simulate multinomial and sum
            sim_counts = np.random.multinomial(int(totals[i]), ambient_props)
            results[i, j] = sim_counts.sum()
    
    return results

def _mc_worker(args):
    """Optimized worker function for multiprocessing Monte Carlo simulation."""
    cell_indices, totals_batch, obs_probs_batch, ambient_props, alpha, n_iter, seed_offset = args
    
    n_cells = len(totals_batch)
    n_above = np.zeros(n_cells, dtype=np.int64)
    
    # Set random seed for reproducibility
    np.random.seed(seed_offset)
    
    # Pre-filter cells with very low counts (optimization)
    valid_mask = totals_batch > 0
    valid_cells = np.where(valid_mask)[0]
    
    if len(valid_cells) == 0:
        return cell_indices, n_above
    
    # Progress tracking for this worker
    cells_processed = 0
    
    # Vectorized simulation approach
    # Instead of calling multinomial for each iteration, we approximate using Poisson
    # This is much faster and gives very similar results for large counts
    
    for i in valid_cells:
        total_i = totals_batch[i]
        obs_prob_i = obs_probs_batch[i]
        
        # Adaptive iteration count based on cell total
        if total_i <= 10:
            actual_iters = min(n_iter, 50)  # Very low counts need fewer iterations
        elif total_i <= 100:
            actual_iters = min(n_iter, 200)  # Low counts
        elif total_i <= 1000:
            actual_iters = min(n_iter, 500)  # Medium counts
        else:
            actual_iters = n_iter  # High counts get full iterations
        
        if total_i <= 10:  # Use exact multinomial for very low counts
            for j in range(actual_iters):
                sim_counts = np.random.multinomial(int(total_i), ambient_props)
                sim_prob = _compute_multinom_prob_single(sim_counts, ambient_props, alpha)
                if sim_prob >= obs_prob_i:
                    n_above[i] += 1
            # Scale up the result if needed
            if actual_iters < n_iter:
                n_above[i] = int(n_above[i] * n_iter / actual_iters)
        else:
            # Fast Poisson approximation for larger counts
            # Sample from Poisson(total * prop_j) for each gene j
            expected_counts = total_i * ambient_props
            
            # Vectorized Poisson sampling with adaptive batch size
            batch_size = min(actual_iters, 500)  # Process in smaller batches
            n_above[i] = 0
            
            for batch_start in range(0, actual_iters, batch_size):
                batch_end = min(batch_start + batch_size, actual_iters)
                current_batch_size = batch_end - batch_start
                
                sim_counts_matrix = np.random.poisson(
                    expected_counts[None, :], 
                    size=(current_batch_size, len(ambient_props))
                )
                
                # Calculate probabilities efficiently for this batch
                for j in range(current_batch_size):
                    sim_total = sim_counts_matrix[j].sum()
                    if sim_total > 0:
                        sim_prob = _compute_multinom_prob_single(sim_counts_matrix[j], ambient_props, alpha)
                        if sim_prob >= obs_prob_i:
                            n_above[i] += 1
            
            # Scale up if we used fewer iterations
            if actual_iters < n_iter:
                n_above[i] = int(n_above[i] * n_iter / actual_iters)
        
        cells_processed += 1
        # Print progress every 100 cells
        if cells_processed % 100 == 0:
            print(f"Worker {seed_offset}: processed {cells_processed}/{len(valid_cells)} cells")
    
    return cell_indices, n_above

@jit(nopython=True)
def _compute_multinom_prob_single(counts, props, alpha):
    """Compute multinomial probability for a single count vector."""
    log_prob = 0.0
    
    # Add log factorial terms
    for i in range(len(counts)):
        if counts[i] > 0:
            log_prob += _numba_gammaln_scalar(counts[i] + 1)
    
    # Add log probability terms
    for i in range(len(counts)):
        if counts[i] > 0 and props[i] > 0:
            log_prob += counts[i] * np.log(props[i])
    
    return log_prob

def permute_counter_mp(
    totals: np.ndarray,
    obs_probs: np.ndarray,
    ambient_props: np.ndarray,
    niters: int,
    alpha: float,
    progress: bool = True,
    adaptive: bool = True,
    min_iters: int = 100,
    max_iters: int = 1000,
    early_stopping: bool = True,
    batch_size: int = 100,
    n_processes: int = None,
    fast_mode: bool = False
) -> np.ndarray:
    """
    Multiprocessing-optimized Monte Carlo permutation testing.
    
    This implementation uses true multiprocessing to achieve high CPU utilization
    by distributing cells across multiple processes.
    
    Parameters
    ----------
    totals : np.ndarray
        Total UMI counts per cell
    obs_probs : np.ndarray
        Observed log-probabilities
    ambient_props : np.ndarray
        Ambient RNA profile proportions
    niters : int
        Number of Monte Carlo iterations
    alpha : float
        Dirichlet concentration parameter
    progress : bool
        Whether to show progress bar
    adaptive : bool
        Whether to use adaptive stopping (currently not implemented)
    min_iters : int
        Minimum iterations before early stopping
    max_iters : int
        Maximum iterations
    early_stopping : bool
        Whether to use early stopping
    batch_size : int
        Size of batches for processing
    n_processes : int
        Number of processes to use (default: all available cores)
    fast_mode : bool
        If True, use extremely fast approximation (10x faster but less accurate)
        
    Returns
    -------
    np.ndarray
        Number of permutations with higher probabilities for each cell
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    n_cells = len(totals)
    
    # Fast mode: drastically reduce iterations for speed
    if fast_mode:
        original_niters = niters
        niters = min(niters, 50)  # Use max 50 iterations in fast mode
        if progress:
            print(f"\nFAST MODE: Using {niters} iterations instead of {original_niters} for speed")
    
    if progress:
        print(f"\nStarting multiprocessing Monte Carlo with {n_processes} processes")
        print(f"Processing {n_cells} cells with {niters} iterations each")
        if fast_mode:
            print("FAST MODE ENABLED: Results will be approximate but much faster")
    
    # Split cells across processes
    cells_per_process = max(1, n_cells // n_processes)
    
    # Prepare arguments for each process
    process_args = []
    for proc_id in range(n_processes):
        start_idx = proc_id * cells_per_process
        end_idx = min(start_idx + cells_per_process, n_cells)
        
        if start_idx >= n_cells:
            break
            
        cell_indices = np.arange(start_idx, end_idx)
        totals_batch = totals[start_idx:end_idx]
        obs_probs_batch = obs_probs[start_idx:end_idx]
        seed_offset = proc_id * 12345 + int(time.time()) % 100000
        
        process_args.append((
            cell_indices,
            totals_batch,
            obs_probs_batch,
            ambient_props,
            alpha,
            niters,
            seed_offset
        ))
    
    # Initialize result array
    n_above = np.zeros(n_cells, dtype=np.int64)
    
    # Run multiprocessing
    if progress:
        pbar = tqdm(total=len(process_args), desc="Monte Carlo processes",
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}')
    
    start_time = time.time()
    
    with mp.Pool(processes=n_processes, maxtasksperchild=1) as pool:
        try:
            # Submit all jobs
            futures = [pool.apply_async(_mc_worker, (args,)) for args in process_args]
            
            # Collect results
            for i, future in enumerate(futures):
                cell_indices, process_n_above = future.get(timeout=3600)  # 1 hour timeout
                n_above[cell_indices] = process_n_above
                
                if progress:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({
                        'Rate': f"{rate:.2f} proc/s",
                        'CPU_Usage': f"{psutil.cpu_percent()}%",
                        'Memory': f"{psutil.virtual_memory().percent:.1f}%"
                    })
                    pbar.update(1)
                    
        except Exception as e:
            print(f"Error in multiprocessing: {e}")
            # Fallback to single process
            if progress:
                print("Falling back to single process...")
            for args in process_args:
                cell_indices, process_n_above = _mc_worker(args)
                n_above[cell_indices] = process_n_above
    
    if progress:
        pbar.close()
        total_time = time.time() - start_time
        print(f"Monte Carlo completed in {format_time(total_time)}")
        print(f"Average processing rate: {len(process_args) / total_time:.2f} processes/second")
    
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
    batch_size: int = CHUNK_SIZE,
    visualize: bool = True,
    confidence_level: float = 0.95,
    n_processes: int = N_PROCESSES,
    use_gpu: bool = HAS_GPU,
    fast_mode: bool = False
) -> pd.DataFrame:
    """
    Enhanced EmptyDrops implementation with performance optimizations.
    """
    start_time = time.time()
    
    if progress:
        print("Starting EmptyDrops analysis...")
    

    
    # Create directory for visualizations if needed
    if visualize:
        os.makedirs('empty_drops_visualizations', exist_ok=True)
    
    # Filter genes with zero counts
    if progress:
        print("Filtering genes...")
    print(f"{(data.X.sum(axis=0).A1 == 0).sum()} genes filtered out since sum(counts) over the gene was 0.")
    sc.pp.filter_genes(data, min_counts=1)
    
    # Get total counts per barcode
    if progress:
        print("Calculating total counts per barcode...")
    totals = data.X.sum(axis=1).A1
    
    # Visualize total counts distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(totals, bins=100, log_scale=True)
        plt.title('Distribution of Total UMI Counts')
        plt.xlabel('Total UMI Counts (log scale)')
        plt.ylabel('Count')
        plt.savefig('empty_drops_visualizations/total_counts_distribution.png')
        plt.close()
        print("Total counts distribution plot saved to 'empty_drops_visualizations/total_counts_distribution.png'")
    
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
        plt.savefig('empty_drops_visualizations/empty_droplet_threshold.png')
        plt.close()
        print("Empty droplet threshold plot saved to 'empty_drops_visualizations/empty_droplet_threshold.png'")
    
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
        plt.savefig('empty_drops_visualizations/ambient_profile.png')
        plt.close()
        print("Ambient profile plot saved to 'empty_drops_visualizations/ambient_profile.png'")
    
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
    
    # Calculate probabilities for non-empty droplets
    if progress:
        print("Calculating probabilities for non-empty droplets...")
    non_empty_mask = ~empty_mask
    if ignore is not None:
        non_empty_mask &= totals > ignore
        
    obs_data = data[non_empty_mask]
    obs_totals = totals[non_empty_mask]
    
    # Calculate multinomial probabilities
    obs_probs = compute_multinom_prob(
        obs_data.X,
        ambient_proportions,
        alpha,
        progress=progress
    )
    
    # Visualize log probabilities
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(obs_probs, bins=50)
        plt.title('Distribution of Log Probabilities')
        plt.xlabel('Log Probability')
        plt.ylabel('Count')
        plt.savefig('empty_drops_visualizations/log_probabilities.png')
        plt.close()
        print("Log probabilities plot saved to 'empty_drops_visualizations/log_probabilities.png'")
    
    # Convert arrays to correct types for Cython function
    print("\nConverting arrays to correct types...")
    print(f"obs_totals dtype before: {obs_totals.dtype}")
    obs_totals = obs_totals.astype(np.int64)
    print(f"obs_totals dtype after: {obs_totals.dtype}")
    
    print(f"obs_probs dtype before: {obs_probs.dtype}")
    obs_probs = obs_probs.astype(np.float64)
    print(f"obs_probs dtype after: {obs_probs.dtype}")
    
    print(f"ambient_proportions dtype before: {ambient_proportions.dtype}")
    ambient_proportions = ambient_proportions.astype(np.float64)
    print(f"ambient_proportions dtype after: {ambient_proportions.dtype}")
    print(f"ambient_proportions shape: {ambient_proportions.shape}")
    print(f"ambient_proportions sum: {ambient_proportions.sum()}")
    
    # Perform Monte Carlo testing
    if progress:
        print("\nPerforming Monte Carlo testing...")
    
    # Use optimized multiprocessing implementation for better CPU utilization
    try:
        n_above = permute_counter_mp(
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
            batch_size=batch_size,
            n_processes=n_processes,
            fast_mode=fast_mode
        )
    except Exception as e:
        if progress:
            print(f"Multiprocessing failed: {e}")
        if HAS_CYTHON:
            if progress:
                print("Falling back to Cython implementation...")
            n_above = cython_permute_counter(
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
        else:
            raise RuntimeError("Neither multiprocessing nor Cython implementation available")
    
    # Calculate p-values
    if progress:
        print("Calculating p-values and FDR...")
    pvals = (n_above + 1) / (niters + 1)
    limited = n_above == 0
    
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
    non_empty_barcodes = data[non_empty_mask].obs_names
    full_results.loc[non_empty_barcodes, 'LogProb'] = obs_probs
    full_results.loc[non_empty_barcodes, 'PValue'] = pvals
    full_results.loc[non_empty_barcodes, 'Limited'] = limited
    
    # Handle retain threshold
    if retain is not None:
        retain_mask = totals >= retain
        full_results.loc[data[retain_mask].obs_names, 'PValue'] = 0
    
    # Apply FDR correction only to cells that were tested
    tested_mask = ~full_results['PValue'].isna()
    if tested_mask.any():
        fdr = multipletests(full_results.loc[tested_mask, 'PValue'], method='fdr_bh')[1]
        full_results.loc[tested_mask, 'FDR'] = fdr
    
    # Visualize p-value distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(full_results.loc[tested_mask, 'PValue'], bins=50)
        plt.title('Distribution of P-values')
        plt.xlabel('P-value')
        plt.ylabel('Count')
        plt.savefig('empty_drops_visualizations/p_values.png')
        plt.close()
        print("P-value distribution plot saved to 'empty_drops_visualizations/p_values.png'")
    
    # Visualize FDR distribution
    if visualize:
        plt.figure(figsize=(10, 6))
        sns.histplot(full_results.loc[tested_mask, 'FDR'], bins=50)
        plt.title('Distribution of FDR')
        plt.xlabel('FDR')
        plt.ylabel('Count')
        plt.savefig('empty_drops_visualizations/fdr.png')
        plt.close()
        print("FDR distribution plot saved to 'empty_drops_visualizations/fdr.png'")
    
    # Create scatter plot of FDR vs Total UMI Counts
    full_results['FDR < 0.05'] = full_results['FDR'] < 0.05
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=full_results[tested_mask], x='Total', y='FDR', 
                   hue='FDR < 0.05', palette=['red', 'blue'], alpha=0.5)
    plt.title('FDR vs Total UMI Counts')
    plt.xlabel('Total UMI Counts')
    plt.ylabel('False Discovery Rate')
    plt.savefig('empty_drops_visualizations/final_results.png')
    plt.close()
    
    # Create gene expression distribution plots for PRM1 and TNP1
    if visualize:
        try:
            _create_gene_expression_plots(data, full_results, genes=['PRM1', 'TNP1'])
        except Exception as e:
            if progress:
                print(f"Warning: Could not create gene expression plots: {e}")

    
    if progress:
        total_time = time.time() - start_time
        print(f"EmptyDrops analysis complete! Total time: {format_time(total_time)}")
    
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

def _create_gene_expression_plots(data, emptydrops_results, genes=['PRM1', 'TNP1'], 
                                 fdr_thresholds=[0.05]):
    """
    Create gene expression distribution plots by cell calling method.
    
    Parameters:
    -----------
    data : AnnData
        Raw single-cell data
    emptydrops_results : pd.DataFrame
        Results from EmptyDrops
    genes : list
        List of genes to analyze (default: ['PRM1', 'TNP1'])
    fdr_thresholds : list
        List of FDR thresholds to analyze (default: [0.001, 0.01, 0.05])
    """
    print("Creating gene expression distribution plots...")
    
    # Get total UMI counts per cell
    total_counts = np.array(data.X.sum(axis=1)).flatten()
    
    # Try to load filtered data for method comparison
    filtered_adata = None
    try:
        filtered_adata = sc.read_10x_h5("data/filtered_feature_bc_matrix.h5")
        print("Loaded filtered data for method comparison")
    except:
        print("Filtered data not available, creating EmptyDrops-only plots")
    
    # Analyze each gene for each FDR threshold
    for gene in genes:
        try:
            # Find gene index
            gene_idx = np.where(data.var_names == gene)[0]
            if len(gene_idx) == 0:
                print(f"Warning: Gene {gene} not found in dataset, skipping...")
                continue
            
            gene_idx = gene_idx[0]
            print(f"Found {gene} at index {gene_idx}")
            
            # Get gene counts
            gene_counts = np.array(data.X[:, gene_idx].todense()).flatten()
            
            # Calculate normalized expression (same as original plot)
            normalized_expr = gene_counts / (total_counts + 1e-10)
            
            for fdr_threshold in fdr_thresholds:
                try:
                    # Format FDR threshold for filename (remove decimal point)
                    fdr_str = str(fdr_threshold).replace('0.', '').replace('.', '')
                    if len(fdr_str) < 4:
                        fdr_str = fdr_str.ljust(4, '0')  # Pad with zeros if needed
                    
                    # Create classification categories if filtered data is available
                    cell_categories = None
                    if filtered_adata is not None:
                        cell_categories = pd.Series(index=data.obs_names, dtype='str')
                        
                        # Get Cell Ranger calls (filtered matrix)
                        cellranger_cells = set(filtered_adata.obs_names)
                        
                        # Get EmptyDrops calls using specified FDR threshold
                        emptydrops_cells = set(emptydrops_results[emptydrops_results['FDR'] < fdr_threshold].index)
                        
                        # Classify each cell into confusion matrix quadrants
                        for barcode in data.obs_names:
                            if barcode in cellranger_cells:
                                if barcode in emptydrops_cells:
                                    cell_categories[barcode] = 'Called by Both Methods'
                                else:
                                    cell_categories[barcode] = 'CellRanger Only'
                            else:
                                if barcode in emptydrops_cells:
                                    cell_categories[barcode] = 'EmptyDrops Only'
                                else:
                                    cell_categories[barcode] = 'Called by Neither'
                    
                    # Create the plot
                    plt.figure(figsize=(15, 10))
                    
                    if cell_categories is not None:
                        # Plot with method comparison (like original)
                        categories = ['Called by Both Methods', 'EmptyDrops Only', 'CellRanger Only', 'Called by Neither']
                        colors = ['green', 'blue', 'orange', 'red']
                        
                        for cat, color in zip(categories, colors):
                            # Create mask for current category
                            mask = cell_categories == cat
                            if mask.any():
                                # Get normalized expression for this category
                                expr = normalized_expr[mask]
                                # Only include cells with non-zero expression
                                nonzero_expr = expr[expr > 0]
                                
                                if len(nonzero_expr) > 0:
                                    sns.kdeplot(
                                        data=nonzero_expr,
                                        label=f'{cat} (n={mask.sum()}, expr={len(nonzero_expr)})',
                                        color=color,
                                        log_scale=True,
                                        linewidth=2
                                    )
                        
                        plt.title(f'{gene} Gene Expression Distribution by Cell Calling Method (FDR < {fdr_threshold})\n'
                                 f'Normalized by Total UMI Count', fontsize=14, pad=20)
                    else:
                        # Plot without method comparison (EmptyDrops only)
                        called_cells_mask = emptydrops_results['FDR'] < fdr_threshold
                        called_indices = emptydrops_results[called_cells_mask].index
                        
                        # Create mask for called vs not called
                        is_called = pd.Series(False, index=data.obs_names)
                        is_called[called_indices] = True
                        
                        for called, color, label in [(True, 'blue', 'Called by EmptyDrops'), 
                                                   (False, 'red', 'Not Called')]:
                            mask = is_called == called
                            if mask.any():
                                expr = normalized_expr[mask]
                                nonzero_expr = expr[expr > 0]
                                
                                if len(nonzero_expr) > 0:
                                    sns.kdeplot(
                                        data=nonzero_expr,
                                        label=f'{label} (n={mask.sum()}, expr={len(nonzero_expr)})',
                                        color=color,
                                        log_scale=True,
                                        linewidth=2
                                    )
                        
                        plt.title(f'{gene} Gene Expression Distribution (FDR < {fdr_threshold})\n'
                                 f'Normalized by Total UMI Count', fontsize=14, pad=20)
                    
                    plt.xlabel('Normalized Gene Expression (log scale)', fontsize=12)
                    plt.ylabel('Density', fontsize=12)
                    
                    # Add grid and legend
                    plt.grid(True, which='both', linestyle=':', alpha=0.3)
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    plt.tight_layout()
                    
                    # Save plot with FDR threshold in filename
                    plot_filename = f'{gene.lower()}_expression_distribution_fdr_{fdr_str}.png'
                    plot_path = os.path.join('empty_drops_visualizations', plot_filename)
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
                    plt.close()
                    
                    print(f"Saved {gene} expression plot (FDR < {fdr_threshold}) to {plot_path}")
                    
                    # Save statistics
                    stats_filename = f'{gene.lower()}_expression_statistics_fdr_{fdr_str}.txt'
                    stats_path = os.path.join('empty_drops_visualizations', stats_filename)
                    
                    with open(stats_path, 'w') as f:
                        f.write(f"{gene} EXPRESSION ANALYSIS (FDR < {fdr_threshold})\n")
                        f.write("=" * 50 + "\n\n")
                        
                        # Overall statistics
                        n_expressing = np.sum(gene_counts > 0)
                        f.write(f"Overall Statistics:\n")
                        f.write(f"  Total cells: {len(gene_counts):,}\n")
                        f.write(f"  Cells expressing {gene}: {n_expressing:,} ({n_expressing/len(gene_counts)*100:.1f}%)\n")
                        f.write(f"  Mean {gene} counts: {gene_counts.mean():.2f}\n")
                        f.write(f"  Mean normalized expression: {normalized_expr.mean():.2%}\n")
                        
                        if cell_categories is not None:
                            f.write(f"\nBy Cell Calling Method:\n")
                            for cat in categories:
                                mask = cell_categories == cat
                                if mask.any():
                                    cat_counts = gene_counts[mask]
                                    cat_expr = normalized_expr[mask]
                                    n_cells = mask.sum()
                                    n_expr = np.sum(cat_counts > 0)
                                    
                                    f.write(f"\n{cat}:\n")
                                    f.write(f"  Cells: {n_cells:,}\n")
                                    f.write(f"  Expressing {gene}: {n_expr:,} ({n_expr/n_cells*100:.1f}%)\n")
                                    f.write(f"  Mean {gene} counts: {cat_counts.mean():.2f}\n")
                                    f.write(f"  Mean normalized expression: {cat_expr.mean():.2%}\n")
                        else:
                            # EmptyDrops only analysis
                            called_mask = is_called
                            for called, label in [(True, 'Called by EmptyDrops'), (False, 'Not Called')]:
                                mask = called_mask == called
                                if mask.any():
                                    cat_counts = gene_counts[mask]
                                    cat_expr = normalized_expr[mask]
                                    n_cells = mask.sum()
                                    n_expr = np.sum(cat_counts > 0)
                                    
                                    f.write(f"\n{label}:\n")
                                    f.write(f"  Cells: {n_cells:,}\n")
                                    f.write(f"  Expressing {gene}: {n_expr:,} ({n_expr/n_cells*100:.1f}%)\n")
                                    f.write(f"  Mean {gene} counts: {cat_counts.mean():.2f}\n")
                                    f.write(f"  Mean normalized expression: {cat_expr.mean():.2%}\n")
                        
                        # Highlight target range for PRM1
                        if gene == 'PRM1':
                            target_range_mask = (normalized_expr >= 0.01) & (normalized_expr <= 0.06)  # 1-6%
                            n_in_target = np.sum(target_range_mask)
                            f.write(f"\nPRM1 Target Range Analysis (1-6% of total transcripts):\n")
                            f.write(f"  Cells in target range: {n_in_target:,} ({n_in_target/len(gene_counts)*100:.1f}%)\n")
                            
                            if cell_categories is not None:
                                # Analyze target range by method
                                for cat in categories:
                                    cat_mask = cell_categories == cat
                                    if cat_mask.any():
                                        cat_target = target_range_mask & cat_mask
                                        n_cat_target = np.sum(cat_target)
                                        n_cat_total = np.sum(cat_mask)
                                        f.write(f"  {cat}: {n_cat_target:,}/{n_cat_total:,} ({n_cat_target/n_cat_total*100:.1f}%)\n")
                            else:
                                # EmptyDrops only analysis for target range
                                called_target = target_range_mask & is_called
                                not_called_target = target_range_mask & ~is_called
                                f.write(f"  Called by EmptyDrops: {np.sum(called_target):,}/{np.sum(is_called):,} ({np.sum(called_target)/np.sum(is_called)*100:.1f}%)\n")
                                f.write(f"  Not Called: {np.sum(not_called_target):,}/{np.sum(~is_called):,} ({np.sum(not_called_target)/np.sum(~is_called)*100:.1f}%)\n")
                    
                    print(f"Saved {gene} statistics (FDR < {fdr_threshold}) to {stats_path}")
                    
                except Exception as e:
                    print(f"Error analyzing {gene} at FDR {fdr_threshold}: {str(e)}")
                    continue
                
        except Exception as e:
            print(f"Error analyzing gene {gene}: {str(e)}")
            continue
    
    print("Gene expression distribution analysis completed")