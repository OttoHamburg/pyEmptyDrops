"""
Performance optimizations for EmptyDrops algorithm.
This module contains highly optimized functions for computationally intensive operations.
"""

import numpy as np
from numba import jit, prange, cuda
from scipy.sparse import issparse, csr_matrix
from scipy.special import gammaln
import cupy as cp
from typing import Tuple, Optional
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import gc

# Constants for optimization
BLOCK_SIZE = 256  # CUDA block size
WARP_SIZE = 32   # Size of CUDA warp
CACHE_LINE_SIZE = 128  # CPU cache line size in bytes

@jit(nopython=True, parallel=True, fastmath=True)
def parallel_multinomial_prob(counts: np.ndarray, props: np.ndarray, alpha: float) -> np.ndarray:
    """
    Compute multinomial probabilities in parallel with SIMD optimization.
    """
    n_cells = counts.shape[0]
    log_probs = np.zeros(n_cells)
    
    for i in prange(n_cells):
        if alpha == np.inf:
            # Standard multinomial
            log_probs[i] = _fast_multinomial_prob(counts[i], props)
        else:
            # Dirichlet-multinomial
            log_probs[i] = _fast_dirichlet_multinomial_prob(counts[i], props, alpha)
    
    return log_probs

@jit(nopython=True, fastmath=True)
def _fast_multinomial_prob(counts: np.ndarray, props: np.ndarray) -> float:
    """
    Fast computation of multinomial probability using pre-computed values.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    
    log_prob = gammaln(total + 1)
    for i in range(len(counts)):
        if counts[i] > 0:
            log_prob += counts[i] * np.log(props[i]) - gammaln(counts[i] + 1)
    
    return log_prob

@jit(nopython=True, fastmath=True)
def _fast_dirichlet_multinomial_prob(counts: np.ndarray, props: np.ndarray, alpha: float) -> float:
    """
    Fast computation of Dirichlet-multinomial probability.
    """
    total = counts.sum()
    if total == 0:
        return 0.0
    
    log_prob = gammaln(total + 1) + gammaln(alpha)
    alpha_props = alpha * props
    
    for i in range(len(counts)):
        if counts[i] > 0:
            log_prob += (gammaln(counts[i] + alpha_props[i]) - 
                        gammaln(counts[i] + 1) - 
                        gammaln(alpha_props[i]))
    
    return log_prob

def optimize_sparse_matrix(X: csr_matrix) -> csr_matrix:
    """
    Optimize sparse matrix operations with memory alignment and cache efficiency.
    """
    if not issparse(X):
        return X
        
    # Convert to CSR format for efficient row operations
    X = X.tocsr()
    
    # Align data to cache line boundaries
    aligned_data = np.zeros(
        ((len(X.data) * X.data.itemsize + CACHE_LINE_SIZE - 1) // CACHE_LINE_SIZE) * CACHE_LINE_SIZE,
        dtype=X.data.dtype
    )
    aligned_data[:len(X.data)] = X.data
    X.data = aligned_data[:len(X.data)]
    
    # Sort indices for better memory access patterns
    X.sort_indices()
    
    # Remove duplicate entries
    X.sum_duplicates()
    
    return X

def parallel_monte_carlo(
    totals: np.ndarray,
    ambient_props: np.ndarray,
    n_iter: int,
    n_processes: Optional[int] = None,
    batch_size: int = 100
) -> np.ndarray:
    """
    Parallel Monte Carlo simulation using multiple processes.
    """
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Split iterations into batches
    n_batches = (n_iter + batch_size - 1) // batch_size
    batches = [(totals, ambient_props, min(batch_size, n_iter - i * batch_size)) 
               for i in range(n_batches)]
    
    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=n_processes) as executor:
        results = list(executor.map(lambda x: _simulate_batch(*x), batches))
    
    # Combine results
    return np.concatenate(results, axis=1)

if cuda.is_available():
    @cuda.jit
    def _cuda_monte_carlo_kernel(
        totals: np.ndarray,
        ambient_props: np.ndarray,
        results: np.ndarray,
        rng_states: np.ndarray
    ):
        """
        CUDA kernel for Monte Carlo simulation.
        """
        idx = cuda.grid(1)
        if idx < len(totals):
            # Use shared memory for ambient profile
            shared_props = cuda.shared.array(shape=(ambient_props.shape[0],), dtype=np.float64)
            if cuda.threadIdx.x == 0:
                for i in range(ambient_props.shape[0]):
                    shared_props[i] = ambient_props[i]
            cuda.syncthreads()
            
            # Generate random numbers using cuRAND
            for j in range(results.shape[1]):
                if totals[idx] > 0:
                    results[idx, j] = cuda.random.xoroshiro128p_uniform_float64(rng_states, idx)
    
    def gpu_monte_carlo(
        totals: np.ndarray,
        ambient_props: np.ndarray,
        n_iter: int,
        batch_size: int = 1000
    ) -> np.ndarray:
        """
        GPU-accelerated Monte Carlo simulation.
        """
        # Move data to GPU
        d_totals = cuda.to_device(totals)
        d_ambient = cuda.to_device(ambient_props)
        d_results = cuda.device_array((len(totals), n_iter), dtype=np.float64)
        
        # Initialize random number generator states
        rng_states = cuda.random.create_xoroshiro128p_states(
            BLOCK_SIZE * ((len(totals) + BLOCK_SIZE - 1) // BLOCK_SIZE),
            seed=np.random.randint(2**31)
        )
        
        # Launch kernel
        threadsperblock = (BLOCK_SIZE,)
        blockspergrid = ((len(totals) + BLOCK_SIZE - 1) // BLOCK_SIZE,)
        
        for i in range(0, n_iter, batch_size):
            batch_size_i = min(batch_size, n_iter - i)
            _cuda_monte_carlo_kernel[blockspergrid, threadsperblock](
                d_totals,
                d_ambient,
                d_results[:, i:i+batch_size_i],
                rng_states
            )
        
        # Copy results back to CPU
        results = d_results.copy_to_host()
        
        # Clean up GPU memory
        del d_totals
        del d_ambient
        del d_results
        del rng_states
        cuda.current_context().deallocations.clear()
        
        return results

def estimate_alpha_fast(
    counts: np.ndarray,
    ambient_props: np.ndarray,
    totals: np.ndarray,
    n_iter: int = 1000
) -> float:
    """
    Fast estimation of alpha parameter using parallel processing.
    """
    from scipy.optimize import minimize_scalar
    
    def objective(alpha: float) -> float:
        if alpha <= 0:
            return np.inf
        return -parallel_multinomial_prob(counts, ambient_props, alpha).sum()
    
    result = minimize_scalar(
        objective,
        bounds=(1e-10, 1e10),
        method='bounded',
        options={'maxiter': n_iter}
    )
    
    return result.x

class MemoryOptimizer:
    """
    Memory optimization utilities for large-scale computations.
    """
    @staticmethod
    def get_chunk_size(data_size: int, available_memory: int) -> int:
        """Calculate optimal chunk size based on available memory."""
        return max(1, available_memory // (data_size * 2))
    
    @staticmethod
    def clear_memory():
        """Force garbage collection and clear memory caches."""
        gc.collect()
        if cuda.is_available():
            cuda.current_context().deallocations.clear()
    
    @staticmethod
    def check_memory_usage() -> float:
        """Check current memory usage percentage."""
        import psutil
        return psutil.Process().memory_percent() 