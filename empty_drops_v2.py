import numpy as np
import pandas as pd
import scanpy as sc
from typing import Optional
import time
import os
import math
from statsmodels.stats.multitest import multipletests
from numba import njit, prange
import numba as nb
from tqdm.auto import tqdm
from scipy.interpolate import interp1d

# --- Core Helper Functions ---

@njit
def _gammaln_numba(x):
    """Use math.lgamma for accurate log-gamma calculation."""
    if x <= 0: return 0.0
    return math.lgamma(x)

@njit
def _log_multinomial_prob_numba_sparse(counts_indices, counts_data, total_count, ambient_props):
    log_prob = _gammaln_numba(total_count + 1)
    for i in range(len(counts_indices)):
        idx, count = counts_indices[i], counts_data[i]
        if ambient_props[idx] > 0:
            log_prob += count * np.log(ambient_props[idx]) - _gammaln_numba(count + 1)
        elif count > 0:
            return -np.inf
    return log_prob


@njit
def _monte_carlo_r_style(unique_totals, total_lengths, ordered_probs, ambient_props, niters, seed):
    """
    Monte Carlo simulation following the R implementation pattern.
    
    Fixed to be sequential to avoid parallel race conditions.
    """
    n_cells_total = np.sum(total_lengths)
    n_above = np.zeros(n_cells_total, dtype=np.int64)
    
    # Set seed once
    np.random.seed(seed)
    
    # Pre-compute log ambient props (avoid repeated log calls)
    log_ambient = np.log(np.maximum(ambient_props, 1e-12))
    
    # Pre-compute factorial terms for all unique totals
    log_factorials = np.zeros(len(unique_totals))
    for i in range(len(unique_totals)):
        if unique_totals[i] > 0:
            log_factorials[i] = _gammaln_numba(unique_totals[i] + 1)
    
    # Process each unique total count sequentially
    for total_idx in range(len(unique_totals)):
        total_count = unique_totals[total_idx]
        n_cells_with_total = total_lengths[total_idx]
        
        if total_count <= 0 or n_cells_with_total == 0:
            continue
            
        # Calculate cell range for this total
        cell_start_idx = np.sum(total_lengths[:total_idx])
        cell_end_idx = cell_start_idx + n_cells_with_total
        
        log_factorial_total = log_factorials[total_idx]
        
        # Run Monte Carlo iterations for this total count
        for i in range(niters):
            # Generate one multinomial sample for this total count
            sim_counts = np.random.multinomial(total_count, ambient_props)
            
            # Fast log-probability calculation with minimal operations
            sim_log_prob = log_factorial_total
            for k in range(len(sim_counts)):
                count = sim_counts[k]
                if count > 0:
                    sim_log_prob += count * log_ambient[k] - _gammaln_numba(count + 1)
            
            # Compare against all cells with this total count
            for cell_idx in range(cell_start_idx, cell_end_idx):
                if sim_log_prob <= ordered_probs[cell_idx]:
                    n_above[cell_idx] += 1
    
    return n_above

def good_turing_ambient_pool(data: sc.AnnData, low_count_gene_sums: np.ndarray):
    """
    Simplified ambient profile estimation matching R's approach.
    R appears to use simple proportions rather than complex Good-Turing.
    """
    gene_names = data.var_names.values
    counts = low_count_gene_sums.A1 if hasattr(low_count_gene_sums, 'A1') else low_count_gene_sums
    
    # Use simple proportional estimate like R
    total_obs_counts = np.sum(counts)
    ambient_props = counts / total_obs_counts
    
    # Apply R's .safe_good_turing logic: handle genes with zero probability
    still_zero = ambient_props <= 0
    if np.any(still_zero):
        pseudo_prob = 1.0 / total_obs_counts  # Like R's 1/sum(ambient.prof)
        n_zero = np.sum(still_zero)
        ambient_props[still_zero] = pseudo_prob / n_zero
        ambient_props[~still_zero] = ambient_props[~still_zero] * (1 - pseudo_prob)
        
    return gene_names, ambient_props, ambient_props * total_obs_counts

# --- Barcode Ranking and Knee Point Detection ---

def _find_curve_bounds_r_style(x, y, exclude_from=50):
    """
    Implement R's .find_curve_bounds function exactly.
    
    From R code:
    d1n <- diff(y)/diff(x)
    skip <- min(length(d1n) - 1, sum(x <= log10(exclude_from)))
    d1n <- tail(d1n, length(d1n) - skip)
    right.edge <- which.min(d1n)
    left.edge <- which.max(d1n[seq_len(right.edge)])
    c(left=left.edge, right=right.edge) + skip
    """
    # Step 1: Calculate first derivative
    d1n = np.diff(y) / np.diff(x)
    
    # Step 2: Calculate skip
    skip = min(len(d1n) - 1, np.sum(x <= np.log10(exclude_from)))
    
    # Step 3: Tail the derivative (remove first 'skip' elements)
    if skip > 0:
        d1n_tail = d1n[skip:]
    else:
        d1n_tail = d1n
    
    # Step 4: Find right edge (minimum derivative)
    right_edge_local = np.argmin(d1n_tail)
    
    # Step 5: Find left edge (maximum derivative up to right edge)
    left_candidates = d1n_tail[:right_edge_local + 1]  # seq_len(right.edge) in R
    if len(left_candidates) > 0:
        left_edge_local = np.argmax(left_candidates)
    else:
        left_edge_local = 0
    
    # Step 6: Add skip back (R: + skip)
    left_edge = left_edge_local + skip
    right_edge = right_edge_local + skip
    
    return {"left": left_edge, "right": right_edge}

def barcode_ranks(
    data: sc.AnnData,
    lower: int = 100,
    exclude_from: int = 50
):
    """
    Implement R's barcodeRanks function as closely as possible.
    
    This follows the R code step-by-step to ensure exact matching.
    """
    # Step 1: Get totals for all cells
    totals = np.asarray(data.X.sum(axis=1)).flatten().astype(int)
    
    # Step 2: Order by decreasing totals (R: o <- order(totals, decreasing=TRUE))
    o = np.argsort(-totals)  # Decreasing order
    
    # Step 3: Run-length encoding (R: stuff <- rle(totals[o]))
    ordered_totals = totals[o]
    
    # R's rle returns values in the order they appear, but np.unique sorts them
    # We need to preserve the order from the sorted array
    run_values = []
    run_lengths = []
    current_val = ordered_totals[0]
    current_count = 1
    
    for i in range(1, len(ordered_totals)):
        if ordered_totals[i] == current_val:
            current_count += 1
        else:
            run_values.append(current_val)
            run_lengths.append(current_count)
            current_val = ordered_totals[i]
            current_count = 1
    
    # Don't forget the last run
    run_values.append(current_val)
    run_lengths.append(current_count)
    
    run_values = np.array(run_values)
    run_lengths = np.array(run_lengths)
    
    # Step 4: Calculate mid-ranks (R: run.rank <- cumsum(stuff$lengths) - (stuff$lengths-1)/2)
    cumsum_lengths = np.cumsum(run_lengths)
    run_rank = cumsum_lengths - (run_lengths - 1) / 2
    
    # Step 5: Filter for totals > lower (R: keep <- run.totals > lower)
    keep_mask = run_values > lower
    
    if np.sum(keep_mask) < 3:
        raise ValueError("Insufficient unique points for computing knee/inflection points")
    
    y = np.log10(run_values[keep_mask])
    x = np.log10(run_rank[keep_mask])
    
    # Step 6: Find curve bounds
    edge_out = _find_curve_bounds_r_style(x, y, exclude_from)
    left_edge = edge_out["left"]
    right_edge = edge_out["right"]
    
    # Step 7: Calculate inflection point (R uses right edge)
    inflection = 10**(y[right_edge])
    
    # Step 8: Restrict to curve fitting region
    new_keep = np.arange(left_edge, right_edge + 1)
    
    if len(new_keep) >= 4:
        curx = x[new_keep]
        cury = y[new_keep]
        
        # Line between first and last points in fitting region
        xbounds = np.array([curx[0], curx[-1]])
        ybounds = np.array([cury[0], cury[-1]])
        
        gradient = (ybounds[1] - ybounds[0]) / (xbounds[1] - xbounds[0])
        intercept = ybounds[0] - xbounds[0] * gradient
        
        # Find points above the line (R: above <- which(cury >= curx * gradient + intercept))
        line_y = curx * gradient + intercept
        above_mask = cury >= line_y
        above_indices = np.where(above_mask)[0]
        
        if len(above_indices) > 0:
            # Calculate distances for points above the line
            curx_above = curx[above_indices]
            cury_above = cury[above_indices]
            
            distances = np.abs(gradient * curx_above - cury_above + intercept) / np.sqrt(gradient**2 + 1)
            
            # Find maximum distance
            max_dist_idx = np.argmax(distances)
            knee_local_idx = above_indices[max_dist_idx]
            knee = 10**(cury[knee_local_idx])
        else:
            knee = 10**(cury[0])
    else:
        knee = 10**(y[new_keep[0]])
    
    return int(knee)


# --- Main EmptyDrops Function ---

def empty_drops_v2(
    data: sc.AnnData,
    lower: int = 100,
    niters: int = 10000,
    retain: Optional[int] = None,
    return_metadata: bool = False
):
    """
    A truly vectorized implementation of the EmptyDrops algorithm using NumPy.
    
    Parameters
    ----------
    data : sc.AnnData
        Single-cell count matrix
    lower : int
        Lower threshold for ambient profile estimation
    niters : int
        Number of Monte Carlo iterations
    retain : Optional[int]
        Retain threshold (auto-detected if None)
    return_metadata : bool
        If True, return (results_df, metadata_dict) tuple
        
    Returns
    -------
    results_df : pd.DataFrame
        EmptyDrops results with FDR values
    metadata_dict : dict (only if return_metadata=True)
        Dictionary containing run metadata for logging
    """
    start_time = time.time()
    print("--- Starting EmptyDrops v2 (Final Numba Implementation) ---")
    
    original_data = data.copy()
    
    # Steps 1-4: Data preparation
    print("Step 1-4: Preparing data and ambient profile...")
    sc.pp.filter_genes(data, min_counts=1)
    totals = np.asarray(data.X.sum(axis=1)).flatten()
    data.obs['total_counts'] = totals
    test_mask = totals > lower
    ambient_gene_sums = np.asarray(data[totals <= lower, :].X.sum(axis=0)).flatten()
    _, ambient_props, _ = good_turing_ambient_pool(data, ambient_gene_sums)
    cells_to_test = data[test_mask, :]
    print(f"Found {cells_to_test.n_obs} cells to test.")
    
    # Step 5: Calculate observed log-probabilities (sequentially, it's fast)
    print("Step 5: Calculating observed log-probabilities...")
    test_matrix_sparse = cells_to_test.X.tocsr()
    obs_log_probs = np.zeros(cells_to_test.n_obs)
    for i in tqdm(range(test_matrix_sparse.shape[0]), desc="Calculating observed probs"):
        start, end = test_matrix_sparse.indptr[i], test_matrix_sparse.indptr[i+1]
        indices, row_data = test_matrix_sparse.indices[start:end], test_matrix_sparse.data[start:end]
        obs_log_probs[i] = _log_multinomial_prob_numba_sparse(indices, row_data, row_data.sum(), ambient_props)

    # Step 6: Monte Carlo simulation following R implementation pattern
    print(f"Step 6: Running {niters} Monte Carlo simulations...")
    
    total_counts_to_test = np.asarray(cells_to_test.obs['total_counts'].values, dtype=int)
    
    # Follow R's approach: order by (totals, probs) like R does with order(totals, probs)
    order_indices = np.lexsort((obs_log_probs, total_counts_to_test))
    ordered_totals = total_counts_to_test[order_indices]
    ordered_probs = obs_log_probs[order_indices]
    
    # Implement R's rle() equivalent - run length encoding of ordered totals
    unique_totals, inverse_indices, total_lengths = np.unique(
        ordered_totals, return_inverse=True, return_counts=True
    )
    
    seed = (os.getpid() + int(time.time() * 1000)) % (2**32 - 1)
    
    print(f"Processing {len(unique_totals)} unique total counts with {len(total_counts_to_test)} cells...")
    print(f"Total counts range: {unique_totals.min()} to {unique_totals.max()}")
    print(f"Starting Monte Carlo simulation...")
    
    start_mc_time = time.time()
    # Run Monte Carlo simulation following R's pattern
    n_above_ordered = _monte_carlo_r_style(
        unique_totals, total_lengths, ordered_probs, ambient_props, niters, seed
    )
    end_mc_time = time.time()
    print(f"Monte Carlo completed in {end_mc_time - start_mc_time:.2f} seconds")
    
    # Reorder results back to original cell order (inverse of R's n.above[o] <- n.above)
    n_above_array = np.zeros_like(n_above_ordered)
    n_above_array[order_indices] = n_above_ordered

    # Step 7: Calculate p-values and FDR
    print("Step 7: Calculating p-values and FDR...")
    p_values = (n_above_array + 1) / (niters + 1)
    results_df = pd.DataFrame(index=original_data.obs_names)
    results_df['Total'] = np.asarray(original_data.X.sum(axis=1)).flatten()
    results_df.loc[original_data.obs_names[test_mask], 'PValue'] = p_values
    
    if retain is None:
        try:
            retain = barcode_ranks(original_data, lower=lower)
            print(f"  --> Automatically determined retain threshold: {retain}")
        except ValueError as e:
            print(f"  --> Knee point detection failed: {e}. Not using retain.")
            retain = np.inf
    
    results_df.attrs.update({'retain': retain, 'lower': lower, 'niters': niters})

    pvals_for_fdr = results_df.loc[original_data.obs_names[test_mask], 'PValue'].copy()
    if retain is not None and np.isfinite(retain):
        pvals_for_fdr[results_df.loc[original_data.obs_names[test_mask], 'Total'] >= retain] = 0
    
    if len(pvals_for_fdr) > 0:
        _, fdr_values, _, _ = multipletests(pvals_for_fdr.dropna(), method='fdr_bh')
        results_df.loc[pvals_for_fdr.dropna().index, 'FDR'] = fdr_values
    
    end_time = time.time()
    total_runtime = end_time - start_time
    print(f"--- EmptyDrops v2 finished in {total_runtime:.2f} seconds ---")
    
    # Calculate FDR summary statistics
    fdr_001 = (results_df['FDR'] <= 0.001).sum()
    fdr_01 = (results_df['FDR'] <= 0.01).sum()
    fdr_05 = (results_df['FDR'] <= 0.05).sum()
    
    print("\n--- Results Summary ---")
    print(f"FDR <= 0.001: {fdr_001}")
    print(f"FDR <= 0.01:  {fdr_01}")
    print(f"FDR <= 0.05:  {fdr_05}")
    
    if return_metadata:
        # Create metadata dictionary for logging
        metadata = {
            'timestamp': pd.Timestamp.now().strftime('%Y%m%d_%H%M%S'),
            'niters': niters,
            'fdr_0_001': int(fdr_001),
            'fdr_0_01': int(fdr_01),
            'fdr_0_05': int(fdr_05),
            'calculated_retain': int(retain) if retain is not None else None,
            'lower': lower,
            'runtime_seconds': round(total_runtime, 2),
            'total_cells': len(results_df),
            'tested_cells': int((results_df['PValue'].notna()).sum()),
            'data_shape': f"{original_data.shape[0]}x{original_data.shape[1]}"
        }
        return results_df, metadata
    else:
        return results_df
