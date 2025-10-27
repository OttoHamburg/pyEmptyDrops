#!/usr/bin/env python3
"""
Detailed component testing following the exact chronological order of EmptyDrops.
This will help identify subtle differences causing FDR collapse at strict thresholds.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import tempfile
import os
from statsmodels.stats.multitest import multipletests

def test_step_by_step_vs_r():
    """Test each step of EmptyDrops in chronological order against R."""
    print("=== Detailed Step-by-Step Testing vs R ===")
    
    # Load the same data
    adata = sc.read_10x_h5('../data/raw_feature_bc_matrix.h5')
    print(f"Original data shape: {adata.shape}")
    
    # Step 1: Gene filtering
    print("\n--- Step 1: Gene Filtering ---")
    genes_before = adata.n_vars
    sc.pp.filter_genes(adata, min_counts=1)
    genes_after = adata.n_vars
    print(f"Python: {genes_before} -> {genes_after} genes (removed {genes_before - genes_after})")
    
    # Step 2: Cell totals and test/ambient split
    print("\n--- Step 2: Cell Classification ---")
    totals = np.asarray(adata.X.sum(axis=1)).flatten()
    adata.obs['total_counts'] = totals
    test_mask = totals > 100
    ambient_mask = totals <= 100
    
    print(f"Python: {ambient_mask.sum()} ambient cells, {test_mask.sum()} test cells")
    
    # Step 3: Ambient profile
    print("\n--- Step 3: Ambient Profile ---")
    ambient_gene_sums = np.asarray(adata[ambient_mask, :].X.sum(axis=0)).flatten()
    from empty_drops_v2 import good_turing_ambient_pool
    _, py_ambient_props, _ = good_turing_ambient_pool(adata, ambient_gene_sums)
    
    print(f"Python ambient profile: sum={py_ambient_props.sum():.10f}, max={py_ambient_props.max():.6f}")
    
    # Step 4: Observed log-probabilities for first few cells
    print("\n--- Step 4: Observed Log-Probabilities (Sample) ---")
    cells_to_test = adata[test_mask, :]
    test_matrix_sparse = cells_to_test.X.tocsr()
    
    from empty_drops_v2 import _log_multinomial_prob_numba_sparse
    
    # Test first 10 cells
    sample_log_probs = []
    sample_totals = []
    for i in range(min(10, cells_to_test.n_obs)):
        start, end = test_matrix_sparse.indptr[i], test_matrix_sparse.indptr[i+1]
        indices, row_data = test_matrix_sparse.indices[start:end], test_matrix_sparse.data[start:end]
        total = int(row_data.sum())
        log_prob = _log_multinomial_prob_numba_sparse(indices, row_data, total, py_ambient_props)
        sample_log_probs.append(log_prob)
        sample_totals.append(total)
    
    print("Python first 10 log-probabilities:")
    for i, (total, lp) in enumerate(zip(sample_totals, sample_log_probs)):
        print(f"  Cell {i}: total={total}, log_prob={lp:.2f}")
    
    # Step 5: Monte Carlo simulation on sample
    print("\n--- Step 5: Monte Carlo Simulation (Sample) ---")
    
    # Test Monte Carlo on first cell with small niters for comparison
    test_total = sample_totals[0]
    test_obs_log_prob = sample_log_probs[0]
    niters = 20
    
    from empty_drops_v2 import _monte_carlo_r_style
    
    # Set up for single cell test
    unique_totals = np.array([test_total])
    total_lengths = np.array([1])
    ordered_probs = np.array([test_obs_log_prob])
    
    np.random.seed(42)
    py_n_above = _monte_carlo_r_style(unique_totals, total_lengths, ordered_probs, py_ambient_props, niters, 42)
    py_p_value = (py_n_above[0] + 1) / (niters + 1)
    
    print(f"Python MC (cell 0): {py_n_above[0]}/{niters} hits, p-value={py_p_value:.4f}")
    
    # Step 6: Compare with R's full pipeline on same data
    print("\n--- Step 6: R Full Pipeline Comparison ---")
    
    # Save test data for R
    test_data = {
        'cell_totals': sample_totals,
        'ambient_counts': ambient_gene_sums[:100],  # First 100 genes for comparison
        'test_cell_0_total': test_total,
        'test_cell_0_obs_log_prob': test_obs_log_prob
    }
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_file = f.name
        pd.DataFrame({
            'ambient_counts': ambient_gene_sums,
            'cell_totals': np.concatenate([np.zeros(len(ambient_gene_sums) - len(sample_totals)), sample_totals])
        }).to_csv(f, index=False)
    
    r_script = f"""
    library(DropletUtils)
    
    # Test the same steps as Python
    cat("=== R Step-by-Step Testing ===\\n")
    
    # Load the same H5 file
    sce_all <- read10xCounts('../data/raw_feature_bc_matrix.h5')
    cat("R original shape:", dim(sce_all), "\\n")
    
    # Filter for GEX only
    gex_mask <- rowData(sce_all)$Type == 'Gene Expression'
    sce <- sce_all[gex_mask, ]
    cat("R GEX-only shape:", dim(sce), "\\n")
    
    # Cell classification
    totals <- colSums(counts(sce))
    ambient_mask <- totals <= 100
    test_mask <- totals > 100
    cat("R cells:", sum(ambient_mask), "ambient,", sum(test_mask), "test\\n")
    
    # Ambient profile
    ambient_counts <- rowSums(counts(sce)[, ambient_mask])
    ambient_props <- ambient_counts / sum(ambient_counts)
    
    # Handle zero probabilities like R's .safe_good_turing
    still_zero <- ambient_props <= 0
    if (any(still_zero)) {{
        pseudo_prob <- 1.0 / sum(ambient_counts)
        n_zero <- sum(still_zero)
        ambient_props[still_zero] <- pseudo_prob / n_zero
        ambient_props[!still_zero] <- ambient_props[!still_zero] * (1 - pseudo_prob)
    }}
    
    cat("R ambient profile: sum=", sum(ambient_props), ", max=", max(ambient_props), "\\n")
    
    # Test log-probability calculation on first few cells
    test_cells <- which(test_mask)[1:10]
    cat("R first 10 log-probabilities:\\n")
    
    for (i in 1:min(10, length(test_cells))) {{
        cell_idx <- test_cells[i]
        cell_counts <- counts(sce)[, cell_idx]
        total <- sum(cell_counts)
        
        # Calculate log-probability
        log_prob <- lgamma(total + 1)
        for (j in 1:length(cell_counts)) {{
            if (cell_counts[j] > 0) {{
                log_prob <- log_prob + cell_counts[j] * log(ambient_props[j]) - lgamma(cell_counts[j] + 1)
            }}
        }}
        
        cat("  Cell", i-1, ": total=", total, ", log_prob=", round(log_prob, 2), "\\n")
        
        # Test Monte Carlo on first cell
        if (i == 1) {{
            set.seed(42)
            n_above <- 0
            niters <- 20
            
            for (iter in 1:niters) {{
                sim_counts <- rmultinom(1, total, ambient_props)[,1]
                sim_log_prob <- lgamma(total + 1)
                for (k in 1:length(sim_counts)) {{
                    if (sim_counts[k] > 0) {{
                        sim_log_prob <- sim_log_prob + sim_counts[k] * log(ambient_props[k]) - lgamma(sim_counts[k] + 1)
                    }}
                }}
                if (sim_log_prob <= log_prob) {{
                    n_above <- n_above + 1
                }}
            }}
            
            p_value <- (n_above + 1) / (niters + 1)
            cat("R MC (cell 0):", n_above, "/", niters, "hits, p-value=", round(p_value, 4), "\\n")
        }}
    }}
    
    # Test knee point detection
    cat("\\n--- R Knee Point Detection ---\\n")
    br <- barcodeRanks(counts(sce), lower=100)
    knee_point <- metadata(br)$knee
    cat("R knee point:", knee_point, "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        r_file = f.name
        f.write(r_script)
    
    try:
        result = subprocess.run(['R', '--vanilla', '-f', r_file], 
                              capture_output=True, text=True, cwd='.')
        print("R output:")
        print(result.stdout)
        if result.stderr:
            print("R errors:")
            print(result.stderr)
            
    except Exception as e:
        print(f"Error running R: {e}")
    finally:
        # Cleanup
        for f in [temp_file, r_file]:
            if os.path.exists(f):
                os.unlink(f)

def test_fdr_correction_vs_r():
    """Test the FDR correction step specifically."""
    print("\n=== Testing FDR Correction ===")
    
    # Create test p-values similar to what we might get
    np.random.seed(42)
    n_cells = 15000
    
    # Simulate p-values with similar distribution to our results
    # Most cells should have moderate p-values, few should be very significant
    p_values = np.random.beta(2, 3, n_cells)  # Beta distribution skewed toward higher p-values
    p_values[:500] = np.random.uniform(0.001, 0.01, 500)  # Some very significant cells
    p_values[500:1000] = np.random.uniform(0.01, 0.05, 500)  # Some moderately significant
    
    print(f"Test p-values: min={p_values.min():.4f}, max={p_values.max():.4f}")
    print(f"P-values < 0.001: {(p_values < 0.001).sum()}")
    print(f"P-values < 0.01: {(p_values < 0.01).sum()}")
    print(f"P-values < 0.05: {(p_values < 0.05).sum()}")
    
    # Python FDR correction
    _, py_fdr, _, _ = multipletests(p_values, method='fdr_bh')
    
    py_fdr_001 = (py_fdr <= 0.001).sum()
    py_fdr_01 = (py_fdr <= 0.01).sum()
    py_fdr_05 = (py_fdr <= 0.05).sum()
    
    print(f"\nPython FDR correction:")
    print(f"FDR <= 0.001: {py_fdr_001}")
    print(f"FDR <= 0.01: {py_fdr_01}")
    print(f"FDR <= 0.05: {py_fdr_05}")
    
    # Save p-values for R
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_file = f.name
        pd.DataFrame({'p_values': p_values}).to_csv(f, index=False)
    
    r_script = f"""
    # Test FDR correction
    p_data <- read.csv('{temp_file}')
    p_values <- p_data$p_values
    
    cat("R p-values: min=", min(p_values), ", max=", max(p_values), "\\n")
    cat("R P-values < 0.001:", sum(p_values < 0.001), "\\n")
    cat("R P-values < 0.01:", sum(p_values < 0.01), "\\n")
    cat("R P-values < 0.05:", sum(p_values < 0.05), "\\n")
    
    # R FDR correction (Benjamini-Hochberg)
    r_fdr <- p.adjust(p_values, method="BH")
    
    cat("\\nR FDR correction:\\n")
    cat("FDR <= 0.001:", sum(r_fdr <= 0.001), "\\n")
    cat("FDR <= 0.01:", sum(r_fdr <= 0.01), "\\n")
    cat("FDR <= 0.05:", sum(r_fdr <= 0.05), "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        r_file = f.name
        f.write(r_script)
    
    try:
        result = subprocess.run(['R', '--vanilla', '-f', r_file], 
                              capture_output=True, text=True, cwd='.')
        print("\nR FDR output:")
        print(result.stdout)
        
    except Exception as e:
        print(f"Error running R: {e}")
    finally:
        # Cleanup
        for f in [temp_file, r_file]:
            if os.path.exists(f):
                os.unlink(f)

if __name__ == "__main__":
    print("Detailed chronological component testing...")
    
    # Test each step in order
    test_step_by_step_vs_r()
    test_fdr_correction_vs_r()
    
    print("\n=== Analysis Complete ===")
    print("Look for discrepancies in:")
    print("1. Log-probability calculations")
    print("2. Monte Carlo hit counts") 
    print("3. Knee point detection")
    print("4. FDR correction procedure")
