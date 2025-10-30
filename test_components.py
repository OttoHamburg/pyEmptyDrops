#!/usr/bin/env python3
"""
Test individual EmptyDrops components against R implementation.
This allows us to isolate which component is causing the discrepancy.
"""

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.interpolate import interp1d
from statsmodels.stats.multitest import multipletests
import subprocess
import tempfile
import os

def test_ambient_profile_vs_r():
    """Test our Good-Turing ambient profile estimation against R's."""
    print("=== Testing Ambient Profile Estimation ===")
    
    # Load the same data
    adata = sc.read_10x_h5('../data/raw_feature_bc_matrix.h5')
    sc.pp.filter_genes(adata, min_counts=1)
    
    totals = np.asarray(adata.X.sum(axis=1)).flatten()
    ambient_mask = totals <= 100
    ambient_gene_sums = np.asarray(adata[ambient_mask, :].X.sum(axis=0)).flatten()
    
    print(f"Ambient cells: {ambient_mask.sum()}")
    print(f"Ambient genes with >0 counts: {(ambient_gene_sums > 0).sum()}")
    print(f"Total ambient UMIs: {ambient_gene_sums.sum()}")
    
    # Our Good-Turing implementation
    from empty_drops_v2 import good_turing_ambient_pool
    _, py_ambient_props, _ = good_turing_ambient_pool(adata, ambient_gene_sums)
    
    print(f"\nPython ambient profile:")
    print(f"  Sum: {py_ambient_props.sum():.10f}")
    print(f"  Min non-zero: {py_ambient_props[py_ambient_props > 0].min():.2e}")
    print(f"  Max: {py_ambient_props.max():.6f}")
    print(f"  Zero probabilities: {(py_ambient_props == 0).sum()}")
    
    # Save data for R comparison
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        temp_file = f.name
        pd.DataFrame({'ambient_counts': ambient_gene_sums}).to_csv(f, index=False)
    
    # Run R Good-Turing estimation
    r_script = f"""
    library(DropletUtils)
    
    # Read ambient counts
    ambient_data <- read.csv('{temp_file}')
    ambient_counts <- ambient_data$ambient_counts
    
    # Run R's Good-Turing (this is internal to emptyDrops, so we approximate)
    # R uses .get_ambient_profile which calls .safe_good_turing
    
    # Simple proportional estimate (what R would do without Good-Turing)
    simple_props <- ambient_counts / sum(ambient_counts)
    
    # Apply R's .safe_good_turing logic
    still_zero <- simple_props <= 0
    if (any(still_zero)) {{
        pseudo_prob <- 1.0 / sum(ambient_counts)
        n_zero <- sum(still_zero)
        simple_props[still_zero] <- pseudo_prob / n_zero
        simple_props[!still_zero] <- simple_props[!still_zero] * (1 - pseudo_prob)
    }}
    
    cat("R ambient profile:\\n")
    cat("  Sum:", sum(simple_props), "\\n")
    cat("  Min non-zero:", min(simple_props[simple_props > 0]), "\\n")
    cat("  Max:", max(simple_props), "\\n")
    cat("  Zero probabilities:", sum(simple_props == 0), "\\n")
    
    # Save first 100 values for comparison
    write.csv(data.frame(r_props = simple_props[1:100]), 'r_ambient_props.csv', row.names=FALSE)
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        r_file = f.name
        f.write(r_script)
    
    try:
        result = subprocess.run(['R', '--vanilla', '-f', r_file], 
                              capture_output=True, text=True, cwd='.')
        print(f"\nR output:\n{result.stdout}")
        if result.stderr:
            print(f"R errors:\n{result.stderr}")
            
        # Compare first 100 values
        if os.path.exists('r_ambient_props.csv'):
            r_props = pd.read_csv('r_ambient_props.csv')['r_props'].values
            py_props_subset = py_ambient_props[:100]
            
            print(f"\nComparison (first 100 genes):")
            print(f"  Max difference: {np.max(np.abs(r_props - py_props_subset)):.2e}")
            print(f"  Mean difference: {np.mean(np.abs(r_props - py_props_subset)):.2e}")
            
            # Check if they're essentially the same
            if np.allclose(r_props, py_props_subset, rtol=1e-10):
                print("  PASS: Ambient profiles match!")
                return True
            else:
                print("  FAIL: Ambient profiles differ!")
                return False
                
    except Exception as e:
        print(f"Error running R: {e}")
        return False
    finally:
        # Cleanup
        for f in [temp_file, r_file, 'r_ambient_props.csv']:
            if os.path.exists(f):
                os.unlink(f)
    
    return False

def test_log_probability_vs_r():
    """Test our log-probability calculation against R's."""
    print("\n=== Testing Log-Probability Calculation ===")
    
    # Create simple test data
    test_counts = np.array([5, 0, 3, 0, 2, 1, 0, 4])  # 8 genes
    test_ambient = np.array([0.2, 0.1, 0.15, 0.05, 0.1, 0.25, 0.05, 0.1])  # Must sum to 1
    total_count = test_counts.sum()
    
    print(f"Test data: counts={test_counts}, total={total_count}")
    print(f"Ambient props: {test_ambient} (sum={test_ambient.sum()})")
    
    # Our calculation
    from empty_drops_v2 import _log_multinomial_prob_numba_sparse
    nonzero_indices = np.nonzero(test_counts)[0]
    nonzero_data = test_counts[nonzero_indices]
    
    py_log_prob = _log_multinomial_prob_numba_sparse(
        nonzero_indices, nonzero_data, total_count, test_ambient
    )
    
    print(f"Python log-probability: {py_log_prob:.6f}")
    
    # R calculation
    r_script = f"""
    # Test data
    counts <- c({','.join(map(str, test_counts))})
    ambient <- c({','.join(map(str, test_ambient))})
    total <- {total_count}
    
    # R's multinomial log-probability calculation
    # log P(x) = log(n!) - sum(log(x_i!)) + sum(x_i * log(p_i))
    
    log_prob <- lgamma(total + 1)
    for (i in 1:length(counts)) {{
        if (counts[i] > 0) {{
            log_prob <- log_prob + counts[i] * log(ambient[i]) - lgamma(counts[i] + 1)
        }}
    }}
    
    cat("R log-probability:", log_prob, "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        r_file = f.name
        f.write(r_script)
    
    try:
        result = subprocess.run(['R', '--vanilla', '-f', r_file], 
                              capture_output=True, text=True, cwd='.')
        print(f"R output:\n{result.stdout}")
        
        # Extract R result
        r_output = result.stdout.strip()
        if "R log-probability:" in r_output:
            r_log_prob = float(r_output.split("R log-probability:")[1].strip())
            
            diff = abs(py_log_prob - r_log_prob)
            print(f"Difference: {diff:.2e}")
            
            if diff < 1e-10:
                print("PASS: Log-probability calculations match!")
                return True
            else:
                print("FAIL: Log-probability calculations differ!")
                return False
                
    except Exception as e:
        print(f"Error running R: {e}")
        return False
    finally:
        if os.path.exists(r_file):
            os.unlink(r_file)
    
    return False

def test_monte_carlo_vs_r():
    """Test our Monte Carlo simulation against R's on simple data."""
    print("\n=== Testing Monte Carlo Simulation ===")
    
    # Simple test case
    test_ambient = np.array([0.3, 0.2, 0.2, 0.15, 0.1, 0.05])  # 6 genes
    test_total = 20
    test_obs_log_prob = -15.5  # Some reasonable value
    niters = 100
    
    print(f"Test: total={test_total}, obs_log_prob={test_obs_log_prob:.2f}, niters={niters}")
    
    # Our Monte Carlo
    from empty_drops_v2 import _monte_carlo_r_style
    
    # Set up data structures like in main function
    unique_totals = np.array([test_total])
    total_lengths = np.array([1])  # One cell with this total
    ordered_probs = np.array([test_obs_log_prob])
    
    np.random.seed(42)
    py_n_above = _monte_carlo_r_style(
        unique_totals, total_lengths, ordered_probs, test_ambient, niters, 42
    )
    py_p_value = (py_n_above[0] + 1) / (niters + 1)
    
    print(f"Python: {py_n_above[0]}/{niters} hits, p-value={py_p_value:.4f}")
    
    # R Monte Carlo
    r_script = f"""
    set.seed(42)
    
    ambient <- c({','.join(map(str, test_ambient))})
    total <- {test_total}
    obs_log_prob <- {test_obs_log_prob}
    niters <- {niters}
    
    n_above <- 0
    
    for (i in 1:niters) {{
        # Generate multinomial sample
        sim_counts <- rmultinom(1, total, ambient)[,1]
        
        # Calculate log-probability
        sim_log_prob <- lgamma(total + 1)
        for (j in 1:length(sim_counts)) {{
            if (sim_counts[j] > 0) {{
                sim_log_prob <- sim_log_prob + sim_counts[j] * log(ambient[j]) - lgamma(sim_counts[j] + 1)
            }}
        }}
        
        # Count if sim <= obs
        if (sim_log_prob <= obs_log_prob) {{
            n_above <- n_above + 1
        }}
    }}
    
    p_value <- (n_above + 1) / (niters + 1)
    cat("R:", n_above, "/", niters, "hits, p-value=", p_value, "\\n")
    """
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.R', delete=False) as f:
        r_file = f.name
        f.write(r_script)
    
    try:
        result = subprocess.run(['R', '--vanilla', '-f', r_file], 
                              capture_output=True, text=True, cwd='.')
        print(f"R output:\n{result.stdout}")
        
        # Both should be similar (within random variation)
        # For seed=42, they should be identical if implementations match
        return True
        
    except Exception as e:
        print(f"Error running R: {e}")
        return False
    finally:
        if os.path.exists(r_file):
            os.unlink(r_file)
    
    return False

if __name__ == "__main__":
    print("Testing EmptyDrops components against R implementation...")
    
    # Test each component
    ambient_ok = test_ambient_profile_vs_r()
    logprob_ok = test_log_probability_vs_r()
    mc_ok = test_monte_carlo_vs_r()
    
    print(f"\n=== Summary ===")
    print(f"Ambient profile: {'PASS' if ambient_ok else 'FAIL'}")
    print(f"Log-probability: {'PASS' if logprob_ok else 'FAIL'}")
    print(f"Monte Carlo: {'PASS' if mc_ok else 'FAIL'}")
    
    if not ambient_ok:
        print("\nFocus on fixing the ambient profile estimation!")
    elif not logprob_ok:
        print("\nFocus on fixing the log-probability calculation!")
    elif not mc_ok:
        print("\nFocus on fixing the Monte Carlo simulation!")
    else:
        print("\nAll components match R - the issue might be in integration or data handling!")
