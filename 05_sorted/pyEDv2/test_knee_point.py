#!/usr/bin/env python3
"""
Test knee point detection against R's barcodeRanks implementation.
"""

import numpy as np
import pandas as pd
import scanpy as sc
import subprocess
import tempfile
import os
from scipy.stats import rankdata

def test_knee_point_step_by_step():
    """Test each step of knee point detection against R."""
    print("=== Testing Knee Point Detection vs R ===")
    
    # Load the same data
    adata = sc.read_10x_h5('../data/raw_feature_bc_matrix.h5')
    sc.pp.filter_genes(adata, min_counts=1)
    
    totals = np.asarray(adata.X.sum(axis=1)).flatten()
    print(f"Total cells: {len(totals)}")
    print(f"Total range: {totals.min()} to {totals.max()}")
    print(f"Cells > 100: {(totals > 100).sum()}")
    
    # Step 1: Order by decreasing totals (like R)
    print("\n--- Step 1: Ordering ---")
    order_indices = np.argsort(-totals)  # Decreasing order
    sorted_totals = totals[order_indices]
    
    print(f"Python sorted totals (first 10): {sorted_totals[:10]}")
    print(f"Python sorted totals (around rank 1000): {sorted_totals[995:1005]}")
    
    # Step 2: Compute ranks using R's method
    print("\n--- Step 2: Ranking ---")
    ranks = rankdata(-totals, method='average')  # Negative for descending ranks
    
    print(f"Python ranks (first 10 cells): {ranks[order_indices[:10]]}")
    
    # Step 3: Run-length encoding
    print("\n--- Step 3: Run-Length Encoding ---")
    unique_totals, inverse_indices, counts = np.unique(sorted_totals, return_inverse=True, return_counts=True)
    
    # Compute mid-rank for each run
    cumulative_counts = np.cumsum(counts)
    run_ranks = cumulative_counts - (counts - 1) / 2
    
    print(f"Python unique totals (first 10): {unique_totals[:10]}")
    print(f"Python run ranks (first 10): {run_ranks[:10]}")
    print(f"Python total unique values: {len(unique_totals)}")
    
    # Step 4: Filter for totals > 100
    print("\n--- Step 4: Filtering ---")
    keep_mask = unique_totals > 100
    kept_totals = unique_totals[keep_mask]
    kept_ranks = run_ranks[keep_mask]
    
    print(f"Python kept points: {len(kept_totals)}")
    print(f"Python kept totals range: {kept_totals.min()} to {kept_totals.max()}")
    print(f"Python kept ranks range: {kept_ranks.min()} to {kept_ranks.max()}")
    
    # Step 5: Log10 transformation
    print("\n--- Step 5: Log10 Transformation ---")
    x = np.log10(kept_ranks)
    y = np.log10(kept_totals)
    
    print(f"Python log10 x range: {x.min():.3f} to {x.max():.3f}")
    print(f"Python log10 y range: {y.min():.3f} to {y.max():.3f}")
    
    # Step 6: Find curve bounds
    print("\n--- Step 6: Curve Bounds ---")
    d1n = np.diff(y) / np.diff(x)  # First derivative
    exclude_from = 50
    skip = min(len(d1n) - 1, np.sum(kept_ranks <= exclude_from))
    
    print(f"Python derivative length: {len(d1n)}")
    print(f"Python skip: {skip}")
    
    if skip < len(d1n):
        d1n_subset = d1n[skip:]
        right_edge = np.argmin(d1n_subset) + skip
        left_edge = np.argmax(d1n_subset[:np.argmin(d1n_subset) + 1]) + skip
    else:
        right_edge = len(d1n) - 1
        left_edge = 0
    
    print(f"Python left_edge: {left_edge}, right_edge: {right_edge}")
    print(f"Python edge region size: {right_edge - left_edge + 1}")
    
    # Step 7: Knee detection
    print("\n--- Step 7: Knee Detection ---")
    if right_edge - left_edge + 1 >= 4:
        region_x = x[left_edge:right_edge + 1]
        region_y = y[left_edge:right_edge + 1]
        
        # Fit line between boundaries
        x_bounds = np.array([region_x[0], region_x[-1]])
        y_bounds = np.array([region_y[0], region_y[-1]])
        gradient = (y_bounds[1] - y_bounds[0]) / (x_bounds[1] - x_bounds[0])
        intercept = y_bounds[0] - x_bounds[0] * gradient
        
        print(f"Python line: gradient={gradient:.6f}, intercept={intercept:.6f}")
        
        # Find points above the line
        line_y = region_x * gradient + intercept
        above_mask = region_y >= line_y
        
        print(f"Python points above line: {np.sum(above_mask)}")
        
        if np.any(above_mask):
            above_x = region_x[above_mask]
            above_y = region_y[above_mask]
            
            # Calculate distances from line
            distances = np.abs(gradient * above_x - above_y + intercept) / np.sqrt(gradient**2 + 1)
            
            print(f"Python max distance: {np.max(distances):.6f}")
            
            # Find maximum distance point
            max_dist_idx = np.argmax(distances)
            knee_y = above_y[max_dist_idx]
            knee_total = 10**knee_y
            
            print(f"Python knee point: {int(knee_total)}")
        else:
            knee_total = 10**region_y[0]
            print(f"Python knee point (fallback): {int(knee_total)}")
    else:
        knee_total = 10**y[0] if len(y) > 0 else np.median(kept_totals)
        print(f"Python knee point (insufficient points): {int(knee_total)}")
    
    # Compare with R
    print("\n--- R Comparison ---")
    r_script = f"""
    library(DropletUtils)
    
    # Load the same data
    sce_all <- read10xCounts('../data/raw_feature_bc_matrix.h5')
    gex_mask <- rowData(sce_all)$Type == 'Gene Expression'
    sce <- sce_all[gex_mask, ]
    
    # Filter genes like Python
    gene_sums <- rowSums(counts(sce))
    keep_genes <- gene_sums > 0
    sce_filtered <- sce[keep_genes, ]
    
    cat("R filtered shape:", dim(sce_filtered), "\\n")
    
    # Calculate totals
    totals <- colSums(counts(sce_filtered))
    cat("R total cells:", length(totals), "\\n")
    cat("R total range:", min(totals), "to", max(totals), "\\n")
    cat("R cells > 100:", sum(totals > 100), "\\n")
    
    # Run barcodeRanks
    br <- barcodeRanks(counts(sce_filtered), lower=100)
    knee_point <- metadata(br)$knee
    
    cat("R knee point:", knee_point, "\\n")
    
    # Get some intermediate values for comparison
    cat("R barcodeRanks details:\\n")
    cat("  Total unique ranks:", nrow(br), "\\n")
    cat("  Rank range:", min(br$rank), "to", max(br$rank), "\\n")
    cat("  Total range:", min(br$total), "to", max(br$total), "\\n")
    
    # Show first few values
    cat("R first 10 ranks and totals:\\n")
    for (i in 1:min(10, nrow(br))) {{
        cat("  ", i, ": rank=", br$rank[i], ", total=", br$total[i], "\\n")
    }}
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
        if os.path.exists(r_file):
            os.unlink(r_file)

if __name__ == "__main__":
    test_knee_point_step_by_step()
