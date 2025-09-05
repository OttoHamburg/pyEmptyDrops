#!/usr/bin/env python3
"""
Deep analysis of R's knee detection mathematics to understand the exact algorithm.
"""

import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import issparse

def analyze_r_knee_detection():
    """Analyze R's knee detection algorithm step by step."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    print(f"Original data shape: {raw_adata.shape}")

    # Get totals
    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    # Test different gene filtering strategies to find the one that gives ~12,600
    print("\n=== Finding the exact gene filtering strategy ===")

    gene_totals = np.array(raw_adata.X.sum(axis=0)).flatten()

    # Test different filtering thresholds
    for threshold in [500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]:
        gene_filter = gene_totals >= threshold
        n_genes_kept = np.sum(gene_filter)

        if n_genes_kept > 100:  # Need enough genes for knee detection
            filtered_adata = raw_adata[:, gene_filter]
            filtered_totals = np.array(filtered_adata.X.sum(axis=1)).flatten()

            # Test knee detection
            knee, inflection = calculate_knee_point(filtered_totals, lower=100)
            diff = abs(knee - 12600)

            print(f"Threshold {threshold:5d}: {n_genes_kept:4d} genes, knee={knee:6.1f}, diff={diff:5.1f}")

            if diff <= 500:
                print(f"🎯 FOUND! Threshold {threshold} gives knee {knee:.1f} (within ±500 of 12,600)")
                return threshold, n_genes_kept, knee

    # If no exact match, find the closest
    print("\n=== Finding closest match ===")
    best_threshold = None
    best_diff = float('inf')
    best_knee = None

    for threshold in [500, 1000, 1500, 2000, 2500, 3000, 5000, 10000]:
        gene_filter = gene_totals >= threshold
        n_genes_kept = np.sum(gene_filter)

        if n_genes_kept > 100:
            filtered_adata = raw_adata[:, gene_filter]
            filtered_totals = np.array(filtered_adata.X.sum(axis=1)).flatten()
            knee, inflection = calculate_knee_point(filtered_totals, lower=100)
            diff = abs(knee - 12600)

            if diff < best_diff:
                best_diff = diff
                best_threshold = threshold
                best_knee = knee

    print(f"Best match: threshold={best_threshold}, knee={best_knee:.1f}, diff={best_diff:.1f}")

    return best_threshold, None, best_knee

def calculate_knee_point(totals: np.ndarray, lower: int = 100, exclude_from: int = 50) -> tuple:
    """Calculate knee point using the exact R algorithm."""
    from empty_drops import _find_curve_bounds

    # Sort totals in descending order and calculate ranks
    sorted_indices = np.argsort(totals)[::-1]
    sorted_totals = totals[sorted_indices]

    # Calculate run-length encoding equivalent for tied values
    unique_totals, inverse_indices, counts = np.unique(sorted_totals, return_inverse=True, return_counts=True)

    # Calculate mid-rank for each unique total (average rank for ties)
    cumulative_counts = np.cumsum(counts)
    run_ranks = cumulative_counts - (counts - 1) / 2.0

    # Filter by lower threshold
    keep = unique_totals > lower
    if np.sum(keep) < 3:
        return 10000.0, 20000.0  # fallback

    # Work in log10 space for numerical stability
    y = np.log10(unique_totals[keep])
    x = np.log10(run_ranks[keep])

    # Find curve bounds using numerical differentiation
    left_edge, right_edge = _find_curve_bounds(x, y, exclude_from)

    # Ensure bounds are valid
    left_edge = max(0, min(left_edge, len(x) - 1))
    right_edge = max(left_edge, min(right_edge, len(x) - 1))

    # Calculate inflection point from the right edge
    inflection = 10.0 ** y[right_edge]

    # Determine fitting region
    new_keep = np.arange(left_edge, right_edge + 1)

    # Calculate knee point using maximum distance method
    if len(new_keep) >= 4:
        curx = x[new_keep]
        cury = y[new_keep]

        # Get bounds of the fitting region
        xbounds = np.array([curx[0], curx[-1]])
        ybounds = np.array([cury[0], cury[-1]])

        # Calculate line parameters (gradient and intercept)
        if xbounds[1] != xbounds[0]:
            gradient = (ybounds[1] - ybounds[0]) / (xbounds[1] - xbounds[0])
        else:
            gradient = 0.0
        intercept = ybounds[0] - xbounds[0] * gradient

        # Find points above the line
        line_y = curx * gradient + intercept
        above_mask = cury >= line_y
        above_indices = np.where(above_mask)[0]

        if len(above_indices) > 0:
            curx_above = curx[above_indices]
            cury_above = cury[above_indices]
            distances = np.abs(gradient * curx_above - cury_above + intercept) / np.sqrt(gradient**2 + 1)

            # Find point with maximum distance
            max_dist_idx = np.argmax(distances)
            knee = 10.0 ** cury_above[max_dist_idx]
        else:
            knee = 10.0 ** cury[0]
    else:
        knee = 10.0 ** y[new_keep[0]] if len(new_keep) > 0 else 10.0 ** y[0]

    return knee, inflection

if __name__ == "__main__":
    threshold, n_genes, knee = analyze_r_knee_detection()
    print(f"\nFinal result: threshold={threshold}, genes={n_genes}, knee={knee:.1f}")
