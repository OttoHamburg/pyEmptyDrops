#!/usr/bin/env python3
"""
Deep analysis of algorithmic differences between R and Python implementations.
"""

import scanpy as sc
import numpy as np
from empty_drops import calculate_knee_point, _find_curve_bounds

def analyze_algorithm_differences():
    """Compare algorithm implementations step by step."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    print("\n=== STEP 1: Gene Filtering Analysis ===")
    print(f"Original genes: {raw_adata.n_vars:,}")

    # Python original filtering (remove zero genes)
    gene_totals = np.array(raw_adata.X.sum(axis=0)).flatten()
    python_filter = gene_totals > 0
    python_filtered_genes = np.sum(python_filter)
    print(f"Python original filtering (> 0): {python_filtered_genes:,} genes")

    # SUPERNOVA filtering (≥ 1000)
    supernova_filter = gene_totals >= 1000
    supernova_filtered_genes = np.sum(supernova_filter)
    print(f"SUPERNOVA filtering (≥ 1000): {supernova_filtered_genes:,} genes")

    # What if R uses a different threshold?
    for threshold in [100, 500, 2000, 5000]:
        r_like_filter = gene_totals >= threshold
        r_like_genes = np.sum(r_like_filter)
        print(f"R-like filtering (≥ {threshold}): {r_like_genes:,} genes")

    print("\n=== STEP 2: Algorithm Step-by-Step Comparison ===")

    # Test with original gene set (20,026 genes)
    knee_original, inflection_original = calculate_knee_point(totals, lower=100)
    print(f"Original genes (20,026): knee={knee_original:.1f}, inflection={inflection_original:.1f}")

    # Test with R-like filtering (let's try ≥ 100)
    r_like_filter = gene_totals >= 100
    r_like_adata = raw_adata[:, r_like_filter]
    r_like_totals = np.array(r_like_adata.X.sum(axis=1)).flatten()
    knee_r_like, inflection_r_like = calculate_knee_point(r_like_totals, lower=100)
    print(f"R-like genes (≥ 100): knee={knee_r_like:.1f}, inflection={inflection_r_like:.1f}")

    # Test with SUPERNOVA filtering (≥ 1000)
    supernova_totals = np.array(raw_adata[:, supernova_filter].X.sum(axis=1)).flatten()
    knee_supernova, inflection_supernova = calculate_knee_point(supernova_totals, lower=100)
    print(f"SUPERNOVA genes (≥ 1000): knee={knee_supernova:.1f}, inflection={inflection_supernova:.1f}")

    print("\n=== STEP 3: Curve Bounds Analysis ===")

    # Analyze curve bounds for different gene sets
    for name, test_totals, n_genes in [
        ("Original (20,026 genes)", totals, raw_adata.n_vars),
        ("R-like (≥ 100)", r_like_totals, np.sum(r_like_filter)),
        ("SUPERNOVA (≥ 1000)", supernova_totals, np.sum(supernova_filter))
    ]:

        # Sort and prepare for curve bounds
        sorted_indices = np.argsort(test_totals)[::-1]
        sorted_totals = test_totals[sorted_indices]

        unique_totals, inverse_indices, counts = np.unique(sorted_totals, return_inverse=True, return_counts=True)
        cumulative_counts = np.cumsum(counts)
        run_ranks = cumulative_counts - (counts - 1) / 2.0

        keep = unique_totals > 100
        if np.sum(keep) < 3:
            print(f"{name}: Insufficient points")
            continue

        y = np.log10(unique_totals[keep])
        x = np.log10(run_ranks[keep])

        left_edge, right_edge = _find_curve_bounds(x, y, exclude_from=50)

        print(f"{name} ({n_genes:,} genes):")
        print(f"  Curve bounds: left={left_edge}, right={right_edge}")
        print(f"  Bound totals: {10**y[left_edge]:.1f} to {10**y[right_edge]:.1f}")
        print(f"  Log space: x=[{x.min():.3f}, {x.max():.3f}], y=[{y.min():.3f}, {y.max():.3f}]")

    print("\n=== STEP 4: Barcode Testing Logic Analysis ===")

    # Analyze why R tests more barcodes
    print("Barcode distribution analysis:")

    # Count barcodes in different ranges
    for threshold in [100, 1000, 5000, 10000, 13138, 17179]:
        count_above = np.sum(totals >= threshold)
        print(f"Barcodes >= {threshold:6,}: {count_above:7,}")

    # Check what R might be doing differently
    print("\nPossible explanations for R testing more barcodes:")
    print("1. R uses a different 'lower' threshold")
    print("2. R has different logic for determining which barcodes to test")
    print("3. R applies retain threshold differently")
    print("4. R processes the matrix differently before knee detection")

    # Test hypothesis: R might use lower=1000 or similar
    print("\nTesting hypothesis: R uses higher lower threshold")
    for test_lower in [100, 500, 1000]:
        # Simulate R's potential logic
        r_like_lower_filter = totals > test_lower
        r_like_testable = r_like_lower_filter & (totals < 12600)  # R's retain
        r_like_tested = np.sum(r_like_testable)
        print(f"If R uses lower={test_lower}: would test {r_like_tested:,} barcodes")

if __name__ == "__main__":
    analyze_algorithm_differences()
