#!/usr/bin/env python3
"""
Investigate why R tests ~33,000 barcodes while Python tests ~15,000.
"""

import scanpy as sc
import numpy as np

def investigate_barcode_testing():
    """Investigate barcode testing differences."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    print("\n=== BARCODE DISTRIBUTION ANALYSIS ===")
    print(f"Total barcodes: {len(totals):,}")

    # Analyze barcode ranges
    ranges = [
        ("Empty droplets (≤100)", totals <= 100),
        ("Low count (101-1000)", (totals > 100) & (totals <= 1000)),
        ("Medium count (1001-5000)", (totals > 1000) & (totals <= 5000)),
        ("High count (5001-10000)", (totals > 5000) & (totals <= 10000)),
        ("Very high count (10001-13138)", (totals > 10000) & (totals <= 13138)),
        ("Ultra high count (>13138)", totals > 13138),
    ]

    for name, mask in ranges:
        count = np.sum(mask)
        print(f"{name:30}: {count:7,}")

    print("\n=== HYPOTHESIS TESTING FOR R'S BARCODE SELECTION ===")

    # Hypothesis 1: R uses a much lower 'lower' threshold
    print("Testing if R uses different 'lower' threshold:")

    # If R used lower=0 (tests all barcodes)
    all_barcodes = totals > 0
    print(f"If R uses lower=0: would test {np.sum(all_barcodes):,} barcodes")

    # If R used lower=50
    lower_50 = totals > 50
    print(f"If R uses lower=50: would test {np.sum(lower_50):,} barcodes")

    # If R used lower=10
    lower_10 = totals > 10
    print(f"If R uses lower=10: would test {np.sum(lower_10):,} barcodes")

    # Hypothesis 2: R doesn't filter genes at all (uses all 22,040 genes)
    print("\nTesting if R uses all genes (22,040) for knee detection:")
    knee_all_genes, inflection_all_genes = calculate_knee_point(totals, lower=100)
    print(f"All genes: knee={knee_all_genes:.1f}, inflection={inflection_all_genes:.1f}")

    # Hypothesis 3: R uses different retain application logic
    print("\nTesting different retain threshold application:")

    # Current Python logic: test barcodes between lower and retain
    current_python = (totals > 100) & (totals < 13138)
    print(f"Current Python logic: {np.sum(current_python):,} barcodes")

    # What if R tests all barcodes above lower, regardless of retain?
    r_hypothesis1 = totals > 100
    print(f"R hypothesis 1 (all > lower): {np.sum(r_hypothesis1):,} barcodes")

    # What if R uses a much lower retain threshold?
    r_hypothesis2 = (totals > 100) & (totals < 5000)  # Much lower retain
    print(f"R hypothesis 2 (retain=5000): {np.sum(r_hypothesis2):,} barcodes")

    # What if R tests barcodes in a different range entirely?
    r_hypothesis3 = (totals > 50) & (totals < 20000)  # Wider range
    print(f"R hypothesis 3 (50 < x < 20000): {np.sum(r_hypothesis3):,} barcodes")

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
    investigate_barcode_testing()
