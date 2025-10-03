#!/usr/bin/env python3
"""
Analyze R's exact logic for determining which barcodes to test.
"""

import scanpy as sc
import numpy as np

def analyze_r_logic():
    """Analyze R's exact barcode testing logic."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    print(f"Total barcodes: {len(totals):,}")

    # R's default logic
    lower = 100

    # Step 1: Determine which barcodes are assumed empty
    assumed_empty = totals <= lower
    print(f"Barcodes <= {lower}: {np.sum(assumed_empty):,}")

    # Step 2: Determine which barcodes to test (non-empty barcodes)
    barcodes_to_test = ~assumed_empty
    print(f"Barcodes to test (> {lower}): {np.sum(barcodes_to_test):,}")

    # Step 3: Calculate knee point on ALL barcodes
    from empty_drops import calculate_knee_point
    knee, inflection = calculate_knee_point(totals, lower=lower)
    print(f"Knee point: {knee:.1f}")

    # Step 4: Automatically retain barcodes >= knee
    automatically_retained = totals >= knee
    print(f"Barcodes >= knee ({knee:.1f}): {np.sum(automatically_retained):,}")

    # Step 5: Test the remaining barcodes
    tested_barcodes = barcodes_to_test & ~automatically_retained
    print(f"Barcodes to actually test: {np.sum(tested_barcodes):,}")

    print("\n=== Expected vs Actual ===")
    print(f"Expected tested barcodes (R's logic): {np.sum(tested_barcodes):,}")
    print("Actual R tested barcodes: 33,393")

    # The issue might be that R uses a different knee calculation
    # Let's see what knee point gives us ~33,393 tested barcodes

    print("\n=== Finding knee point that gives ~33,393 tested barcodes ===")

    for test_knee in [10000, 11000, 12000, 12615, 13000, 14000]:
        tested_at_knee = barcodes_to_test & (totals < test_knee)
        count = np.sum(tested_at_knee)
        print(f"Knee={test_knee:5,}: tested={count:6,}")

    # Another possibility: R tests ALL barcodes > lower, not just non-empty ones
    print("\n=== Alternative hypothesis: R tests ALL barcodes > lower ===")

    all_above_lower = totals > lower
    print(f"All barcodes > {lower}: {np.sum(all_above_lower):,}")

    for test_knee in [10000, 11000, 12000, 12615, 13000, 14000]:
        tested_alt = all_above_lower & (totals < test_knee)
        count = np.sum(tested_alt)
        print(f"Knee={test_knee:5,}: tested={count:6,}")

if __name__ == "__main__":
    analyze_r_logic()
