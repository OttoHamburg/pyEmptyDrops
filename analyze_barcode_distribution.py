#!/usr/bin/env python3
"""
Analyze the barcode distribution to understand why SUPERNOVA still tests 15,000 barcodes.
"""

import scanpy as sc
import numpy as np

def analyze_barcode_distribution():
    """Analyze the distribution of barcode totals."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    # Get totals
    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    print(f"Total barcodes: {len(totals):,}")
    print(f"Total range: {totals.min()} to {totals.max()}")

    # Analyze different ranges
    lower = 100

    # Original retain threshold (17,179)
    retain_original = 17179
    range_original = (totals > lower) & (totals < retain_original)
    count_original = np.sum(range_original)
    print(f"\nOriginal retain threshold: {retain_original:,}")
    print(f"Barcodes in range {lower} < x < {retain_original}: {count_original:,}")

    # SUPERNOVA retain threshold (13,138)
    retain_supernova = 13138
    range_supernova = (totals > lower) & (totals < retain_supernova)
    count_supernova = np.sum(range_supernova)
    print(f"\nSUPERNOVA retain threshold: {retain_supernova:,}")
    print(f"Barcodes in range {lower} < x < {retain_supernova}: {count_supernova:,}")

    # Check if there are more barcodes above the SUPERNOVA threshold
    above_supernova = totals >= retain_supernova
    count_above = np.sum(above_supernova)
    print(f"Barcodes >= {retain_supernova}: {count_above:,}")

    # Check the distribution around the thresholds
    print("\n=== Distribution Analysis ===")

    # Check around the original threshold
    around_original = (totals >= retain_original - 1000) & (totals <= retain_original + 1000)
    count_around_original = np.sum(around_original)
    print(f"Barcodes within ±1000 of original threshold ({retain_original}): {count_around_original:,}")

    # Check around the SUPERNOVA threshold
    around_supernova = (totals >= retain_supernova - 1000) & (totals <= retain_supernova + 1000)
    count_around_supernova = np.sum(around_supernova)
    print(f"Barcodes within ±1000 of SUPERNOVA threshold ({retain_supernova}): {count_around_supernova:,}")

    # Show the actual distribution
    print("\n=== Detailed Distribution ===")
    thresholds = [100, 1000, 5000, 10000, 13138, 17179, 20000]
    for thresh in thresholds:
        count = np.sum(totals >= thresh)
        print(f"Barcodes >= {thresh:6,}: {count:7,}")

    # Show why we get exactly 15,000 tested barcodes
    print("\n=== Why exactly 15,000? ===")

    # Count barcodes between 100 and 13,138
    tested_range = (totals > 100) & (totals < 13138)
    tested_count = np.sum(tested_range)
    print(f"Barcodes > 100 and < 13,138: {tested_count:,}")

    # But wait, the function tests 15,000, not 14,794 as we'd expect
    # This suggests there might be some additional filtering or the exact threshold matters

    # Let's check the exact count at different thresholds around 13,138
    for test_thresh in [13000, 13138, 13200, 13500, 14000]:
        tested_at_thresh = (totals > 100) & (totals < test_thresh)
        count_at_thresh = np.sum(tested_at_thresh)
        print(f"Barcodes > 100 and < {test_thresh:5,}: {count_at_thresh:6,}")

if __name__ == "__main__":
    analyze_barcode_distribution()
