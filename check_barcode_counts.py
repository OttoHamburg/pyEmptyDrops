#!/usr/bin/env python3
"""
Check the exact barcode counts in the dataset.
"""

import scanpy as sc
import numpy as np

def check_barcode_counts():
    """Check exact barcode counts."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    print(f"Total barcodes: {len(totals):,}")
    print(f"Barcodes > 0: {np.sum(totals > 0):,}")
    print(f"Barcodes <= 100: {np.sum(totals <= 100):,}")
    print(f"Barcodes > 100: {np.sum(totals > 100):,}")
    print(f"Barcodes > 1000: {np.sum(totals > 1000):,}")
    print(f"Barcodes > 5000: {np.sum(totals > 5000):,}")
    print(f"Barcodes > 10000: {np.sum(totals > 10000):,}")
    print(f"Barcodes > 12615: {np.sum(totals > 12615):,}")
    print(f"Barcodes > 13138: {np.sum(totals > 13138):,}")
    print(f"Barcodes > 17179: {np.sum(totals > 17179):,}")

    # Check what R's tested_barcodes of 33,393 could mean
    print("\nPossible interpretations of R's 33,393 tested barcodes:")
    print(f"1. All barcodes > 0: {np.sum(totals > 0):,}")
    print(f"2. All barcodes > 50: {np.sum(totals > 50):,}")
    print(f"3. All barcodes > 10: {np.sum(totals > 10):,}")
    print(f"4. All barcodes > 100: {np.sum(totals > 100):,}")

    # Check if maybe the R validation uses test.ambient=TRUE
    print("\nIf R used test.ambient=TRUE (tests all > 0):")
    all_above_0 = totals > 0
    print(f"  Barcodes tested: {np.sum(all_above_0):,}")

    # Check if R's retain threshold affects the count
    retain_12615 = totals >= 12615
    print(f"  Barcodes >= 12615: {np.sum(retain_12615):,}")
    print(f"  Barcodes < 12615 but > 0: {np.sum(all_above_0 & ~retain_12615):,}")

if __name__ == "__main__":
    check_barcode_counts()
