#!/usr/bin/env python3
"""
Test script to verify R compatibility of modified SUPERNOVA implementation.
"""

import scanpy as sc
import numpy as np
from empty_drops_supernova import empty_drops

def test_r_compatibility():
    """Test that SUPERNOVA now matches R's behavior."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    # Get totals for analysis
    totals = np.array(raw_adata.X.sum(axis=1)).flatten()

    print(f"Total barcodes: {len(totals):,}")

    # Test the expected behavior
    lower = 100
    retain = 12615  # R's exact retain threshold

    # Calculate expected tested barcodes (R's logic)
    expected_tested = np.sum(totals > lower)
    expected_retained = np.sum(totals >= retain)

    print("\n=== Expected R Behavior ===")
    print(f"Barcodes > {lower}: {expected_tested:,}")
    print(f"Barcodes >= {retain}: {expected_retained:,}")
    print(f"Expected tested barcodes: {expected_tested - expected_retained:,}")

    # Now test our implementation
    print("\n=== Testing SUPERNOVA Implementation ===")

    # Run with R-compatible parameters
    results = empty_drops(
        data=raw_adata,
        lower=100,
        retain=12615,  # Force R's exact retain threshold
        niters=1000,   # Use fewer iterations for faster testing
        progress=True,
        visualize=False,  # Skip visualizations for speed
        use_cache=False   # Don't use cache for fresh test
    )

    # Analyze results
    tested_barcodes = results['PValue'].notna().sum()
    significant_cells = (results['FDR'] <= 0.001).sum()

    print("\n=== SUPERNOVA Results ===")
    print(f"Tested barcodes: {tested_barcodes:,}")
    print(f"Significant cells (FDR ≤ 0.001): {significant_cells:,}")

    # Check if we match R's behavior
    print("\n=== Compatibility Check ===")
    if tested_barcodes == (expected_tested - expected_retained):
        print("✅ SUCCESS: Tested barcodes match R's expectation!")
    else:
        print(f"❌ MISMATCH: Expected {expected_tested - expected_retained:,}, got {tested_barcodes:,}")

    if abs(tested_barcodes - 33393) <= 100:  # Allow small variation due to data differences
        print("✅ SUCCESS: Close to R's 33,393 tested barcodes!")
    else:
        print(f"⚠️  WARNING: {tested_barcodes:,} differs from R's 33,393")

    return results

if __name__ == "__main__":
    test_r_compatibility()
