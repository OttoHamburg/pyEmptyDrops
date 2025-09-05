#!/usr/bin/env python3
"""
Test the supernova version against R's knee point of 12,600.
"""

import scanpy as sc
import numpy as np
from empty_drops_supernova import empty_drops

def test_supernova():
    """Test the supernova version."""

    print("Loading data...")
    raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    raw_adata.var_names_make_unique()

    print(f"Data shape: {raw_adata.shape}")

    # Clear cache to force fresh run
    import os
    import shutil
    cache_dir = "empty_drops_cache"
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    print("\n=== Testing SUPERNOVA EmptyDrops (R-compatible knee detection) ===")

    # Run with supernova version
    results = empty_drops(
        data=raw_adata,
        lower=100,
        retain=None,  # Let it calculate the knee point
        niters=1000,  # Reduced for faster testing
        alpha=np.inf,
        test_ambient=False,
        progress=True,
        visualize=False,  # Skip visualizations for speed
        dataset_name="supernova_test"
    )

    print(f"\n=== SUPERNOVA Results ===")
    print(f"Total barcodes: {len(results):,}")

    # Count tested barcodes
    tested_mask = ~results['PValue'].isna()
    tested_count = tested_mask.sum()
    print(f"Tested barcodes: {tested_count:,}")

    # Count significant cells at different FDR thresholds
    fdr_001 = (results['FDR'] <= 0.001).sum()
    fdr_01 = (results['FDR'] <= 0.01).sum()
    fdr_05 = (results['FDR'] <= 0.05).sum()

    print(f"Significant cells:")
    print(f"  FDR ≤ 0.001: {fdr_001:,}")
    print(f"  FDR ≤ 0.01:  {fdr_01:,}")
    print(f"  FDR ≤ 0.05:  {fdr_05:,}")

    print(f"\n=== Comparison with R ===")
    print(f"  R tested: ~33,393 barcodes")
    print(f"  SUPERNOVA tested: {tested_count:,} barcodes")
    print(f"  Difference: {abs(tested_count - 33393):,}")

    # Check the knee point from the results
    # We need to extract it from the metadata or run metadata
    metadata_path = "emptydrops_runs/supernova_test/20251009_171500_run/metadata/run_metadata.json"
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        knee_point = metadata['parameters']['retain']
        print(f"\nSUPERNOVA knee point: {knee_point}")

        diff_from_r = abs(knee_point - 12600)
        if diff_from_r <= 500:
            print(f"🎉 SUCCESS! Knee point {knee_point} is within ±500 of R's 12,600!")
            print(f"Difference: {diff_from_r:.1f}")
        else:
            print(f"❌ Knee point {knee_point} differs by {diff_from_r:.1f} from R's 12,600")

    except Exception as e:
        print(f"Could not read knee point from metadata: {e}")

if __name__ == "__main__":
    test_supernova()
