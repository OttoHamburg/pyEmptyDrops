#!/usr/bin/env python3
"""
Test the final SUPERNOVA version against R's exact results.
"""

import scanpy as sc
import numpy as np
from empty_drops_supernova import empty_drops

def test_supernova_final():
    """Test the final SUPERNOVA version."""

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

    print("\n=== Testing SUPERNOVA EmptyDrops (Final Version) ===")

    # Run with exact same parameters as R validation
    results = empty_drops(
        data=raw_adata,
        lower=100,
        retain=None,  # Let it calculate the knee point
        niters=10000,  # Same as R
        alpha=np.inf,
        test_ambient=False,  # Same as R
        ignore=None,  # Same as R
        round=True,  # Same as R
        by_rank=None,  # Same as R
        progress=True,
        visualize=False,  # Skip visualizations for speed
        dataset_name="supernova_final_test"
    )

    print(f"\n=== SUPERNOVA Results ===")
    print(f"Total barcodes: {len(results):,}")

    # Count tested barcodes (non-NA p-values)
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

    print(f"\n=== Comparison with R Validation ===")
    print(f"  R tested barcodes: 33,393")
    print(f"  SUPERNOVA tested: {tested_count:,}")
    print(f"  Difference: {abs(tested_count - 33393):,}")

    print(f"  R cells FDR ≤ 0.001: 23,452")
    print(f"  SUPERNOVA cells: {fdr_001:,}")
    print(f"  Difference: {abs(fdr_001 - 23452):,}")

    print(f"  R cells FDR ≤ 0.01: 25,053")
    print(f"  SUPERNOVA cells: {fdr_01:,}")
    print(f"  Difference: {abs(fdr_01 - 25053):,}")

    print(f"  R cells FDR ≤ 0.05: 26,188")
    print(f"  SUPERNOVA cells: {fdr_05:,}")
    print(f"  Difference: {abs(fdr_05 - 26188):,}")

    # Check the knee point from the results
    metadata_path = "emptydrops_runs/supernova_final_test/20251009_172500_run/metadata/run_metadata.json"
    try:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        knee_point = metadata['parameters']['retain']
        print(f"\nSUPERNOVA knee point: {knee_point}")

        diff_from_r = abs(knee_point - 12615)
        if diff_from_r <= 500:
            print(f"🎉 KNEE POINT SUCCESS! {knee_point} is within ±500 of R's 12,615!")
        else:
            print(f"❌ Knee point {knee_point} differs by {diff_from_r:.1f} from R's 12,615")

    except Exception as e:
        print(f"Could not read knee point from metadata: {e}")

if __name__ == "__main__":
    test_supernova_final()
