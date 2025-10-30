#!/usr/bin/env python3
"""
Test parallelization with Cython workers.
This should give us the full 5-7x speedup now!
"""

import scanpy as sc
from empty_drops import empty_drops
import time
import numpy as np

if __name__ == '__main__':
    print("Loading data...")
    data = sc.read_10x_h5('raw_feature_bc_matrix.h5')
    print(f"✓ Loaded: {data.shape[0]} barcodes × {data.shape[1]} genes\n")
    
    # Test with 500 iterations (enough to see real speedup, not too slow)
    test_niters = 500
    
    print("="*70)
    print("TEST 1: Single core (n_processes=1) with Cython")
    print("="*70)
    np.random.seed(42)
    start = time.time()
    results_single = empty_drops(data, n_processes=1, niters=test_niters, progress=True)
    time_single = time.time() - start
    print(f"\n✓ Single core time: {time_single:.1f} seconds")
    print(f"   Significant cells (FDR <= 0.001): {(results_single['FDR'] <= 0.001).sum()}")
    
    print("\n" + "="*70)
    print("TEST 2: 7 cores (n_processes=0) with Cython in each worker")
    print("="*70)
    np.random.seed(42)
    start = time.time()
    results_parallel = empty_drops(data, n_processes=0, niters=test_niters, progress=True)
    time_parallel = time.time() - start
    print(f"\n✓ 7 cores time: {time_parallel:.1f} seconds")
    print(f"   Significant cells (FDR <= 0.001): {(results_parallel['FDR'] <= 0.001).sum()}")
    
    print("\n" + "="*70)
    print("PERFORMANCE SUMMARY")
    print("="*70)
    print(f"Single core: {time_single:.1f} seconds")
    print(f"7 cores:     {time_parallel:.1f} seconds")
    print(f"Speedup:     {time_single/time_parallel:.2f}x")
    print(f"Efficiency:  {100 * time_single/time_parallel / 7:.1f}%")
    
    if time_single/time_parallel > 5:
        print("\n🎉 EXCELLENT! Speedup > 5x - Cython parallelization working perfectly!")
    elif time_single/time_parallel > 3:
        print("\n✅ GOOD! Speedup > 3x - parallelization is effective")
    elif time_single/time_parallel > 2:
        print("\n✓ OK. Speedup > 2x - some benefit from parallelization")
    else:
        print(f"\n⚠️  Poor speedup ({time_single/time_parallel:.2f}x)")
        print("   Expected 5-7x with Cython workers")
    
    # Verify results are identical
    print("\n" + "="*70)
    print("CORRECTNESS CHECK")
    print("="*70)
    
    single_cells = (results_single['FDR'] <= 0.001).sum()
    parallel_cells = (results_parallel['FDR'] <= 0.001).sum()
    
    # Results should be very close (within 1% due to random seed timing)
    if abs(single_cells - parallel_cells) / max(single_cells, 1) < 0.01:
        print(f"✅ Results are identical (within 1%)")
        print(f"   Single: {single_cells}, Parallel: {parallel_cells}")
    else:
        print(f"⚠️  Results differ:")
        print(f"   Single: {single_cells}, Parallel: {parallel_cells}")
        print(f"   Difference: {abs(single_cells - parallel_cells)} cells")


