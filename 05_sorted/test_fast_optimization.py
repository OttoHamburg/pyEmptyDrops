#!/usr/bin/env python3
"""
Fast EmptyDrops Optimization Test Script

This script provides a quick way to test the enhanced optuna implementation
with fewer trials for development and validation purposes.
"""

import os
import sys
import time
import logging
from datetime import datetime
import scanpy as sc
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append('.')

from hyperparameter_optimization import (
    EmptyDropsOptimizer, 
    run_optimization_with_visualization,
    calculate_multi_gene_score,
    OPTIMIZATION_DIR, RESULTS_DIR, PLOTS_DIR
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def quick_optimization_test(n_trials=5, data_subset_size=None):
    """
    Run a quick optimization test with reduced trials and optional data subsetting.
    
    Parameters:
    -----------
    n_trials : int
        Number of optimization trials (default: 5 for quick testing)
    data_subset_size : int, optional
        Number of cells to use for testing (None = use all data)
    """
    print("=== Fast EmptyDrops Optimization Test ===\n")
    
    # Load data
    print("Loading data...")
    try:
        raw_adata = sc.read_10x_h5("data/raw_feature_bc_matrix.h5")
        print(f"Original data shape: {raw_adata.shape}")
        
        # Optionally subset data for faster testing
        if data_subset_size and data_subset_size < len(raw_adata):
            # Select cells with highest total UMI counts for testing
            total_counts = np.array(raw_adata.X.sum(axis=1)).flatten()
            top_cells_idx = np.argsort(total_counts)[-data_subset_size:]
            raw_adata = raw_adata[top_cells_idx].copy()
            print(f"Using subset of {data_subset_size} cells for testing")
            print(f"Subset data shape: {raw_adata.shape}")
        
    except FileNotFoundError:
        print("Error: Could not find 'data/raw_feature_bc_matrix.h5'")
        print("Please ensure the data file exists in the correct location.")
        return None
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None
    
    # Test different optimization configurations
    configs = [
        {
            'name': 'Multi-Objective PRM1+TNP1',
            'target_genes': ['PRM1', 'TNP1'],
            'multi_objective': True,
            'convergence_patience': 3
        }
    ]
    
    results_summary = []
    
    for config in configs:
        print(f"\n--- Testing {config['name']} ---")
        
        # Create study name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"test_{config['name'].lower().replace(' ', '_').replace('+', '_')}_{timestamp}"
        
        start_time = time.time()
        
        try:
            # Initialize optimizer
            optimizer = EmptyDropsOptimizer(
                raw_adata, 
                target_genes=config['target_genes'],
                study_name=study_name,
                suppress_output=True,
                multi_objective=config['multi_objective'],
                convergence_patience=config['convergence_patience']
            )
            
            # Run optimization
            print(f"Running {n_trials} trials...")
            study = optimizer.optimize(n_trials=n_trials)
            
            optimization_time = time.time() - start_time
            
            # Collect results
            result_info = {
                'configuration': config['name'],
                'study_name': study_name,
                'n_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'best_score': study.best_value,
                'best_params': study.best_params,
                'optimization_time': optimization_time,
                'early_stopped': optimizer.early_stopped,
                'convergence_counter': optimizer.convergence_counter
            }
            
            results_summary.append(result_info)
            
            print(f"✅ {config['name']} completed:")
            print(f"   Best score: {study.best_value:.2f}")
            print(f"   Completed trials: {result_info['completed_trials']}/{n_trials}")
            print(f"   Time: {optimization_time:.1f}s")
            print(f"   Early stopped: {optimizer.early_stopped}")
            
            # Test final parameters
            if study.best_params:
                print(f"   Testing best parameters...")
                test_results = test_best_parameters(raw_adata, study.best_params, config['target_genes'])
                result_info['test_results'] = test_results
                
        except Exception as e:
            print(f"❌ Error in {config['name']}: {str(e)}")
            logger.error(f"Configuration {config['name']} failed: {str(e)}")
    
    # Print summary
    print("\n=== OPTIMIZATION TEST SUMMARY ===")
    for result in results_summary:
        print(f"\n{result['configuration']}:")
        print(f"  Best Score: {result['best_score']:.2f}")
        print(f"  Time: {result['optimization_time']:.1f}s")
        print(f"  Trials: {result['completed_trials']}/{result['n_trials']}")
        print(f"  Early Stopped: {result['early_stopped']}")
        
        if 'test_results' in result:
            test = result['test_results']
            print(f"  Final Test - Called Cells: {test['total_called_cells']}")
            for gene in result['test_results']:
                if gene in ['PRM1', 'TNP1']:
                    print(f"    {gene} target cells: {test.get(gene, 0)}")
    
    return results_summary

def test_best_parameters(raw_adata, best_params, target_genes):
    """Test the best parameters from optimization."""
    from empty_drops import empty_drops
    
    try:
        # Extract FDR threshold
        test_params = best_params.copy()
        fdr_threshold = test_params.pop('fdr_threshold', 0.01)
        
        # Run EmptyDrops with best parameters
        results = empty_drops(
            raw_adata,
            progress=False,
            visualize=False,
            fast_mode=True,
            **test_params
        )
        
        # Calculate final scores
        scores = calculate_multi_gene_score(
            results,
            raw_adata,
            target_genes=target_genes,
            fdr_threshold=fdr_threshold
        )
        
        return scores
        
    except Exception as e:
        logger.error(f"Error testing best parameters: {str(e)}")
        return {}

def performance_comparison():
    """Compare performance of different optimization strategies."""
    print("=== Performance Comparison ===\n")
    
    # Test with different data sizes
    data_sizes = [1000, 5000, 10000]
    n_trials = 3  # Very quick for comparison
    
    for size in data_sizes:
        print(f"Testing with {size} cells...")
        results = quick_optimization_test(n_trials=n_trials, data_subset_size=size)
        if results:
            total_time = sum(r['optimization_time'] for r in results)
            print(f"Total time for {size} cells: {total_time:.1f}s")
        print()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fast EmptyDrops Optimization Test')
    parser.add_argument('--trials', type=int, default=5, 
                       help='Number of optimization trials (default: 5)')
    parser.add_argument('--subset', type=int, default=None,
                       help='Use subset of N cells for faster testing')
    parser.add_argument('--compare', action='store_true',
                       help='Run performance comparison across different data sizes')
    
    args = parser.parse_args()
    
    if args.compare:
        performance_comparison()
    else:
        results = quick_optimization_test(n_trials=args.trials, data_subset_size=args.subset)
        
        if results:
            print(f"\n📊 Results saved to: {OPTIMIZATION_DIR}")
            print(f"📈 Plots available in: {PLOTS_DIR}")
            print(f"📄 Detailed results in: {RESULTS_DIR}") 