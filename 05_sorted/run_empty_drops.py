"""
Run EmptyDrops analysis on raw feature-barcode matrix and validate against filtered dataset.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from empty_drops import empty_drops
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import time
import os

def load_matrices():
    """Load raw and filtered feature-barcode matrices."""
    print("Loading matrices...")
    start_time = time.time()
    
    # Load raw matrix
    raw_adata = sc.read_10x_h5('data/raw_feature_bc_matrix.h5')
    print(f"Raw matrix loaded: {raw_adata.shape}")
    
    # Load filtered matrix for validation
    filtered_adata = sc.read_10x_h5('data/filtered_feature_bc_matrix.h5')
    print(f"Filtered matrix loaded: {filtered_adata.shape}")
    
    load_time = time.time() - start_time
    print(f"Loading completed in {load_time:.2f} seconds")
    
    return raw_adata, filtered_adata

def run_empty_drops_analysis(raw_adata, lower=150, retain=1000):
    """Run EmptyDrops analysis with parameters optimized for background removal.
    
    Parameters chosen to minimize false positives (empty droplets called as cells):
    - lower=150: Moderate threshold for empty droplets
    - retain=1000: Automatically keep only very high-count droplets
    - alpha=0.5: Stricter ambient profile modeling
    - FDR=0.001: Very stringent false discovery rate
    """
    print("\nRunning EmptyDrops analysis with background-removal optimized parameters...")
    print(f"lower threshold: {lower}")
    print(f"retain threshold: {retain}")
    start_time = time.time()
    

    
    # Run EmptyDrops with practical parameters optimized for speed
    results = empty_drops(
        raw_adata,
        lower=lower,          # Moderate threshold for empty droplets
        retain=retain,        # Only very high-count droplets automatically kept
        test_ambient=True,
        niters=500,          # Reduced iterations for practical runtime
        alpha=0.5,           # Stricter ambient profile modeling
        progress=True,
        min_iters=100,
        max_iters=500,       # Reduced max iterations
        early_stopping=True,
        batch_size=100,      # Larger batch size for efficiency
        confidence_level=0.95 # Practical confidence level
    )
    
    analysis_time = time.time() - start_time
    print(f"Analysis completed in {analysis_time:.2f} seconds")
    
    return results

def validate_results(raw_adata, filtered_adata, results):
    """Validate the results of EmptyDrops by comparing with the filtered data."""
    print("\nValidating results...")
    
    # Get the barcodes that are in the filtered data (ground truth)
    filtered_barcodes = set(filtered_adata.obs_names)
    
    # Get the barcodes that EmptyDrops identified as cells using stricter FDR
    empty_drops_barcodes = set(results[results['FDR'] < 0.001].index)  # Much stricter FDR threshold
    
    # Calculate metrics
    true_positives = len(filtered_barcodes.intersection(empty_drops_barcodes))
    false_positives = len(empty_drops_barcodes - filtered_barcodes)
    true_negatives = len(set(raw_adata.obs_names) - filtered_barcodes - empty_drops_barcodes)
    false_negatives = len(filtered_barcodes - empty_drops_barcodes)
    
    # Create confusion matrix
    cm = np.array([[true_negatives, false_positives],
                   [false_negatives, true_positives]])
    
    # Calculate metrics
    total = true_positives + false_positives + true_negatives + false_negatives
    accuracy = (true_positives + true_negatives) / total if total > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print validation results
    print("\nValidation Results:")
    print(f"True Positives: {true_positives}")
    print(f"False Positives: {false_positives}")
    print(f"True Negatives: {true_negatives}")
    print(f"False Negatives: {false_negatives}")
    print(f"\nPrecision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1 Score: {f1_score:.3f}")
    print(f"Accuracy: {accuracy:.3f}")
    
    # Save confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Empty', 'Cell'],
                yticklabels=['Empty', 'Cell'])
    plt.title('Confusion Matrix (Strict Background Removal)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('empty_drops_visualizations/confusion_matrix.png')
    plt.close()
    
    # Additional visualization: UMI count distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=pd.DataFrame({
        'UMI_Counts': raw_adata.X.sum(axis=1).A1,
        'Category': ['Cell' if bc in empty_drops_barcodes else 'Empty' for bc in raw_adata.obs_names]
    }), x='UMI_Counts', hue='Category', bins=100, log_scale=True)
    plt.title('UMI Count Distribution by Category')
    plt.savefig('empty_drops_visualizations/umi_distribution.png')
    plt.close()
    
    return {
        'true_positives': true_positives,
        'false_positives': false_positives,
        'true_negatives': true_negatives,
        'false_negatives': false_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }

def main():
    """Main function to run the analysis pipeline."""
    # Load matrices
    raw_adata, filtered_adata = load_matrices()
    
    # Run EmptyDrops analysis with strict parameters
    results = run_empty_drops_analysis(raw_adata)
    
    # Validate results
    metrics = validate_results(raw_adata, filtered_adata, results)
    
    # Save results
    results.to_csv('empty_drops_visualizations/empty_drops_results.csv')
    
    print("\nAnalysis complete! Results saved to 'empty_drops_visualizations/empty_drops_results.csv'")
    print("Confusion matrix saved as 'empty_drops_visualizations/confusion_matrix.png'")
    print("UMI distribution plot saved as 'empty_drops_visualizations/umi_distribution.png'")

if __name__ == "__main__":
    main() 