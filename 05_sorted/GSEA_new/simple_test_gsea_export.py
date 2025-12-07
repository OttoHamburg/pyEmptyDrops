#!/usr/bin/env python3
"""
Simple but comprehensive test for GSEA export functionality.
This script creates a test dataset and validates the complete export process.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import tempfile
import shutil
import warnings

# Import our GSEA export functionality
from prepare_gsea_files import GSEAFileGenerator

warnings.filterwarnings('ignore')
sc.settings.verbosity = 0

def create_realistic_test_data(n_cells=3000, n_genes=2000):
    """Create a realistic test dataset that mimics single-cell data."""
    print(f"Creating realistic test dataset: {n_cells} cells × {n_genes} genes...")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    # Create realistic gene expression matrix
    # Most genes have low expression with occasional high values
    base_counts = np.random.negative_binomial(n=2, p=0.3, size=(n_cells, n_genes))
    
    # Add some highly expressed genes (housekeeping-like)
    n_high_genes = int(n_genes * 0.05)
    high_expr_indices = np.random.choice(n_genes, n_high_genes, replace=False)
    base_counts[:, high_expr_indices] += np.random.negative_binomial(n=10, p=0.2, size=(n_cells, n_high_genes))
    
    # Add some silent genes
    n_silent_genes = int(n_genes * 0.1)
    silent_indices = np.random.choice(n_genes, n_silent_genes, replace=False)
    base_counts[:, silent_indices] = 0
    
    # Convert to sparse matrix
    X = sp.csr_matrix(base_counts.astype(np.float32))
    
    # Create gene names (mix of ENSEMBL-like and symbols)
    gene_names = []
    gene_symbols = []
    for i in range(n_genes):
        gene_name = f"ENSG{i:011d}" if i < n_genes//2 else f"GENE_{i}"
        gene_symbol = f"SYMBOL_{i}"
        gene_names.append(gene_name)
        gene_symbols.append(gene_symbol)
    
    # Create cell barcodes
    cell_barcodes = [f"CELL_{i:06d}-1" for i in range(n_cells)]
    
    # Create AnnData object
    adata = sc.AnnData(
        X=X,
        obs=pd.DataFrame(index=cell_barcodes),
        var=pd.DataFrame(
            {'gene_symbols': gene_symbols},
            index=gene_names
        )
    )
    
    # Add realistic metadata
    adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
    adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
    adata.obs['percent_mito'] = np.random.uniform(0.05, 0.25, n_cells)
    
    # Apply basic filtering
    sc.pp.filter_cells(adata, min_genes=50)
    sc.pp.filter_genes(adata, min_cells=3)
    
    # Add processing steps
    adata.raw = adata
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    
    # Create expression groups for phenotypes
    median_expr = np.median(adata.obs['total_counts'])
    adata.obs['expression_group'] = ['HIGH' if x > median_expr else 'LOW' 
                                   for x in adata.obs['total_counts']]
    
    print(f"✅ Created dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
    return adata

def validate_gct_file(gct_path):
    """Validate GCT file format and content."""
    print(f"🔍 Validating GCT file: {os.path.basename(gct_path)}")
    
    if not os.path.exists(gct_path):
        raise ValueError("GCT file does not exist")
    
    with open(gct_path, 'r') as f:
        lines = f.readlines()
    
    # Check header
    if lines[0].strip() != "#1.2":
        raise ValueError(f"Invalid GCT version: {lines[0].strip()}")
    
    # Parse dimensions
    dimensions = lines[1].strip().split('\t')
    n_genes, n_samples = int(dimensions[0]), int(dimensions[1])
    
    # Check header line
    header_parts = lines[2].strip().split('\t')
    if len(header_parts) != n_samples + 2:
        raise ValueError(f"Header mismatch: expected {n_samples + 2}, got {len(header_parts)}")
    
    if header_parts[0] != "NAME" or header_parts[1] != "Description":
        raise ValueError("Invalid header columns")
    
    # Check data lines
    data_lines = len(lines) - 3
    if data_lines != n_genes:
        raise ValueError(f"Data line mismatch: expected {n_genes}, got {data_lines}")
    
    # Validate first few data lines
    for i in range(3, min(8, len(lines))):
        parts = lines[i].strip().split('\t')
        if len(parts) != n_samples + 2:
            raise ValueError(f"Data line {i-2} has wrong number of columns")
        
        # Check that values are numeric
        for val in parts[2:5]:  # Check first few values
            try:
                float_val = float(val)
                if np.isnan(float_val) or np.isinf(float_val):
                    raise ValueError(f"Invalid numeric value: {val}")
            except ValueError:
                raise ValueError(f"Non-numeric value: {val}")
    
    print(f"  ✅ Valid GCT format: {n_genes:,} genes × {n_samples:,} samples")
    print(f"  ✅ File size: {os.path.getsize(gct_path) / (1024*1024):.2f} MB")
    return n_genes, n_samples

def validate_cls_file(cls_path, expected_samples):
    """Validate CLS file format and content."""
    print(f"🔍 Validating CLS file: {os.path.basename(cls_path)}")
    
    if not os.path.exists(cls_path):
        raise ValueError("CLS file does not exist")
    
    with open(cls_path, 'r') as f:
        lines = f.readlines()
    
    if len(lines) < 3:
        raise ValueError("CLS file too short")
    
    # Parse header
    header_parts = lines[0].strip().split()
    n_samples, n_classes, continuous = int(header_parts[0]), int(header_parts[1]), int(header_parts[2])
    
    if n_samples != expected_samples:
        raise ValueError(f"Sample count mismatch: expected {expected_samples}, got {n_samples}")
    
    if n_classes != 2:
        raise ValueError(f"Expected 2 classes, got {n_classes}")
    
    if continuous != 1:
        raise ValueError(f"Expected continuous=1, got {continuous}")
    
    # Check class labels
    class_labels = lines[1].strip().split()
    if len(class_labels) != 2:
        raise ValueError(f"Expected 2 class labels, got {len(class_labels)}")
    
    # Remove # prefix if present (CLS format uses # in header)
    clean_labels = [label.lstrip('#') for label in class_labels]
    if 'HIGH' not in clean_labels or 'LOW' not in clean_labels:
        raise ValueError(f"Expected HIGH and LOW labels, got {clean_labels}")
    
    # Check sample assignments
    assignments = lines[2].strip().split()
    if len(assignments) != n_samples:
        raise ValueError(f"Assignment count mismatch: expected {n_samples}, got {len(assignments)}")
    
    # Count class distribution
    class_counts = {'0': 0, '1': 0}
    for assignment in assignments:
        if assignment not in ['0', '1']:
            raise ValueError(f"Invalid assignment: {assignment}")
        class_counts[assignment] += 1
    
    print(f"  ✅ Valid CLS format: {n_samples:,} samples, 2 classes")
    print(f"  ✅ Class distribution: {class_counts}")
    return class_counts

def run_comprehensive_test():
    """Run the comprehensive test suite."""
    print("🧪 COMPREHENSIVE GSEA EXPORT TEST")
    print("="*60)
    
    # Create temporary directory
    test_dir = tempfile.mkdtemp(prefix="gsea_test_")
    output_dir = os.path.join(test_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Test 1: Create realistic test data
        print("\n1️⃣ Creating realistic test dataset...")
        adata = create_realistic_test_data(n_cells=3000, n_genes=2000)
        
        # Save test data
        test_h5ad_path = os.path.join(test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        print(f"  ✅ Saved test data: {test_h5ad_path}")
        
        # Test 2: Initialize GSEA generator
        print("\n2️⃣ Initializing GSEA generator...")
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=output_dir)
        print(f"  ✅ Generator initialized with {generator.adata.shape[0]} cells × {generator.adata.shape[1]} genes")
        
        # Test 3: Generate GCT file
        print("\n3️⃣ Generating expression dataset (GCT)...")
        gct_path = generator.create_expression_dataset_gct(
            filter_genes=True,
            min_cells=10,
            use_raw=False
        )
        print(f"  ✅ GCT file created: {os.path.basename(gct_path)}")
        
        # Test 4: Validate GCT file
        print("\n4️⃣ Validating GCT file...")
        n_genes, n_samples = validate_gct_file(gct_path)
        
        # Test 5: Generate CLS file
        print("\n5️⃣ Generating phenotype labels (CLS)...")
        # First create phenotype groups
        generator.create_phenotype_groups(method="expression_level")
        cls_path = generator.create_phenotype_labels_cls()
        print(f"  ✅ CLS file created: {os.path.basename(cls_path)}")
        
        # Test 6: Validate CLS file
        print("\n6️⃣ Validating CLS file...")
        class_counts = validate_cls_file(cls_path, n_samples)
        
        # Test 7: File compatibility test
        print("\n7️⃣ Testing file compatibility...")
        
        # Check that files can be read properly
        with open(gct_path, 'r') as f:
            gct_content = f.read()
        
        with open(cls_path, 'r') as f:
            cls_content = f.read()
        
        if 'nan' in gct_content.lower() or 'inf' in gct_content.lower():
            raise ValueError("GCT file contains invalid values")
        
        print("  ✅ Files are compatible with GSEA format")
        
        # Test 8: Edge case testing
        print("\n8️⃣ Testing edge cases...")
        
        # Test with very small dataset
        small_adata = create_realistic_test_data(n_cells=100, n_genes=200)
        small_test_path = os.path.join(test_dir, "small_test.h5ad")
        small_adata.write_h5ad(small_test_path)
        
        small_generator = GSEAFileGenerator(h5ad_path=small_test_path, output_dir=output_dir)
        small_gct = small_generator.create_expression_dataset_gct(filter_genes=False)
        small_generator.create_phenotype_groups()  # Create groups first
        small_cls = small_generator.create_phenotype_labels_cls()
        
        if not os.path.exists(small_gct) or not os.path.exists(small_cls):
            raise ValueError("Edge case test failed")
        
        print("  ✅ Edge cases handled correctly")
        
        # Final success message
        print("\n🎉 ALL TESTS PASSED!")
        print("="*60)
        print("GSEA export functionality is working flawlessly!")
        print(f"📊 Final dataset: {n_genes:,} genes × {n_samples:,} samples")
        print(f"📁 Files generated:")
        print(f"  - Expression dataset: {os.path.basename(gct_path)}")
        print(f"  - Phenotype labels: {os.path.basename(cls_path)}")
        print("✅ Ready for GSEA analysis!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Clean up
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)

def test_with_real_data():
    """Test with real filtered data if available."""
    print("\n🔬 TESTING WITH REAL DATA")
    print("="*40)
    
    # Check for available data files
    data_files = [
        "data/emptydrops_all_detected_cells.h5ad",
        "data/filtered_feature_bc_matrix.h5"
    ]
    
    available_file = None
    for file_path in data_files:
        if os.path.exists(file_path):
            available_file = file_path
            break
    
    if not available_file:
        print("⚠️ No real data files found, skipping real data test")
        return True
    
    print(f"📂 Using real data: {available_file}")
    
    try:
        # Create output directory
        output_dir = "real_data_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize generator
        if available_file.endswith('.h5ad'):
            generator = GSEAFileGenerator(h5ad_path=available_file, output_dir=output_dir)
        else:
            generator = GSEAFileGenerator(h5_path=available_file, output_dir=output_dir)
        
        n_cells, n_genes = generator.adata.shape
        print(f"📊 Real dataset: {n_cells:,} cells × {n_genes:,} genes")
        
        # Ensure required metadata exists
        if 'total_counts' not in generator.adata.obs.columns:
            generator.adata.obs['total_counts'] = np.array(generator.adata.X.sum(axis=1)).flatten()
        
        # Generate files
        gct_path = generator.create_expression_dataset_gct()
        generator.create_phenotype_groups()  # Create groups first
        cls_path = generator.create_phenotype_labels_cls()
        
        # Validate
        validate_gct_file(gct_path)
        validate_cls_file(cls_path, n_cells)
        
        print("✅ Real data test PASSED!")
        print(f"📁 Files ready in: {os.path.abspath(output_dir)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Real data test failed: {e}")
        return False

def main():
    """Main test function."""
    print("🧬 GSEA EXPORT VALIDATION SUITE")
    print("="*80)
    
    # Run comprehensive test
    test1_passed = run_comprehensive_test()
    
    # Run real data test
    test2_passed = test_with_real_data()
    
    # Summary
    print(f"\n📋 TEST SUMMARY")
    print("="*40)
    print(f"Comprehensive test: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Real data test: {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    
    if test1_passed and test2_passed:
        print("\n🎉 ALL TESTS SUCCESSFUL!")
        print("Your GSEA export is working perfectly!")
    else:
        print("\n⚠️ Some tests failed. Please review the output above.")
    
    return 0 if (test1_passed and test2_passed) else 1

if __name__ == "__main__":
    sys.exit(main())
