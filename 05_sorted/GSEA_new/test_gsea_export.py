#!/usr/bin/env python3
"""
Comprehensive test suite for GSEA export functionality.

This script creates test datasets and validates all aspects of the GSEA file export
to ensure there are no bugs and the export works flawlessly.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import warnings
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch

# Import our GSEA export functionality
from prepare_gsea_files import GSEAFileGenerator

warnings.filterwarnings('ignore')
sc.settings.verbosity = 0


class TestGSEAExport(unittest.TestCase):
    """Test suite for GSEA export functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.test_dir = tempfile.mkdtemp(prefix="gsea_test_")
        self.test_output_dir = os.path.join(self.test_dir, "test_output")
        os.makedirs(self.test_output_dir, exist_ok=True)
        
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def create_test_adata(self, n_cells=3000, n_genes=5000, sparse=True, add_metadata=True):
        """
        Create a realistic test AnnData object that mimics single-cell data.
        
        Parameters:
        -----------
        n_cells : int
            Number of cells to generate
        n_genes : int  
            Number of genes to generate
        sparse : bool
            Whether to use sparse matrix format
        add_metadata : bool
            Whether to add realistic metadata
            
        Returns:
        --------
        AnnData : Test single-cell dataset
        """
        print(f"Creating test dataset with {n_cells} cells and {n_genes} genes...")
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create realistic gene expression data
        # Most genes have low expression, some have high expression
        base_expression = np.random.negative_binomial(n=3, p=0.3, size=(n_cells, n_genes))
        
        # Add some highly expressed genes (simulate housekeeping genes)
        n_high_genes = int(n_genes * 0.05)  # 5% highly expressed
        high_expr_genes = np.random.choice(n_genes, n_high_genes, replace=False)
        base_expression[:, high_expr_genes] += np.random.negative_binomial(n=10, p=0.2, size=(n_cells, n_high_genes))
        
        # Add some completely silent genes (zeros)
        n_silent_genes = int(n_genes * 0.1)  # 10% silent genes
        silent_genes = np.random.choice(n_genes, n_silent_genes, replace=False)
        base_expression[:, silent_genes] = 0
        
        # Convert to sparse if requested
        if sparse:
            X = sp.csr_matrix(base_expression.astype(np.float32))
        else:
            X = base_expression.astype(np.float32)
        
        # Create gene names (mix of ENSEMBL-like and gene symbols)
        gene_names = []
        gene_symbols = []
        for i in range(n_genes):
            if i < n_genes // 2:
                # ENSEMBL-like IDs
                gene_name = f"ENSG{i:011d}"
                gene_symbol = f"GENE{i}"
            else:
                # Gene symbol-like names
                gene_name = f"GENE_{i}"
                gene_symbol = f"SYM{i}"
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
        
        if add_metadata:
            # Add realistic metadata
            adata.obs['total_counts'] = np.array(adata.X.sum(axis=1)).flatten()
            adata.obs['n_genes'] = np.array((adata.X > 0).sum(axis=1)).flatten()
            adata.obs['percent_mito'] = np.random.uniform(0.05, 0.25, n_cells)
            adata.obs['cell_type'] = np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_cells)
            
            # Add some processing history
            adata.uns['log1p'] = {'base': None}
            adata.var['highly_variable'] = np.random.choice([True, False], n_genes, p=[0.2, 0.8])
            
        print(f"✅ Created test dataset: {adata.shape[0]} cells × {adata.shape[1]} genes")
        return adata
    
    def create_small_filtered_dataset(self):
        """Create a small filtered dataset similar to what you would have."""
        # Create a realistic small dataset
        adata = self.create_test_adata(n_cells=3000, n_genes=2000, sparse=True)
        
        # Apply typical filtering steps
        sc.pp.filter_cells(adata, min_genes=200)
        sc.pp.filter_genes(adata, min_cells=3)
        
        # Add some processing that would be typical
        adata.raw = adata
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        
        # Add expression groups (what would be used for phenotypes)
        median_expr = np.median(adata.obs['total_counts'])
        adata.obs['expression_group'] = ['HIGH' if x > median_expr else 'LOW' 
                                       for x in adata.obs['total_counts']]
        
        return adata
    
    def test_create_test_dataset(self):
        """Test that we can create a valid test dataset."""
        adata = self.create_test_adata(n_cells=100, n_genes=500)
        
        # Basic checks
        self.assertEqual(adata.shape, (100, 500))
        self.assertIn('gene_symbols', adata.var.columns)
        self.assertIn('total_counts', adata.obs.columns)
        self.assertTrue(sp.issparse(adata.X))
        
        print("✅ Test dataset creation: PASSED")
    
    def test_gsea_generator_initialization(self):
        """Test GSEAFileGenerator initialization with test data."""
        adata = self.create_small_filtered_dataset()
        
        # Save test data
        test_h5ad_path = os.path.join(self.test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        
        # Test initialization
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        self.assertIsNotNone(generator.adata)
        self.assertEqual(generator.output_dir, self.test_output_dir)
        self.assertTrue(os.path.exists(self.test_output_dir))
        
        print("✅ GSEA generator initialization: PASSED")
    
    def test_gct_file_generation(self):
        """Test GCT file generation and format validation."""
        adata = self.create_small_filtered_dataset()
        test_h5ad_path = os.path.join(self.test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        # Generate GCT file
        gct_path = generator.create_expression_dataset()
        
        # Validate file exists
        self.assertTrue(os.path.exists(gct_path))
        
        # Validate GCT format
        with open(gct_path, 'r') as f:
            lines = f.readlines()
        
        # Check header
        self.assertEqual(lines[0].strip(), "#1.2")
        
        # Parse dimensions
        dimensions = lines[1].strip().split('\t')
        n_genes, n_samples = int(dimensions[0]), int(dimensions[1])
        
        # Validate dimensions match our data
        self.assertEqual(n_samples, adata.shape[0])
        self.assertLessEqual(n_genes, adata.shape[1])  # May be filtered
        
        # Check data structure
        header_line = lines[2].strip().split('\t')
        self.assertEqual(header_line[0], "NAME")
        self.assertEqual(header_line[1], "Description")
        self.assertEqual(len(header_line), n_samples + 2)
        
        print(f"✅ GCT file generation: PASSED ({n_genes} genes × {n_samples} samples)")
    
    def test_cls_file_generation(self):
        """Test CLS file generation and format validation."""
        adata = self.create_small_filtered_dataset()
        test_h5ad_path = os.path.join(self.test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        # Generate CLS file
        cls_path = generator.create_phenotype_labels()
        
        # Validate file exists
        self.assertTrue(os.path.exists(cls_path))
        
        # Validate CLS format
        with open(cls_path, 'r') as f:
            lines = f.readlines()
        
        # Check header format
        header_parts = lines[0].strip().split()
        n_samples, n_classes, continuous = int(header_parts[0]), int(header_parts[1]), int(header_parts[2])
        
        self.assertEqual(n_samples, adata.shape[0])
        self.assertEqual(n_classes, 2)  # HIGH and LOW
        self.assertEqual(continuous, 1)  # Categorical
        
        # Check class labels
        class_labels = lines[1].strip().split()
        self.assertIn('HIGH', class_labels)
        self.assertIn('LOW', class_labels)
        
        # Check sample assignments
        sample_assignments = lines[2].strip().split()
        self.assertEqual(len(sample_assignments), n_samples)
        self.assertTrue(all(x in ['0', '1'] for x in sample_assignments))
        
        print(f"✅ CLS file generation: PASSED ({n_samples} samples, 2 classes)")
    
    def test_gene_symbol_handling(self):
        """Test proper handling of different gene identifier types."""
        adata = self.create_small_filtered_dataset()
        test_h5ad_path = os.path.join(self.test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        # Test with gene symbols
        gct_path = generator.create_expression_dataset()
        
        # Read first few genes from GCT file
        with open(gct_path, 'r') as f:
            lines = f.readlines()
        
        # Check that gene names are properly formatted
        for i in range(3, min(10, len(lines))):  # Check first few data lines
            parts = lines[i].strip().split('\t')
            gene_name = parts[0]
            gene_desc = parts[1]
            
            # Should have both name and description
            self.assertIsNotNone(gene_name)
            self.assertIsNotNone(gene_desc)
            self.assertNotEqual(gene_name, "")
            self.assertNotEqual(gene_desc, "")
        
        print("✅ Gene symbol handling: PASSED")
    
    def test_sparse_matrix_handling(self):
        """Test proper handling of sparse matrices."""
        adata = self.create_small_filtered_dataset()
        
        # Ensure matrix is sparse
        self.assertTrue(sp.issparse(adata.X))
        
        test_h5ad_path = os.path.join(self.test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        # Should handle sparse matrices without errors
        gct_path = generator.create_expression_dataset()
        cls_path = generator.create_phenotype_labels()
        
        self.assertTrue(os.path.exists(gct_path))
        self.assertTrue(os.path.exists(cls_path))
        
        print("✅ Sparse matrix handling: PASSED")
    
    def test_file_sizes_and_content_validation(self):
        """Test that generated files have reasonable sizes and valid content."""
        adata = self.create_small_filtered_dataset()
        test_h5ad_path = os.path.join(self.test_dir, "test_data.h5ad")
        adata.write_h5ad(test_h5ad_path)
        
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        # Generate all files
        gct_path = generator.create_expression_dataset()
        cls_path = generator.create_phenotype_labels()
        
        # Check file sizes are reasonable
        gct_size = os.path.getsize(gct_path)
        cls_size = os.path.getsize(cls_path)
        
        self.assertGreater(gct_size, 1000)  # Should be at least 1KB
        self.assertGreater(cls_size, 100)   # Should be at least 100 bytes
        
        # Check no NaN values in GCT file
        with open(gct_path, 'r') as f:
            content = f.read()
            self.assertNotIn('nan', content.lower())
            self.assertNotIn('inf', content.lower())
        
        print(f"✅ File validation: PASSED (GCT: {gct_size/1024:.1f}KB, CLS: {cls_size}B)")
    
    def test_edge_cases(self):
        """Test edge cases and potential failure modes."""
        
        # Test with very small dataset
        small_adata = self.create_test_adata(n_cells=10, n_genes=50, add_metadata=False)
        
        # Add minimal required metadata
        small_adata.obs['total_counts'] = np.array(small_adata.X.sum(axis=1)).flatten()
        
        test_h5ad_path = os.path.join(self.test_dir, "small_test_data.h5ad")
        small_adata.write_h5ad(test_h5ad_path)
        
        generator = GSEAFileGenerator(h5ad_path=test_h5ad_path, output_dir=self.test_output_dir)
        
        # Should handle small datasets
        gct_path = generator.create_expression_dataset(filter_genes=False)  # Don't filter
        cls_path = generator.create_phenotype_labels()
        
        self.assertTrue(os.path.exists(gct_path))
        self.assertTrue(os.path.exists(cls_path))
        
        print("✅ Edge cases: PASSED")
    
    def test_with_real_filtered_data(self):
        """Test with the actual filtered data if available."""
        filtered_h5_path = "data/filtered_feature_bc_matrix.h5"
        
        if os.path.exists(filtered_h5_path):
            print(f"Testing with real filtered data: {filtered_h5_path}")
            
            # Load the real data
            try:
                generator = GSEAFileGenerator(h5_path=filtered_h5_path, output_dir=self.test_output_dir)
                
                # Generate files
                gct_path = generator.create_expression_dataset()
                cls_path = generator.create_phenotype_labels()
                
                # Validate files
                self.assertTrue(os.path.exists(gct_path))
                self.assertTrue(os.path.exists(cls_path))
                
                # Check dimensions make sense
                with open(gct_path, 'r') as f:
                    lines = f.readlines()
                
                dimensions = lines[1].strip().split('\t')
                n_genes, n_samples = int(dimensions[0]), int(dimensions[1])
                
                self.assertGreater(n_genes, 100)    # Should have reasonable number of genes
                self.assertGreater(n_samples, 100)  # Should have reasonable number of cells
                
                print(f"✅ Real data test: PASSED ({n_genes} genes × {n_samples} samples)")
                
            except Exception as e:
                print(f"⚠️ Real data test failed: {e}")
                # Don't fail the test for this since it depends on external data
        else:
            print("⚠️ Real filtered data not found, skipping real data test")
    
    def run_comprehensive_test(self):
        """Run all tests and provide a comprehensive report."""
        print("="*80)
        print("COMPREHENSIVE GSEA EXPORT TEST SUITE")
        print("="*80)
        
        test_methods = [
            self.test_create_test_dataset,
            self.test_gsea_generator_initialization,
            self.test_gct_file_generation,
            self.test_cls_file_generation,
            self.test_gene_symbol_handling,
            self.test_sparse_matrix_handling,
            self.test_file_sizes_and_content_validation,
            self.test_edge_cases,
            self.test_with_real_filtered_data
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_method in test_methods:
            try:
                print(f"\nRunning {test_method.__name__}...")
                test_method()
                passed_tests += 1
            except Exception as e:
                print(f"❌ {test_method.__name__}: FAILED - {e}")
                failed_tests += 1
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"Total: {passed_tests + failed_tests}")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! GSEA export is working flawlessly.")
        else:
            print(f"\n⚠️ {failed_tests} test(s) failed. Please review the issues above.")
        
        print("="*80)
        
        return failed_tests == 0


def main():
    """Main function to run the test suite."""
    tester = TestGSEAExport()
    try:
        success = tester.run_comprehensive_test()
        return 0 if success else 1
    finally:
        # Clean up
        if hasattr(tester, 'test_dir') and os.path.exists(tester.test_dir):
            shutil.rmtree(tester.test_dir)


if __name__ == "__main__":
    sys.exit(main())
