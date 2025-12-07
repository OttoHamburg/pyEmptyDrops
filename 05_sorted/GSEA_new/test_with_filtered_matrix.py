#!/usr/bin/env python3
"""
Test GSEA export specifically with the filtered 3000-cell matrix.

This script tests the GSEA export functionality with your specific filtered dataset
and provides detailed validation of the output files.
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import warnings
from pathlib import Path

# Import our GSEA export functionality
from prepare_gsea_files import GSEAFileGenerator

warnings.filterwarnings('ignore')
sc.settings.verbosity = 1


def validate_gct_file(gct_path, expected_samples=None):
    """
    Thoroughly validate a GCT file format and content.
    
    Parameters:
    -----------
    gct_path : str
        Path to the GCT file
    expected_samples : int, optional
        Expected number of samples
        
    Returns:
    --------
    dict : Validation results
    """
    print(f"\n📋 Validating GCT file: {gct_path}")
    
    if not os.path.exists(gct_path):
        return {"valid": False, "error": "File does not exist"}
    
    try:
        with open(gct_path, 'r') as f:
            lines = f.readlines()
        
        # Check minimum file length
        if len(lines) < 4:
            return {"valid": False, "error": "File too short (less than 4 lines)"}
        
        # Validate header
        if not lines[0].strip() == "#1.2":
            return {"valid": False, "error": f"Invalid version header: {lines[0].strip()}"}
        
        # Parse dimensions
        try:
            dimensions = lines[1].strip().split('\t')
            n_genes, n_samples = int(dimensions[0]), int(dimensions[1])
        except:
            return {"valid": False, "error": "Cannot parse dimensions"}
        
        # Validate dimensions
        if n_genes <= 0 or n_samples <= 0:
            return {"valid": False, "error": f"Invalid dimensions: {n_genes} × {n_samples}"}
        
        if expected_samples and n_samples != expected_samples:
            return {"valid": False, "error": f"Sample count mismatch: expected {expected_samples}, got {n_samples}"}
        
        # Check header line
        header_parts = lines[2].strip().split('\t')
        if len(header_parts) != n_samples + 2:
            return {"valid": False, "error": f"Header length mismatch: expected {n_samples + 2}, got {len(header_parts)}"}
        
        if header_parts[0] != "NAME" or header_parts[1] != "Description":
            return {"valid": False, "error": "Invalid header columns"}
        
        # Check data lines
        expected_data_lines = n_genes
        actual_data_lines = len(lines) - 3  # Subtract header lines
        
        if actual_data_lines != expected_data_lines:
            return {"valid": False, "error": f"Data line count mismatch: expected {expected_data_lines}, got {actual_data_lines}"}
        
        # Validate first few data lines
        invalid_values = 0
        for i in range(3, min(3 + 10, len(lines))):  # Check first 10 data lines
            parts = lines[i].strip().split('\t')
            
            if len(parts) != n_samples + 2:
                return {"valid": False, "error": f"Data line {i-2} has wrong number of columns"}
            
            # Check gene name and description
            if not parts[0] or not parts[1]:
                return {"valid": False, "error": f"Missing gene name or description at line {i-2}"}
            
            # Check expression values
            for j, val in enumerate(parts[2:]):
                try:
                    float_val = float(val)
                    if np.isnan(float_val) or np.isinf(float_val):
                        invalid_values += 1
                except ValueError:
                    return {"valid": False, "error": f"Invalid expression value '{val}' at line {i-2}, column {j+2}"}
        
        # File size check
        file_size = os.path.getsize(gct_path)
        expected_min_size = n_genes * n_samples * 2  # Rough estimate
        
        validation_results = {
            "valid": True,
            "n_genes": n_genes,
            "n_samples": n_samples,
            "file_size_mb": file_size / (1024 * 1024),
            "invalid_values": invalid_values,
            "lines_checked": min(10, actual_data_lines)
        }
        
        print(f"  ✅ Format: Valid GCT v1.2")
        print(f"  ✅ Dimensions: {n_genes:,} genes × {n_samples:,} samples")
        print(f"  ✅ File size: {validation_results['file_size_mb']:.2f} MB")
        if invalid_values > 0:
            print(f"  ⚠️ Invalid values found: {invalid_values}")
        else:
            print(f"  ✅ Data quality: No invalid values in sample")
        
        return validation_results
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def validate_cls_file(cls_path, expected_samples=None):
    """
    Thoroughly validate a CLS file format and content.
    
    Parameters:
    -----------
    cls_path : str
        Path to the CLS file
    expected_samples : int, optional
        Expected number of samples
        
    Returns:
    --------
    dict : Validation results
    """
    print(f"\n📋 Validating CLS file: {cls_path}")
    
    if not os.path.exists(cls_path):
        return {"valid": False, "error": "File does not exist"}
    
    try:
        with open(cls_path, 'r') as f:
            lines = f.readlines()
        
        # Check minimum file length
        if len(lines) < 3:
            return {"valid": False, "error": "File too short (less than 3 lines)"}
        
        # Parse header
        try:
            header_parts = lines[0].strip().split()
            n_samples, n_classes, continuous = int(header_parts[0]), int(header_parts[1]), int(header_parts[2])
        except:
            return {"valid": False, "error": "Cannot parse header"}
        
        # Validate header
        if n_samples <= 0 or n_classes <= 0:
            return {"valid": False, "error": f"Invalid header: {n_samples} samples, {n_classes} classes"}
        
        if continuous != 1:
            return {"valid": False, "error": f"Expected continuous=1, got {continuous}"}
        
        if expected_samples and n_samples != expected_samples:
            return {"valid": False, "error": f"Sample count mismatch: expected {expected_samples}, got {n_samples}"}
        
        # Parse class labels
        class_labels = lines[1].strip().split()
        if len(class_labels) != n_classes:
            return {"valid": False, "error": f"Class label count mismatch: expected {n_classes}, got {len(class_labels)}"}
        
        # Parse sample assignments
        sample_assignments = lines[2].strip().split()
        if len(sample_assignments) != n_samples:
            return {"valid": False, "error": f"Sample assignment count mismatch: expected {n_samples}, got {len(sample_assignments)}"}
        
        # Validate assignments are valid indices
        valid_indices = set(str(i) for i in range(n_classes))
        invalid_assignments = [x for x in sample_assignments if x not in valid_indices]
        if invalid_assignments:
            return {"valid": False, "error": f"Invalid class assignments: {invalid_assignments[:5]}"}
        
        # Count class distribution
        class_counts = {}
        for assignment in sample_assignments:
            class_idx = int(assignment)
            class_name = class_labels[class_idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        validation_results = {
            "valid": True,
            "n_samples": n_samples,
            "n_classes": n_classes,
            "class_labels": class_labels,
            "class_counts": class_counts,
            "file_size_bytes": os.path.getsize(cls_path)
        }
        
        print(f"  ✅ Format: Valid CLS")
        print(f"  ✅ Samples: {n_samples:,}")
        print(f"  ✅ Classes: {n_classes} ({', '.join(class_labels)})")
        print(f"  ✅ Distribution: {dict(class_counts)}")
        
        return validation_results
        
    except Exception as e:
        return {"valid": False, "error": f"Validation error: {str(e)}"}


def test_filtered_matrix():
    """Test GSEA export with the filtered matrix."""
    print("🧪 TESTING GSEA EXPORT WITH FILTERED MATRIX")
    print("="*60)
    
    # Check for filtered matrix
    filtered_h5_path = "data/filtered_feature_bc_matrix.h5"
    emptydrops_h5ad_path = "data/emptydrops_all_detected_cells.h5ad"
    
    test_data_path = None
    data_source = None
    
    # Try different data sources
    if os.path.exists(emptydrops_h5ad_path):
        test_data_path = emptydrops_h5ad_path
        data_source = "EmptyDrops filtered h5ad"
        print(f"📂 Using EmptyDrops filtered data: {test_data_path}")
    elif os.path.exists(filtered_h5_path):
        test_data_path = filtered_h5_path
        data_source = "10X filtered matrix"
        print(f"📂 Using 10X filtered data: {test_data_path}")
    else:
        print("❌ No filtered matrix found!")
        print("Expected files:")
        print(f"  - {emptydrops_h5ad_path}")
        print(f"  - {filtered_h5_path}")
        return False
    
    # Create output directory
    output_dir = "test_gsea_output"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Initialize GSEA generator
        print(f"\n🔧 Initializing GSEA generator with {data_source}...")
        
        if test_data_path.endswith('.h5ad'):
            generator = GSEAFileGenerator(h5ad_path=test_data_path, output_dir=output_dir)
        else:
            generator = GSEAFileGenerator(h5_path=test_data_path, output_dir=output_dir)
        
        # Get dataset info
        n_cells, n_genes = generator.adata.shape
        print(f"📊 Dataset: {n_cells:,} cells × {n_genes:,} genes")
        
        # Check for required metadata
        required_obs = ['total_counts']
        missing_obs = [col for col in required_obs if col not in generator.adata.obs.columns]
        if missing_obs:
            print(f"⚠️ Missing required obs columns: {missing_obs}")
            print("Adding missing metadata...")
            generator.adata.obs['total_counts'] = np.array(generator.adata.X.sum(axis=1)).flatten()
        
        # Generate GSEA files
        print(f"\n🔄 Generating GSEA files...")
        
        # Create expression dataset (GCT)
        print("  Creating expression dataset (GCT)...")
        gct_path = generator.create_expression_dataset_gct(
            filter_genes=True,
            min_cells=10,
            use_raw=False
        )
        
        # Create phenotype labels (CLS)
        print("  Creating phenotype labels (CLS)...")
        generator.create_phenotype_groups(method='expression_level')
        cls_path = generator.create_phenotype_labels_cls()
        
        print(f"\n📁 Generated files:")
        print(f"  - Expression dataset: {gct_path}")
        print(f"  - Phenotype labels: {cls_path}")
        
        # Validate generated files
        print(f"\n✅ VALIDATION RESULTS")
        print("-" * 40)
        
        # Validate GCT file
        gct_results = validate_gct_file(gct_path, expected_samples=n_cells)
        if not gct_results["valid"]:
            print(f"❌ GCT validation failed: {gct_results['error']}")
            return False
        
        # Validate CLS file
        cls_results = validate_cls_file(cls_path, expected_samples=n_cells)
        if not cls_results["valid"]:
            print(f"❌ CLS validation failed: {cls_results['error']}")
            return False
        
        # Final summary
        print(f"\n🎉 SUCCESS! GSEA export completed successfully")
        print(f"📊 Final dataset: {gct_results['n_genes']:,} genes × {gct_results['n_samples']:,} samples")
        print(f"📋 Phenotypes: {cls_results['class_counts']}")
        print(f"💾 Total output size: {gct_results['file_size_mb']:.2f} MB")
        
        # GSEA usage instructions
        print(f"\n📖 READY FOR GSEA ANALYSIS!")
        print("="*60)
        print("Files are ready for GSEA desktop application:")
        print(f"1. Expression dataset: {os.path.abspath(gct_path)}")
        print(f"2. Phenotype labels: {os.path.abspath(cls_path)}")
        print("\nRecommended GSEA settings:")
        print("- Gene sets: MSigDB Hallmark, C2 Canonical Pathways, C5 Gene Ontology")
        print("- Permutation: Phenotype")
        print("- Metric: Signal2Noise")
        print("- Collapse dataset: No")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main function."""
    success = test_filtered_matrix()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
