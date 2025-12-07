#!/usr/bin/env python3
"""
Demonstration script to prepare GSEA files from your EmptyDrops h5ad data.

This script shows different ways to prepare your data for GSEA analysis
following the requirements from GSEAguide.txt
"""

import os
import sys
from prepare_gsea_files import GSEAFileGenerator

def main():
    """Run GSEA file preparation for your data."""
    
    print("GSEA FILE PREPARATION FOR EMPTYDROPS DATA")
    print("="*50)
    
    # Define data paths (adjust these to match your files)
    data_paths = {
        'h5ad_emptydrops': 'data/emptydrops_all_detected_cells.h5ad',
        'h5_raw': 'data/raw_feature_bc_matrix.h5',
        'h5_filtered': 'data/filtered_feature_bc_matrix.h5'
    }
    
    # Check which files exist
    available_files = {}
    for name, path in data_paths.items():
        if os.path.exists(path):
            available_files[name] = path
            print(f"✅ Found: {path}")
        else:
            print(f"❌ Not found: {path}")
    
    if not available_files:
        print("\n❌ No data files found. Please check your data paths.")
        print("Expected files:")
        for name, path in data_paths.items():
            print(f"  - {path}")
        return 1
    
    # Choose the best available file (prefer EmptyDrops filtered)
    if 'h5ad_emptydrops' in available_files:
        input_file = available_files['h5ad_emptydrops']
        file_type = 'h5ad'
        print(f"\n🎯 Using EmptyDrops filtered data: {input_file}")
    elif 'h5_filtered' in available_files:
        input_file = available_files['h5_filtered']
        file_type = 'h5'
        print(f"\n🎯 Using CellRanger filtered data: {input_file}")
    else:
        input_file = available_files['h5_raw']
        file_type = 'h5'
        print(f"\n🎯 Using raw data: {input_file}")
        print("⚠️  Note: Raw data may need additional filtering")
    
    # Create different GSEA file sets for different analysis approaches
    analysis_configs = [
        {
            'name': 'expression_based',
            'output_dir': 'gsea_files_expression',
            'phenotype_method': 'expression_level',
            'description': 'Groups cells by total expression level (high vs low)'
        },
        {
            'name': 'cluster_based',
            'output_dir': 'gsea_files_clusters',
            'phenotype_method': 'clusters',
            'description': 'Uses existing cluster assignments if available'
        }
    ]
    
    # Add pseudotime analysis if we have processed data
    if file_type == 'h5ad':
        analysis_configs.append({
            'name': 'pseudotime_based',
            'output_dir': 'gsea_files_pseudotime',
            'phenotype_method': 'pseudotime',
            'description': 'Groups cells by developmental pseudotime (early vs late)'
        })
    
    print(f"\n📋 Will create {len(analysis_configs)} different GSEA file sets:")
    for config in analysis_configs:
        print(f"  - {config['name']}: {config['description']}")
    
    # Create GSEA files for each configuration
    successful_configs = []
    
    for config in analysis_configs:
        print(f"\n{'='*60}")
        print(f"CREATING: {config['name'].upper()}")
        print(f"{'='*60}")
        
        try:
            # Initialize generator
            if file_type == 'h5ad':
                generator = GSEAFileGenerator(
                    h5ad_path=input_file,
                    output_dir=config['output_dir']
                )
            else:
                generator = GSEAFileGenerator(
                    h5_path=input_file,
                    output_dir=config['output_dir']
                )
            
            # Create files
            files_created = generator.create_all_gsea_files(
                phenotype_method=config['phenotype_method'],
                output_prefix=f"{config['name']}_"
            )
            
            config['files_created'] = files_created
            successful_configs.append(config)
            
            print(f"✅ Successfully created {len(files_created)} files in '{config['output_dir']}'")
            
        except Exception as e:
            print(f"❌ Failed to create {config['name']}: {e}")
            print("Continuing with other configurations...")
            continue
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if successful_configs:
        print(f"✅ Successfully created GSEA files for {len(successful_configs)} configurations:")
        
        for config in successful_configs:
            print(f"\n📁 {config['name'].upper()}:")
            print(f"   Directory: {config['output_dir']}")
            print(f"   Method: {config['description']}")
            print(f"   Files: {len(config['files_created'])}")
            
            # List the files
            for file_path in config['files_created']:
                filename = os.path.basename(file_path)
                print(f"     - {filename}")
        
        print(f"\n🎯 NEXT STEPS:")
        print("1. Choose the most appropriate configuration for your analysis")
        print("2. Open GSEA desktop application")
        print("3. Load the files from your chosen configuration")
        print("4. Select appropriate gene set collections:")
        print("   - GO_Biological_Process_2023 (for biological processes)")
        print("   - KEGG_2021_Human (for pathways)")
        print("   - MSigDB_Hallmark_2020 (for hallmark gene sets)")
        print("5. Run the analysis with these recommended settings:")
        print("   - Collapse/Remap to gene symbols: Collapse")
        print("   - Permutation type: phenotype")
        print("   - Number of permutations: 1000")
        print("   - Metric for ranking genes: Signal2Noise")
        
        print(f"\n📖 For detailed instructions, see the GSEA_INSTRUCTIONS.txt file in each directory.")
        
        # Recommend best configuration
        if len(successful_configs) > 1:
            print(f"\n💡 RECOMMENDATION:")
            if any(c['name'] == 'expression_based' for c in successful_configs):
                print("   Start with 'expression_based' configuration - it's most similar to your current analysis")
            elif any(c['name'] == 'pseudotime_based' for c in successful_configs):
                print("   Use 'pseudotime_based' if you're interested in developmental trajectories")
            else:
                print("   Use 'cluster_based' if you want to compare different cell types")
        
    else:
        print("❌ No GSEA files were successfully created.")
        print("Please check your data files and try again.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
