#!/usr/bin/env python3
"""
Simple GSEA file export for EmptyDrops h5ad data.

This script creates the essential files needed for GSEA analysis in the correct formats
according to GSEAguide.txt requirements.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

def load_emptydrops_data(h5ad_path="data/emptydrops_all_detected_cells.h5ad"):
    """Load and basic preprocessing of EmptyDrops data."""
    print(f"Loading EmptyDrops data from: {h5ad_path}")
    
    if not os.path.exists(h5ad_path):
        print(f"❌ File not found: {h5ad_path}")
        print("Available files in data/:")
        if os.path.exists("data/"):
            for f in os.listdir("data/"):
                print(f"  - {f}")
        return None
    
    adata = sc.read_h5ad(h5ad_path)
    print(f"Data loaded: {adata.shape[0]} cells × {adata.shape[1]} genes")
    
    # Show data structure
    print(f"Gene names (first 5): {list(adata.var_names[:5])}")
    print(f"Cell metadata columns: {list(adata.obs.columns)}")
    print(f"Gene metadata columns: {list(adata.var.columns)}")
    
    return adata

def create_expression_groups(adata, method="expression_level"):
    """Create simple binary phenotype groups."""
    print(f"\nCreating phenotype groups using {method}...")
    
    if method == "expression_level":
        # Group by total UMI counts
        total_counts = np.array(adata.X.sum(axis=1)).flatten()
        median_count = np.median(total_counts)
        
        groups = []
        for count in total_counts:
            if count >= median_count:
                groups.append("HIGH")
            else:
                groups.append("LOW")
        
        adata.obs['phenotype'] = pd.Categorical(groups, categories=["HIGH", "LOW"])
        
        print(f"Groups created:")
        print(f"  HIGH: {(adata.obs['phenotype'] == 'HIGH').sum()} cells")
        print(f"  LOW: {(adata.obs['phenotype'] == 'LOW').sum()} cells")
        
    return adata

def export_expression_gct(adata, output_dir="gsea_files", filename="expression_dataset.gct"):
    """Export expression data in GCT format for GSEA."""
    print(f"\nExporting expression data to GCT format...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Convert sparse matrix to dense
    if hasattr(adata.X, 'toarray'):
        expr_matrix = adata.X.toarray()
    else:
        expr_matrix = adata.X
    
    # Filter lowly expressed genes
    gene_counts = (expr_matrix > 0).sum(axis=0)
    keep_genes = gene_counts >= 5  # At least 5 cells
    
    expr_matrix = expr_matrix[:, keep_genes]
    gene_names = adata.var_names[keep_genes]
    
    print(f"Filtered to {len(gene_names)} genes (≥5 cells expressing)")
    
    # Check if data needs log transformation
    max_val = expr_matrix.max()
    if max_val > 50:  # Likely raw counts
        expr_matrix = np.log1p(expr_matrix)
        print("Applied log1p transformation")
    
    # Create DataFrame (genes as rows, samples as columns)
    expr_df = pd.DataFrame(
        expr_matrix.T,
        index=gene_names,
        columns=adata.obs_names
    )
    
    # Write GCT file
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        # GCT header
        f.write('#1.2\n')
        f.write(f'{len(expr_df)}\t{len(expr_df.columns)}\n')
        
        # Column headers: NAME, Description, then sample names
        f.write('NAME\tDescription\t' + '\t'.join(expr_df.columns) + '\n')
        
        # Data rows
        for gene_name in expr_df.index:
            row_data = expr_df.loc[gene_name]
            values_str = '\t'.join([f'{val:.6f}' for val in row_data.values])
            f.write(f'{gene_name}\t{gene_name}\t{values_str}\n')
    
    print(f"✅ Expression dataset saved: {output_path}")
    print(f"   Format: {len(expr_df)} genes × {len(expr_df.columns)} samples")
    
    return output_path

def export_phenotype_cls(adata, output_dir="gsea_files", filename="phenotype_labels.cls"):
    """Export phenotype labels in CLS format for GSEA."""
    print(f"\nExporting phenotype labels to CLS format...")
    
    if 'phenotype' not in adata.obs.columns:
        print("❌ No phenotype groups found. Run create_expression_groups first.")
        return None
    
    phenotypes = adata.obs['phenotype']
    unique_phenotypes = phenotypes.cat.categories.tolist()
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        # Line 1: number of samples, number of classes, 1
        f.write(f'{len(phenotypes)} {len(unique_phenotypes)} 1\n')
        
        # Line 2: class names (with # prefix)
        f.write('#' + ' '.join(unique_phenotypes) + '\n')
        
        # Line 3: class assignments (0-indexed)
        class_indices = []
        for phenotype in phenotypes:
            class_indices.append(str(unique_phenotypes.index(phenotype)))
        
        f.write(' '.join(class_indices) + '\n')
    
    print(f"✅ Phenotype labels saved: {output_path}")
    print(f"   Classes: {unique_phenotypes}")
    
    return output_path

def export_chip_annotation(adata, output_dir="gsea_files", filename="chip_annotation.chip"):
    """Export chip annotation file for gene symbol mapping."""
    print(f"\nExporting chip annotation file...")
    
    output_path = os.path.join(output_dir, filename)
    
    with open(output_path, 'w') as f:
        # Header
        f.write('Probe Set ID\tGene Symbol\tGene Title\n')
        
        # Data rows
        for gene_name in adata.var_names:
            # Use gene name as both probe ID and symbol
            f.write(f'{gene_name}\t{gene_name}\t{gene_name}\n')
    
    print(f"✅ Chip annotation saved: {output_path}")
    
    return output_path

def create_instructions_file(output_dir="gsea_files", files_created=None):
    """Create instructions file for GSEA analysis."""
    instructions_path = os.path.join(output_dir, "GSEA_INSTRUCTIONS.txt")
    
    with open(instructions_path, 'w') as f:
        f.write("GSEA ANALYSIS INSTRUCTIONS\n")
        f.write("="*50 + "\n\n")
        
        f.write("FILES CREATED:\n")
        if files_created:
            for i, file_path in enumerate(files_created, 1):
                filename = os.path.basename(file_path)
                f.write(f"{i}. {filename}\n")
        
        f.write("\nGSEA ANALYSIS STEPS:\n")
        f.write("1. Open GSEA desktop application (https://www.gsea-msigdb.org/gsea/)\n")
        f.write("2. Go to 'Run GSEA' tab\n")
        f.write("3. Load your files:\n")
        f.write("   - Expression dataset: expression_dataset.gct\n")
        f.write("   - Phenotype labels: phenotype_labels.cls\n")
        f.write("   - Gene sets: Choose from MSigDB (recommended below)\n")
        f.write("   - Chip platform: chip_annotation.chip (optional)\n")
        f.write("4. Settings:\n")
        f.write("   - Collapse/Remap to gene symbols: No_Collapse\n")
        f.write("   - Permutation type: phenotype\n")
        f.write("   - Number of permutations: 1000\n")
        f.write("   - Metric for ranking genes: Signal2Noise\n")
        f.write("5. Click 'Run GSEA'\n")
        
        f.write("\nRECOMMENDED GENE SET COLLECTIONS:\n")
        f.write("For spermatogenesis/reproductive biology:\n")
        f.write("- GO_Biological_Process_2023\n")
        f.write("- KEGG_2021_Human\n")
        f.write("- Reactome_2022\n")
        f.write("- MSigDB_Hallmark_2020\n")
        
        f.write("\nKEY POINTS:\n")
        f.write("- Your data compares HIGH vs LOW expression cells\n")
        f.write("- Use 'No_Collapse' since gene names are already symbols\n")
        f.write("- Look for pathways enriched in HIGH group (mature sperm markers)\n")
        f.write("- Check p-values and FDR q-values for significance\n")
        
    print(f"✅ Instructions saved: {instructions_path}")
    
    return instructions_path

def main():
    """Main function to create all GSEA files."""
    print("SIMPLE GSEA FILE EXPORT")
    print("="*30)
    
    # Load data
    adata = load_emptydrops_data()
    if adata is None:
        return 1
    
    # Create phenotype groups
    adata = create_expression_groups(adata, method="expression_level")
    
    # Create output directory
    output_dir = "gsea_files"
    
    # Export all files
    files_created = []
    
    try:
        # 1. Expression dataset (required)
        gct_file = export_expression_gct(adata, output_dir)
        files_created.append(gct_file)
        
        # 2. Phenotype labels (required)
        cls_file = export_phenotype_cls(adata, output_dir)
        files_created.append(cls_file)
        
        # 3. Chip annotation (optional)
        chip_file = export_chip_annotation(adata, output_dir)
        files_created.append(chip_file)
        
        # 4. Instructions
        instructions_file = create_instructions_file(output_dir, files_created)
        
        print(f"\n✅ SUCCESS!")
        print(f"Created {len(files_created)} GSEA files in '{output_dir}/'")
        print(f"\nFiles created:")
        for file_path in files_created:
            print(f"  - {os.path.basename(file_path)}")
        
        print(f"\nNext steps:")
        print("1. Open GSEA desktop application")
        print("2. Load the files from the gsea_files/ directory")
        print("3. Choose gene set collections from MSigDB")
        print("4. Run the analysis")
        print(f"\nSee {output_dir}/GSEA_INSTRUCTIONS.txt for detailed instructions.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error creating files: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
