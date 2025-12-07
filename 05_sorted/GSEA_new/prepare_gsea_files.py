#!/usr/bin/env python3
"""
Prepare h5ad EmptyDrops matrix for GSEA Analysis

This script converts your h5ad single-cell data into the required GSEA file formats:
1. Expression dataset file (.gct format)
2. Phenotype labels file (.cls format)
3. Gene sets file (optional, using MSigDB)
4. Chip annotation file (optional, for gene symbol mapping)

Based on GSEAguide.txt requirements for GSEA desktop application.
"""

import os
import numpy as np
import pandas as pd
import scanpy as sc
import warnings
warnings.filterwarnings('ignore')

class GSEAFileGenerator:
    """Generate GSEA-compatible files from h5ad data."""
    
    def __init__(self, h5ad_path=None, h5_path=None, output_dir="gsea_files"):
        """
        Initialize with either h5ad or h5 file path.
        
        Parameters:
        -----------
        h5ad_path : str, optional
            Path to h5ad file (processed EmptyDrops data)
        h5_path : str, optional  
            Path to h5 file (10x format)
        output_dir : str
            Directory to save GSEA files
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load data
        if h5ad_path and os.path.exists(h5ad_path):
            print(f"Loading h5ad file: {h5ad_path}")
            self.adata = sc.read_h5ad(h5ad_path)
        elif h5_path and os.path.exists(h5_path):
            print(f"Loading h5 file: {h5_path}")
            self.adata = sc.read_10x_h5(h5_path)
            # Make gene names unique
            if not self.adata.var_names.is_unique:
                self.adata.var_names_make_unique()
        else:
            raise FileNotFoundError("Neither h5ad nor h5 file found. Please provide valid path.")
        
        print(f"Data loaded: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        self._analyze_data_structure()
        
    def _analyze_data_structure(self):
        """Analyze the data structure and gene identifiers."""
        print("\n" + "="*50)
        print("DATA STRUCTURE ANALYSIS")
        print("="*50)
        
        # Check gene identifiers
        print(f"Gene names (first 10): {list(self.adata.var_names[:10])}")
        print(f"Gene var columns: {list(self.adata.var.columns)}")
        
        # Check if we have gene symbols vs other identifiers
        if 'gene_symbols' in self.adata.var.columns:
            symbols_available = self.adata.var['gene_symbols'].notna().sum()
            print(f"Gene symbols available: {symbols_available}/{len(self.adata.var)}")
        
        # Check observation data
        print(f"Cell obs columns: {list(self.adata.obs.columns)}")
        print(f"Data matrix type: {type(self.adata.X)}")
        print(f"Data range: {self.adata.X.min():.3f} to {self.adata.X.max():.3f}")
        
    def create_phenotype_groups(self, method="expression_level", n_groups=2, 
                              custom_grouping=None, group_names=None):
        """
        Create phenotype groups for GSEA analysis.
        
        Parameters:
        -----------
        method : str
            Method for grouping: 'expression_level', 'pseudotime', 'clusters', 'custom'
        n_groups : int
            Number of groups to create (for expression_level method)
        custom_grouping : array-like
            Custom grouping variable (for custom method)
        group_names : list
            Names for the groups
        """
        print(f"\nCreating phenotype groups using method: {method}")
        
        if method == "expression_level":
            # Group by total expression level (similar to your current approach)
            total_counts = np.array(self.adata.X.sum(axis=1)).flatten()
            
            if n_groups == 2:
                # Binary grouping (high vs low)
                median_count = np.median(total_counts)
                groups = ["HIGH" if x >= median_count else "LOW" for x in total_counts]
                if group_names is None:
                    group_names = ["HIGH", "LOW"]
            else:
                # Multiple groups using quantiles
                groups = pd.cut(total_counts, bins=n_groups, 
                              labels=[f"GROUP_{i+1}" for i in range(n_groups)])
                if group_names is None:
                    group_names = [f"GROUP_{i+1}" for i in range(n_groups)]
                    
        elif method == "pseudotime" and 'dpt_pseudotime' in self.adata.obs.columns:
            # Group by pseudotime
            pseudotime = self.adata.obs['dpt_pseudotime']
            median_time = pseudotime.median()
            groups = ["LATE" if x >= median_time else "EARLY" for x in pseudotime]
            if group_names is None:
                group_names = ["EARLY", "LATE"]
                
        elif method == "clusters" and 'louvain' in self.adata.obs.columns:
            # Use existing cluster labels
            groups = self.adata.obs['louvain'].astype(str).tolist()
            unique_groups = sorted(self.adata.obs['louvain'].unique().astype(str))
            if group_names is None:
                group_names = unique_groups
                
        elif method == "custom" and custom_grouping is not None:
            groups = custom_grouping
            if group_names is None:
                group_names = sorted(set(groups))
                
        else:
            raise ValueError(f"Invalid method '{method}' or required data not available")
        
        self.adata.obs['gsea_phenotype'] = pd.Categorical(groups, categories=group_names)
        
        print(f"Phenotype groups created:")
        group_counts = self.adata.obs['gsea_phenotype'].value_counts()
        for group, count in group_counts.items():
            print(f"  {group}: {count} cells")
            
        return groups, group_names
    
    def create_expression_dataset_gct(self, filename="expression_dataset.gct", 
                                    use_raw=False, log_transform=True,
                                    filter_genes=True, min_cells=5):
        """
        Create expression dataset file in GCT format for GSEA.
        
        Parameters:
        -----------
        filename : str
            Output filename
        use_raw : bool
            Use raw counts instead of processed data
        log_transform : bool
            Apply log1p transformation
        filter_genes : bool
            Filter out lowly expressed genes
        min_cells : int
            Minimum cells expressing a gene (for filtering)
        """
        print(f"\nCreating expression dataset file: {filename}")
        
        # Use appropriate data
        if use_raw and self.adata.raw is not None:
            expr_data = self.adata.raw.X
            gene_names = self.adata.raw.var_names
        else:
            expr_data = self.adata.X
            gene_names = self.adata.var_names
        
        # Convert to dense array if sparse
        if hasattr(expr_data, 'toarray'):
            expr_data = expr_data.toarray()
        
        # Filter genes if requested
        if filter_genes:
            # Count cells expressing each gene
            gene_expression_counts = (expr_data > 0).sum(axis=0)
            if hasattr(gene_expression_counts, 'A1'):  # Handle sparse matrix
                gene_expression_counts = gene_expression_counts.A1
            keep_genes = gene_expression_counts >= min_cells
            expr_data = expr_data[:, keep_genes]
            gene_names = gene_names[keep_genes]
            print(f"Filtered to {len(gene_names)} genes (min {min_cells} cells)")
        
        # Log transform if requested
        if log_transform and not use_raw:
            # Check if data is already log-transformed
            max_val = expr_data.max()
            if max_val > 50:  # Likely raw counts
                expr_data = np.log1p(expr_data)
                print("Applied log1p transformation")
            else:
                print("Data appears already log-transformed")
        
        # Create DataFrame
        expr_df = pd.DataFrame(
            expr_data.T,  # Transpose: genes as rows, samples as columns
            index=gene_names,
            columns=self.adata.obs_names
        )
        
        # Add gene descriptions (use gene names if no other info available)
        if 'gene_symbols' in self.adata.var.columns and not use_raw:
            # Get descriptions for filtered genes
            try:
                descriptions = []
                for gene_name in gene_names:
                    if gene_name in self.adata.var.index:
                        symbol = self.adata.var.loc[gene_name, 'gene_symbols']
                        if pd.isna(symbol):
                            descriptions.append(gene_name)
                        else:
                            descriptions.append(symbol)
                    else:
                        descriptions.append(gene_name)
            except:
                descriptions = gene_names
        else:
            descriptions = gene_names
        
        # Create GCT format
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            # GCT header
            f.write('#1.2\n')
            f.write(f'{len(expr_df)}\t{len(expr_df.columns)}\n')
            
            # Column headers
            f.write('NAME\tDescription\t' + '\t'.join(expr_df.columns) + '\n')
            
            # Data rows
            for i, (gene, row) in enumerate(expr_df.iterrows()):
                desc = descriptions[i] if i < len(descriptions) else gene
                values = '\t'.join([f'{val:.6f}' for val in row.values])
                f.write(f'{gene}\t{desc}\t{values}\n')
        
        print(f"✅ Expression dataset saved: {output_path}")
        print(f"   Format: {len(expr_df)} genes × {len(expr_df.columns)} samples")
        
        return output_path
    
    def create_phenotype_labels_cls(self, filename="phenotype_labels.cls", 
                                  phenotype_col='gsea_phenotype'):
        """
        Create phenotype labels file in CLS format for GSEA.
        
        Parameters:
        -----------
        filename : str
            Output filename
        phenotype_col : str
            Column name containing phenotype labels
        """
        print(f"\nCreating phenotype labels file: {filename}")
        
        if phenotype_col not in self.adata.obs.columns:
            raise ValueError(f"Phenotype column '{phenotype_col}' not found. "
                           "Run create_phenotype_groups() first.")
        
        phenotypes = self.adata.obs[phenotype_col]
        unique_phenotypes = phenotypes.cat.categories.tolist()
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            # CLS header line 1: number of samples, number of classes, 1
            f.write(f'{len(phenotypes)} {len(unique_phenotypes)} 1\n')
            
            # CLS header line 2: class names
            f.write('#' + ' '.join(unique_phenotypes) + '\n')
            
            # CLS data line: class assignments (0-indexed)
            class_indices = [unique_phenotypes.index(phenotype) for phenotype in phenotypes]
            f.write(' '.join(map(str, class_indices)) + '\n')
        
        print(f"✅ Phenotype labels saved: {output_path}")
        print(f"   Classes: {unique_phenotypes}")
        print(f"   Samples per class: {phenotypes.value_counts().to_dict()}")
        
        return output_path
    
    def create_chip_annotation(self, filename="chip_annotation.chip"):
        """
        Create chip annotation file for gene symbol mapping.
        
        Parameters:
        -----------
        filename : str
            Output filename
        """
        print(f"\nCreating chip annotation file: {filename}")
        
        # Check if we have gene symbol information
        if 'gene_symbols' in self.adata.var.columns:
            gene_symbols = self.adata.var['gene_symbols'].fillna(self.adata.var_names)
        else:
            print("No gene symbols found, using gene names as symbols")
            gene_symbols = self.adata.var_names
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            # Chip header
            f.write('Probe Set ID\tGene Symbol\tGene Title\n')
            
            # Data rows
            for probe_id, symbol in zip(self.adata.var_names, gene_symbols):
                # Use symbol as title if no other info available
                title = symbol if symbol != probe_id else "NA"
                f.write(f'{probe_id}\t{symbol}\t{title}\n')
        
        print(f"✅ Chip annotation saved: {output_path}")
        print(f"   Mapped {len(self.adata.var)} genes to symbols")
        
        return output_path
    
    def create_gene_sets_gmt(self, filename="custom_gene_sets.gmt", 
                           marker_genes=None, cluster_col='louvain'):
        """
        Create custom gene sets file from cluster markers.
        
        Parameters:
        -----------
        filename : str
            Output filename
        marker_genes : dict, optional
            Custom gene sets as {set_name: [genes]}
        cluster_col : str
            Column containing cluster labels for marker analysis
        """
        print(f"\nCreating gene sets file: {filename}")
        
        gene_sets = {}
        
        # Add custom gene sets if provided
        if marker_genes:
            gene_sets.update(marker_genes)
        
        # Add cluster markers if available
        if cluster_col in self.adata.obs.columns:
            try:
                # Find marker genes for each cluster
                sc.tl.rank_genes_groups(self.adata, groupby=cluster_col, 
                                      method='wilcoxon', n_genes=50)
                
                marker_df = sc.get.rank_genes_groups_df(self.adata, group=None)
                
                # Create gene sets for each cluster
                for cluster in marker_df['group'].unique():
                    cluster_markers = marker_df[
                        (marker_df['group'] == cluster) & 
                        (marker_df['pvals_adj'] < 0.05) &
                        (marker_df['logfoldchanges'] > 0.5)
                    ]['names'].tolist()
                    
                    if len(cluster_markers) > 5:  # Only include if enough markers
                        gene_sets[f'CLUSTER_{cluster}_MARKERS'] = cluster_markers[:30]
                
                print(f"Added {len([k for k in gene_sets.keys() if 'CLUSTER' in k])} cluster marker sets")
                
            except Exception as e:
                print(f"Could not create cluster markers: {e}")
        
        # Add some default spermatogenesis-related gene sets
        spermatogenesis_sets = {
            'SPERMATOGENESIS_EARLY': ['SYCP1', 'SYCP3', 'DMC1', 'MLH1', 'MSH4', 'MSH5'],
            'SPERMATOGENESIS_LATE': ['PRM1', 'PRM2', 'TNP1', 'TNP2', 'SPAG16', 'ODF1', 'ODF2'],
            'CHROMATIN_CONDENSATION': ['PRM1', 'PRM2', 'TNP1', 'TNP2', 'H1T', 'H2AL1Q'],
            'FLAGELLUM_ASSEMBLY': ['SPAG16', 'ODF1', 'ODF2', 'DNAI1', 'DNAH5', 'TEKT1'],
            'ACROSOME_FORMATION': ['ACRV1', 'CRISP2', 'SPAM1', 'ZP3R', 'IZUMO1']
        }
        
        # Only add sets where we have genes in our data
        for set_name, genes in spermatogenesis_sets.items():
            available_genes = [g for g in genes if g in self.adata.var_names]
            if len(available_genes) >= 3:
                gene_sets[set_name] = available_genes
        
        output_path = os.path.join(self.output_dir, filename)
        
        with open(output_path, 'w') as f:
            for set_name, genes in gene_sets.items():
                # GMT format: set_name, description, gene1, gene2, ...
                f.write(f'{set_name}\t{set_name}\t' + '\t'.join(genes) + '\n')
        
        print(f"✅ Gene sets saved: {output_path}")
        print(f"   Created {len(gene_sets)} gene sets")
        
        return output_path
    
    def create_all_gsea_files(self, phenotype_method="expression_level", 
                            output_prefix="", **kwargs):
        """
        Create all required GSEA files at once.
        
        Parameters:
        -----------
        phenotype_method : str
            Method for creating phenotype groups
        output_prefix : str
            Prefix for all output files
        **kwargs : dict
            Additional arguments for individual file creation methods
        """
        print("\n" + "="*60)
        print("CREATING ALL GSEA FILES")
        print("="*60)
        
        # Create phenotype groups
        self.create_phenotype_groups(method=phenotype_method)
        
        # Create all files
        files_created = []
        
        # 1. Expression dataset (required)
        gct_file = self.create_expression_dataset_gct(
            filename=f"{output_prefix}expression_dataset.gct"
        )
        files_created.append(gct_file)
        
        # 2. Phenotype labels (required)
        cls_file = self.create_phenotype_labels_cls(
            filename=f"{output_prefix}phenotype_labels.cls"
        )
        files_created.append(cls_file)
        
        # 3. Chip annotation (optional but recommended)
        chip_file = self.create_chip_annotation(
            filename=f"{output_prefix}chip_annotation.chip"
        )
        files_created.append(chip_file)
        
        # 4. Custom gene sets (optional)
        gmt_file = self.create_gene_sets_gmt(
            filename=f"{output_prefix}custom_gene_sets.gmt"
        )
        files_created.append(gmt_file)
        
        # Create summary
        self._create_gsea_summary(files_created, output_prefix)
        
        return files_created
    
    def _create_gsea_summary(self, files_created, prefix=""):
        """Create a summary file with GSEA analysis instructions."""
        summary_path = os.path.join(self.output_dir, f"{prefix}GSEA_INSTRUCTIONS.txt")
        
        with open(summary_path, 'w') as f:
            f.write("GSEA ANALYSIS FILES - USAGE INSTRUCTIONS\n")
            f.write("="*50 + "\n\n")
            
            f.write("FILES CREATED:\n")
            for i, file_path in enumerate(files_created, 1):
                filename = os.path.basename(file_path)
                f.write(f"{i}. {filename}\n")
            
            f.write(f"\nDATA SUMMARY:\n")
            f.write(f"- Total cells: {self.adata.shape[0]}\n")
            f.write(f"- Total genes: {self.adata.shape[1]}\n")
            f.write(f"- Phenotype groups: {list(self.adata.obs['gsea_phenotype'].cat.categories)}\n")
            
            f.write(f"\nGSEA ANALYSIS STEPS:\n")
            f.write("1. Open GSEA desktop application\n")
            f.write("2. Go to 'Run GSEA' tab\n")
            f.write("3. Load files:\n")
            f.write(f"   - Expression dataset: {os.path.basename(files_created[0])}\n")
            f.write(f"   - Phenotype labels: {os.path.basename(files_created[1])}\n")
            f.write("   - Gene sets: Use MSigDB collections or custom file\n")
            f.write(f"   - Chip platform: {os.path.basename(files_created[2]) if len(files_created) > 2 else 'Optional'}\n")
            f.write("4. Set parameters:\n")
            f.write("   - Collapse/Remap to gene symbols: Collapse (if using chip file)\n")
            f.write("   - Permutation type: phenotype\n")
            f.write("   - Number of permutations: 1000\n")
            f.write("   - Metric for ranking genes: Signal2Noise\n")
            f.write("5. Run analysis\n")
            
            f.write(f"\nRECOMMENDED GENE SET COLLECTIONS:\n")
            f.write("- GO_Biological_Process_2023\n")
            f.write("- KEGG_2021_Human\n")
            f.write("- Reactome_2022\n")
            f.write("- MSigDB_Hallmark_2020\n")
            
        print(f"✅ Instructions saved: {summary_path}")


def main():
    """Main function to demonstrate usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare GSEA files from h5ad data')
    parser.add_argument('--h5ad', help='Path to h5ad file')
    parser.add_argument('--h5', help='Path to h5 file')
    parser.add_argument('--output-dir', default='gsea_files', help='Output directory')
    parser.add_argument('--phenotype-method', default='expression_level',
                       choices=['expression_level', 'pseudotime', 'clusters', 'custom'],
                       help='Method for creating phenotype groups')
    parser.add_argument('--prefix', default='', help='Prefix for output files')
    
    args = parser.parse_args()
    
    # Use default paths if none provided
    if not args.h5ad and not args.h5:
        # Try to find files in current directory
        if os.path.exists('data/emptydrops_all_detected_cells.h5ad'):
            args.h5ad = 'data/emptydrops_all_detected_cells.h5ad'
        elif os.path.exists('data/raw_feature_bc_matrix.h5'):
            args.h5 = 'data/raw_feature_bc_matrix.h5'
        else:
            print("No input file specified and no default files found.")
            print("Please provide --h5ad or --h5 argument.")
            return
    
    # Create GSEA file generator
    try:
        generator = GSEAFileGenerator(
            h5ad_path=args.h5ad,
            h5_path=args.h5,
            output_dir=args.output_dir
        )
        
        # Create all files
        files_created = generator.create_all_gsea_files(
            phenotype_method=args.phenotype_method,
            output_prefix=args.prefix
        )
        
        print(f"\n✅ SUCCESS! Created {len(files_created)} GSEA files in '{args.output_dir}'")
        print("\nNext steps:")
        print("1. Open GSEA desktop application")
        print("2. Load the created files")
        print("3. Choose appropriate gene set collections from MSigDB")
        print("4. Run the analysis")
        print(f"\nSee {args.output_dir}/GSEA_INSTRUCTIONS.txt for detailed instructions.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
