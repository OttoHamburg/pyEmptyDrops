"""
Analyze genes with differential detection between CellRanger and EmptyDrops methods.
This script extends the PRM1 analysis to identify and visualize genes that show
significant differences in detection between the two methods.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.sparse import issparse, csr_matrix
import os
from typing import Dict, List, Tuple
import warnings
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests

class DifferentialGeneAnalyzer:
    def __init__(self, raw_h5_path: str, filtered_h5_path: str, emptydrops_results_path: str):
        """
        Initialize the analyzer with paths to required data files.
        
        Parameters
        ----------
        raw_h5_path : str
            Path to the raw feature-barcode matrix HDF5 file
        filtered_h5_path : str
            Path to the filtered feature-barcode matrix HDF5 file
        emptydrops_results_path : str
            Path to the EmptyDrops results CSV file
        """
        self.raw_h5_path = raw_h5_path
        self.filtered_h5_path = filtered_h5_path
        self.emptydrops_results_path = emptydrops_results_path
        
        # Create output directory
        self.output_dir = "differential_gene_analysis"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load data
        self._load_data()
    
    def _load_data(self):
        """Load and prepare all necessary data."""
        print("Loading data...")
        
        # Load matrices
        self.raw_adata = sc.read_10x_h5(self.raw_h5_path)
        self.filtered_adata = sc.read_10x_h5(self.filtered_h5_path)
        
        # Make gene names unique
        self.raw_adata.var_names_make_unique()
        self.filtered_adata.var_names_make_unique()
        
        # Load EmptyDrops results
        self.results = pd.read_csv(self.emptydrops_results_path, index_col=0)
        
        # Get total UMI counts
        self.total_counts = np.array(self.raw_adata.X.sum(axis=1)).flatten()
        
        print(f"Loaded {self.raw_adata.shape[0]} total droplets and {self.raw_adata.shape[1]} genes")
    
    def classify_droplets(self, fdr_threshold: float = 0.001) -> pd.Series:
        """
        Classify droplets based on CellRanger and EmptyDrops calls.
        
        Parameters
        ----------
        fdr_threshold : float
            FDR threshold for EmptyDrops classification
            
        Returns
        -------
        pd.Series
            Classification for each droplet
        """
        cell_categories = pd.Series(index=self.raw_adata.obs_names, dtype='str')
        
        # Get Cell Ranger and EmptyDrops calls
        cellranger_cells = set(self.filtered_adata.obs_names)
        emptydrops_cells = set(self.results[self.results['FDR'] < fdr_threshold].index)
        
        # Classify each droplet
        for barcode in self.raw_adata.obs_names:
            if barcode in cellranger_cells:
                if barcode in emptydrops_cells:
                    cell_categories[barcode] = 'Both'
                else:
                    cell_categories[barcode] = 'CellRanger_Only'
            else:
                if barcode in emptydrops_cells:
                    cell_categories[barcode] = 'EmptyDrops_Only'
                else:
                    cell_categories[barcode] = 'Neither'
        
        return cell_categories
    
    @staticmethod
    def _analyze_gene_batch(batch_data: Tuple[np.ndarray, List[str], pd.Series, int]) -> List[Dict]:
        """
        Analyze a batch of genes for differential detection.
        
        Parameters
        ----------
        batch_data : tuple
            Contains (gene_matrix, gene_names, categories, min_cells)
            
        Returns
        -------
        list
            List of dictionaries with gene statistics
        """
        gene_matrix, gene_names, categories, min_cells = batch_data
        results = []
        
        # Pre-calculate category masks
        category_masks = {}
        for category in ['Both', 'EmptyDrops_Only', 'CellRanger_Only', 'Neither']:
            category_masks[category] = categories == category
        
        # Process each gene in the batch
        for gene_idx in range(gene_matrix.shape[1]):
            gene_name = gene_names[gene_idx]
            gene_counts = gene_matrix[:, gene_idx].toarray().flatten()
            
            stats_dict = {
                'gene': gene_name,
                'total_expressing_cells': np.sum(gene_counts > 0)
            }
            
            # Calculate detection rates and statistics for each category
            total_cells = len(categories)
            for category in ['Both', 'EmptyDrops_Only', 'CellRanger_Only', 'Neither']:
                mask = category_masks[category]
                cells_in_category = np.sum(mask)
                
                if cells_in_category > 0:
                    gene_counts_category = gene_counts[mask]
                    expressing_cells = np.sum(gene_counts_category > 0)
                    detection_rate = expressing_cells / cells_in_category
                    mean_expression = np.mean(gene_counts_category)
                    median_expression = np.median(gene_counts_category)
                    
                    stats_dict.update({
                        f'{category}_cells': cells_in_category,
                        f'{category}_expressing': expressing_cells,
                        f'{category}_detection_rate': detection_rate,
                        f'{category}_mean_expression': mean_expression,
                        f'{category}_median_expression': median_expression
                    })
                else:
                    stats_dict.update({
                        f'{category}_cells': 0,
                        f'{category}_expressing': 0,
                        f'{category}_detection_rate': 0,
                        f'{category}_mean_expression': 0,
                        f'{category}_median_expression': 0
                    })
            
            # Calculate overall representation using all cells
            stats_dict['overall_detection_rate'] = stats_dict['total_expressing_cells'] / total_cells
            
            # Calculate fold changes and significance if enough cells
            if (stats_dict['EmptyDrops_Only_cells'] >= min_cells and 
                stats_dict['Both_cells'] >= min_cells):
                
                # Calculate fold change between EmptyDrops_Only and Both categories
                fold_change = (stats_dict['EmptyDrops_Only_detection_rate'] + 1e-10) / \
                            (stats_dict['Both_detection_rate'] + 1e-10)
                
                stats_dict['fold_change'] = fold_change
                
                # Fisher's exact test for detection rate difference
                contingency = np.array([
                    [stats_dict['EmptyDrops_Only_expressing'], 
                     stats_dict['EmptyDrops_Only_cells'] - stats_dict['EmptyDrops_Only_expressing']],
                    [stats_dict['Both_expressing'],
                     stats_dict['Both_cells'] - stats_dict['Both_expressing']]
                ])
                
                _, p_value = stats.fisher_exact(contingency)
                stats_dict['p_value'] = p_value
                
                results.append(stats_dict)
        
        return results

    def find_differential_genes(self, fdr_threshold: float = 0.001, 
                              min_cells: int = 10,
                              min_fold_change: float = 1.5,
                              min_detection_rate: float = 0.01,
                              n_processes: int = None,
                              batch_size: int = 1000,
                              genes_of_interest: List[str] = ['PRM1']) -> pd.DataFrame:
        """
        Find genes with differential detection between EmptyDrops and CellRanger cells.
        
        Parameters
        ----------
        fdr_threshold : float
            FDR threshold for EmptyDrops classification
        min_cells : int
            Minimum number of cells required in each category for analysis
        min_fold_change : float
            Minimum fold change in detection rate to consider a gene differential
        min_detection_rate : float
            Minimum detection rate in EmptyDrops_Only cells
        n_processes : int
            Number of processes to use for parallel processing
        batch_size : int
            Number of genes to process in each batch
        genes_of_interest : List[str]
            List of genes to include regardless of statistics
            
        Returns
        -------
        pd.DataFrame
            DataFrame containing statistics for differential genes
        """
        print("Finding differential genes...")
        
        # Classify droplets
        categories = self.classify_droplets(fdr_threshold)
        
        # Set up parallel processing
        if n_processes is None:
            n_processes = max(1, cpu_count() - 1)
        
        # Prepare batches
        gene_matrix = self.raw_adata.X
        gene_names = self.raw_adata.var_names.tolist()
        n_genes = len(gene_names)
        
        batches = []
        for i in range(0, n_genes, batch_size):
            end_idx = min(i + batch_size, n_genes)
            batch_data = (gene_matrix[:, i:end_idx], gene_names[i:end_idx], categories, min_cells)
            batches.append(batch_data)
        
        # Process batches in parallel
        with Pool(n_processes) as pool:
            results = list(tqdm(
                pool.imap(self._analyze_gene_batch, batches),
                total=len(batches),
                desc="Processing gene batches"
            ))
        
        # Flatten results
        all_results = []
        for batch_results in results:
            all_results.extend(batch_results)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(all_results)
        
        # Add FDR correction
        if not results_df.empty:
            _, results_df['FDR'], _, _ = multipletests(
                results_df['p_value'],
                method='fdr_bh'
            )
        
        # Filter results
        mask = (
            (results_df['EmptyDrops_Only_detection_rate'] >= min_detection_rate) &
            (results_df['fold_change'] >= min_fold_change) &
            (results_df['FDR'] < 0.05)
        )
        
        # Always include genes of interest
        for gene in genes_of_interest:
            if gene in results_df['gene'].values:
                mask |= (results_df['gene'] == gene)
        
        filtered_results = results_df[mask].copy()
        
        # Sort by fold change
        filtered_results = filtered_results.sort_values('fold_change', ascending=False)
        
        # Save results
        filtered_results.to_csv(os.path.join(self.output_dir, 'differential_genes.csv'))
        
        print(f"Found {len(filtered_results)} differential genes")
        return filtered_results

def main():
    """Main function to run the analysis."""
    # Set up paths
    raw_h5_path = "data/raw_feature_bc_matrix.h5"
    filtered_h5_path = "data/filtered_feature_bc_matrix.h5"
    emptydrops_results_path = "empty_drops/empty_drops_results.csv"
    
    # Create analyzer
    analyzer = DifferentialGeneAnalyzer(
        raw_h5_path=raw_h5_path,
        filtered_h5_path=filtered_h5_path,
        emptydrops_results_path=emptydrops_results_path
    )
    
    # Find differential genes
    differential_genes = analyzer.find_differential_genes(
        fdr_threshold=0.001,
        min_cells=10,
        min_fold_change=1.5,
        min_detection_rate=0.01,
        genes_of_interest=['PRM1']
    )
    
    print("\nTop 10 differential genes:")
    print(differential_genes.head(10))

if __name__ == "__main__":
    main() 