"""
Analyze gene expression patterns in confusion matrix quadrants.
This script focuses on understanding the characteristics of droplets where
CellRanger and EmptyDrops disagree on cell calling.
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
from typing import Dict, List, Tuple
import warnings
from tqdm import tqdm
from multiprocessing import Pool
import argparse

def process_gene_batch(args):
    """
    Process a batch of genes for normalization.
    
    Parameters
    ----------
    args : tuple
        (gene_indices, expression_matrix, total_umi_counts)
    
    Returns
    -------
    list
        List of (gene_index, (mean_pct, (q1, median, q3, whislo, whishi, fliers))) tuples
    """
    gene_indices, expression_matrix, total_umi_counts = args
    results = []
    for idx in gene_indices:
        gene_expr = np.array(expression_matrix[:, idx].todense()).flatten()
        # Only consider cells that express this gene
        expressing_cells = gene_expr > 0
        if np.any(expressing_cells):
            # Calculate percentage of total counts in expressing cells
            gene_pct = (gene_expr[expressing_cells] / total_umi_counts[expressing_cells]) * 100
            mean_pct = np.mean(gene_pct)
            
            # Calculate distribution statistics
            q1, median, q3 = np.percentile(gene_pct, [25, 50, 75])
            iqr = q3 - q1
            whislo = max(np.min(gene_pct), q1 - 1.5 * iqr)
            whishi = min(np.max(gene_pct), q3 + 1.5 * iqr)
            
            # Find outliers
            fliers = gene_pct[(gene_pct < whislo) | (gene_pct > whishi)]
            
            # Store statistics
            stats = (q1, median, q3, whislo, whishi, fliers)
            results.append((idx, (mean_pct, stats)))
        else:
            results.append((idx, (0, None)))
            print("results: ", results)
    return results

class QuadrantAnalyzer:
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
        self.output_dir = "confusion_matrix_analysis"
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
        Classify droplets into confusion matrix quadrants.
        
        Parameters
        ----------
        fdr_threshold : float
            FDR threshold for EmptyDrops classification
            
        Returns
        -------
        pd.Series
            Classification for each droplet
        """
        # Get Cell Ranger and EmptyDrops calls
        cellranger_cells = set(self.filtered_adata.obs_names)
        emptydrops_cells = set(self.results[self.results['FDR'] < fdr_threshold].index)
        
        # Create classification series
        categories = pd.Series(index=self.raw_adata.obs_names, dtype='str')
        
        # Classify each droplet
        for barcode in self.raw_adata.obs_names:
            if barcode in cellranger_cells:
                if barcode in emptydrops_cells:
                    categories[barcode] = 'Both_Called'  # True Positive
                else:
                    categories[barcode] = 'CellRanger_Only'  # False Negative
            else:
                if barcode in emptydrops_cells:
                    categories[barcode] = 'EmptyDrops_Only'  # False Positive
                else:
                    categories[barcode] = 'Neither_Called'  # True Negative
        
        return categories
    
    def analyze_quadrants(self, fdr_threshold: float = 0.001, 
                         min_cells: int = 10,
                         genes_of_interest: List[str] = None):
        """
        Analyze gene expression patterns in each confusion matrix quadrant.
        
        Parameters
        ----------
        fdr_threshold : float
            FDR threshold for EmptyDrops classification
        min_cells : int
            Minimum number of cells required in a category for analysis
        genes_of_interest : List[str], optional
            List of specific genes to analyze in detail
        """
        print(f"Analyzing confusion matrix quadrants (FDR < {fdr_threshold})...")
        
        # Get droplet classifications
        categories = self.classify_droplets(fdr_threshold)
        
        # Create quadrant directories
        quadrant_dirs = {}
        for quadrant in ['Both_Called', 'EmptyDrops_Only', 'CellRanger_Only', 'Neither_Called']:
            quadrant_dir = os.path.join(self.output_dir, quadrant)
            os.makedirs(quadrant_dir, exist_ok=True)
            quadrant_dirs[quadrant] = quadrant_dir
        
        # Calculate basic statistics for each quadrant
        quadrant_stats = {}
        for quadrant in quadrant_dirs:
            mask = categories == quadrant
            cells_in_quadrant = np.sum(mask)
            
            if cells_in_quadrant > 0:
                # Get expression data for this quadrant
                quadrant_expr = self.raw_adata[mask]
                
                # Calculate statistics
                total_counts = np.array(quadrant_expr.X.sum(axis=1)).flatten()
                genes_detected = np.array((quadrant_expr.X > 0).sum(axis=1)).flatten()
                
                stats = {
                    'n_cells': cells_in_quadrant,
                    'median_total_counts': np.median(total_counts),
                    'median_genes_detected': np.median(genes_detected),
                    'mean_total_counts': np.mean(total_counts),
                    'mean_genes_detected': np.mean(genes_detected)
                }
                quadrant_stats[quadrant] = stats
        
        # Save quadrant statistics
        stats_df = pd.DataFrame(quadrant_stats).T
        stats_df.to_csv(os.path.join(self.output_dir, 'quadrant_statistics.csv'))
        
        # Plot quadrant statistics
        self._plot_quadrant_stats(stats_df)
        
        # Analyze disagreement cases in detail
        self._analyze_disagreements(categories, quadrant_dirs)
        
        # Analyze specific genes if provided
        if genes_of_interest:
            self._analyze_genes_across_quadrants(categories, genes_of_interest)
    
    def _plot_quadrant_stats(self, stats_df: pd.DataFrame):
        """Plot basic statistics for each quadrant."""
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        fig.suptitle('Quadrant Statistics Comparison', fontsize=16, y=1.02)
        
        # Plot cell counts
        sns.barplot(data=stats_df.reset_index(), x='index', y='n_cells', ax=axes[0,0])
        axes[0,0].set_title('Number of Cells per Quadrant')
        axes[0,0].set_xticklabels(axes[0,0].get_xticklabels(), rotation=45)
        
        # Plot median total counts
        sns.barplot(data=stats_df.reset_index(), x='index', y='median_total_counts', ax=axes[0,1])
        axes[0,1].set_title('Median Total Counts per Cell')
        axes[0,1].set_xticklabels(axes[0,1].get_xticklabels(), rotation=45)
        
        # Plot median genes detected
        sns.barplot(data=stats_df.reset_index(), x='index', y='median_genes_detected', ax=axes[1,0])
        axes[1,0].set_title('Median Genes Detected per Cell')
        axes[1,0].set_xticklabels(axes[1,0].get_xticklabels(), rotation=45)
        
        # Plot mean vs median comparison
        comparison_data = pd.melt(stats_df.reset_index(), 
                                id_vars=['index'],
                                value_vars=['mean_total_counts', 'median_total_counts'])
        sns.barplot(data=comparison_data, x='index', y='value', hue='variable', ax=axes[1,1])
        axes[1,1].set_title('Mean vs Median Total Counts')
        axes[1,1].set_xticklabels(axes[1,1].get_xticklabels(), rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'quadrant_statistics.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _analyze_disagreements(self, categories: pd.Series, quadrant_dirs: Dict[str, str]):
        """Analyze cases where EmptyDrops and CellRanger disagree."""
        # Focus on disagreement quadrants
        disagreement_categories = ['EmptyDrops_Only', 'CellRanger_Only']
        
        # Set up multiprocessing
        n_processes = max(1, os.cpu_count() - 1)
        
        for category in disagreement_categories:
            mask = categories == category
            if np.sum(mask) == 0:
                continue
            
            print(f"\nAnalyzing {category} category...")
            
            # Get expression data for this category
            category_expr = self.raw_adata[mask]
            
            # Calculate total UMI counts per cell for normalization
            total_umi_counts = np.array(category_expr.X.sum(axis=1)).flatten()
            
            # Calculate gene-level statistics
            n_cells = category_expr.shape[0]
            detection_rate = np.array((category_expr.X > 0).sum(axis=0)).flatten() / n_cells
            
            # Split genes into batches for parallel processing
            n_genes = category_expr.shape[1]
            batch_size = max(100, n_genes // (n_processes * 10))  # Ensure reasonable batch size
            gene_batches = [range(i, min(i + batch_size, n_genes)) 
                          for i in range(0, n_genes, batch_size)]
            
            # Prepare arguments for parallel processing
            process_args = [(indices, category_expr.X, total_umi_counts) for indices in gene_batches]
            
            # Process batches in parallel
            normalized_expr = [0] * n_genes  # Initialize with zeros
            expr_stats = [None] * n_genes  # Store distribution statistics
            
            with Pool(n_processes) as pool:
                # Process batches with progress bar
                results = []
                for batch_results in tqdm(
                    pool.imap(process_gene_batch, process_args),
                    total=len(gene_batches),
                    desc="Calculating normalized expression"
                ):
                    results.extend(batch_results)
                
                # Collect results
                for idx, (mean_pct, stats) in results:
                    normalized_expr[idx] = mean_pct
                    expr_stats[idx] = stats
            
            # Create gene statistics DataFrame
            gene_stats = pd.DataFrame({
                'gene': self.raw_adata.var_names,
                'detection_rate': detection_rate,
                'mean_pct_of_cell': normalized_expr,
                'expr_stats': expr_stats
            })
            
            # Sort by detection rate and save
            gene_stats = gene_stats.sort_values('detection_rate', ascending=False)
            gene_stats.to_csv(os.path.join(quadrant_dirs[category], 'gene_statistics.csv'))
            
            # Plot top genes
            self._plot_top_genes(gene_stats.head(20), category, quadrant_dirs[category])
    
    def _plot_top_genes(self, gene_stats: pd.DataFrame, category: str, output_dir: str):
        """Plot statistics for top genes in a category."""
        # Create figure and primary axis
        fig, ax1 = plt.subplots(figsize=(15, 8))  # Increased figure width for box plots
        
        # Plot detection rates as bars on primary axis
        bars = ax1.bar(range(len(gene_stats)), gene_stats['detection_rate'] * 100, color='skyblue', alpha=0.6)
        ax1.set_xticks(range(len(gene_stats)))
        ax1.set_xticklabels(gene_stats['gene'], rotation=45, ha='right')
        ax1.set_ylabel('Detection Rate (%)', color='skyblue', fontsize=10)
        ax1.tick_params(axis='y', labelcolor='skyblue')
        
        # Create second y-axis for normalized expression
        ax2 = ax1.twinx()
        ax2.set_ylabel('Expression (% of Cell\'s Total)', color='red', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='red')
        
        # Plot box plots and mean values
        for i, row in enumerate(gene_stats.itertuples()):
            if row.expr_stats is not None:
                # Extract statistics
                q1, median, q3, whislo, whishi, fliers = row.expr_stats
                
                # Plot box plot
                box_width = 0.4
                box_color = 'red'
                alpha = 0.3
                
                # Box
                ax2.add_patch(plt.Rectangle((i - box_width/2, q1), box_width, q3-q1,
                                         facecolor=box_color, alpha=alpha))
                
                # Median line
                ax2.plot([i - box_width/2, i + box_width/2], [median, median],
                        color=box_color, linewidth=2, alpha=0.6)
                
                # Whiskers
                ax2.plot([i, i], [q3, whishi], color=box_color, linestyle='-', alpha=alpha)
                ax2.plot([i, i], [q1, whislo], color=box_color, linestyle='-', alpha=alpha)
                
                # Whisker caps
                cap_width = box_width/3
                ax2.plot([i - cap_width/2, i + cap_width/2], [whishi, whishi],
                        color=box_color, linestyle='-', alpha=alpha)
                ax2.plot([i - cap_width/2, i + cap_width/2], [whislo, whislo],
                        color=box_color, linestyle='-', alpha=alpha)
                
                # Plot outliers if any
                if len(fliers) > 0:
                    ax2.plot([i] * len(fliers), fliers, 'r.',
                            markersize=4, alpha=0.3)
                
                # Add mean value as a red dot
                ax2.plot(i, row.mean_pct_of_cell, 'ro', alpha=0.8)
                
                # Add text for mean, median, and quartiles
                ax2.text(i, whishi, f'Mean: {row.mean_pct_of_cell:.2f}%\nMedian: {median:.2f}%',
                        ha='center', va='bottom', color='red', fontsize=8)
        
        # Set title
        plt.title(f'Top 20 Genes by Detection Rate in {category}', pad=20)
        
        # Add legend
        legend_elements = [
            plt.Rectangle((0,0), 1, 1, facecolor='skyblue', alpha=0.6, label='Detection Rate (% of cells)'),
            plt.Rectangle((0,0), 1, 1, facecolor='red', alpha=0.3, label='Expression Distribution'),
            plt.Line2D([0], [0], marker='o', color='red', label='Mean Expression',
                      markerfacecolor='red', markersize=8, linestyle='None', alpha=0.8)
        ]
        ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1.1))
        
        # Add explanation text box
        plt.figtext(0.02, 0.02,
                   'Detection Rate = % of cells expressing the gene\n' +
                   'Box Plot = Distribution of expression in expressing cells\n' +
                   'Red Dot = Mean expression as % of cell\'s total transcripts\n' +
                   'Box = 25th to 75th percentile, Line = Median',
                   fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, 'top_genes.png'),
                   dpi=300,
                   bbox_inches='tight')
        plt.close()
    
    def _analyze_genes_across_quadrants(self, categories: pd.Series, genes: List[str]):
        """Analyze specific genes across all quadrants."""
        for gene in genes:
            try:
                gene_idx = np.where(self.raw_adata.var_names == gene)[0][0]
            except IndexError:
                print(f"Warning: Gene {gene} not found in dataset")
                continue
            
            # Get gene expression data
            gene_expr = np.array(self.raw_adata.X[:, gene_idx].todense()).flatten()
            
            # Create gene-specific directory in differential_gene_analysis
            gene_dir = os.path.join('differential_gene_analysis', f'gene_{gene}')
            os.makedirs(gene_dir, exist_ok=True)
            
            # Create distribution plot
            plt.figure(figsize=(12, 8))
            categories_order = ['Both_Called', 'EmptyDrops_Only', 'CellRanger_Only', 'Neither_Called']
            data_to_plot = []
            labels = []
            
            for category in categories_order:
                mask = categories == category
                if np.sum(mask) > 0:
                    expr_in_category = gene_expr[mask]
                    expr_in_category = expr_in_category[expr_in_category > 0]  # Only positive values for log scale
                    if len(expr_in_category) > 0:
                        data_to_plot.append(expr_in_category)
                        labels.append(f"{category}\n(n={len(expr_in_category)})")
            
            plt.violinplot(data_to_plot, showmeans=True, showextrema=True)
            plt.yscale('log')
            plt.xticks(range(1, len(labels) + 1), labels, rotation=45)
            plt.title(f'{gene} Expression Distribution Across Cell Categories')
            plt.ylabel('Expression Level (log scale)')
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(gene_dir, f'{gene}_distribution.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            total_counts = np.array(self.raw_adata.X.sum(axis=1)).flatten()
            
            colors = {'Both_Called': 'blue', 'EmptyDrops_Only': 'green', 
                     'CellRanger_Only': 'red', 'Neither_Called': 'gray'}
            
            for category in categories_order:
                mask = categories == category
                if np.sum(mask) > 0:
                    plt.scatter(total_counts[mask], gene_expr[mask], 
                              c=colors[category], label=category, alpha=0.5, s=1)
            
            plt.xscale('log')
            plt.yscale('log')
            plt.xlabel('Total UMI Counts (log scale)')
            plt.ylabel(f'{gene} Expression (log scale)')
            plt.title(f'{gene} Expression vs Total UMI Counts')
            plt.legend()
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.tight_layout()
            plt.savefig(os.path.join(gene_dir, f'{gene}_scatter.png'), dpi=300, bbox_inches='tight')
            plt.close()

def main():
    """Run the confusion matrix analysis."""
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze confusion matrix quadrants for gene expression.')
    parser.add_argument('--fdr_threshold', type=float, default=0.001,
                       help='FDR threshold for EmptyDrops classification (default: 0.001)')
    parser.add_argument('--gene', type=str, default=None,
                       help='Specific gene to analyze')
    
    args = parser.parse_args()
    
    # Initialize analyzer with data paths
    analyzer = QuadrantAnalyzer(
        raw_h5_path='raw_feature_bc_matrix.h5',
        filtered_h5_path='filtered_feature_bc_matrix.h5',
        emptydrops_results_path='empty_drops_visualizations/empty_drops_results.csv'
    )
    
    # Run analysis with specified FDR threshold and focus on INSL3
    analyzer.analyze_quadrants(
        fdr_threshold=args.fdr_threshold,
        genes_of_interest=['INSL3']
    )
    
    print("\nAnalysis complete! Results are saved in the 'confusion_matrix_analysis' directory.")

if __name__ == '__main__':
    main() 