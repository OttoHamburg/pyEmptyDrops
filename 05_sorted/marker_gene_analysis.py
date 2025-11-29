#!/usr/bin/env python3
"""
Marker Gene Analysis for Cells Called by Both Methods

This script analyzes gene expression patterns in cells that are called by both
EmptyDrops and CellRanger methods to identify high-quality marker genes and
create comprehensive visualization matrices.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
from typing import List, Tuple, Dict, Optional
from tqdm import tqdm
import json
from datetime import datetime

# Set up directories
ANALYSIS_DIR = "marker_gene_analysis"
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")
TABLES_DIR = os.path.join(ANALYSIS_DIR, "tables")
RESULTS_DIR = os.path.join(ANALYSIS_DIR, "results")

for dir_path in [ANALYSIS_DIR, PLOTS_DIR, TABLES_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

class MarkerGeneAnalyzer:
    """
    Comprehensive analyzer for marker genes in cells called by both methods.
    """
    
    def __init__(self, raw_h5_path: str, filtered_h5_path: str, emptydrops_results_path: str, 
                 fdr_threshold: float = 0.05):
        """
        Initialize the marker gene analyzer.
        
        Parameters:
        -----------
        raw_h5_path : str
            Path to raw feature-barcode matrix
        filtered_h5_path : str
            Path to filtered feature-barcode matrix  
        emptydrops_results_path : str
            Path to EmptyDrops results CSV
        fdr_threshold : float
            FDR threshold for EmptyDrops calls
        """
        self.raw_h5_path = raw_h5_path
        self.filtered_h5_path = filtered_h5_path
        self.emptydrops_results_path = emptydrops_results_path
        self.fdr_threshold = fdr_threshold
        
        # Data containers
        self.raw_adata = None
        self.filtered_adata = None
        self.emptydrops_results = None
        self.cell_categories = None
        self.marker_analysis = None
        
        print(f"Initialized MarkerGeneAnalyzer with FDR threshold: {fdr_threshold}")
    
    def load_data(self):
        """Load all required datasets."""
        print("Loading datasets...")
        
        # Load raw data
        print("  Loading raw data...")
        self.raw_adata = sc.read_10x_h5(self.raw_h5_path)
        print(f"    Raw data shape: {self.raw_adata.shape}")
        
        # Load filtered data
        print("  Loading filtered data...")
        self.filtered_adata = sc.read_10x_h5(self.filtered_h5_path)
        print(f"    Filtered data shape: {self.filtered_adata.shape}")
        
        # Load EmptyDrops results
        print("  Loading EmptyDrops results...")
        if os.path.exists(self.emptydrops_results_path):
            self.emptydrops_results = pd.read_csv(self.emptydrops_results_path, index_col=0)
            print(f"    EmptyDrops results shape: {self.emptydrops_results.shape}")
        else:
            # Try to find the most recent EmptyDrops results
            print("    Specified EmptyDrops results not found, looking for recent results...")
            results_dir = "empty_drops_visualizations"
            if os.path.exists(results_dir):
                # Look for CSV files with FDR results
                potential_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
                if potential_files:
                    print(f"    Found potential results files: {potential_files}")
                    # For now, we'll create a simple analysis without EmptyDrops results
                    self.emptydrops_results = None
                else:
                    self.emptydrops_results = None
            else:
                self.emptydrops_results = None
        
        print("✅ Data loading completed!")
    
    def categorize_cells(self):
        """Categorize cells based on calling methods."""
        print("Categorizing cells by calling methods...")
        
        # Get cell barcodes from each method
        cellranger_cells = set(self.filtered_adata.obs_names)
        print(f"  CellRanger called cells: {len(cellranger_cells):,}")
        
        if self.emptydrops_results is not None:
            emptydrops_cells = set(
                self.emptydrops_results[self.emptydrops_results['FDR'] < self.fdr_threshold].index
            )
            print(f"  EmptyDrops called cells: {len(emptydrops_cells):,}")
        else:
            # Create a dummy EmptyDrops result for demonstration
            print("  No EmptyDrops results available, creating analysis for CellRanger cells only")
            emptydrops_cells = set()
        
        # Create cell categories
        self.cell_categories = pd.Series(index=self.raw_adata.obs_names, dtype='str')
        
        for barcode in self.raw_adata.obs_names:
            if barcode in cellranger_cells:
                if barcode in emptydrops_cells:
                    self.cell_categories[barcode] = 'Called by Both Methods'
                else:
                    self.cell_categories[barcode] = 'CellRanger Only'
            else:
                if barcode in emptydrops_cells:
                    self.cell_categories[barcode] = 'EmptyDrops Only'
                else:
                    self.cell_categories[barcode] = 'Called by Neither'
        
        # Print summary
        category_counts = self.cell_categories.value_counts()
        print("  Cell categorization summary:")
        for category, count in category_counts.items():
            print(f"    {category}: {count:,}")
        
        return category_counts
    
    def analyze_marker_genes(self, min_cells: int = 50, min_detection_rate: float = 0.1,
                           target_genes: List[str] = None):
        """
        Analyze potential marker genes in cells called by both methods.
        
        Parameters:
        -----------
        min_cells : int
            Minimum number of cells expressing the gene
        min_detection_rate : float
            Minimum detection rate in target cells
        target_genes : list
            Specific genes to include in analysis
        """
        print("Analyzing marker genes...")
        
        if target_genes is None:
            target_genes = ['PRM1', 'TNP1', 'SYCP3', 'ACRV1', 'ODF1', 'TNP2', 'SMCP', 'ACROSIN']
        
        # Focus ONLY on cells called by both methods for main analysis
        if 'Called by Both Methods' in self.cell_categories.values:
            target_cells_mask = self.cell_categories == 'Called by Both Methods'
            analysis_label = "Cells Called by Both Methods"
        else:
            print("  WARNING: No cells called by both methods found!")
            target_cells_mask = self.cell_categories == 'CellRanger Only'
            analysis_label = "CellRanger Called Cells (fallback)"
        
        target_cells = self.cell_categories[target_cells_mask].index
        print(f"  Analyzing {len(target_cells):,} {analysis_label.lower()}")
        
        # Get expression data for target cells
        target_cell_data = self.raw_adata[target_cells]
        
        # Calculate total UMI counts per cell
        total_counts = np.array(target_cell_data.X.sum(axis=1)).flatten()
        
        # Analyze genes
        gene_analysis = []
        
        print("  Calculating gene statistics...")
        for gene_idx, gene_name in enumerate(tqdm(self.raw_adata.var_names, desc="Processing genes")):
            # Get gene expression
            gene_counts = np.array(target_cell_data.X[:, gene_idx].todense()).flatten()
            
            # Calculate metrics
            n_expressing = np.sum(gene_counts > 0)
            detection_rate = n_expressing / len(target_cells)
            
            # Skip genes with low detection (except target genes which we always include)
            if n_expressing < min_cells or detection_rate < min_detection_rate:
                if gene_name not in target_genes:
                    continue
                else:
                    print(f"  Including low-expressing target gene {gene_name} ({n_expressing} cells, {detection_rate:.3f} detection rate)")
            
            # Calculate expression metrics
            mean_counts = np.mean(gene_counts)
            median_counts = np.median(gene_counts)
            max_counts = np.max(gene_counts)
            
            # Calculate normalized expression
            normalized_expr = gene_counts / (total_counts + 1e-10)
            mean_normalized = np.mean(normalized_expr)
            median_normalized = np.median(normalized_expr)
            
            # Calculate expression in target range (1-6% for sperm markers)
            target_range_mask = (normalized_expr >= 0.01) & (normalized_expr <= 0.06)
            n_in_target_range = np.sum(target_range_mask)
            target_range_percentage = n_in_target_range / len(target_cells) * 100
            
            # Store results
            gene_analysis.append({
                'gene': gene_name,
                'n_expressing_cells': n_expressing,
                'detection_rate': detection_rate,
                'mean_counts': mean_counts,
                'median_counts': median_counts,
                'max_counts': max_counts,
                'mean_normalized_expr': mean_normalized,
                'median_normalized_expr': median_normalized,
                'n_in_target_range': n_in_target_range,
                'target_range_percentage': target_range_percentage,
                'is_target_gene': gene_name in target_genes
            })
        
        # Convert to DataFrame
        self.marker_analysis = pd.DataFrame(gene_analysis)
        
        # Sort by target range percentage and detection rate
        self.marker_analysis = self.marker_analysis.sort_values(
            ['target_range_percentage', 'detection_rate'], 
            ascending=[False, False]
        )
        
        print(f"  Found {len(self.marker_analysis)} genes meeting criteria")
        print(f"  Target genes found: {self.marker_analysis[self.marker_analysis['is_target_gene']]['gene'].tolist()}")
        
        return self.marker_analysis
    
    def create_comparison_table(self, top_n: int = 150):
        """Create a comprehensive comparison table of top marker genes."""
        print(f"Creating comparison table for top {top_n} genes...")
        
        if self.marker_analysis is None:
            print("  No marker analysis available. Run analyze_marker_genes() first.")
            return None
        
        # Get top genes + all target genes
        top_genes = self.marker_analysis.head(top_n).copy()
        target_genes_df = self.marker_analysis[self.marker_analysis['is_target_gene'] == True]
        
        # Combine and remove duplicates
        combined_genes = pd.concat([top_genes, target_genes_df]).drop_duplicates(subset=['gene'])
        top_genes = combined_genes.copy()
        
        # Add category analysis
        category_analysis = []
        
        for _, gene_row in top_genes.iterrows():
            gene_name = gene_row['gene']
            gene_idx = list(self.raw_adata.var_names).index(gene_name)
            
            # Calculate expression by category
            category_stats = {}
            for category in self.cell_categories.unique():
                if pd.isna(category):
                    continue
                
                category_mask = self.cell_categories == category
                category_cells = self.cell_categories[category_mask].index
                
                if len(category_cells) == 0:
                    continue
                
                # Get expression data
                gene_counts = np.array(self.raw_adata[category_cells, gene_idx].X.todense()).flatten()
                total_counts = np.array(self.raw_adata[category_cells].X.sum(axis=1)).flatten()
                
                # Calculate normalized expression
                normalized_expr = gene_counts / (total_counts + 1e-10)
                
                # Calculate statistics
                n_expressing = np.sum(gene_counts > 0)
                detection_rate = n_expressing / len(category_cells) if len(category_cells) > 0 else 0
                mean_normalized = np.mean(normalized_expr)
                
                category_stats[category] = {
                    'n_cells': len(category_cells),
                    'n_expressing': n_expressing,
                    'detection_rate': detection_rate,
                    'mean_normalized_expr': mean_normalized
                }
            
            category_analysis.append({
                'gene': gene_name,
                'category_stats': category_stats
            })
        
        # Create detailed table
        detailed_table = []
        for i, (_, gene_row) in enumerate(top_genes.iterrows()):
            gene_name = gene_row['gene']
            cat_stats = category_analysis[i]['category_stats']
            
            row = {
                'rank': i + 1,
                'gene': gene_name,
                'is_target_gene': gene_row['is_target_gene'],
                'overall_detection_rate': gene_row['detection_rate'],
                'target_range_percentage': gene_row['target_range_percentage'],
                'mean_normalized_expr': gene_row['mean_normalized_expr']
            }
            
            # Add category-specific statistics
            for category in ['Called by Both Methods', 'CellRanger Only', 'EmptyDrops Only', 'Called by Neither']:
                if category in cat_stats:
                    stats = cat_stats[category]
                    row[f'{category}_n_cells'] = stats['n_cells']
                    row[f'{category}_detection_rate'] = stats['detection_rate']
                    row[f'{category}_mean_expr'] = stats['mean_normalized_expr']
                else:
                    row[f'{category}_n_cells'] = 0
                    row[f'{category}_detection_rate'] = 0.0
                    row[f'{category}_mean_expr'] = 0.0
            
            detailed_table.append(row)
        
        detailed_df = pd.DataFrame(detailed_table)
        
        # Save table
        table_path = os.path.join(TABLES_DIR, 'marker_gene_comparison_table.csv')
        detailed_df.to_csv(table_path, index=False)
        print(f"  Saved detailed comparison table to: {table_path}")
        
        # Create summary table for display
        summary_cols = [
            'rank', 'gene', 'is_target_gene', 'overall_detection_rate', 
            'target_range_percentage', 'Called by Both Methods_detection_rate',
            'CellRanger Only_detection_rate', 'EmptyDrops Only_detection_rate'
        ]
        
        summary_df = detailed_df[summary_cols].copy()
        summary_df.columns = [
            'Rank', 'Gene', 'Target Gene', 'Overall Detection Rate', 
            'Target Range %', 'Both Methods Detection', 
            'CellRanger Detection', 'EmptyDrops Detection'
        ]
        
        # Round numerical columns
        numerical_cols = ['Overall Detection Rate', 'Target Range %', 'Both Methods Detection', 
                         'CellRanger Detection', 'EmptyDrops Detection']
        for col in numerical_cols:
            if col in summary_df.columns:
                summary_df[col] = summary_df[col].round(3)
        
        # Save summary
        summary_path = os.path.join(TABLES_DIR, 'marker_gene_summary_table.csv')
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved summary table to: {summary_path}")
        
        # Display top 20 and all target genes
        print("\n📊 TOP 20 MARKER GENES SUMMARY:")
        print("=" * 100)
        print(summary_df.head(20).to_string(index=False))
        
        # Show all target genes
        target_genes_in_table = summary_df[summary_df['Target Gene'] == True]
        if len(target_genes_in_table) > 0:
            print("\n🎯 ALL TARGET GENES IN ANALYSIS:")
            print("=" * 100)
            print(target_genes_in_table.to_string(index=False))
        
        return detailed_df, summary_df
    
    def create_kernel_density_matrix(self, genes: List[str] = None, figsize: Tuple[int, int] = (20, 16)):
        """
        Create a matrix of kernel density plots for top marker genes.
        
        Parameters:
        -----------
        genes : list
            List of genes to plot. If None, uses top genes from analysis.
        figsize : tuple
            Figure size for the plot matrix
        """
        print("Creating kernel density plot matrix...")
        
        if genes is None:
            if self.marker_analysis is None:
                print("  No marker analysis available. Run analyze_marker_genes() first.")
                return None
            
            # Always include priority genes in fixed positions
            priority_genes = ['PRM1', 'TNP1', 'SYCP3', 'ACRV1']
            
            # Get top performing genes (excluding priority genes to avoid duplicates)
            top_performing = []
            for _, row in self.marker_analysis.iterrows():
                gene_name = row['gene']
                if gene_name not in priority_genes and len(top_performing) < 12:
                    top_performing.append(gene_name)
            
            # Combine: priority genes first (consistent positioning), then top performers
            genes = priority_genes + top_performing
            
            # Ensure we only include genes that exist in the data
            existing_genes = []
            for gene in genes:
                if gene in self.raw_adata.var_names:
                    existing_genes.append(gene)
                else:
                    print(f"  Warning: Gene {gene} not found in dataset")
            
            genes = existing_genes[:16]  # 4x4 matrix
        
        n_genes = len(genes)
        n_cols = 4
        n_rows = int(np.ceil(n_genes / n_cols))
        
        print(f"  Creating {n_rows}x{n_cols} matrix for {n_genes} genes")
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Plot each gene
        for i, gene in enumerate(genes):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            try:
                self._plot_gene_kernel_density(gene, ax)
            except Exception as e:
                print(f"  Warning: Could not plot {gene}: {e}")
                ax.text(0.5, 0.5, f'Error plotting\n{gene}', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(gene)
        
        # Hide empty subplots
        for i in range(n_genes, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(PLOTS_DIR, 'marker_gene_kernel_density_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        plt.close()
        
        print(f"  Saved kernel density matrix to: {plot_path}")
        return plot_path
    
    def _plot_gene_kernel_density(self, gene: str, ax):
        """Plot kernel density for a single gene."""
        if gene not in self.raw_adata.var_names:
            ax.text(0.5, 0.5, f'Gene {gene}\nnot found', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(gene)
            return
        
        gene_idx = list(self.raw_adata.var_names).index(gene)
        
        # Get total UMI counts
        total_counts = np.array(self.raw_adata.X.sum(axis=1)).flatten()
        
        # Define colors for categories
        colors = {
            'Called by Both Methods': 'green',
            'EmptyDrops Only': 'blue', 
            'CellRanger Only': 'orange',
            'Called by Neither': 'red'
        }
        
        # Plot for each category
        for category, color in colors.items():
            if category not in self.cell_categories.values:
                continue
                
            category_mask = self.cell_categories == category
            category_cells = self.cell_categories[category_mask].index
            
            if len(category_cells) == 0:
                continue
            
            # Get gene expression
            gene_counts = np.array(self.raw_adata[category_cells, gene_idx].X.todense()).flatten()
            cat_total_counts = total_counts[category_mask]
            
            # Calculate normalized expression
            normalized_expr = gene_counts / (cat_total_counts + 1e-10)
            
            # Only plot cells with non-zero expression
            nonzero_expr = normalized_expr[normalized_expr > 0]
            
            if len(nonzero_expr) > 10:  # Need enough points for KDE
                try:
                    sns.kdeplot(
                        data=nonzero_expr,
                        label=f'{category} (n={len(nonzero_expr)})',
                        color=color,
                        log_scale=True,
                        linewidth=2,
                        ax=ax
                    )
                except:
                    # Fallback to histogram if KDE fails
                    ax.hist(nonzero_expr, bins=20, alpha=0.3, color=color, 
                           label=f'{category} (n={len(nonzero_expr)})', density=True)
        
        ax.set_xlabel('Normalized Expression (log scale)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(gene, fontsize=12, weight='bold')
        ax.grid(True, which='both', linestyle=':', alpha=0.3)
        ax.legend(fontsize=8, loc='upper right')
    
    def run_complete_analysis(self, top_genes_for_plots: int = 16):
        """Run the complete marker gene analysis pipeline."""
        print("🧬 Starting Complete Marker Gene Analysis")
        print("=" * 60)
        
        # Load data
        self.load_data()
        
        # Categorize cells
        category_counts = self.categorize_cells()
        
        # Analyze marker genes
        marker_df = self.analyze_marker_genes()
        
        # Create comparison tables
        detailed_table, summary_table = self.create_comparison_table()
        
        # Create kernel density matrix
        if len(marker_df) > 0:
            # Get top genes including target genes
            top_genes = marker_df.head(12)
            target_genes = marker_df[marker_df['is_target_gene']]
            
            # Combine and get unique genes for plotting
            all_genes = pd.concat([target_genes, top_genes]).drop_duplicates(subset=['gene'])
            genes_to_plot = all_genes['gene'].head(top_genes_for_plots).tolist()
            
            plot_path = self.create_kernel_density_matrix(genes_to_plot)
            
            print(f"\n🎯 Analysis complete!")
            print(f"📊 Tables saved to: {TABLES_DIR}")
            print(f"📈 Plots saved to: {PLOTS_DIR}")
            print(f"💾 Results available in: {RESULTS_DIR}")
            
            # Save analysis summary
            summary = {
                'analysis_date': datetime.now().isoformat(),
                'fdr_threshold': self.fdr_threshold,
                'category_counts': category_counts.to_dict(),
                'total_genes_analyzed': len(marker_df),
                'top_genes': genes_to_plot,
                'target_genes_found': marker_df[marker_df['is_target_gene']]['gene'].tolist()
            }
            
            summary_path = os.path.join(RESULTS_DIR, 'analysis_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
        else:
            print("⚠️  No marker genes found meeting the criteria")
            return None

def main():
    """Main function to run the marker gene analysis."""
    print("🧬 Marker Gene Analysis for Cells Called by Both Methods")
    print("=" * 70)
    
    # Configuration
    raw_h5_path = "data/raw_feature_bc_matrix.h5"
    filtered_h5_path = "data/filtered_feature_bc_matrix.h5" 
    emptydrops_results_path = "empty_drops_visualizations/empty_drops_results.csv"  # Updated pathçc
    
    # Initialize analyzer
    analyzer = MarkerGeneAnalyzer(
        raw_h5_path=raw_h5_path,
        filtered_h5_path=filtered_h5_path,
        emptydrops_results_path=emptydrops_results_path,
        fdr_threshold=0.05
    )
    
    # Run complete analysis
    try:
        summary = analyzer.run_complete_analysis(top_genes_for_plots=16)
        
        if summary:
            print("\n✅ Analysis Summary:")
            print(f"   Target genes found: {summary['target_genes_found']}")
            print(f"   Total genes analyzed: {summary['total_genes_analyzed']}")
            print(f"   Plots created for: {len(summary['top_genes'])} genes")
        
    except Exception as e:
        print(f"❌ Error during analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 