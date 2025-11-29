import optuna
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc
import h5py
import os
import sys
import logging
import time
import ast
import pickle
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import contextlib
import io
from tqdm import tqdm
from sklearn.metrics import silhouette_score
from scipy import stats

from empty_drops import empty_drops

# Create directories for optimization results
OPTIMIZATION_DIR = "optuna_optimization"
STUDIES_DIR = os.path.join(OPTIMIZATION_DIR, "studies")
PLOTS_DIR = os.path.join(OPTIMIZATION_DIR, "plots")
RESULTS_DIR = os.path.join(OPTIMIZATION_DIR, "results")

for dir_path in [OPTIMIZATION_DIR, STUDIES_DIR, PLOTS_DIR, RESULTS_DIR]:
    os.makedirs(dir_path, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OPTIMIZATION_DIR, 'optimization.log')),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Context manager to suppress console output
@contextlib.contextmanager
def suppress_stdout_stderr():
    """Context manager to suppress all stdout and stderr output."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr

@contextlib.contextmanager
def capture_output():
    """Context manager to capture stdout and stderr output."""
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        yield stdout_capture, stderr_capture
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr

def calculate_multi_gene_score(emptydrops_results, raw_adata, target_genes=['PRM1', 'TNP1'], 
                              fdr_threshold=0.01, iteration=None):
    """
    Calculate multi-objective score for optimization targeting multiple genes.
    Now includes normalized metrics to prevent parameter inflation bias.
    
    Parameters:
    -----------
    emptydrops_results : pd.DataFrame
        Results from empty_drops function
    raw_adata : AnnData
        Raw data matrix
    target_genes : list
        List of genes to optimize for
    fdr_threshold : float
        FDR threshold for calling cells
    iteration : int, optional
        Current iteration number for saving results
    
    Returns:
    --------
    dict : Multi-objective scores and metrics including normalized versions
    """
    try:
        # Get cells called by EmptyDrops
        called_cells = emptydrops_results[emptydrops_results['FDR'] < fdr_threshold]
        called_cells = called_cells.sort_values(by='FDR', ascending=True)
    
        if len(called_cells) == 0:
            logger.warning(f"No cells called with FDR < {fdr_threshold}")
            return {gene: 0 for gene in target_genes}
        
        gene_scores = {}
        total_cells = len(called_cells)
        
        # Raw counts for each gene (original approach)
        for gene in target_genes:
            if gene not in raw_adata.var_names:
                logger.warning(f"{gene} gene not found in data")
                gene_scores[gene] = 0
                continue
            
            # Extract gene expression for called cells
            gene_indices = called_cells.index
            gene_counts = raw_adata[gene_indices, gene].X.toarray().flatten()
            total_counts = raw_adata[gene_indices].X.sum(axis=1).A1
        
            # Calculate gene expression as percentage of total transcripts
            gene_percentages = np.divide(gene_counts, total_counts, 
                                       out=np.zeros_like(gene_counts, dtype=float), 
                                       where=total_counts!=0) * 100
        
            # Count cells in target range (1-6% for sperm cells)
            target_cells = np.sum((gene_percentages >= 1.0) & (gene_percentages <= 6.0))
            gene_scores[gene] = float(target_cells)
            
            # NEW: Calculate normalized efficiency metrics
            efficiency = target_cells / total_cells if total_cells > 0 else 0
            gene_scores[f'{gene}_efficiency'] = efficiency  # Percentage of called cells in target range
            
            # NEW: Calculate precision (target cells vs cells with any expression)
            expressing_cells = np.sum(gene_counts > 0)
            precision = target_cells / expressing_cells if expressing_cells > 0 else 0
            gene_scores[f'{gene}_precision'] = precision
            
            # NEW: Calculate biological quality score (mean expression in target range)
            target_mask = (gene_percentages >= 1.0) & (gene_percentages <= 6.0)
            if np.sum(target_mask) > 0:
                quality_score = np.mean(gene_percentages[target_mask])
                gene_scores[f'{gene}_quality'] = quality_score
            else:
                gene_scores[f'{gene}_quality'] = 0
        
        # Basic composite metrics
        gene_scores['total_called_cells'] = total_cells
        gene_scores['combined_score'] = sum(gene_scores[gene] for gene in target_genes)
        gene_scores['weighted_score'] = gene_scores.get('PRM1', 0) * 0.6 + gene_scores.get('TNP1', 0) * 0.4
        
        # NEW: Normalized composite scores
        # Efficiency-weighted score (rewards quality over quantity)
        efficiency_score = (
            gene_scores.get('PRM1_efficiency', 0) * 0.6 + 
            gene_scores.get('TNP1_efficiency', 0) * 0.4
        ) * 1000  # Scale to make it comparable to raw counts
        gene_scores['efficiency_weighted_score'] = efficiency_score
        
        # Combined efficiency and precision score
        combined_efficiency = (
            (gene_scores.get('PRM1_efficiency', 0) + gene_scores.get('PRM1_precision', 0)) * 0.6 +
            (gene_scores.get('TNP1_efficiency', 0) + gene_scores.get('TNP1_precision', 0)) * 0.4
        ) * 500  # Scale appropriately
        gene_scores['combined_efficiency_score'] = combined_efficiency
        
        # Quality-adjusted score (efficiency * quality)
        quality_adjusted = (
            gene_scores.get('PRM1_efficiency', 0) * gene_scores.get('PRM1_quality', 0) * 0.6 +
            gene_scores.get('TNP1_efficiency', 0) * gene_scores.get('TNP1_quality', 0) * 0.4
        ) * 100
        gene_scores['quality_adjusted_score'] = quality_adjusted
        
        # NEW: Penalty for over-calling (discourage extremely liberal parameters)
        if total_cells > 50000:  # Penalty threshold
            overcall_penalty = min(0.5, (total_cells - 50000) / 100000)  # Up to 50% penalty
            gene_scores['overcall_penalty'] = overcall_penalty
        else:
            gene_scores['overcall_penalty'] = 0
        
        # Final normalized score (main optimization target)
        # Combines efficiency, precision, and quality while penalizing over-calling
        final_score = (
            efficiency_score * 0.4 +  # Efficiency weight
            combined_efficiency * 0.3 +  # Combined efficiency weight  
            quality_adjusted * 0.3  # Quality weight
        ) * (1 - gene_scores['overcall_penalty'])  # Apply penalty
        
        gene_scores['final_normalized_score'] = final_score
        
        # Calculate expression diversity (higher is better for capturing different cell types)
        if len(called_cells) > 10:
            try:
                # Use top variable genes for diversity calculation
                gene_expression = raw_adata[called_cells.index].X.toarray()
                if gene_expression.shape[1] > 100:
                    # Sample genes for computational efficiency
                    n_genes_sample = min(500, gene_expression.shape[1])
                    gene_indices = np.random.choice(gene_expression.shape[1], n_genes_sample, replace=False)
                    gene_expression = gene_expression[:, gene_indices]
                
                # Calculate coefficient of variation for expression diversity
                mean_expr = np.mean(gene_expression, axis=0)
                std_expr = np.std(gene_expression, axis=0)
                cv = np.mean(std_expr / (mean_expr + 1e-10))
                gene_scores['expression_diversity'] = float(cv)
            except:
                gene_scores['expression_diversity'] = 0.0
        else:
            gene_scores['expression_diversity'] = 0.0
        
        # Save detailed results if iteration is provided
        if iteration is not None:
            output_path = os.path.join(RESULTS_DIR, f'multi_gene_analysis_iter_{iteration}.csv')
            called_cells_analysis = called_cells.copy()
            
            for gene in target_genes:
                if gene in raw_adata.var_names:
                    gene_indices = called_cells.index
                    gene_counts = raw_adata[gene_indices, gene].X.toarray().flatten()
                    total_counts = raw_adata[gene_indices].X.sum(axis=1).A1
                    gene_percentages = np.divide(gene_counts, total_counts, 
                                               out=np.zeros_like(gene_counts, dtype=float), 
                                               where=total_counts!=0) * 100
                    called_cells_analysis[f'{gene}_percentage'] = gene_percentages
                    called_cells_analysis[f'{gene}_in_target'] = (gene_percentages >= 1.0) & (gene_percentages <= 6.0)
            
            called_cells_analysis.to_csv(output_path, index=True)
            
            logger.info(f"Iteration {iteration}: Called {total_cells} cells, "
                       f"PRM1: {gene_scores.get('PRM1', 0)} raw ({gene_scores.get('PRM1_efficiency', 0):.3f} eff), "
                       f"TNP1: {gene_scores.get('TNP1', 0)} raw ({gene_scores.get('TNP1_efficiency', 0):.3f} eff), "
                       f"Final score: {final_score:.2f}")

        return gene_scores
        
    except Exception as e:
        logger.error(f"Error calculating multi-gene score: {str(e)}")
        return {gene: 0.0 for gene in target_genes}

# Function to calculate the PRM1 score based on the EmptyDrops results (in KernelDensityPlot)
def calculate_prm1_score(emptydrops_results, raw_adata, fdr_threshold=0.01, iteration=None):
    """
    Calculate PRM1 score for optimization.
    
    Parameters:
    -----------
    emptydrops_results : pd.DataFrame
        Results from empty_drops function
    raw_adata : AnnData
        Raw data matrix
    fdr_threshold : float
        FDR threshold for calling cells
    iteration : int, optional
        Current iteration number for saving results
    
    Returns:
    --------
    float : PRM1 score (number of cells in 1-6% PRM1 range)
    """
    try:
        # Get cells called by EmptyDrops
        called_cells = emptydrops_results[emptydrops_results['FDR'] < fdr_threshold]
        called_cells = called_cells.sort_values(by='FDR', ascending=True)
    
        if len(called_cells) == 0:
            logger.warning(f"No cells called with FDR < {fdr_threshold}")
            return 0
        
        # Check if PRM1 gene exists
        if 'PRM1' not in raw_adata.var_names:
            logger.warning("PRM1 gene not found in data")
            # Return total number of called cells as fallback
            return len(called_cells)
        
        # Extract PRM1 expression for called cells
        prm1_indices = called_cells.index
        prm1_counts = raw_adata[prm1_indices, 'PRM1'].X.toarray().flatten()
        total_counts = raw_adata[prm1_indices].X.sum(axis=1).A1
    
        # Calculate PRM1 as percentage of total transcripts
        prm1_percentages = np.divide(prm1_counts, total_counts, 
                                   out=np.zeros_like(prm1_counts, dtype=float), 
                                   where=total_counts!=0) * 100
    
        # Count cells in 1-6% range
        target_cells = np.sum((prm1_percentages >= 1.0) & (prm1_percentages <= 6.0))

        # Save the called cells to a csv file if iteration is provided
        if iteration is not None:
            output_path = os.path.join(RESULTS_DIR, f'called_cells_iter_{iteration}.csv')
            called_cells_with_prm1 = called_cells.copy()
            called_cells_with_prm1['PRM1_percentage'] = prm1_percentages
            called_cells_with_prm1.to_csv(output_path, index=True)
            
            logger.info(f"Iteration {iteration}: Called {len(called_cells)} cells, "
                       f"{target_cells} in PRM1 1-6% range")

        return float(target_cells)  # Ensure return type is float
        
    except Exception as e:
        logger.error(f"Error calculating PRM1 score: {str(e)}")
        return 0.0

# Function to save called barcodes/cells to a csv file
def save_called_cells(emptydrops_results, output_path, fdr_threshold=0.01):
    """Save called cells to CSV file."""
    try:
        called_cells = emptydrops_results[emptydrops_results['FDR'] < fdr_threshold]
        called_cells.to_csv(output_path, index=True)
        logger.info(f"Saved {len(called_cells)} called cells to {output_path}")
    except Exception as e:
        logger.error(f"Error saving called cells: {str(e)}")

def analyze_prm1_distribution(called_cells_data, raw_adata, output_dir=None):
    """
    Analyze PRM1 distribution in called cells.
    
    Parameters:
    -----------
    called_cells_data : pd.DataFrame
        Called cells from EmptyDrops
    raw_adata : AnnData
        Raw data matrix
    output_dir : str, optional
        Directory to save plots
    
    Returns:
    --------
    dict : Analysis results including counts and plot data
    """
    try:
        if 'PRM1' not in raw_adata.var_names:
            logger.warning("PRM1 gene not found for distribution analysis")
            return {'n_target_range': 0, 'distribution_plot': None, 'kde_analysis': None}
        
        # Calculate PRM1 percentages
        prm1_indices = called_cells_data.index
        prm1_counts = raw_adata[prm1_indices, 'PRM1'].X.toarray().flatten()
        total_counts = raw_adata[prm1_indices].X.sum(axis=1).A1
        
        prm1_percentages = np.divide(prm1_counts, total_counts, 
                                   out=np.zeros_like(prm1_counts, dtype=float), 
                                   where=total_counts!=0) * 100
        
        # Count cells in target range
        target_range_mask = (prm1_percentages >= 1.0) & (prm1_percentages <= 6.0)
        n_target_range = np.sum(target_range_mask)
        
        # Create kernel density plot
        if output_dir:
            plt.figure(figsize=(12, 8))
            
            # Main distribution plot
            plt.subplot(2, 2, 1)
            plt.hist(prm1_percentages, bins=50, alpha=0.7, density=True, color='skyblue')
            plt.axvspan(1.0, 6.0, alpha=0.3, color='red', label=f'Target Range (n={n_target_range})')
            plt.xlabel('PRM1 Percentage')
            plt.ylabel('Density')
            plt.title('PRM1 Distribution in Called Cells')
            plt.legend()
            
            # Log scale plot
            plt.subplot(2, 2, 2)
            plt.hist(prm1_percentages[prm1_percentages > 0], bins=50, alpha=0.7, 
                    density=True, color='lightgreen')
            plt.axvspan(1.0, 6.0, alpha=0.3, color='red')
            plt.xscale('log')
            plt.xlabel('PRM1 Percentage (log)')
            plt.ylabel('Density')
            plt.title('PRM1 Distribution (Log Scale)')
            
            # Box plot
            plt.subplot(2, 2, 3)
            plt.boxplot([prm1_percentages[target_range_mask], 
                        prm1_percentages[~target_range_mask]], 
                       labels=['Target Range', 'Other'])
            plt.ylabel('PRM1 Percentage')
            plt.title('PRM1 Distribution by Range')
            
            # Summary statistics
            plt.subplot(2, 2, 4)
            stats_text = f"""
            Total Cells: {len(called_cells_data)}
            Target Range (1-6%): {n_target_range}
            Percentage in Target: {n_target_range/len(called_cells_data)*100:.2f}%
            
            PRM1 Statistics:
            Mean: {np.mean(prm1_percentages):.3f}%
            Median: {np.median(prm1_percentages):.3f}%
            Std: {np.std(prm1_percentages):.3f}%
            Max: {np.max(prm1_percentages):.3f}%
            """
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes, 
                    fontsize=10, verticalalignment='center')
            plt.axis('off')
            plt.title('Summary Statistics')
            
            plt.tight_layout()
            plot_path = os.path.join(output_dir, 'prm1_distribution_analysis.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"PRM1 distribution analysis saved to {plot_path}")
        
        return {
            'n_target_range': n_target_range,
            'total_cells': len(called_cells_data),
            'target_percentage': n_target_range/len(called_cells_data)*100 if len(called_cells_data) > 0 else 0,
            'prm1_stats': {
                'mean': float(np.mean(prm1_percentages)),
                'median': float(np.median(prm1_percentages)),
                'std': float(np.std(prm1_percentages)),
                'max': float(np.max(prm1_percentages))
            }
        }
        
    except Exception as e:
        logger.error(f"Error in PRM1 distribution analysis: {str(e)}")
        return {'n_target_range': 0, 'distribution_plot': None, 'kde_analysis': None}

# Enhanced EmptyDrops optimizer with advanced Optuna features
class EmptyDropsOptimizer:
    def __init__(self, raw_data, target_genes=['PRM1'], study_name=None, storage_url=None, 
                 suppress_output=True, multi_objective=True, convergence_patience=10):
        """
        Initialize the advanced EmptyDrops optimizer.
        
        Parameters:
        -----------
        raw_data : AnnData
            Raw single-cell data
        target_genes : list
            List of genes to optimize for (default: ['PRM1'])
        study_name : str, optional
            Name for the Optuna study
        storage_url : str, optional
            URL for persistent storage of study
        suppress_output : bool, optional
            Whether to suppress console output from empty_drops during optimization
        multi_objective : bool
            Whether to use multi-objective optimization
        convergence_patience : int
            Number of trials without improvement before early stopping
        """
        self.raw_data = raw_data
        self.target_genes = target_genes if isinstance(target_genes, list) else [target_genes]
        self.multi_objective = multi_objective
        self.convergence_patience = convergence_patience
        self.trial_counter = 0
        self.best_score = 0
        self.best_params = None
        self.suppress_output = suppress_output
        
        # Advanced tracking
        self.score_history = []
        self.convergence_counter = 0
        self.early_stopped = False
        
        # Progress bar for optimization when output is suppressed
        self.progress_bar = None
        self.optimization_start_time = None
        
        # Generate study name if not provided
        if study_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            obj_type = "multi" if multi_objective else "single"
            study_name = f"emptydrops_{obj_type}obj_optimization_{timestamp}"
        self.study_name = study_name
        
        # Set up storage
        if storage_url is None:
            storage_url = f"sqlite:///{os.path.join(STUDIES_DIR, self.study_name)}.db"
        self.storage_url = storage_url
        
        logger.info(f"Initialized EmptyDrops optimizer for study: {study_name}")
        logger.info(f"Data shape: {raw_data.shape}")
        logger.info(f"Target genes: {self.target_genes}")
        logger.info(f"Multi-objective: {multi_objective}")
        logger.info(f"Convergence patience: {convergence_patience}")
        
    def create_study(self, direction='maximize', sampler=None, pruner=None):
        """Create or load an Optuna study."""
        try:
            # Create study with advanced features
            if sampler is None:
                sampler = optuna.samplers.TPESampler(
                    n_startup_trials=10,
                    n_ei_candidates=24,
                    seed=42
                )
                
            if pruner is None:
                pruner = optuna.pruners.MedianPruner(
                    n_startup_trials=5,
                    n_warmup_steps=10
                )
            
            study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage_url,
                direction=direction,
                sampler=sampler,
                pruner=pruner,
                load_if_exists=True
            )
            
            logger.info(f"Created/loaded study with {len(study.trials)} existing trials")
            return study
            
        except Exception as e:
            logger.error(f"Error creating study: {str(e)}")
            # Fallback to in-memory study
            return optuna.create_study(direction=direction, sampler=sampler, pruner=pruner)
    
    def objective(self, trial):
        """
        Advanced objective function for Optuna optimization with multi-objective support.
        
        Parameters:
        -----------
        trial : optuna.Trial
            Trial object for parameter suggestion
            
        Returns:
        --------
        float or tuple : Objective value(s)
        """
        try:
            self.trial_counter += 1
            trial_start_time = time.time()
            
            # Improved parameter suggestions with better constraints
            params = {}
            
            # Lower threshold with adaptive range based on data
            total_umis = np.array(self.raw_data.X.sum(axis=1)).flatten()
            umi_percentiles = np.percentile(total_umis, [10, 50, 90])
            lower_min = max(50, int(umi_percentiles[0] * 0.1))
            lower_max = max(lower_min + 100, min(2000, int(umi_percentiles[1] * 0.5)))
            params['lower'] = trial.suggest_int('lower', lower_min, lower_max, step=25)
            
            # Retain parameter with adaptive suggestions
            retain_options = [None]
            if len(self.raw_data) > 5000:
                retain_options.extend([1000, 2000, 3000, 5000])
            else:
                retain_options.extend([500, 1000, 1500])
            params['retain'] = trial.suggest_categorical('retain', retain_options) 
            
            # Alpha with more granular control
            alpha_options = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, np.inf]
            params['alpha'] = trial.suggest_categorical('alpha', alpha_options)
            
            # Iterations based on convergence requirements
            niters_options = [100, 200, 500, 1000, 2000] if not self.suppress_output else [100, 200, 500]
            params['niters'] = trial.suggest_categorical('niters', niters_options)
            
            # FDR threshold with better resolution in important range
            params['fdr_threshold'] = trial.suggest_float('fdr_threshold', 0.0001, 0.2, log=True)
            
            # Additional adaptive parameters
            if hasattr(self.raw_data, 'n_obs') and self.raw_data.n_obs > 50000:
                params['batch_size'] = trial.suggest_categorical('batch_size', [100, 200, 500, 1000])
            
            # Update progress bar if output is suppressed
            if self.suppress_output and self.progress_bar is not None:
                elapsed = time.time() - self.optimization_start_time
                avg_time = elapsed / self.trial_counter if self.trial_counter > 0 else 0
                self.progress_bar.set_description(
                    f"Trial {self.trial_counter} | Best: {self.best_score:.1f} | "
                    f"Conv: {self.convergence_counter}/{self.convergence_patience} | "
                    f"Avg: {avg_time:.1f}s/trial"
                )
                self.progress_bar.update(1)
            
            logger.info(f"Trial {self.trial_counter}: Testing parameters {params}")
            
            # Run EmptyDrops with suggested parameters
            fdr_threshold = params.pop('fdr_threshold')  # Remove from params for empty_drops
            
            # Run EmptyDrops with optional output suppression
            if self.suppress_output:
                with suppress_stdout_stderr():
                    results = empty_drops(
                        self.raw_data, 
                        progress=False,
                        visualize=False,
                        fast_mode=True,
                        **params
                    )
            else:
                results = empty_drops(
                    self.raw_data, 
                    progress=False,
                    visualize=False,
                    fast_mode=True,
                    **params
                )
            
            trial_time = time.time() - trial_start_time
            
            # Calculate scores based on optimization mode
            if self.multi_objective and len(self.target_genes) > 1:
                # Multi-objective optimization with normalized scoring
                gene_scores = calculate_multi_gene_score(
                    results, 
                    self.raw_data, 
                    target_genes=self.target_genes,
                    fdr_threshold=fdr_threshold,
                    iteration=self.trial_counter
                )
                
                # Primary objective: normalized score that prevents parameter inflation
                primary_score = gene_scores.get('final_normalized_score', 0)
                
                # Secondary objectives (for analysis and user attributes)
                secondary_scores = {
                    'expression_diversity': gene_scores.get('expression_diversity', 0),
                    'total_called_cells': gene_scores.get('total_called_cells', 0),
                    'efficiency_weighted': gene_scores.get('efficiency_weighted_score', 0),
                    'overcall_penalty': gene_scores.get('overcall_penalty', 0)
                }
                
                # Store individual gene scores and metrics as user attributes
                for gene in self.target_genes:
                    trial.set_user_attr(f'{gene}_target_cells', gene_scores.get(gene, 0))
                    trial.set_user_attr(f'{gene}_efficiency', gene_scores.get(f'{gene}_efficiency', 0))
                    trial.set_user_attr(f'{gene}_precision', gene_scores.get(f'{gene}_precision', 0))
                    trial.set_user_attr(f'{gene}_quality', gene_scores.get(f'{gene}_quality', 0))
                
                # Store composite scores
                trial.set_user_attr('raw_weighted_score', gene_scores.get('weighted_score', 0))
                trial.set_user_attr('efficiency_score', gene_scores.get('efficiency_weighted_score', 0))
                trial.set_user_attr('quality_adjusted', gene_scores.get('quality_adjusted_score', 0))
                trial.set_user_attr('overcall_penalty', gene_scores.get('overcall_penalty', 0))
                
                # NEW: Add parameter relationship metrics for better dashboard analysis
                lower_val = params.get('lower', 0)
                retain_val = params.get('retain', None)
                
                if retain_val is not None:
                    # Parameter span (testing range)
                    param_span = retain_val - lower_val
                    trial.set_user_attr('parameter_span', param_span)
                    
                    # Relative span (span as ratio of lower threshold)
                    relative_span = param_span / lower_val if lower_val > 0 else 0
                    trial.set_user_attr('relative_span', relative_span)
                    
                    # Testing efficiency (target cells per unit span)
                    span_efficiency = gene_scores.get('combined_score', 0) / param_span if param_span > 0 else 0
                    trial.set_user_attr('span_efficiency', span_efficiency)
                else:
                    trial.set_user_attr('parameter_span', float('inf'))  # No upper limit
                    trial.set_user_attr('relative_span', float('inf'))
                    trial.set_user_attr('span_efficiency', 0)
                
                # Parameter category for easier analysis
                if retain_val is None:
                    param_category = f"lower_{lower_val}_no_retain"
                else:
                    param_category = f"lower_{lower_val}_retain_{retain_val}"
                trial.set_user_attr('parameter_category', param_category)
                
                score = primary_score
                
            else:
                # Single-objective optimization (backward compatibility)
                primary_gene = self.target_genes[0] if self.target_genes else 'PRM1'
                if primary_gene == 'PRM1':
                    raw_score = calculate_prm1_score(
                        results, 
                        self.raw_data, 
                        fdr_threshold=fdr_threshold,
                        iteration=self.trial_counter
                    )
                    # For single objective, apply simple normalization
                    total_called = (results['FDR'] < fdr_threshold).sum()
                    efficiency = raw_score / total_called if total_called > 0 else 0
                    # Apply overcall penalty for single objective too
                    overcall_penalty = min(0.3, max(0, (total_called - 30000) / 100000))
                    score = (raw_score + efficiency * 1000) * (1 - overcall_penalty)
                    
                    # Store metrics
                    trial.set_user_attr('raw_score', raw_score)
                    trial.set_user_attr('efficiency', efficiency)
                    trial.set_user_attr('overcall_penalty', overcall_penalty)
                else:
                    gene_scores = calculate_multi_gene_score(
                        results, 
                        self.raw_data, 
                        target_genes=[primary_gene],
                        fdr_threshold=fdr_threshold,
                        iteration=self.trial_counter
                    )
                    score = gene_scores.get('final_normalized_score', 0)
            
            # Report intermediate values for pruning
            trial.report(score, step=0)
            
            # Check for early stopping (convergence)
            self.score_history.append(score)
            if len(self.score_history) >= self.convergence_patience:
                recent_scores = self.score_history[-self.convergence_patience:]
                if self._check_convergence(recent_scores):
                    self.convergence_counter += 1
                    if self.convergence_counter >= self.convergence_patience:
                        self.early_stopped = True
                        trial.study.stop()
                        logger.info(f"Early stopping triggered after {self.trial_counter} trials")
                else:
                    self.convergence_counter = 0
            
            # Log trial results
            called_cells = (results['FDR'] < fdr_threshold).sum()
            logger.info(f"Trial {self.trial_counter}: Score={score:.2f}, "
                       f"Called cells={called_cells}, Time={trial_time:.2f}s")
            
            # Update best score tracking
            if score > self.best_score:
                self.best_score = score
                self.best_params = {**params, 'fdr_threshold': fdr_threshold}
                self.convergence_counter = 0  # Reset convergence counter on improvement
                logger.info(f"New best score: {score:.2f}")
            
            # Store comprehensive trial information
            trial.set_user_attr('called_cells', int(called_cells))
            trial.set_user_attr('trial_time', trial_time)
            trial.set_user_attr('timestamp', datetime.now().isoformat())
            trial.set_user_attr('fdr_threshold_used', fdr_threshold)
            
            return score
            
        except Exception as e:
            logger.error(f"Error in trial {self.trial_counter}: {str(e)}")
            return 0.0  # Return worst possible score for failed trials
    
    def _check_convergence(self, recent_scores, threshold=0.01):
        """Check if optimization has converged based on recent scores."""
        if len(recent_scores) < 3:
            return False
        
        # Check if improvement is less than threshold
        max_score = max(recent_scores)
        min_score = min(recent_scores)
        relative_improvement = (max_score - min_score) / (max_score + 1e-10)
        
        return relative_improvement < threshold
    
    def optimize(self, n_trials=100, timeout=None, n_jobs=1, callbacks=None):
        """
        Run optimization with advanced features.
        
        Parameters:
        -----------
        n_trials : int
            Number of trials to run
        timeout : int, optional
            Timeout in seconds
        n_jobs : int
            Number of parallel jobs
        callbacks : list, optional
            List of callback functions
            
        Returns:
        --------
        optuna.Study : Completed study object
        """
        logger.info(f"Starting optimization with {n_trials} trials")
        self.optimization_start_time = time.time()
        
        # Initialize progress bar if output is suppressed
        if self.suppress_output:
            self.progress_bar = tqdm(
                total=n_trials,
                desc="EmptyDrops Optimization",
                unit="trial",
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {desc}',
                position=0,
                leave=True
            )
            print("🚀 Starting EmptyDrops hyperparameter optimization with fast mode enabled...")
            print("📊 Console output from EmptyDrops is suppressed - progress shown below:")
        
        # Create study
        study = self.create_study()
        
        # Add default callbacks
        if callbacks is None:
            callbacks = []
        
        # Add logging callback
        def logging_callback(study, trial):
            if trial.number % 10 == 0:
                logger.info(f"Completed {trial.number} trials, "
                           f"best value: {study.best_value:.2f}")
        
        callbacks.append(logging_callback)
        
        try:
            # Run optimization
            study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                n_jobs=n_jobs,
                callbacks=callbacks,
                show_progress_bar=not self.suppress_output  # Use our progress bar when output is suppressed
            )
            
            total_time = time.time() - self.optimization_start_time
            
            # Close progress bar if it was used
            if self.suppress_output and self.progress_bar is not None:
                self.progress_bar.close()
                print("\n✅ Optimization completed!")
                print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
                print(f"🏆 Best score: {study.best_value:.2f}")
                print(f"📋 Best parameters: {study.best_params}")
                print(f"💾 Results saved to: {RESULTS_DIR}")
            
            logger.info(f"Optimization completed in {total_time:.2f} seconds")
            logger.info(f"Best parameters: {study.best_params}")
            logger.info(f"Best value: {study.best_value}")
            
            # Save study results
            self.save_study_results(study)
            
            # Generate visualizations
            self.create_optimization_plots(study)
            
            return study
            
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            # Close progress bar on error
            if self.suppress_output and self.progress_bar is not None:
                self.progress_bar.close()
            return study
    
    def save_study_results(self, study):
        """Save study results to files."""
        try:
            # Save study summary
            results_file = os.path.join(RESULTS_DIR, f"{self.study_name}_results.txt")
            with open(results_file, 'w') as f:
                f.write(f"EmptyDrops Optimization Results\n")
                f.write(f"Study Name: {self.study_name}\n")
                f.write(f"Number of Trials: {len(study.trials)}\n")
                f.write(f"Best Value: {study.best_value}\n")
                f.write(f"Best Parameters:\n")
                for param, value in study.best_params.items():
                    f.write(f"  {param}: {value}\n")
                f.write(f"\nCompleted Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
                f.write(f"Failed Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
                f.write(f"Pruned Trials: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n")
            
            # Save trials dataframe
            trials_df = study.trials_dataframe()
            trials_file = os.path.join(RESULTS_DIR, f"{self.study_name}_trials.csv")
            trials_df.to_csv(trials_file, index=False)
            
            # Save study object
            study_file = os.path.join(RESULTS_DIR, f"{self.study_name}_study.pkl")
            with open(study_file, 'wb') as f:
                pickle.dump(study, f)
            
            logger.info(f"Study results saved to {RESULTS_DIR}")
            
        except Exception as e:
            logger.error(f"Error saving study results: {str(e)}")
    
    def create_optimization_plots(self, study):
        """Create comprehensive optimization plots."""
        try:
            logger.info("Creating optimization plots...")
            
            # Plot 1: Optimization History
            fig_history = optuna.visualization.plot_optimization_history(study)
            fig_history.write_html(os.path.join(PLOTS_DIR, f"{self.study_name}_history.html"))
            
            # Plot 2: Parameter Importances
            try:
                fig_importance = optuna.visualization.plot_param_importances(study)
                fig_importance.write_html(os.path.join(PLOTS_DIR, f"{self.study_name}_importance.html"))
            except:
                logger.warning("Could not create parameter importance plot")
            
            # Plot 3: Parallel Coordinate Plot with objective value
            # Include objective value as a parameter to see performance clearly
            fig_parallel = optuna.visualization.plot_parallel_coordinate(
                study, 
                params=['lower', 'alpha', 'niters', 'fdr_threshold', 'retain'],
                target=lambda t: t.value,  # Add objective value column
                target_name="Objective"
            )
            fig_parallel.write_html(os.path.join(PLOTS_DIR, f"{self.study_name}_parallel.html"))
            
            # Plot 4: Slice Plot
            fig_slice = optuna.visualization.plot_slice(study)
            fig_slice.write_html(os.path.join(PLOTS_DIR, f"{self.study_name}_slice.html"))
            
            # Plot 5: Contour Plot (for parameter pairs)
            try:
                fig_contour = optuna.visualization.plot_contour(study)
                fig_contour.write_html(os.path.join(PLOTS_DIR, f"{self.study_name}_contour.html"))
            except:
                logger.warning("Could not create contour plot")
            
            # Custom plots using matplotlib
            self.create_custom_plots(study)
            
            logger.info(f"Optimization plots saved to {PLOTS_DIR}")
            
            # Suggest dashboard usage
            print(f"\n🌐 TIP: Launch Optuna Dashboard for real-time monitoring:")
            print(f"   python3 launch_optuna_dashboard.py")
            print(f"   This provides interactive visualization and study comparison!")
            
        except Exception as e:
            logger.error(f"Error creating optimization plots: {str(e)}")
    
    def create_custom_plots(self, study):
        """Create custom matplotlib plots."""
        try:
            trials_df = study.trials_dataframe()
            
            # Filter completed trials
            completed = trials_df[trials_df['state'] == 'COMPLETE']
            
            if len(completed) == 0:
                logger.warning("No completed trials for plotting")
                return
            
            # Plot 1: Parameter vs Objective scatter plots
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Parameter vs Objective Analysis - {self.study_name}', fontsize=16)
            
            params = ['params_lower', 'params_alpha', 'params_niters', 'params_fdr_threshold']
            
            for i, param in enumerate(params):
                if param in completed.columns:
                    row, col = i // 3, i % 3
                    
                    # Handle categorical parameters
                    if param == 'params_alpha':
                        # Convert inf to a large number for plotting
                        alpha_values = completed[param].replace([np.inf], [1000])
                        axes[row, col].scatter(alpha_values, completed['value'], alpha=0.6)
                        axes[row, col].set_xscale('log')
                    else:
                        axes[row, col].scatter(completed[param], completed['value'], alpha=0.6)
                    
                    axes[row, col].set_xlabel(param.replace('params_', ''))
                    axes[row, col].set_ylabel('Objective Value')
                    axes[row, col].set_title(f'{param.replace("params_", "")} vs Objective')
            
            # Plot trial progression
            axes[1, 2].plot(completed['number'], completed['value'])
            axes[1, 2].set_xlabel('Trial Number')
            axes[1, 2].set_ylabel('Objective Value')
            axes[1, 2].set_title('Trial Progression')
            
            # Add best value line
            best_value = completed['value'].max()
            axes[1, 2].axhline(y=best_value, color='r', linestyle='--', 
                              label=f'Best: {best_value:.2f}')
            axes[1, 2].legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{self.study_name}_custom_analysis.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Plot 2: Distribution of parameters
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'Parameter Distributions - {self.study_name}', fontsize=16)
            
            # Lower threshold distribution
            axes[0, 0].hist(completed['params_lower'], bins=20, alpha=0.7, color='skyblue')
            axes[0, 0].set_xlabel('Lower Threshold')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].set_title('Distribution of Lower Threshold')
            
            # FDR threshold distribution (log scale)
            axes[0, 1].hist(completed['params_fdr_threshold'], bins=20, alpha=0.7, color='lightgreen')
            axes[0, 1].set_xlabel('FDR Threshold')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].set_title('Distribution of FDR Threshold')
            axes[0, 1].set_xscale('log')
            
            # Number of iterations distribution
            if 'params_niters' in completed.columns:
                axes[1, 0].hist(completed['params_niters'], bins=10, alpha=0.7, color='orange')
                axes[1, 0].set_xlabel('Number of Iterations')
                axes[1, 0].set_ylabel('Frequency')
                axes[1, 0].set_title('Distribution of Iterations')
            
            # Objective value distribution
            axes[1, 1].hist(completed['value'], bins=20, alpha=0.7, color='coral')
            axes[1, 1].set_xlabel('Objective Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Distribution of Objective Values')
            
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_DIR, f"{self.study_name}_distributions.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating custom plots: {str(e)}")

# Function to let the user insert the parameters
def user_input_parameters():
    print("Insert the parameters in the following format:")
    print("1. fixed format: [1, 20, 35, 40, 50, 67, 100, 123]")
    print("2. compact format: [start, end, step]")
    print("3. mixed format: 100,[start, end, step], 150")
    print("4. chained format: 100,[start, end, step],[start, end, step] 150, 200")
    print("5. special value: inf, none")

    # Parameter definitions
    param_configs = {}
    parameters = {
        'lower': 'Lower threshold for empty droplets',
        'retain': 'Retain threshold (use "none" for None)',
        'alpha': 'Alpha parameter (use "inf" for infinity)',
        'niters': 'Number of Monte Carlo iterations',
        'fdr_threshold': 'FDR threshold for analysis'
    }

    for param_name, description in parameters.items():
        while True:
            try:
                user_input = input(f"\n{param_name} ({description}): ").strip()
                parsed_values = parse_parameter_input(user_input)
                param_configs[param_name] = parsed_values
                print(f"✓ Parsed {param_name}: {parsed_values}")
                break
            except Exception as e:
                print(f"❌ Error parsing {param_name}: {e}")
                print("Please try again with correct format.")
    
    return param_configs

# Function to parse the user input for the paramaters
def parse_parameter_input(input_string):
    """
    Parse user input string into list of values.
    Handles: fixed values, ranges, mixed formats, special values.
    """
    if not input_string:
        raise ValueError("Empty input")
    
    # Remove extra whitespace
    input_string = input_string.strip()
    
    # Split by commas, but respect brackets
    parts = split_respecting_brackets(input_string)
    
    values = []
    for part in parts:
        part = part.strip()
        
        # Handle special values
        if part.lower() == 'none':
            values.append(None)
        elif part.lower() == 'inf':
            values.append(np.inf)
        elif part.startswith('[') and part.endswith(']'):
            # Handle range format [start,end,step]
            try:
                range_values = ast.literal_eval(part)
                if len(range_values) != 3:
                    raise ValueError("Range must have exactly 3 values: [start,end,step]")
                start, end, step = range_values
                expanded = list(np.arange(start, end + step/2, step))
                values.extend(expanded)
            except Exception as e:
                raise ValueError(f"Invalid range format '{part}': {e}")
        else:
            # Handle regular numeric values
            try:
                if '.' in part:
                    values.append(float(part))
                else:
                    values.append(int(part))
            except ValueError:
                raise ValueError(f"Cannot parse numeric value: '{part}'")
    
    if not values:
        raise ValueError("No valid values found")
    
    return values

# Function to split the user input by commas but respect bracket groups
def split_respecting_brackets(input_string):
    """
    Split string by commas but respect bracket groups.
    Example: "50,100,[150,300,25],500" → ["50", "100", "[150,300,25]", "500"]
    """
    parts = []
    current_part = ""
    bracket_depth = 0
    
    for char in input_string:
        if char == '[':
            bracket_depth += 1
            current_part += char
        elif char == ']':
            bracket_depth -= 1
            current_part += char
        elif char == ',' and bracket_depth == 0:
            if current_part.strip():
                parts.append(current_part.strip())
            current_part = ""
        else:
            current_part += char
    
    # Add the last part
    if current_part.strip():
        parts.append(current_part.strip())
    
    return parts

def run_optimization_with_visualization(raw_adata, n_trials=50, study_name=None, 
                                      suppress_output=True):
    """
    Run complete optimization with visualization and analysis.
    
    Parameters:
    -----------
    raw_adata : AnnData
        Raw single-cell data
    n_trials : int
        Number of optimization trials
    study_name : str, optional
        Name for the study
    suppress_output : bool, optional
        Whether to suppress console output from empty_drops during optimization
        
    Returns:
    --------
    dict : Results including best parameters and study object
    """
    logger.info("Starting complete optimization workflow")
    
    # Initialize enhanced optimizer
    optimizer = EmptyDropsOptimizer(
        raw_adata, 
        target_genes=['PRM1', 'TNP1'], 
        study_name=study_name,
        suppress_output=suppress_output,
        multi_objective=True,
        convergence_patience=10
    )
    
    # Run optimization
    study = optimizer.optimize(n_trials=n_trials)
    
    # Get best parameters
    best_params = study.best_params.copy()
    
    # Run final analysis with best parameters
    logger.info("Running final analysis with best parameters...")
    fdr_threshold = 0.01  # Default FDR threshold for final analysis
    if 'fdr_threshold' in best_params:
        fdr_threshold = best_params.pop('fdr_threshold')
    
    final_results = empty_drops(
        raw_adata, 
        progress=True,
        visualize=True,
        **best_params
    )
    
    # Analyze results
    called_cells = final_results[final_results['FDR'] < fdr_threshold]
    prm1_analysis = analyze_prm1_distribution(called_cells, raw_adata, output_dir=RESULTS_DIR)
    
    # Save final results
    save_called_cells(final_results, 
                     os.path.join(RESULTS_DIR, f"{optimizer.study_name}_final_called_cells.csv"),
                     fdr_threshold=fdr_threshold)
    
    # Create summary report
    create_optimization_summary(study, final_results, prm1_analysis, 
                               fdr_threshold, optimizer.study_name)
    
    results = {
        'study': study,
        'best_params': {**best_params, 'fdr_threshold': fdr_threshold},
        'best_score': study.best_value,
        'final_results': final_results,
        'called_cells': called_cells,
        'prm1_analysis': prm1_analysis,
        'study_name': optimizer.study_name
    }
    
    logger.info("Optimization workflow completed successfully!")
    return results

def create_optimization_summary(study, final_results, prm1_analysis, fdr_threshold, study_name):
    """Create a comprehensive optimization summary report."""
    try:
        summary_file = os.path.join(RESULTS_DIR, f"{study_name}_optimization_summary.md")
        
        with open(summary_file, 'w') as f:
            f.write(f"# EmptyDrops Optimization Summary\n\n")
            f.write(f"**Study Name:** {study_name}\n")
            f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"## Optimization Results\n\n")
            f.write(f"- **Number of Trials:** {len(study.trials)}\n")
            f.write(f"- **Best Objective Value:** {study.best_value:.2f}\n")
            f.write(f"- **Completed Trials:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])}\n")
            f.write(f"- **Failed Trials:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])}\n")
            f.write(f"- **Pruned Trials:** {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}\n\n")
            
            f.write(f"## Best Parameters\n\n")
            for param, value in study.best_params.items():
                f.write(f"- **{param}:** {value}\n")
            f.write(f"- **fdr_threshold:** {fdr_threshold}\n\n")
            
            f.write(f"## Final Analysis Results\n\n")
            total_called = (final_results['FDR'] < fdr_threshold).sum()
            f.write(f"- **Total Cells Called:** {total_called}\n")
            f.write(f"- **Cells in PRM1 Target Range (1-6%):** {prm1_analysis['n_target_range']}\n")
            f.write(f"- **Percentage in Target Range:** {prm1_analysis['target_percentage']:.2f}%\n\n")
            
            if 'prm1_stats' in prm1_analysis:
                f.write(f"## PRM1 Statistics\n\n")
                stats = prm1_analysis['prm1_stats']
                f.write(f"- **Mean PRM1 Percentage:** {stats['mean']:.3f}%\n")
                f.write(f"- **Median PRM1 Percentage:** {stats['median']:.3f}%\n")
                f.write(f"- **Standard Deviation:** {stats['std']:.3f}%\n")
                f.write(f"- **Maximum PRM1 Percentage:** {stats['max']:.3f}%\n\n")
            
            f.write(f"## Files Generated\n\n")
            f.write(f"- Optimization plots: `{PLOTS_DIR}/`\n")
            f.write(f"- Final results: `{RESULTS_DIR}/{study_name}_final_called_cells.csv`\n")
            f.write(f"- Trial data: `{RESULTS_DIR}/{study_name}_trials.csv`\n")
            f.write(f"- Study object: `{RESULTS_DIR}/{study_name}_study.pkl`\n")
            f.write(f"- PRM1 analysis: `{RESULTS_DIR}/prm1_distribution_analysis.png`\n")
        
        logger.info(f"Optimization summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error creating optimization summary: {str(e)}")

# ------------------------------------------------------------------------------------------------
# MAIN FUNCTION to run the optimization
# ------------------------------------------------------------------------------------------------
def main():
    """Main function to run EmptyDrops optimization."""
    try:
        print("=== EmptyDrops Hyperparameter Optimization ===\n")
        
        # Configuration options
        print("Configuration options:")
        print("1. Quick test (10 trials)")
        print("2. Standard optimization (50 trials)")
        print("3. Extensive optimization (100 trials)")
        print("4. Full-scale optimization (30 trials, recommended for full dataset)")
        print("5. Custom configuration")
        
        choice = input("\nSelect option (1-5): ").strip()
        
        if choice == '1':
            n_trials = 10
        elif choice == '2':
            n_trials = 50
        elif choice == '3':
            n_trials = 100
        elif choice == '4':
            n_trials = 30
            print("🎯 Full-scale optimization selected:")
            print("   • 30 trials (estimated 20-30 hours)")
            print("   • Optimized for 720k+ cells")
            print("   • Can be stopped/resumed anytime with Ctrl+C")
        elif choice == '5':
            try:
                n_trials = int(input("Number of trials: "))
            except ValueError:
                print("Invalid input, using default 50 trials")
                n_trials = 50
        else:
            print("Invalid choice, using standard optimization")
            n_trials = 50
        
        # Optional study name
        study_name = input("Study name (press Enter for auto-generated): ").strip()
        if not study_name:
            study_name = None
        
        # Output suppression option
        suppress_choice = input("Suppress EmptyDrops console output during optimization? (Y/n): ").strip().lower()
        suppress_output = suppress_choice != 'n'
        
        # Load data
        print(f"\nLoading data...")
        try:
            raw_adata = sc.read_10x_h5("data/raw_feature_bc_matrix.h5")
            logger.info(f"Loaded data with shape: {raw_adata.shape}")
        except FileNotFoundError:
            print("Error: Could not find 'data/raw_feature_bc_matrix.h5'")
            print("Please ensure the data file exists in the correct location.")
            return
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return
        
        # Run optimization
        print(f"\nStarting optimization with {n_trials} trials...")
        if suppress_output:
            print("EmptyDrops console output will be suppressed during optimization.")
            print("Only Optuna optimization messages will be shown.")
        else:
            print("EmptyDrops console output will be visible during optimization.")
        print(f"Results will be saved to: {OPTIMIZATION_DIR}")
        
        results = run_optimization_with_visualization(
            raw_adata, 
            n_trials=n_trials, 
            study_name=study_name,
            suppress_output=suppress_output
        )
        
        # Print summary
        print(f"\n=== Optimization Complete ===")
        print(f"Study name: {results['study_name']}")
        print(f"Best score: {results['best_score']:.2f}")
        print(f"Best parameters:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        print(f"\nFinal results:")
        print(f"  Total cells called: {len(results['called_cells'])}")
        print(f"  Cells in PRM1 target range: {results['prm1_analysis']['n_target_range']}")
        print(f"  Target percentage: {results['prm1_analysis']['target_percentage']:.2f}%")
        
        print(f"\nResults saved to: {OPTIMIZATION_DIR}")
        print(f"View plots in: {PLOTS_DIR}")
        
    except KeyboardInterrupt:
        print("\nOptimization interrupted by user")
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()