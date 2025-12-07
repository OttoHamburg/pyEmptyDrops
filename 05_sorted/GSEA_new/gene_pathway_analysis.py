#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import scanpy as sc
import gseapy as gp
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import scipy.stats as stats
from matplotlib.patches import Rectangle
from datetime import datetime
import json
import time
from tqdm import tqdm

# Configuration
warnings.filterwarnings('ignore')
sc.settings.verbosity = 1
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
plt.rcParams['figure.figsize'] = (12, 8)

class SpermatogenesisAnalyzer:
    """
    Advanced analyzer for spermatogenesis single-cell data with focus on 
    creating biologically meaningful continuous trajectories.
    """
    
    def __init__(self, 
                data_path="data/emptydrops_all_detected_cells.h5ad", 
                output_dir=None, 
                CREDcomparison=False, 
                activate_advanced_filtering=False, 
                activate_hvg=True):
        self.data_path = data_path
        self.adata = None
        self.spermatogenesis_markers = None
        self.trajectory_markers = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.CREDcomparison = CREDcomparison
        self.activate_advanced_filtering = activate_advanced_filtering
        self.activate_hvg = activate_hvg
        # Create timestamped output directory
        if output_dir is None:
            self.output_dir = f"GSEA_{self.timestamp}"
        else:
            self.output_dir = f"{output_dir}/GSEA_{self.timestamp}"
            
        # Create the output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory created: {self.output_dir}")
        
        # Initialize parameters tracking
        self.analysis_parameters = {
            'timestamp': self.timestamp,
            'data_path': data_path,
            'output_dir': self.output_dir
        }
        
        self.setup_spermatogenesis_markers()
        
    def save_parameters(self):
        """Save analysis parameters to a text file."""
        param_file = os.path.join(self.output_dir, "parameters.txt")
        
        with open(param_file, 'w') as f:
            f.write("SPERMATOGENESIS ANALYSIS PARAMETERS\n")
            f.write("="*50 + "\n\n")
            
            f.write(f"Analysis Timestamp: {self.analysis_parameters['timestamp']}\n")
            f.write(f"Data Path: {self.analysis_parameters['data_path']}\n")
            f.write(f"Output Directory: {self.analysis_parameters['output_dir']}\n\n")
            
            if hasattr(self, 'adata') and self.adata is not None:
                f.write(f"Dataset Shape: {self.adata.shape[0]} cells × {self.adata.shape[1]} features\n")
                
            for key, value in self.analysis_parameters.items():
                if key not in ['timestamp', 'data_path', 'output_dir']:
                    f.write(f"{key}: {value}\n")
                    
            f.write("\nSpermatogenesis Markers Used:\n")
            f.write("-" * 30 + "\n")
            for cell_type, markers in self.spermatogenesis_markers.items():
                f.write(f"{cell_type}: {', '.join(markers)}\n")
                
        print(f"Parameters saved to: {param_file}")
        
    def setup_spermatogenesis_markers(self):
        """Define comprehensive spermatogenesis markers for pig."""
        
        # stage-specific markers based on spermatogenesis biology
        self.spermatogenesis_markers = {
            #Spermatogonien (mitotische vorläuferzellen & SSC)
            'spermatogonia': [
                'ID4', 'GFRA1', 'ZBTB16', 'NANOS2', 'UTF1', 'SALL4', 'DMRT1', 'SOHLH1', 'SOHLH2'
            ],
            # spermatocyten
            'spermatocytes': [
                'STRA8', 'DMRT6', 'SYCP1', 'SYCP2', 'SYCP3', 'HORMAD1', 'HORMAD2', 'TEX101', 'MLH1'
            ],
            # spermatiden (rund)
            'round_spermatids': [
                'TNP1', 'TNP2', 'ACRV1', 'CRISP2', 'GSTM3', 'SPATA16'
            ],
            # spermatiden (elongating/condensed)
            'elongated_spermatids': [
                'PRM1', 'PRM2', 'TNP1', 'TNP2', 'GSTM3', 'CRISP2', 'OAZ3', 'SPATA16'
            ],
            # leydig zellen
            'leydig_cells': [
                'CYP17A1', 'CYP11A1', 'HSD3B1', 'HSD17B3', 'STAR', 'INSL3', 'NR5A1', 'PDGFRA', 'NR2F2'
            ],
            # sertoli zellen
            'sertoli_cells': [
                'SOX9', 'AMH', 'GATA4', 'WT1', 'INHA', 'CLDN11', 'VIM', 'TF', 'RBP1', 'FSHR'
            ],
            # peritubular zellen
            'peritubular_cells': [
                'ACTA2', 'MYH11', 'CNN1', 'TAGLN', 'PDGFRB', 'DES', 'CSPG4'
            ]
            # 'Early_Sgonia': [
            #     "GFRA1", "ID4", "SOHLH1", "SOHLH2", "POU5F1", "UCHL1", "ITGB1", "ITGA6", "NGN3", "SOX3", "RET", "LIN28A", "ZBTB16", "NOS2", "EPCAM", "BMI1"
            # ],
            # 'Late_Sgonia': [
            #     "DAZL", "RAD51", "STRA8", "HIST1H1A", "PCNA", "TEX11", "NANOS3", "E2F1", "HIST1H3A", "WSB2", "ESX1", "USP26", "SALL4", "NLRP4C", "CCNA2", "CCNE2", "CCND1", "CRABP1", "ESRP1", "ADGRA3", "CDKN1C", "PER1", "LIN28A", "FIGLA", "UTF1", "HIST1H4M", "FTHL17"
            # ],
            # 'Early_Scytes': [
            #     "DMC1", "MEIOB", "GPAT2"
            # ],
            # 'Late_Scytes': [
            #     "KDM3A", "POU5F2"
            # ],
            # 'Round_Stids': [
            #     "SLX", "HILS1", "KLF17", "CRISP2", "AKAP4"
            # ],
            # 'Later_Stids': [
            #     "SUFU", "DLL1", "1700024P04RIK", "PRM1", "DYRK4", "TNP1", "HSPA1L"
            # ],
            # 'Sertoli': [
            #     "ID3", "SOX8", "RHOX5", "TFCP2", "WT1", "GATA1", "KITL"
            # ],
            # 'Leydig': [
            #     "CYP17A1", "HSD17B11", "CSF1", "VCAM1", "HSD3B6"
            # ]
        }
        
        # Create trajectory-ordered markers (early to late)
        self.trajectory_markers = (
            # self.spermatogenesis_markers['Early_Sgonia'] +
            # self.spermatogenesis_markers['Late_Sgonia'] +
            # self.spermatogenesis_markers['Early_Scytes'] +
            # self.spermatogenesis_markers['Late_Scytes'] +
            # self.spermatogenesis_markers['Round_Stids'] +
            # self.spermatogenesis_markers['Later_Stids']
            self.spermatogenesis_markers['spermatogonia'] +
            self.spermatogenesis_markers['spermatocytes'] +
            self.spermatogenesis_markers['round_spermatids'] +
            self.spermatogenesis_markers['elongated_spermatids']
        )
        
        # Remove duplicates while preserving order
        seen = set()
        self.trajectory_markers = [x for x in self.trajectory_markers 
                                 if not (x in seen or seen.add(x))]
        
        print(f"Defined {len(self.trajectory_markers)} trajectory markers of germ cells (5 categories)")
        print(f"Cell type markers: {sum(len(v) for v in self.spermatogenesis_markers.values())} total including leydig/sertoli etc")
    
    def load_and_preprocess_data(self):
        """Load and preprocess the EmptyDrops data with spermatogenesis-specific steps."""
        
        #print(f"Loading data from: {self.data_path}")
        
        # Load data
        if self.data_path.endswith('.h5ad'):
            self.adata = sc.read_h5ad(self.data_path)
        else:
            self.adata = sc.read_10x_h5(self.data_path)
            self.adata.var_names_make_unique()
        
        print(f"Original data shape: {self.adata.shape}")
        
        # Basic QC metrics
        self.adata.var['mt'] = self.adata.var_names.str.startswith(('MT-', 'Mt-', 'mt-'))
        sc.pp.calculate_qc_metrics(self.adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
        
        # Add UMI counts for compatibility
        if 'total_counts' not in self.adata.obs.columns:
            self.adata.obs['total_counts'] = np.array(self.adata.X.sum(axis=1)).flatten()
        
        # Store raw counts
        self.adata.layers['counts'] = self.adata.X.copy()
        
        print(f"Data loaded and QC metrics calculated")
        return self.adata
    
    def advanced_filtering(self, 
                          min_genes=100, 
                          max_genes=6000, 
                          min_counts=500,
                          max_counts=50000,
                          max_mt_percent=20, 
                          activate=False):
        """Advanced filtering tailored for spermatogenesis data."""
        
        # Track filtering parameters
        self.analysis_parameters.update({
            'filtering_min_genes': min_genes,
            'filtering_max_genes': max_genes,
            'filtering_min_counts': min_counts,
            'filtering_max_counts': max_counts,
            'filtering_max_mt_percent': max_mt_percent,
            'filtering_activated': activate
        })
        
        if activate:
            print(f"\nADVANCED FILTERING")
            print(f"Original: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
            
            # Cell filtering
            sc.pp.filter_cells(self.adata, min_genes=min_genes)
            print(f"After min_genes filter: {self.adata.shape[0]} cells")
            
            # Gene filtering - be more permissive for rare spermatogenesis genes
            sc.pp.filter_genes(self.adata, min_cells=10)  # More permissive
            print(f"After gene filter: {self.adata.shape[1]} genes")
            
            # Additional cell quality filters
            self.adata = self.adata[self.adata.obs.n_genes_by_counts < max_genes, :]
            self.adata = self.adata[self.adata.obs.total_counts > min_counts, :]
            self.adata = self.adata[self.adata.obs.total_counts < max_counts, :]
            
            if 'pct_counts_mt' in self.adata.obs.columns and self.adata.obs['pct_counts_mt'].max() > 0:
                self.adata = self.adata[self.adata.obs.pct_counts_mt < max_mt_percent, :]
            
            print(f"Final filtered data: {self.adata.shape[0]} cells × {self.adata.shape[1]} genes")
        
        
        # Quality control plots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Total counts
        axes[0].hist(self.adata.obs['total_counts'], bins=50, alpha=0.7)
        axes[0].set_xlabel('Total UMI counts')
        axes[0].set_ylabel('Number of cells')
        axes[0].set_title('UMI Count Distribution')
        
        # Number of genes
        axes[1].hist(self.adata.obs['n_genes_by_counts'], bins=50, alpha=0.7)
        axes[1].set_xlabel('Number of genes')
        axes[1].set_ylabel('Number of cells')
        axes[1].set_title('Gene Count Distribution')
        
        # Scatter plot
        if 'pct_counts_mt' in self.adata.obs.columns:
            sc.pl.scatter(self.adata, 
                        x='total_counts', 
                        y='n_genes_by_counts', 
                        color='pct_counts_mt', 
                        ax=axes[2], 
                        show=False, 
                        legend_loc='best'
                
            )
            fig.axes[-1].remove() #remove the colorbar/legend due to error still persisting: https://github.com/scverse/scanpy/issues/1258
        else:
            sc.pl.scatter(self.adata, 
                        x='total_counts', 
                        y='n_genes_by_counts',
                        ax=axes[2], 
                        show=False
            )

        
        plt.tight_layout()
        
        # Save the plot
        plot_path = os.path.join(self.output_dir, "filtering_qc_plots.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return self.adata
    
    def normalize_and_scale(self, hvg_custom=True):
        """Normalization and scaling optimized for spermatogenesis data."""
        
        print(f"\nNORMALIZATION & SCALING")
        
        # Save raw data
        self.adata.raw = self.adata
        
        # Normalize to 10,000 UMI per cell
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        
        # Log transform
        sc.pp.log1p(self.adata)
        
        hvg_min_mean = 0.01     # Lower threshold for lowly expressed genes
        hvg_max_mean = 8        # Higher threshold for highly expressed genes
        hvg_min_disp = 0.3      # Moderate dispersion threshold
        hvg_n_top_genes = 3000  # Select top 3000 HVGs, disables all other arguments
        if hvg_custom:
            sc.pp.highly_variable_genes(
                self.adata, 
                min_mean= hvg_min_mean,  
                max_mean= hvg_max_mean,
                min_disp= hvg_min_disp,
                n_top_genes= hvg_n_top_genes
            )
            self.analysis_parameters.update({
                'hvg_custom': hvg_custom,
                'hvg_min_mean': hvg_min_mean,
                'hvg_max_mean': hvg_max_mean,
                'hvg_min_disp': hvg_min_disp,
                'hvg_n_top_genes': hvg_n_top_genes
            })
        else: 
            sc.pp.highly_variable_genes(self.adata)
            self.analysis_parameters.update({
                'hvg_custom': hvg_custom
             })
        
        # save parameters
        self.save_parameters()

        # save hvg parameters
        if hvg_custom:
            print(f"Custom hvg parameters: {self.analysis_parameters['hvg_min_mean']}, {self.analysis_parameters['hvg_max_mean']}, {self.analysis_parameters['hvg_min_disp']}, {self.analysis_parameters['hvg_n_top_genes']}")
            print(f"Highly variable genes: {sum(self.adata.var['highly_variable'])}")

            # Plot hvg genes directly without using scanpy's save parameter
            sc.pl.highly_variable_genes(self.adata, show=False)
            plt.savefig(f"{self.output_dir}/hvg_genes.png", dpi=300, bbox_inches='tight')
            plt.close()

            # Check which spermatogenesis markers are highly variable
            available_markers = [gene for gene in self.trajectory_markers 
                            if gene in self.adata.var_names]
            hvg_markers = [gene for gene in available_markers 
                        if self.adata.var.loc[gene, 'highly_variable']]
            
            print(f"In data available spermatogenesis markers: {len(available_markers)}/{len(self.trajectory_markers)}")
            print(f"Markers not found in data: {set(self.trajectory_markers) - set(available_markers)}")
            print(f"HVG spermatogenesis markers: {len(hvg_markers)}")
            print(f"Top 12 HVG: {hvg_markers[:12]}")
            
            # Add all important spermatogenesis markers to HVG even if no select by scnapy
            for marker in available_markers[:100]:  # Top 100 most important
                if marker in self.adata.var_names:
                    self.adata.var.loc[marker, 'highly_variable'] = True
            
            # Keep only highly variable genes for downstream analysis
            if hvg_custom:
                self.adata = self.adata[:, self.adata.var.highly_variable]

        else:
            print(f"NO custom hvg filter but just for your information:")
            print(f"There would be {sum(self.adata.var['highly_variable'])} highly variable genes available")
        
        # Scale data
        #sc.pp.scale(self.adata, max_value=10) #removed for now
        
        #print(f"Normalized: {self.adata.shape[1]} features")
        
        return self.adata
    
    def advanced_dimensionality_reduction(self):
        """ optimized for trajectory inference."""
        
        print(f"\nADVANCED DIMENSIONALITY REDUCTION")
        
        # PCA with more components for trajectory analysis
        sc.tl.pca(self.adata, svd_solver='arpack', n_comps=50, random_state=42)
        
        # Compute neighborhood graph with optimized parameters for trajectory
        sc.pp.neighbors(
            self.adata, 
            n_neighbors=50,     # More neighbors for better trajectory
            n_pcs=30,           # Use more PCs
            metric='cosine',    # Cosine similarity often better for sparse data
            random_state=42
        )
        
        progress_steps = ["Computing diffmap", "Computing PAGA", "Compute UMAP with PAGA", "Compute diffusion-based UMAP"]
        with tqdm(total=4, desc="Analysis Progress", unit="step") as pbar:

            # Diffusion pseudotime for trajectory inference
            #print("Computing diffusion maps...")
            pbar.set_description(progress_steps[0])
            sc.tl.diffmap(self.adata, n_comps=15)
            pbar.update(1)
            
            # PAGA for trajectory topology
            #print("Computing PAGA...")
            pbar.set_description(progress_steps[1])
            sc.tl.leiden(self.adata, resolution=0.3, random_state=42, key_added='leiden_coarse')
            sc.tl.paga(self.adata, groups='leiden_coarse')
            pbar.update(1)
            
            # Use PAGA initialization for UMAP
            #print("Computing UMAP with PAGA initialization...")
            pbar.set_description(progress_steps[2])
            sc.pl.paga(self.adata, plot=False)  # Initialize positions
            sc.tl.umap(self.adata, init_pos='paga', min_dist=0.1, spread=1.0, random_state=42)
            pbar.update(1)
            
            # Alternative: Diffusion map-based UMAP
            #print("Computing diffusion-based UMAP...")
            pbar.set_description(progress_steps[3])
            sc.pp.neighbors(self.adata, use_rep='X_diffmap', n_neighbors=30, random_state=42)
            sc.tl.umap(self.adata, min_dist=0.05, spread=0.8, random_state=42)
            pbar.update(1)
            
            # Final clustering with multiple resolutions
            resolutions = [0.1, 0.3, 0.5, 0.8]
            for res in resolutions:
                sc.tl.leiden(self.adata, resolution=res, random_state=42, 
                            key_added=f'leiden_{res}')
        
        return self.adata
    
    def infer_spermatogenesis_trajectory(self):
        """Infer spermatogenesis trajectory using biological knowledge."""
        
        print(f"\nSPERMATOGENESIS TRAJECTORY INFERENCE")
        
        # Find clusters enriched in different stages
        stage_scores = {}
        
        for stage, markers in self.spermatogenesis_markers.items():
            available_markers = [m for m in markers if m in self.adata.var_names]
            if len(available_markers) > 0:
                #print(f"{stage}: {len(available_markers)} markers available")
                
                # Calculate stage score
                sc.tl.score_genes(self.adata, available_markers, 
                                 score_name=f'{stage}_score')
                stage_scores[stage] = f'{stage}_score'
        
        # Identify root cluster (likely spermatogonia-enriched)
        if 'spermatogonia_score' in self.adata.obs.columns:
            # Find cluster with highest spermatogonia score
            cluster_scores = self.adata.obs.groupby('leiden_0.3')['spermatogonia_score'].mean()
            root_cluster = cluster_scores.idxmax()
            
            # Set root for pseudotime
            root_cells = self.adata.obs['leiden_0.3'] == root_cluster
            if root_cells.sum() > 0:
                self.adata.uns['iroot'] = np.flatnonzero(root_cells)[0]
                
                # Compute diffusion pseudotime
                sc.tl.dpt(self.adata)
                print(f"Root cluster set: {root_cluster}")
            
        # Create spermatogenesis progression score
        # early_markers = (self.spermatogenesis_markers['Early_Sgonia'] + 
        #                  self.spermatogenesis_markers['Late_Sgonia'] +
        #                  self.spermatogenesis_markers['Early_Scytes'])
        # late_markers = (self.spermatogenesis_markers['Late_Scytes'] +
        #                 self.spermatogenesis_markers['Round_Stids'] +
        #                 self.spermatogenesis_markers['Later_Stids'])
        early_markers = (self.spermatogenesis_markers['spermatogonia'] +
                         self.spermatogenesis_markers['spermatocytes'])
        late_markers = (self.spermatogenesis_markers['round_spermatids'] +
                        self.spermatogenesis_markers['elongated_spermatids'])

        
        # Score early vs late markers
        early_available = [m for m in early_markers if m in self.adata.var_names]
        late_available = [m for m in late_markers if m in self.adata.var_names]

        if len(early_available) > 0:
            sc.tl.score_genes(self.adata, early_available, score_name='early_score')
        if len(late_available) > 0:
            sc.tl.score_genes(self.adata, late_available, score_name='late_score')
        
        # Create progression score (late - early)
        if 'early_score' in self.adata.obs.columns and 'late_score' in self.adata.obs.columns:
            self.adata.obs['progression_score'] = (
                self.adata.obs['late_score'] - self.adata.obs['early_score']
            )
        
        return self.adata
    
    def run_gsea_analysis(self):
        print(f"\nGSEA PATHWAY ANALYSIS")
        
        # Create phenotype based on progression score
        if 'progression_score' in self.adata.obs.columns:
            # Split into high/low progression
            mean_score = np.mean(self.adata.obs['progression_score'])
            self.adata.obs['progression_group'] = [
                'LATE' if score > mean_score else 'EARLY' 
                for score in self.adata.obs['progression_score']
            ]
        else:
            # Fallback to expression-based grouping
            mean_expr = np.mean(self.adata.obs['total_counts'])
            self.adata.obs['progression_group'] = [
                'HIGH' if count > mean_expr else 'LOW' 
                for count in self.adata.obs['total_counts']
            ]
        
        print(f"Progression groups: {self.adata.obs['progression_group'].value_counts()}")
        
        # Run GSEA with multiple gene sets
        gene_sets = [
            'GO_Biological_Process_2023',
            #'KEGG_2021_Human', 
            #'Reactome_2022',
            #'MSigDB_Hallmark_2020'
        ]
        
        gsea_results = {}
        
        for idx, gene_set in enumerate(gene_sets):
            try:
                print(f"({idx+1}/{len(gene_sets)}) Running GSEA with {gene_set}...")
                
                # Prepare data for GSEA (transpose: genes as rows, cells as columns)
                expr_df = pd.DataFrame(
                    self.adata.raw.X.toarray().T,
                    index=self.adata.raw.var_names,
                    columns=self.adata.obs_names
                )
                
                # Run GSEA
                res = gp.gsea(
                    data=expr_df,
                    gene_sets=gene_set,
                    cls=self.adata.obs['progression_group'],
                    permutation_num=500,  # Reduced for speed
                    permutation_type='phenotype',
                    outdir=None,
                    method='s2n',
                    threads=4,
                    seed=42
                )
                
                gsea_results[gene_set] = res
                print(f"   -> {len(res.res2d)} pathways found")
                
            except Exception as e:
                print(f"!!!!!!!!! Error with {gene_set}: {e}")
                continue
        
        # Save best results
        self.gsea_results = gsea_results
        
        # Display top pathways
        if gsea_results:
            best_results = list(gsea_results.values())[0]
            print(f"\nTOP PATHWAYS (first database):")
            print(best_results.res2d[['Term', 'ES', 'NES', 'FDR q-val']].head(10))
        
        return gsea_results
    
    def identify_cluster_markers(self):
        """Identify and validate cluster markers for spermatogenesis stages."""
        
        print(f"\nCLUSTER MARKER IDENTIFICATION")
        
        # Find markers for each cluster
        sc.tl.rank_genes_groups(
            self.adata, 
            groupby='leiden_0.3', 
            method='wilcoxon',
            n_genes=50,
            use_raw=True
        )
        
        # Get marker results
        marker_df = sc.get.rank_genes_groups_df(self.adata, group=None)
        
        # Annotate clusters based on top markers
        cluster_annotations = {}
        top_markers_per_cluster = {}
        
        for cluster in self.adata.obs['leiden_0.3'].unique():
            #####
            ## for cluster, annotation in self.cluster_annotations.items():
            #####
            cluster_markers = marker_df[
                (marker_df['group'] == cluster) & 
                (marker_df['pvals_adj'] < 0.05) &
                (marker_df['logfoldchanges'] > 0.5)
            ]['names'].head(10).tolist()

            top_markers_per_cluster[cluster] = cluster_markers
            
            # Annotate based on known markers
            annotation = self._annotate_cluster(cluster_markers)
            cluster_annotations[cluster] = annotation

            ############ might cause problem            
            count = (self.adata.obs['leiden_0.3'] == cluster).sum()
            print(f"Cluster {cluster} ({annotation}): {cluster_markers[:5]}\n   -> {count} cells")
        
        # Add annotations to adata
        self.adata.obs['cluster_annotation'] = [
            cluster_annotations.get(cluster, f"Cluster_{cluster}")
            for cluster in self.adata.obs['leiden_0.3']
        ]
        
        self.cluster_annotations = cluster_annotations
        self.top_markers_per_cluster = top_markers_per_cluster
        
        return marker_df
    
    def _annotate_cluster(self, cluster_markers):
        """Annotate cluster based on marker genes."""
        
        # Count markers for each cell type
        type_scores = {}
        
        for cell_type, known_markers in self.spermatogenesis_markers.items():
            overlap = len(set(cluster_markers) & set(known_markers))
            type_scores[cell_type] = overlap
        
        # Find best match
        if max(type_scores.values()) > 0:
            best_match = max(type_scores, key=type_scores.get)
            return best_match
        else:
            return "Unknown"
    
    def create_plots(self):
        """Create comprehensive visualization plots."""
        
        print(f"\nCREATING PLOTS")
        
        import matplotlib.pyplot as plt
        
        # 1. Overview UMAP plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Clusters
        sc.pl.umap(self.adata, color='leiden_0.3', legend_loc='on data', 
                  legend_fontsize=8, ax=axes[0,0], show=False, frameon=False)
        axes[0,0].set_title('Leiden Clusters')
        
        # Cluster annotations
        sc.pl.umap(self.adata, color='cluster_annotation', legend_loc='right margin',
                  ax=axes[0,1], show=False, frameon=False)
        axes[0,1].set_title('Cluster Annotations')
        
        # Pseudotime
        if 'dpt_pseudotime' in self.adata.obs.columns:
            sc.pl.umap(self.adata, color='dpt_pseudotime', color_map='viridis',
                      ax=axes[0,2], show=False, frameon=False)
            axes[0,2].set_title('Diffusion Pseudotime')
        
        # Progression score
        if 'progression_score' in self.adata.obs.columns:
            sc.pl.umap(self.adata, color='progression_score', color_map='RdYlBu_r',
                      ax=axes[1,0], show=False, frameon=False)
            axes[1,0].set_title('Spermatogenesis Progression')
        
        # Total UMI
        sc.pl.umap(self.adata, color='total_counts', color_map='plasma',
                  ax=axes[1,1], show=False, frameon=False)
        axes[1,1].set_title('Total UMI Counts')
        
        # Number of genes
        sc.pl.umap(self.adata, color='n_genes_by_counts', color_map='viridis',
                  ax=axes[1,2], show=False, frameon=False)
        axes[1,2].set_title('Number of Genes')
        
        plt.tight_layout()
        
        # Save the overview plot
        overview_path = os.path.join(self.output_dir, "overview_plots.png")
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        plt.show()
        

        # 2. Plot top 3 genes out of each cluster in one plot and iterate over all clusters
        if hasattr(self, 'top_markers_per_cluster'):
            print(f"\nCreating cluster-specific marker plots...")
            
            for cluster in sorted(self.adata.obs['leiden_0.3'].unique()):
                # Get top 3 markers for this cluster
                cluster_markers = self.top_markers_per_cluster.get(cluster, [])
                top_3_markers = cluster_markers[:3]
                
                # Filter markers that are available in raw data
                available_markers = [m for m in top_3_markers if m in self.adata.raw.var_names]
                
                if len(available_markers) == 0:
                    print(f"!!!! No markers available for cluster {cluster}")
                    continue
                
                # Create subplot layout: 1 cluster plot + up to 3 marker plots
                n_plots = 1 + len(available_markers)
                n_cols = min(4, n_plots)
                n_rows = (n_plots + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
                
                # Properly handle axes array indexing
                if n_rows == 1 and n_cols == 1:
                    # Single plot case
                    axes = [axes]
                elif n_rows == 1 or n_cols == 1:
                    # 1D array case - flatten to 1D
                    axes = axes.flatten()
                else:
                    # 2D array case - keep as is
                    pass
                
                # Plot 1: Cluster overview with highlighted cluster
                ax_cluster = axes[0] if n_rows == 1 or n_cols == 1 else axes[0, 0]
                
                # Create a copy of leiden labels for highlighting
                cluster_highlight = self.adata.obs['leiden_0.3'].copy()
                cluster_highlight = cluster_highlight.astype(str)
                cluster_highlight[cluster_highlight != str(cluster)] = 'Other'
                
                # Create temporary obs column for plotting
                self.adata.obs['temp_cluster_highlight'] = cluster_highlight
                
                sc.pl.umap(self.adata, color='temp_cluster_highlight', 
                          palette=['red', 'lightgray'], 
                          ax=ax_cluster, show=False, frameon=False,
                          legend_loc='right margin')
                ax_cluster.set_title(f'Cluster {cluster} Highlighted')
                
                # Plot 2-4: Top marker genes
                colors = ['Reds', 'Reds', 'Reds']
                for i, marker in enumerate(available_markers):
                    plot_idx = i + 1
                    if n_rows == 1 or n_cols == 1:
                        # 1D indexing
                        ax = axes[plot_idx]
                    else:
                        # 2D indexing
                        row, col = plot_idx // n_cols, plot_idx % n_cols
                        ax = axes[row, col]
                    
                    sc.pl.umap(self.adata, color=marker, use_raw=True,
                              color_map=colors[i % len(colors)], 
                              ax=ax, show=False, frameon=False)
                    ax.set_title(f'{marker} Expression')
                
                # Hide empty subplots
                for i in range(n_plots, n_rows * n_cols):
                    if n_rows == 1 or n_cols == 1:
                        # 1D indexing
                        if i < len(axes):
                            axes[i].set_visible(False)
                    else:
                        # 2D indexing
                        row, col = i // n_cols, i % n_cols
                        axes[row, col].set_visible(False)
                
                # Add overall title
                annotation = self.cluster_annotations.get(cluster, 'Unknown')
                fig.suptitle(f'Cluster {cluster} ({annotation}) - Top Markers', 
                           fontsize=16, y=0.98)
                
                plt.tight_layout()
                
                # Save cluster-specific plot
                cluster_plot_path = os.path.join(self.output_dir, f"cluster_{cluster}_markers.png")
                plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
                # do not show the plot
                plt.close()
                
                # Clean up temporary column
                del self.adata.obs['temp_cluster_highlight']
        
        
        
        # 3. Key marker genes
        key_markers = ['PRM1', 'TNP1', 'ID4', 'STRA8', 'ZBTB16', 'SYCP3']
        available_key_markers = [m for m in key_markers if m in self.adata.raw.var_names]
        
        if len(available_key_markers) > 0:
            n_markers = len(available_key_markers)
            n_cols = min(3, n_markers)
            n_rows = (n_markers + n_cols - 1) // n_cols
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
            
            # Properly handle axes array indexing
            if n_rows == 1 and n_cols == 1:
                # Single plot case
                axes = [axes]
            elif n_rows == 1 or n_cols == 1:
                # 1D array case - flatten to 1D
                axes = axes.flatten()
            else:
                # 2D array case - keep as is
                pass
            
            for i, marker in enumerate(available_key_markers):
                if n_rows == 1 or n_cols == 1:
                    # 1D indexing
                    ax = axes[i]
                else:
                    # 2D indexing
                    row, col = i // n_cols, i % n_cols
                    ax = axes[row, col]
                
                sc.pl.umap(self.adata, color=marker, use_raw=True, 
                          color_map='Reds', ax=ax, show=False, frameon=False)
                ax.set_title(f'{marker} Expression')
            
            # Hide empty subplots
            for i in range(len(available_key_markers), n_rows * n_cols):
                if n_rows == 1 or n_cols == 1:
                    # 1D indexing
                    if i < len(axes):
                        axes[i].set_visible(False)
                else:
                    # 2D indexing
                    row, col = i // n_cols, i % n_cols
                    axes[row, col].set_visible(False)
            
            plt.tight_layout()
            
            # Save key markers plot
            key_markers_path = os.path.join(self.output_dir, "key_marker_genes.png")
            plt.savefig(key_markers_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        # 3. Stage scores heatmap
        stage_cols = [col for col in self.adata.obs.columns if col.endswith('_score')]
        if len(stage_cols) > 0:
            plt.figure(figsize=(12, 8))
            
            # Calculate mean scores per cluster
            score_matrix = []
            for cluster in sorted(self.adata.obs['leiden_0.3'].unique()):
                cluster_cells = self.adata.obs['leiden_0.3'] == cluster
                cluster_scores = self.adata.obs.loc[cluster_cells, stage_cols].mean()
                score_matrix.append(cluster_scores)
            
            score_df = pd.DataFrame(score_matrix, 
                                  index=sorted(self.adata.obs['leiden_0.3'].unique()),
                                  columns=stage_cols)
            
            sns.heatmap(score_df, annot=True, cmap='RdYlBu_r', center=0,
                       cbar_kws={'label': 'Average Stage Score'})
            plt.title('Spermatogenesis Stage Scores by Cluster')
            plt.xlabel('Spermatogenesis Stage')
            plt.ylabel('Leiden Cluster')
            plt.tight_layout()
            
            # Save heatmap
            heatmap_path = os.path.join(self.output_dir, "stage_scores_heatmap.png")
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.show()
        
        # 4. Trajectory plot if pseudotime available
        if 'dpt_pseudotime' in self.adata.obs.columns:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            
            # Pseudotime vs progression score
            if 'progression_score' in self.adata.obs.columns:
                scatter = axes[0,0].scatter(self.adata.obs['dpt_pseudotime'], 
                                          self.adata.obs['progression_score'],
                                          c=self.adata.obs['leiden_0.3'].astype(int),
                                          cmap='tab10', alpha=0.6, s=1)
                axes[0,0].set_xlabel('DPT Pseudotime')
                axes[0,0].set_ylabel('Progression Score')
                axes[0,0].set_title('Trajectory Validation')
                plt.colorbar(scatter, ax=axes[0,0], label='Cluster')
            
            # Pseudotime distribution
            axes[0,1].hist(self.adata.obs['dpt_pseudotime'], bins=50, alpha=0.7)
            axes[0,1].set_xlabel('DPT Pseudotime')
            axes[0,1].set_ylabel('Number of Cells')
            axes[0,1].set_title('Pseudotime Distribution')
            
            # Cluster vs pseudotime
            cluster_pseudotime = []
            cluster_labels = []
            for cluster in sorted(self.adata.obs['leiden_0.3'].unique()):
                cluster_cells = self.adata.obs['leiden_0.3'] == cluster
                cluster_pseudotime.extend(self.adata.obs.loc[cluster_cells, 'dpt_pseudotime'])
                cluster_labels.extend([f'C{cluster}'] * cluster_cells.sum())
            
            df_plot = pd.DataFrame({'Pseudotime': cluster_pseudotime, 'Cluster': cluster_labels})
            
            # Violin plot
            import matplotlib.pyplot as plt
            clusters = sorted(df_plot['Cluster'].unique())
            positions = range(len(clusters))
            
            violin_parts = axes[1,0].violinplot([df_plot[df_plot['Cluster'] == c]['Pseudotime'] 
                                               for c in clusters], positions=positions)
            axes[1,0].set_xticks(positions)
            axes[1,0].set_xticklabels(clusters)
            axes[1,0].set_xlabel('Cluster')
            axes[1,0].set_ylabel('DPT Pseudotime')
            axes[1,0].set_title('Pseudotime by Cluster')
            
            # PAGA plot
            if 'paga' in self.adata.uns:
                sc.pl.paga(self.adata, color='leiden_0.3', ax=axes[1,1], 
                          show=False, frameon=False)
                axes[1,1].set_title('PAGA Trajectory Graph')
            
            plt.tight_layout()
            
            # Save trajectory plots
            trajectory_path = os.path.join(self.output_dir, "trajectory_analysis.png")
            plt.savefig(trajectory_path, dpi=300, bbox_inches='tight')
            plt.show()

    def cellranger_emptydrops_comparison(self):
        if self.CREDcomparison:
            print(f"\nCOMPARING CELLRANGER AND EMPTYDROPS, set CREDcomparison=False in the class initialization to deactivate this plot")
            
            # check if we got the union with the added column "detection_method", otherwise we throw an error and check if there are only 0,1 and 2 values in the column
            if 'detection_method' not in self.adata.obs.columns:
                raise ValueError("Detection method column not found, no union provided, cannot proceed with comparison")
            if not all(self.adata.obs['detection_method'].isin([0,1,2])):
                raise ValueError("Detection method column contains invalid values, be sure only 0,1 and 2 values are present")
            
            print(f"Found {len(self.adata.obs[self.adata.obs['detection_method'] == 0])}x CellRanger only cells")
            print(f"Found {len(self.adata.obs[self.adata.obs['detection_method'] == 1])}x EmptyDrops only cells")
            print(f"Found {len(self.adata.obs[self.adata.obs['detection_method'] == 2])}x detected by both methods cells")

            stats_data = []
            method_names = ['CellRanger only (0)', 'EmptyDrops only (1)', 'Both methods (2)']
            
            for method_val, method_name in zip([0, 1, 2], method_names):
                mask = self.adata.obs['detection_method'] == method_val
                n_cells = mask.sum()
                if n_cells > 0:
                    subset_adata = self.adata[mask].copy()
                    if 'n_genes_by_counts' in subset_adata.obs.columns:
                        mean_genes = subset_adata.obs['n_genes_by_counts'].mean()
                    else:
                        mean_genes = (subset_adata.X > 0).sum(axis=1).mean()
                    if 'total_counts' in subset_adata.obs.columns:
                        mean_counts = subset_adata.obs['total_counts'].mean()
                    else:
                        mean_counts = np.array(subset_adata.X.sum(axis=1)).flatten().mean()
                    gene_sums = np.array(subset_adata.X.sum(axis=0)).flatten()
                    top_genes_idx = gene_sums.argsort()[-10:][::-1]
                    top_genes = [subset_adata.var_names[i] for i in top_genes_idx]
                    if 'cluster_annotation' in subset_adata.obs.columns:
                        cluster_counts = subset_adata.obs['cluster_annotation'].value_counts()
                        top_clusters = cluster_counts.head(3).to_dict()
                    else:
                        top_clusters = "No annotations available"
                    
                    stats_data.append({
                        'Detection Method': method_name,
                        'Number of Cells': n_cells,
                        'Mean Genes per Cell': f"{mean_genes:.1f}",
                        'Mean Counts per Cell': f"{mean_counts:.1f}",
                        'Top 10 Genes': ', '.join(top_genes),
                        'Top Cell Types': str(top_clusters)
                    })
                else:
                    stats_data.append({
                        'Detection Method': method_name,
                        'Number of Cells': 0,
                        'Mean Genes per Cell': 'N/A',
                        'Mean Counts per Cell': 'N/A',
                        'Top 10 Genes': 'N/A',
                        'Top Cell Types': 'N/A'
                    })
            
            # Create and display the comparison table
            comparison_df = pd.DataFrame(stats_data)
            print("\n" + "="*300)
            print("DETECTION METHOD COMPARISON TABLE")
            print("="*300)
            print(comparison_df.to_string(index=False))
            print("="*300)

            # create plots to show the difference between the two methods
            fig, axes = plt.subplots(1, 4, figsize=(20, 5))
            method_names = ['CellRanger only', 'EmptyDrops only', 'Both methods', 'All methods combined']
            for i, (method_val, name) in enumerate(zip([0, 1, 2], method_names[:3])):
                mask = self.adata.obs['detection_method'] == method_val
                sc.pl.umap(self.adata, ax=axes[i], show=False, frameon=False, s=10, alpha=0.3)
                if mask.sum() > 0:
                    #palette={"Emptydrops": "yellow", "Cellranger": "orange", "both methods": "green"}
                    sc.pl.umap(self.adata[mask], color= ['detection_method'], ax=axes[i], show=False, frameon=False, s=15, alpha=0.8, color_map="magma")
                axes[i].set_title(f'{name} (n={mask.sum()})')
            detection_method_colors = {0: 'red', 1: 'yellow', 2: 'orange'}
            # Plot points in order ED only (1), Both methods (2), CellRanger only 0 so that CR lays on top
            plot_order = [1, 2, 0]
            for method_val in plot_order:
                mask = self.adata.obs['detection_method'] == method_val
                if mask.sum() > 0:
                    color = detection_method_colors[method_val]
                    # Set alpha to 0.8 explicitly for CR only , rest is 0.1 
                    alpha_val = 0.8 if method_val == 0 else 0.1
                    axes[3].scatter(self.adata.obsm['X_umap'][mask, 0], self.adata.obsm['X_umap'][mask, 1], 
                                    c=color, s=15, alpha=alpha_val)
            red_patch = mpatches.Patch(color='red', label='CellRanger only')
            yellow_patch = mpatches.Patch(color='yellow', label='EmptyDrops only')
            orange_patch = mpatches.Patch(color='orange', label='Both methods')
            axes[3].legend(handles=[red_patch, yellow_patch, orange_patch], loc='upper right')
            axes[3].set_title(f'{method_names[3]} (n={len(self.adata)})')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/cellranger_emptydrops_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            
            # Get total counts for each cell and plot distribution of each detection method
            if 'total_counts' in self.adata.obs.columns:
                counts = self.adata.obs['total_counts'].values
            else:
                counts = np.array(self.adata.X.sum(axis=1)).flatten()
            detection_methods = self.adata.obs['detection_method'].values
            sorted_indices = np.argsort(counts)[::-1]
            sorted_counts = counts[sorted_indices] 
            print(sorted_counts[:10])
            sorted_methods = detection_methods[sorted_indices] 
            color_map = {0: 'red', 1: 'blue', 2: 'green'}  # Changed yellow to blue for better contrast
            line_colors = [color_map[int(m)] for m in sorted_methods]
            cr_only_mask = sorted_methods == 0
            cr_only_ranks = np.where(cr_only_mask)[0] + 1  # +1 to make ranks 1-based
            print(f"Ranks of CellRanger only cells: {cr_only_ranks.tolist()}")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            n_bins = 50
            bin_edges = np.linspace(0, len(sorted_counts), n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # count the cells in each bin for each method
            cr_counts = np.zeros(n_bins)
            ed_counts = np.zeros(n_bins)
            both_counts = np.zeros(n_bins)
            for i in range(n_bins):
                start_idx = int(bin_edges[i])
                end_idx = int(bin_edges[i + 1])
                bin_methods = sorted_methods[start_idx:end_idx]
                
                cr_counts[i] = np.sum(bin_methods == 0)
                ed_counts[i] = np.sum(bin_methods == 1)
                both_counts[i] = np.sum(bin_methods == 2)
            # left plot
            ax1.plot(bin_centers, cr_counts, color='red', linewidth=2, alpha=0.8, label=f'CellRanger only (n={np.sum(sorted_methods==0)})', marker='o', markersize=4)
            ax1.plot(bin_centers, ed_counts, color='blue', linewidth=2, alpha=0.8, label=f'EmptyDrops only (n={np.sum(sorted_methods==1)})', marker='s', markersize=4)
            ax1.plot(bin_centers, both_counts, color='green', linewidth=2, alpha=0.8, label=f'Both methods (n={np.sum(sorted_methods==2)})', marker='^', markersize=4)
            ax1.set_xlabel('Cell Ranks (sorted by total UMI counts, descending, bin size=50)', fontsize=12)
            ax1.set_ylabel('Number of cells per bin', fontsize=12)
            ax1.set_title('Distribution of Detection Methods Across UMI Count Ranks', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right', fontsize=10)
            ax1.grid(axis='both', alpha=0.3, linestyle='--')
            ax1.set_axisbelow(True)
            # Right plot
            ax2.bar(bin_centers, cr_counts, color='red', alpha=0.8, label=f'CellRanger only (n={np.sum(sorted_methods==0)})', width=bin_centers[1]-bin_centers[0])
            ax2.bar(bin_centers, ed_counts, bottom=cr_counts, color='blue', alpha=0.8, label=f'EmptyDrops only (n={np.sum(sorted_methods==1)})', width=bin_centers[1]-bin_centers[0])
            ax2.bar(bin_centers, both_counts, bottom=cr_counts+ed_counts, color='green', alpha=0.8, label=f'Both methods (n={np.sum(sorted_methods==2)})', width=bin_centers[1]-bin_centers[0])
            ax2.set_xlabel('Cell Ranks (sorted by total UMI counts, descending, bin size=50)', fontsize=12)
            ax2.set_ylabel('Number of cells per bin', fontsize=12)
            ax2.set_title('Distribution of Detection Methods Across UMI Count Ranks', fontsize=14, fontweight='bold')
            ax2.legend(loc='upper right', fontsize=10)
            ax2.grid(axis='y', alpha=0.3, linestyle='--')
            ax2.set_axisbelow(True)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/cellranger_emptydrops_comparison_umicounts.png", dpi=300, bbox_inches='tight')
            plt.show()


    def export_results(self):
        """Export all results for further analysis."""
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 1. Save processed data
        self.adata.write_h5ad(f"{self.output_dir}/processed_spermatogenesis_data.h5ad")
        
        # 2. Export cluster markers
        if hasattr(self, 'top_markers_per_cluster'):
            markers_df = []
            for cluster, markers in self.top_markers_per_cluster.items():
                for i, marker in enumerate(markers):
                    markers_df.append({
                        'cluster': cluster,
                        'gene': marker,
                        'rank': i+1,
                        'annotation': self.cluster_annotations.get(cluster, 'Unknown')
                    })
            
            pd.DataFrame(markers_df).to_csv(f"{self.output_dir}/cluster_markers.csv", index=False)
        
        # 3. Export for Metascape
        marker_df = sc.get.rank_genes_groups_df(self.adata, group=None)
        significant_markers = marker_df[
            (marker_df['pvals_adj'] < 0.05) & 
            (marker_df['logfoldchanges'] > 1.0)
        ].groupby('group').head(30)
        
        significant_markers.to_csv(f"{self.output_dir}/metascape_markers.csv", index=False)
        pd.DataFrame({'Gene': significant_markers['names'].tolist()}).to_csv(
            f"{self.output_dir}/metascape_gene_list.csv", index=False
        )
        
        # 4. Export GSEA results
        if hasattr(self, 'gsea_results'):
            for db_name, res in self.gsea_results.items():
                res.res2d.to_csv(f"{self.output_dir}/gsea_{db_name}.csv", index=False)
        
        # 5. Export cell metadata
        obs_df = self.adata.obs.copy()
        obs_df['UMAP1'] = self.adata.obsm['X_umap'][:, 0]
        obs_df['UMAP2'] = self.adata.obsm['X_umap'][:, 1]
        obs_df.to_csv(f"{self.output_dir}/cell_metadata.csv")
        
        # Print summary
        print(f"\nANALYSIS SUMMARY")
        print(f"{'='*50}")
        print(f"Final data: {self.adata.shape[0]} cells × {self.adata.shape[1]} features")
        print(f"Clusters identified: {len(self.adata.obs['leiden_0.3'].unique())}")
        if hasattr(self, 'cluster_annotations'):
            print(f"Annotated clusters: {len(self.cluster_annotations)}")
        
        if 'dpt_pseudotime' in self.adata.obs.columns:
            print(f"Pseudotime range: {self.adata.obs['dpt_pseudotime'].min():.3f} - {self.adata.obs['dpt_pseudotime'].max():.3f}")
        
        if hasattr(self, 'gsea_results'):
            print(f"GSEA databases: {len(self.gsea_results)}")
        
        print(f"{'='*50}")
    
    def run_complete_analysis(self):
        """Run the complete spermatogenesis analysis pipeline."""
        
        print(f"{'='*60}")
        
        # 1: Load and preprocess
        self.load_and_preprocess_data()
        
        # 2: Advanced filtering
        self.advanced_filtering(activate=self.activate_advanced_filtering)
        
        # 3 Normalization and scaling
        self.normalize_and_scale(hvg_custom=self.activate_hvg)
        
        # Step 4: Dimensionality reduction
        self.advanced_dimensionality_reduction()
        
        # 5: Trajectory inference
        self.infer_spermatogenesis_trajectory()
        
        # 6: GSEA analysis
        self.run_gsea_analysis()
        
        # 7 Cluster markers
        self.identify_cluster_markers()
        
        # Step 8: Comprehensive plots
        self.create_plots()

        # Step nine:
        self.cellranger_emptydrops_comparison()
        
        # Step 10: Export results
        self.export_results()
        # i included timestamp saving dict to init in class so func is not needed anymore
        
        return self.adata

def main():
    # Initialize analyzer
    analyzer = SpermatogenesisAnalyzer()
    
    # Run complete analysis
    adata = analyzer.run_complete_analysis()
    
    
    return adata, analyzer

if __name__ == "__main__":
    adata, analyzer = main()
