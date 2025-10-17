import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load data
print("Loading data...")
raw_adata = sc.read_10x_h5('raw_feature_bc_matrix.h5')
filtered_adata = sc.read_10x_h5('filtered_feature_bc_matrix.h5')
empty_drops_results = pd.read_csv('empty_drops_results.csv', index_col=0)

# Calculate UMI counts per cell
raw_umi_counts = raw_adata.X.sum(axis=1)
if isinstance(raw_umi_counts, np.matrix):
    raw_umi_counts = raw_umi_counts.A1

# Create comparison dataframe
df_compare = pd.DataFrame({
    'UMI_Counts': raw_umi_counts,
    'In_CellRanger': raw_adata.obs_names.isin(filtered_adata.obs_names),
    'In_EmptyDrops': empty_drops_results['FDR < 0.05']
})

# 1. UMI Distribution Comparison
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(data=df_compare, x='UMI_Counts', hue='In_CellRanger', 
             bins=50, log_scale=True, stat='density')
plt.title('UMI Distribution (Cell Ranger)')
plt.xlabel('UMI Counts (log scale)')

plt.subplot(1, 2, 2)
sns.histplot(data=df_compare, x='UMI_Counts', hue='In_EmptyDrops', 
             bins=50, log_scale=True, stat='density')
plt.title('UMI Distribution (EmptyDrops)')
plt.xlabel('UMI Counts (log scale)')
plt.tight_layout()
plt.savefig('umi_distribution_comparison.png')
plt.close()

# 2. Venn Diagram-like comparison
from matplotlib_venn import venn2
plt.figure(figsize=(8, 8))
venn2([set(df_compare[df_compare['In_CellRanger']].index),
       set(df_compare[df_compare['In_EmptyDrops']].index)],
      set_labels=('Cell Ranger', 'EmptyDrops'))
plt.title('Overlap between Cell Ranger and EmptyDrops Cells')
plt.savefig('cell_overlap_venn.png')
plt.close()

# 3. Scatter plot of gene counts vs UMI counts
gene_counts = (raw_adata.X > 0).sum(axis=1)
if isinstance(gene_counts, np.matrix):
    gene_counts = gene_counts.A1

df_compare['Gene_Counts'] = gene_counts

plt.figure(figsize=(10, 8))
plt.scatter(df_compare[~df_compare['In_CellRanger']]['UMI_Counts'],
           df_compare[~df_compare['In_CellRanger']]['Gene_Counts'],
           alpha=0.1, label='Not Called', color='gray')
plt.scatter(df_compare[df_compare['In_EmptyDrops']]['UMI_Counts'],
           df_compare[df_compare['In_EmptyDrops']]['Gene_Counts'],
           alpha=0.05, label='EmptyDrops', color='red')
plt.scatter(df_compare[df_compare['In_CellRanger']]['UMI_Counts'],
           df_compare[df_compare['In_CellRanger']]['Gene_Counts'],
           alpha=0.3, label='Cell Ranger', color='blue')
plt.xscale('log')
plt.yscale('log')
plt.xlabel('UMI Counts (log scale)')
plt.ylabel('Gene Counts (log scale)')
plt.title('Gene Counts vs UMI Counts')
plt.legend()
plt.savefig('gene_umi_scatter.png')
plt.close()

# Print summary statistics
print("\nSummary Statistics:")
print("\nCell Ranger Cells:")
print(df_compare[df_compare['In_CellRanger']]['UMI_Counts'].describe())
print("\nEmptyDrops Cells:")
print(df_compare[df_compare['In_EmptyDrops']]['UMI_Counts'].describe())

# Current parameters:
#empty_drops(
#    data,
#    lower=100,  # Current threshold
#    retain=None,
#    niters=10000,
#    alpha=None
#)

# Suggested modifications:
#empty_drops(
#    data,
#    lower=500,  # Increase to match Cell Ranger's apparent threshold
#    retain=3000,  # Set based on Cell Ranger's median UMI count
#    niters=10000,
#    alpha=0.1  # Add concentration parameter to account for high ambient RNA
#)

def additional_qc_metrics(adata):
    # Gene detection rate
    adata.obs['n_genes'] = (adata.X > 0).sum(axis=1)
    
    # Mitochondrial content
    adata.var['mt'] = adata.var_names.str.startswith('MT-')
    adata.obs['percent_mt'] = adata.X[:, adata.var['mt']].sum(axis=1) / adata.X.sum(axis=1) * 100
    
    # RNA complexity (genes per UMI)
    adata.obs['complexity'] = adata.obs['n_genes'] / adata.obs['n_counts']
    
    # Expression of known cell markers
    # (Add specific markers for your cell types) 