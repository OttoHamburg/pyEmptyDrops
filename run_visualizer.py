import scanpy as sc
from empty_drops_visualizer import visualize_empty_drops

# Load your data
# Using the raw data file that exists in your directory
adata = sc.read_10x_h5("raw_feature_bc_matrix.h5")

# Run EmptyDrops with visualization
results, intermediate_data = visualize_empty_drops(
    data=adata,
    lower=100,  # Adjust parameters as needed
    niters=1000,
    progress=True
)

# Print summary of results
print(f"Found {(results['FDR'] < 0.05).sum()} cells with FDR < 0.05")

# You can also use the additional visualization functions
from empty_drops_visualizer import plot_umi_vs_genes
plot_umi_vs_genes(adata, results, output_file="empty_drops_visualizations/umi_vs_genes.png")