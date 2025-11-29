import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import ndimage
from sklearn.decomposition import NMF
import scanpy as sc
from scipy.stats import gaussian_kde

def load_emptydrops_matrix():
    """Load the EmptyDrops filtered matrix properly."""
    adata = sc.read_h5ad("data/emptydrops_all_detected_cells.h5ad")
    return adata.X.toarray(), adata

def smooth_component(component, sigma=2):
    """Apply Gaussian smoothing to reduce spikiness."""
    return ndimage.gaussian_filter1d(component, sigma=sigma)

def plot_nmf_components_improved(matrix, n_components=5, figsize=(15, 12)):
    """
    Plot NMF components with multiple improved visualization methods.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Gene expression matrix (cells x genes)
    n_components : int
        Number of NMF components
    figsize : tuple
        Figure size
    """
    print(f"🔬 Running NMF with {n_components} components...")
    
    # Fit NMF
    model = NMF(n_components=n_components, random_state=42, max_iter=500)
    W = model.fit_transform(matrix)  # Cell loadings
    H = model.components_  # Gene loadings (components)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=figsize)
    
    for i in range(n_components):
        component = H[i, :]  # Gene loadings for component i
        
        # 1. Heatmap of top genes (much better than line plot!)
        plt.subplot(n_components, 4, i*4 + 1)
        top_genes_idx = np.argsort(component)[-100:]  # Top 100 genes
        top_values = component[top_genes_idx]
        sns.heatmap(top_values.reshape(-1, 1), 
                   cmap='Reds', cbar=True, 
                   yticklabels=False, xticklabels=[f'Comp {i+1}'])
        plt.title(f'Component {i+1}\nTop 100 Genes')
        
        # 2. Histogram of component values
        plt.subplot(n_components, 4, i*4 + 2)
        plt.hist(component[component > 0], bins=50, alpha=0.7, color='skyblue')
        plt.xlabel('Component Value')
        plt.ylabel('Count')
        plt.title(f'Component {i+1}\nValue Distribution')
        plt.yscale('log')
        
        # 3. Smoothed component plot (reduces spikiness)
        plt.subplot(n_components, 4, i*4 + 3)
        smoothed = smooth_component(component, sigma=5)
        plt.plot(smoothed, color='red', alpha=0.8, linewidth=2, label='Smoothed')
        plt.plot(component, color='gray', alpha=0.3, linewidth=0.5, label='Original')
        plt.xlabel('Gene Index')
        plt.ylabel('Loading')
        plt.title(f'Component {i+1}\nSmoothed vs Original')
        plt.legend()
        
        # 4. Kernel density of non-zero values
        plt.subplot(n_components, 4, i*4 + 4)
        nonzero_vals = component[component > np.percentile(component, 90)]  # Top 10%
        if len(nonzero_vals) > 10:
            try:
                kde = gaussian_kde(nonzero_vals)
                x_range = np.linspace(nonzero_vals.min(), nonzero_vals.max(), 100)
                plt.plot(x_range, kde(x_range), color='purple', linewidth=2)
                plt.fill_between(x_range, kde(x_range), alpha=0.3, color='purple')
                plt.xlabel('Component Value')
                plt.ylabel('Density')
                plt.title(f'Component {i+1}\nKDE (Top 10%)')
            except:
                plt.text(0.5, 0.5, 'Not enough\nvariation for KDE', 
                        ha='center', va='center', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
    
    return W, H, model

def plot_component_summary(H, figsize=(12, 8)):
    """Create a summary heatmap of all components."""
    plt.figure(figsize=figsize)
    
    # Show heatmap of all components (genes x components)
    plt.subplot(2, 2, 1)
    # Take top genes from each component
    top_genes_per_comp = []
    for i in range(H.shape[0]):
        top_idx = np.argsort(H[i, :])[-50:]  # Top 50 genes per component
        top_genes_per_comp.extend(top_idx)
    
    unique_top_genes = list(set(top_genes_per_comp))
    subset_H = H[:, unique_top_genes]
    
    sns.heatmap(subset_H, cmap='Reds', cbar=True)
    plt.title('All Components\n(Top Contributing Genes Only)')
    plt.xlabel('Top Genes')
    plt.ylabel('Components')
    
    # Component sparsity
    plt.subplot(2, 2, 2)
    sparsity = [np.sum(H[i, :] > 0) / H.shape[1] for i in range(H.shape[0])]
    plt.bar(range(1, len(sparsity)+1), sparsity, color='lightcoral')
    plt.xlabel('Component')
    plt.ylabel('Fraction of Non-zero Genes')
    plt.title('Component Sparsity')
    
    # Component max values
    plt.subplot(2, 2, 3)
    max_vals = [np.max(H[i, :]) for i in range(H.shape[0])]
    plt.bar(range(1, len(max_vals)+1), max_vals, color='lightblue')
    plt.xlabel('Component')
    plt.ylabel('Maximum Value')
    plt.title('Component Maximum Values')
    
    # Component variance
    plt.subplot(2, 2, 4)
    variances = [np.var(H[i, :]) for i in range(H.shape[0])]
    plt.bar(range(1, len(variances)+1), variances, color='lightgreen')
    plt.xlabel('Component')
    plt.ylabel('Variance')
    plt.title('Component Variance')
    
    plt.tight_layout()
    plt.show()

def plot_sorted_components(H, n_top_genes=100, figsize=(12, 8)):
    """Plot components with genes sorted by loading values."""
    n_components = H.shape[0]
    
    fig, axes = plt.subplots(n_components, 1, figsize=figsize)
    if n_components == 1:
        axes = [axes]
    
    for i in range(n_components):
        # Sort genes by component loading
        sorted_idx = np.argsort(H[i, :])[::-1]  # Descending order
        sorted_values = H[i, sorted_idx]
        
        # Plot only top genes to reduce noise
        axes[i].plot(range(n_top_genes), sorted_values[:n_top_genes], 
                    color='red', linewidth=2, alpha=0.8)
        axes[i].fill_between(range(n_top_genes), sorted_values[:n_top_genes], 
                           alpha=0.3, color='red')
        
        axes[i].set_title(f'Component {i+1} - Top {n_top_genes} Genes (Sorted)')
        axes[i].set_xlabel('Gene Rank')
        axes[i].set_ylabel('Loading Value')
        axes[i].grid(True, alpha=0.3)
        
        # Add threshold line
        threshold = np.percentile(sorted_values, 95)
        axes[i].axhline(y=threshold, color='black', linestyle='--', 
                       alpha=0.5, label=f'95th percentile')
        axes[i].legend()
    
    plt.tight_layout()
    plt.show()

# Example usage function
def demo_nmf_visualization():
    """Demo function showing proper NMF component visualization."""
    print("📊 Loading EmptyDrops matrix...")
    matrix, adata = load_emptydrops_matrix()
    
    print(f"Matrix shape: {matrix.shape}")
    print("🎨 Creating improved NMF visualizations...")
    
    # Use smaller subset for demo (faster computation)
    subset_matrix = matrix[:1000, :1000]  # First 1000 cells and genes
    
    # Run the improved visualization
    W, H, model = plot_nmf_components_improved(subset_matrix, n_components=3)
    
    # Show summary
    plot_component_summary(H)
    
    # Show sorted components
    plot_sorted_components(H, n_top_genes=50)
    
    return W, H, model

if __name__ == "__main__":
    demo_nmf_visualization() 