# pyEmptyDrops Documentation

Production-ready command-line tool and Python module for running EmptyDrops analysis on 10x Genomics single-cell RNA sequencing data.

## Overview

`run_empty_drops.py` provides a streamlined interface to the EmptyDrops algorithm, which identifies real cells from ambient RNA in droplet-based single-cell sequencing experiments. The script can be used both as a command-line tool and as an importable Python module.

## Installation

### Requirements

The following packages are required:

- Python >= 3.9
- numpy >= 1.26.0
- pandas >= 2.2.0
- scanpy >= 1.10.0
- matplotlib >= 3.9.0
- scipy >= 1.13.0
- statsmodels >= 0.14.0
- numba >= 0.60.0
- tqdm >= 4.67.0

**Note**: `scanpy` automatically installs `anndata` as a dependency.

### Recommended: Use a Virtual Environment

It is strongly recommended to use a Python virtual environment to avoid conflicts with other Python packages:

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Why use a virtual environment?**
- Prevents conflicts with system-wide Python packages
- Ensures reproducible environments across different machines
- Easily isolate project dependencies
- Can be deleted and recreated without affecting system Python

### Installation Methods

**Option 1: Using pip**

```bash
pip install numpy pandas scanpy matplotlib scipy statsmodels numba tqdm
```

**Option 2: Using requirements.txt**

Create a `requirements.txt` file with:
```
numpy>=1.26.0
pandas>=2.2.0
scanpy>=1.10.0
matplotlib>=3.9.0
scipy>=1.13.0
statsmodels>=0.14.0
numba>=0.60.0
tqdm>=4.67.0
```

Then install:
```bash
pip install -r requirements.txt
```

**Tested Versions:**
- Python 3.9.6
- numpy 1.26.4
- pandas 2.2.3
- scanpy 1.10.3
- matplotlib 3.9.4
- scipy 1.13.1
- statsmodels 0.14.4
- numba 0.60.0
- tqdm 4.67.1
- anndata 0.10.9

## Usage

### Command-Line Interface

Basic usage:

```bash
python run_empty_drops.py input.h5
```

With custom output directory:

```bash
python run_empty_drops.py input.h5 -o results/
```

With custom parameters:

```bash
python run_empty_drops.py input.h5 --lower 200 --niters 5000
```

Without plotting:

```bash
python run_empty_drops.py input.h5 --no-plot
```

### Python Module

```python
from run_empty_drops import run_empty_drops

results_df, metadata, adata = run_empty_drops(
    'input.h5',
    output_dir='results/',
    lower=100,
    niters=10000,
    plot=True
)
```

## Parameters

### Required Arguments

- `input_file` (str): Path to input 10x Genomics H5 file (format: `raw_feature_bc_matrix.h5`)

### Optional Arguments

- `-o, --output-dir` (str, default: "."): Output directory for results files
- `--output-prefix` (str, default: None): Custom prefix for output files. If not specified, uses the input filename without extension
- `--lower` (int, default: 100): Lower threshold for testing. Only barcodes with UMI counts above this value will be tested for empty droplets
- `--niters` (int, default: 10000): Number of Monte Carlo iterations for p-value calculation. Higher values provide more accurate p-values but increase computation time
- `--retain` (int, default: None): Retain threshold (knee point). Barcodes with UMI counts above this value automatically have FDR=0. If None, the knee point is automatically calculated using barcode ranks
- `--max-batches` (int, default: 100): Maximum number of batches for optimization. Controls the batching strategy to reduce computational overhead. Lower values may reduce accuracy slightly
- `--no-plot` (flag): Disable barcode rank plot generation
- `--gex-only` (flag, default: True): Use only Gene Expression data from multi-modal 10x files
- `--all-features` (flag): Use all features including ATAC-seq or other modalities (overrides --gex-only)

## Output Files

For an input file named `sample.h5`, the script generates the following output files:

### CSV Results (`sample_results.csv`)

A comma-separated values file containing EmptyDrops results with the following columns:

- `Total`: Total UMI count per barcode
- `PValue`: P-value from EmptyDrops test (only for tested barcodes)
- `FDR`: False Discovery Rate adjusted p-values using Benjamini-Hochberg method

**Note**: Only barcodes tested (with Total > `lower`) will have PValue and FDR values. Other barcodes will have NaN for these columns.

### H5AD File (`sample_results.h5ad`)

An AnnData object (HDF5 format) containing **only cells with FDR <= 0.05**:

- Filtered expression data (only cells passing FDR threshold)
- Results in `obs` (observations/barcodes) table:
  - `Total`: Total UMI count per barcode
  - `PValue`: EmptyDrops p-values
  - `FDR`: False Discovery Rate values
- Metadata in `uns['empty_drops']` dictionary:
  - `timestamp`: Analysis timestamp
  - `niters`: Number of Monte Carlo iterations used
  - `fdr_0_001`, `fdr_0_01`, `fdr_0_05`: Counts of barcodes with FDR <= 0.001, 0.01, 0.05
  - `calculated_retain`: Automatically calculated knee point
  - `lower`: Lower threshold used
  - `runtime_seconds`: Total runtime
  - `total_cells`: Total number of barcodes (before filtering)
  - `tested_cells`: Number of barcodes tested
  - `data_shape`: Original data dimensions (before filtering)
  - `version`: Algorithm version used
  - `batches_used`: Number of batches used for optimization
  - `reduction_factor`: Computational reduction factor achieved

**Note**: The H5AD file contains only cells with FDR <= 0.05 for easier downstream analysis. The CSV file contains all cells for reference.

### Metadata JSON (`sample_metadata.json`)

A JSON file containing all metadata from the analysis run, useful for tracking and reproducibility.

### Barcode Rank Plot (`sample_barcode_ranks.png`)

A log-log plot showing:

- Barcode ranks (x-axis, log10 scale)
- Total UMI counts (y-axis, log10 scale)
- Horizontal lines indicating:
  - Lower threshold (orange, dotted line)
  - Retain threshold/knee point (red, dashed line)

This visualization helps assess data quality and verify threshold selection.

## Algorithm Details

The EmptyDrops algorithm compares observed gene expression patterns against an ambient RNA profile constructed from low-count barcodes. The analysis proceeds through these steps:

1. **Data Preparation**: Filters genes with zero counts and identifies test vs. ambient barcodes
2. **Ambient Profile**: Calculates ambient RNA profile from barcodes with counts <= `lower`
3. **Observed Probabilities**: Computes log-probabilities for observed gene expression patterns
4. **Monte Carlo Simulation**: Generates simulated barcodes under the null hypothesis (ambient RNA only)
5. **P-value Calculation**: Compares observed vs. simulated probabilities
6. **FDR Correction**: Applies Benjamini-Hochberg correction for multiple testing
7. **Knee Point Detection**: Automatically identifies the transition point between real cells and empty droplets

## Performance

The batched implementation provides significant performance improvements:

- Up to 68x reduction in computational overhead through intelligent batching
- Typical runtime: ~2-3 minutes for 10,000 iterations on datasets with ~15,000 test barcodes
- Comparable accuracy to the R implementation within 1-80 cells difference

## Examples

### Standard Analysis

```bash
python run_empty_drops.py data/raw_feature_bc_matrix.h5
```

This will:
- Automatically detect the knee point
- Run 10,000 Monte Carlo iterations
- Generate all output files (CSV, H5AD, JSON, PNG)
- Save results in the current directory

### Quick Test Run

```bash
python run_empty_drops.py data/raw_feature_bc_matrix.h5 --niters 1000 --lower 200
```

Reduces iterations for faster testing and uses a higher lower threshold.

### Production Run with Custom Output

```bash
python run_empty_drops.py data/raw_feature_bc_matrix.h5 \
    --output-dir results/ \
    --output-prefix experiment1 \
    --niters 20000 \
    --max-batches 150
```

### Batch Processing

```python
import glob
from run_empty_drops import run_empty_drops

for h5_file in glob.glob("data/*.h5"):
    print(f"Processing {h5_file}...")
    run_empty_drops(
        h5_file,
        output_dir=f"results/{Path(h5_file).stem}/",
        niters=10000
    )
```

## Troubleshooting

### Insufficient Memory

If you encounter memory errors, try:
- Reducing `--max-batches` (though this may slightly reduce accuracy)
- Processing samples individually rather than in batch

### Knee Point Detection Fails

If automatic knee point detection fails (rare), you can:
- Manually specify `--retain` based on your data
- Check the barcode rank plot to visually identify the knee point
- Adjust `--lower` parameter if too many barcodes are filtered out

### Large Datasets

For very large datasets (>100,000 barcodes):
- The analysis may take longer (30+ minutes)
- Consider using `--niters 5000` for initial exploration
- Increase `--max-batches` if needed (though default 100 is usually optimal)

## Citation

If you use this EmptyDrops implementation, please cite:

Lun ATL, Riesenfeld S, Andrews T, Gomes T, Marioni JC (2019). "EmptyDrops: distinguishing cells from empty droplets in droplet-based single-cell RNA sequencing data." Genome Biology 20:63.

## License

[Add your license information here]

## Contact

[Add contact information here]

