library(SingleCellExperiment); library(DropletUtils); library(zellkonverter); library(celda)

# Load raw and set barcodes
sce_raw <- DropletUtils::read10xCounts("00_data/raw.h5")
if (is.null(colnames(sce_raw)) || length(colnames(sce_raw))==0)
  colnames(sce_raw) <- as.character(SummarizedExperiment::colData(sce_raw)[["Barcode"]])

# Foreground barcodes
ad_fg <- zellkonverter::readH5AD("00_data/foreground.h5ad")
norm <- function(x) sub("-1$","", x)
keep_fg <- norm(colnames(sce_raw)) %in% norm(colnames(ad_fg))
sce_fg <- sce_raw[, keep_fg, drop=FALSE]

# Read cluster_type produced in Python
lab <- read.csv("51_annot_cells_types.csv", row.names=1, check.names = FALSE)
m <- match(norm(colnames(sce_fg)), norm(rownames(lab)))
sce_fg$cluster_annotation <- factor(lab$cluster_annotation[m])

if (any(is.na(sce_fg$cluster_annotation))) stop("Some barcodes did not match labels. Check CSV/barcodes.")

set.seed(42)
sce_out <- celda::decontX(sce_fg, ,background = sce_raw, z = "cluster_annotation")

# Save
dir.create("53_results_clusters", showWarnings=FALSE)
HDF5Array::saveHDF5SummarizedExperiment(sce_out, dir="53_results_clusters/decontx_with_clusters_SCEh5", replace=TRUE)
zellkonverter::writeH5AD(sce_out, "53_results_clusters/decontx_with_clusters.h5ad", X_name="decontXcounts")