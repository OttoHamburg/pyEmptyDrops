# decontX: foreground (ED) vs background (raw minus ED) without UMI thresholds

library(SingleCellExperiment)
library(DropletUtils)
library(celda)
library(zellkonverter)
library(Matrix)
library(HDF5Array)

# Paths (inside 11_decontX/data)
data_dir <- "./data"
raw_h5 <- file.path(data_dir, "raw.h5")
fg_h5ad <- file.path(data_dir, "foreground.h5ad")

# Load raw counts (master counts source)
sce_raw <- DropletUtils::read10xCounts(raw_h5)

if (is.null(colnames(sce_raw)) || length(colnames(sce_raw))==0) { colnames(sce_raw) <- as.character(SummarizedExperiment::colData(sce_raw)[["Barcode"]]) }
# Helper to harmonize barcodes
norm_bc <- function(x) sub("-1$", "", x)
raw_bc_norm <- norm_bc(colnames(sce_raw))

# Load foreground and background barcodes from H5AD (barcodes only)
fg_bc <- character(0)
if (file.exists(fg_h5ad)) {
  ad_fg <- zellkonverter::readH5AD(fg_h5ad)
  fg_bc <- norm_bc(colnames(ad_fg))
}

# Define foreground as raw ∩ foreground.h5ad
keep_fg <- raw_bc_norm %in% fg_bc
if (!any(keep_fg)) stop("No overlap between Foreground and raw barcodes.")
sce_fg <- sce_raw[, keep_fg, drop = FALSE]

# Optional prior clusters from Foreground H5AD (e.g., 'leiden')
pass_z <- NULL
if (exists("ad_fg")) {
  cd_cols <- colnames(SummarizedExperiment::colData(ad_fg))
  if ("leiden" %in% cd_cols) {
    m <- match(norm_bc(colnames(sce_fg)), norm_bc(colnames(ad_fg)))
    prior <- rep(NA_character_, ncol(sce_fg))
    prior[!is.na(m)] <- as.character(SummarizedExperiment::colData(ad_fg)$leiden[m[!is.na(m)]])
    sce_fg$prior_cluster <- prior
    pass_z <- "prior_cluster"
  }
}

# Run decontX with explicit background (no UMI thresholding)
set.seed(42)
if (is.null(pass_z)) {
  sce_out <- celda::decontX(sce_fg, background = sce_raw)
} else {
  sce_out <- celda::decontX(sce_fg, background = sce_raw, z = pass_z)
}

# Export decontaminated counts and H5AD
decon <- celda::decontXcounts(sce_out)
out_prefix <- "decontx_ED_bg_nothresh"
HDF5Array::saveHDF5SummarizedExperiment(sce_out, dir = paste0(out_prefix, "_SCEh5"), replace=TRUE)
try({ decon_csr <- as(Matrix::Matrix(decon, sparse=TRUE), "dgCMatrix"); Matrix::writeMM(decon_csr, paste0(out_prefix, "_matrix.mtx")) }, silent=TRUE)
  zellkonverter::writeH5AD(sce_out, paste0(out_prefix, ".h5ad"), X_name = "decontXcounts")
