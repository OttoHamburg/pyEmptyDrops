import scanpy as sc
import numpy as np
from fontTools.subset.svg import update_glyph_href_links
from numba import typeof
from scipy.sparse import csr_matrix
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gamma, gammaln, factorial  # gamma function and Log of the gamma function
import scipy.stats as ss
from collections import Counter
from tqdm import tqdm


"""
Functionality:  Checks a CSR matrix for float values.
                Prompts the user to continue or exit if floats are found.
Returns:        True/False.
"""
def check_matrix_contain_floats(matrix):
    try:
        check_values(matrix)
    except ValueError as e:
        # Catch the specific ValueError raised
        print(e)
        #response = input("Do you want to continue processing? (yes/no) otherwise type (int) if you want to convert all floats to integers: ")
        response = 'int'
        if response.lower() == 'yes':
            print("Continuing with processing...")
            return True, matrix
        elif response.lower() == 'int':
            matrix = matrix.astype(int)  # Convert all float values to integers
            print("Converted matrix to integers.")
            return True, matrix  # Return the converted matrix
        else:
            print("Exiting process.")
            return False, None

"""
Functionality:  Counts float values in the count_matrix and raises a ValueError
Returns:        if any floats found, detailing the count and an example.
"""
def check_values(matrix):
    float_count = 0
    if isinstance(matrix, csr_matrix):
        # count floats in matrix
        for value in matrix.data:
            if isinstance(value, np.float32):
                float_count += 1
        # throw error if previously found a float and print the first one found
        if float_count >= 1:
            for value in matrix.data:
                if isinstance(value, np.float32):
                    raise ValueError(f"Found a total of {float_count} floats in the counts_matrix, for example: {value}.")
    else:
        print("The provided matrix is not a CSR matrix.")


def good_turing_ambient_pool(data, low_count_gene_sums):
    # flatten the 1x22040 2D np-matrix since it led to errors in hashing using "Counter"
    counts = np.asarray(low_count_gene_sums)
    counts = counts.flatten() if counts.ndim > 1 else counts

    # Calculate frequency distribution (sorted increasing)
    frequency_distribution = Counter(counts)
    # Total number of unique items
    n = sum(frequency_distribution.values())
    # Initialize Good-Turing proportions dictionary
    good_turing_proportions = {}

    # Calculate Good-Turing proportions to avoid having genes with zero counts in the ambient pool
    for r in tqdm(range(max(counts) + 1), desc="Calculating Good-Turing Proportions"):
        n_r = frequency_distribution.get(r, 0)  # Number of items with count r
        n_r_plus_1 = frequency_distribution.get(r + 1, 1)  # Number of items with count r + 1

        if n_r > 0:  # Only calculate for observed frequencies
            # habe jetzt einfach n_r_plus_1 auf =1 gesetzt falls es nicht ex., sonst vielleicht auch einfach and n_r_plus_1 > 0 fordern?
            good_turing_proportions[r] = (r + 1) * n_r_plus_1 / n

    # Handle unseen events (count = 0) => common practice = N1/N
    good_turing_proportions[0] = frequency_distribution.get(1, 0) / n #ändert in dem fall nix, da erstes N0 = 0, somit (0+1)*N1 und somit N1/N gilt

    # Normalize the proportions to sum to 1
    total_proportion = sum(good_turing_proportions.values())
    for r in tqdm(good_turing_proportions, desc="Normalizing Proportions"):
        good_turing_proportions[r] /= total_proportion

    # Ensure normalization is correct
    normalized_total = sum(good_turing_proportions.values())
    assert abs(normalized_total - 1.0) < 1e-6, f"Proportions do not sum to 1: {normalized_total}"

    # Create an array to hold results
    gene_expectations = np.zeros_like(counts, dtype=float)
    # Fill in the results regarding the proportions
    ambient_proportions = {}

    # Get all gene_names
    gene_names = data.var_names

    for i, count in tqdm(enumerate(counts), desc="Filling Ambient Proportions", total=len(counts)):
        proportion = good_turing_proportions.get(count, 0)
        gene_expectations[i] = proportion
        ambient_proportions[gene_names[i]] = proportion

    return gene_names, ambient_proportions, gene_expectations


def estimate_alpha(ambient_set_g, gene_proportions):
    """
    Estimates alpha by maximizing the Dirichlet-Multinomial likelihood.

    # we need to compute the log-likelihood for all barcodes in the ambient set G, by summing the log L_b values for each barcode

    Parameters:
        counts (numpy.ndarray): Gene-by-barcode count matrix (G x B).
        proportions (numpy.ndarray): Ambient RNA proportions (G,).

    Returns:
        float: Estimated alpha.
    """
    alpha = 1.0 # initial guessed alpha
    total_log_likelihood = 0
    product = [1.0]

    batch_size = ambient_set_g.shape[0] // 100

    # iterate over all barcodes
    for i in tqdm(range(0, ambient_set_g.shape[0] - 1, batch_size), desc="Estimate (global) alpha", position=0, leave=True):
        # batch extraction
        batch = ambient_set_g[i: i+ batch_size]

        # p_g = proportions
        total_counts = batch.X.sum(axis=0).A1
        # reduce dimension of total_counts

        y_gb = np.asarray(batch.X.A)  # .flatten() #convert sparse matrix to dense array for calculations
        #alpha_p_g = gene_proportions * alpha
        # product = product * np.log( gamma(y_gb + alpha_p_g) / ( factorial(y_gb) * gamma(alpha_p_g) ) )

        product = product * np.log((gamma(y_gb + gene_proportions)) / (factorial(y_gb) * (gene_proportions)))
    l_b = (factorial(total_counts) * gamma(alpha)) / np.log(gamma(total_counts + alpha) * product)

    # Convert L_b to log-space and sum
    l_b = np.nanmean(l_b)
    total_log_likelihood += np.log(l_b)
    print(f"total_log_likelihood: {total_log_likelihood}")

    return -total_log_likelihood # Return negative for minimization



def maximum_likelihood_estimate(alpha, batch_barcodes, p_g, batch_size=10000):
    # p_g = proportions
    total_counts = batch_barcodes.X.sum(axis=0).A1
    # reduce dimension of total_counts
    product = 1.0

    y_gb = np.asarray(batch_barcodes.X.A)#.flatten() #convert sparse matrix to dense array for calculations
    alpha_p_g = p_g * alpha
    #product = product * np.log( gamma(y_gb + alpha_p_g) / ( factorial(y_gb) * gamma(alpha_p_g) ) )
    product = product * ( np.log(gamma(y_gb + alpha_p_g)) / (factorial(y_gb) * (alpha_p_g)) )
    l_b = ( factorial(total_counts) * gammaln(alpha) ) / np.log( gamma(total_counts + alpha) * product )

    return l_b

    # total_counts = data.X[barcode].sum()
    # product = 1
    # rows, cols = data.X[barcode].nonzero()
    #
    # for gene_idx in cols:
    #     y_gb = data.X[barcode, gene_idx]
    #
    #     gene_name = data.var_names[gene_idx]
    #     p_g = ambient_proportions[gene_name]
    #
    #     alpha_p_g = alpha * p_g
    #
    #     if y_gb== 0:
    #         print("y_gb: ", y_gb)
    #     if p_g == 0:
    #         print("p_g: ", p_g)
    #
    #     # log reingefuchst, da sonst das produkt der MLE immer kleiner wird und so gegen 0 geht
    #     product = product * log( gamma(y_gb + alpha_p_g) / ( factorial(y_gb) * gamma(alpha_p_g) )
    #                         )
    #
    # l_b = ( factorial(total_counts) * gamma(alpha) ) / ( gamma(total_counts + alpha) * product )
    #
    # return l_b



"""
Distinguish between droplets containing cells and ambient RNA in a droplet-based single-cell RNA sequencing experiment.
------ INPUT ------
data.X             = A numeric matrix object - usually a dgTMatrix or dgCMatrix - containing droplet data prior to
                any filtering or cell calling. Columns represent barcoded droplets, rows represent genes.
lower         = numeric scalar specifying the lower bound on the total UMI count
retain        = Barcodes that contain more than retain total counts are always retained. This ensures that large
                cells with profiles that are very similar to the ambient pool are not inadvertently discarded.
                If retain is not specified, it is set to the total count at the knee point detected by
               barcodeRanks. Manual specification of retain may be useful if the knee point was not correctly
                identified in complex log-rank curves. Users can also set retain=Inf to disable automatic#                 retention of barcodes with large totals.
barcode_args  = Further arguments to pass to barcodeRanks.
test_ambient  = A logical scalar indicating whether results should be returned for barcodes with totals less
                than or equal to lower.
------ OUTPUT ------
emptyDrops will return a DataFrame with the following components:
      Total:   Integer, the total UMI count for each barcode.
      LogProb: Numeric, the log-probability of observing the barcode's count vector under the null model.
      PValue:  Numeric, the Monte Carlo p-value against the null model.
      Limited: Logical, indicating whether a lower p-value could be obtained by increasing niters.
      FDR:     field
The metadata of the output DataFrame will contains the ambient profile in ambient, the estimated/specified value
of alpha, the specified value of lower (possibly altered by use.rank) and the number of iterations in niters.
For emptyDrops, the metadata will also contain the retention threshold in retain.
"""
def empty_drops(data, lower=100, retain=None, barcode_args: list = None, test_ambient=False):
    # remove all genes which aren't representif in one barcode
    print(f"{(data.X.sum(axis=0).A1 == 0).sum()} genes filtered out since sum(counts) over the gene was 0.")
    sc.pp.filter_genes(data, min_counts=1)

    low_count_barcode_mask = data.X.sum(axis=1).A1 <= 50  # axis=1 summing every value in row
    high_count_barcodes = data[low_count_barcode_mask==False]
    # Sum counts for each gene (axis=0/column) but only for low-count barcodes (-> in low_count_mask rows labeled as True)
    low_count_gene_sums = data.X[low_count_barcode_mask].sum(axis=0)

    print


    # ----------------------
    # CREATE AMBIENT PROFILE
    # ----------------------
    # gene_expectations is np array with 20025 genes, ambient_proportions is dict with genes and their expectation
    gene_names, ambient_proportions, gene_expectations = good_turing_ambient_pool(data, low_count_gene_sums)

    # calculate global alpha out of low_count_barcodes
    ambient_set_g = data[low_count_barcode_mask]
    alpha = estimate_alpha(ambient_set_g, gene_expectations)
    # dauert knapp 47 Stunden


    # calculate likelihoods using global alpha for high_count_barcodes
    likelihoods = []
    for idx, barcode in enumerate (high_count_barcodes):
        likelihood = maximum_likelihood_est

    # test for barcode 2
    likelihood_barcode = maximum_likelihood_estimate(1, 2, ambient_proportions, data)
    print("likelihood_barcode 2: ", likelihood_barcode)




    # log_probs = np.array([ss.multinomial.logpmf(data.X[:, i], n=data.X[:, i].sum(), p=0.1) for i in range(data.X.shape[1])])

    # significant_deviations = compute_significant_deviations(high_count_barcodes, ambient_proportions, gene_names)
    #print(f"Identified {len(significant_deviations)} barcodes with significant deviations.")

    # wir haben nun high_count barcodes gegen deviation getestet, teste NUN *ALLE BARCODES* gegen deviations
    # the null model assumes that the counts in each barcode (bzw. droplet) originate solely from the ambient RNA pool,
    # meaning they are FULLY random (+++representative of the background noise).

    # For every barcode, the probability of obbtaining the specific count vector based on the null model is computed.


def main():
    raw_h5_path = '../01_Daten/raw_feature_bc_matrix.h5'
    raw_adata = sc.read_10x_h5(raw_h5_path, gex_only=True)
    filtered_h5_path = '../01_Daten/filtered_feature_bc_matrix.h5'
    filtered_adata = sc.read_10x_h5(filtered_h5_path, gex_only=True)
    print("LOADED MATRICES")

    go_on, m = check_matrix_contain_floats(raw_adata.X)

    # assign the new count matrix with floats changed to ints (if user wanted)
    raw_adata.X = m

    if not go_on:
        print("EXIT PROGRAM")
        return 0
    else:
        print("START EMPTY_DROPS")
        empty_drops(raw_adata)


main()