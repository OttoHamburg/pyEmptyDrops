import scanpy as sc
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.stats import multinomial
from collections import Counter
from statsmodels.stats.multitest import multipletests
from scipy.special import gammaln
from scipy.optimize import minimize_scalar



"""
Functionality:  Checks a CSR matrix for float values.
                Prompts the user to continue or exit if floats are found.
Returns:        True/False.
"""
def check_matrix_contain_floats(m):
    try:
        check_values(m)
    except ValueError as e:
        # Catch the specific ValueError raised
        print(e)
        response = input("Do you want to continue processing? (yes/no) otherwise type (int) if you want to convert all floats to integers: ")
        if response.lower() == 'yes':
            print("Continuing with processing...")
            return True, m
        elif response.lower() == 'int':
            m = m.astype(int)  # Convert all float values to integers
            print("Converted matrix to integers.")
            return True, m  # Return the converted matrix
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



"""
Distinguish between droplets containing cells and ambient RNA in a droplet-based single-cell RNA sequencing experiment.
------ INPUT ------
m             = A numeric matrix object - usually a dgTMatrix or dgCMatrix - containing droplet data prior to
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


def empty_drops(data, lower=100, ignore=None, niters=10000, fdr=0.001):
    """Run the EmptyDrops procedure."""
    m = data.X
    gene_sums_mask = m.sum(axis=0).A1 == 0
    m = m[:, ~gene_sums_mask]
    print(f"{gene_sums_mask.sum()} genes filtered out as their sum across all barcodes is zero.")

    ambient_proportions = ambient_profile(m, lower)

    if ignore is not None:
        print(f"Ignoring barcodes with total counts ≤ {ignore}.")
        high_count_mask = (m.sum(axis=1).A1 > lower) & (m.sum(axis=1).A1> ignore)
    else:
        high_count_mask = m.sum(axis=1).A1 > lower

    results = test_empty_drops(m[high_count_mask], ambient_proportions, niters=niters)

    # Adjust p-values
    adjusted_pvalues = multipletests(results['PValue'], method='fdr_bh', alpha=fdr)[1]
    results['AdjPValue'] = adjusted_pvalues
    results['IsCell'] = results['AdjPValue'] < fdr
    return results


def test_empty_drops(m, ambient_proportions, niters=10000):
    """Test each barcode for significant deviation from the ambient profile."""
    results = []
    total_counts = m.sum(axis=1).A
    for idx in range(m.shape[0]):
        count_vector = m[idx].A
        if count_vector.sum() == 0:  # Skip empty rows
            continue

        # Compute observed multinomial probability
        ambient_vector = np.array([ambient_proportions.get(c, 0) for c in count_vector])
        ambient_vector /= ambient_vector.sum()
        observed_prob = multinomial.pmf(count_vector, count_vector.sum(), ambient_vector)

        # Simulate null probabilities
        simulated_probs = []
        for _ in range(niters):
            sim_vector = np.random.multinomial(count_vector.sum(), ambient_vector)
            sim_prob = multinomial.pmf(sim_vector, count_vector.sum(), ambient_vector)
            simulated_probs.append(sim_prob)

        p_value = np.mean([sim <= observed_prob for sim in simulated_probs])
        results.append({'Barcode': idx, 'TotalCounts': total_counts[idx], 'PValue': p_value})

    return pd.DataFrame(results)


def ambient_profile(m, lower, by_rank = None, good_turing = True):
    """
    Computes proportions for low-count sums in an HDF5 sparse matrix.

    Parameters:
    m : scipy.sparse matrix
        The input sparse matrix.
    lower : int
        Threshold for determining low-count rows based on their sum.
    by_rank : None or array-like, optional
        Ignored in this implementation, but kept for potential future extensions.
    good_turing : bool, optional
        Whether to apply Good-Turing smoothing to proportions.

    Returns:
    np.ndarray
        Array of proportions computed from low-count sums.
    """
    low_count_mask = np.array(m.sum(axis=1).A1 <= lower).flatten()
    low_count_sums = m[low_count_mask].sum(axis=0).A1
    if good_turing:
        proportions = good_turing_proportions(low_count_sums)
    else:
        proportions = low_count_sums / low_count_sums.sum()
    return proportions



def good_turing_proportions(counts):
    frequency_distribution = Counter(counts)
    n = sum(frequency_distribution.values())
    proportions = {}

    for r in range(max(counts) + 1):
        n_r = frequency_distribution.get(r, 0)
        n_r_plus_1 = frequency_distribution.get(r + 1, 0)
        if n_r > 0:
            proportions[r] = (r + 1) * n_r_plus_1 / n
    proportions[0] = frequency_distribution.get(1, 0) / n
    total_proportion = sum(proportions.values())
    for r in proportions:
        proportions[r] /= total_proportion
    return proportions





def compute_multinom_prob_rest(totals, alpha=np.inf):
    """
    Efficiently calculates the total-dependent component of the multinomial log-probability.

    Parameters:
    totals : np.ndarray
        Array of total counts.
    alpha : float, optional
        The dispersion parameter. Default is infinity (no dispersion).

    Returns:
    np.ndarray
        The computed multinomial log-probabilities for the given totals and alpha.
    """
    # Check for infinite alpha (no dispersion case)
    if np.isinf(alpha):
        # Compute log-factorial of totals
        return gammaln(totals + 1)
    else:
        # Compute the log-probability component for given totals and alpha
        return gammaln(totals + 1) + gammaln(alpha) - gammaln(totals + alpha)



def estimate_alpha(h5_sparse_matrix, prop, totals, interval=(0.01, 10000)):
    """
    Efficiently finds the MLE for the overdispersion parameter of a Dirichlet-multinomial distribution
    using an HDF5 sparse matrix as input.

    Parameters:
    h5_sparse_matrix : scipy.sparse.csr_matrix
        The input HDF5 sparse matrix.
    prop : np.ndarray
        Proportions array corresponding to the rows of the matrix.
    totals : np.ndarray
        Array of total counts.
    interval : tuple, optional
        Interval for optimization (default: (0.01, 10000)).

    Returns:
    float
        The estimated alpha value (MLE).
    """

    # Ensure matrix is in CSR format for efficient row access
    if not isinstance(h5_sparse_matrix, csr_matrix):
        h5_sparse_matrix = csr_matrix(h5_sparse_matrix)

    # Extract non-zero indices and values
    row_indices, col_indices = h5_sparse_matrix.nonzero()
    nz_vals = np.array(h5_sparse_matrix.data)

    # Map proportions to the non-zero rows
    per_prop = prop[row_indices]

    def log_likelihood(alpha):
        """
        Computes the log-likelihood for a given alpha.

        Parameters:
        alpha : float
            The current value of alpha.

        Returns:
        float
            The log-likelihood for the given alpha.
        """
        if alpha <= 0:
            return -np.inf  # Log-likelihood undefined for non-positive alpha

        prop_alpha = per_prop * alpha
        log_lik = (
                gammaln(alpha) * len(totals)
                - np.sum(gammaln(totals + alpha))
                + np.sum(gammaln(nz_vals + prop_alpha))
                - np.sum(gammaln(prop_alpha))
        )
        return log_lik

    # Optimize to find the alpha that maximizes the log-likelihood
    result = minimize_scalar(
        lambda alpha: -log_likelihood(alpha),  # Minimize negative log-likelihood
        bounds=interval,
        method='bounded'
    )

    return result.x



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