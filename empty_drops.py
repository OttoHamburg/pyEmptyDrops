import scanpy as sc
import numpy as np
from scipy.sparse import csr_matrix
import scipy.stats as ss
from collections import Counter






"""
Functionality:  Checks a CSR matrix for float values.
                Prompts the user to continue or exit if floats are found.
Returns:        True/False.
"""
def check_count_matrix_values(m):
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
def empty_drops(m, lower=100, retain=None, barcode_args: list = None, test_ambient=False):

    stats = test_empty_drops(m, lower=lower, test_ambient=test_ambient)





""" 
obtains an estimate of the composition of the ambient pool of RNA based on the barcodes with total UMI counts less than or equal to lower
"""
def test_empty_drops(m, lower=100, niters = 10000, test_ambient = False, ignore = None, alpha = None, round = True, by_rank = None):
    # obtains an estimate of the composition of the ambient pool of RNA based on the barcodes with total UMI counts less than or equal to lower
    # Counts for the low-count barcodes are pooled together, and an estimate of the proportion vector for the ambient pool is calculated using goodTuringProportions
    ambient_proportions = estimateAmbience(m, lower)

    # Test for significant deviation from ambient proportions for each barcode above lower
    high_count_mask = m.sum(axis=1).A1 > lower  # Mask for high-count barcodes
    high_count_barcodes = m[high_count_mask]

    low_count_mask = m.sum(axis=1).A1 <= lower  # Mask for high-count barcodes
    low_count_barcodes = m[low_count_mask]
    low_count_mean = m[low_count_mask].mean(axis=0)
    low_count_prop = low_count_mean / low_count_mean.sum()

    significant_deviations = {}

    for i in range(high_count_barcodes.shape[0]):
        barcode_counts = high_count_barcodes[i].A  # Get counts for the current barcode
        expected_counts = ambient_proportions * barcode_counts.sum()
        deviation = np.abs(barcode_counts - expected_counts)

        if np.any(deviation > expected_counts * 0.1):  # check if 10% deviation is significant
            significant_deviations[i] = deviation

    # wir haben nun high_count barcodes gegen deviation getestet, teste NUN *ALLE BARCODES* gegen deviations
    # the null model assumes that the counts in each barcode (bzw. droplet) originate solely from the ambient RNA pool,
    # meaning they are fully random and representative of the background noise.

    # For each barcode, the probability of obtaining its count vector based on the null model is computed.
    log_probs = np.array([ss.multinomial.logpmf(m[:, i].A1, n=m[:, i].sum(), p=prop)
                          for i in range(m.shape[1])])

    # Then, niters count vectors
    # are simulated from the null model. The proportion of simulated vectors with probabilities lower than the observed multinomial
    # probability for that barcode is used to calculate the p-value.

    # We use this Monte Carlo approach as an exact multinomial p-value is difficult to calculate. However, the p-value is lower-bounded
    # by the value of niters (Phipson and Smyth, 2010), which can result in loss of power if niters is too small. Users can check whether
    # this loss of power has any practical consequence by checking the Limited field in the output. If any barcodes have Limited=TRUE but
    # does not reject the null hypothesis, it suggests that niters should be increased.

    # The stability of the Monte Carlo $p$-values depends on niters, which is only set to a default of 10000 for speed. Larger values
    # improve stability with the only cost being that of time, so users should set niters to the largest value they are willing to wait for.

    # The ignore argument can also be set to ignore barcodes with total counts less than or equal to ignore. This differs from the lower
    # argument in that the ignored barcodes are not necessarily used to compute the ambient profile. Users can interpret ignore as the
    # minimum total count required for a barcode to be considered as a potential cell. In contrast, lower is the maximum total count below
    # which all barcodes are assumed to be empty droplets.

    # Implement Monte Carlo approach to compute p-values for each barcode based on the multinomial sampling
    p_values = {}

    # Generate niters simulated count vectors from the ambient proportions
    monte_carlo_counts = np.random.multinomial(m.sum(), ambient_proportions, size=niters)

    for idx, deviation in significant_deviations.items():
        barcode_counts = high_count_barcodes[idx].A.squeeze()
        # Observing count vector in the multinomial distribution
        obs_ll = ss.multinomial.pmf(barcode_counts, n=barcode_counts.sum(), p=ambient_proportions)

        simulated_ll = np.array([
            ss.multinomial.pmf(sim_counts, n=sim_counts.sum(), p=ambient_proportions)
            for sim_counts in monte_carlo_counts
        ])

        # Calculate p-value by comparing observed log-likelihood to the simulated log-likelihoods
        p_value = np.sum(simulated_ll < obs_ll) / niters
        p_values[idx] = p_value


    return significant_deviations



""" 
Estimate the composition of the ambient pool of RNA based on barcodes with total UMI counts less than or equal to 'lower'.
    Parameters:
        m (csr_matrix): A CSR matrix containing RNA count data.
        lower (float): The threshold for low UMI counts.
    Returns:
        np.ndarray: Estimated proportions for each transcript in the ambient pool. 
"""
def estimateAmbience(m, lower, by_rank = None, good_turing = True):
    # creates mask for low-count barcodes (finds every cell with less than lower active genes)
    low_count_mask = m.sum(axis=1).A1 <= lower #axis=1 summing every value in row
    # Sum counts for each gene (axis=0/column) but only for low-count barcodes (-> in low_count_mask rows labeled as True)
    low_count_sums = m[low_count_mask].sum(axis=0)
    # Estimate proportions using Good-Turing
    proportions = goodTuringProportions(low_count_sums)
    return proportions


""" 
Estimate Good-Turing proportions based on counts.
    Parameters:
        counts (np.ndarray): A 1D array of counts for each gene (but only for barcodes with total <lower)
    Returns:
        np.ndarray: Estimated proportions for each gene. 
"""
def goodTuringProportions(counts):
    # flatten the 1x22040 2D np-matrix since it let to errors in hasing using "Counter"
    if isinstance(counts, np.matrix):
        counts = np.asarray(counts)

    counts = counts.flatten() if counts.ndim > 1 else counts

    total_counts = counts.sum()
    # Handle case where total_counts is zero
    if total_counts == 0:
        # create array full of zeros shaped like counts/low_count_sums array
        return np.zeros_like(counts)
    # Calculate frequency distribution (sorted increasing)
    frequency_distribution = Counter(counts)
    # Total number of unique items
    n = sum(frequency_distribution.values())
    # Initialize Good-Turing proportions dictionary
    good_turing_proportions = {}

    # Calculate Good-Turing proportions
    for r in range(max(counts) + 1):
        n_r = frequency_distribution.get(r, 0)  # Number of items with count r
        n_r_plus_1 = frequency_distribution.get(r + 1, 0)  # Number of items with count r + 1

        if n_r > 0:  # Only calculate for observed frequencies
            good_turing_proportions[r] = (r + 1) * n_r_plus_1 / n

    # Handle unseen events (count = 0) => common praxis = N1/N
    good_turing_proportions[0] = frequency_distribution.get(1, 0) / n

    # rreate an array for hold results
    results = np.zeros_like(counts)

    # fill in the results regarding the proportions
    for r in range(len(results)):
        if r in good_turing_proportions:
            results[r] = good_turing_proportions[r]

    return results



def main():
    raw_h5_path = '../01_Daten/raw_feature_bc_matrix.h5'
    raw_adata = sc.read_10x_h5(raw_h5_path, gex_only=True)
    filtered_h5_path = '../01_Daten/filtered_feature_bc_matrix.h5'
    filtered_adata = sc.read_10x_h5(filtered_h5_path, gex_only=True)
    print("LOADED MATRICES")

    go_on, m = check_count_matrix_values(raw_adata.X)
    if not go_on:
        print("EXIT PROGRAM")
        return 0
    else:
        print("START EMPTY_DROPS")
        empty_drops(m) #only pass the counts to the function


main()








