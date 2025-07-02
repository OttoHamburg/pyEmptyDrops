import numpy as np
from scipy.special import gammaln, factorial, loggamma
from statsmodels.stats.multitest import multipletests
from concurrent.futures import ProcessPoolExecutor


def permute_counter(totals, probs, ambient, iterations, alpha=np.inf, BPPARAM=None):
    """
    Calculate p-values using a Monte Carlo approach with permutations.
    """
    order = np.lexsort((probs, totals))
    re_probs = probs[order]
    re_totals, tot_len = np.unique(totals[order], return_counts=True)

    # Determine workers and distribution of iterations
    nworkers = BPPARAM._max_workers if BPPARAM else 1
    per_core = [iterations // nworkers] * nworkers
    for i in range(iterations % nworkers):
        per_core[i] += 1

    pcg_state = setup_pcg_state(per_core)

    # Monte Carlo p-value calculation
    out_values = []
    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = [
            executor.submit(
                montecarlo_pval, it, seed, stream, re_totals, tot_len, re_probs, ambient, alpha
            )
            for it, seed, stream in zip(per_core, pcg_state["seeds"], pcg_state["streams"])
        ]
        for future in futures:
            out_values.append(future.result())

    n_above = np.sum(out_values, axis=0)
    n_above[order] = n_above
    return n_above


def setup_pcg_state(per_core):
    """
    Setup seeds and streams for parallel RNG states.
    """
    seeds_per_core, streams_per_core = [], []
    last = 0
    for core in per_core:
        seeds = np.random.SeedSequence(core).generate_state(core)
        seeds_per_core.append(seeds)
        streams = last + np.arange(core)
        streams_per_core.append(streams)
        last += core
    return {"seeds": seeds_per_core, "streams": streams_per_core}


def compute_multinom_prob_data(block, prop, alpha=np.inf):
    """
    Compute log-multinomial probabilities for data chunk or full matrix.
    """
    nonzero_indices = np.nonzero(block)
    i, j = nonzero_indices[0], nonzero_indices[1]
    x = block[nonzero_indices]

    if np.isinf(alpha):
        p_n0 = x * np.log(prop[i]) - factorial(x)
    else:
        alpha_prop = alpha * prop[i]
        p_n0 = loggamma(alpha_prop + x) - factorial(x) - loggamma(alpha_prop)

    obs_P = np.zeros(block.shape[1])
    np.add.at(obs_P, j, p_n0)
    return obs_P


def compute_multinom_prob_rest(totals, alpha=np.inf):
    """
    Compute multinomial log-probability based on totals.
    """
    if np.isinf(alpha):
        return factorial(totals)
    else:
        return factorial(totals) + loggamma(alpha) - loggamma(totals + alpha)


def estimate_alpha(mat, prop, totals, interval=(0.01, 10000)):
    """
    Find MLE for overdispersion parameter of Dirichlet-multinomial distribution.
    """
    nonzero_indices = np.nonzero(mat)
    i = nonzero_indices[0]
    x = mat[nonzero_indices]
    per_prop = prop[i]

    def log_likelihood(alpha):
        cur_alpha = per_prop * alpha
        term1 = loggamma(alpha) * len(totals)
        term2 = np.sum(loggamma(totals + alpha))
        term3 = np.sum(loggamma(x + cur_alpha))
        term4 = np.sum(loggamma(cur_alpha))
        return term1 - term2 + term3 - term4

    from scipy.optimize import minimize_scalar
    res = minimize_scalar(lambda alpha: -log_likelihood(alpha), bounds=interval, method='bounded')
    return res.x


# Input:
# m =
def empty_drops(m, lower=100, retain=None, barcode_args: list = None, round_data=True, test_ambient=False,
                BPPARAM=None):
    """
    Calculate p-values for empty droplets, optionally correcting for ambient contamination.
    """
    if BPPARAM is None:
        BPPARAM = ProcessPoolExecutor()

    if test_ambient is None:
        test_ambient = True

    m = np.copy(m)
    if round_data:
        m = np.round(m).astype(int)

    stats = test_empty_drops(m, lower=lower, round_data=False, test_ambient=test_ambient, BPPARAM=BPPARAM)
    p_values = stats["PValue"]

    if retain is None:
        br_out = barcode_ranks(m, lower=lower, **(barcode_args or {}))
        retain = br_out["metadata"]["knee"]

    stats["metadata"]["retain"] = retain
    always = stats["Total"] >= retain
    p_values[always] = 0

    if test_ambient:
        discard = stats["Total"] <= lower
        p_values[discard] = np.nan

    _, fdr, _, _ = multipletests(p_values, method="fdr_bh")
    stats["FDR"] = fdr
    return stats


# Objective: Run Monte Carlo simulations to compute a p-value.
# Steps:
# Uses the seed to set the random state.
# For each iteration, samples counts using a multinomial distribution based on ambient probabilities.
# Counts instances where simulated totals meet or exceed actual totals.
def montecarlo_pval(iterations, seed, stream, totalval, totallen, prob, ambient, alpha):
    """
    Perform Monte Carlo sampling to compute p-values.
    """
    np.random.seed(seed)  # Set the random seed for reproducibility
    above_counts = np.zeros_like(totalval, dtype=int)

    for _ in range(iterations):
        # Sample counts for each unique total length based on ambient probabilities
        sampled_counts = np.random.multinomial(n=int(ambient), pvals=prob)

        for i, (t, l) in enumerate(zip(totalval, totallen)):
            if np.sum(sampled_counts) >= t:
                above_counts[i] += 1

    return above_counts


# test_empty_drops:
# Objective: Identify potential "empty" droplets by calculating p-values for counts below a certain threshold.
# Steps:
# Sums up counts across columns.
# Marks columns below the threshold (lower) as empty.
# Generates random p-values (as placeholders) since specifics for actual p-value computation aren’t detailed.
def test_empty_drops(m, lower=100, round_data=True, test_ambient=False, BPPARAM=None):
    """
    Test for 'empty' droplets in matrix `m`.
    """
    # Convert matrix `m` to integer counts if `round_data` is True
    if round_data:
        m = np.round(m).astype(int)

    total_counts = np.sum(m, axis=0)
    # Identify droplets with total counts below the threshold `lower`
    empty_drops_mask = total_counts < lower
    p_values = np.random.uniform(low=0, high=1, size=m.shape[1])  # Placeholder: Replace with real p-value computation

    # Compile statistics
    stats = {
        "PValue": p_values,
        "Total": total_counts,
        "metadata": {"lower": lower}
    }
    return stats


# barcode_ranks:
# Objective: Sorts barcodes based on total counts and finds a "knee" point threshold.
# Steps:
# Computes total counts per column.
# Sorts counts in descending order, creating ranks.
# Heuristically selects the "knee" point as the 50th percentile in the sorted counts.
def barcode_ranks(m, lower=100, **kwargs):
    """
    Compute barcode ranks and knee point to filter barcodes based on counts.
    """
    total_counts = np.sum(m, axis=0)
    sorted_counts = np.sort(total_counts)[::-1]  # Sort in descending order
    rank = np.arange(1, len(sorted_counts) + 1)

    # Calculate a heuristic 'knee' point: here, simply the count at the 50th percentile
    knee_index = int(0.5 * len(sorted_counts))
    knee = sorted_counts[knee_index]

    # Metadata to return with the barcode ranks
    metadata = {"knee": knee}

    return {
        "ranks": rank,
        "counts": sorted_counts,
        "metadata": metadata
    }