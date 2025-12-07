.rounded_to_integer <- function(m, round=TRUE) {
    if (round) {
        m <- round(m)
    }
    m
}

#' @importFrom Matrix colSums
.intColSums <- function(m) {
    # Enforcing discreteness mainly for emptyDrops()'s Monte Carlo step.
    as.integer(round(colSums(m)))
}

#' @importClassesFrom SparseArray COO_SparseArray
#' @importFrom DelayedArray setAutoBPPARAM DelayedArray
#' @importClassesFrom DelayedArray DelayedArray
.realize_DA_to_memory <- function(m, BPPARAM) {
    if (is(m, "DelayedArray")) {
        if (!is(m@seed, "COO_SparseArray")) {
            # Coercion from DelayedArray to COO_SparseArray uses block
            # processing.
            old <- .parallelize(BPPARAM)
            on.exit(setAutoBPPARAM(old))
            new_seed <- as(m, "COO_SparseArray")

            m <- DelayedArray(new_seed)
        }
    }

    m
}

#' @importFrom DelayedArray getAutoBPPARAM setAutoBPPARAM
.parallelize <- function(BPPARAM) {
    old <- getAutoBPPARAM()
    setAutoBPPARAM(BPPARAM)
    old
}
