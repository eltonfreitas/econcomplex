"""
Fitness-Complexity method (non-linear iterative).

References
----------
Tacchella et al. (2012) "A New Metrics for Countries' Fitness and Products' Complexity".
Cristelli et al. (2013).
"""

import warnings

import numpy as np
import pandas as pd
from typing import Tuple, Union

from ..core.utils import validate_matrix, safe_divide, binarize
from ..core.rca import rca as compute_rca


def fitness_complexity(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    iterations: int = 20,
    extremality: float = 1.0,
    tol: float = 1e-10,
    log_fitness: bool = False,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Fitness-Complexity algorithm.

    Iterative update rules (normalized at each step):
      F_r^{(n)} = sum_c M_{rc} * Q_c^{(n-1)}
      Q_c^{(n)} = 1 / ( sum_r M_{rc} * (1/F_r^{(n-1)})^alpha )^{1/alpha}

    where alpha = `extremality` (default 1).

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    iterations : int
        Maximum iterations (default 20, matching the R `economiccomplexity`
        package). The loop stops earlier as soon as `tol` is reached.
        At 20 iterations the algorithm is practically converged on typical
        data (relative change ~1e-7); a RuntimeWarning is issued only when
        the final relative change still exceeds 1e-3, which signals real
        instability (e.g. oscillation on pathological matrices).
    extremality : float
        Non-linearity parameter alpha (default 1.0).
    tol : float
        Convergence tolerance on relative change.
    log_fitness : bool
        If True, return natural log of fitness and complexity
        (Cristelli et al. 2015 recommend the log scale for analysis).
        Zeros are returned as NaN.

    Returns
    -------
    (fitness, complexity) as pd.Series or ndarrays.
    fitness = region/country fitness score.
    complexity = product/activity complexity score.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    arr = validate_matrix(mat)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    n_r, n_c = m.shape
    fitness = np.ones(n_r)
    complexity = np.ones(n_c)

    converged = False
    delta_f = delta_q = np.inf
    for _ in range(iterations):
        fitness_new = m @ complexity
        # Normalize by mean
        mean_f = fitness_new.mean()
        if mean_f > 0:
            fitness_new /= mean_f

        # Q update with extremality
        inv_f = safe_divide(1.0, fitness_new ** extremality)
        q_denom = (m.T @ inv_f) ** (1.0 / extremality)
        complexity_new = safe_divide(1.0, q_denom)
        mean_q = complexity_new.mean()
        if mean_q > 0:
            complexity_new /= mean_q

        # Convergence check
        delta_f = np.max(np.abs(fitness_new - fitness)) / (np.max(np.abs(fitness)) + 1e-15)
        delta_q = np.max(np.abs(complexity_new - complexity)) / (np.max(np.abs(complexity)) + 1e-15)

        fitness = fitness_new
        complexity = complexity_new

        if delta_f < tol and delta_q < tol:
            converged = True
            break

    if not converged and (delta_f > 1e-3 or delta_q > 1e-3):
        warnings.warn(
            f"fitness_complexity did not converge within {iterations} "
            f"iterations (final relative change {max(delta_f, delta_q):.1e}); "
            "results may be unstable. Increase `iterations`.",
            RuntimeWarning,
            stacklevel=2,
        )

    if log_fitness:
        with np.errstate(divide="ignore", invalid="ignore"):
            fitness = np.where(fitness > 0, np.log(fitness), np.nan)
            complexity = np.where(complexity > 0, np.log(complexity), np.nan)

    if is_df:
        return (
            pd.Series(fitness, index=row_index, name="fitness"),
            pd.Series(complexity, index=col_index, name="complexity"),
        )
    return fitness, complexity
