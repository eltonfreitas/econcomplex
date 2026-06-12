"""
ECI / PCI — single entry point for all complexity methods.

`eci_pci(mat, method=...)` is the recommended way to compute economic
complexity with this library. It dispatches between the three methods
(mirroring `complexity_measures()` of the R `economiccomplexity` package),
pre-trims degenerate units, and returns results aligned with the input.

The underlying implementations remain available for advanced use:
`eci_pci_eigenvector` (module `eigenvector`), `method_of_reflections`
(module `reflections`), and `fitness_complexity` (module `fitness`).
"""

import numpy as np
import pandas as pd
from typing import Literal, Optional, Tuple, Union

from .eigenvector import eci_pci_eigenvector
from .reflections import method_of_reflections
from .fitness import fitness_complexity


def eci_pci(
    mat: Union[np.ndarray, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    method: Literal["eigenvector", "reflections", "fitness"] = "eigenvector",
    iterations: Optional[int] = None,
    extremality: float = 1.0,
    tol: float = 1e-10,
    log_fitness: bool = False,
    trim: bool = True,
    dmin: int = 1,
    umin: int = 1,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:
    """
    Economic Complexity Index (ECI) and Product Complexity Index (PCI).

    Single entry point for the three complexity methods (mirrors the
    `complexity_measures()` interface of the R `economiccomplexity`
    package):

    - 'eigenvector' (default): second eigenvector of the Markov-style
      co-occurrence matrices (Hidalgo & Hausmann 2009, OEC Atlas form).
      ECI/PCI are z-score normalized, sign-corrected so that ECI
      correlates positively with diversity and PCI negatively with
      ubiquity.
    - 'reflections': iterative Method of Reflections
      (delegates to `method_of_reflections`).
    - 'fitness': non-linear Fitness-Complexity algorithm of
      Tacchella et al. (2012) (delegates to `fitness_complexity`;
      returns raw fitness/complexity scores, not z-scores).

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    method : str
        'eigenvector', 'reflections', or 'fitness'.
    iterations : int, optional
        Iterations for 'reflections' and 'fitness' (default 20 for both,
        matching the R `economiccomplexity` package; for 'fitness' it is
        a cap — the loop stops at convergence and warns if the cap is hit
        first). Ignored by 'eigenvector'.
    extremality : float
        Non-linearity parameter alpha for 'fitness' (default 1.0).
    tol : float
        Convergence tolerance for 'reflections' and 'fitness'.
    log_fitness : bool
        For 'fitness': return the natural log of fitness/complexity
        (Cristelli et al. 2015). Ignored by the other methods.
    trim : bool
        If True (default), pre-trim the matrix with `trim_core` so that
        degenerate units — locations with zero diversity and activities
        with zero ubiquity — are excluded from the calculation. Trimmed
        units are returned as NaN, preserving the original index/shape.
    dmin, umin : int
        Diversity/ubiquity thresholds passed to `trim_core` (default 1).
        Use 2 for the well-connected core recommended for very sparse
        networks.

    Returns
    -------
    (eci, pci) as pd.Series or ndarrays, aligned with the input matrix
    (NaN for units removed by trimming).
    """
    if trim:
        from ..core.preprocess import trim_core
        is_df_in = isinstance(mat, pd.DataFrame)
        df = mat if is_df_in else pd.DataFrame(np.asarray(mat, dtype=float))
        trimmed = trim_core(df, dmin=dmin, umin=umin,
                            use_rca=use_rca, threshold=threshold)
        if trimmed.shape[0] < 2 or trimmed.shape[1] < 2:
            raise ValueError(
                f"After trimming to the ({dmin}, {umin})-core the matrix has "
                f"shape {trimmed.shape}; not enough connected units to "
                "compute complexity."
            )
        if trimmed.shape != df.shape:
            res_r, res_c = eci_pci(
                trimmed, use_rca=use_rca, threshold=threshold, method=method,
                iterations=iterations, extremality=extremality, tol=tol,
                log_fitness=log_fitness, trim=False,
            )
            res_r = res_r.reindex(df.index)
            res_c = res_c.reindex(df.columns)
            if not is_df_in:
                return res_r.values, res_c.values
            return res_r, res_c

    if method == "eigenvector":
        return eci_pci_eigenvector(mat, use_rca=use_rca, threshold=threshold)
    if method == "reflections":
        return method_of_reflections(
            mat, use_rca=use_rca, threshold=threshold,
            iterations=iterations if iterations is not None else 20,
            tol=tol,
        )
    if method == "fitness":
        return fitness_complexity(
            mat, use_rca=use_rca, threshold=threshold,
            iterations=iterations if iterations is not None else 20,
            extremality=extremality, tol=tol, log_fitness=log_fitness,
        )
    raise ValueError(
        "method must be 'eigenvector', 'reflections', or 'fitness'."
    )
