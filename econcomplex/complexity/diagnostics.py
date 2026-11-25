"""
Diagnostics of the eigenvector ECI.

References
----------
Mealy, P., Farmer, J. D. & Teytelboym, A. (2019). Interpreting economic
complexity. Science Advances, 5(1), eaau1705.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from typing import Optional, Union

from ..core.utils import validate_matrix, binarize, safe_divide
from ..core.rca import rca as compute_rca
from .eci_pci import eci_pci


def _rank_corr(a, b):
    ok = ~np.isnan(a) & ~np.isnan(b)
    if ok.sum() < 3:
        return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok])
    return np.corrcoef(ra, rb)[0, 1] if ra.std() > 0 and rb.std() > 0 else np.nan


def eci_diagnostics(
    mat: Union[np.ndarray, pd.DataFrame],
    n_boot: int = 100,
    share: float = 0.8,
    use_rca: bool = True,
    threshold: float = 1.0,
    seed: Optional[int] = None,
) -> pd.Series:
    """
    How much can the ECI ranking of this matrix be trusted?

    The eigenvector ECI is the second eigenvector of a region-to-region
    similarity matrix: it splits the regions along the main axis of
    difference in what they do (Mealy, Farmer & Teytelboym 2019). That
    reading only holds when the split is clear and stable, which small
    or sparse subnational matrices do not guarantee. Three checks:

    - **Spectral gap.** ``eigenvalue_2`` close to 1 with a wide ``gap``
      to ``eigenvalue_3`` means one dominant axis. A narrow gap means
      the second and third eigenvectors are nearly interchangeable and
      the ranking can change with small changes in the data.
    - **Relation with diversity.** ``corr_diversity`` (Pearson) and
      ``rank_corr_diversity`` (Spearman). An eigenvector has no sign of
      its own; this library, like most, orients the ECI so that it
      correlates positively with diversity. That works for countries,
      but Mealy et al. find no correlation for US states and a negative
      one for UK local authorities, where the less diversified places
      are the richer ones. When the correlation is weak, the orientation
      is arbitrary and "high ECI" may mean the opposite of "complex":
      check it against an outside variable such as income or wages.
    - **Stability.** The ECI is recomputed on `n_boot` random subsets of
      ``share`` of the activities. ``rank_stability_mean`` and
      ``rank_stability_p05`` summarize the rank correlation with the
      full-sample ECI; ``sign_flips`` is the share of subsets in which
      the ranking comes out inverted.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    n_boot : int
        Number of random subsets of activities (0 skips the stability
        check).
    share : float
        Share of the activities kept in each subset.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    seed : int, optional
        Random seed.

    Returns
    -------
    pd.Series with ``n_regions`` and ``n_activities`` (after trimming),
    ``eigenvalue_2``, ``eigenvalue_3``, ``gap``, ``corr_diversity``,
    ``rank_corr_diversity``, ``rank_stability_mean``,
    ``rank_stability_p05`` and ``sign_flips``.
    """
    df = mat if isinstance(mat, pd.DataFrame) else pd.DataFrame(validate_matrix(mat))
    arr = validate_matrix(df)
    full = binarize(compute_rca(arr), threshold) if use_rca else binarize(arr, threshold)
    diversity = full.sum(axis=1)
    m = full[diversity > 0][:, full.sum(axis=0) > 0]      # the (1, 1)-core
    kc, kp = m.sum(axis=1), m.sum(axis=0)
    mcc = safe_divide(m, kc[:, None]) @ safe_divide(m, kp[None, :]).T
    eig = np.sort(np.linalg.eigvals(mcc).real)[::-1]

    eci, _ = eci_pci(df, use_rca=use_rca, threshold=threshold)
    eci = eci.to_numpy()
    ok = ~np.isnan(eci)

    out = {
        "n_regions": float(m.shape[0]), "n_activities": float(m.shape[1]),
        "eigenvalue_2": eig[1], "eigenvalue_3": eig[2] if eig.size > 2 else np.nan,
        "gap": eig[1] - eig[2] if eig.size > 2 else np.nan,
        "corr_diversity": np.corrcoef(eci[ok], diversity[ok])[0, 1],
        "rank_corr_diversity": _rank_corr(eci, diversity),
        "rank_stability_mean": np.nan, "rank_stability_p05": np.nan, "sign_flips": np.nan,
    }
    if n_boot:
        rng = np.random.default_rng(seed)
        n_keep = max(2, int(round(share * df.shape[1])))
        corrs = []
        for _ in range(n_boot):
            cols = np.sort(rng.choice(df.shape[1], n_keep, replace=False))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sub, _ = eci_pci(df.iloc[:, cols], use_rca=use_rca, threshold=threshold)
            except (ValueError, np.linalg.LinAlgError):
                continue
            corrs.append(_rank_corr(eci, sub.to_numpy()))
        corrs = np.array(corrs, dtype=float)
        corrs = corrs[~np.isnan(corrs)]
        if corrs.size:
            out.update({"rank_stability_mean": corrs.mean(),
                        "rank_stability_p05": np.percentile(corrs, 5),
                        "sign_flips": float((corrs < 0).mean())})
    return pd.Series(out, name="eci_diagnostics")
