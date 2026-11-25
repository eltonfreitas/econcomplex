"""
Relatedness density, distance, and related indicators.

References
----------
Hidalgo et al. (2007); Hausmann & Klinger (2007);
Balland et al. (2019).
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, safe_divide, binarize, align_square
from ..core.rca import rca as compute_rca


def _get_mcp_and_phi(
    mat,
    phi,
    use_rca: bool,
    threshold: float,
    proximity_method: str,
):
    """Helper: returns binary M and proximity matrix as ndarrays."""
    is_df = isinstance(mat, pd.DataFrame)

    arr = validate_matrix(mat)

    if use_rca:
        m = binarize(compute_rca(arr), threshold)
    else:
        m = binarize(arr, threshold)

    if phi is None:
        from .proximity import proximity as _prox
        phi_dict = _prox(
            m, use_rca=False, threshold=0.5,
            method=proximity_method, compute="product",
        )
        phi_arr = (
            phi_dict["product"].values
            if isinstance(phi_dict["product"], pd.DataFrame)
            else phi_dict["product"]
        )
    else:
        phi_arr = align_square(phi, mat.columns if is_df else None, m.shape[1])

    return m, phi_arr, is_df


def relatedness_density(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
    percent: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relatedness density for each (region, activity) pair.

    density_{rc} = (M * Phi)_{rc} / rowSums(Phi)_c * 100

    Fraction of activities related to c that region r already has,
    expressed as a percentage (0-100) by default, or as a share (0-1)
    with ``percent=False``.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    phi : array-like (C x C), optional
        Pre-computed product proximity matrix. Computed internally if None.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    proximity_method : str
        Normalization method for proximity ('max', 'sqrt', 'min').
    percent : bool
        If True (default) return 0-100; if False return 0-1, the scale
        used by `distance` (= 1 - density) and `cross_relatedness`.

    Returns
    -------
    R x C relatedness density matrix.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> ec.relatedness_density(mat).round(1).values.tolist()
    [[0.0, 33.3, 33.3], [100.0, 66.7, 66.7], [100.0, 100.0, 100.0]]
    >>> ec.relatedness_density(mat, percent=False).round(2).loc["R1"].tolist()
    [0.0, 0.33, 0.33]
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)  # 1 x C
    numerator = m @ phi_arr                              # R x C
    density = safe_divide(numerator, col_sums_phi) * (100 if percent else 1)

    if is_df:
        return pd.DataFrame(density, index=row_index, columns=col_index)
    return density


def distance(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Distance (1 - density/100).

    Weighted fraction of related activities that region r does NOT have.

    Returns
    -------
    R x C distance matrix (values 0–1).
    """
    dens = relatedness_density(mat, phi=phi, use_rca=use_rca,
                                threshold=threshold, proximity_method=proximity_method)
    if isinstance(dens, pd.DataFrame):
        return 1 - dens / 100
    return 1 - dens / 100


def relatedness_density_internal(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
    percent: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Internal relatedness density: density values for activities
    the region ALREADY has (M_{rc} = 1).
    Other cells are NaN.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)
    numerator = m @ phi_arr
    density = safe_divide(numerator, col_sums_phi) * (100 if percent else 1)

    mask = m == 0
    density_internal = density.copy()
    density_internal[mask] = np.nan

    if is_df:
        return pd.DataFrame(density_internal, index=row_index, columns=col_index)
    return density_internal


def relatedness_density_external(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
    percent: bool = True,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    External relatedness density: density values for activities
    the region does NOT yet have (M_{rc} = 0).
    Other cells are NaN.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)
    numerator = m @ phi_arr
    density = safe_divide(numerator, col_sums_phi) * (100 if percent else 1)

    mask = m == 1
    density_external = density.copy()
    density_external[mask] = np.nan

    if is_df:
        return pd.DataFrame(density_external, index=row_index, columns=col_index)
    return density_external


def relative_relatedness(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relative relatedness (Pinheiro et al. 2022, eq. 7): z-transform of the
    relatedness density against the statistics of the region's option set
    (activities it does NOT currently hold, M_{rc} = 0).

    relative_density_{rc} = (density_{rc} - mean_non_held_r) / std_non_held_r
    for cells where M_{rc} = 0; NaN otherwise.

    References
    ----------
    Pinheiro, Hartmann, Boschma & Hidalgo (2022) "The time and frequency
    of unrelated diversification", Research Policy 51, 104323.

    Returns
    -------
    R x C standardized density matrix.
    """
    is_df = isinstance(mat, pd.DataFrame)
    row_index = mat.index if is_df else None
    col_index = mat.columns if is_df else None

    m, phi_arr, _ = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)

    col_sums_phi = phi_arr.sum(axis=0, keepdims=True)
    numerator = m @ phi_arr
    density = safe_divide(numerator, col_sums_phi) * 100

    result = np.full_like(density, np.nan)
    for i in range(m.shape[0]):
        non_held = m[i] == 0
        vals = density[i, non_held]
        std = vals.std()
        if std > 0:
            result[i, non_held] = (vals - vals.mean()) / std

    if is_df:
        return pd.DataFrame(result, index=row_index, columns=col_index)
    return result


# Short alias matching the documented API
density = relatedness_density


def full_relatedness(
    mat: Union[np.ndarray, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    part: str = "full",
    weights: str = "iapadre",
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relatedness density with continuous specialization (Full Relatedness).

    The standard density only counts the related activities in which the
    region is specialized (RCA >= 1); everything below the cut-off is
    treated as absent. Cicerone, McCann & Venhorst (2026) weight every
    related activity by how specialized the region is in it:

    FullRel_{rc} = sum_{z != c} phi_{cz} w_{rz} / sum_{z != c} phi_{cz}

    with w = RSCA + 1, the symmetric index of revealed advantage shifted
    to [0, 2]. The index then splits exactly into the part that comes
    from the activities in which the region is specialized and the part
    that comes from those in which it is not:

    FullRel = Spec_FullRel + NoSpec_FullRel

    In their data the second part predicts new specializations beyond
    the first, more so in high-income regions.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    phi : array-like (C x C), optional
        Proximity matrix. Computed internally (from the binary RCA
        matrix) if None. Its diagonal is ignored.
    part : {'full', 'specialized', 'non_specialized'}
        The whole index or one of its two parts.
    weights : {'iapadre', 'dalum', 'capped'}
        How specialization enters. ``'iapadre'`` (default, the paper's
        choice) and ``'dalum'`` use RSCA + 1 with the corresponding
        symmetric index of `rsca`; the result ranges from 0 to 2.
        ``'capped'`` uses min(RCA / threshold, 1): activities above the
        threshold count fully and those below count in proportion, so
        the result ranges from 0 to 1 and its specialized part is the
        standard density (as a share).
    threshold : float
        RCA from which an activity counts as specialized (the split
        between the two parts, and the proximity matrix when `phi` is
        None).
    proximity_method : str
        Normalization of the proximity computed internally.

    Returns
    -------
    R x C matrix (DataFrame if the input was one), for every cell,
    present or not. Unlike `relatedness_density` it is not a percentage.

    Notes
    -----
    Eqs. (8)-(10) of the paper. The printed condition for the split,
    "RSTA >= 1", is read here as RSTA + 1 >= 1, that is RSTA >= 0, which
    is the paper's own definition of a specialization ("positive values
    indicate that the region is relatively specialised") and coincides
    with RCA >= 1. The ``'capped'`` weighting is not from the paper.

    References
    ----------
    Cicerone, G., McCann, P. & Venhorst, V. A. (2026). Disentangling
    relatedness density: the complementary impact of non-specialised
    related technologies. Spatial Economic Analysis, 21(3), 460-477.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[6, 2, 0], [1, 5, 4], [3, 3, 6]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> full = ec.full_relatedness(mat)
    >>> spec = ec.full_relatedness(mat, part="specialized")
    >>> rest = ec.full_relatedness(mat, part="non_specialized")
    >>> bool(((spec + rest - full).abs() < 1e-12).all().all())
    True
    """
    if part not in ("full", "specialized", "non_specialized"):
        raise ValueError("part must be 'full', 'specialized' or 'non_specialized'.")
    if weights not in ("iapadre", "dalum", "capped"):
        raise ValueError("weights must be 'iapadre', 'dalum' or 'capped'.")
    from ..core.rca import rsca

    m, phi_arr, is_df = _get_mcp_and_phi(mat, phi, True, threshold, proximity_method)
    arr = validate_matrix(mat)
    if weights == "capped":
        w = np.minimum(compute_rca(arr) / threshold, 1.0)
    else:
        w = rsca(arr, method=weights) + 1.0
    if part == "specialized":
        w = w * m
    elif part == "non_specialized":
        w = w * (1 - m)

    p = phi_arr.copy()
    np.fill_diagonal(p, 0.0)
    out = safe_divide(w @ p, p.sum(axis=0, keepdims=True))

    if is_df:
        return pd.DataFrame(out, index=mat.index, columns=mat.columns)
    return out


def density_contributions(
    mat: Union[np.ndarray, pd.DataFrame],
    region,
    activity,
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
    percent: bool = True,
) -> pd.Series:
    """
    Which activities of a region make another activity feasible there.

    Splits the relatedness density of one (region, activity) pair into
    the contribution of each activity the region already has:

    contribution_{c'} = M_{rc'} * phi_{c'c} / sum_{c''} phi_{c''c}

    The contributions add up to ``relatedness_density`` of the pair, so
    they read as "c is close to r's portfolio because r already has A, B
    and C" — the simplest explanation of a diversification opportunity.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    region, activity
        Labels of the pair (row and column positions if `mat` is an
        array).
    phi : array-like (C x C), optional
        Pre-computed proximity matrix. Computed internally if None.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    proximity_method : str
        Normalization method for proximity ('max', 'sqrt', 'min').
    percent : bool
        Same scale as `relatedness_density`: 0-100 (default) or 0-1.

    Returns
    -------
    pd.Series indexed by activity, largest contribution first. Only the
    activities that contribute (present in the region and with positive
    proximity to `activity`) are listed.

    Examples
    --------
    >>> import pandas as pd
    >>> import econcomplex as ec
    >>> mat = pd.DataFrame([[4, 0, 0], [0, 2, 2], [4, 2, 2]],
    ...                    index=["R1", "R2", "R3"], columns=["A", "B", "C"])
    >>> ec.density_contributions(mat, "R2", "A").round(1).to_dict()
    {'B': 50.0, 'C': 50.0}
    """
    m, phi_arr, is_df = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)
    r = mat.index.get_loc(region) if is_df else region
    c = mat.columns.get_loc(activity) if is_df else activity

    total = phi_arr[:, c].sum()
    contrib = m[r] * phi_arr[:, c] / total if total > 0 else np.zeros(m.shape[1])
    out = pd.Series(contrib * (100 if percent else 1),
                    index=mat.columns if is_df else None, name="contribution")
    return out[out > 0].sort_values(ascending=False, kind="stable")


def decomposed_density(
    density: Union[np.ndarray, pd.DataFrame],
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> Union[pd.DataFrame, np.ndarray]:
    """
    Relatedness density net of region and activity fixed effects.

    Density is mechanically high in diversified regions (they have more of
    everything) and differs across activities with their position in the
    network. This function removes both level effects and returns the
    residual of the two-way fixed-effects regression
    d_{rc} = a_r + b_c + e_{rc}, i.e. what is specific to the pair. On a
    complete matrix the residual is

    e_{rc} = d_{rc} - mean_r - mean_c + mean

    NaN cells are left out of the estimation and stay NaN, so the fixed
    effects can be estimated on a subset of cells — for instance only on
    the activities each region does not have yet, by passing
    ``relatedness_density_external(mat)``.

    Parameters
    ----------
    density : array-like (R x C)
        A density matrix (any scale), possibly with NaN.
    tol : float
        Convergence tolerance of the fixed effects (incomplete matrices
        are solved by alternating projections).
    max_iter : int
        Maximum number of iterations.

    Returns
    -------
    R x C matrix of residuals, in the units of the input, with zero mean
    in every row and every column.

    Notes
    -----
    Nomaler & Verspagen (2024) regress density on the diversity of the
    country and the ubiquity of the product and call the fitted part
    "unrelated variety" and the residual "related variety". This function
    is a generalization, not a replication: region and activity fixed
    effects absorb diversity and ubiquity together with any other
    region-level or activity-level factor.

    References
    ----------
    Nomaler, Ö. & Verspagen, B. (2024). Related or unrelated
    diversification: What is smart specialization? Structural Change and
    Economic Dynamics, 70, 503-515.

    Examples
    --------
    >>> import numpy as np
    >>> import econcomplex as ec
    >>> additive = np.add.outer([1.0, 2.0, 5.0], [0.0, 10.0])   # a_r + b_c
    >>> bool(np.allclose(ec.decomposed_density(additive), 0))
    True
    """
    is_df = isinstance(density, pd.DataFrame)
    d = np.array(density, dtype=float)
    if d.ndim != 2:
        raise ValueError("density must be a 2-D matrix.")
    if np.isinf(d).any():
        raise ValueError("density has infinite cells.")

    valid = ~np.isnan(d)
    n_row = np.maximum(valid.sum(axis=1, keepdims=True), 1)
    n_col = np.maximum(valid.sum(axis=0, keepdims=True), 1)
    res = np.where(valid, d, 0.0)
    for _ in range(max_iter):
        row_mean = res.sum(axis=1, keepdims=True) / n_row
        res = (res - row_mean) * valid
        col_mean = res.sum(axis=0, keepdims=True) / n_col
        res = (res - col_mean) * valid
        if max(np.abs(row_mean).max(), np.abs(col_mean).max()) < tol:
            break
    res[~valid] = np.nan

    if is_df:
        return pd.DataFrame(res, index=density.index, columns=density.columns)
    return res
