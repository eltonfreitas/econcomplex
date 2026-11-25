"""
Indicators for lists of activities.

A *list* is any set of activities a policy cares about: green products,
the industries of a mission or of a value chain, a technological
frontier, activities at risk in a transition. The functions here ask
what a region already has from a list, how much of the rest is within
reach, and how much of its capabilities come from the list.

Lists are given as ``{name: [activity labels]}`` or as a boolean
DataFrame (activities x lists).

References
----------
Mealy, P. & Teytelboym, A. (2022). Economic complexity and the green
economy. Research Policy, 51(8), 103948.
Andres, P., Mealy, P., Handler, N. & Fankhauser, S. (2023). Stranded
nations? Transition risks and opportunities towards a clean economy.
Environmental Research Letters, 18(4), 045004.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import (
    validate_matrix, safe_divide, binarize, align_vector, align_matrix,
)
from ..core.rca import rca as compute_rca
from ..relatedness.density import _get_mcp_and_phi


def _list_masks(lists, labels):
    """Return (names, L x C boolean array) for lists over `labels`."""
    labels = pd.Index(labels)
    if isinstance(lists, pd.DataFrame):
        masks = lists.reindex(labels, fill_value=False).to_numpy(dtype=bool).T
        return list(lists.columns), masks
    names, masks = [], []
    for name, members in lists.items():
        members = pd.Index(members)
        unknown = members.difference(labels)
        if len(unknown):
            raise ValueError(
                f"List {name!r} has {len(unknown)} activity label(s) that are "
                f"not in the matrix, e.g. {list(unknown[:3])}."
            )
        names.append(name)
        masks.append(labels.isin(members))
    return names, np.array(masks, dtype=bool).reshape(len(names), len(labels))


def _presence(mat, use_rca, threshold):
    arr = validate_matrix(mat)
    m = binarize(compute_rca(arr), threshold) if use_rca else binarize(arr, threshold)
    if isinstance(mat, pd.DataFrame):
        return m, mat.index, mat.columns
    return m, pd.RangeIndex(m.shape[0]), pd.RangeIndex(m.shape[1])


def list_complexity_potential(
    mat: Union[np.ndarray, pd.DataFrame],
    value: Union[np.ndarray, pd.Series],
    feasibility: Union[np.ndarray, pd.DataFrame],
    lists: Union[dict, pd.DataFrame],
    use_rca: bool = True,
    threshold: float = 1.0,
    average: str = "list",
) -> pd.DataFrame:
    """
    What a region has, and could reach, within a list of activities.

    For each region r and list L:

    presence_{rL}  = sum_{c in L} M_{rc} v_c
    potential_{rL} = sum_{c in L, M_{rc} = 0} f_{rc} v_c / N

    where v_c is a value per activity (PCI, wage, ...) and f_{rc} the
    feasibility of c in r (relatedness density as a share, or a predicted
    entry probability). Presence is the value of the list's activities
    the region is specialized in; potential is the value of the remaining
    ones, discounted by how reachable they are. N is the size of the
    list (``average="list"``, default) or the number of its activities
    the region lacks (``average="missing"``).

    These generalize the Green Complexity Index and Green Complexity
    Potential of Mealy & Teytelboym (2022) to any list, value and
    feasibility measure (see Notes). With the list of all activities and
    f = density, potential x |L| is the Complexity Outlook Index.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    value : array-like (length C)
        Value of each activity, matched by label. Activities without a
        value (NaN) add nothing.
    feasibility : array-like (R x C)
        Feasibility of each activity in each region, e.g.
        ``relatedness_density(mat, percent=False)``. NaN cells add
        nothing.
    lists : dict or pd.DataFrame
        ``{name: [activities]}`` or a boolean DataFrame (activities x
        lists).
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.
    average : {'list', 'missing'}
        Denominator of the potential. ``'list'``: the size of the list,
        so the potential falls as the region completes the list.
        ``'missing'``: the number of list activities the region lacks,
        i.e. the mean discounted value of what is left (NaN when nothing
        is left).

    Returns
    -------
    pd.DataFrame indexed by (region, list) with columns ``presence``,
    ``potential``, ``n_present`` (activities of the list the region is
    specialized in) and ``n_list`` (|L|). Use
    ``out["potential"].unstack()`` for a region x list table.

    Notes
    -----
    Mealy & Teytelboym (2022, eqs. 8-9) define, over the green products
    g, GCI_c = sum_g M_cg PCI_g and GCP_c as the mean of density_cg x
    PCI_g over the green products the country is not competitive in,
    with PCI rescaled to [0, 1], and standardize both across countries.
    To reproduce them pass the green list, ``value=normalize_01(pci)``,
    ``feasibility=relatedness_density(mat, percent=False)`` and
    ``average="missing"``, then z-score ``presence`` and ``potential``
    across regions.
    """
    if average not in ("list", "missing"):
        raise ValueError("average must be 'list' or 'missing'.")
    m, rows, cols = _presence(mat, use_rca, threshold)
    v = np.nan_to_num(align_vector(value, cols if isinstance(mat, pd.DataFrame) else None,
                                   m.shape[1], "value", allow_missing=True))
    f = np.nan_to_num(align_matrix(feasibility, rows, cols, "feasibility"))
    names, masks = _list_masks(lists, cols)
    size = masks.sum(axis=1)

    k = masks.T.astype(float)                                   # C x L
    n_present = m @ k
    if average == "list":
        denom = np.broadcast_to(size[None, :].astype(float), n_present.shape)
    else:
        denom = size[None, :] - n_present
    potential = np.full(n_present.shape, np.nan)
    np.divide(((1 - m) * f * v) @ k, denom, out=potential, where=denom > 0)
    out = pd.DataFrame({
        "presence": ((m * v) @ k).ravel(),
        "potential": potential.ravel(),
        "n_present": n_present.ravel().astype(int),
        "n_list": np.tile(size, len(rows)),
    }, index=pd.MultiIndex.from_product([rows, names], names=["region", "list"]))
    return out


def list_density(
    mat: Union[np.ndarray, pd.DataFrame],
    lists: Union[dict, pd.DataFrame],
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
    percent: bool = True,
) -> dict:
    """
    Part of the relatedness density that comes from each list.

    density^{L}_{rc} = sum_{c' in L} M_{rc'} phi_{c'c} / sum_{c''} phi_{c''c}

    The denominator is the one of `relatedness_density`, so the parts are
    additive: the part of a list plus the part of its complement is the
    total density, and lists that partition the activities (sectors,
    technology classes) decompose it exactly. Reads as "how much of the
    region's proximity to c is built on the activities of L" — e.g. how
    far existing brown capabilities support the entry into green ones.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    lists : dict or pd.DataFrame
        ``{name: [activities]}`` or a boolean DataFrame (activities x
        lists).
    phi : array-like (C x C), optional
        Pre-computed proximity matrix. Computed internally if None.
    use_rca, threshold, proximity_method, percent
        As in `relatedness_density`.

    Returns
    -------
    dict ``{name: R x C matrix}`` on the scale of `relatedness_density`.
    The part due to everything outside a list is
    ``relatedness_density(mat, phi=phi) - parts[name]``.
    """
    m, phi_arr, is_df = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)
    cols = mat.columns if is_df else pd.RangeIndex(m.shape[1])
    names, masks = _list_masks(lists, cols)
    total = phi_arr.sum(axis=0, keepdims=True)
    scale = 100 if percent else 1

    parts = {}
    for name, mask in zip(names, masks):
        part = safe_divide((m * mask) @ phi_arr, total) * scale
        parts[name] = pd.DataFrame(part, index=mat.index, columns=cols) if is_df else part
    return parts


def missing_links(
    mat: Union[np.ndarray, pd.DataFrame],
    links: Union[dict, pd.DataFrame],
    feasibility: Union[np.ndarray, pd.DataFrame],
    cut: float,
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Which links of a chain a region has, could reach, or lacks.

    The activities of a value chain are grouped into links (stages). For
    each region, a link is

    - ``present``  if the region is specialized in at least one of its
      activities;
    - ``feasible`` if it is not present but at least one of its
      activities has feasibility of `cut` or more;
    - ``missing``  otherwise.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix.
    links : dict or pd.DataFrame
        ``{link: [activities]}`` or a boolean DataFrame (activities x
        links). For several chains at once use tuple keys,
        ``{(chain, link): [...]}``.
    feasibility : array-like (R x C)
        Feasibility of each activity in each region (density, predicted
        probability...).
    cut : float
        Feasibility from which an activity counts as within reach, on the
        scale of `feasibility`.
    use_rca : bool
        Compute RCA before binarizing.
    threshold : float
        Binarization threshold.

    Returns
    -------
    pd.DataFrame (regions x links) with the status of each link.
    """
    m, rows, cols = _presence(mat, use_rca, threshold)
    f = np.nan_to_num(align_matrix(feasibility, rows, cols, "feasibility"), nan=-np.inf)
    names, masks = _list_masks(links, cols)
    k = masks.T.astype(float)

    present = (m @ k) > 0
    feasible = (((1 - m) * (f >= cut)) @ k) > 0
    status = np.where(present, "present", np.where(feasible, "feasible", "missing"))
    return pd.DataFrame(status, index=rows, columns=pd.Index(names, tupleize_cols=True))


def transition_risk(
    mat: Union[np.ndarray, pd.DataFrame],
    exposed: Union[list, pd.Index, np.ndarray],
    pci: Optional[Union[np.ndarray, pd.Series]] = None,
    targets: Optional[Union[list, pd.Index, np.ndarray]] = None,
    phi: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    use_rca: bool = True,
    threshold: float = 1.0,
    proximity_method: str = "max",
) -> pd.DataFrame:
    """
    Exposure of each region to a list of activities at risk, and how
    easily it can move out of them.

    Written for the low-carbon transition — `exposed` is the list of
    "brown" activities and `targets` the green or climate-compatible
    ones (Andres et al. 2023) — but it applies to any list of activities
    expected to shrink.

    For each region r, with s_{rb} the share of activity b in the
    region's total value:

    ``exposure``  sum_b s_{rb}: share of the region's value in the
                  exposed activities.
    ``lock_in``   sum_b s_{rb} (1 - PCI_b), with PCI rescaled to [0, 1]:
                  the Brown Lock-in Index. Exposure counts more when it
                  sits in low-complexity activities, whose capabilities
                  are harder to redeploy. Needs `pci`.
    ``n_exposed`` number of exposed activities the region is specialized
                  in.
    ``outlook``   Transition Outlook: mean, over the exposed activities
                  the region is specialized in, of
                  TO_b = (mean proximity of b to the targets) /
                         (mean proximity of b to all activities).
                  Above 1 the region's exposed activities are closer to
                  the targets than to the average activity. NaN if the
                  region has no exposed specialization.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix (employment, exports).
    exposed : list of activity labels
        Activities at risk (column positions if `mat` is an array).
    pci : array-like (length C), optional
        Activity complexity, matched by label; rescaled to [0, 1] by
        min-max over the activities with a value.
    targets : list of activity labels, optional
        Activities to move into. Default: every activity not in
        `exposed` (the "overall" outlook of the paper).
    phi : array-like (C x C), optional
        Proximity matrix. Computed internally if None.
    use_rca, threshold, proximity_method
        As in `relatedness_density`.

    Returns
    -------
    pd.DataFrame indexed by region with the four columns above
    (``lock_in`` is NaN without `pci`).

    Notes
    -----
    Follows eqs. (1), (3) and (4) of Andres et al. (2023), who then
    standardize every index across countries (mean 0, s.d. 1); the
    values here are not standardized. The paper does not spell out how
    PCI is rescaled to [0, 1]; min-max is used.
    """
    m, phi_arr, is_df = _get_mcp_and_phi(mat, phi, use_rca, threshold, proximity_method)
    arr = validate_matrix(mat)
    rows = mat.index if is_df else pd.RangeIndex(m.shape[0])
    cols = mat.columns if is_df else pd.RangeIndex(m.shape[1])
    _, masks = _list_masks({"exposed": exposed}, cols)
    brown = masks[0]
    if targets is None:
        target = ~brown
    else:
        target = _list_masks({"targets": targets}, cols)[1][0]
    if not brown.any() or not target.any():
        raise ValueError("exposed and targets must each hold at least one activity.")

    share = safe_divide(arr, arr.sum(axis=1, keepdims=True))
    lock_in = np.full(m.shape[0], np.nan)
    if pci is not None:
        q = align_vector(pci, cols if is_df else None, m.shape[1], "pci", allow_missing=True)
        span = np.nanmax(q) - np.nanmin(q)
        q01 = (q - np.nanmin(q)) / span if span > 0 else np.zeros_like(q)
        lock_in = (share[:, brown] * np.nan_to_num(1 - q01[brown])).sum(axis=1)

    to_b = safe_divide(phi_arr[:, target].mean(axis=1), phi_arr.mean(axis=1))    # per activity
    n_exposed = m[:, brown].sum(axis=1)
    outlook = np.full(m.shape[0], np.nan)
    np.divide(m[:, brown] @ to_b[brown], n_exposed, out=outlook, where=n_exposed > 0)

    return pd.DataFrame({
        "exposure": share[:, brown].sum(axis=1),
        "lock_in": lock_in,
        "n_exposed": n_exposed.astype(int),
        "outlook": outlook,
    }, index=rows.rename("region") if rows.name is None else rows)
