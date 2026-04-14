"""
High-level pipeline: compute all complexity indicators in one call.

Mirrors the API of py-ecomplexity but with richer output and support
for all indicator methods.
"""

from __future__ import annotations

import warnings
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd

from .core.rca import rca, rpop, mcp
from .core.diversity import diversity, ubiquity
from .complexity.reflections import method_of_reflections
from .complexity.eigenvector import eci_pci
from .complexity.fitness import fitness_complexity
from .relatedness.proximity import proximity, continuous_proximity
from .relatedness.density import relatedness_density, distance
from .outlook.coi_cog import complexity_outlook_index, complexity_outlook_gain
from .core.utils import pivot_to_matrix, binarize, safe_divide


def compute_complexity(
    data: pd.DataFrame,
    cols: Dict[str, str],
    *,
    method: Literal["eigenvector", "reflections", "fitness"] = "eigenvector",
    presence_test: Literal["rca", "rpop", "both", "manual"] = "rca",
    threshold: float = 1.0,
    iterations: int = 20,
    proximity_type: Literal["discrete", "continuous"] = "discrete",
    proximity_method: Literal["max", "sqrt", "min"] = "max",
    pop: Optional[pd.DataFrame] = None,
    compute_coi_cog: bool = True,
    time_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Compute a full suite of economic complexity indicators.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format DataFrame.
    cols : dict
        Mapping with keys:
          'loc'  → column name for locations/regions
          'act'  → column name for activities/products
          'val'  → column name for values
        Optionally:
          'time' → column name for time periods (enables panel mode)
    method : str
        ECI/PCI method: 'eigenvector', 'reflections', or 'fitness'.
    presence_test : str
        How to binarize: 'rca', 'rpop', 'both', 'manual'.
    threshold : float
        Binarization threshold (default 1.0).
    iterations : int
        Iterations for reflections/fitness methods.
    proximity_type : str
        'discrete' (co-occurrence) or 'continuous' (correlation).
    proximity_method : str
        Proximity normalization: 'max', 'sqrt', or 'min'.
    pop : pd.DataFrame, optional
        Population data with columns ['loc', 'time'?, 'pop'].
        Required when presence_test is 'rpop' or 'both'.
    compute_coi_cog : bool
        Whether to compute COI and COG (slower).
    time_col : str, optional
        Override for time column name (or use cols['time']).

    Returns
    -------
    Original DataFrame with new columns:
      rca, mcp, diversity, ubiquity, eci, pci, density, distance, [coi, cog]
      [fitness, complexity] (when method='fitness')
    """
    loc_col = cols.get("loc") or cols.get("location")
    act_col = cols.get("act") or cols.get("activity") or cols.get("product")
    val_col = cols.get("val") or cols.get("value")
    t_col = time_col or cols.get("time")

    if not all([loc_col, act_col, val_col]):
        raise ValueError("cols must contain 'loc', 'act', and 'val' keys.")

    if t_col and t_col in data.columns:
        periods = data[t_col].unique()
        results = []
        for period in sorted(periods):
            period_data = data[data[t_col] == period].copy()
            pop_period = None
            if pop is not None:
                if t_col in pop.columns:
                    pop_period = pop[pop[t_col] == period]
                else:
                    pop_period = pop
            period_result = _compute_single_period(
                period_data, loc_col, act_col, val_col,
                method=method,
                presence_test=presence_test,
                threshold=threshold,
                iterations=iterations,
                proximity_type=proximity_type,
                proximity_method=proximity_method,
                pop=pop_period,
                compute_coi_cog=compute_coi_cog,
            )
            results.append(period_result)
        return pd.concat(results, ignore_index=True)
    else:
        return _compute_single_period(
            data, loc_col, act_col, val_col,
            method=method,
            presence_test=presence_test,
            threshold=threshold,
            iterations=iterations,
            proximity_type=proximity_type,
            proximity_method=proximity_method,
            pop=pop,
            compute_coi_cog=compute_coi_cog,
        )


def _compute_single_period(
    data: pd.DataFrame,
    loc_col: str,
    act_col: str,
    val_col: str,
    *,
    method: str,
    presence_test: str,
    threshold: float,
    iterations: int,
    proximity_type: str,
    proximity_method: str,
    pop: Optional[pd.DataFrame],
    compute_coi_cog: bool,
) -> pd.DataFrame:
    """Compute indicators for a single time period."""

    # --- Build pivot matrix ---
    mat = pivot_to_matrix(data, index=loc_col, columns=act_col, values=val_col)

    # --- RCA ---
    rca_mat = rca(mat)

    # --- Population for RPOP ---
    pop_vec = None
    if presence_test in ("rpop", "both") and pop is not None:
        pop_col = [c for c in pop.columns if c not in (loc_col,)][0]
        pop_aligned = pop.set_index(loc_col).reindex(mat.index)[pop_col]
        pop_vec = pop_aligned.values

    # --- Mcp binary matrix ---
    mcp_mat = mcp(
        mat,
        presence_test=presence_test,
        rca_threshold=threshold,
        rpop_threshold=threshold,
        pop=pop_vec,
    )

    # --- Diversity & Ubiquity ---
    div = mcp_mat.sum(axis=1)
    ubi = mcp_mat.sum(axis=0)

    # --- ECI / PCI ---
    if method == "eigenvector":
        eci_s, pci_s = eci_pci(mat, use_rca=True, threshold=threshold)
    elif method == "reflections":
        eci_s, pci_s = method_of_reflections(mat, use_rca=True,
                                               threshold=threshold,
                                               iterations=iterations)
    elif method == "fitness":
        eci_s, pci_s = fitness_complexity(mat, use_rca=True,
                                           threshold=threshold,
                                           iterations=iterations)
        eci_s.name = "eci"
        pci_s.name = "pci"
    else:
        raise ValueError("method must be 'eigenvector', 'reflections', or 'fitness'.")

    # --- Proximity ---
    if proximity_type == "continuous":
        phi = continuous_proximity(rca_mat)
    else:
        phi_dict = proximity(mcp_mat, use_rca=False, threshold=0.5,
                             method=proximity_method, compute="product")
        phi = phi_dict["product"]

    # --- Density & Distance ---
    dens_mat = relatedness_density(mcp_mat, phi=phi, use_rca=False, threshold=0.5)
    dist_mat = 1 - dens_mat / 100

    # --- COI / COG (optional) ---
    coi_s = None
    cog_mat = None
    if compute_coi_cog:
        pci_arr = pci_s.values if isinstance(pci_s, pd.Series) else pci_s
        coi_s = complexity_outlook_index(
            mcp_mat, pci_arr, phi=phi, use_rca=False, threshold=0.5
        )
        cog_mat = complexity_outlook_gain(
            mcp_mat, pci_arr, phi=phi, use_rca=False, threshold=0.5
        )

    # --- Merge everything back into long format ---
    result = data.copy()

    # Add location-level indicators
    loc_indicators = pd.DataFrame({
        "diversity": div,
        "eci": eci_s,
        "coi": coi_s if coi_s is not None else np.nan,
    })
    result = result.merge(
        loc_indicators.reset_index().rename(columns={"index": loc_col}),
        on=loc_col,
        how="left",
    )

    # Add activity-level indicators
    act_indicators = pd.DataFrame({
        "ubiquity": ubi,
        "pci": pci_s,
    })
    result = result.merge(
        act_indicators.reset_index().rename(columns={"index": act_col}),
        on=act_col,
        how="left",
    )

    # Add cell-level indicators
    rca_long = (
        rca_mat.reset_index()
        .melt(id_vars=rca_mat.index.name, var_name=act_col, value_name="rca")
    )
    mcp_long = (
        mcp_mat.reset_index()
        .melt(id_vars=mcp_mat.index.name, var_name=act_col, value_name="mcp")
    )
    dens_long = (
        dens_mat.reset_index()
        .melt(id_vars=dens_mat.index.name, var_name=act_col, value_name="density")
    )
    dist_long = (
        dist_mat.reset_index()
        .melt(id_vars=dist_mat.index.name, var_name=act_col, value_name="distance")
    )

    for long_df in [rca_long, mcp_long, dens_long, dist_long]:
        result = result.merge(long_df, on=[loc_col, act_col], how="left")

    if cog_mat is not None:
        cog_long = (
            cog_mat.reset_index()
            .melt(id_vars=cog_mat.index.name, var_name=act_col, value_name="cog")
        )
        result = result.merge(cog_long, on=[loc_col, act_col], how="left")

    return result
