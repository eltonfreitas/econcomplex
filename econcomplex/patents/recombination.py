"""
Patent-based recombination complexity.

References
----------
Fleming & Sorenson (2001) "Technology as a Complex Adaptive System:
Evidence from Patent Data".
"""

import numpy as np
import pandas as pd
from typing import Union

from ..core.utils import validate_matrix, safe_divide


def ease_of_recombination(
    incidence: Union[np.ndarray, pd.DataFrame],
) -> Union[pd.Series, np.ndarray]:
    """
    Ease of Recombination (EOR) for each technology/class.

    EOR_t = (number of patents in which t co-occurs with any other tech)
            / (total number of patents containing t)

    A high EOR means the technology is easily combined with others.

    Parameters
    ----------
    incidence : array-like (P x T)
        Patent × technology binary incidence matrix.

    Returns
    -------
    pd.Series indexed by technology.
    """
    is_df = isinstance(incidence, pd.DataFrame)
    col_index = incidence.columns if is_df else None

    arr = validate_matrix(incidence)

    # For each patent, does it contain more than one technology?
    patents_with_multiple = (arr.sum(axis=1) > 1).astype(float)  # P

    # For each technology: number of patents where it appears with others
    numerator = arr.T @ patents_with_multiple   # T

    # Total patents containing each technology
    denominator = arr.sum(axis=0)               # T

    result = safe_divide(numerator, denominator)

    if is_df:
        return pd.Series(result, index=col_index, name="ease_of_recombination")
    return result


def modular_complexity(
    incidence: Union[np.ndarray, pd.DataFrame],
    eor: Union[np.ndarray, pd.Series, None] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Modular Complexity of each patent.

    MC_p = (number of technologies in patent p)
           / (sum of EOR for each technology in patent p)

    Higher MC = technologies are harder to recombine = more complex patent.

    Parameters
    ----------
    incidence : array-like (P x T)
        Patent × technology binary incidence matrix.
    eor : array-like (length T), optional
        Pre-computed EOR vector. Computed internally if None.

    Returns
    -------
    pd.Series indexed by patent.
    """
    is_df = isinstance(incidence, pd.DataFrame)
    row_index = incidence.index if is_df else None

    arr = validate_matrix(incidence)

    if eor is None:
        eor_arr = ease_of_recombination(arr)
        if isinstance(eor_arr, pd.Series):
            eor_arr = eor_arr.values
    else:
        eor_arr = np.array(eor, dtype=float)

    # number of technologies per patent
    n_techs = arr.sum(axis=1)  # P

    # sum of EOR for technologies present in each patent
    eor_sum = arr @ eor_arr   # P

    result = safe_divide(n_techs, eor_sum)

    if is_df:
        return pd.Series(result, index=row_index, name="modular_complexity")
    return result


def modular_complexity_avg(
    incidence: Union[np.ndarray, pd.DataFrame],
    eor: Union[np.ndarray, pd.Series, None] = None,
) -> Union[pd.Series, np.ndarray]:
    """
    Average Modular Complexity aggregated per technology (column).

    avg_MC_t = mean(MC_p for all patents p containing technology t)

    Returns
    -------
    pd.Series indexed by technology.
    """
    is_df = isinstance(incidence, pd.DataFrame)
    col_index = incidence.columns if is_df else None

    arr = validate_matrix(incidence)
    mc = modular_complexity(arr, eor=eor)
    if isinstance(mc, pd.Series):
        mc_arr = mc.values
    else:
        mc_arr = mc

    col_sums = arr.sum(axis=0)
    numerator = arr.T @ mc_arr
    result = safe_divide(numerator, col_sums)

    if is_df:
        return pd.Series(result, index=col_index, name="modular_complexity_avg")
    return result
