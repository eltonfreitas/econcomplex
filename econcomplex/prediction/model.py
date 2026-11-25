"""
Machine-learning prediction of entry.

Optional module: needs scikit-learn (``pip install econcomplex[ml]``),
imported only inside the functions so that the rest of the library works
without it.

`build_features` turns matrices into a table of (region, activity)
cells; `predict_entry` returns out-of-sample probabilities of entry,
ready for `validate_entry`; `explain_entry` says which features carry
the prediction.

References
----------
Breiman, L. (2001). Random forests. Machine Learning, 45, 5-32
(permutation importance).
Tacchella, A., Zaccaria, A., Miccheli, M. & Pietronero, L. (2023).
Relatedness in the era of machine learning. Chaos, Solitons & Fractals,
176, 114071.
Albora, G., Pietronero, L., Tacchella, A. & Zaccaria, A. (2023). Product
progression: a machine learning approach to forecasting industrial
upgrading. Scientific Reports, 13, 1481.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union

from ..core.utils import validate_matrix, binarize, align_matrix
from ..core.rca import rca as compute_rca
from .entry import _auc

CELL = ["region", "activity"]


def build_features(
    mat: Union[np.ndarray, pd.DataFrame],
    layers: Optional[dict] = None,
    at_risk: Optional[Union[np.ndarray, pd.DataFrame]] = None,
    region_vars: Optional[pd.DataFrame] = None,
    activity_vars: Optional[pd.DataFrame] = None,
    controls: bool = True,
    use_rca: bool = True,
    threshold: float = 1.0,
) -> pd.DataFrame:
    """
    Table of features for each (region, activity) cell.

    Stacks any number of region x activity matrices — relatedness
    densities computed with different proximity matrices (co-location,
    co-occupation, labour flows, input-output), spatial density,
    relative relatedness... — into the columns of a long table, and adds
    the usual controls computed from the value matrix.

    Parameters
    ----------
    mat : array-like (R x C)
        Value matrix at the start of the window (t0).
    layers : dict, optional
        ``{name: R x C matrix}``; each becomes a column. DataFrames are
        matched to `mat` by label.
    at_risk : array-like (R x C) of bool, optional
        Keep only these cells (second output of `entry_targets`).
    region_vars : pd.DataFrame (regions x variables), optional
        Extra region-level columns, repeated across activities.
    activity_vars : pd.DataFrame (activities x variables), optional
        Extra activity-level columns (e.g. national growth), repeated
        across regions.
    controls : bool
        If True (default) add ``rca`` (initial RCA), ``log_region_size``,
        ``diversity``, ``log_activity_size`` and ``ubiquity``. They show
        how much the layers add to what size and frequency already
        explain.
    use_rca : bool
        Compute RCA before binarizing (for diversity and ubiquity).
    threshold : float
        Binarization threshold.

    Returns
    -------
    pd.DataFrame indexed by (region, activity), one row per cell.

    Notes
    -----
    To avoid leakage, every layer must be computed with data up to t0
    only — including the proximity matrix behind a density.
    """
    is_df = isinstance(mat, pd.DataFrame)
    arr = validate_matrix(mat)
    rows = mat.index if is_df else pd.RangeIndex(arr.shape[0])
    cols = mat.columns if is_df else pd.RangeIndex(arr.shape[1])
    index = pd.MultiIndex.from_product([rows, cols], names=CELL)

    data = {name: align_matrix(layer, rows, cols, name).ravel()
            for name, layer in (layers or {}).items()}
    if controls:
        r = compute_rca(arr)
        m = binarize(r, threshold) if use_rca else binarize(arr, threshold)
        shape = np.ones_like(arr)
        data.update({
            "rca": r.ravel(),
            "log_region_size": (np.log1p(arr.sum(axis=1))[:, None] * shape).ravel(),
            "diversity": (m.sum(axis=1)[:, None] * shape).ravel(),
            "log_activity_size": (np.log1p(arr.sum(axis=0))[None, :] * shape).ravel(),
            "ubiquity": (m.sum(axis=0)[None, :] * shape).ravel(),
        })
    out = pd.DataFrame(data, index=index)
    if region_vars is not None:
        out = out.join(region_vars.rename_axis(CELL[0]), on=CELL[0])
    if activity_vars is not None:
        out = out.join(activity_vars.rename_axis(CELL[1]), on=CELL[1])
    if at_risk is not None:
        out = out[align_matrix(at_risk, rows, cols, "at_risk").ravel() > 0]
    return out


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_model(model, calibrate, seed):
    try:
        from sklearn.base import clone
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as err:                                  # pragma: no cover
        raise ImportError(
            "This function needs scikit-learn: pip install econcomplex[ml]"
        ) from err

    if model == "gbm":
        est = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             random_state=seed)
    elif model == "rf":
        est = make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                   n_jobs=-1, random_state=seed))
    elif model == "logit":
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000))
    elif isinstance(model, str):
        raise ValueError("model must be 'gbm', 'rf', 'logit' or a scikit-learn classifier.")
    else:
        est = clone(model)
    if calibrate:
        est = CalibratedClassifierCV(est, method="isotonic", cv=3)
    return est


def _stack_target(target, index):
    """Target (R x C matrix or Series on cells) aligned to the feature rows."""
    if isinstance(target, pd.DataFrame):
        target = pd.Series(
            target.to_numpy(dtype=float).ravel(),
            index=pd.MultiIndex.from_product([target.index, target.columns]))
    return pd.Series(target).reindex(index).to_numpy(dtype=float)


def _fold_of(groups, n_splits, seed):
    """Fold number of each row, whole groups together, shuffled by seed."""
    codes, uniq = pd.factorize(np.asarray(groups))
    if len(uniq) < n_splits:
        raise ValueError(f"n_splits={n_splits} needs at least that many groups; got {len(uniq)}.")
    order = np.random.default_rng(seed).permutation(len(uniq))
    fold = np.empty(len(uniq), dtype=int)
    fold[order] = np.arange(len(uniq)) % n_splits
    return fold[codes]


def _region_groups(regions, groups):
    if groups is None:
        return np.asarray(regions)
    mapped = pd.Series(regions).map(pd.Series(groups))
    if mapped.isna().any():
        raise ValueError("groups must give a group for every region.")
    return mapped.to_numpy()


def _oof(X, y, fold, make):
    """Out-of-fold probabilities; a fold whose training set has one class is NaN."""
    p = np.full(len(y), np.nan)
    for k in np.unique(fold):
        train, test = fold != k, fold == k
        if y[train].min() == y[train].max():
            continue
        p[test] = make().fit(X[train], y[train]).predict_proba(X[test])[:, 1]
    return p


def predict_entry(
    features: pd.DataFrame,
    target: Union[pd.DataFrame, pd.Series],
    model="gbm",
    mode: str = "layers",
    new_features: Optional[pd.DataFrame] = None,
    groups: Optional[Union[dict, pd.Series]] = None,
    n_splits: int = 5,
    calibrate: bool = False,
    min_entries: int = 5,
    seed: Optional[int] = None,
    return_model: bool = False,
):
    """
    Predicted probability of entry, out of sample.

    Without `new_features` the function cross-validates by region: the
    regions are split into `n_splits` groups and each cell is predicted
    by a model that never saw its region. The result is an honest score
    to compare with relatedness density in `validate_entry`.

    With `new_features` the model is fitted on all cells and applied to
    the new table — fit on a past window (t0 -> t1), predict the next one
    from the features observed at t1.

    Two designs are available:

    ``mode="layers"``
        One model for all cells. `features` is the long table of
        `build_features` (densities by layer plus controls).
    ``mode="vector"``
        One model per activity. `features` is the R x C matrix of RCA at
        t0 and each model predicts the entry into one activity from the
        region's whole specialization vector, without any proximity
        matrix (Tacchella et al. 2023; Albora et al. 2023). Slower: C
        models per fold.

    Parameters
    ----------
    features : pd.DataFrame
        Long table indexed by (region, activity) (``mode="layers"``) or
        R x C matrix (``mode="vector"``).
    target : pd.DataFrame (R x C) or pd.Series on (region, activity)
        1 = entered, 0 = did not, NaN = not at risk (first output of
        `entry_targets`). Cells that are NaN are not used.
    model : {'gbm', 'rf', 'logit'} or scikit-learn classifier
        Gradient boosting (default), random forest, or logistic
        regression on standardized features — the transparent benchmark.
        Any classifier with ``predict_proba`` is accepted.
    mode : {'layers', 'vector'}
    new_features : pd.DataFrame, optional
        Same layout as `features`, for the cells to predict.
    groups : dict or pd.Series, optional
        Maps each region to a larger unit (state, country) to keep whole
        units out of the training folds. By default each region is its
        own group.
    n_splits : int
        Number of folds of the cross-validation.
    calibrate : bool
        Calibrate the probabilities (isotonic regression, 3 internal
        folds). Use it when the probability itself matters — e.g. when
        it is multiplied by a size in `potential_employment` — and not
        only the ranking.
    min_entries : int
        ``mode="vector"`` only: activities with fewer entries (or fewer
        non-entries) than this are not modelled and stay NaN.
    seed : int, optional
        Random seed (folds and models).
    return_model : bool
        Also return the model fitted on all cells (a dict
        ``{activity: model}`` in vector mode).

    Returns
    -------
    R x C DataFrame of probabilities, NaN for the cells that were not
    predicted; with ``return_model=True``, a tuple (probabilities, model).

    Notes
    -----
    Rough cost on 510 regions x 660 activities (about 250 thousand cells
    at risk): a few seconds for ``mode="layers"`` with gradient
    boosting; in ``mode="vector"`` about 0.6 s per activity with a random
    forest and 3 s with gradient boosting.

    Examples
    --------
    Compare a model with relatedness density, out of sample::

        target, at_risk = ec.entry_targets(ec.rca(mat_t0), ec.rca(mat_t1))
        X = ec.build_features(mat_t0, {"density": density_t0}, at_risk=at_risk)
        scores = {"gbm": ec.predict_entry(X, target, seed=1),
                  "density": density_t0}
        ec.validate_entry(scores, target)
    """
    if mode not in ("layers", "vector"):
        raise ValueError("mode must be 'layers' or 'vector'.")
    make = lambda: _make_model(model, calibrate, seed)     # noqa: E731

    if mode == "vector":
        return _predict_vector(features, target, new_features, groups, n_splits,
                               min_entries, seed, return_model, make)

    if features.index.nlevels != 2:
        raise ValueError("features must be indexed by (region, activity); see build_features.")
    y = _stack_target(target, features.index)
    use = ~np.isnan(y)
    X, y_use = features.to_numpy(dtype=float)[use], y[use]
    if y_use.min() == y_use.max():
        raise ValueError("target has a single class among the feature rows.")

    fitted = None
    if new_features is not None or return_model:
        fitted = make().fit(X, y_use)
    if new_features is None:
        regions = features.index.get_level_values(0)[use]
        fold = _fold_of(_region_groups(regions, groups), n_splits, seed)
        probs = pd.Series(_oof(X, y_use, fold, make), index=features.index[use])
    else:
        missing = [c for c in features.columns if c not in new_features.columns]
        if missing:
            raise ValueError(f"new_features lacks the column(s) {missing}.")
        probs = pd.Series(
            fitted.predict_proba(new_features[features.columns].to_numpy(dtype=float))[:, 1],
            index=new_features.index)

    out = probs.unstack()
    if isinstance(target, pd.DataFrame) and new_features is None:
        out = out.reindex(index=target.index, columns=target.columns)
    return (out, fitted) if return_model else out


def _predict_vector(features, target, new_features, groups, n_splits,
                    min_entries, seed, return_model, make):
    if not isinstance(features, pd.DataFrame) or not isinstance(target, pd.DataFrame):
        raise ValueError("mode='vector' needs R x C DataFrames for features and target.")
    X = features.to_numpy(dtype=float)
    Y = target.reindex(index=features.index, columns=features.columns).to_numpy(dtype=float)
    fold = _fold_of(_region_groups(features.index, groups), n_splits, seed)
    X_new = None
    if new_features is not None:
        X_new = new_features.reindex(columns=features.columns).to_numpy(dtype=float)

    shape = (len(new_features if X_new is not None else features), X.shape[1])
    probs = np.full(shape, np.nan)
    models = {}
    for j, activity in enumerate(features.columns):
        use = ~np.isnan(Y[:, j])
        y = Y[use, j]
        if min(y.sum(), (1 - y).sum()) < min_entries:
            continue
        if X_new is not None or return_model:
            models[activity] = make().fit(X[use], y)
        if X_new is None:
            probs[use, j] = _oof(X[use], y, fold[use], make)
        else:
            probs[:, j] = models[activity].predict_proba(X_new)[:, 1]

    index = new_features.index if X_new is not None else features.index
    out = pd.DataFrame(probs, index=index, columns=features.columns)
    return (out, models) if return_model else out


def _permutation_drops(X, y, fold, make, n_repeats, rng):
    """
    Fall in held-out AUC when each column is shuffled (features x draws),
    and the sign of the relation between each column and the prediction.
    """
    drops = [[] for _ in range(X.shape[1])]
    sign = np.zeros(X.shape[1])
    for k in np.unique(fold):
        train, test = fold != k, fold == k
        y_test = y[test] > 0.5
        if y[train].min() == y[train].max() or y_test.min() == y_test.max():
            continue
        fitted = make().fit(X[train], y[train])
        pred = fitted.predict_proba(X[test])[:, 1]
        base = _auc(y_test, pred)
        for j in range(X.shape[1]):
            col = X[test][:, j]
            ok = ~np.isnan(col)
            if ok.sum() > 2 and col[ok].std() > 0 and pred[ok].std() > 0:
                sign[j] += np.corrcoef(col[ok], pred[ok])[0, 1]
            for _ in range(n_repeats):
                shuffled = X[test].copy()
                shuffled[:, j] = rng.permutation(col)
                drops[j].append(base - _auc(y_test, fitted.predict_proba(shuffled)[:, 1]))
    return np.array(drops), np.sign(sign)


def _importance_table(drops, sign, names, alpha):
    out = pd.DataFrame({
        "importance": drops.mean(axis=1),
        "std": drops.std(axis=1),
        "p_value": (1 + (drops <= 0).sum(axis=1)) / (1 + drops.shape[1]),
        "direction": np.where(sign > 0, "+", np.where(sign < 0, "-", "")),
    }, index=names)
    out["significant"] = out["p_value"] < alpha
    return out.sort_values("importance", ascending=False, kind="stable")


def explain_entry(
    features: pd.DataFrame,
    target: Union[pd.DataFrame, pd.Series],
    model="gbm",
    mode: str = "layers",
    activities: Optional[list] = None,
    groups: Optional[Union[dict, pd.Series]] = None,
    n_splits: int = 5,
    n_repeats: int = 10,
    alpha: float = 0.05,
    min_entries: int = 5,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Which features carry the prediction of entry, out of sample.

    Permutation importance on held-out regions: the model is fitted on
    the training regions, and each feature is shuffled in turn in the
    test regions; the importance is the resulting fall in AUC. A feature
    the model does not rely on has importance around zero, so the share
    of shuffles that did not reduce the AUC is a p-value for "this
    feature adds nothing out of sample".

    Importance is conditional on the other features: two layers that
    carry the same information share it, and each may look weak even
    though the pair matters. Read it together with the AUC of each layer
    alone in `validate_entry`.

    ``mode="layers"``
        One table for the model on `build_features`: which layers and
        controls matter.
    ``mode="vector"``
        One table per target activity, for the models that predict entry
        from the region's whole RCA vector: which of the activities a
        region has make the entry into the target more likely. It is a
        data-driven counterpart of `density_contributions`. Slow: one
        cross-validated model and (number of activities x `n_repeats`)
        predictions per fold and target, so pass the targets of interest
        in `activities`.

    Parameters
    ----------
    features, target, model, groups, n_splits, min_entries, seed
        As in `predict_entry`.
    mode : {'layers', 'vector'}
    activities : list, optional
        ``mode="vector"`` only: target activities to explain (default:
        all that have enough entries).
    n_repeats : int
        Shuffles of each feature in each fold.
    alpha : float
        Significance level for the ``significant`` column.

    Returns
    -------
    pd.DataFrame indexed by feature (by (target, feature) in vector
    mode), most important first, with ``importance`` (mean fall in AUC),
    ``std``, ``p_value``, ``direction`` (``+`` when higher values of the
    feature go with a higher predicted probability, ``-`` otherwise) and
    ``significant``.
    """
    if mode not in ("layers", "vector"):
        raise ValueError("mode must be 'layers' or 'vector'.")
    rng = np.random.default_rng(seed)
    make = lambda: _make_model(model, False, seed)     # noqa: E731

    if mode == "layers":
        y = _stack_target(target, features.index)
        use = ~np.isnan(y)
        X, y = features.to_numpy(dtype=float)[use], y[use]
        regions = features.index.get_level_values(0)[use]
        fold = _fold_of(_region_groups(regions, groups), n_splits, seed)
        drops, sign = _permutation_drops(X, y, fold, make, n_repeats, rng)
        if drops.size == 0:
            raise ValueError("no fold had both entries and non-entries; use fewer splits.")
        return _importance_table(drops, sign, features.columns, alpha)

    if not isinstance(features, pd.DataFrame) or not isinstance(target, pd.DataFrame):
        raise ValueError("mode='vector' needs R x C DataFrames for features and target.")
    X = features.to_numpy(dtype=float)
    Y = target.reindex(index=features.index, columns=features.columns)
    fold = _fold_of(_region_groups(features.index, groups), n_splits, seed)
    tables = {}
    for activity in (features.columns if activities is None else activities):
        y = Y[activity].to_numpy(dtype=float)
        use = ~np.isnan(y)
        if min(y[use].sum(), (1 - y[use]).sum()) < min_entries:
            continue
        drops, sign = _permutation_drops(X[use], y[use], fold[use], make, n_repeats, rng)
        if drops.size:
            tables[activity] = _importance_table(drops, sign, features.columns, alpha)
    if not tables:
        raise ValueError("no target activity has enough entries and non-entries.")
    return pd.concat(tables, names=["target", "feature"])
