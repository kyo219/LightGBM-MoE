# coding: utf-8
"""Auto-K selection for MoE models — choose the optimal number of experts via information criteria or CV."""

import copy
import warnings

import numpy as np

from . import engine


def select_num_experts(
    params,
    train_set,
    k_range=range(2, 8),
    criterion="bic",
    num_boost_round=100,
    valid_sets=None,
    callbacks=None,
    nfold=5,
    seed=0,
    verbose=True,
):
    """Select the optimal number of experts (K) for a MoE model.

    Trains a MoE model for each candidate K and selects the best one according
    to the specified information criterion or cross-validation metric.

    Parameters
    ----------
    params : dict
        LightGBM parameters. Must include ``boosting="mixture"``.
    train_set : Dataset
        Training dataset. For IC-based criteria (bic/aic/icl), the dataset must
        retain raw data (use ``Dataset(X, y, free_raw_data=False)``).
    k_range : sequence of int, default range(2, 8)
        Candidate values of K (number of experts) to evaluate.
    criterion : str, default "bic"
        Selection criterion: ``"bic"``, ``"aic"``, ``"icl"``, or ``"cv_rmse"``.
    num_boost_round : int, default 100
        Number of boosting iterations per candidate model.
    valid_sets : list of Dataset or None, default None
        Validation datasets (used only for IC-based criteria, not cv_rmse).
    callbacks : list or None, default None
        Callbacks passed to ``train()`` or ``cv()``.
    nfold : int, default 5
        Number of CV folds (only used when ``criterion="cv_rmse"``).
    seed : int, default 0
        Random seed for CV fold splitting.
    verbose : bool, default True
        If True, print a results table after evaluation.

    Returns
    -------
    best_k : int
        The K value that minimises the criterion.
    results : dict
        Dictionary with keys ``"k_values"``, ``"criterion_values"``,
        ``"models"`` (list of trained Boosters, None for cv_rmse),
        and ``"details"`` (list of per-K detail dicts).
    """
    criterion = criterion.lower()
    if criterion not in ("bic", "aic", "icl", "cv_rmse"):
        raise ValueError(f"criterion must be 'bic', 'aic', 'icl', or 'cv_rmse', got '{criterion}'")

    p = copy.deepcopy(params)
    if p.get("boosting") != "mixture":
        raise ValueError("select_num_experts requires boosting='mixture' in params")

    k_values = list(k_range)
    criterion_values = []
    models = []
    details = []

    for k in k_values:
        p_k = copy.deepcopy(p)
        p_k["mixture_num_experts"] = k

        try:
            if criterion == "cv_rmse":
                crit, detail = _eval_cv_rmse(p_k, train_set, num_boost_round, nfold, seed, callbacks)
                models.append(None)
            else:
                crit, detail, model = _eval_ic(
                    p_k, train_set, num_boost_round, valid_sets, callbacks, criterion
                )
                models.append(model)
        except Exception as e:
            warnings.warn(f"K={k} failed: {e}")
            criterion_values.append(float("inf"))
            details.append({"error": str(e)})
            models.append(None)
            continue

        criterion_values.append(crit)
        details.append(detail)

    if all(v == float("inf") for v in criterion_values):
        raise ValueError("All K candidates failed during evaluation")

    best_idx = int(np.argmin(criterion_values))
    best_k = k_values[best_idx]

    results = {
        "k_values": k_values,
        "criterion_values": criterion_values,
        "models": models,
        "details": details,
    }

    if verbose:
        _print_table(k_values, criterion_values, details, best_idx, criterion)

    return best_k, results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _eval_cv_rmse(params, train_set, num_boost_round, nfold, seed, callbacks):
    """Evaluate a single K via cross-validation RMSE."""
    p = copy.deepcopy(params)
    p.setdefault("objective", "regression")
    p["metric"] = "rmse"

    cv_result = engine.cv(
        p,
        train_set,
        num_boost_round=num_boost_round,
        nfold=nfold,
        seed=seed,
        callbacks=callbacks,
        stratified=False,
    )

    # Find the RMSE key — pattern: "valid rmse-mean"
    rmse_key = None
    for key in cv_result:
        if "rmse" in key and "mean" in key:
            rmse_key = key
            break
    if rmse_key is None:
        raise RuntimeError(f"Could not find RMSE metric in cv results. Keys: {list(cv_result.keys())}")

    best_rmse = min(cv_result[rmse_key])
    best_iter = int(np.argmin(cv_result[rmse_key]))

    detail = {
        "cv_rmse": best_rmse,
        "best_iteration": best_iter,
    }
    return best_rmse, detail


def _eval_ic(params, train_set, num_boost_round, valid_sets, callbacks, criterion):
    """Evaluate a single K via information criterion (BIC/AIC/ICL)."""
    model = engine.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        callbacks=callbacks,
    )

    # Get X and y from train_set for prediction-based IC computation
    train_set.construct()
    X = train_set.get_data()
    if X is None:
        raise RuntimeError(
            "Cannot retrieve training data from Dataset (raw data was freed). "
            "Use Dataset(X, y, free_raw_data=False) for IC-based criteria."
        )
    y = train_set.get_label()
    N = len(y)

    # MSE on training set
    preds = model.predict(X)
    mse = float(np.mean((preds - y) ** 2))

    # Effective number of parameters: total leaf count across all trees
    model_dump = model.dump_model()
    n_params = _count_leaves(model_dump)

    # Log-likelihood under Gaussian assumption
    log_lik = -N / 2.0 * np.log(2 * np.pi * max(mse, 1e-15)) - N / 2.0

    # Compute criterion
    if criterion == "bic":
        crit_val = -2 * log_lik + n_params * np.log(N)
    elif criterion == "aic":
        crit_val = -2 * log_lik + 2 * n_params
    elif criterion == "icl":
        bic_val = -2 * log_lik + n_params * np.log(N)
        # Entropy penalty from gate probabilities
        regime_proba = model.predict_regime_proba(X)
        eps = 1e-12
        ent = -np.sum(regime_proba * np.log(regime_proba + eps))
        crit_val = bic_val + 2 * ent
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    detail = {
        "mse": mse,
        "n_params": n_params,
        "log_lik": log_lik,
        criterion: float(crit_val),
    }
    if criterion == "icl":
        detail["bic"] = float(bic_val)
        detail["gate_entropy_sum"] = float(ent)

    return float(crit_val), detail, model


def _count_leaves(model_dump):
    """Count total number of leaves across all trees in a dumped model."""
    total = 0
    for tree_info in model_dump.get("tree_info", []):
        total += _count_leaves_in_tree(tree_info.get("tree_structure", {}))
    return total


def _count_leaves_in_tree(node):
    """Recursively count leaves in a single tree node."""
    if "leaf_value" in node:
        return 1
    count = 0
    if "left_child" in node:
        count += _count_leaves_in_tree(node["left_child"])
    if "right_child" in node:
        count += _count_leaves_in_tree(node["right_child"])
    return count


def _print_table(k_values, criterion_values, details, best_idx, criterion):
    """Print a formatted results table."""
    crit_name = criterion.upper()
    is_ic = criterion in ("bic", "aic", "icl")

    print()
    print(f"Auto-K Selection (criterion={criterion})")
    print("=" * 50)

    if is_ic:
        print(f"  {'K':>3}  |  {'MSE':>10}  | {'#Params':>7} | {'Log-Lik':>10} | {crit_name:>12}")
        print("-" * 5 + "+" + "-" * 13 + "+" + "-" * 9 + "+" + "-" * 12 + "+" + "-" * 14)
        for i, k in enumerate(k_values):
            marker = "  <-- best" if i == best_idx else ""
            d = details[i]
            if "error" in d:
                print(f"  {k:>3}  |  {'FAILED':>10}  | {'':>7} | {'':>10} | {'inf':>12}{marker}")
            else:
                print(
                    f"  {k:>3}  |  {d['mse']:>10.6f}  | {d['n_params']:>7d} | {d['log_lik']:>10.1f} | "
                    f"{criterion_values[i]:>12.1f}{marker}"
                )
    else:
        # cv_rmse
        print(f"  {'K':>3}  |  {'CV-RMSE':>10}  | {'Best Iter':>9}")
        print("-" * 5 + "+" + "-" * 13 + "+" + "-" * 11)
        for i, k in enumerate(k_values):
            marker = "  <-- best" if i == best_idx else ""
            d = details[i]
            if "error" in d:
                print(f"  {k:>3}  |  {'FAILED':>10}  | {'':>9}{marker}")
            else:
                print(f"  {k:>3}  |  {d['cv_rmse']:>10.6f}  | {d['best_iteration']:>9d}{marker}")

    print()
