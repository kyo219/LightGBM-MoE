# coding: utf-8
"""MoE Regime Diagnostics — label-free diagnostics for mixture-of-experts models."""

import numpy as np


def diagnose_moe(model, X, y, print_report=True):
    """Diagnose whether a trained MoE model is functioning as an effective switching model.

    All metrics are computed without regime ground-truth labels.

    Parameters
    ----------
    model : lightgbm_moe.Booster
        A trained MoE (boosting="mixture") model.
    X : array-like
        Feature matrix used for diagnosis.
    y : array-like
        Target values used for computing prediction errors.
    print_report : bool, default True
        If True, print a human-readable diagnostic report.

    Returns
    -------
    dict
        Diagnostic metrics including gate entropy, expert specialization,
        routing gain, expert correlation, utilization, and an overall verdict.
    """
    y = np.asarray(y, dtype=np.float64)

    regime_proba = model.predict_regime_proba(X)
    expert_preds = model.predict_expert_pred(X)
    # The MoE combined prediction is exactly sum_k gate_prob_k * expert_pred_k
    # (MixtureGBDT::Predict); deriving it from the two arrays above avoids a
    # third full inference pass over the gate + all expert forests.
    moe_preds = np.einsum("ij,ij->i", regime_proba, expert_preds)
    # argmax of gate probabilities — identical to predict_regime(X),
    # without a redundant prediction pass.
    regime_pred = np.argmax(regime_proba, axis=1)
    K = regime_proba.shape[1]

    # =========================================================================
    # [1] Gate Entropy
    # =========================================================================
    eps = 1e-12
    log_proba = np.empty_like(regime_proba)
    np.add(regime_proba, eps, out=log_proba)
    np.log(log_proba, out=log_proba)
    entropy_per_sample = -np.einsum("ij,ij->i", regime_proba, log_proba)
    del log_proba
    max_entropy = np.log(K)
    mean_entropy = float(np.mean(entropy_per_sample))
    median_entropy = float(np.median(entropy_per_sample))
    confidence_ratio = float(np.mean(entropy_per_sample < 0.3 * max_entropy))

    # =========================================================================
    # [2] Expert Specialization
    # =========================================================================
    se_all = np.empty_like(expert_preds)
    np.subtract(expert_preds, y[:, None], out=se_all)
    np.square(se_all, out=se_all)  # (N, K)
    assigned_se = se_all[np.arange(len(y)), regime_pred]

    # Mean SE of non-assigned experts
    other_se_mean = (se_all.sum(axis=1) - assigned_se) / max(K - 1, 1)

    wins = assigned_se < other_se_mean
    specialization_rate = float(np.mean(wins))

    improvement_where_wins = np.zeros_like(assigned_se)
    improvement_mask = wins & (other_se_mean > 0)
    np.divide(
        other_se_mean - assigned_se,
        other_se_mean + eps,
        out=improvement_where_wins,
        where=improvement_mask,
    )
    mean_loss_improvement = float(np.sum(improvement_where_wins) / max(np.sum(wins), 1))

    # =========================================================================
    # [3] Routing Gain
    # =========================================================================
    moe_rmse = float(np.sqrt(np.mean((moe_preds - y) ** 2)))
    expert_rmses = np.sqrt(np.mean(se_all, axis=0)).tolist()
    best_single_rmse = min(expert_rmses)
    routing_gain = (best_single_rmse - moe_rmse) / (best_single_rmse + eps) * 100

    # =========================================================================
    # [4] Expert Correlation
    # =========================================================================
    if K > 1:
        corr_matrix = np.corrcoef(expert_preds, rowvar=False)
        corrs = corr_matrix[np.triu_indices(K, k=1)]
        expert_corr_max = float(np.max(corrs))
        expert_corr_min = float(np.min(corrs))
    else:
        expert_corr_max = expert_corr_min = 0.0
    expert_collapsed = expert_corr_max > 0.99

    # =========================================================================
    # [5] Expert Utilization
    # =========================================================================
    utilization = (np.bincount(regime_pred, minlength=K) / max(len(regime_pred), 1)).tolist()
    utilization_min = min(utilization)
    any_underutilized = utilization_min < 0.05

    # =========================================================================
    # [6] Verdict
    # =========================================================================
    if expert_collapsed or utilization_min < 0.01 or specialization_rate < 0.3:
        verdict = "Not Switching (Collapsed)"
    elif specialization_rate > 0.6 and confidence_ratio > 0.5 and routing_gain > 1.0 and not expert_collapsed:
        verdict = "Effective Switching"
    else:
        verdict = "Weak Switching"

    result = {
        "K": K,
        "mean_entropy": mean_entropy,
        "median_entropy": median_entropy,
        "max_entropy": float(max_entropy),
        "confidence_ratio": confidence_ratio,
        "entropy_per_sample": entropy_per_sample,
        "specialization_rate": specialization_rate,
        "mean_loss_improvement": mean_loss_improvement,
        "moe_rmse": moe_rmse,
        "expert_rmses": expert_rmses,
        "best_single_rmse": best_single_rmse,
        "routing_gain": routing_gain,
        "expert_corr_max": expert_corr_max,
        "expert_corr_min": expert_corr_min,
        "expert_collapsed": expert_collapsed,
        "utilization": utilization,
        "utilization_min": utilization_min,
        "any_underutilized": any_underutilized,
        "verdict": verdict,
    }

    if print_report:
        _print_report(result)

    return result


def _print_report(r):
    """Print a human-readable diagnostic report."""
    K = r["K"]
    print()
    print("MoE Regime Diagnostics")
    print("======================")
    print(f"Model: K={K} experts")

    print()
    print("[1] Gate Entropy")
    print(f"    Mean entropy       : {r['mean_entropy']:.3f} / {r['max_entropy']:.3f} (max)")
    print(f"    Confidence ratio   : {r['confidence_ratio']:.1%}")

    print()
    print("[2] Expert Specialization")
    print(f"    Specialization rate: {r['specialization_rate']:.1%}")
    print(f"    Mean loss improvement: {r['mean_loss_improvement']:.1%}")

    print()
    print("[3] Routing Gain")
    print(f"    MoE RMSE           : {r['moe_rmse']:.4f}")
    expert_strs = "  ".join(f"E{k}={r['expert_rmses'][k]:.4f}" for k in range(K))
    print(f"    Expert RMSEs       : {expert_strs}")
    print(f"    Routing gain       : {r['routing_gain']:+.1f}%")

    print()
    print("[4] Expert Correlation")
    print(f"    Pairwise corr      : {r['expert_corr_max']:.2f} (max)  {r['expert_corr_min']:.2f} (min)")
    print(f"    Collapsed          : {'Yes' if r['expert_collapsed'] else 'No'}")

    print()
    print("[5] Expert Utilization")
    util_strs = "   ".join(f"E{k}: {r['utilization'][k]:.1%}" for k in range(K))
    print(f"    {util_strs}")

    print()
    if r["verdict"] == "Effective Switching":
        print(f"Verdict: {r['verdict']} ✓")
    elif r["verdict"] == "Not Switching (Collapsed)":
        print(f"Verdict: {r['verdict']} ✗")
    else:
        print(f"Verdict: {r['verdict']} ~")
    print()
