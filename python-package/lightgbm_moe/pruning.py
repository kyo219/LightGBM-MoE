# coding: utf-8
"""Expert pruning and merging for trained MoE models."""

import numpy as np


class PrunedMoEModel:
    """A wrapper around a trained MoE Booster that masks pruned/merged experts.

    Provides the same prediction interface as ``Booster`` so it can be used
    directly with ``diagnose_moe()``.

    Attributes
    ----------
    original_model : Booster
        The original trained MoE model.
    active_mask : np.ndarray
        Boolean mask of shape ``(K_orig,)`` indicating which experts are active.
    merge_weights : dict
        Mapping ``{survivor_idx: {orig_idx: weight, ...}}`` for merged experts.
    merged_into : dict
        Mapping ``{victim_idx: survivor_idx}`` for merged (not pruned) experts.
    pruning_report : dict
        Summary of the pruning/merging operations performed.
    """

    def __init__(self, original_model, active_mask, merge_weights=None, merged_into=None, pruning_report=None):
        self.original_model = original_model
        self.active_mask = np.asarray(active_mask, dtype=bool)
        self.merge_weights = merge_weights or {}
        self.merged_into = merged_into or {}
        self.pruning_report = pruning_report or {}

    def num_experts(self):
        """Return the number of active experts."""
        return int(self.active_mask.sum())

    def is_mixture(self):
        """Return True (this is always a mixture model wrapper)."""
        return True

    def predict_regime_proba(self, X):
        """Return gate probabilities for active experts only, renormalised.

        Merge victims donate their gate mass to their survivor (so merging
        two identical experts is prediction-preserving); mass of genuinely
        pruned experts is redistributed proportionally via renormalisation.

        Returns
        -------
        np.ndarray of shape ``(N, K_active)``
        """
        full_proba = self.original_model.predict_regime_proba(X)  # (N, K_orig)
        if self.merged_into:
            full_proba = full_proba.copy()
            for victim, survivor in self.merged_into.items():
                full_proba[:, survivor] += full_proba[:, victim]
        active_proba = full_proba[:, self.active_mask]  # (N, K_active)
        row_sums = active_proba.sum(axis=1, keepdims=True)
        np.maximum(row_sums, 1e-15, out=row_sums)
        np.divide(active_proba, row_sums, out=active_proba)
        return active_proba

    def predict_expert_pred(self, X):
        """Return per-expert predictions for active experts (with merge weighting).

        Returns
        -------
        np.ndarray of shape ``(N, K_active)``
        """
        full_preds = self.original_model.predict_expert_pred(X)  # (N, K_orig)
        active_indices = np.where(self.active_mask)[0]

        result = np.empty((full_preds.shape[0], len(active_indices)), dtype=np.float64)
        merge_scratch = None
        for col, orig_idx in enumerate(active_indices):
            if orig_idx in self.merge_weights:
                # Accumulate directly into the result column; avoid an extra
                # N-sized temporary for every survivor group.
                sources = iter(self.merge_weights[orig_idx].items())
                src_idx, weight = next(sources)
                np.multiply(full_preds[:, src_idx], weight, out=result[:, col])
                for src_idx, weight in sources:
                    if merge_scratch is None:
                        merge_scratch = np.empty(full_preds.shape[0], dtype=np.float64)
                    np.multiply(full_preds[:, src_idx], weight, out=merge_scratch)
                    np.add(result[:, col], merge_scratch, out=result[:, col])
            else:
                result[:, col] = full_preds[:, orig_idx]

        return result

    def predict(self, X):
        """Return weighted MoE prediction using active experts only.

        Returns
        -------
        np.ndarray of shape ``(N,)``
        """
        proba = self.predict_regime_proba(X)  # (N, K_active)
        preds = self.predict_expert_pred(X)  # (N, K_active)
        return np.einsum("ij,ij->i", proba, preds)

    def predict_regime(self, X):
        """Return argmax regime index among active experts.

        Returns
        -------
        np.ndarray of shape ``(N,)``
        """
        proba = self.predict_regime_proba(X)
        return np.argmax(proba, axis=1).astype(np.int32)


def prune_experts(
    model,
    X,
    y=None,
    correlation_threshold=0.95,
    min_utilization=0.02,
    print_report=True,
):
    """Prune underutilised and redundant experts from a trained MoE model.

    Parameters
    ----------
    model : Booster
        A trained MoE model (``boosting="mixture"``).
    X : array-like
        Feature matrix used to evaluate expert utilisation and correlation.
    y : array-like or None, default None
        Target values (not used in the current implementation, reserved for
        future loss-based pruning).
    correlation_threshold : float, default 0.95
        Expert pairs with prediction correlation above this threshold are
        merged (the less-utilised expert is folded into the more-utilised one).
    min_utilization : float, default 0.02
        Experts with utilisation below this fraction of samples are pruned.
    print_report : bool, default True
        If True, print a summary of pruning actions.

    Returns
    -------
    PrunedMoEModel
        A wrapped model with pruned/merged experts masked out.
    """
    K = model.num_experts()
    regime_pred = model.predict_regime(X)
    expert_preds = model.predict_expert_pred(X)

    # Compute utilization per expert
    utilization = np.bincount(regime_pred, minlength=K).astype(np.float64)
    utilization /= max(len(regime_pred), 1)

    active = np.ones(K, dtype=bool)
    actions = []
    merge_weights = {}

    # Step 1: Prune underutilised experts
    for k in range(K):
        if utilization[k] < min_utilization:
            active[k] = False
            actions.append(f"  E{k}: underutilized ({utilization[k]:.1%} < {min_utilization:.0%})")

    # Step 2: Merge highly correlated expert pairs (among remaining active)
    active_indices = list(np.where(active)[0])
    merged_into = {}  # maps pruned index -> survivor index

    # Compute pairwise correlations among active experts
    pairs_to_merge = []
    corr_matrix = np.corrcoef(expert_preds, rowvar=False) if len(active_indices) > 1 else None
    for i_pos, i in enumerate(active_indices):
        for j in active_indices[i_pos + 1 :]:
            corr = float(corr_matrix[i, j])
            if corr > correlation_threshold:
                pairs_to_merge.append((i, j, corr))

    # Sort by correlation descending — merge most similar first
    pairs_to_merge.sort(key=lambda x: -x[2])

    for i, j, corr in pairs_to_merge:
        if not active[i] or not active[j]:
            continue  # One already pruned/merged
        # Keep the one with higher utilization
        if utilization[i] >= utilization[j]:
            survivor, victim = i, j
        else:
            survivor, victim = j, i

        active[victim] = False
        # Transitive remap (union-find style): experts previously merged into
        # the new victim now belong to its survivor, so chained merges
        # (A -> B, then B -> C) never lose A's membership.
        for prev_victim, prev_surv in merged_into.items():
            if prev_surv == victim:
                merged_into[prev_victim] = survivor
        merged_into[victim] = survivor

        actions.append(f"  E{victim}: merged into E{survivor} (corr={corr:.2f})")

    # Build merge weight maps once all merges are resolved, so every key is an
    # active survivor and each weight dict covers the survivor plus all of its
    # (transitively) merged victims.
    survivor_groups = {}
    for victim, survivor in merged_into.items():
        survivor_groups.setdefault(survivor, [survivor]).append(victim)
    for survivor, group in survivor_groups.items():
        total_util = sum(utilization[g] for g in group)
        if total_util > 0:
            weights = {g: float(utilization[g] / total_util) for g in group}
        else:
            weights = {g: 1.0 / len(group) for g in group}
        merge_weights[survivor] = weights

    # Safety: ensure at least one expert remains
    if not active.any():
        best = int(np.argmax(utilization))
        active[best] = True
        actions.append(f"  E{best}: kept as fallback (all experts would be pruned)")

    # Build active expert info
    active_indices_final = list(np.where(active)[0])
    active_utilizations = {}
    for idx in active_indices_final:
        if idx in merge_weights:
            merged_sources = list(merge_weights[idx].keys())
            total_u = sum(utilization[s] for s in merged_sources)
            includes = [s for s in merged_sources if s != idx]
            active_utilizations[idx] = (total_u, includes)
        else:
            active_utilizations[idx] = (utilization[idx], [])

    report = {
        "K_original": K,
        "K_active": int(active.sum()),
        "active_indices": active_indices_final,
        "utilization": utilization.tolist(),
        "actions": actions,
        "active_utilizations": active_utilizations,
    }

    if print_report:
        _print_prune_report(report)

    return PrunedMoEModel(
        original_model=model,
        active_mask=active,
        merge_weights=merge_weights,
        merged_into=merged_into,
        pruning_report=report,
    )


def merge_experts(
    model,
    X,
    expert_groups,
    print_report=True,
):
    """Manually merge specified groups of experts.

    Parameters
    ----------
    model : Booster
        A trained MoE model.
    X : array-like
        Feature matrix used to compute utilisation-based merge weights.
    expert_groups : list of list of int
        Each inner list specifies expert indices to merge. The first index
        in each group is the survivor.
    print_report : bool, default True
        If True, print a summary of merging actions.

    Returns
    -------
    PrunedMoEModel
        A wrapped model with merged experts.
    """
    K = model.num_experts()
    regime_pred = model.predict_regime(X)
    utilization = np.bincount(regime_pred, minlength=K).astype(np.float64)
    utilization /= max(len(regime_pred), 1)

    active = np.ones(K, dtype=bool)
    merge_weights = {}
    merged_into = {}
    actions = []

    for group in expert_groups:
        if len(group) < 2:
            continue
        survivor = group[0]
        victims = group[1:]

        total_util = sum(utilization[g] for g in group)
        if total_util > 0:
            weights = {g: float(utilization[g] / total_util) for g in group}
        else:
            weights = {g: 1.0 / len(group) for g in group}
        merge_weights[survivor] = weights

        for v in victims:
            active[v] = False
            merged_into[v] = survivor
            actions.append(f"  E{v}: merged into E{survivor}")

    active_indices_final = list(np.where(active)[0])

    report = {
        "K_original": K,
        "K_active": int(active.sum()),
        "active_indices": active_indices_final,
        "utilization": utilization.tolist(),
        "actions": actions,
    }

    if print_report:
        _print_merge_report(report)

    return PrunedMoEModel(
        original_model=model,
        active_mask=active,
        merge_weights=merge_weights,
        merged_into=merged_into,
        pruning_report=report,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _print_prune_report(report):
    """Print a human-readable pruning report."""
    print()
    print("Expert Pruning Report")
    print("=" * 40)
    print(f"Original: K={report['K_original']} experts -> Active: K={report['K_active']} experts")

    if report["actions"]:
        print()
        print("Actions:")
        for a in report["actions"]:
            print(a)

    print()
    print("Active experts:", [f"E{i}" for i in report["active_indices"]])
    for idx, (util, includes) in report["active_utilizations"].items():
        inc_str = f" (includes merged E{',E'.join(str(i) for i in includes)})" if includes else ""
        print(f"  E{idx}: utilization {util:.1%}{inc_str}")
    print()


def _print_merge_report(report):
    """Print a human-readable merge report."""
    print()
    print("Expert Merge Report")
    print("=" * 40)
    print(f"Original: K={report['K_original']} experts -> Active: K={report['K_active']} experts")

    if report["actions"]:
        print()
        print("Actions:")
        for a in report["actions"]:
            print(a)

    print()
    print("Active experts:", [f"E{i}" for i in report["active_indices"]])
    print()
