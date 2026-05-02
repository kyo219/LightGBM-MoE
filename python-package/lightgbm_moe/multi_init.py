"""Multi-init restart for MixtureGBDT.

The empirical sweep in ``examples/em_strength_sweep.py`` showed that EM
trained from a single init lands in an init-specific local optimum: the
final regime structure barely overlaps across init schemes (pairwise ARI
< 0.15) and the per-init RMSE varies by 30%+. Even DA-EM only rescues
*bad* inits — for good inits it can hurt by softening the prior.

The right engineering response — taken straight from sklearn's
``GaussianMixture(n_init=10)`` playbook — is to train multiple independent
runs with different inits/seeds and pick the best by a score (RMSE, ELBO,
or a user-supplied callable). Each attempt is fully independent, so the
strategy parallels naturally if needed; we run sequentially here for
simplicity and predictable memory use.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .basic import Booster, Dataset, LightGBMError
from .engine import train as _train


@dataclass
class MultiInitTrial:
    """One attempt within a multi-init restart."""
    index: int
    seed: int
    init_scheme: str
    score: float
    elapsed_s: float = 0.0
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MultiInitResult:
    """Aggregate result of :func:`train_multi_init`."""
    best_booster: Booster
    best_trial: MultiInitTrial
    trials: List[MultiInitTrial]
    score_metric: str

    def summary_table(self) -> str:
        """Pretty-printed comparison across attempts."""
        lines = [f"{'idx':>3}  {'seed':>6}  {'init':>16s}  "
                 f"{self.score_metric:>10s}  {'time':>6s}"]
        for t in self.trials:
            marker = " ★" if t.index == self.best_trial.index else "  "
            lines.append(
                f"{t.index:>3}  {t.seed:>6}  {t.init_scheme:>16s}  "
                f"{t.score:>10.4f}  {t.elapsed_s:>5.1f}s{marker}"
            )
        return "\n".join(lines)


_LOWER_IS_BETTER = {"rmse", "mae", "loss", "marginal_log_lik_neg"}
_HIGHER_IS_BETTER = {"marginal_log_lik", "elbo", "r2", "accuracy"}


def _score_booster(
    booster: Booster,
    score_metric: Union[str, Callable[[Booster], float]],
    score_data: Optional[Tuple[np.ndarray, np.ndarray]],
) -> float:
    """Resolve the scoring function. RMSE is the safe default — works on
    any regression MoE without needing extra C++ plumbing for ELBO export."""
    if callable(score_metric):
        return float(score_metric(booster))

    if score_metric == "rmse":
        if score_data is None:
            raise ValueError(
                "score_metric='rmse' needs score_data=(X, y); pass valid_sets "
                "with the booster's existing predict() chain or override the "
                "metric to a callable that computes from the booster alone."
            )
        X, y = score_data
        yhat = booster.predict(X)
        return float(np.sqrt(np.mean((np.asarray(y) - yhat) ** 2)))

    raise ValueError(
        f"Unknown score_metric={score_metric!r}. Use 'rmse' or a callable."
    )


def train_multi_init(
    params: Dict[str, Any],
    train_set: Dataset,
    num_boost_round: int = 100,
    n_inits: int = 5,
    init_schemes: Optional[Sequence[str]] = None,
    base_seed: Optional[int] = None,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    score_metric: Union[str, Callable[[Booster], float]] = "rmse",
    score_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    callbacks_factory: Optional[Callable[[int], List[Callable]]] = None,
    verbose: bool = False,
) -> MultiInitResult:
    """Train ``n_inits`` MoE models with varied inits/seeds, return the best.

    Parameters
    ----------
    params : dict
        Base LightGBM params; must include ``boosting='mixture'``.
    train_set : lightgbm_moe.Dataset
        Training dataset. Reused across all attempts (no copy needed —
        Dataset is read-only at training time).
    num_boost_round : int, default=100
        Boosting rounds per attempt.
    n_inits : int, default=5
        Number of independent attempts to run.
    init_schemes : sequence of str or None
        If given, cycle through these as ``mixture_init`` for successive
        attempts. ``None`` keeps whatever ``params['mixture_init']`` is set
        to (so the variation is purely from the seed). A reasonable default
        for "discover the regime" use cases is
        ``['kmeans_features', 'gmm', 'quantile', 'random', 'kmeans_features']``
        — three deterministic schemes plus two randomized ones.
    base_seed : int or None
        Starting seed. Each attempt uses ``base_seed + idx * 100``. If
        ``None``, falls back to ``params.get('seed', 42)``.
    valid_sets, valid_names : optional
        Forwarded to ``lgb.train`` for early stopping / metric collection.
    score_metric : "rmse" or callable, default="rmse"
        How to compare attempts. Callables receive the trained booster and
        must return a float; lower is better unless the metric name is
        recognised as higher-is-better.
    score_data : (X, y) or None
        Required when ``score_metric='rmse'``. The held-out (or training)
        data on which RMSE is computed.
    callbacks_factory : callable, default=None
        ``callbacks_factory(trial_idx)`` should return a fresh list of
        callbacks for the given attempt. Use this when you want one
        ``RegimeEvolutionRecorder`` per trial (recorders are stateful, so
        a single instance can't safely be reused across attempts).
    verbose : bool
        Print per-trial progress.

    Returns
    -------
    MultiInitResult
        ``.best_booster`` is the model with the best score; ``.trials`` is
        the per-attempt summary; ``.summary_table()`` is human-readable.
    """
    import time

    if "boosting" in params and params["boosting"] != "mixture":
        raise ValueError(
            f"train_multi_init expects boosting='mixture', got "
            f"{params['boosting']!r}; use lgb.train for a non-MoE booster."
        )

    base_seed = base_seed if base_seed is not None else int(params.get("seed", 42))

    metric_name = (
        score_metric if isinstance(score_metric, str) else "custom"
    )
    higher_is_better = metric_name in _HIGHER_IS_BETTER

    trials: List[MultiInitTrial] = []
    boosters: List[Booster] = []

    for i in range(n_inits):
        attempt_params = copy.deepcopy(params)
        attempt_seed = base_seed + i * 100
        attempt_params["seed"] = attempt_seed

        if init_schemes is not None and len(init_schemes) > 0:
            attempt_init = init_schemes[i % len(init_schemes)]
            attempt_params["mixture_init"] = attempt_init
        else:
            attempt_init = attempt_params.get("mixture_init", "quantile")

        if not verbose:
            attempt_params["verbose"] = -1

        callbacks = callbacks_factory(i) if callbacks_factory else None

        t0 = time.time()
        booster = _train(
            attempt_params,
            train_set,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        elapsed = time.time() - t0

        try:
            score = _score_booster(booster, score_metric, score_data)
        except LightGBMError as exc:  # pragma: no cover — propagate train errors
            raise RuntimeError(f"trial {i} scoring failed: {exc}") from exc

        trials.append(MultiInitTrial(
            index=i, seed=attempt_seed, init_scheme=attempt_init,
            score=score, elapsed_s=elapsed,
        ))
        boosters.append(booster)

        if verbose:
            print(f"[multi_init {i+1}/{n_inits}] init={attempt_init:<16s} "
                  f"seed={attempt_seed:<6d} score={score:.4f} "
                  f"time={elapsed:.1f}s")

    if higher_is_better:
        best_idx = int(np.argmax([t.score for t in trials]))
    else:
        best_idx = int(np.argmin([t.score for t in trials]))

    return MultiInitResult(
        best_booster=boosters[best_idx],
        best_trial=trials[best_idx],
        trials=trials,
        score_metric=metric_name,
    )


__all__ = ["train_multi_init", "MultiInitResult", "MultiInitTrial"]
