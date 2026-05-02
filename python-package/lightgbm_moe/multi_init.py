"""Multi-init restart for MixtureGBDT.

The empirical sweep in ``examples/em_strength_sweep.py`` showed that EM
trained from a single init lands in an init-specific local optimum: the
final regime structure barely overlaps across init schemes (pairwise ARI
< 0.15) and the per-init RMSE varies by 30%+. Even DA-EM only rescues
*bad* inits — for good inits it can hurt by softening the prior.

The right engineering response — taken straight from sklearn's
``GaussianMixture(n_init=10)`` playbook — is to train multiple independent
runs with different inits/seeds and pick the best by a score.

Speedups:

  * ``n_jobs`` runs trials in parallel processes (each gets a slice of the
    available OMP threads to avoid oversubscription).
  * ``prescreen_rounds`` / ``prescreen_keep`` runs all attempts cheaply
    first, drops the losers, and only retrains the survivors at full
    ``num_boost_round`` — successive halving in spirit. Best gains when
    most of the per-attempt cost is in the long tail of training rounds
    rather than init / setup.

Constraints with ``n_jobs > 1``:
  - ``score_metric`` must be the string ``"rmse"`` (callables aren't
    pickled across subprocess boundaries safely).
  - ``callbacks_factory`` is silently ignored (recorders are stateful and
    don't survive subprocess death). Use ``n_jobs=1`` to keep recorders.
"""

from __future__ import annotations

import copy
import os
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


# --------------------------------------------------------------------------- #
# Subprocess worker — must be a top-level function for pickling.              #
# --------------------------------------------------------------------------- #

def _trial_worker(args: Dict[str, Any]) -> Dict[str, Any]:
    """Run one training attempt; return a results dict with the model
    serialized as a string (Booster objects don't pickle cleanly).

    This function executes inside a subprocess when ``n_jobs > 1`` and
    inline in the main process when ``n_jobs == 1``. Either way it must
    only depend on top-level imports.
    """
    import time

    # Avoid OMP oversubscription when multiple workers run in parallel —
    # each gets a slice of the cores via params['num_threads'] (already
    # set by the orchestrator) and we forbid the OMP runtime from spawning
    # more threads than that on top.
    if "OMP_NUM_THREADS" not in os.environ:
        nth = args["params"].get("num_threads", 1)
        os.environ["OMP_NUM_THREADS"] = str(max(1, int(nth)))

    # Imports inside so spawn-method workers (Windows/macOS default) get a
    # clean LightGBM import within the subprocess.
    import numpy as _np

    from .basic import Dataset as _Dataset
    from .engine import train as _train_local

    p = args["params"]
    X = args["X"]
    y = args["y"]
    num_rounds = args["num_boost_round"]

    dset = _Dataset(X, label=y)
    t0 = time.time()
    booster = _train_local(p, dset, num_boost_round=num_rounds)
    elapsed = time.time() - t0

    # Score: RMSE on the supplied (X_score, y_score). We only support RMSE
    # in the worker because callable score_metrics aren't reliably picklable.
    if args.get("score_X") is None:
        # No score data — the orchestrator will compute it later in-process.
        score = float("nan")
    else:
        yhat = booster.predict(args["score_X"])
        score = float(_np.sqrt(_np.mean((args["score_y"] - yhat) ** 2)))

    return {
        "score": score,
        "model_str": booster.model_to_string(),
        "elapsed_s": elapsed,
        "init_scheme": args["init_scheme"],
        "seed": args["seed"],
        "trial_index": args["trial_index"],
    }


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


def _resolve_train_data(train_set):
    """Pull raw (X, y) out of a Dataset (or accept a tuple). Required for
    parallel execution since lightgbm Dataset doesn't pickle cleanly across
    subprocess boundaries."""
    if isinstance(train_set, tuple) and len(train_set) == 2:
        return np.asarray(train_set[0]), np.asarray(train_set[1])
    if isinstance(train_set, Dataset):
        if train_set.data is None or train_set.label is None:
            raise ValueError(
                "train_multi_init: the Dataset object has no cached data/label "
                "(constructed from a file path?). Pass raw (X, y) tuple "
                "instead, or use a Dataset built from numpy/pandas in memory."
            )
        return np.asarray(train_set.data), np.asarray(train_set.label)
    raise TypeError(
        f"train_set must be a Dataset or (X, y) tuple, got {type(train_set)!r}"
    )


def _build_trial_args(i, base_params, base_seed, init_schemes, X, y,
                       score_data, num_rounds, num_threads):
    p = copy.deepcopy(base_params)
    seed = base_seed + i * 100
    p["seed"] = seed
    if init_schemes:
        init_scheme = init_schemes[i % len(init_schemes)]
        p["mixture_init"] = init_scheme
    else:
        init_scheme = p.get("mixture_init", "quantile")
    p["verbose"] = -1
    if num_threads is not None:
        p["num_threads"] = num_threads
    return {
        "trial_index": i,
        "seed": seed,
        "init_scheme": init_scheme,
        "params": p,
        "X": X,
        "y": y,
        "score_X": score_data[0] if score_data is not None else None,
        "score_y": score_data[1] if score_data is not None else None,
        "num_boost_round": num_rounds,
    }


def _run_trials(
    trial_args_list,
    n_jobs,
    callbacks_factory=None,
    score_metric=None,
    score_data=None,
    verbose=False,
):
    """Run trials sequentially or in parallel. Returns list of result dicts
    with ``score``, ``model_str``, ``elapsed_s``, ``init_scheme``, ``seed``,
    ``trial_index``.

    When ``n_jobs > 1``, the worker is run in subprocesses — callbacks_factory
    and callable score_metric are ignored (documented constraint).
    """
    if n_jobs <= 1 and (callbacks_factory is not None
                        or callable(score_metric)):
        # In-process: support callbacks and custom metrics.
        results = []
        for args in trial_args_list:
            i = args["trial_index"]
            cbs = callbacks_factory(i) if callbacks_factory else None
            import time as _time
            dset = Dataset(args["X"], label=args["y"])
            t0 = _time.time()
            booster = _train(args["params"], dset,
                             num_boost_round=args["num_boost_round"],
                             callbacks=cbs)
            elapsed = _time.time() - t0
            score = _score_booster(booster, score_metric, score_data)
            results.append({
                "score": score,
                "model_str": booster.model_to_string(),
                "elapsed_s": elapsed,
                "init_scheme": args["init_scheme"],
                "seed": args["seed"],
                "trial_index": i,
                "_booster_in_process": booster,  # keep alive for caller
            })
            if verbose:
                print(f"[multi_init {i+1}] init={args['init_scheme']:<16s} "
                      f"seed={args['seed']:<6d} score={score:.4f} "
                      f"time={elapsed:.1f}s")
        return results

    if n_jobs <= 1:
        # In-process worker (same code path as parallel, just no pool).
        results = [_trial_worker(args) for args in trial_args_list]
        if verbose:
            for r in results:
                print(f"[multi_init {r['trial_index']+1}] "
                      f"init={r['init_scheme']:<16s} seed={r['seed']:<6d} "
                      f"score={r['score']:.4f} time={r['elapsed_s']:.1f}s")
        return results

    # Parallel via ProcessPoolExecutor
    from concurrent.futures import ProcessPoolExecutor, as_completed
    results = [None] * len(trial_args_list)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        future_to_idx = {
            pool.submit(_trial_worker, args): idx
            for idx, args in enumerate(trial_args_list)
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            r = fut.result()
            results[idx] = r
            if verbose:
                print(f"[multi_init {r['trial_index']+1}] "
                      f"init={r['init_scheme']:<16s} seed={r['seed']:<6d} "
                      f"score={r['score']:.4f} time={r['elapsed_s']:.1f}s")
    return results


def _hydrate_booster(result):
    """Reconstruct a Booster from a serialized model string. Used after
    parallel runs (in-process runs keep the original Booster alive)."""
    if "_booster_in_process" in result:
        return result["_booster_in_process"]
    return Booster(model_str=result["model_str"])


def train_multi_init(
    params: Dict[str, Any],
    train_set,
    num_boost_round: int = 100,
    n_inits: int = 5,
    init_schemes: Optional[Sequence[str]] = None,
    base_seed: Optional[int] = None,
    valid_sets: Optional[List[Dataset]] = None,
    valid_names: Optional[List[str]] = None,
    score_metric: Union[str, Callable[[Booster], float]] = "rmse",
    score_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    callbacks_factory: Optional[Callable[[int], List[Callable]]] = None,
    n_jobs: int = 1,
    prescreen_rounds: Optional[int] = None,
    prescreen_keep: Optional[int] = None,
    verbose: bool = False,
) -> MultiInitResult:
    """Train ``n_inits`` MoE models with varied inits/seeds, return the best.

    Parameters
    ----------
    params : dict
        Base LightGBM params; must include ``boosting='mixture'``.
    train_set : Dataset or (X, y) tuple
        Training data. A ``Dataset`` is unwrapped to its underlying numpy
        arrays. Required when ``n_jobs > 1`` (subprocess workers can't
        share a constructed Dataset).
    num_boost_round : int, default=100
        Boosting rounds for the (final) full-training pass per surviving
        attempt.
    n_inits : int, default=5
        Number of independent attempts to run.
    init_schemes : sequence of str or None
        Cycled as ``mixture_init`` across attempts. ``None`` keeps
        ``params['mixture_init']`` for every trial (variation is then
        purely from the seed).
    base_seed : int or None
        Each attempt uses ``base_seed + idx * 100``. Defaults to
        ``params.get('seed', 42)``.
    valid_sets, valid_names :
        Forwarded to ``lgb.train`` only in the in-process / sequential
        path (subprocesses construct their own Dataset).
    score_metric : "rmse" or callable, default="rmse"
        ``"rmse"`` works in any execution mode. Custom callables only
        work with ``n_jobs=1`` (they aren't pickled to subprocesses).
    score_data : (X_score, y_score) or None
        Required when ``score_metric='rmse'``.
    callbacks_factory : callable, default=None
        ``callbacks_factory(trial_idx)`` returns a fresh callback list per
        attempt. **Only honored when ``n_jobs=1``** — recorder state
        doesn't survive subprocess death.
    n_jobs : int, default=1
        Parallel workers. 1 = sequential (current behavior). Higher uses
        ``ProcessPoolExecutor``. Each worker gets ``num_threads //
        n_jobs`` OMP threads (clamped to ≥ 1) to avoid oversubscription.
    prescreen_rounds : int or None
        If set, run *all* ``n_inits`` attempts at this much shorter
        ``num_boost_round`` first, then advance only the top
        ``prescreen_keep`` to a full re-train at ``num_boost_round``.
        Successive-halving in spirit. Combine with ``n_jobs`` for the
        biggest wall-clock win on large problems.
    prescreen_keep : int or None
        Number of survivors to keep from the prescreen. Required when
        ``prescreen_rounds`` is given. Must be ≥ 1.
    verbose : bool
        Print per-trial progress.

    Returns
    -------
    MultiInitResult
        ``.best_booster`` is the model with the best score; ``.trials`` is
        the per-attempt summary (only the *full-trained* trials when
        prescreen is in effect); ``.summary_table()`` is human-readable.
    """
    if "boosting" in params and params["boosting"] != "mixture":
        raise ValueError(
            f"train_multi_init expects boosting='mixture', got "
            f"{params['boosting']!r}; use lgb.train for a non-MoE booster."
        )
    if score_metric == "rmse" and score_data is None:
        raise ValueError(
            "score_metric='rmse' needs score_data=(X, y); pass valid_sets "
            "with the booster's existing predict() chain or override the "
            "metric to a callable that computes from the booster alone."
        )
    if prescreen_rounds is not None and prescreen_keep is None:
        raise ValueError("prescreen_keep is required when prescreen_rounds is given")
    if prescreen_keep is not None and prescreen_keep < 1:
        raise ValueError(f"prescreen_keep must be >= 1, got {prescreen_keep}")
    if n_jobs > 1 and callable(score_metric):
        raise ValueError(
            "n_jobs > 1 does not support callable score_metric (pickle "
            "limitation). Use score_metric='rmse' or n_jobs=1."
        )
    if n_jobs > 1 and callbacks_factory is not None:
        # Soft warning rather than error — recorders won't survive subprocess
        # death but the run will still produce a valid best booster.
        import warnings
        warnings.warn(
            "callbacks_factory is ignored when n_jobs > 1 (callbacks are "
            "stateful and can't be returned from worker subprocesses). "
            "Re-run with n_jobs=1 to keep recorder snapshots.",
            stacklevel=2,
        )
        callbacks_factory = None

    base_seed = base_seed if base_seed is not None else int(params.get("seed", 42))

    metric_name = (
        score_metric if isinstance(score_metric, str) else "custom"
    )
    higher_is_better = metric_name in _HIGHER_IS_BETTER

    # Extract raw arrays (needed for parallel; cheap for sequential).
    X, y = _resolve_train_data(train_set)

    # Allocate OMP threads per worker so we don't oversubscribe physical cores.
    user_num_threads = params.get("num_threads", None)
    if n_jobs > 1:
        # Default: split available cores across workers.
        if user_num_threads is None:
            try:
                cpu = os.cpu_count() or 1
            except Exception:
                cpu = 1
            per_worker = max(1, cpu // n_jobs)
        else:
            per_worker = max(1, int(user_num_threads) // n_jobs)
    else:
        per_worker = user_num_threads  # leave alone; lgb default

    # ---- Phase 1: prescreen (or full, if no prescreen) ---------------------
    phase1_rounds = (prescreen_rounds if prescreen_rounds is not None
                     else num_boost_round)
    trial_args = [
        _build_trial_args(i, params, base_seed, init_schemes, X, y,
                          score_data, phase1_rounds, per_worker)
        for i in range(n_inits)
    ]
    if verbose and prescreen_rounds is not None:
        print(f"[multi_init] prescreen phase: {n_inits} trials × "
              f"{prescreen_rounds} rounds (n_jobs={n_jobs})")
    phase1_results = _run_trials(
        trial_args, n_jobs,
        callbacks_factory=callbacks_factory,
        score_metric=score_metric,
        score_data=score_data,
        verbose=verbose,
    )

    if prescreen_rounds is None:
        # No prescreen — phase 1 is the final pass.
        final_results = phase1_results
    else:
        # ---- Phase 2: full retrain on survivors ----------------------------
        phase1_sorted = sorted(
            phase1_results,
            key=lambda r: r["score"],
            reverse=higher_is_better,
        )
        survivors = phase1_sorted[:prescreen_keep]
        if verbose:
            kept = ", ".join(f"#{r['trial_index']}={r['score']:.4f}"
                             for r in survivors)
            print(f"[multi_init] phase 1 done; advancing top {prescreen_keep}: "
                  f"{kept}")
            print(f"[multi_init] full phase: {prescreen_keep} trials × "
                  f"{num_boost_round} rounds (n_jobs={n_jobs})")
        # Rebuild trial args for survivors with full rounds. Reuse the same
        # (init_scheme, seed) pair so prescreen ordering matches.
        full_args = []
        for r in survivors:
            full_args.append(_build_trial_args(
                r["trial_index"], params, base_seed, init_schemes, X, y,
                score_data, num_boost_round, per_worker,
            ))
        final_results = _run_trials(
            full_args, n_jobs,
            callbacks_factory=callbacks_factory,
            score_metric=score_metric,
            score_data=score_data,
            verbose=verbose,
        )

    # ---- Pick best ---------------------------------------------------------
    if higher_is_better:
        best_idx = int(np.argmax([r["score"] for r in final_results]))
    else:
        best_idx = int(np.argmin([r["score"] for r in final_results]))

    trials = [
        MultiInitTrial(
            index=r["trial_index"], seed=r["seed"],
            init_scheme=r["init_scheme"], score=r["score"],
            elapsed_s=r["elapsed_s"],
        )
        for r in final_results
    ]
    best_booster = _hydrate_booster(final_results[best_idx])

    return MultiInitResult(
        best_booster=best_booster,
        best_trial=trials[best_idx],
        trials=trials,
        score_metric=metric_name,
    )


__all__ = ["train_multi_init", "MultiInitResult", "MultiInitTrial"]
