"""Diagnostic visualization for the EM dynamics of a Mixture-of-Experts model.

The :class:`RegimeEvolutionRecorder` is a ``lgb.train`` callback that snapshots
the responsibility matrix ``r_ik`` at user-configurable iterations. After
training, :meth:`RegimeEvolutionRecorder.plot` renders a four-panel diagnostic
that shows how the regime assignment evolves from the initialization
(``mixture_init='gmm'`` / ``'kmeans_features'`` / ...) through the EM loop.

The renderer adapts to whether the data is time-ordered or tabular:

- **timeseries** (default when ``mixture_r_smoothing`` is ``markov``/``ema``/
  ``momentum`` or when the user passes a ``time_axis``): the tape's x-axis is
  time and the top panel is ``y(t)``. Drift across iterations along *time*
  reveals how EM redistributes regime assignments.
- **tabular**: samples are sorted by the *final* argmax regime, so the tape's
  x-axis becomes "regime cluster rank". Drift then reveals how many samples
  EM moved between init clusters and final clusters. The top panel turns into
  per-regime ``y`` violins.

Both modes share the same convergence diagnostics (mean responsibility entropy
and per-step flip rate) and the per-expert load chart.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

ArrayLike = Union[np.ndarray, Sequence[float]]


class RegimeEvolutionRecorder:
    """Snapshot training-time responsibilities for post-hoc EM diagnostics.

    Pass to ``lgb.train(..., callbacks=[recorder])``. After training is done,
    call :meth:`plot` to render the diagnostic figure, or inspect the raw
    snapshots via :attr:`snapshots`.

    Parameters
    ----------
    every : int, default=10
        Capture a snapshot every ``every`` iterations. ``every=1`` captures
        every iteration (memory-heavy on long runs; rely on ``max_snapshots``
        to keep the trace bounded).
    mode : {"auto", "timeseries", "tabular"}, default="auto"
        How :meth:`plot` should interpret the data axis. ``"auto"`` picks
        ``"timeseries"`` when either (a) the booster's params contain
        ``mixture_r_smoothing in {markov, ema, momentum}`` or (b) the user
        supplied a ``time_axis``. Otherwise ``"tabular"``.
    time_axis : array-like or {"row_index"} or None, default=None
        For timeseries mode: the x-axis values to use on the tape and the
        top ``y(t)`` panel. ``None`` and ``"row_index"`` both mean "use
        ``np.arange(num_data)``" — the right choice when the user has
        already pre-sorted their dataframe. Pass an array of timestamps
        (datetime or numeric) for explicit time labels.
    max_snapshots : int, default=50
        Cap on the number of snapshots retained. Once exceeded the recorder
        thins the trace by keeping a uniform stride from first to last
        snapshot — first and last are always preserved.
    capture_iter_zero : bool, default=True
        Force capture of iteration 0 even when ``every > 1``. With
        ``mixture_warmup_iters >= 1`` this snapshot is the unmodified
        :func:`InitResponsibilities` output (e.g. GMM init), which is the
        whole point of the "before EM" baseline.

    Attributes
    ----------
    snapshots : list of (int, np.ndarray)
        ``(iteration, r)`` pairs in chronological order. Each ``r`` has shape
        ``(num_data, num_experts)`` and is a copy (safe to keep past further
        training).
    """

    def __init__(
        self,
        every: int = 10,
        mode: str = "auto",
        time_axis: Optional[Union[ArrayLike, str]] = None,
        max_snapshots: int = 50,
        capture_iter_zero: bool = True,
    ) -> None:
        if every < 1:
            raise ValueError("`every` must be >= 1")
        if mode not in ("auto", "timeseries", "tabular"):
            raise ValueError(f"unknown mode {mode!r}")
        if max_snapshots < 2:
            raise ValueError("`max_snapshots` must be >= 2")
        self.every = int(every)
        self.mode = mode
        self.time_axis = time_axis
        self.max_snapshots = int(max_snapshots)
        self.capture_iter_zero = bool(capture_iter_zero)
        self.snapshots: List[Tuple[int, np.ndarray]] = []
        self._params_seen: Optional[dict] = None

    # ---- LightGBM callback protocol ----------------------------------------
    @property
    def order(self) -> int:
        # Run after early stopping (order=0) and most other callbacks so the
        # recorder sees the *final* responsibilities of the iteration.
        return 30

    @property
    def before_iteration(self) -> bool:
        return False

    def __call__(self, env: Any) -> None:
        i = env.iteration
        if not (i % self.every == 0 or (self.capture_iter_zero and i == 0)):
            return
        try:
            r = env.model.get_responsibilities()
        except Exception:  # noqa: BLE001 — diagnostic; never break training
            return
        if r.size == 0:
            return
        # Cache params on first capture for later mode auto-detection.
        if self._params_seen is None:
            self._params_seen = dict(env.params) if env.params else {}
        self.snapshots.append((i, r.copy()))
        self._maybe_trim()

    def _maybe_trim(self) -> None:
        if len(self.snapshots) <= self.max_snapshots:
            return
        # Keep a uniform stride; np.linspace endpoints always include the first
        # and last entry, which is what we want for the trace.
        keep = np.unique(
            np.linspace(0, len(self.snapshots) - 1, self.max_snapshots).astype(int)
        )
        self.snapshots = [self.snapshots[j] for j in keep]

    # ---- Derived metrics ---------------------------------------------------
    @property
    def num_snapshots(self) -> int:
        return len(self.snapshots)

    @property
    def iterations(self) -> np.ndarray:
        return np.array([it for it, _ in self.snapshots])

    def regime_argmax(self) -> np.ndarray:
        """``(num_snapshots, num_data)`` argmax regime per sample per snapshot."""
        if not self.snapshots:
            raise RuntimeError("no snapshots recorded")
        return np.stack([np.argmax(r, axis=1) for _, r in self.snapshots], axis=0)

    def expert_load(self) -> np.ndarray:
        """``(num_snapshots, num_experts)`` mean responsibility per expert."""
        if not self.snapshots:
            raise RuntimeError("no snapshots recorded")
        return np.stack([r.mean(axis=0) for _, r in self.snapshots], axis=0)

    def mean_entropy(self) -> np.ndarray:
        """``(num_snapshots,)`` mean per-sample entropy. Drops as EM converges."""
        out = []
        for _, r in self.snapshots:
            r_safe = np.clip(r, 1e-12, 1.0)
            out.append(float(-(r_safe * np.log(r_safe)).sum(axis=1).mean()))
        return np.array(out)

    def flip_rate(self) -> np.ndarray:
        """``(num_snapshots - 1,)`` fraction of samples whose argmax changed."""
        am = self.regime_argmax()
        if am.shape[0] < 2:
            return np.array([])
        return np.mean(am[1:] != am[:-1], axis=1)

    # ---- Plot --------------------------------------------------------------
    def plot(
        self,
        y: Optional[ArrayLike] = None,
        X: Optional[ArrayLike] = None,
        mode: Optional[str] = None,
        params: Optional[dict] = None,
        figsize: Tuple[float, float] = (11.0, 8.0),
        cmap: str = "tab10",
        title: Optional[str] = None,
    ) -> Any:
        """Render the four-panel diagnostic.

        Parameters
        ----------
        y : array-like or None
            Target values, length = num_data. Used in the top panel
            (``y(t)`` for timeseries, per-regime violin for tabular). Optional.
        X : array-like or None
            Currently unused; reserved for future tabular feature overlays.
        mode : {"auto", "timeseries", "tabular"} or None
            Override the recorder's stored mode for this call.
        params : dict or None
            Override the params dict used for ``mode="auto"`` resolution.
            Defaults to whatever the recorder saw during training.
        figsize : tuple, default=(11, 8)
        cmap : str, default="tab10"
            Discrete colormap for argmax regimes. ``tab10`` works well up to
            K=10 experts. Use ``tab20`` for larger K.
        title : str or None
            Suptitle. If ``None``, a default is generated.

        Returns
        -------
        matplotlib.figure.Figure
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm, ListedColormap
            from matplotlib.gridspec import GridSpec
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "matplotlib is required for RegimeEvolutionRecorder.plot()"
            ) from exc

        if not self.snapshots:
            raise RuntimeError(
                "No snapshots recorded — did you pass the recorder to "
                "lgb.train(callbacks=[...])?"
            )

        argmax_matrix = self.regime_argmax()  # (S, N)
        num_snap, num_data = argmax_matrix.shape
        num_experts = self.snapshots[0][1].shape[1]

        if y is not None:
            y_arr = np.asarray(y).ravel()
            if y_arr.shape[0] != num_data:
                raise ValueError(
                    f"y length {y_arr.shape[0]} != num_data {num_data}"
                )
        else:
            y_arr = None

        resolved_mode = mode or self._resolve_mode(params or self._params_seen)
        time_x = self._resolve_time_axis(num_data)

        # In tabular mode reorder samples by final argmax so cluster
        # structure is visible as horizontal bands across the tape.
        if resolved_mode == "tabular":
            order_idx = np.argsort(argmax_matrix[-1])
            argmax_plot = argmax_matrix[:, order_idx]
            x_label = "sample (sorted by final regime)"
            x_for_top = np.arange(num_data)
            x_for_top_label = x_label
            y_for_top = y_arr[order_idx] if y_arr is not None else None
        else:
            argmax_plot = argmax_matrix
            order_idx = np.arange(num_data)
            x_label = "time / sample index"
            x_for_top = time_x
            x_for_top_label = "time"
            y_for_top = y_arr

        # Discrete colormap with exactly num_experts colors.
        base_cmap = plt.get_cmap(cmap, max(num_experts, 2))
        listed = ListedColormap(base_cmap(np.arange(num_experts)))
        bounds = np.arange(num_experts + 1) - 0.5
        norm = BoundaryNorm(bounds, ncolors=num_experts)

        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 2.0, 1.0])

        # --- Panel ① top: y(t) line OR per-regime violin -------------------
        ax_top = fig.add_subplot(gs[0, :])
        if y_for_top is not None:
            if resolved_mode == "tabular":
                final_am = argmax_matrix[-1]
                groups = [y_arr[final_am == k] for k in range(num_experts)]
                positions = list(range(num_experts))
                # Filter empty regimes from the violin (matplotlib chokes).
                non_empty = [(p, g) for p, g in zip(positions, groups) if len(g) > 0]
                if non_empty:
                    pp, gg = zip(*non_empty)
                    parts = ax_top.violinplot(gg, positions=pp, showmeans=True)
                    for k_idx, body in zip(pp, parts["bodies"]):
                        body.set_facecolor(listed(k_idx))
                        body.set_alpha(0.7)
                ax_top.set_xticks(range(num_experts))
                ax_top.set_xticklabels([f"k={k}" for k in range(num_experts)])
                ax_top.set_xlabel("regime (final argmax)")
                ax_top.set_ylabel("y")
                ax_top.set_title("Per-regime y distribution")
            else:
                ax_top.plot(x_for_top, y_arr, color="black", lw=0.7, alpha=0.85)
                ax_top.set_xlabel(x_for_top_label)
                ax_top.set_ylabel("y")
                ax_top.set_title("Target series")
                ax_top.margins(x=0)
        else:
            ax_top.text(
                0.5, 0.5, "(no y supplied)",
                ha="center", va="center", transform=ax_top.transAxes,
                color="gray", fontsize=11,
            )
            ax_top.set_axis_off()

        # --- Panel ② tape: argmax(r) over iterations -----------------------
        ax_tape = fig.add_subplot(gs[1, :])
        # imshow with sample on x, snapshot on y. extent so x lines up with
        # whatever x_for_top uses; for tabular it's just rank.
        x_left, x_right = (0.0, float(num_data))
        if resolved_mode == "timeseries" and np.issubdtype(time_x.dtype, np.number):
            x_left, x_right = float(time_x[0]), float(time_x[-1])
        ax_tape.imshow(
            argmax_plot,
            aspect="auto",
            interpolation="nearest",
            cmap=listed,
            norm=norm,
            extent=(x_left, x_right, num_snap - 0.5, -0.5),
            origin="upper",
        )
        ax_tape.set_yticks(range(num_snap))
        ax_tape.set_yticklabels(
            [f"iter={it}" for it in self.iterations],
            fontsize=8,
        )
        ax_tape.set_xlabel(x_label)
        ax_tape.set_ylabel("snapshot")
        # Inline colorbar for regime IDs.
        cbar = fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=listed),
            ax=ax_tape, ticks=range(num_experts),
            shrink=0.6, pad=0.01,
        )
        cbar.set_label("regime k = argmax r")

        # --- Panel ③ entropy + flip rate ------------------------------------
        ax_diag = fig.add_subplot(gs[2, 0])
        iters = self.iterations
        ent = self.mean_entropy()
        ln1 = ax_diag.plot(
            iters, ent, "o-", color="C3", label="mean entropy of r",
        )
        ax_diag.set_xlabel("iteration")
        ax_diag.set_ylabel("mean entropy", color="C3")
        ax_diag.tick_params(axis="y", labelcolor="C3")
        ax_diag.set_ylim(bottom=0)
        ax_diag.grid(alpha=0.3)

        if num_snap >= 2:
            ax_flip = ax_diag.twinx()
            fr = self.flip_rate()
            mid_iters = (iters[:-1] + iters[1:]) / 2
            ln2 = ax_flip.plot(
                mid_iters, fr, "s--", color="C0", alpha=0.7,
                label="argmax flip rate",
            )
            ax_flip.set_ylabel("flip rate", color="C0")
            ax_flip.tick_params(axis="y", labelcolor="C0")
            ax_flip.set_ylim(0, max(0.05, float(fr.max()) * 1.1))
            lines = ln1 + ln2
            ax_diag.legend(lines, [l.get_label() for l in lines],
                           loc="upper right", fontsize=8)
        ax_diag.set_title("EM convergence")

        # --- Panel ④ expert load over iterations (stacked area) ------------
        ax_load = fig.add_subplot(gs[2, 1])
        load = self.expert_load()  # (S, K)
        colors = [listed(k) for k in range(num_experts)]
        ax_load.stackplot(
            iters, load.T,
            labels=[f"expert {k}" for k in range(num_experts)],
            colors=colors, alpha=0.85,
        )
        ax_load.set_xlabel("iteration")
        ax_load.set_ylabel("mean responsibility")
        ax_load.set_ylim(0, 1)
        ax_load.set_title("Expert load (collapse warning if a band → 0 or 1)")
        ax_load.legend(loc="upper right", fontsize=8, ncol=min(num_experts, 3))
        ax_load.axhline(1.0 / num_experts, color="gray", lw=0.7,
                        ls=":", label="uniform")

        if title is None:
            title = (
                f"Regime evolution — mode={resolved_mode}, "
                f"K={num_experts}, snapshots={num_snap}, "
                f"iters {iters[0]}→{iters[-1]}"
            )
        fig.suptitle(title)
        return fig

    # ---- helpers -----------------------------------------------------------
    def _resolve_mode(self, params: Optional[dict]) -> str:
        if self.mode != "auto":
            return self.mode
        smoothing = (params or {}).get("mixture_r_smoothing", "none")
        if smoothing in ("markov", "ema", "momentum"):
            return "timeseries"
        if self.time_axis is not None:
            return "timeseries"
        return "tabular"

    def _resolve_time_axis(self, n: int) -> np.ndarray:
        if self.time_axis is None or (
            isinstance(self.time_axis, str) and self.time_axis == "row_index"
        ):
            return np.arange(n)
        if isinstance(self.time_axis, str) and self.time_axis.startswith("column:"):
            raise NotImplementedError(
                "time_axis='column:N' is not supported yet — pass an explicit "
                "array instead"
            )
        ax = np.asarray(self.time_axis)
        if ax.shape[0] != n:
            raise ValueError(
                f"time_axis length {ax.shape[0]} != num_data {n}"
            )
        return ax


__all__ = ["RegimeEvolutionRecorder"]
