"""Bayesian Optimization agent for multi-phase experiments.

Ported from the legacy ``temp/rtm_pymmcore/agents/bo_optimization_gpax.py``.
The BO internals (GP fitting, acquisition functions, scaling, penalty) are
unchanged.  The integration layer is adapted to the new Controller-based
agent architecture:

* Inherits :class:`InterPhaseAgent` instead of the legacy ``Agent``.
* Uses ``controller.run_experiment()`` / ``continue_experiment()`` /
  ``finish_experiment()`` for experiment control.
* Template methods renamed to match the event-based API.

Requires ``gpax`` and ``jax`` as dependencies (not imported at module level
so that importing this module without them installed does not crash).
"""

from __future__ import annotations

import dataclasses
import os
import time
from abc import abstractmethod
from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from faro.agents.base import InterPhaseAgent

if TYPE_CHECKING:
    from faro.core.data_structures import RTMEvent


# ---------------------------------------------------------------------------
# Dataclasses for BO configuration
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class BO_Parameter:
    """A controllable parameter for Bayesian Optimization."""

    name: str
    bounds: tuple
    param_type: str = "float"  # 'int' or 'float'
    spacing: float = 10.0
    log_scale: bool = False
    initial_value: float = None


@dataclasses.dataclass
class BO_Covariate:
    """A non-controllable observed variable (e.g., baseline cell area)."""

    name: str
    bounds: tuple = None
    log_scale: bool = False


@dataclasses.dataclass
class BO_Objective:
    """The single objective metric to optimize."""

    name: str
    goal: str  # 'minimize' or 'maximize'
    log_scale: bool = False


# ---------------------------------------------------------------------------
# Bounds-based scaler (normalizes to [-1, 1] using parameter bounds)
# ---------------------------------------------------------------------------


class StandardScalerBounds:
    """Custom scaler that normalizes using parameter bounds.

    For a feature with bounds ``(min, max)`` we set
    ``mean = (min + max) / 2`` and ``std = (max - min) / 2`` so that
    values at the bounds map to roughly +/-1.  Supports per-feature log
    transforms.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.log_scale = None

    def fit(self, X, bounds=None, log_scale=None):
        import jax.numpy as jnp

        X = jnp.asarray(X)
        self.log_scale = log_scale

        if log_scale is not None:
            for i, use_log in enumerate(log_scale):
                if use_log:
                    X = X.at[:, i].set(jnp.log(X[:, i]))

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)

        if bounds is not None and log_scale is not None:
            for i, bound in enumerate(bounds):
                if bound is not None and log_scale[i]:
                    minv, maxv = jnp.log(bound[0]), jnp.log(bound[1])
                    self.mean_ = self.mean_.at[i].set((minv + maxv) / 2.0)
                    self.std_ = self.std_.at[i].set((maxv - minv) / 2.0)
                elif bound is not None:
                    minv, maxv = bound
                    self.mean_ = self.mean_.at[i].set((minv + maxv) / 2.0)
                    self.std_ = self.std_.at[i].set((maxv - minv) / 2.0)

        self.std_ = jnp.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        import jax.numpy as jnp

        X = jnp.asarray(X)
        if self.log_scale is not None:
            for i, use_log in enumerate(self.log_scale):
                if use_log:
                    X = X.at[:, i].set(jnp.log(X[:, i]))
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, bounds=None, log_scale=None):
        return self.fit(X, bounds=bounds, log_scale=log_scale).transform(X)

    def inverse_transform(self, X):
        import jax.numpy as jnp

        X = jnp.asarray(X)
        X = X * self.std_ + self.mean_
        if self.log_scale is not None:
            for i, use_log in enumerate(self.log_scale):
                if use_log:
                    X = X.at[:, i].set(jnp.exp(X[:, i]))
        return X


# ---------------------------------------------------------------------------
# BOptGPAX — Bayesian Optimization with GPax
# ---------------------------------------------------------------------------


class BOptGPAX(InterPhaseAgent):
    """Bayesian Optimization agent using GPax (Gaussian Process with JAX).

    This is an inter-phase agent: it owns the experiment loop and calls
    ``controller.run_experiment()`` / ``controller.continue_experiment()``
    between BO iterations.

    Subclasses must implement two template methods:

    * :meth:`_create_events_for_cycle` -- build ``RTMEvent`` objects for
      one experiment cycle with the given BO parameters.
    * :meth:`_preprocess_results` -- extract the objective metric from
      the pipeline's tracking output.

    Args:
        storage_path: Path where the pipeline stores tracks/ parquet files.
        parameters_to_optimize: List of :class:`BO_Parameter` defining the
            controllable search space.
        objective_metric: :class:`BO_Objective` defining what to optimize.
        bo_covariates: List of :class:`BO_Covariate` (uncontrolled variables).
        n_iterations: Number of BO iterations after initial samples.
        acquisition_function: ``"ei"`` (Expected Improvement) or ``"ucb"``
            (Upper Confidence Bound).
        ucb_beta: Exploration parameter for UCB.
        ei_xi: Exploration offset for EI (decays over iterations).
        ei_xi_final: Final value of ``ei_xi`` after decay.
        ei_xi_decay_fraction: Fraction of iterations over which ``ei_xi``
            decays from ``ei_xi`` to ``ei_xi_final``.
        ei_num_samples: Number of posterior samples for EI/UCB computation.
        ei_noiseless: Whether to use noiseless predictions.
        n_cov_samples: Number of covariate samples for marginalisation.
        cov_marginalization_mode: ``"joint"`` (default, recommended) or
            ``"mesh"``.  ``"joint"`` samples *rows* from ``df_results`` so
            each covariate sample is a real (correlated) combination seen in
            the data — the total number of scenarios per grid point is
            ``n_cov_samples``, regardless of how many covariates you have.
            ``"mesh"`` resamples each covariate independently and takes
            the outer product, giving ``n_cov_samples ** n_covariates``
            scenarios — this grows explosively and is usually much slower
            without being more accurate (it ignores cross-covariate
            correlations in the observed data).
        penalty: ``None``, ``"delta"``, or ``"inverse_distance"`` to
            discourage re-evaluation at/near recent points.
        penalty_factor: Strength of the inverse-distance penalty.
        n_conditions_per_iter: Number of distinct parameter sets ("conditions")
            evaluated per BO phase.  ``1`` (default) preserves the legacy
            single-condition behaviour.  ``> 1`` enables **batch BO**: each
            phase picks ``n_conditions_per_iter`` parameter sets via
            sequential greedy acquisition with local penalisation, splits
            the configured FOVs evenly between them, and evaluates them
            simultaneously.  Subclasses must override
            :meth:`_create_events_for_batch` (and typically
            :meth:`_preprocess_results`) to map conditions to FOV subsets.
        batch_penalty_factor: Penalty factor used during the inner sequential
            greedy loop in :meth:`_select_batch_parameters` to keep the
            ``n_conditions_per_iter`` selected points apart.  Only used
            when ``n_conditions_per_iter > 1``.
        n_initial_phases: Number of phases at the start of the experiment
            that use **farthest-point initial sampling** instead of BO
            acquisition.  Default ``1`` (legacy behaviour — only the
            first phase is initial).  Increase to ``2`` (or more) to
            give a high-dimensional GP more distinct controllable
            coordinates to learn lengthscales from before trusting EI.
            A rule of thumb is to aim for roughly ``2 * n_controllable``
            distinct coordinates before BO kicks in; with
            ``n_conditions_per_iter=3`` and 2 controllable dims, that
            means ``n_initial_phases=2`` (6 coordinates total).  All
            ``n_initial_phases * n_conditions_per_iter`` initial picks
            are generated in one pass on the first initial phase via
            farthest-point sampling on ``x_unmeasured``, then doled out
            ``n_conditions_per_iter`` at a time.  This gives mutually
            well-spread initial coordinates across phases rather than
            independently-spread batches that might cluster.
        initial_exploration_cap: Optional fraction in ``(0, 1]`` that, if set,
            restricts farthest-point sampling during the initial phases to
            the lower-``cap`` slice of each parameter's range.  For a
            parameter with bounds ``[lo, hi]``, only grid points with
            ``x <= lo + cap * (hi - lo)`` are eligible.  Example:
            ``cap=0.75`` limits initial picks to the bottom 75% of each
            parameter axis, useful when the upper corner of the grid would
            violate a cycle-time or physical budget and you only want to
            avoid it during random exploration.  Default ``None`` = no cap
            (use the full grid, legacy behaviour).
        verbose: If True, plot acquisition landscape each iteration.
    """

    def __init__(
        self,
        storage_path: str,
        parameters_to_optimize: List[BO_Parameter],
        objective_metric: BO_Objective,
        bo_covariates: List[BO_Covariate] = None,
        n_iterations: int = 10,
        true_marginalization: bool = False,
        acquisition_function: str = "ei",
        ucb_beta: float = 4.0,
        ei_xi: float = 0.0,
        ei_xi_final: Optional[float] = None,
        ei_xi_decay_fraction: float = 0.7,
        ei_num_samples: int = 16,
        ei_noiseless: bool = True,
        n_cov_samples: int = 10,
        cov_marginalization_mode: str = "joint",
        penalty: Optional[str] = None,
        penalty_factor: float = 1.0,
        n_conditions_per_iter: int = 1,
        batch_penalty_factor: float = 2.0,
        n_initial_phases: int = 1,
        initial_exploration_cap: Optional[float] = None,
        prior=None,
        verbose: bool = False,
    ):
        super().__init__(storage_path)
        if bo_covariates is None:
            bo_covariates = []

        self.n_iterations = n_iterations
        self.iteration = 0
        self.parameters_to_optimize = parameters_to_optimize
        self.objective_metric = objective_metric
        self.model = None

        self.fovs: list[int] = []

        self.x_covariate = None
        self._seed = 42

        self.x = None
        self.y = None

        self.x_total_linespace = self._generate_total_parameter_space()
        self.x_unmeasured = self._generate_total_parameter_space()
        self.x_performed_experiments = None
        self.bo_covariates = bo_covariates
        self.true_marginalization = true_marginalization

        self.acquisition_function = acquisition_function.lower().strip()
        if self.acquisition_function not in {"ucb", "ei"}:
            raise ValueError(
                f"Unsupported acquisition_function='{acquisition_function}'. "
                f"Use 'ucb' or 'ei'."
            )
        self.ucb_beta = float(ucb_beta)
        self.ei_xi = float(ei_xi)
        self.ei_xi_final = (
            float(ei_xi_final) if ei_xi_final is not None else float(ei_xi)
        )
        self.ei_xi_decay_fraction = float(ei_xi_decay_fraction)
        self.ei_num_samples = int(ei_num_samples)
        self.ei_noiseless = bool(ei_noiseless)
        self.n_cov_samples = int(n_cov_samples)
        cov_marginalization_mode = cov_marginalization_mode.lower().strip()
        if cov_marginalization_mode not in {"joint", "mesh"}:
            raise ValueError(
                f"Unknown cov_marginalization_mode={cov_marginalization_mode!r}, "
                f"expected 'joint' or 'mesh'."
            )
        self.cov_marginalization_mode = cov_marginalization_mode
        self._rng = np.random.default_rng(self._seed)

        if penalty is not None:
            penalty = penalty.lower().strip()
            if penalty not in {"delta", "inverse_distance"}:
                raise ValueError(
                    f"Unsupported penalty='{penalty}'. "
                    f"Use None, 'delta', or 'inverse_distance'."
                )
        self.penalty = penalty
        self.penalty_factor = float(penalty_factor)

        if n_conditions_per_iter < 1:
            raise ValueError(
                f"n_conditions_per_iter must be >= 1, got {n_conditions_per_iter}"
            )
        self.n_conditions_per_iter = int(n_conditions_per_iter)
        self.batch_penalty_factor = float(batch_penalty_factor)

        if n_initial_phases < 0:
            raise ValueError(f"n_initial_phases must be >= 0, got {n_initial_phases}")
        # n_initial_phases=0 is valid when df_results is pre-seeded with
        # prior observations (e.g. memory/bootstrap from a previous run).
        # The initial-sampling code path is skipped entirely in that case.
        self.n_initial_phases = int(n_initial_phases)

        if initial_exploration_cap is not None:
            cap = float(initial_exploration_cap)
            if not (0.0 < cap <= 1.0):
                raise ValueError(
                    f"initial_exploration_cap must be in (0, 1], got {cap}"
                )
            self.initial_exploration_cap = cap
        else:
            self.initial_exploration_cap = None
        # Lazy-initialised FIFO of pre-generated farthest-point picks for
        # the initial phases.  Populated on the first initial phase with
        # ``n_initial_phases * n_conditions_per_iter`` mutually-spread
        # coordinates, then consumed ``n_conditions_per_iter`` at a time.
        self._initial_pick_queue: list[dict] | None = None

        # Per-phase state populated by run_one_phase / run for batch BO.
        # Subclasses (or their _create_events_for_batch /
        # _preprocess_results overrides) read these to know which condition
        # a given FOV belongs to and which phase id is currently active.
        self._phase_counter: int = 0
        self._fov_index_offset: int = 0
        self._current_phase_id: int = 0
        self._current_condition_map: dict[int, dict] = {}

        self._acquisition_used_this_round = self.acquisition_function

        self.verbose = verbose
        self.save_phase_logs = True
        self._iteration_means: list[float] = []

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def add_fovs(self, fovs: list[int]):
        """Set the FOV indices used by this agent."""
        self.fovs = list(fovs)

    # ------------------------------------------------------------------
    # Template methods — subclasses must implement these
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_events_for_cycle(self, parameters: dict) -> list[RTMEvent]:
        """Create RTMEvents for one experiment cycle with given BO parameters.

        This replaces the legacy ``_create_df_acquire_for_exp_cycle``.

        Args:
            parameters: Dict mapping parameter names to selected values
                (e.g. ``{"stim_angle": 0.3, "stim_radial": 0.5}``).

        Returns:
            List of :class:`RTMEvent` objects for this cycle.
        """
        ...

    @abstractmethod
    def _preprocess_results(self, fov_tracks: dict[int, pd.DataFrame]) -> pd.DataFrame:
        """Extract the objective metric from pipeline tracking output.

        Args:
            fov_tracks: ``{fov_index: df_tracked}`` for each FOV.

        Returns:
            DataFrame with columns matching all :attr:`parameters_to_optimize`
            names, all :attr:`bo_covariates` names, and the
            :attr:`objective_metric` name.  Each row is one observation.

        Note:
            In **batch BO** mode (``n_conditions_per_iter > 1``) the base
            class populates :attr:`_current_phase_id` and
            :attr:`_current_condition_map` before calling this method, so
            subclasses can look up which condition a FOV belongs to.
        """
        ...

    def _create_events_for_batch(self, param_list: list[dict]) -> list:
        """Build RTMEvents for one *batch* of conditions.

        Default implementation only handles the single-condition case
        (``len(param_list) == 1``) and delegates to
        :meth:`_create_events_for_cycle` so existing single-condition
        subclasses keep working unchanged.

        Subclasses that set ``n_conditions_per_iter > 1`` must override
        this method to:

        1. Split :attr:`fov_positions` into ``len(param_list)`` chunks
           (one per condition) -- the conventional layout is equal-sized
           chunks of ``len(self.fovs) // len(param_list)`` FOVs.
        2. Build per-condition events, applying any per-condition FOV-index
           offset (``self._fov_index_offset``) so phase-specific FOV ids
           remain unique across phases.
        3. Populate :attr:`_current_condition_map` so
           :meth:`_preprocess_results` can map each FOV back to the
           parameter set that generated it.
        4. Return the merged event list (typically passed through
           ``faro.core.utils.apply_fov_batching``).
        """
        if len(param_list) != 1:
            raise NotImplementedError(
                f"Default _create_events_for_batch only supports single-condition "
                f"batches (got {len(param_list)} conditions). Override "
                f"_create_events_for_batch in your subclass for batch BO."
            )
        return self._create_events_for_cycle(param_list[0])

    def _on_phase_complete(self, df_new: pd.DataFrame, phase_id: int) -> None:
        """Hook called at the end of each batch BO phase.

        Default implementation is a no-op.  Override to add live
        plotting, checkpoint saving, or other side effects.  Called from
        :meth:`_run_one_phase_batch` and :meth:`_run_batch_loop` *after*
        the new observations have been concatenated into
        :attr:`df_results`, so ``self.df_results`` reflects the full
        history at call time.

        Args:
            df_new: The fresh observations from this phase only.
            phase_id: Zero-based phase index.
        """
        pass

    # ------------------------------------------------------------------
    # Phase logging
    # ------------------------------------------------------------------

    def _open_phase_log(self, phase_id: int):
        """Return a context manager that tees stdout/stderr to a log file.

        Log files are written to ``{storage_path}/logs/phase_{phase_id:03d}.log``.
        Output is **not** suppressed — it still appears in the notebook /
        console as usual.
        """
        import contextlib
        import sys

        log_dir = os.path.join(self.storage_path, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"phase_{phase_id:03d}.log")

        class _Tee:
            def __init__(self, original, log_file):
                self._original = original
                self._log = log_file

            def write(self, msg):
                self._original.write(msg)
                self._log.write(msg)
                self._log.flush()

            def flush(self):
                self._original.flush()
                self._log.flush()

        @contextlib.contextmanager
        def _ctx():
            f = open(log_path, "w", encoding="utf-8")
            old_out, old_err = sys.stdout, sys.stderr
            sys.stdout = _Tee(old_out, f)
            sys.stderr = _Tee(old_err, f)
            try:
                yield
            finally:
                sys.stdout = old_out
                sys.stderr = old_err
                f.close()

        return _ctx()

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | None = None) -> str | None:
        """Save the trained GP + scalers + metadata for post-hoc analysis.

        Pickles a single dict via ``joblib`` containing everything you
        need to (a) recreate predictions on new (parameter, covariate)
        combinations, (b) compute per-dimension sensitivity from the
        ARD lengthscale posterior, and (c) plot the marginalised
        landscape outside the experiment notebook.  The same scalers
        used during training are bundled, so any external code can do
        ``x_full_scaled = payload['x_scaler'].transform(x_full)`` then
        ``payload['model'].predict(...)`` directly.

        Called automatically after each phase from
        :class:`OscillationBO._on_phase_complete`, so even if the
        experiment crashes mid-way the most recent fit is on disk.
        Can also be invoked manually after ``composed_agent.run()``
        with a custom path if you want a separate snapshot.

        Args:
            path: Destination ``.joblib`` file.  Defaults to
                ``{storage_path}/models/bo_model_iter_{iteration:03d}.joblib``
                so every BO phase gets its own snapshot.

        Returns:
            The path written to, or ``None`` if no model has been fit
            yet (e.g. called during the initial-spread phases) or if
            saving failed for any reason (errors are caught and printed
            so they cannot interrupt the experiment loop).
        """
        import joblib

        if self.model is None:
            print("  [save_model] no model to save (BO has not fit yet)")
            return None

        if path is None:
            models_dir = os.path.join(self.storage_path, "models")
            os.makedirs(models_dir, exist_ok=True)
            path = os.path.join(
                models_dir, f"bo_model_iter_{self.iteration:03d}.joblib"
            )

        try:
            rng = getattr(self, "_rng_key_predict", None)

            # Serialize the GP's trained state (MCMC/SVI samples) rather
            # than the live model object.  The gpax model holds JAX-traced
            # kernel functions that can't be pickled, but the posterior
            # samples (a plain dict of numpy arrays) are all you need to
            # recreate predictions later via gpax.ExactGP.predict().
            model_state = None
            if self.model is not None:
                samples = getattr(self.model, "mcmc", None)
                if samples is not None:
                    # ExactGP (MCMC): .mcmc is a numpyro MCMC object;
                    # .get_samples() returns a dict[str, jax.Array].
                    # Store the kernel name as a string, not the kernel
                    # function itself (which is a JAX-traced callable
                    # that can't be pickled).
                    kernel_attr = getattr(self.model, "kernel", "Matern")
                    kernel_name = (
                        kernel_attr
                        if isinstance(kernel_attr, str)
                        else getattr(kernel_attr, "__name__", "Matern")
                    )
                    model_state = {
                        "backend": "ExactGP",
                        "kernel": kernel_name,
                        "input_dim": getattr(self.model, "input_dim", None),
                        "samples": {
                            k: np.asarray(v) for k, v in samples.get_samples().items()
                        },
                        "X_train": (
                            np.asarray(self.model.X_train)
                            if self.model.X_train is not None
                            else None
                        ),
                        "y_train": (
                            np.asarray(self.model.y_train)
                            if self.model.y_train is not None
                            else None
                        ),
                    }
                else:
                    # viSparseGP / viGP: store params + guide state
                    params = getattr(self.model, "kernel_params", None)
                    model_state = {
                        "backend": type(self.model).__name__,
                        "kernel_params": (
                            {k: np.asarray(v) for k, v in params.items()}
                            if params is not None
                            else None
                        ),
                    }

            payload = dict(
                model_state=model_state,
                x_scaler=getattr(self, "_x_scaler", None),
                y_scaler=getattr(self, "_y_scaler", None),
                rng_key_predict=(np.asarray(rng) if rng is not None else None),
                parameter_names=[p.name for p in self.parameters_to_optimize],
                parameter_bounds=[tuple(p.bounds) for p in self.parameters_to_optimize],
                parameter_log_scale=[
                    bool(p.log_scale) for p in self.parameters_to_optimize
                ],
                covariate_names=[c.name for c in self.bo_covariates],
                covariate_bounds=[
                    tuple(c.bounds) if c.bounds is not None else None
                    for c in self.bo_covariates
                ],
                objective_name=self.objective_metric.name,
                objective_goal=self.objective_metric.goal,
                df_results=(
                    self.df_results.copy()
                    if getattr(self, "df_results", None) is not None
                    else None
                ),
                iteration=self.iteration,
                n_iterations=self.n_iterations,
                n_initial_phases=self.n_initial_phases,
                n_conditions_per_iter=self.n_conditions_per_iter,
                acquisition_function=self.acquisition_function,
                cov_marginalization_mode=self.cov_marginalization_mode,
                n_cov_samples=self.n_cov_samples,
            )
            joblib.dump(payload, path)
            return path
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"  Warning: could not save BO model to {path}: {exc}")
            return None

    # ------------------------------------------------------------------
    # Per-phase API (composable) + main experiment loop
    # ------------------------------------------------------------------

    def _ensure_results_df(self) -> None:
        """Lazily create the per-instance results DataFrame."""
        if not hasattr(self, "df_results") or self.df_results is None:
            self.df_results = pd.DataFrame()

    def run_one_phase(
        self,
        phase_id: int,
        fov_positions: list | None = None,
        fovs: list[int] | None = None,
    ) -> dict | None:
        """Run a single BO phase.

        Dispatches to :meth:`_run_one_phase_single` or
        :meth:`_run_one_phase_batch` based on
        :attr:`n_conditions_per_iter`.

        For single-condition BO (``n_conditions_per_iter == 1``) this is
        one BO iteration: pick the next parameter set (initial spread for
        early phases, GP-acquisition otherwise), build events, run/continue
        the experiment, wait for the pipeline, read tracks and append
        observations to :attr:`df_results`.

        For batch BO (``n_conditions_per_iter > 1``) it picks
        ``n_conditions_per_iter`` parameter sets via sequential greedy
        acquisition with local penalisation, splits the configured FOVs
        between them, evaluates them simultaneously, and shifts FOV ids by
        ``phase_id * len(fov_positions)`` so each phase gets globally
        unique FOV indices (avoids stale tracker state across phases).

        This method is the primary integration point for
        :class:`ComposedAgent` — composing BO with a per-phase
        :class:`PreExperimentAgent` (e.g. ``FOVFinderAgent``) just means
        calling ``finder.run()`` and then this method in a loop.

        Args:
            phase_id: Zero-based phase index.  ``phase_id == 0`` triggers
                ``controller.run_experiment``; later phases use
                ``continue_experiment``.
            fov_positions: Optional list of stage positions for this
                phase.  When provided, replaces the agent's current FOVs.
            fovs: Optional list of FOV indices.  Single-condition mode
                only -- in batch mode the FOV ids are derived from
                ``phase_id`` and ignored if supplied.

        Returns:
            ``{"params": ..., "df_new": ..., "phase_id": phase_id}`` or
            ``None`` if no results were produced.  In batch mode
            ``params`` is a list of condition dicts.
        """
        if self.n_conditions_per_iter > 1:
            return self._run_one_phase_batch(phase_id, fov_positions=fov_positions)
        return self._run_one_phase_single(
            phase_id, fov_positions=fov_positions, fovs=fovs
        )

    def _run_one_phase_single(
        self,
        phase_id: int,
        fov_positions: list | None = None,
        fovs: list[int] | None = None,
    ) -> dict | None:
        """Single-condition variant of :meth:`run_one_phase`."""
        self._ensure_results_df()

        # Update FOVs if positions were supplied for this phase.
        if fov_positions is not None:
            self.fov_positions = list(fov_positions)
            if fovs is None:
                fovs = list(range(len(fov_positions)))
            self.add_fovs(fovs)

        if not self.fovs:
            raise ValueError(
                "No FOVs configured. Call add_fovs() before run_one_phase(), "
                "or pass fov_positions=... to this call."
            )

        # --- Pick next parameter set ----------------------------------
        # See _run_one_phase_batch for the rationale behind n_initial_phases
        # and the pre-generated farthest-point queue.  In single-condition
        # mode (n_conditions_per_iter == 1) the queue holds exactly one
        # pick per initial phase.
        do_initial = self.iteration < self.n_initial_phases or (
            self.df_results.empty or len(self.df_results) < 4
        )
        if do_initial:
            if self._initial_pick_queue is None or len(self._initial_pick_queue) == 0:
                remaining_initial_phases = max(
                    1, self.n_initial_phases - self.iteration
                )
                total_initial_picks = (
                    remaining_initial_phases * self.n_conditions_per_iter
                )
                all_initial, _ = self._select_initial_samples(k=total_initial_picks)
                if isinstance(all_initial, dict):
                    all_initial = [all_initial]
                self._initial_pick_queue = list(all_initial)
            params = self._initial_pick_queue[0]
            self._initial_pick_queue = self._initial_pick_queue[1:]
            print(
                f"=== Phase {phase_id}: initial sample "
                f"{self.iteration + 1}/{self.n_initial_phases}: {params} ==="
            )
        else:
            params = self._determine_next_parameters(
                self.df_results, verbose=self.verbose
            )
            print(f"=== Phase {phase_id}: BO-selected {params} ===")

        # --- Build events and run -------------------------------------
        events = self._create_events_for_cycle(params)
        if phase_id == 0:
            self.controller.run_experiment(events, validate=False)
        else:
            self.controller.continue_experiment(events, validate=False)

        # --- Wait + collect results -----------------------------------
        self._wait_for_pipeline(timeout=self._phase_pipeline_timeout())
        tracks = {fov: self.read_tracks(fov) for fov in self.fovs}
        df_new = self._preprocess_results(tracks)
        if not df_new.empty:
            self.df_results = pd.concat([self.df_results, df_new], ignore_index=True)
            self._iteration_means.append(
                float(df_new[self.objective_metric.name].mean())
            )

        if not self.df_results.empty:
            self.x = self._extract_x_from_df(self.df_results)
            self.y = self._extract_y_from_df(self.df_results)

        self.iteration += 1
        return {"params": params, "df_new": df_new, "phase_id": phase_id}

    def _run_one_phase_batch(
        self,
        phase_id: int,
        fov_positions: list | None = None,
    ) -> dict | None:
        """Batch variant of :meth:`run_one_phase`.

        Picks :attr:`n_conditions_per_iter` parameter sets, splits the
        FOVs evenly between them, builds events via
        :meth:`_create_events_for_batch`, runs/continues the experiment,
        and reads phase-suffixed tracks.

        FOV indices are shifted by ``phase_id * len(fov_positions)`` so
        each phase gets globally unique FOV ids.  Subclasses' overrides of
        :meth:`_create_events_for_batch` should respect
        ``self._fov_index_offset`` when building per-condition position
        lists.
        """
        self._ensure_results_df()

        if fov_positions is None:
            raise ValueError(
                "Batch BO run_one_phase requires fov_positions "
                "(typically supplied per-phase by a pre-phase agent like "
                "FOVFinderAgent)."
            )
        n_per_phase = len(fov_positions)
        if n_per_phase % self.n_conditions_per_iter != 0:
            raise ValueError(
                f"Number of FOVs per phase ({n_per_phase}) must be divisible "
                f"by n_conditions_per_iter ({self.n_conditions_per_iter})."
            )

        self.fov_positions = list(fov_positions)
        self._fov_index_offset = phase_id * n_per_phase
        self.fovs = list(
            range(self._fov_index_offset, self._fov_index_offset + n_per_phase)
        )
        self._phase_counter = phase_id
        self._current_phase_id = phase_id

        # --- Capture phase output to a log file ----------------------
        _log_ctx = None
        if self.save_phase_logs:
            _log_ctx = self._open_phase_log(phase_id)
            _log_ctx.__enter__()

        try:
            return self._run_one_phase_batch_inner(phase_id, batch_params=None)
        finally:
            if _log_ctx is not None:
                _log_ctx.__exit__(None, None, None)

    def _run_one_phase_batch_inner(
        self, phase_id: int, *, batch_params=None
    ) -> dict | None:
        """Inner body of :meth:`_run_one_phase_batch` (extracted for logging)."""

        # --- Pick batch params (initial spread or BO acquisition) -----
        # Initial phases: farthest-point spread sampling on the full
        # discrete grid.  All picks across every initial phase are
        # pre-generated in ONE call so they are mutually farthest-point
        # spread, then doled out n_conditions_per_iter at a time.  This
        # is strictly better than independent spread-sampling per phase
        # (which can cluster the later phases near earlier picks when
        # the farthest-point seed lands by chance near a pre-existing
        # selection).  The fallback (df_results too tiny for GP fit) is
        # a safety net in case n_initial_phases is set to 1 and the
        # first phase's measurements all fail.
        do_initial = self.iteration < self.n_initial_phases or (
            self.df_results.empty or len(self.df_results) < 4
        )
        if do_initial:
            if self._initial_pick_queue is None or len(self._initial_pick_queue) == 0:
                remaining_initial_phases = max(
                    1, self.n_initial_phases - self.iteration
                )
                total_initial_picks = (
                    remaining_initial_phases * self.n_conditions_per_iter
                )
                all_initial, _ = self._select_initial_samples(k=total_initial_picks)
                if isinstance(all_initial, dict):
                    all_initial = [all_initial]
                self._initial_pick_queue = list(all_initial)
            batch_params = self._initial_pick_queue[: self.n_conditions_per_iter]
            self._initial_pick_queue = self._initial_pick_queue[
                self.n_conditions_per_iter :
            ]
            print(
                f"=== Phase {phase_id}: initial batch "
                f"{self.iteration + 1}/{self.n_initial_phases} ==="
            )
        else:
            batch_params = self._select_batch_parameters(
                self.df_results, n_conditions=self.n_conditions_per_iter
            )
            print(f"=== Phase {phase_id}: BO-selected batch ===")

        for i, p in enumerate(batch_params):
            print(f"  Condition {i}: {p}")

        # Stash the selected picks so _plot_live can annotate them
        self._current_batch_picks = [
            [p[param.name] for param in self.parameters_to_optimize]
            for p in batch_params
        ]

        # --- Pre-measurement plot (shows what BO picked) ---------------
        if hasattr(self, "plot_live") and self.plot_live and not do_initial:
            try:
                self._plot_live(
                    self.df_results,
                    f"Phase {phase_id} — selected conditions (before measurement)",
                    save_subdir="before",
                )
            except Exception as exc:
                print(f"  Warning: pre-measurement plot failed: {exc}")

        # --- Build events and run -------------------------------------
        events = self._create_events_for_batch(batch_params)
        if phase_id == 0:
            self.controller.run_experiment(events, validate=False)
        else:
            # Fresh FOVs every phase → timesteps restart from 0.
            # TODO: ideally the controller should auto-detect this from
            # the FOV indices (new p → no offset, reused p → offset).
            # For now offset_timepoints is an explicit flag.
            self.controller.continue_experiment(
                events, validate=False, offset_timepoints=False
            )

        # --- Wait + collect results -----------------------------------
        self._wait_for_pipeline(timeout=self._phase_pipeline_timeout())
        tracks = {fov: self.read_tracks(fov, phase_id=phase_id) for fov in self.fovs}
        df_new = self._preprocess_results(tracks)
        if not df_new.empty:
            self.df_results = pd.concat([self.df_results, df_new], ignore_index=True)
            self._iteration_means.append(
                float(df_new[self.objective_metric.name].mean())
            )
            self._on_phase_complete(df_new, phase_id)

        if not self.df_results.empty:
            self.x = self._extract_x_from_df(self.df_results)
            self.y = self._extract_y_from_df(self.df_results)

        self.iteration += 1
        return {"params": batch_params, "df_new": df_new, "phase_id": phase_id}

    def _select_batch_parameters(
        self, df_results: pd.DataFrame, n_conditions: int
    ) -> list[dict]:
        """Select ``n_conditions`` distinct parameter sets via sequential greedy.

        After the first acquisition the inverse-distance penalty is
        temporarily enabled so subsequent picks are pushed away from
        already-chosen points within the same batch.  The original
        :attr:`penalty` / :attr:`penalty_factor` are restored on return.

        Used by :meth:`_run_one_phase_batch` (and the legacy ``run`` loop)
        when :attr:`n_conditions_per_iter > 1`.
        """
        if df_results.empty or len(df_results) < 4:
            initial, _ = self._select_initial_samples(k=n_conditions)
            if isinstance(initial, dict):
                initial = [initial]
            return list(initial)[:n_conditions]

        # Snapshot already-performed experiments BEFORE the batch loop so
        # the final verbose plot can show them in gray while the new picks
        # are highlighted in cyan.
        prev_performed = (
            self.x_performed_experiments.copy()
            if self.x_performed_experiments is not None
            else None
        )
        self._last_plot_context = None

        # Fit the GP ONCE for the whole batch.  ``_batch_fit_reuse`` tells
        # ``_determine_next_parameters`` to stash the first call's fit in
        # ``_cached_gp_fit``; subsequent calls in this loop will reuse it,
        # saving 2x MCMC runs per phase for a 3-condition batch.  Only
        # the penalty (inverse-distance to previously-picked points)
        # changes between acquisitions, so reusing the fit is exact.
        self._cached_gp_fit = None
        self._batch_fit_reuse = True

        selected: list[dict] = []
        orig_penalty = self.penalty
        orig_penalty_factor = self.penalty_factor
        try:
            for i in range(n_conditions):
                if i > 0:
                    self.penalty = "inverse_distance"
                    self.penalty_factor = self.batch_penalty_factor
                try:
                    # verbose=False throughout — we want a SINGLE plot at
                    # the end of the batch showing all n_conditions picks,
                    # not one-per-call showing only the first.
                    params = self._determine_next_parameters(df_results, verbose=False)
                    selected.append(params)
                except Exception as e:
                    print(f"  Warning: batch selection {i} failed: {e}")
                    if len(self.x_unmeasured) > 0:
                        idx = self._rng.integers(len(self.x_unmeasured))
                        params = {
                            p.name: float(self.x_unmeasured[idx, j])
                            for j, p in enumerate(self.parameters_to_optimize)
                        }
                        selected.append(params)
        finally:
            self.penalty = orig_penalty
            self.penalty_factor = orig_penalty_factor
            self._batch_fit_reuse = False
            self._cached_gp_fit = None

        # Render the batch diagnostic plot with ALL picks annotated.
        # Uses the plot context stashed by the first successful
        # _determine_next_parameters call (which captured the clean,
        # pre-penalty GP landscape and acquisition).
        if (
            self.verbose
            and getattr(self, "_last_plot_context", None) is not None
            and len(selected) > 0
        ):
            try:
                picks_array = np.array(
                    [
                        [p[param.name] for param in self.parameters_to_optimize]
                        for p in selected
                    ],
                    dtype=float,
                )
                self._plot_bo_landscape(
                    **self._last_plot_context,
                    next_point=None,
                    next_points=picks_array,
                    prev_experiments=prev_performed,
                )
            except Exception as exc:  # pragma: no cover - best-effort viz
                print(f"  Warning: batch BO plot failed: {exc}")

        return selected

    def run(self) -> None:
        """Execute the full BO experiment loop on a fixed set of FOVs.

        Convenience wrapper that calls :meth:`run_one_phase` for the
        configured number of iterations and then finalises the
        controller.  Equivalent to:

        .. code-block:: python

            for k in range(self.n_iterations + 1):
                agent.run_one_phase(k)
            agent.controller.finish_experiment()

        Dispatches to :meth:`_run_single_loop` or :meth:`_run_batch_loop`
        based on :attr:`n_conditions_per_iter`.

        Raises:
            ValueError: If no FOVs have been configured via :meth:`add_fovs`.
        """
        if not self.fovs:
            raise ValueError("No FOVs configured. Call add_fovs() before run().")

        if self.n_conditions_per_iter > 1:
            self._run_batch_loop()
        else:
            self._run_single_loop()

        self.controller.finish_experiment()

    def _run_single_loop(self) -> None:
        """Legacy single-condition ``run()`` body."""
        # Initial samples spread across the parameter space.
        initial_parameters, _ = self._select_initial_samples(k=4)
        if isinstance(initial_parameters, dict):
            initial_parameters = [initial_parameters]

        df_results = pd.DataFrame()
        self.df_results = df_results

        for i, params in enumerate(initial_parameters):
            print(
                f"=== Initial sample {i + 1}/{len(initial_parameters)}: "
                f"{params} ==="
            )
            events = self._create_events_for_cycle(params)
            if i == 0:
                self.controller.run_experiment(events, validate=False)
            else:
                self.controller.continue_experiment(events, validate=False)

            self._wait_for_pipeline(timeout=self._phase_pipeline_timeout())
            tracks = {fov: self.read_tracks(fov) for fov in self.fovs}
            df_new_results = self._preprocess_results(tracks)
            if not df_new_results.empty:
                df_results = pd.concat([df_results, df_new_results], ignore_index=True)
                self._iteration_means.append(
                    float(df_new_results[self.objective_metric.name].mean())
                )
            self.df_results = df_results

        for e in range(self.n_iterations):
            print(f"\n=== BO Iteration {e + 1}/{self.n_iterations} ===")
            next_params = self._determine_next_parameters(
                df_results, verbose=self.verbose
            )
            print(f"Selected parameters for next experiment: {next_params}")
            events = self._create_events_for_cycle(next_params)
            self.controller.continue_experiment(events, validate=False)

            self._wait_for_pipeline(timeout=self._phase_pipeline_timeout())
            tracks = {fov: self.read_tracks(fov) for fov in self.fovs}
            df_new_results = self._preprocess_results(tracks)
            if not df_new_results.empty:
                df_results = pd.concat([df_results, df_new_results], ignore_index=True)
                self._iteration_means.append(
                    float(df_new_results[self.objective_metric.name].mean())
                )
            self.iteration += 1
            self.df_results = df_results

        # Store final results for external access
        if not df_results.empty:
            self.x = self._extract_x_from_df(df_results)
            self.y = self._extract_y_from_df(df_results)
        self.df_results = df_results

    def _run_batch_loop(self) -> None:
        """Batch BO ``run()`` body (re-uses the configured FOVs each iteration).

        Unlike :meth:`_run_one_phase_batch`, this loop does *not* shift
        FOV indices between iterations -- the same physical FOVs are
        reimaged every batch.  Use this for in-place batch BO; for
        per-phase fresh-FOV batch BO drive the agent via
        :class:`ComposedAgent` (which calls :meth:`run_one_phase`).
        """
        df_results = pd.DataFrame()
        self.df_results = df_results

        n_per_iter = len(self.fovs)
        if n_per_iter % self.n_conditions_per_iter != 0:
            raise ValueError(
                f"Number of FOVs ({n_per_iter}) must be divisible by "
                f"n_conditions_per_iter ({self.n_conditions_per_iter})."
            )

        # --- Initial batch (spread points, no GP yet) ----------------
        initial_params, _ = self._select_initial_samples(k=self.n_conditions_per_iter)
        if isinstance(initial_params, dict):
            initial_params = [initial_params]
        initial_params = list(initial_params)[: self.n_conditions_per_iter]

        print(f"=== Initial batch: {self.n_conditions_per_iter} conditions ===")
        for i, p in enumerate(initial_params):
            print(f"  Condition {i}: {p}")

        self._fov_index_offset = 0
        self._current_phase_id = self._phase_counter
        events = self._create_events_for_batch(initial_params)
        self.controller.run_experiment(events, validate=False)

        self._wait_for_pipeline(timeout=self._phase_pipeline_timeout())
        tracks = {
            fov: self.read_tracks(fov, phase_id=self._current_phase_id)
            for fov in self.fovs
        }
        df_new = self._preprocess_results(tracks)
        if not df_new.empty:
            df_results = pd.concat([df_results, df_new], ignore_index=True)
            self._iteration_means.append(
                float(df_new[self.objective_metric.name].mean())
            )
            self.df_results = df_results
            self._on_phase_complete(df_new, self._phase_counter)
        self._phase_counter += 1
        self.df_results = df_results

        # --- BO iterations -------------------------------------------
        for e in range(self.n_iterations):
            print(f"\n{'='*60}")
            print(f"=== BO Iteration {e + 1}/{self.n_iterations} ===")
            print(f"  Data so far: {len(df_results)} observations")
            print(f"{'='*60}")

            batch_params = self._select_batch_parameters(
                df_results, n_conditions=self.n_conditions_per_iter
            )
            for i, p in enumerate(batch_params):
                print(f"  Selected condition {i}: {p}")

            self._current_phase_id = self._phase_counter
            events = self._create_events_for_batch(batch_params)
            self.controller.continue_experiment(events, validate=False)

            self._wait_for_pipeline(timeout=self._phase_pipeline_timeout())
            tracks = {
                fov: self.read_tracks(fov, phase_id=self._current_phase_id)
                for fov in self.fovs
            }
            df_new = self._preprocess_results(tracks)
            if not df_new.empty:
                df_results = pd.concat([df_results, df_new], ignore_index=True)
                self._iteration_means.append(
                    float(df_new[self.objective_metric.name].mean())
                )
                self.df_results = df_results
                self._on_phase_complete(df_new, self._phase_counter)
            self._phase_counter += 1
            self.iteration += 1
            self.df_results = df_results

        if not df_results.empty:
            self.x = self._extract_x_from_df(df_results)
            self.y = self._extract_y_from_df(df_results)
        self.df_results = df_results

    # _wait_for_pipeline is inherited from InterPhaseAgent

    def _phase_pipeline_timeout(self) -> float:
        """Estimated upper bound on how long a phase needs to drain.

        Derived from the configured frame layout so the timeout
        auto-scales when ``n_frames`` / ``time_between_timesteps`` change
        in subclasses.  Formula:

            expected = n_frames * time_between_timesteps + 1800   # +30 min slack
            timeout  = max(3600, 2 * expected)

        i.e. at least 1 hour, but otherwise 2x the (acquisition + 30 min
        processing slack) estimate.  The 2x safety factor covers the
        gap between the per-frame loop time the acquisition was queued
        with and the actual pipeline drain time, which depends on
        Cellpose throughput, FOV-batching factor, and disk speed.

        Subclasses without ``n_frames`` / ``time_between_timesteps``
        attributes (i.e. the abstract base) fall back to 1 hour.
        """
        n_frames = getattr(self, "n_frames", None)
        dt = getattr(self, "time_between_timesteps", None)
        if n_frames is None or dt is None:
            return 3600.0
        expected = float(n_frames) * float(dt) + 1800.0
        return max(3600.0, 2.0 * expected)

    # ------------------------------------------------------------------
    # EI exploration decay
    # ------------------------------------------------------------------

    def _current_ei_xi(self) -> float:
        """Iteration-dependent EI exploration offset.

        Decays linearly from ``ei_xi`` to ``ei_xi_final`` over the first
        ``ei_xi_decay_fraction * n_iterations`` BO iterations.
        """
        xi_start = float(self.ei_xi)
        xi_end = float(self.ei_xi_final)
        if self.n_iterations <= 0:
            return xi_end

        frac = max(0.0, min(1.0, float(self.ei_xi_decay_fraction)))
        decay_steps = max(1, int(np.ceil(frac * self.n_iterations)))
        t = min(max(self.iteration, 0), decay_steps)
        alpha = t / decay_steps
        return (1.0 - alpha) * xi_start + alpha * xi_end

    # ------------------------------------------------------------------
    # Penalty
    # ------------------------------------------------------------------

    def _apply_penalty(self, acq, x_scaled, recent_points, penalty, penalty_factor):
        """Apply penalty to acquisition function based on recent points."""
        import jax.numpy as jnp

        recent_scaled = jnp.asarray(recent_points)

        if penalty == "delta":
            for rp in recent_scaled:
                distances = jnp.linalg.norm(x_scaled - rp, axis=1)
                mask = distances < 1e-8
                acq = jnp.where(mask, -jnp.inf, acq)

        elif penalty == "inverse_distance":
            min_distances = jnp.full(x_scaled.shape[0], jnp.inf)
            for rp in recent_scaled:
                distances = jnp.linalg.norm(x_scaled - rp, axis=1)
                min_distances = jnp.minimum(min_distances, distances)
            epsilon = 1e-6
            penalty_term = 1.0 / (min_distances + epsilon)
            acq = acq / (1.0 + penalty_factor * penalty_term)

        return acq

    def _is_flat_acquisition(self, acq_values, range_tol=1e-6, relative_tol=0.01):
        """Check whether acquisition values are effectively flat."""
        import jax.numpy as jnp

        acq = jnp.asarray(acq_values)
        finite = jnp.isfinite(acq)
        if not bool(jnp.any(finite)):
            return True
        acq_f = acq[finite]
        acq_range = float(jnp.max(acq_f) - jnp.min(acq_f))
        if acq_range < range_tol:
            return True
        acq_scale = float(jnp.max(jnp.abs(acq_f)))
        if acq_scale > 0 and acq_range / acq_scale < relative_tol:
            return True
        return False

    # ------------------------------------------------------------------
    # Parameter space generation
    # ------------------------------------------------------------------

    def _generate_total_parameter_space(self):
        param_grids = []
        for param in self.parameters_to_optimize:
            spacing = param.spacing if param.spacing is not None else 1.0
            grid = np.arange(param.bounds[0], param.bounds[1] + spacing, spacing)
            param_grids.append(grid)
        mesh = np.meshgrid(*param_grids, indexing="ij")
        flat_mesh = [m.flatten() for m in mesh]
        return np.column_stack(flat_mesh)

    # ------------------------------------------------------------------
    # Data extraction helpers
    # ------------------------------------------------------------------

    def _extract_x_from_df(self, df: pd.DataFrame) -> np.ndarray:
        X = []
        for param in self.parameters_to_optimize:
            X.append(df[param.name].values)
        for covariate in self.bo_covariates:
            X.append(df[covariate.name].values)
        return np.column_stack(X)

    def _extract_y_from_df(self, df: pd.DataFrame) -> np.ndarray:
        y = df[self.objective_metric.name].values
        return y.reshape(-1, 1)

    def _get_bounds_and_log_scale(self):
        bounds = []
        log_scale = []
        for param in self.parameters_to_optimize:
            bounds.append(param.bounds)
            log_scale.append(param.log_scale)
        for covariate in self.bo_covariates:
            bounds.append(covariate.bounds)
            log_scale.append(covariate.log_scale)
        return bounds, log_scale

    def _sample_covariate_grid(self, df_results: pd.DataFrame) -> np.ndarray | None:
        """Return the covariate-sample grid to marginalise over.

        Shape is ``(n_samples, n_covariates)``.  In ``"joint"`` mode
        (default) ``n_samples == self.n_cov_samples`` regardless of the
        number of covariates — each row is a real observation from
        ``df_results``, preserving cross-covariate correlations.  In
        ``"mesh"`` mode the returned grid has
        ``n_samples == self.n_cov_samples ** n_covariates`` rows (outer
        product of independently resampled covariates), which matches the
        legacy behaviour but scales poorly.

        Returns ``None`` when there are no covariates.
        """
        if len(self.bo_covariates) == 0:
            return None

        n_samples = int(self.n_cov_samples)
        n_rows = len(df_results)

        if self.cov_marginalization_mode == "joint":
            if n_rows == 0:
                return np.zeros((n_samples, len(self.bo_covariates)), dtype=float)
            row_idx = self._rng.integers(0, n_rows, size=n_samples)
            return np.column_stack(
                [
                    np.asarray(df_results[c.name].values, dtype=float)[row_idx]
                    for c in self.bo_covariates
                ]
            )

        # --- mesh mode (legacy) ---
        cov_samples_list = []
        for covariate in self.bo_covariates:
            cov_vals = np.asarray(df_results[covariate.name].values, dtype=float)
            if cov_vals.size == 0:
                cov_samples = np.array([0.0])
            else:
                cov_samples = self._rng.choice(cov_vals, size=n_samples, replace=True)
            cov_samples_list.append(cov_samples)

        if len(cov_samples_list) == 1:
            return cov_samples_list[0].reshape(-1, 1)
        cov_mesh = np.meshgrid(*cov_samples_list, indexing="ij")
        return np.stack([m.ravel() for m in cov_mesh], axis=1)

    # ------------------------------------------------------------------
    # Initial sample selection (spread across parameter space)
    # ------------------------------------------------------------------

    def _extract_spread_points(self, X, k=3):
        rng = np.random.default_rng(self._seed)
        n = X.shape[0]
        idx = [rng.integers(n)]
        d = np.linalg.norm(X - X[idx[0]], axis=1)
        for _ in range(1, k):
            next_idx = np.argmax(d)
            idx.append(next_idx)
            d = np.minimum(d, np.linalg.norm(X - X[next_idx], axis=1))
        idx_array = np.array(idx)
        return np.array(X[idx_array]), idx

    def _select_initial_samples(self, k=4):
        x_scaler = StandardScalerBounds()
        bounds, log_scale = self._get_bounds_and_log_scale()

        # Optionally restrict the farthest-point pool to a lower fraction of
        # each parameter's range.  Keeps initial exploration inside a feasible
        # subspace (e.g. to respect a cycle-time budget on worst-case stim
        # exposure) while leaving BO-guided phases free to pick the full grid.
        if self.initial_exploration_cap is not None:
            cap = self.initial_exploration_cap
            mask = np.ones(len(self.x_unmeasured), dtype=bool)
            for i, param in enumerate(self.parameters_to_optimize):
                lo, hi = param.bounds
                thr = lo + cap * (hi - lo)
                mask &= self.x_unmeasured[:, i] <= thr
            if not mask.any():
                raise ValueError(
                    f"initial_exploration_cap={cap} produced an empty pool. "
                    f"Check parameter bounds or lower the cap."
                )
            pool_global_indices = np.flatnonzero(mask)
            pool = self.x_unmeasured[mask]
        else:
            pool_global_indices = np.arange(len(self.x_unmeasured))
            pool = self.x_unmeasured

        X_scaled = x_scaler.fit_transform(pool, bounds=bounds, log_scale=log_scale)
        X_selected_scaled, selected_local_indices = self._extract_spread_points(
            X_scaled, k=k
        )
        X_selected = x_scaler.inverse_transform(X_selected_scaled)
        # Map pool-local indices back to global x_unmeasured indices
        selected_indices = pool_global_indices[np.asarray(selected_local_indices)]
        self.x_unmeasured = np.delete(self.x_unmeasured, selected_indices, axis=0)
        self.x_performed_experiments = (
            np.concatenate([self.x_performed_experiments, X_selected], axis=0)
            if self.x_performed_experiments is not None
            else X_selected
        )

        selected_parameter_dicts = []
        for selected_row in np.asarray(X_selected):
            selected_parameter_dicts.append(
                {
                    param.name: selected_row[i]
                    for i, param in enumerate(self.parameters_to_optimize)
                }
            )
        return selected_parameter_dicts, selected_indices

    # ------------------------------------------------------------------
    # Acquisition functions (EI, UCB)
    # ------------------------------------------------------------------

    def _compute_mc_ei(
        self,
        rng_key,
        gp_model,
        x_scaled,
        best_f_scaled,
        maximize: bool,
        batch_size=1000,
        xi: float = 0.0,
        penalty: Optional[str] = None,
        recent_points: Optional[np.ndarray] = None,
        penalty_factor: float = 1.0,
    ):
        """Expected Improvement using GPax posterior samples."""
        import jax.numpy as jnp
        from jax.scipy.stats import norm as jnorm

        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_scaled,
            batch_size=batch_size,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )

        mean_broadcast = mean_pred[None, None, :]
        y_sampled = jnp.where(jnp.isfinite(y_sampled), y_sampled, mean_broadcast)

        mean = jnp.mean(y_sampled, axis=(0, 1))
        var = jnp.var(y_sampled, axis=(0, 1))
        sigma = jnp.sqrt(jnp.maximum(var, 1e-12))

        u = (mean - (best_f_scaled + xi)) / sigma
        if not maximize:
            u = -u

        ei = sigma * (jnorm.pdf(u) + u * jnorm.cdf(u))
        ei = jnp.where(jnp.isfinite(ei), ei, 0.0)

        if penalty is not None and recent_points is not None and len(recent_points) > 0:
            ei = self._apply_penalty(
                ei, x_scaled, recent_points, penalty, penalty_factor
            )

        return ei

    def _compute_mc_ucb(
        self,
        rng_key,
        gp_model,
        x_scaled,
        maximize: bool,
        beta: float = 4.0,
        batch_size=1000,
        penalty: Optional[str] = None,
        recent_points: Optional[np.ndarray] = None,
        penalty_factor: float = 1.0,
    ):
        """Monte-Carlo UCB using GPax posterior samples."""
        import jax.numpy as jnp

        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_scaled,
            batch_size=batch_size,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )

        mean_broadcast = mean_pred[None, None, :]
        y_sampled = jnp.where(jnp.isfinite(y_sampled), y_sampled, mean_broadcast)

        mean = jnp.mean(y_sampled, axis=(0, 1))
        var = jnp.var(y_sampled, axis=(0, 1))
        delta = jnp.sqrt(beta * var)
        ucb = mean + delta if maximize else -(mean - delta)
        ucb = jnp.where(jnp.isfinite(ucb), ucb, 0.0)

        if penalty is not None and recent_points is not None and len(recent_points) > 0:
            ucb = self._apply_penalty(
                ucb, x_scaled, recent_points, penalty, penalty_factor
            )

        return ucb

    # ------------------------------------------------------------------
    # Robust acquisition (marginalised over covariates)
    # ------------------------------------------------------------------

    def _compute_robust_acq(
        self,
        rng_key,
        gp_model,
        x_grid,
        c_samples,
        x_scaler,
        y_scaled,
        batch_size=1000,
        xi: Optional[float] = None,
    ):
        """Compute robust acquisition marginalised over covariate samples."""
        import jax.numpy as jnp
        from jax.scipy.stats import norm as jnorm

        n_grid = x_grid.shape[0]
        n_mc = c_samples.shape[0] if c_samples.ndim > 1 else len(c_samples)
        if c_samples.ndim == 1:
            c_samples = c_samples.reshape(-1, 1)

        x_repeated = jnp.repeat(x_grid, n_mc, axis=0)
        c_tiled = jnp.tile(c_samples, (n_grid, 1))
        x_full_batch = jnp.hstack([x_repeated, c_tiled])
        x_full_batch_scaled = x_scaler.transform(x_full_batch)

        n_total = x_full_batch_scaled.shape[0]
        batch_size = min(batch_size, n_total)
        print(
            f"Computing robust acquisition over {n_total} scenarios "
            f"({n_grid} grid points x {n_mc} covariate samples)..."
        )

        maximize = self.objective_metric.goal == "maximize"

        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_full_batch_scaled,
            batch_size=batch_size,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )

        mean_broadcast = mean_pred[None, None, :]
        y_sampled = jnp.where(jnp.isfinite(y_sampled), y_sampled, mean_broadcast)

        mean_flat = jnp.mean(y_sampled, axis=(0, 1))
        var_flat = jnp.var(y_sampled, axis=(0, 1))

        mean_matrix = mean_flat.reshape(n_grid, n_mc)
        var_matrix = var_flat.reshape(n_grid, n_mc)

        if self.acquisition_function == "ei":
            xi_value = self.ei_xi if xi is None else float(xi)
            # MC-integrated EI: compare every (x, c_i) prediction against a
            # SINGLE scalar reference, then average EI across covariate
            # samples.  We use the **GP-predicted best marginal mean**
            # over the control grid as the reference (BoTorch "noisy EI"
            # / "qLogNoisyEI" style), NOT the raw observed max.
            #
            # Rationale: with a noisy, sparse objective like
            # ``frac_oscillating`` (mostly 0-0.05, occasional FOV
            # outliers at 0.15), the raw observed max sits several σ
            # above the mean after z-scoring.  EI then computes
            # ``u = (mu - best_f) / sigma << 0`` everywhere, the
            # acquisition collapses to zero, and we silently fall back
            # to UCB every phase (pure variance-seeking → jumps to
            # grid corners instead of exploiting the top region).
            # Using the GP's best predicted mean smooths over outlier
            # FOVs and keeps EI numerically non-flat so it can actually
            # drive exploitation near the observed optimum.
            mean_marg_ref = jnp.mean(mean_matrix, axis=1)
            if maximize:
                best_f_scalar = jnp.max(mean_marg_ref)
            else:
                best_f_scalar = jnp.min(mean_marg_ref)

            print(
                f"  best_f (scaled, from GP predicted mean over grid): "
                f"{float(best_f_scalar):.6f}"
            )

            sigma_matrix = jnp.sqrt(jnp.maximum(var_matrix, 1e-12))
            u = (mean_matrix - (best_f_scalar + xi_value)) / sigma_matrix
            if not maximize:
                u = -u
            acq_matrix = sigma_matrix * (jnorm.pdf(u) + u * jnorm.cdf(u))
            acq_matrix = jnp.where(jnp.isfinite(acq_matrix), acq_matrix, 0.0)
        else:
            delta = jnp.sqrt(self.ucb_beta * var_matrix)
            if maximize:
                acq_matrix = mean_matrix + delta
            else:
                acq_matrix = -(mean_matrix - delta)
            acq_matrix = jnp.where(jnp.isfinite(acq_matrix), acq_matrix, 0.0)

        robust_acq = jnp.mean(acq_matrix, axis=1)

        # Penalty (controllable params only)
        if self.penalty is not None and self.x_performed_experiments is not None:
            n_ctrl = len(self.parameters_to_optimize)
            recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl]

            ctrl_log = (
                x_scaler.log_scale[:n_ctrl] if x_scaler.log_scale is not None else None
            )
            ctrl_mean = x_scaler.mean_[:n_ctrl]
            ctrl_std = x_scaler.std_[:n_ctrl]

            def _scale_ctrl(X):
                X = jnp.asarray(X)
                if ctrl_log is not None:
                    for i, use_log in enumerate(ctrl_log):
                        if use_log:
                            X = X.at[:, i].set(jnp.log(X[:, i]))
                return (X - ctrl_mean) / ctrl_std

            x_grid_ctrl_scaled = _scale_ctrl(x_grid)
            recent_ctrl_scaled = _scale_ctrl(jnp.asarray(recent_ctrl_raw))

            robust_acq = self._apply_penalty(
                robust_acq,
                x_grid_ctrl_scaled,
                np.asarray(recent_ctrl_scaled),
                self.penalty,
                self.penalty_factor,
            )

        print(
            f"  Robust acq stats: min={float(jnp.min(robust_acq)):.6f}, "
            f"max={float(jnp.max(robust_acq)):.6f}"
        )
        return robust_acq

    # ------------------------------------------------------------------
    # Main BO decision logic
    # ------------------------------------------------------------------

    def _determine_next_parameters(
        self, df_results: pd.DataFrame, verbose=False
    ) -> dict:
        """Fit GP, compute acquisition, and return the next parameter dict.

        In batch BO (``_select_batch_parameters``) this method may be
        called N times per phase with the **same** ``df_results`` — only
        the penalty differs between calls.  Re-fitting the GP each time
        wastes compute (MCMC dominates runtime).  If ``_cached_gp_fit``
        has been populated by an earlier call in the same batch loop,
        we reuse the cached fit and skip straight to acquisition.  The
        batch driver is responsible for clearing the cache before the
        next phase; non-batch callers leave ``_cached_gp_fit`` at ``None``
        so behaviour there is unchanged.
        """
        import gpax
        import gpax.utils
        import jax.numpy as jnp

        cache = getattr(self, "_cached_gp_fit", None)
        # Capture BEFORE the fit step runs — the fit step populates the
        # cache when _batch_fit_reuse is set, so checking after would
        # always read True on call 1 too.  We need the pre-fit state to
        # know whether this is the first call in a batch (or a non-batch
        # call), which controls whether we stash the plot context below.
        is_first_call_in_batch = cache is None
        if cache is not None:
            gp_model = cache["gp_model"]
            x_scaler = cache["x_scaler"]
            y_scaler = cache["y_scaler"]
            rng_key_predict = cache["rng_key_predict"]
            x = cache["x"]
            y = cache["y"]
            print("  [reusing cached GP fit from earlier in this batch]")
        else:
            x_scaler = StandardScalerBounds()
            bounds, log_scale = self._get_bounds_and_log_scale()

            y_scaler = StandardScalerBounds()
            y_log_scale = [self.objective_metric.log_scale]

            x_raw = self._extract_x_from_df(df_results)
            y_raw = self._extract_y_from_df(df_results)

            x = x_scaler.fit_transform(x_raw, bounds=bounds, log_scale=log_scale)
            y = y_scaler.fit_transform(y_raw, bounds=None, log_scale=y_log_scale)

            rng_key, rng_key_predict = gpax.utils.get_keys()
            gp_model = gpax.ExactGP(kernel="Matern", input_dim=x.shape[1])
            gp_model.fit(
                rng_key,
                X=x,
                y=y,
                progress_bar=True,
                num_warmup=400,
                num_samples=800,
            )

            # Store for post-hoc analysis
            self.model = gp_model
            self._x_scaler = x_scaler
            self._y_scaler = y_scaler
            self._rng_key_predict = rng_key_predict

            # If we're inside a batch selection loop, cache the fit so
            # subsequent picks in the same batch skip MCMC.
            if getattr(self, "_batch_fit_reuse", False):
                self._cached_gp_fit = dict(
                    gp_model=gp_model,
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                    rng_key_predict=rng_key_predict,
                    x=x,
                    y=y,
                )

        x_grid_ctrl = self.x_unmeasured.copy()
        acquisition_used = self.acquisition_function
        current_xi = self._current_ei_xi()

        if len(self.bo_covariates) > 0:
            # Robust acquisition: marginalise over observed covariate distribution
            cov_grid = self._sample_covariate_grid(df_results)

            acq_values_total = self._compute_robust_acq(
                rng_key_predict,
                gp_model,
                x_grid_ctrl,
                cov_grid,
                x_scaler,
                y,
                xi=current_xi,
            )

            # Fallback: EI can become flat if model is overconfident
            if acquisition_used == "ei" and self._is_flat_acquisition(
                acq_values_total, range_tol=1e-6
            ):
                print(
                    "  EI is flat (max-min < 1e-6). "
                    "Falling back to UCB for this round."
                )
                acquisition_used = "ucb"
                old_acq = self.acquisition_function
                try:
                    self.acquisition_function = "ucb"
                    acq_values_total = self._compute_robust_acq(
                        rng_key_predict,
                        gp_model,
                        x_grid_ctrl,
                        cov_grid,
                        x_scaler,
                        y,
                        xi=current_xi,
                    )
                finally:
                    self.acquisition_function = old_acq
            self._acquisition_used_this_round = acquisition_used
        else:
            # No covariates: acquisition on controllable grid directly
            x_total_scaled = x_scaler.transform(x_grid_ctrl)
            maximize = self.objective_metric.goal == "maximize"

            recent_ctrl = None
            if self.penalty is not None and self.x_performed_experiments is not None:
                n_ctrl_params = len(self.parameters_to_optimize)
                recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl_params]
                recent_ctrl = np.asarray(
                    x_scaler.transform(jnp.asarray(recent_ctrl_raw))
                )

            if self.acquisition_function == "ei":
                # best_f from GP-predicted mean over grid (not observed
                # max) — robust to single-point outliers.  See the
                # comment in _compute_robust_acq for full rationale.
                _mean_pred, _y_samp = gp_model.predict_in_batches(
                    rng_key_predict,
                    x_total_scaled,
                    batch_size=1000,
                    n=self.ei_num_samples,
                    noiseless=self.ei_noiseless,
                )
                _grid_mean = jnp.mean(_y_samp, axis=(0, 1))
                best_f_scaled = jnp.max(_grid_mean) if maximize else jnp.min(_grid_mean)
                acq_values_total = self._compute_mc_ei(
                    rng_key_predict,
                    gp_model,
                    x_total_scaled,
                    best_f_scaled=best_f_scaled,
                    maximize=maximize,
                    batch_size=1000,
                    xi=current_xi,
                    penalty=self.penalty,
                    recent_points=recent_ctrl,
                    penalty_factor=self.penalty_factor,
                )
                if self._is_flat_acquisition(acq_values_total, range_tol=1e-6):
                    print(
                        "  EI is flat (max-min < 1e-6). "
                        "Falling back to UCB for this round."
                    )
                    acquisition_used = "ucb"
                    acq_values_total = self._compute_mc_ucb(
                        rng_key_predict,
                        gp_model,
                        x_total_scaled,
                        maximize=maximize,
                        beta=self.ucb_beta,
                        batch_size=1000,
                        penalty=self.penalty,
                        recent_points=recent_ctrl,
                        penalty_factor=self.penalty_factor,
                    )
            else:
                acq_values_total = self._compute_mc_ucb(
                    rng_key_predict,
                    gp_model,
                    x_total_scaled,
                    maximize=maximize,
                    beta=self.ucb_beta,
                    batch_size=1000,
                    penalty=self.penalty,
                    recent_points=recent_ctrl,
                    penalty_factor=self.penalty_factor,
                )
            self._acquisition_used_this_round = acquisition_used

        next_measurement_idx = jnp.argmax(acq_values_total)
        next_parameters = np.asarray(x_grid_ctrl[int(next_measurement_idx)])

        # Remove chosen point from unmeasured set
        self.x_unmeasured = np.delete(
            self.x_unmeasured, int(next_measurement_idx), axis=0
        )
        self.x_performed_experiments = (
            np.concatenate(
                [self.x_performed_experiments, next_parameters.reshape(1, -1)],
                axis=0,
            )
            if self.x_performed_experiments is not None
            else next_parameters.reshape(1, -1)
        )

        next_parameters_dict = {
            param.name: next_parameters[i]
            for i, param in enumerate(self.parameters_to_optimize)
        }

        # Stash the plot context so the batch caller can render a single
        # diagnostic plot with all picks overlaid.  IMPORTANT: only stash
        # on the FIRST call of a batch — i.e. when the GP fit cache is
        # still empty.  Subsequent calls in the same batch have:
        #   (a) inverse-distance penalty applied to the acquisition, and
        #   (b) earlier picks removed from x_unmeasured,
        # so their acquisition surfaces are warped and have holes where
        # picks 1..i-1 used to be.  Stashing those would make the final
        # overlay misleading: picks 1 and 2 would sit on zero-padded
        # holes (rendered blue) instead of on the clean, unpenalised EI
        # landscape that produced the batch in the first place.  Using
        # the FIRST call's stash gives the intuitive view: clean EI
        # landscape + 3 crosses showing where the greedy + diversity
        # penalty ended up placing each batch pick.
        if is_first_call_in_batch:
            self._last_plot_context = dict(
                df_results=df_results,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                gp_model=gp_model,
                rng_key_predict=rng_key_predict,
                x_unmeasured_at_computation=x_grid_ctrl.copy(),
                acq_values_total=np.asarray(acq_values_total),
                acquisition_used=acquisition_used,
                current_xi=current_xi,
                cov_grid=cov_grid if len(self.bo_covariates) > 0 else None,
                y=np.asarray(y),
            )

        if verbose:
            self._plot_bo_landscape(
                df_results,
                x_scaler,
                y_scaler,
                gp_model,
                rng_key_predict,
                x_grid_ctrl,
                acq_values_total,
                next_parameters,
                acquisition_used,
                current_xi,
                cov_grid if len(self.bo_covariates) > 0 else None,
                y,
            )

        return next_parameters_dict

    # ------------------------------------------------------------------
    # Visualization (optional, for verbose mode)
    # ------------------------------------------------------------------

    def _get_ground_truth_grid(self, unique_x1, unique_x2, cov_mean):
        """Compute ground truth on a (x1, x2) grid.

        Returns an array of shape ``(len(unique_x1), len(unique_x2))``
        or ``None`` if no analytical ground truth is available.

        Override in subclasses that have a known ground truth.
        """
        return None

    def _plot_bo_landscape(
        self,
        df_results,
        x_scaler,
        y_scaler,
        gp_model,
        rng_key_predict,
        x_unmeasured_at_computation,
        acq_values_total,
        next_point,
        acquisition_used,
        current_xi,
        cov_grid,
        y,
        next_points=None,
        prev_experiments=None,
    ):
        """Plot measured data, model predictions, and acquisition function.

        Creates 4 subplots:
        1. Measured points (x1, x2) with y as colour
        2. Ground truth (if available) or GP model uncertainty
        3. Acquisition function (with all batch picks annotated)
        4. GP predicted landscape

        The figure is also saved to ``{storage_path}/figures/bo_iter_*.png``
        so every phase is archived alongside the experiment data.

        Args:
            next_point: legacy single-pick annotation. Kept for backwards
                compatibility with single-condition callers.
            next_points: array of shape ``(n_picks, n_params)`` with every
                pick made in the current phase. Takes priority over
                ``next_point`` — used by batch BO to annotate all picks
                (e.g. 3 cyan crosses when ``n_conditions_per_iter=3``).
            prev_experiments: snapshot of ``x_performed_experiments`` from
                BEFORE the current batch was picked. When provided, used
                for the "Already measured" overlay so the current-phase
                picks don't double up on both gray and cyan markers. Falls
                back to ``self.x_performed_experiments`` if ``None``.
        """
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        x_total_ctrl = self.x_total_linespace.copy()
        unique_x1 = np.unique(x_total_ctrl[:, 0])
        unique_x2 = np.unique(x_total_ctrl[:, 1])

        # Covariate samples for marginalisation.  Match what the BO
        # acquisition does in _compute_robust_acq with cov_marginalization_mode
        # 'joint': draw whole rows from df_results so cross-covariate
        # correlations are preserved.  The previous implementation varied
        # only the first covariate on a linspace and held the rest at their
        # mean — inconsistent with the BO target and biased when covariates
        # are correlated.
        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_cols = [c.name for c in self.bo_covariates]
            cov_vals_full = np.asarray(df_results[cov_cols].to_numpy(), dtype=float)
            cov_mean = float(np.nanmean(cov_vals_full[:, 0]))
            n_cov_samples = 50
            _plot_rng = np.random.default_rng(0)
            row_idx = _plot_rng.integers(0, cov_vals_full.shape[0], size=n_cov_samples)
            cov_samples_joint = cov_vals_full[row_idx]  # (n_cov_samples, n_cov)
        else:
            cov_mean = 0.0
            n_cov_samples = 1
            cov_samples_joint = None

        # Build (x_ctrl, c) grid: repeat each (x1, x2) across every joint
        # covariate sample.  Layout mirrors _compute_robust_acq so the
        # reshape to (n_ctrl, n_cov) further down is unambiguous.
        if len(self.bo_covariates) > 0 and cov_samples_joint is not None:
            n_ctrl = len(unique_x1) * len(unique_x2)
            ctrl_grid = np.array([[x1, x2] for x1 in unique_x1 for x2 in unique_x2])
            x_grid_full = np.hstack(
                [
                    np.repeat(ctrl_grid, n_cov_samples, axis=0),
                    np.tile(cov_samples_joint, (n_ctrl, 1)),
                ]
            )
        else:
            x_grid_full = np.array([[x1, x2] for x1 in unique_x1 for x2 in unique_x2])

        x_grid_scaled = x_scaler.transform(x_grid_full)
        # Batched prediction — ``gp_model.predict(...)`` would materialise
        # a ``(num_mcmc, n_samples_per_draw, n_test)`` array of posterior
        # samples in one shot; for a 10k-point (x, c) grid this blew up
        # to >170 GB and OOM'd the GPU.  ``predict_in_batches`` streams
        # the test grid in chunks, identical API as used by
        # ``_compute_robust_acq``.
        y_pred_scaled, y_samples_scaled = gp_model.predict_in_batches(
            rng_key_predict,
            x_grid_scaled,
            batch_size=1000,
            n=self.ei_num_samples,
            noiseless=True,
        )
        y_pred = y_scaler.inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).flatten()

        y_std_scaled = jnp.std(y_samples_scaled, axis=(0, 1))
        y_std = np.asarray(y_std_scaled) * float(jnp.abs(y_scaler.std_[0]))

        # Marginalise
        if len(self.bo_covariates) > 0:
            n_ctrl_points = len(unique_x1) * len(unique_x2)
            y_pred_marg = y_pred.reshape(n_ctrl_points, n_cov_samples).mean(axis=1)
            y_std_marg = y_std.reshape(n_ctrl_points, n_cov_samples).mean(axis=1)
            acq_marg = np.asarray(acq_values_total)
        else:
            y_pred_marg = y_pred
            y_std_marg = y_std
            acq_marg = np.asarray(acq_values_total)

        # For full-grid acq, recompute on the complete linespace if needed
        if len(acq_marg) != len(unique_x1) * len(unique_x2):
            # acq was computed on unmeasured subset; pad to full grid
            acq_full = np.zeros(len(unique_x1) * len(unique_x2))
            # Map unmeasured points to their position in full grid
            x_full = self.x_total_linespace
            for j, pt in enumerate(x_unmeasured_at_computation):
                diffs = np.abs(x_full - pt).sum(axis=1)
                idx = np.argmin(diffs)
                if j < len(acq_marg):
                    acq_full[idx] = float(acq_marg[j])
            acq_marg = acq_full

        y_pred_2d = y_pred_marg.reshape(len(unique_x1), len(unique_x2))
        acq_2d = acq_marg.reshape(len(unique_x1), len(unique_x2))
        X_mesh, Y_mesh = np.meshgrid(unique_x1, unique_x2, indexing="ij")

        y_true = self._get_ground_truth_grid(unique_x1, unique_x2, cov_mean)

        if y_true is not None and self.objective_metric.goal == "maximize":
            y_vmin = 0.0
            y_vmax = float(np.nanmax(y_true))
        else:
            all_y = []
            if not df_results.empty:
                all_y.append(df_results[self.objective_metric.name].values)
            all_y.append(y_pred_marg)
            if y_true is not None:
                all_y.append(y_true.flatten())
            all_y = np.concatenate(all_y)
            y_vmin, y_vmax = float(np.nanmin(all_y)), float(np.nanmax(all_y))

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(
            f"BO Iteration {self.iteration + 1}/{self.n_iterations} "
            f"({len(df_results)} total measurements)",
            fontsize=14,
        )

        # Subplot 1: Measured points
        ax1 = axes[0, 0]
        if not df_results.empty:
            scatter = ax1.scatter(
                df_results[self.parameters_to_optimize[0].name],
                df_results[self.parameters_to_optimize[1].name],
                c=df_results[self.objective_metric.name],
                cmap="viridis",
                vmin=y_vmin,
                vmax=y_vmax,
                s=100,
                edgecolor="k",
                linewidth=1.5,
                zorder=3,
            )
            if len(self.bo_covariates) > 0:
                grouped = df_results.groupby(
                    [
                        self.parameters_to_optimize[0].name,
                        self.parameters_to_optimize[1].name,
                    ]
                )
                for (x1, x2), group in grouped:
                    # One line per covariate: shows the mean of the FOV
                    # observations at this (x1, x2) condition so the
                    # viewer can see what covariate context the GP has
                    # for each measured point.  FOVs are fed to the GP
                    # as individual observations (not aggregated), so
                    # these means are display-only — the GP itself sees
                    # the within-condition spread.
                    lines = [
                        f"{cov.name}={group[cov.name].mean():.1f}"
                        for cov in self.bo_covariates
                    ]
                    ax1.annotate(
                        "\n".join(lines),
                        xy=(x1, x2),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        bbox=dict(
                            boxstyle="round,pad=0.2", facecolor="white", alpha=0.7
                        ),
                    )
        ax1.set_xlabel(self.parameters_to_optimize[0].name)
        ax1.set_ylabel(self.parameters_to_optimize[1].name)
        ax1.set_title("Measured Points")
        ax1.grid(True, alpha=0.3)

        cov_label = " (marginalized)" if len(self.bo_covariates) > 0 else ""

        # Subplot 2: Ground Truth or Uncertainty
        ax2 = axes[0, 1]
        if y_true is not None:
            ax2.pcolormesh(
                X_mesh,
                Y_mesh,
                y_true,
                cmap="viridis",
                vmin=y_vmin,
                vmax=y_vmax,
                shading="nearest",
            )
            opt_idx = np.unravel_index(
                (
                    y_true.argmax()
                    if self.objective_metric.goal == "maximize"
                    else y_true.argmin()
                ),
                y_true.shape,
            )
            ax2.scatter(
                unique_x1[opt_idx[0]],
                unique_x2[opt_idx[1]],
                c="red",
                s=300,
                marker="*",
                edgecolors="black",
                linewidths=2,
                label="True optimum",
                zorder=3,
            )
            ax2.legend()
            ax2.set_title(f"Ground Truth (cov={cov_mean:.2f})")
        else:
            y_std_2d = y_std_marg.reshape(len(unique_x1), len(unique_x2))
            im2 = ax2.pcolormesh(
                X_mesh,
                Y_mesh,
                y_std_2d,
                cmap="YlOrRd",
                shading="nearest",
            )
            fig.colorbar(im2, ax=ax2, label=f"std({self.objective_metric.name})")
            ax2.set_title(f"GP Model Uncertainty{cov_label}")
        ax2.set_xlabel(self.parameters_to_optimize[0].name)
        ax2.set_ylabel(self.parameters_to_optimize[1].name)
        ax2.grid(True, alpha=0.3)

        # Subplot 3: Acquisition Function
        ax3 = axes[1, 0]
        ax3.pcolormesh(X_mesh, Y_mesh, acq_2d, cmap="coolwarm", shading="nearest")

        # "Already measured" overlay — prefer the pre-batch snapshot so
        # current-phase picks don't show up on BOTH gray and cyan markers.
        _prev = (
            prev_experiments
            if prev_experiments is not None
            else self.x_performed_experiments
        )
        if _prev is not None and len(_prev) > 0:
            _prev_arr = np.asarray(_prev)
            ax3.scatter(
                _prev_arr[:, 0],
                _prev_arr[:, 1],
                c="gray",
                s=80,
                marker="x",
                linewidths=2,
                label="Already measured",
                alpha=0.6,
                zorder=2,
            )

        # Current-phase picks — `next_points` (plural) takes priority so
        # batch BO can show all n_conditions_per_iter picks as cyan crosses.
        _picks = None
        if next_points is not None and len(next_points) > 0:
            _picks = np.asarray(next_points, dtype=float).reshape(
                -1, len(self.parameters_to_optimize)
            )
        elif next_point is not None:
            _picks = np.asarray(next_point, dtype=float).reshape(
                1, len(self.parameters_to_optimize)
            )
        if _picks is not None and len(_picks) > 0:
            ax3.scatter(
                _picks[:, 0],
                _picks[:, 1],
                c="cyan",
                s=220,
                marker="X",
                edgecolors="black",
                linewidths=1.5,
                label=f"Chosen next ({len(_picks)})",
                zorder=4,
            )
        ax3.legend()
        ax3.set_xlabel(self.parameters_to_optimize[0].name)
        ax3.set_ylabel(self.parameters_to_optimize[1].name)
        acq_label = str(
            getattr(self, "_acquisition_used_this_round", self.acquisition_function)
        ).upper()
        ax3.set_title(f"Acquisition ({acq_label}){cov_label}")
        ax3.grid(True, alpha=0.3)

        # Subplot 4: GP Predicted Landscape
        ax4 = axes[1, 1]
        im4 = ax4.pcolormesh(
            X_mesh,
            Y_mesh,
            y_pred_2d,
            cmap="viridis",
            vmin=y_vmin,
            vmax=y_vmax,
            shading="nearest",
        )
        gp_opt_idx = np.unravel_index(
            (
                y_pred_2d.argmax()
                if self.objective_metric.goal == "maximize"
                else y_pred_2d.argmin()
            ),
            y_pred_2d.shape,
        )
        ax4.scatter(
            unique_x1[gp_opt_idx[0]],
            unique_x2[gp_opt_idx[1]],
            c="red",
            s=300,
            marker="*",
            edgecolors="black",
            linewidths=2,
            label="GP optimum",
            zorder=3,
        )
        if not df_results.empty:
            ax4.scatter(
                df_results[self.parameters_to_optimize[0].name],
                df_results[self.parameters_to_optimize[1].name],
                c="white",
                s=50,
                marker="x",
                linewidths=2,
                label="Measured",
                zorder=3,
                alpha=0.8,
            )
        ax4.legend()
        ax4.set_xlabel(self.parameters_to_optimize[0].name)
        ax4.set_ylabel(self.parameters_to_optimize[1].name)
        ax4.set_title(f"GP Predicted Landscape{cov_label}")
        ax4.grid(True, alpha=0.3)

        plt.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.94, 0.15, 0.02, 0.7])
        fig.colorbar(
            scatter if not df_results.empty else im4,
            cax=cbar_ax,
            label=self.objective_metric.name,
        )

        # Persist the figure to {storage_path}/figures/ so every BO phase
        # is archived alongside the experiment data. Errors here should
        # never interrupt the experiment loop — log and continue.
        try:
            figures_dir = os.path.join(self.storage_path, "figures")
            os.makedirs(figures_dir, exist_ok=True)
            fig_path = os.path.join(
                figures_dir, f"bo_iter_{self.iteration + 1:03d}.png"
            )
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"  Saved BO diagnostic plot to {fig_path}")
        except Exception as exc:  # pragma: no cover - best-effort
            print(f"  Warning: could not save BO plot: {exc}")

        plt.show()


# ---------------------------------------------------------------------------
# Mock utilities for testing (from legacy code)
# ---------------------------------------------------------------------------

_MOCK_RNG = np.random.default_rng(42)


def mock_df_results(
    df_acquire: pd.DataFrame,
    n_samples: int = 10,
    *,
    noise_scale: float = 1.0,
    noise_base: float = 2.0,
    noise_cov_scale: float = 2.0,
) -> pd.DataFrame:
    """Generate mock measurements for testing.

    The objective is maximised around a moving optimum influenced by ``cov1``.
    """
    rng = _MOCK_RNG
    rows = []
    for _, row in df_acquire.iterrows():
        x1 = float(row["x1"])
        x2 = float(row["x2"])

        cov1_samples = rng.uniform(0.0, 1.0, size=n_samples)
        x1_opt = 6.0 + 1.5 * (cov1_samples - 0.5)
        x2_opt = 2.5 - 1.0 * (cov1_samples - 0.5)

        base = 200.0
        quad = -2.0 * (x1 - x1_opt) ** 2 - 3.0 * (x2 - x2_opt) ** 2
        interaction = 6.0 * np.sin(0.5 * x1) * np.cos(0.7 * x2)

        noise_std = noise_scale * (noise_base + noise_cov_scale * cov1_samples)
        noise = rng.normal(0.0, noise_std, size=n_samples)

        y_samples = base + quad + interaction + 10.0 * cov1_samples + noise

        for i in range(n_samples):
            rows.append(
                {"x1": x1, "x2": x2, "cov1": cov1_samples[i], "y": y_samples[i]}
            )

    return pd.DataFrame(rows)


def compute_ground_truth(x1_grid, x2_grid, cov1_mean):
    """Compute ground truth function values on a grid.

    Returns array with shape ``(len(x1_grid), len(x2_grid))`` using
    ``'ij'`` indexing.
    """
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid, indexing="ij")

    x1_opt = 6.0 + 1.5 * (cov1_mean - 0.5)
    x2_opt = 2.5 - 1.0 * (cov1_mean - 0.5)

    base = 200.0
    quad = -2.0 * (x1_mesh - x1_opt) ** 2 - 3.0 * (x2_mesh - x2_opt) ** 2
    interaction = 6.0 * np.sin(0.5 * x1_mesh) * np.cos(0.7 * x2_mesh)

    return base + quad + interaction + 10.0 * cov1_mean
