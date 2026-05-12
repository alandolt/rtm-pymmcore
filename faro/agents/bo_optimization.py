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
        penalty: ``None``, ``"delta"``, or ``"inverse_distance"`` to
            discourage re-evaluation at/near recent points.
        penalty_factor: Strength of the inverse-distance penalty.
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
        penalty: Optional[str] = None,
        penalty_factor: float = 1.0,
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

        self._acquisition_used_this_round = self.acquisition_function

        self.verbose = verbose
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
        """
        ...

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

        For single-condition BO this is one BO iteration: pick the next
        parameter set (initial spread for early phases, GP-acquisition
        otherwise), build events, run/continue the experiment, wait for
        the pipeline, read tracks and append observations to
        :attr:`df_results`.

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
            fovs: Optional list of FOV indices.  Defaults to
                ``range(len(fov_positions))`` when *fov_positions* is given.

        Returns:
            ``{"params": ..., "df_new": ..., "phase_id": phase_id}`` or
            ``None`` if no results were produced.
        """
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
        if self.df_results.empty or len(self.df_results) < 4:
            initial, _ = self._select_initial_samples(k=1)
            if isinstance(initial, dict):
                params = initial
            else:
                params = initial[0]
            print(f"=== Phase {phase_id}: initial sample {params} ===")
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
        self._wait_for_pipeline()
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

    def run(self) -> None:
        """Execute the full BO experiment loop.

        Convenience wrapper that calls :meth:`run_one_phase` for the
        configured number of iterations and then finalises the
        controller.  Equivalent to:

        .. code-block:: python

            for k in range(self.n_iterations + 1):
                agent.run_one_phase(k)
            agent.controller.finish_experiment()

        Raises:
            ValueError: If no FOVs have been configured via :meth:`add_fovs`.
        """
        if not self.fovs:
            raise ValueError("No FOVs configured. Call add_fovs() before run().")

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

            self._wait_for_pipeline()
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

            self._wait_for_pipeline()
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

        self.controller.finish_experiment()

    # _wait_for_pipeline is inherited from InterPhaseAgent

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
        X_scaled = x_scaler.fit_transform(
            self.x_unmeasured, bounds=bounds, log_scale=log_scale
        )
        X_selected_scaled, selected_indices = self._extract_spread_points(X_scaled, k=k)
        X_selected = x_scaler.inverse_transform(X_selected_scaled)
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
            if maximize:
                best_f_per_cov = jnp.max(mean_matrix, axis=0)
            else:
                best_f_per_cov = jnp.min(mean_matrix, axis=0)

            print(
                f"  best_f per covariate (scaled): "
                f"min={float(jnp.min(best_f_per_cov)):.6f}, "
                f"max={float(jnp.max(best_f_per_cov)):.6f}"
            )

            sigma_matrix = jnp.sqrt(jnp.maximum(var_matrix, 1e-12))
            u = (mean_matrix - (best_f_per_cov[None, :] + xi_value)) / sigma_matrix
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
        """Fit GP, compute acquisition, and return the next parameter dict."""
        import gpax
        import gpax.utils
        import jax.numpy as jnp

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
            num_warmup=200,
            num_samples=400,
        )

        # Store for post-hoc analysis
        self.model = gp_model
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler
        self._rng_key_predict = rng_key_predict

        x_grid_ctrl = self.x_unmeasured.copy()
        acquisition_used = self.acquisition_function
        current_xi = self._current_ei_xi()

        if len(self.bo_covariates) > 0:
            # Robust acquisition: sample covariates from observed distribution
            cov_samples_list = []
            for covariate in self.bo_covariates:
                cov_vals = np.asarray(df_results[covariate.name].values, dtype=float)
                if cov_vals.size == 0:
                    cov_samples = np.array([0.0])
                else:
                    cov_samples = self._rng.choice(
                        cov_vals, size=self.n_cov_samples, replace=True
                    )
                cov_samples_list.append(cov_samples)

            if len(cov_samples_list) == 1:
                cov_grid = cov_samples_list[0].reshape(-1, 1)
            else:
                cov_mesh = np.meshgrid(*cov_samples_list, indexing="ij")
                cov_grid = np.stack([m.ravel() for m in cov_mesh], axis=1)

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
                best_f_scaled = jnp.max(y) if maximize else jnp.min(y)
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
    ):
        """Plot measured data, model predictions, and acquisition function.

        Creates 4 subplots:
        1. Measured points (x1, x2) with y as colour
        2. Ground truth (if available) or GP model uncertainty
        3. Acquisition function
        4. GP predicted landscape
        """
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        x_total_ctrl = self.x_total_linespace.copy()
        unique_x1 = np.unique(x_total_ctrl[:, 0])
        unique_x2 = np.unique(x_total_ctrl[:, 1])

        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_vals = np.asarray(
                df_results[self.bo_covariates[0].name].values, dtype=float
            )
            cov_min = float(np.nanmin(cov_vals))
            cov_max = float(np.nanmax(cov_vals))
            cov_mean = float(np.nanmean(cov_vals))
            n_cov_samples = 10
            cov_samples = np.linspace(cov_min, cov_max, n_cov_samples)
        else:
            cov_mean = 0.0
            cov_samples = np.array([0.0])
            n_cov_samples = 1

        # Build grid for predictions
        x_grid_full = []
        for x1 in unique_x1:
            for x2 in unique_x2:
                for cov in cov_samples:
                    x_grid_full.append(
                        [x1, x2, cov] if len(self.bo_covariates) > 0 else [x1, x2]
                    )
        x_grid_full = np.array(x_grid_full)

        x_grid_scaled = x_scaler.transform(x_grid_full)
        y_pred_scaled, y_samples_scaled = gp_model.predict(
            rng_key_predict, x_grid_scaled, noiseless=True
        )
        y_pred = y_scaler.inverse_transform(y_pred_scaled).flatten()

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
                    cov_val = group[self.bo_covariates[0].name].mean()
                    ax1.annotate(
                        f"{cov_val:.2f}",
                        xy=(x1, x2),
                        xytext=(0, 8),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=8,
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
        if (
            self.x_performed_experiments is not None
            and len(self.x_performed_experiments) > 0
        ):
            ax3.scatter(
                self.x_performed_experiments[:, 0],
                self.x_performed_experiments[:, 1],
                c="gray",
                s=80,
                marker="x",
                linewidths=2,
                label="Already measured",
                alpha=0.6,
                zorder=2,
            )
        if next_point is not None:
            ax3.scatter(
                float(next_point[0]),
                float(next_point[1]),
                c="cyan",
                s=220,
                marker="X",
                edgecolors="black",
                linewidths=1.5,
                label="Chosen next",
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
