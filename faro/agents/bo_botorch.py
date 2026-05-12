"""Bayesian Optimization agent using BoTorch (PyTorch / GPyTorch).

Drop-in replacement for :class:`faro.agents.bo_optimization.BOptGPAX` that
swaps the gpax/JAX backend for BoTorch/PyTorch.  Uses:

* ``SingleTaskGP`` with Matern 5/2 kernel (MLE via L-BFGS)
* Built-in ``Normalize`` / ``Standardize`` transforms
* Closed-form EI / UCB acquisition with covariate marginalisation
* Sequential greedy batch acquisition with local penalisation
  (inherited from :class:`BOptGPAX`)

The class overrides only the GP-related methods
(:meth:`_determine_next_parameters`, :meth:`_select_initial_samples`,
:meth:`save_model`); the run loop, event creation, batch selection,
FOV-index offsetting, and all template methods are inherited unchanged
from :class:`BOptGPAX`.

Requires ``botorch``, ``gpytorch``, and ``torch`` (imported lazily so
importing this module without them installed does not crash).
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from faro.agents.bo_optimization import BOptGPAX


# ---------------------------------------------------------------------------
# Pure-numpy scaler (replaces JAX-based StandardScalerBounds)
# ---------------------------------------------------------------------------


class _NumpyScaler:
    """Bounds-based scaler using pure numpy.

    Functionally equivalent to
    :class:`faro.agents.bo_optimization.StandardScalerBounds` but without
    any JAX dependency.  Used only for initial-sample spread-point
    normalisation inside :class:`BOptBoTorch`.
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.log_scale = None

    def fit(self, X, bounds=None, log_scale=None):
        X = np.asarray(X, dtype=np.float64).copy()
        self.log_scale = log_scale
        if log_scale is not None:
            for i, use_log in enumerate(log_scale):
                if use_log:
                    X[:, i] = np.log(X[:, i])
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        if bounds is not None:
            for i, bound in enumerate(bounds):
                if bound is not None:
                    lo, hi = float(bound[0]), float(bound[1])
                    if log_scale is not None and log_scale[i]:
                        lo, hi = np.log(lo), np.log(hi)
                    self.mean_[i] = (lo + hi) / 2.0
                    self.std_[i] = (hi - lo) / 2.0
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        if self.log_scale is not None:
            for i, use_log in enumerate(self.log_scale):
                if use_log:
                    X[:, i] = np.log(X[:, i])
        return (X - self.mean_) / self.std_

    def fit_transform(self, X, bounds=None, log_scale=None):
        return self.fit(X, bounds=bounds, log_scale=log_scale).transform(X)

    def inverse_transform(self, X):
        X = np.asarray(X, dtype=np.float64).copy()
        X = X * self.std_ + self.mean_
        if self.log_scale is not None:
            for i, use_log in enumerate(self.log_scale):
                if use_log:
                    X[:, i] = np.exp(X[:, i])
        return X


# ---------------------------------------------------------------------------
# BOptBoTorch
# ---------------------------------------------------------------------------


class BOptBoTorch(BOptGPAX):
    """Bayesian Optimization agent backed by BoTorch.

    Subclasses :class:`BOptGPAX` and overrides the GP fitting /
    acquisition layer.  All run-loop machinery, template methods, batch
    management, and phase logging are inherited unchanged.

    This is a plug-and-play replacement for :class:`BOptGPAX`: subclass
    it exactly the same way (implement ``_create_events_for_cycle`` /
    ``_preprocess_results``) or combine it with existing experiment
    subclasses via multiple inheritance (see
    :class:`faro.agents.bo_botorch_oscillation.OscillationBOBoTorch`).

    Additional args (all others forwarded to :class:`BOptGPAX`):
        gp_num_restarts: Random restarts for GP hyper-parameter MLE.
        gp_raw_samples: Raw random candidate count for acquisition
            optimisation (only used when ``optimize_acqf`` is called
            internally — not currently used for discrete grids, but
            reserved for future continuous-relaxation support).
    """

    def __init__(
        self,
        *,
        gp_num_restarts: int = 5,
        gp_raw_samples: int = 64,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.gp_num_restarts = int(gp_num_restarts)
        self.gp_raw_samples = int(gp_raw_samples)

    # ------------------------------------------------------------------
    # Numpy helpers (no JAX)
    # ------------------------------------------------------------------

    def _apply_log_transforms(self, X: np.ndarray) -> np.ndarray:
        """Apply per-feature log transforms based on param/covariate config."""
        X = np.asarray(X, dtype=np.float64).copy()
        _, log_scale = self._get_bounds_and_log_scale()
        for i, use_log in enumerate(log_scale):
            if use_log:
                X[:, i] = np.log(X[:, i])
        return X

    def _get_torch_bounds(self, train_X=None):
        """Return ``(2, d)`` bounds tensor in the (log-)transformed space.

        Uses explicit parameter / covariate bounds when available; falls
        back to the training-data range (with 10 % margin) otherwise.
        """
        import torch

        bounds_list, log_scale = self._get_bounds_and_log_scale()
        lower, upper = [], []
        for i, b in enumerate(bounds_list):
            if b is not None:
                lo, hi = float(b[0]), float(b[1])
                if log_scale[i]:
                    lo, hi = np.log(lo), np.log(hi)
                lower.append(lo)
                upper.append(hi)
            elif train_X is not None:
                col = (
                    train_X[:, i].numpy()
                    if hasattr(train_X, "numpy")
                    else train_X[:, i]
                )
                lo, hi = float(np.min(col)), float(np.max(col))
                margin = 0.1 * (hi - lo + 1e-6)
                lower.append(lo - margin)
                upper.append(hi + margin)
            else:
                lower.append(0.0)
                upper.append(1.0)
        return torch.tensor([lower, upper], dtype=torch.double)

    @staticmethod
    def _is_flat_acq_np(
        acq_values: np.ndarray,
        range_tol: float = 1e-6,
        relative_tol: float = 0.01,
    ) -> bool:
        """Check whether acquisition values are effectively flat (numpy)."""
        finite = np.isfinite(acq_values)
        if not np.any(finite):
            return True
        acq_f = acq_values[finite]
        acq_range = float(np.max(acq_f) - np.min(acq_f))
        if acq_range < range_tol:
            return True
        acq_scale = float(np.max(np.abs(acq_f)))
        if acq_scale > 0 and acq_range / acq_scale < relative_tol:
            return True
        return False

    # ------------------------------------------------------------------
    # Initial sampling (pure numpy — no JAX scaler)
    # ------------------------------------------------------------------

    def _select_initial_samples(self, k=4):
        scaler = _NumpyScaler()
        bounds, log_scale = self._get_bounds_and_log_scale()
        X_scaled = scaler.fit_transform(
            self.x_unmeasured, bounds=bounds, log_scale=log_scale
        )
        X_selected_scaled, selected_indices = self._extract_spread_points(X_scaled, k=k)
        X_selected = scaler.inverse_transform(X_selected_scaled)
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
    # Core BO decision logic (BoTorch)
    # ------------------------------------------------------------------

    def _determine_next_parameters(
        self, df_results: pd.DataFrame, verbose: bool = False
    ) -> dict:
        """Fit SingleTaskGP and compute acquisition using BoTorch.

        Mirrors :meth:`BOptGPAX._determine_next_parameters` but uses a
        ``SingleTaskGP`` (Matern 5/2, MLE) with built-in input/output
        transforms and closed-form EI / UCB.

        Supports the ``_cached_gp_fit`` / ``_batch_fit_reuse`` pattern
        used by the inherited :meth:`_select_batch_parameters` so the
        GP is fit only once per batch phase.
        """
        import torch
        from botorch.fit import fit_gpytorch_mll
        from botorch.models import SingleTaskGP
        from botorch.models.transforms.input import Normalize
        from botorch.models.transforms.outcome import Standardize
        from gpytorch.mlls import ExactMarginalLogLikelihood
        from scipy.stats import norm

        # --- Check for cached GP fit (batch BO reuse) -------------------
        cache = getattr(self, "_cached_gp_fit", None)
        is_first_call_in_batch = cache is None

        if cache is not None:
            model = cache["gp_model"]
            train_X = cache["train_X"]
            train_Y = cache["train_Y"]
            print("  [reusing cached GP fit from earlier in this batch]")
        else:
            # --- Prepare training data ----------------------------------
            x_raw = self._extract_x_from_df(df_results)
            y_raw = self._extract_y_from_df(df_results)

            x_transformed = self._apply_log_transforms(x_raw)
            y_transformed = y_raw.copy()
            if self.objective_metric.log_scale:
                y_transformed = np.log(y_transformed)

            train_X = torch.tensor(x_transformed, dtype=torch.double)
            train_Y = torch.tensor(y_transformed, dtype=torch.double)

            bounds_tensor = self._get_torch_bounds(train_X)

            # --- Fit GP -------------------------------------------------
            print(
                f"  Fitting SingleTaskGP on {train_X.shape[0]} observations "
                f"({train_X.shape[1]}-D)..."
            )
            model = SingleTaskGP(
                train_X,
                train_Y,
                input_transform=Normalize(d=train_X.shape[-1], bounds=bounds_tensor),
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            # Store for post-hoc analysis
            self.model = model
            self._train_X = train_X
            self._train_Y = train_Y

            if getattr(self, "_batch_fit_reuse", False):
                self._cached_gp_fit = dict(
                    gp_model=model, train_X=train_X, train_Y=train_Y
                )

        # --- Build prediction grid --------------------------------------
        x_grid_ctrl = self.x_unmeasured.copy()
        n_grid = x_grid_ctrl.shape[0]
        n_ctrl = len(self.parameters_to_optimize)
        current_xi = self._current_ei_xi()
        maximize = self.objective_metric.goal == "maximize"

        # --- Evaluate acquisition (with covariate marginalisation) ------
        if len(self.bo_covariates) > 0:
            cov_grid = self._sample_covariate_grid(df_results)
            n_mc = cov_grid.shape[0]

            x_repeated = np.repeat(x_grid_ctrl, n_mc, axis=0)
            c_tiled = np.tile(cov_grid, (n_grid, 1))
            x_full = np.hstack([x_repeated, c_tiled])
            x_full_transformed = self._apply_log_transforms(x_full)
            X_pred = torch.tensor(x_full_transformed, dtype=torch.double)

            print(
                f"  Predicting {X_pred.shape[0]} points "
                f"({n_grid} grid x {n_mc} cov samples)..."
            )
            with torch.no_grad():
                posterior = model.posterior(X_pred)
                mean_flat = posterior.mean.squeeze(-1).numpy()
                var_flat = posterior.variance.squeeze(-1).numpy()

            mean_matrix = mean_flat.reshape(n_grid, n_mc)
            var_matrix = var_flat.reshape(n_grid, n_mc)
            sigma_matrix = np.sqrt(np.maximum(var_matrix, 1e-12))

            # best_f: GP-predicted best marginal mean (NOT observed max).
            # On sparse objectives like frac_oscillating the observed max
            # can sit 4-6 sigma above the mean, making closed-form EI
            # collapse to zero (see bo_optimization.py for full rationale).
            mean_marg_ref = mean_matrix.mean(axis=1)
            best_f = float(mean_marg_ref.max() if maximize else mean_marg_ref.min())
            acq_values_total, acquisition_used = self._acq_with_fallback(
                mean_matrix,
                sigma_matrix,
                best_f,
                current_xi,
                maximize,
                is_matrix=True,
            )
        else:
            x_ctrl_transformed = self._apply_log_transforms(x_grid_ctrl)
            X_pred = torch.tensor(x_ctrl_transformed, dtype=torch.double)

            print(f"  Predicting {X_pred.shape[0]} grid points...")
            with torch.no_grad():
                posterior = model.posterior(X_pred)
                mean = posterior.mean.squeeze(-1).numpy()
                variance = posterior.variance.squeeze(-1).numpy()

            sigma = np.sqrt(np.maximum(variance, 1e-12))
            # best_f: GP-predicted best mean (robust to observed outliers)
            best_f = float(mean.max() if maximize else mean.min())
            acq_values_total, acquisition_used = self._acq_with_fallback(
                mean,
                sigma,
                best_f,
                current_xi,
                maximize,
                is_matrix=False,
            )

        self._acquisition_used_this_round = acquisition_used

        # --- Apply penalty (batch diversity) ----------------------------
        acq_values_total = self._apply_penalty_np(acq_values_total, x_grid_ctrl, n_ctrl)

        # --- Select argmax and update state -----------------------------
        next_idx = int(np.argmax(acq_values_total))
        next_parameters = np.asarray(x_grid_ctrl[next_idx])

        self.x_unmeasured = np.delete(self.x_unmeasured, next_idx, axis=0)
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

        # Stash plot context for the first call in a batch (clean,
        # unpenalised landscape — see BOptGPAX._determine_next_parameters
        # docstring for rationale).
        if is_first_call_in_batch:
            self._last_plot_context = dict(
                acq_values_total=np.asarray(acq_values_total),
                acquisition_used=acquisition_used,
                x_unmeasured_at_computation=x_grid_ctrl.copy(),
                current_xi=current_xi,
                gp_model=model,
                train_X=train_X,
                train_Y=train_Y,
                backend="botorch",
            )

        if verbose:
            print(
                f"  Acq ({acquisition_used}): "
                f"min={float(np.min(acq_values_total)):.6f}, "
                f"max={float(np.max(acq_values_total)):.6f}"
            )

        return next_parameters_dict

    # ------------------------------------------------------------------
    # Acquisition helpers
    # ------------------------------------------------------------------

    def _acq_with_fallback(
        self,
        mean,
        sigma,
        best_f: float,
        xi: float,
        maximize: bool,
        *,
        is_matrix: bool,
    ) -> tuple[np.ndarray, str]:
        """Compute EI (with UCB fallback if flat).

        If ``is_matrix`` is True, ``mean`` and ``sigma`` have shape
        ``(n_grid, n_mc)`` and the result is marginalised over the last
        axis.  Otherwise they are 1-D arrays of shape ``(n_grid,)``.
        """
        from scipy.stats import norm

        acq, used = self._compute_acq_np(
            self.acquisition_function,
            mean,
            sigma,
            best_f,
            xi,
            maximize,
            is_matrix=is_matrix,
        )
        if used == "ei" and self._is_flat_acq_np(acq):
            print(
                "  EI is flat (max-min < 1e-6). " "Falling back to UCB for this round."
            )
            acq, used = self._compute_acq_np(
                "ucb",
                mean,
                sigma,
                best_f,
                xi,
                maximize,
                is_matrix=is_matrix,
            )
        return acq, used

    def _compute_acq_np(
        self,
        acq_type: str,
        mean: np.ndarray,
        sigma: np.ndarray,
        best_f: float,
        xi: float,
        maximize: bool,
        *,
        is_matrix: bool,
    ) -> tuple[np.ndarray, str]:
        """Closed-form EI or UCB in numpy."""
        from scipy.stats import norm

        if acq_type == "ei":
            u = (mean - (best_f + xi)) / sigma
            if not maximize:
                u = -u
            acq = sigma * (u * norm.cdf(u) + norm.pdf(u))
        else:  # ucb
            beta_sqrt = np.sqrt(self.ucb_beta)
            if maximize:
                acq = mean + beta_sqrt * sigma
            else:
                acq = -(mean - beta_sqrt * sigma)

        acq = np.where(np.isfinite(acq), acq, 0.0)

        if is_matrix:
            acq = np.mean(acq, axis=1)  # marginalise over covariates

        return acq, acq_type

    def _apply_penalty_np(
        self,
        acq_values: np.ndarray,
        x_grid_ctrl: np.ndarray,
        n_ctrl: int,
    ) -> np.ndarray:
        """Apply penalty for batch diversity (pure numpy)."""
        if self.penalty is None or self.x_performed_experiments is None:
            return acq_values

        recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl]
        _, log_scale = self._get_bounds_and_log_scale()

        def _scale_ctrl(X: np.ndarray) -> np.ndarray:
            X = np.asarray(X, dtype=np.float64).copy()
            for i, param in enumerate(self.parameters_to_optimize):
                if log_scale[i]:
                    X[:, i] = np.log(X[:, i])
                lo, hi = float(param.bounds[0]), float(param.bounds[1])
                if log_scale[i]:
                    lo, hi = np.log(lo), np.log(hi)
                rng = hi - lo
                if rng > 0:
                    X[:, i] = (X[:, i] - lo) / rng
            return X

        grid_scaled = _scale_ctrl(x_grid_ctrl)
        recent_scaled = _scale_ctrl(recent_ctrl_raw)

        acq = acq_values.copy()
        if self.penalty == "delta":
            for rp in recent_scaled:
                distances = np.linalg.norm(grid_scaled - rp, axis=1)
                acq = np.where(distances < 1e-8, -np.inf, acq)
        elif self.penalty == "inverse_distance":
            min_distances = np.full(grid_scaled.shape[0], np.inf)
            for rp in recent_scaled:
                distances = np.linalg.norm(grid_scaled - rp, axis=1)
                min_distances = np.minimum(min_distances, distances)
            penalty_term = 1.0 / (min_distances + 1e-6)
            acq = acq / (1.0 + self.penalty_factor * penalty_term)

        return acq

    # ------------------------------------------------------------------
    # Model persistence
    # ------------------------------------------------------------------

    def save_model(self, path: str | None = None) -> str | None:
        """Save the trained BoTorch GP + metadata for post-hoc analysis.

        Analogous to :meth:`BOptGPAX.save_model` but serialises a
        PyTorch ``state_dict`` instead of NumPyro MCMC samples.
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
            import torch

            model_state = {
                "backend": "botorch_SingleTaskGP",
                "state_dict": {
                    k: v.detach().cpu().numpy()
                    for k, v in self.model.state_dict().items()
                },
                "train_X": (
                    self._train_X.numpy()
                    if hasattr(self, "_train_X") and self._train_X is not None
                    else None
                ),
                "train_Y": (
                    self._train_Y.numpy()
                    if hasattr(self, "_train_Y") and self._train_Y is not None
                    else None
                ),
            }

            payload = dict(
                model_state=model_state,
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
        except Exception as exc:
            print(f"  Warning: could not save BO model to {path}: {exc}")
            return None
