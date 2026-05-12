"""Sparse / variational Bayesian Optimization agent.

Companion to :mod:`faro.agents.bo_optimization`.  ``BOptGPAXSparse`` is a
drop-in replacement for :class:`faro.agents.bo_optimization.BOptGPAX` that
swaps the ExactGP+MCMC backend for ``gpax.viSparseGP`` (or ``gpax.viGP``)
and uses **closed-form** Expected Improvement / Upper Confidence Bound
acquisition instead of Monte-Carlo sampling.

Why a separate class?  ExactGP scales as ``O(n^3)`` in the number of
observations, so per-cell BO loops with thousands of observations per
iteration become infeasible after ~10 phases.  ``viSparseGP`` reduces
this to ``O(n * m^2)`` (with ``m`` inducing points) and returns
``(mean, variance)`` directly, making the closed-form acquisition
expressions valid.

The class only overrides :meth:`_determine_next_parameters`; everything
else (event creation, batch selection, run loop, FOV-index offsetting,
``run_one_phase`` composition, EI decay schedule, penalty handling,
covariate marginalisation logic) is inherited unchanged from
``BOptGPAX``.

Use this class for *generic* sparse-GP BO; subclass it (just like you
would ``BOptGPAX``) to add experiment-specific event creation and
result preprocessing.

Requires ``gpax`` and ``jax`` as dependencies (imported lazily inside
``_determine_next_parameters`` so importing this module without them
installed does not crash).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from faro.agents.bo_optimization import BOptGPAX, StandardScalerBounds


def _safe_batch_size(n_rows: int, requested: int) -> int:
    """Clamp *requested* batch size to avoid two bugs in gpax's
    ``split_in_batches`` (``viSparseGP.predict_in_batches``):

    1. ``batch_size >= n_rows`` → internal ``for`` loop iterates zero
       times and a post-loop reference to the loop variable raises
       ``UnboundLocalError``.
    2. ``n_rows % batch_size == 1`` → the trailing batch of 1 row
       gets squeezed to a 0-D array and ``jnp.concatenate`` fails with
       ``Cannot concatenate arrays with different numbers of dimensions``.

    This helper returns a batch size that avoids both: strictly less
    than *n_rows*, and arranged so no batch has exactly 1 row.
    """
    bs = min(int(requested), n_rows - 1)
    if bs < 1:
        bs = 1
    # Bug 2: walk bs down until no batch (including remainder) has size 1.
    while bs > 1 and n_rows % bs == 1:
        bs -= 1
    return bs


class BOptGPAXSparse(BOptGPAX):
    """Bayesian Optimization agent backed by a variational sparse GP.

    Subclasses :class:`faro.agents.bo_optimization.BOptGPAX` and
    overrides only :meth:`_determine_next_parameters` to use
    ``gpax.viSparseGP`` (or ``gpax.viGP``) plus closed-form EI/UCB
    acquisition.  All other behaviour (template methods, run loop,
    ``run_one_phase``, batch acquisition, EI decay schedule, penalty
    handling, covariate marginalisation, plotting) is inherited.

    Args:
        gp_backend: ``"vi_sparse"`` (default, scales best with inducing
            points) or ``"vi"`` (full variational, no inducing points --
            use for medium datasets where the speedup is unnecessary).
        inducing_points_ratio: Fraction of training points used as
            inducing points (only used when ``gp_backend="vi_sparse"``).
            ``0.05`` = 5% inducing points, a reasonable starting point.
        num_svi_steps: Number of SVI optimisation steps for the
            variational fit.
        svi_step_size: Adam step size for SVI.
        gp_batch_size: Batch size for ``predict_in_batches`` (caps
            peak GPU memory during prediction on large grids).
        **kwargs: Forwarded to :class:`BOptGPAX`.
    """

    def __init__(
        self,
        *,
        gp_backend: str = "vi_sparse",
        inducing_points_ratio: float = 0.1,
        num_svi_steps: int = 1000,
        svi_step_size: float = 5e-3,
        gp_batch_size: int = 2000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if gp_backend not in {"vi_sparse", "vi"}:
            raise ValueError(
                f"Unknown gp_backend={gp_backend!r}, use 'vi_sparse' or 'vi'"
            )
        self.gp_backend = gp_backend
        self.inducing_points_ratio = float(inducing_points_ratio)
        self.num_svi_steps = int(num_svi_steps)
        self.svi_step_size = float(svi_step_size)
        self.gp_batch_size = int(gp_batch_size)

    # ------------------------------------------------------------------
    # viSparseGP fit + analytic EI/UCB (overrides ExactGP+MC)
    # ------------------------------------------------------------------

    def _determine_next_parameters(
        self, df_results: pd.DataFrame, verbose: bool = False
    ) -> dict:
        """Fit viSparseGP/viGP and compute analytic acquisition.

        Unlike :meth:`BOptGPAX._determine_next_parameters`, the
        variational backend returns ``(mean, variance)`` instead of
        posterior samples, so EI / UCB use closed-form expressions
        rather than Monte Carlo averages.
        """
        import gpax
        import gpax.utils
        from scipy.stats import norm

        x_scaler = StandardScalerBounds()
        bounds, log_scale = self._get_bounds_and_log_scale()
        y_scaler = StandardScalerBounds()
        y_log_scale = [self.objective_metric.log_scale]

        x_raw = self._extract_x_from_df(df_results)
        y_raw = self._extract_y_from_df(df_results)

        x = x_scaler.fit_transform(x_raw, bounds=bounds, log_scale=log_scale)
        y = y_scaler.fit_transform(y_raw, bounds=None, log_scale=y_log_scale)
        x_np = np.asarray(x)
        y_flat = np.asarray(y).flatten()

        rng_key, rng_key_predict = gpax.utils.get_keys()

        # ----- Fit variational GP -----
        print(
            f"  Fitting {self.gp_backend} on {x_np.shape[0]} observations "
            f"({x_np.shape[1]}-D)..."
        )
        if self.gp_backend == "vi_sparse":
            gp_model = gpax.viSparseGP(input_dim=x_np.shape[1], kernel="Matern")
            gp_model.fit(
                rng_key,
                X=x_np,
                y=y_flat,
                inducing_points_ratio=self.inducing_points_ratio,
                num_steps=self.num_svi_steps,
                step_size=self.svi_step_size,
                progress_bar=verbose,
                print_summary=verbose,
            )
        else:  # "vi"
            gp_model = gpax.viGP(input_dim=x_np.shape[1], kernel="Matern")
            gp_model.fit(
                rng_key,
                X=x_np,
                y=y_flat,
                num_steps=self.num_svi_steps,
                step_size=self.svi_step_size,
                progress_bar=verbose,
                print_summary=verbose,
            )

        # Print ARD lengthscales in the same shape as the ExactGP backend
        # (small lengthscale = high relevance).  Always on; cheap, and one
        # of the most useful diagnostics for a sparse GP.
        try:
            samples = gp_model.get_samples()
            k_length = np.asarray(samples["k_length"]).squeeze()
            k_scale = float(np.asarray(samples.get("k_scale", np.nan)).mean())
            noise = float(np.asarray(samples.get("noise", np.nan)).mean())
            dim_names = [p.name for p in self.parameters_to_optimize] + [
                c.name for c in self.bo_covariates
            ]
            print("  ARD lengthscales (small = highly relevant):")
            for name, ls in zip(dim_names, np.atleast_1d(k_length)):
                print(f"    {name:28s}  lengthscale={float(ls):6.3f}")
            print(f"    k_scale={k_scale:.3f}   noise={noise:.3f}")
        except Exception as exc:
            print(f"  (k_length readout unavailable: {type(exc).__name__}: {exc})")

        # Store for post-hoc analysis (mirrors BOptGPAX)
        self.model = gp_model
        self._x_scaler = x_scaler
        self._y_scaler = y_scaler
        self._rng_key_predict = rng_key_predict

        x_grid_ctrl = self.x_unmeasured.copy()
        n_grid = x_grid_ctrl.shape[0]
        n_ctrl = len(self.parameters_to_optimize)
        current_xi = self._current_ei_xi()
        maximize = self.objective_metric.goal == "maximize"

        # ----- Build prediction grid (with covariate marginalisation) -----
        if len(self.bo_covariates) > 0:
            cov_grid = self._sample_covariate_grid(df_results)
            n_mc = cov_grid.shape[0]
            x_repeated = np.repeat(x_grid_ctrl, n_mc, axis=0)
            c_tiled = np.tile(cov_grid, (n_grid, 1))
            x_full = np.hstack([x_repeated, c_tiled])
            x_full_scaled = np.asarray(x_scaler.transform(x_full))

            print(
                f"  Predicting {x_full_scaled.shape[0]} points "
                f"({n_grid} grid x {n_mc} cov samples)..."
            )
            _bs = _safe_batch_size(x_full_scaled.shape[0], self.gp_batch_size)
            mean_pred, var_pred = gp_model.predict_in_batches(
                rng_key_predict,
                x_full_scaled,
                batch_size=_bs,
                noiseless=True,
            )
            mean_matrix = np.asarray(mean_pred).reshape(n_grid, n_mc)
            var_matrix = np.asarray(var_pred).reshape(n_grid, n_mc)
            sigma_matrix = np.sqrt(np.maximum(var_matrix, 1e-12))

            if self.acquisition_function == "ei":
                # Best mean per covariate scenario (analog of base class behavior)
                if maximize:
                    best_f_per_cov = np.max(mean_matrix, axis=0)
                else:
                    best_f_per_cov = np.min(mean_matrix, axis=0)
                u = (
                    mean_matrix - (best_f_per_cov[None, :] + current_xi)
                ) / sigma_matrix
                if not maximize:
                    u = -u
                # Closed-form EI: sigma * (u * Phi(u) + phi(u))
                acq_matrix = sigma_matrix * (u * norm.cdf(u) + norm.pdf(u))
            else:  # UCB
                beta_sqrt = np.sqrt(self.ucb_beta)
                if maximize:
                    acq_matrix = mean_matrix + beta_sqrt * sigma_matrix
                else:
                    acq_matrix = -(mean_matrix - beta_sqrt * sigma_matrix)

            acq_matrix = np.where(np.isfinite(acq_matrix), acq_matrix, 0.0)
            # Marginalise over covariate samples
            acq_values_total = np.mean(acq_matrix, axis=1)
        else:
            x_total_scaled = np.asarray(x_scaler.transform(x_grid_ctrl))
            print(f"  Predicting {x_total_scaled.shape[0]} grid points...")
            _bs = _safe_batch_size(x_total_scaled.shape[0], self.gp_batch_size)
            mean_pred, var_pred = gp_model.predict_in_batches(
                rng_key_predict,
                x_total_scaled,
                batch_size=_bs,
                noiseless=True,
            )
            mean = np.asarray(mean_pred)
            sigma = np.sqrt(np.maximum(np.asarray(var_pred), 1e-12))

            if self.acquisition_function == "ei":
                best_f_scaled = float(np.max(y_flat) if maximize else np.min(y_flat))
                u = (mean - (best_f_scaled + current_xi)) / sigma
                if not maximize:
                    u = -u
                acq_values_total = sigma * (u * norm.cdf(u) + norm.pdf(u))
            else:
                beta_sqrt = np.sqrt(self.ucb_beta)
                if maximize:
                    acq_values_total = mean + beta_sqrt * sigma
                else:
                    acq_values_total = -(mean - beta_sqrt * sigma)
            acq_values_total = np.where(
                np.isfinite(acq_values_total), acq_values_total, 0.0
            )

        # ----- Apply penalty (for batch diversity) -----
        if self.penalty is not None and self.x_performed_experiments is not None:
            recent_ctrl_raw = self.x_performed_experiments[:, :n_ctrl]
            ctrl_log = (
                x_scaler.log_scale[:n_ctrl] if x_scaler.log_scale is not None else None
            )
            ctrl_mean = np.asarray(x_scaler.mean_)[:n_ctrl]
            ctrl_std = np.asarray(x_scaler.std_)[:n_ctrl]

            def _scale_ctrl(X):
                X = np.asarray(X, dtype=float).copy()
                if ctrl_log is not None:
                    for i, use_log in enumerate(ctrl_log):
                        if use_log:
                            X[:, i] = np.log(X[:, i])
                return (X - ctrl_mean) / ctrl_std

            x_grid_ctrl_scaled = _scale_ctrl(x_grid_ctrl)
            recent_ctrl_scaled = _scale_ctrl(recent_ctrl_raw)

            if self.penalty == "delta":
                for rp in recent_ctrl_scaled:
                    distances = np.linalg.norm(x_grid_ctrl_scaled - rp, axis=1)
                    mask = distances < 1e-8
                    acq_values_total = np.where(mask, -np.inf, acq_values_total)
            elif self.penalty == "inverse_distance":
                min_distances = np.full(x_grid_ctrl_scaled.shape[0], np.inf)
                for rp in recent_ctrl_scaled:
                    distances = np.linalg.norm(x_grid_ctrl_scaled - rp, axis=1)
                    min_distances = np.minimum(min_distances, distances)
                penalty_term = 1.0 / (min_distances + 1e-6)
                acq_values_total = acq_values_total / (
                    1.0 + self.penalty_factor * penalty_term
                )

        # Stash diagnostic context for _plot_live in subclasses.
        # Matches the schema used by BOptGPAX so OscillationBO-style
        # live plots work without a per-backend branch.  Only stash on
        # the first call in a batch (later calls see a penalised
        # landscape which would be misleading); use the same "no
        # existing context" test the live plot gate below relies on.
        if getattr(self, "_last_plot_context", None) is None:
            self._last_plot_context = dict(
                df_results=df_results,
                x_scaler=x_scaler,
                y_scaler=y_scaler,
                gp_model=gp_model,
                rng_key_predict=rng_key_predict,
                x_unmeasured_at_computation=x_grid_ctrl.copy(),
                acq_values_total=np.asarray(acq_values_total),
                acquisition_used=self.acquisition_function,
                current_xi=current_xi,
                cov_grid=cov_grid if len(self.bo_covariates) > 0 else None,
                y=np.asarray(y),
            )

        # ----- Select argmax and update state -----
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

        return {
            param.name: next_parameters[i]
            for i, param in enumerate(self.parameters_to_optimize)
        }
