"""Batch BO agent for ERK oscillation optimisation (BoTorch backend).

Plug-and-play replacement for :class:`faro.agents.bo_oscillation.OscillationBO`
that uses the :class:`faro.agents.bo_botorch.BOptBoTorch` GP backend
(BoTorch / PyTorch) instead of gpax / JAX.

The experiment-specific logic (event creation, oscillation classification,
result preprocessing, live plotting, checkpointing) is inherited from
:class:`OscillationBO`.  The GP fitting, acquisition, initial sampling,
and model persistence are inherited from :class:`BOptBoTorch`.

Python MRO ensures methods resolve correctly::

    OscillationBOBoTorch
      -> OscillationBO       (events, preprocessing, plotting)
      -> BOptBoTorch          (GP, acquisition, save_model)
      -> BOptGPAX             (run loop, batch management)
      -> InterPhaseAgent      (base)

Usage is identical to :class:`OscillationBO` — just swap the class::

    # Before:
    agent = OscillationBO(storage_path=..., ...)

    # After:
    agent = OscillationBOBoTorch(storage_path=..., ...)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from faro.agents.bo_botorch import BOptBoTorch
from faro.agents.bo_oscillation import OscillationBO


class OscillationBOBoTorch(OscillationBO, BOptBoTorch):
    """Batch BO for ERK oscillation with BoTorch GP backend.

    Inherits experiment-specific methods from :class:`OscillationBO`
    and BoTorch GP methods from :class:`BOptBoTorch`.

    Accepts all keyword arguments of both parent classes.  The
    BoTorch-specific additions are:

    * ``gp_num_restarts`` (default 5)
    * ``gp_raw_samples`` (default 64)

    All other arguments (``n_frames``, ``first_frame_stim``,
    ``parameters_to_optimize``, ``n_conditions_per_iter``, etc.) are
    passed through unchanged.
    """

    # OscillationBO._plot_landscape_and_acq_from_context uses gpax's
    # predict_in_batches API.  Override with a BoTorch-compatible version.

    def _plot_landscape_and_acq_from_context(
        self,
        ctx: dict,
        df_results: pd.DataFrame,
        ax_mean,
        ax_acq,
        fig,
    ) -> None:
        """Render GP landscape + acquisition heatmaps (BoTorch)."""
        import torch

        model = ctx["gp_model"]
        acq_values_total = ctx["acq_values_total"]
        acquisition_used = ctx["acquisition_used"]
        x_unmeasured = ctx["x_unmeasured_at_computation"]

        x_total_ctrl = self.x_total_linespace.copy()
        unique_x1 = np.unique(x_total_ctrl[:, 0])
        unique_x2 = np.unique(x_total_ctrl[:, 1])
        n_ctrl = len(unique_x1) * len(unique_x2)

        # Covariate samples for marginalisation
        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_cols = [c.name for c in self.bo_covariates]
            cov_vals_full = df_results[cov_cols].to_numpy(dtype=float)
            n_cov_samples = 50
            _plot_rng = np.random.default_rng(0)
            row_idx = _plot_rng.integers(0, cov_vals_full.shape[0], size=n_cov_samples)
            cov_samples = cov_vals_full[row_idx]
        else:
            n_cov_samples = 1
            cov_samples = None

        ctrl_grid = np.array([[x1, x2] for x1 in unique_x1 for x2 in unique_x2])

        if cov_samples is not None:
            x_grid_full = np.hstack(
                [
                    np.repeat(ctrl_grid, n_cov_samples, axis=0),
                    np.tile(cov_samples, (n_ctrl, 1)),
                ]
            )
        else:
            x_grid_full = ctrl_grid

        x_transformed = self._apply_log_transforms(x_grid_full)
        X_pred = torch.tensor(x_transformed, dtype=torch.double)

        with torch.no_grad():
            posterior = model.posterior(X_pred)
            y_pred = posterior.mean.squeeze(-1).numpy()

        # Inverse log-transform if needed
        if self.objective_metric.log_scale:
            y_pred = np.exp(y_pred)

        # Marginalise
        if cov_samples is not None:
            y_pred_marg = y_pred.reshape(n_ctrl, n_cov_samples).mean(axis=1)
        else:
            y_pred_marg = y_pred

        X_mesh, Y_mesh = np.meshgrid(unique_x1, unique_x2, indexing="ij")
        y_pred_2d = y_pred_marg.reshape(len(unique_x1), len(unique_x2))

        # --- Subplot: GP predicted landscape ---
        im1 = ax_mean.pcolormesh(
            X_mesh, Y_mesh, y_pred_2d, cmap="viridis", shading="auto"
        )
        fig.colorbar(im1, ax=ax_mean, label=f"predicted {self.objective_metric.name}")
        ax_mean.scatter(
            df_results["stim_exposure"],
            df_results["ramp"],
            c="white",
            s=15,
            alpha=0.6,
            marker="x",
            linewidths=0.8,
        )
        ax_mean.set_xlabel("stim_exposure (ms)")
        ax_mean.set_ylabel("ramp (ms/frame)")
        ax_mean.set_title("GP predicted landscape (marginalised)")

        # --- Subplot: Acquisition function ---
        acq_marg = np.asarray(acq_values_total)
        if len(acq_marg) != n_ctrl:
            acq_full = np.zeros(n_ctrl)
            x_full_ref = self.x_total_linespace
            for j, pt in enumerate(x_unmeasured):
                diffs = np.abs(x_full_ref - pt).sum(axis=1)
                idx = np.argmin(diffs)
                if j < len(acq_marg):
                    acq_full[idx] = float(acq_marg[j])
            acq_marg = acq_full

        acq_2d = acq_marg.reshape(len(unique_x1), len(unique_x2))
        im2 = ax_acq.pcolormesh(X_mesh, Y_mesh, acq_2d, cmap="inferno", shading="auto")
        acq_label = acquisition_used.upper()
        fig.colorbar(im2, ax=ax_acq, label=f"{acq_label} acquisition")
        ax_acq.scatter(
            df_results["stim_exposure"],
            df_results["ramp"],
            c="white",
            s=15,
            alpha=0.6,
            marker="x",
            linewidths=0.8,
        )

        # Overlay next batch picks (cyan crosses)
        picks = getattr(self, "_current_batch_picks", None)
        if picks:
            picks_arr = np.array(picks, dtype=float)
            ax_acq.scatter(
                picks_arr[:, 0],
                picks_arr[:, 1],
                c="cyan",
                s=120,
                marker="X",
                edgecolors="k",
                linewidths=1.0,
                zorder=10,
                label="next conditions",
            )
            ax_acq.legend(loc="upper right", fontsize=7)

        ax_acq.set_xlabel("stim_exposure (ms)")
        ax_acq.set_ylabel("ramp (ms/frame)")
        ax_acq.set_title(f"Acquisition {acq_label} (marginalised)")
