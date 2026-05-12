"""Structured GP BO agent for single-pulse dose-response experiments.

Uses :class:`gpax.StructuredGP` with a **Hill curve prior** to efficiently
characterise the dose-response relationship between a single stimulation
pulse duration and the cellular ERK-KTR response.

Protocol per FOV (per phase)
-----------------------------
1. **Baseline**: ``n_frames_baseline`` imaging frames (no stimulation).
2. **Pulse**: one stimulation frame at frame index ``n_frames_baseline``
   with exposure = ``pulse_duration`` ms (the BO-controllable parameter).
3. **Observation**: ``n_frames_observation`` imaging frames post-pulse.

Computed metrics per FOV
------------------------
* ``mean_delta_auc`` -- mean per-cell |AUC| of (cnr - baseline_cnr) over
  the observation window.  Primary objective by default.
* ``frac_responding`` -- fraction of cells whose ``delta_auc`` exceeds
  ``response_threshold``.  Stored as extra column for analysis.

Structured GP prior
-------------------
The Hill function ``V_max * d^n / (K^n + d^n)`` is used as the mean
function prior for the :class:`gpax.StructuredGP`.  The GP kernel then
captures residual structure (including covariate effects).  All Hill
parameters (``V_max``, ``K``, ``n``) are learnable via MCMC, so the
prior adapts to the data.

Subclasses :class:`faro.agents.bo_optimization.BOptGPAX` -- all batch-BO
machinery (sequential greedy acquisition, FOV-index offsets, EI/UCB,
covariate marginalisation) is inherited.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd

from faro.agents.bo_optimization import BOptGPAX, StandardScalerBounds
from faro.core.data_structures import RTMSequence

if TYPE_CHECKING:
    pass

import faro.core.utils as utils


class DoseResponseBO(BOptGPAX):
    """Structured GP BO for single-pulse dose-response characterisation.

    Each BO phase tests ``n_conditions_per_iter`` pulse durations
    simultaneously, splitting configured FOVs evenly between conditions.
    When driven by :class:`ComposedAgent` the FOVs are fresh every phase
    (chosen by ``FOVFinderAgent``); the GP accumulates observations
    across all phases.

    Args:
        n_frames_baseline: Number of imaging-only frames before the pulse.
        n_frames_observation: Number of imaging frames after the pulse.
        time_between_timesteps: Seconds between consecutive frames.
        imaging_channels: Tuple of imaging channels acquired every frame.
        stim_channel: Stimulation channel (exposure overridden per condition).
        optocheck_channel: Optional reference channel acquired on the last
            frame (e.g. mCitrine for optoRTK expression readout).  Pass
            ``None`` to skip.
        response_threshold: Per-cell ``delta_auc`` threshold for
            classifying a cell as "responding".  Default ``0.1``.
        min_track_fraction: Minimum fraction of experiment frames a cell
            must be tracked for.  Default ``0.8``.
        max_baseline_cnr: Cells with mean baseline CNR above this value
            are excluded (segmentation artefacts / already active).
            ``None`` disables filter.  Default ``0.8``.
        save_checkpoints: Write ``df_results`` to parquet after every phase.
        plot_live: Show live 1x3 progress plot after every phase.
        **kwargs: Forwarded to :class:`BOptGPAX` (includes
            ``parameters_to_optimize``, ``objective_metric``,
            ``n_conditions_per_iter``, etc.).
    """

    def __init__(
        self,
        *,
        n_frames_baseline: int = 10,
        n_frames_observation: int = 30,
        time_between_timesteps: float = 60.0,
        imaging_channels,
        stim_channel,
        optocheck_channel=None,
        response_threshold: float = 0.1,
        min_track_fraction: float = 0.8,
        max_baseline_cnr: float | None = 0.8,
        save_checkpoints: bool = True,
        plot_live: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_frames_baseline = int(n_frames_baseline)
        self.n_frames_observation = int(n_frames_observation)
        # total = baseline + 1 stim frame + observation
        self.n_frames = self.n_frames_baseline + 1 + self.n_frames_observation
        self.stim_frame = self.n_frames_baseline  # 0-based index of the pulse frame
        self.time_between_timesteps = float(time_between_timesteps)

        self.imaging_channels = imaging_channels
        self.stim_channel = stim_channel
        self.optocheck_channel = optocheck_channel

        self.response_threshold = float(response_threshold)
        self.min_track_fraction = float(min_track_fraction)
        self.max_baseline_cnr = (
            float(max_baseline_cnr) if max_baseline_cnr is not None else None
        )

        self.save_checkpoints = bool(save_checkpoints)
        self.plot_live = bool(plot_live)

    @property
    def fovs_per_condition(self) -> int:
        if not self.fovs:
            raise RuntimeError(
                "fovs_per_condition is undefined until FOVs have been "
                "configured (call run_one_phase with fov_positions=...)."
            )
        return len(self.fovs) // self.n_conditions_per_iter

    # ------------------------------------------------------------------
    # Event creation
    # ------------------------------------------------------------------

    def _create_events_for_cycle(self, parameters: dict) -> list:
        """Build events for a single-condition single-pulse experiment."""
        pulse_dur = float(parameters["pulse_duration"])
        acq = RTMSequence(
            time_plan={
                "interval": self.time_between_timesteps,
                "loops": self.n_frames,
            },
            stage_positions=self.fov_positions,
            channels=self.imaging_channels,
            stim_channels=(self.stim_channel,),
            stim_frames=frozenset({self.stim_frame}),
            stim_exposure=pulse_dur,
            ref_channels=((self.optocheck_channel,) if self.optocheck_channel else ()),
            ref_frames=(
                frozenset({self.n_frames - 1})
                if self.optocheck_channel
                else frozenset()
            ),
            rtm_metadata={
                "phase_name": f"dose_response_pulse_{pulse_dur:.0f}ms",
                "phase_id": self._phase_counter,
                "pulse_duration": pulse_dur,
            },
        )
        return list(acq)

    def _create_events_for_batch(self, param_list: list[dict]) -> list:
        """Build RTMEvents for all conditions in one batch iteration.

        Creates one :class:`RTMSequence` per condition (each with
        :attr:`fovs_per_condition` FOVs), remaps FOV indices for global
        uniqueness, and populates :attr:`_current_condition_map`.
        """
        phase_id = self._phase_counter
        all_events = []
        fovs_per_condition = self.fovs_per_condition

        self._current_condition_map = {}

        for cond_idx, params in enumerate(param_list):
            start = cond_idx * fovs_per_condition
            end = start + fovs_per_condition
            cond_positions = self.fov_positions[start:end]

            for fov_idx in range(start, end):
                self._current_condition_map[fov_idx + self._fov_index_offset] = params

            pulse_dur = float(params["pulse_duration"])

            acq = RTMSequence(
                time_plan={
                    "interval": self.time_between_timesteps,
                    "loops": self.n_frames,
                },
                stage_positions=cond_positions,
                channels=self.imaging_channels,
                stim_channels=(self.stim_channel,),
                stim_frames=frozenset({self.stim_frame}),
                stim_exposure=pulse_dur,
                ref_channels=(
                    (self.optocheck_channel,) if self.optocheck_channel else ()
                ),
                ref_frames=(
                    frozenset({self.n_frames - 1})
                    if self.optocheck_channel
                    else frozenset()
                ),
                rtm_metadata={
                    "phase_name": f"BO_iter_{phase_id}_cond_{cond_idx}",
                    "phase_id": phase_id,
                    "condition_idx": cond_idx,
                    "pulse_duration": pulse_dur,
                },
            )

            p_offset = cond_idx * fovs_per_condition + self._fov_index_offset
            for ev in acq:
                local_p = ev.index.get("p", 0)
                all_events.append(
                    ev.model_copy(
                        update={"index": {**dict(ev.index), "p": local_p + p_offset}}
                    )
                )

        all_events = utils.apply_fov_batching(all_events, time_per_fov=2.0)

        print(f"  Created {len(all_events)} events for phase {phase_id}")
        for i, p in enumerate(param_list):
            fov_start = i * fovs_per_condition + self._fov_index_offset
            fov_end = fov_start + fovs_per_condition - 1
            print(
                f"    Cond {i} (FOVs {fov_start}-{fov_end}): "
                f"pulse_duration={p['pulse_duration']:.0f}ms"
            )
        return all_events

    # ------------------------------------------------------------------
    # Result preprocessing -- AUC & response probability
    # ------------------------------------------------------------------

    def _preprocess_results(self, fov_tracks: dict) -> pd.DataFrame:
        """Compute per-cell AUC and response probability per FOV.

        For each cell that passes quality gates:

        1. ``baseline_cnr_cell`` = mean cnr over the first
           ``n_frames_baseline`` frames.
        2. ``delta_auc`` = trapezoid integral of
           ``(cnr - baseline_cnr_cell)`` over the observation window
           (frames ``stim_frame + 1`` to ``n_frames - 1``).
        3. A cell *responds* if ``delta_auc > response_threshold``.

        Per FOV the method returns ``mean_delta_auc`` (mean across cells)
        and ``frac_responding`` (fraction of responders).
        """
        phase_id = self._current_phase_id
        results = []

        for fov_idx, df_tracks in fov_tracks.items():
            if df_tracks.empty or "particle" not in df_tracks.columns:
                continue

            # --- Filter to current phase ---
            if "phase_id" in df_tracks.columns:
                df_phase = df_tracks[df_tracks["phase_id"] == phase_id]
                if df_phase.empty:
                    continue
            else:
                df_phase = df_tracks

            params = self._current_condition_map.get(fov_idx)
            if params is None:
                print(
                    f"  Warning: no condition mapping for FOV {fov_idx}, " f"skipping"
                )
                continue

            # Detect cnr column
            cnr_col = (
                "cnr"
                if "cnr" in df_phase.columns
                else "cnr_median" if "cnr_median" in df_phase.columns else None
            )
            if cnr_col is None:
                print(f"  Warning: no cnr column in FOV {fov_idx}, skipping")
                continue

            all_particles = df_phase["particle"].unique()
            n_cells_total = len(all_particles)

            # --- Quality gate: tracking length ---
            min_frames = int(self.min_track_fraction * self.n_frames)
            frames_per_cell = df_phase.groupby("particle")["fov_timestep"].nunique()

            # --- Per-cell baseline CNR ---
            baseline_df = df_phase[df_phase["fov_timestep"] < self.n_frames_baseline]
            per_cell_baseline = pd.Series(dtype=float)
            if not baseline_df.empty:
                per_cell_baseline = (
                    baseline_df.groupby("particle")[cnr_col].mean().dropna()
                )

            # FOV-level baseline_cnr covariate (all cells, before filter)
            baseline_cnr_fov = (
                float(per_cell_baseline.mean()) if len(per_cell_baseline) > 0 else 0.0
            )

            # optoRTK expression covariate
            optortk_vals = []
            if "ref_mean_intensity" in df_phase.columns:
                optortk_vals = (
                    df_phase.groupby("particle")["ref_mean_intensity"]
                    .first()
                    .dropna()
                    .values
                )

            # --- Build set of cells passing quality gates ---
            valid_particles = set(all_particles)
            valid_particles &= set(frames_per_cell[frames_per_cell >= min_frames].index)
            if self.max_baseline_cnr is not None and len(per_cell_baseline) > 0:
                valid_particles &= set(
                    per_cell_baseline[per_cell_baseline < self.max_baseline_cnr].index
                )

            n_cells = len(valid_particles)
            n_responding = 0
            cell_aucs = []

            # --- Per-cell AUC computation ---
            obs_start = self.stim_frame + 1
            obs_end = self.n_frames

            for particle, grp in df_phase.groupby("particle"):
                if particle not in valid_particles:
                    continue
                grp = grp.sort_values("fov_timestep")

                # Baseline mean for this cell
                bl = grp[grp["fov_timestep"] < self.n_frames_baseline]
                if bl.empty:
                    continue
                baseline_val = bl[cnr_col].mean()

                # Observation window
                obs = grp[
                    (grp["fov_timestep"] >= obs_start) & (grp["fov_timestep"] < obs_end)
                ]
                if len(obs) < 3:
                    continue

                # AUC via trapezoid rule on (cnr - baseline)
                t = obs["fov_timestep"].values.astype(float)
                y = obs[cnr_col].values.astype(float) - baseline_val
                delta_auc = float(np.trapz(y, t))
                cell_aucs.append(delta_auc)

                if delta_auc > self.response_threshold:
                    n_responding += 1

            if n_cells < 3 or len(cell_aucs) == 0:
                print(
                    f"  Warning: FOV {fov_idx} has only {n_cells} valid "
                    f"cells (of {n_cells_total} total), skipping"
                )
                continue

            mean_delta_auc = float(np.mean(cell_aucs))
            frac_responding = n_responding / n_cells
            optortk_expression = (
                float(np.mean(optortk_vals)) if len(optortk_vals) > 0 else 0.0
            )

            results.append(
                {
                    "pulse_duration": params["pulse_duration"],
                    "n_cells": float(n_cells),
                    "baseline_cnr": baseline_cnr_fov,
                    "optortk_expression": optortk_expression,
                    "mean_delta_auc": mean_delta_auc,
                    "frac_responding": frac_responding,
                }
            )

        if not results:
            print(f"Warning: no valid FOVs in phase {phase_id}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        print(
            f"  Phase {phase_id}: {len(df)} FOVs, "
            f"mean_delta_auc={df['mean_delta_auc'].mean():.4f}, "
            f"frac_responding={df['frac_responding'].mean():.4f}"
        )
        return df

    # ------------------------------------------------------------------
    # Structured GP with Hill curve prior
    # ------------------------------------------------------------------

    @staticmethod
    def _make_hill_prior(dose_col: int = 0):
        """Return a numpyro Hill function for use as StructuredGP mean_fn.

        The Hill equation ``V_max * d^n / (K^n + d^n)`` models the
        expected monotonically-increasing, saturating dose-response
        relationship.  All three parameters are learnable via MCMC.

        Because the GP operates on *scaled* inputs (approximately
        ``[-1, 1]``), the dose is shifted to ``[0, 2]`` before applying
        the Hill function so that the domain stays non-negative.

        Args:
            dose_col: Column index of the dose (pulse_duration) in the
                scaled input array.  Defaults to ``0`` (first parameter).
        """

        def hill_fn(x):
            import jax.numpy as jnp
            import numpyro
            import numpyro.distributions as dist

            vmax = numpyro.sample("hill_vmax", dist.HalfNormal(2.0))
            k_half = numpyro.sample("hill_k", dist.HalfNormal(1.0))
            n_hill = numpyro.sample("hill_n", dist.HalfNormal(2.0))

            # Shift scaled dose from ~[-1,1] to [0,2] so Hill is defined
            dose = jnp.clip(x[:, dose_col] + 1.0, 1e-6, None)
            return vmax * dose**n_hill / (k_half**n_hill + dose**n_hill)

        return hill_fn

    def _determine_next_parameters(
        self, df_results: pd.DataFrame, verbose=False
    ) -> dict:
        """Fit StructuredGP with Hill prior, compute acquisition, return next point.

        Overrides the base-class method to swap ``gpax.ExactGP`` for
        ``gpax.StructuredGP`` with a Hill curve mean function.  All
        acquisition logic (EI/UCB, covariate marginalisation, batch
        caching, penalty) is inherited unchanged.
        """
        import gpax
        import gpax.utils
        import jax.numpy as jnp

        cache = getattr(self, "_cached_gp_fit", None)
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

            # --- KEY DIFFERENCE: StructuredGP with Hill prior ---
            gp_model = gpax.StructuredGP(
                input_dim=x.shape[1],
                kernel="Matern",
                mean_fn=self._make_hill_prior(dose_col=0),
            )
            gp_model.fit(
                rng_key,
                X=x,
                y=y,
                progress_bar=True,
                num_warmup=500,
                num_samples=1000,
            )

            self.model = gp_model
            self._x_scaler = x_scaler
            self._y_scaler = y_scaler
            self._rng_key_predict = rng_key_predict

            if getattr(self, "_batch_fit_reuse", False):
                self._cached_gp_fit = dict(
                    gp_model=gp_model,
                    x_scaler=x_scaler,
                    y_scaler=y_scaler,
                    rng_key_predict=rng_key_predict,
                    x=x,
                    y=y,
                )

        # --- Acquisition (identical to base class) ---
        x_grid_ctrl = self.x_unmeasured.copy()
        acquisition_used = self.acquisition_function
        current_xi = self._current_ei_xi()

        if len(self.bo_covariates) > 0:
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

        # --- Select next point ---
        next_measurement_idx = jnp.argmax(acq_values_total)
        next_parameters = np.asarray(x_grid_ctrl[int(next_measurement_idx)])

        self.x_unmeasured = np.delete(
            self.x_unmeasured, int(next_measurement_idx), axis=0
        )
        self.x_performed_experiments = (
            np.concatenate(
                [
                    self.x_performed_experiments,
                    next_parameters.reshape(1, -1),
                ],
                axis=0,
            )
            if self.x_performed_experiments is not None
            else next_parameters.reshape(1, -1)
        )

        next_parameters_dict = {
            param.name: next_parameters[i]
            for i, param in enumerate(self.parameters_to_optimize)
        }

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
                cov_grid=(cov_grid if len(self.bo_covariates) > 0 else None),
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
    # Per-phase hook (live plot + checkpoint)
    # ------------------------------------------------------------------

    def _on_phase_complete(self, df_new: pd.DataFrame, phase_id: int) -> None:
        if self.save_checkpoints:
            self._save_checkpoint(self.df_results)
            self.save_model()
        if self.plot_live:
            self._plot_live(self.df_results, f"After phase {phase_id + 1}")

    def _save_checkpoint(self, df_results: pd.DataFrame) -> None:
        ckpt_path = os.path.join(self.storage_path, "bo_results_checkpoint.parquet")
        df_results.to_parquet(ckpt_path, index=False)
        ckpt_dir = os.path.join(self.storage_path, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        df_results.to_parquet(
            os.path.join(
                ckpt_dir,
                f"bo_results_phase_{self._current_phase_id:03d}.parquet",
            ),
            index=False,
        )

    def _plot_bo_landscape(self, *args, **kwargs) -> None:
        """Suppress the base-class 2x2 plot -- replaced by _plot_live."""
        pass

    def _plot_live(
        self,
        df_results: pd.DataFrame,
        iteration_label: str,
        save_subdir: str = "after",
    ) -> None:
        """1x3 plot: measured dose-response, GP landscape, acquisition.

        For a 1-D dose (pulse_duration only), the GP landscape and
        acquisition are plotted as curves.  For 2-D input (pulse_duration +
        another parameter) they fall back to 2-D heatmaps.
        """
        import matplotlib.pyplot as plt

        n_ctrl = len(self.parameters_to_optimize)
        obj_name = self.objective_metric.name

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        fig.suptitle(iteration_label, fontsize=13, fontweight="bold")

        # --- Subplot 1: measured dose-response per FOV ---
        axes[0].scatter(
            df_results["pulse_duration"],
            df_results[obj_name],
            c="steelblue",
            s=30,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.3,
        )
        axes[0].set_xlabel("pulse_duration (ms)")
        axes[0].set_ylabel(obj_name)
        axes[0].set_title(f"Measured {obj_name}")

        # Also plot frac_responding on twin axis if available
        if "frac_responding" in df_results.columns and obj_name != "frac_responding":
            ax_twin = axes[0].twinx()
            ax_twin.scatter(
                df_results["pulse_duration"],
                df_results["frac_responding"],
                c="coral",
                s=20,
                alpha=0.5,
                marker="^",
            )
            ax_twin.set_ylabel("frac_responding", color="coral")
            ax_twin.tick_params(axis="y", labelcolor="coral")

        # --- Subplots 2 & 3: GP landscape + acquisition ---
        ctx = getattr(self, "_last_plot_context", None)
        if ctx is None:
            for ax in axes[1:]:
                ax.text(
                    0.5,
                    0.5,
                    "GP not fit yet\n(initial batch)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                    color="gray",
                )
            axes[1].set_title("GP predicted landscape")
            axes[2].set_title("Acquisition")
        elif n_ctrl == 1:
            self._plot_1d_landscape(ctx, df_results, axes[1], axes[2], fig)
        else:
            try:
                self._plot_landscape_and_acq_from_context(
                    ctx, df_results, axes[1], axes[2], fig
                )
            except Exception as e:
                for ax in axes[1:]:
                    ax.text(
                        0.5,
                        0.5,
                        f"Plot failed:\n{type(e).__name__}: {e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="red",
                    )

        plt.tight_layout()
        plots_dir = os.path.join(self.storage_path, "plots", save_subdir)
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(
            os.path.join(plots_dir, f"phase_{self._current_phase_id:03d}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def _plot_1d_landscape(self, ctx, df_results, ax_mean, ax_acq, fig):
        """1-D dose-response curve + acquisition function."""
        import jax.numpy as jnp

        gp_model = ctx["gp_model"]
        x_scaler = ctx["x_scaler"]
        y_scaler = ctx["y_scaler"]
        rng_key_predict = ctx["rng_key_predict"]
        acq_values_total = ctx["acq_values_total"]
        acquisition_used = ctx["acquisition_used"]
        x_unmeasured = ctx["x_unmeasured_at_computation"]

        obj_name = self.objective_metric.name
        x_total_ctrl = self.x_total_linespace.copy()
        unique_doses = np.sort(np.unique(x_total_ctrl[:, 0]))

        # Predict on full dose grid (marginalising covariates)
        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_cols = [c.name for c in self.bo_covariates]
            cov_vals = df_results[cov_cols].to_numpy().astype(float)
            n_cov = min(50, len(cov_vals))
            _rng = np.random.default_rng(0)
            cov_samples = cov_vals[_rng.integers(0, len(cov_vals), size=n_cov)]
            x_grid_full = np.hstack(
                [
                    np.repeat(unique_doses[:, None], n_cov, axis=0),
                    np.tile(cov_samples, (len(unique_doses), 1)),
                ]
            )
        else:
            n_cov = 1
            x_grid_full = unique_doses[:, None]

        x_grid_scaled = x_scaler.transform(x_grid_full)
        y_pred_scaled, y_samples = gp_model.predict_in_batches(
            rng_key_predict,
            x_grid_scaled,
            batch_size=1000,
            n=self.ei_num_samples,
            noiseless=True,
        )
        y_pred = y_scaler.inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).flatten()

        if len(self.bo_covariates) > 0:
            y_pred_marg = y_pred.reshape(len(unique_doses), n_cov).mean(axis=1)
            # Compute std across covariate samples for uncertainty band
            y_std_marg = y_pred.reshape(len(unique_doses), n_cov).std(axis=1)
        else:
            y_pred_marg = y_pred
            # Use posterior variance for uncertainty
            y_var_scaled = np.asarray(jnp.var(y_samples, axis=(0, 1)))
            y_std_marg = np.sqrt(np.maximum(y_var_scaled, 0.0))
            y_std_marg = y_scaler.inverse_transform(y_std_marg.reshape(-1, 1)).flatten()
            # Rough uncertainty from std of scaled predictions
            y_std_marg = np.abs(
                y_std_marg
                - y_scaler.inverse_transform(np.zeros((len(y_std_marg), 1))).flatten()
            )

        # GP predicted curve
        ax_mean.plot(unique_doses, y_pred_marg, "b-", linewidth=1.5, label="GP mean")
        ax_mean.fill_between(
            unique_doses,
            y_pred_marg - 2 * y_std_marg,
            y_pred_marg + 2 * y_std_marg,
            alpha=0.2,
            color="blue",
            label="GP +/- 2 std",
        )
        ax_mean.scatter(
            df_results["pulse_duration"],
            df_results[obj_name],
            c="red",
            s=25,
            zorder=5,
            label="observations",
        )
        ax_mean.set_xlabel("pulse_duration (ms)")
        ax_mean.set_ylabel(obj_name)
        ax_mean.set_title("GP predicted dose-response (marginalised)")
        ax_mean.legend(fontsize=8)

        # Acquisition function
        acq_vals = np.asarray(acq_values_total)
        if len(acq_vals) == len(x_unmeasured):
            # Map back to full dose grid
            acq_full = np.zeros(len(unique_doses))
            for j, pt in enumerate(x_unmeasured):
                idx = np.argmin(np.abs(unique_doses - pt[0]))
                acq_full[idx] = max(acq_full[idx], float(acq_vals[j]))
            acq_vals = acq_full

        if len(acq_vals) == len(unique_doses):
            ax_acq.fill_between(unique_doses, 0, acq_vals, alpha=0.4, color="orange")
            ax_acq.plot(unique_doses, acq_vals, "k-", linewidth=1)
        else:
            ax_acq.bar(range(len(acq_vals)), acq_vals, alpha=0.6, color="orange")
        ax_acq.set_xlabel("pulse_duration (ms)")
        ax_acq.set_ylabel(acquisition_used.upper())
        ax_acq.set_title(f"Acquisition ({acquisition_used.upper()})")

    def _plot_landscape_and_acq_from_context(
        self, ctx, df_results, ax_mean, ax_acq, fig
    ):
        """2-D heatmap plot (when >1 controllable parameter)."""
        import jax.numpy as jnp

        gp_model = ctx["gp_model"]
        x_scaler = ctx["x_scaler"]
        y_scaler = ctx["y_scaler"]
        rng_key_predict = ctx["rng_key_predict"]
        acq_values_total = ctx["acq_values_total"]
        acquisition_used = ctx["acquisition_used"]
        x_unmeasured = ctx["x_unmeasured_at_computation"]

        obj_name = self.objective_metric.name
        x_total_ctrl = self.x_total_linespace.copy()
        unique_x1 = np.unique(x_total_ctrl[:, 0])
        unique_x2 = np.unique(x_total_ctrl[:, 1])
        n_ctrl = len(unique_x1) * len(unique_x2)

        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_cols = [c.name for c in self.bo_covariates]
            cov_vals_full = df_results[cov_cols].to_numpy().astype(float)
            n_cov_samples = 50
            _plot_rng = np.random.default_rng(0)
            row_idx = _plot_rng.integers(0, cov_vals_full.shape[0], size=n_cov_samples)
            cov_samples_joint = cov_vals_full[row_idx]
        else:
            n_cov_samples = 1
            cov_samples_joint = None

        ctrl_grid = np.array([[x1, x2] for x1 in unique_x1 for x2 in unique_x2])
        if cov_samples_joint is not None:
            x_grid_full = np.hstack(
                [
                    np.repeat(ctrl_grid, n_cov_samples, axis=0),
                    np.tile(cov_samples_joint, (n_ctrl, 1)),
                ]
            )
        else:
            x_grid_full = ctrl_grid

        x_grid_scaled = x_scaler.transform(x_grid_full)
        y_pred_scaled, _ = gp_model.predict_in_batches(
            rng_key_predict,
            x_grid_scaled,
            batch_size=1000,
            n=self.ei_num_samples,
            noiseless=True,
        )
        y_pred = y_scaler.inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).flatten()
        if len(self.bo_covariates) > 0:
            y_pred_marg = y_pred.reshape(n_ctrl, n_cov_samples).mean(axis=1)
        else:
            y_pred_marg = y_pred

        X_mesh, Y_mesh = np.meshgrid(unique_x1, unique_x2, indexing="ij")
        y_pred_2d = y_pred_marg.reshape(len(unique_x1), len(unique_x2))

        p1_name = self.parameters_to_optimize[0].name
        p2_name = self.parameters_to_optimize[1].name

        im1 = ax_mean.pcolormesh(
            X_mesh, Y_mesh, y_pred_2d, cmap="viridis", shading="auto"
        )
        fig.colorbar(im1, ax=ax_mean, label=f"predicted {obj_name}")
        ax_mean.scatter(
            df_results[p1_name],
            df_results[p2_name],
            c="white",
            s=15,
            alpha=0.6,
            marker="x",
            linewidths=0.8,
        )
        ax_mean.set_xlabel(p1_name)
        ax_mean.set_ylabel(p2_name)
        ax_mean.set_title("GP predicted landscape (marginalised)")

        acq_marg = np.asarray(acq_values_total)
        if len(acq_marg) != n_ctrl:
            acq_full = np.zeros(n_ctrl)
            x_full = self.x_total_linespace
            for j, pt in enumerate(x_unmeasured):
                diffs = np.abs(x_full - pt).sum(axis=1)
                idx = np.argmin(diffs)
                if j < len(acq_marg):
                    acq_full[idx] = float(acq_marg[j])
            acq_marg = acq_full
        acq_2d = acq_marg.reshape(len(unique_x1), len(unique_x2))
        im2 = ax_acq.pcolormesh(X_mesh, Y_mesh, acq_2d, cmap="inferno", shading="auto")
        fig.colorbar(im2, ax=ax_acq, label=acquisition_used.upper())
        ax_acq.set_xlabel(p1_name)
        ax_acq.set_ylabel(p2_name)
        ax_acq.set_title(f"Acquisition ({acquisition_used.upper()})")
