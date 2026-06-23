"""v4 dose-response BO: peak-amplitude primary readout, honest model selection.

Subclass of :class:`TwoTaskDoseResponseBO` that implements the v4 design
agreed after the v3 post-mortem
(``bo_erk_dose_response_v3_pitch_and_critique.ipynb``) and the readout
replay (``bo_erk_dose_response_v3_peak_amplitude_replay.ipynb``).

What changed vs v3, and why
---------------------------
* **Primary readout is peak amplitude, not AUC.**  The primary task fits
  ``mean_peak_amp`` = mean over surviving cells of ``max(cnr - baseline)``
  over the observation window.  Peak reports the instantaneous gain of the
  ultrasensitive RAS->RAF->MEK->ERK step *before* slow negative feedback
  (DUSP, ERK->SOS) engages, so it has a real ceiling (ERK pool / receptor
  occupancy) and a dose-response that *can* saturate -- unlike AUC, which
  convolves amplitude with a non-saturating duration/adaptation integral
  and climbs ~log(dose) forever (the v3 failure).  ``mean_delta_auc`` is
  still computed and stored every phase as a logged diagnostic column, but
  is no longer the fitted objective.

  Implementation note: to inherit the parent's two-task acquisition,
  plotting and checkpoint machinery unchanged, the **internal task key
  ``"auc"`` is repurposed to carry the peak-amplitude readout**
  (``self.task_columns["auc"] == "mean_peak_amp"``).  Renaming the key to
  ``"peak"`` would force re-deriving the parent's
  ``_determine_next_parameters`` (which references ``"auc"`` literally),
  so the key name is kept as an implementation detail.  Everything the
  user sees (objective metric, plot labels, the logged column) says
  *peak amplitude*.

* **Honest model selection restored, with a non-saturating escape and an
  in-range Hill.**  v3's "Hill only" deleted the one safety valve that
  could flag non-saturation.  v4 ships:

  - ``power_law`` (``A * d**b``) -- a non-saturating family the eps-greedy
    selector can *declare a winner*, i.e. the agent reports
    "non-saturating, exponent b" instead of fabricating an EC50;
  - an **in-range Hill** whose half-max ``K`` is sampled from a
    ``Beta(2, 2)`` scaled onto the shifted-dose range ``(0, 2)``.  Beta(2,2)
    has zero density at the boundaries, so the sampler cannot park ``K`` at
    the upper dose boundary (the v3 ``K``-runs-to-5000ms pathology).  If
    the data does not saturate, this Hill fits badly and eps-greedy rotates
    away from it.
  - ``exponential`` (soft saturation, no cooperativity) is kept.

  Default peak candidates: ``{hill (in-range), exponential, power_law}``.
  Default frac candidates: ``{hill (in-range), exponential}`` -- a
  recruitment fraction must saturate at 1, so no non-saturating family is
  offered there (and ``frac``'s half-recruitment dose is the one EC50 that
  is always identifiable -- used as the convergence companion below).

* **Cell-count floor on the baseline filter.**  v3 only rejected an FOV
  when ``< 3`` cells survived ``max_baseline_cnr``, producing 3-9 cell FOV
  means that are pure noise.  v4 rejects an FOV unless
  ``>= min_valid_cells`` (default 20) survive.

* **frac down-weighted in the joint acquisition** (``frac_acq_weight``
  default 0.15, was 0.3) so the ceilinged recruitment fraction stops
  pulling the MaxVar sampler.

* **optoRTK expression handled by stratification, not just
  marginalisation.**  ``optortk_expression`` stays a GP input axis, but
  acquisition optionally marginalises over a *uniform* low->high expression
  grid (``uniform_expression_grid=True``) so the chosen doses reduce
  uncertainty evenly across expression strata.  Reporting (notebook) slices
  the same surface at expression percentiles.  ``baseline_cnr`` is kept as
  a covariate (its surviving-cell FOV mean still carries residual signal;
  interpret its effect predictively, not causally).

* **Convergence keyed to a quantity that can converge.**  After each
  phase v4 records the predicted peak dose-response curve and the
  ``frac`` half-recruitment dose, and exposes :meth:`convergence_report`
  / :attr:`is_converged` (peak curve stable to < ``stop_rms_tol`` for
  ``stop_patience`` consecutive phases, with a stable selected family).
  It does *not* auto-halt the run (``ComposedAgent.run`` runs a fixed
  phase count); use it to decide whether to start the next power block.

Power blocking
--------------
v4 keeps a single controllable axis (``pulse_duration``).  Stimulation
*power* is non-linear, so rather than putting it on the GP axis it is a
**blocking factor**: run one v4 agent per power level (5 / 10 / 25 / 100)
and compare the per-power peak dose-response curves.  The notebook drives
the blocking loop; this class is agnostic to it.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

import numpy as np
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions.transforms import AffineTransform

from faro.agents.two_task_dose_response import (
    TwoTaskDoseResponseBO,
    hill_mean_fn,
    exponential_mean_fn,
    exponential_mean_fn_prior,
    _shifted_dose,
    _inv_logit,
    _extract_optortk_vals,
    _GPAX_PREDICT_SAFE_BATCH,
)


# ---------------------------------------------------------------------------
# New / reparametrised candidate mean functions
# ---------------------------------------------------------------------------


def hill_inrange_mean_fn_prior():
    """Hill prior with the half-max ``K`` forced into the observed dose range.

    Same functional form as the v3 Hill (reuse :func:`hill_mean_fn`); only
    the ``K`` prior differs.  ``K`` is drawn from ``Beta(2, 2)`` affine-mapped
    onto the shifted-dose interval ``(0, 2)`` (which spans the full measured
    dose range).  Beta(2, 2) has zero density at 0 and 2, so the sampler
    cannot park ``K`` at the dose boundary -- the v3 EC50-unidentifiability
    trap.  ``V_max`` is widened to ``HalfNormal(3.0)`` so the asymptote is
    not forced below the data.
    """
    return {
        "hill_vmax": numpyro.sample("hill_vmax", dist.HalfNormal(3.0)),
        "hill_k": numpyro.sample(
            "hill_k",
            dist.TransformedDistribution(
                dist.Beta(2.0, 2.0), AffineTransform(0.0, 2.0)
            ),
        ),
        "hill_n": numpyro.sample("hill_n", dist.Uniform(0.5, 4.0)),
    }


def power_law_mean_fn(x, p):
    """Power law: ``a * d**b`` -- the non-saturating escape family.

    Its single shape parameter ``b`` reads off directly: ``b < 1``
    decelerating-but-non-saturating, ``b ~ 1`` linear, ``b > 1``
    accelerating.  It is monotone-increasing from the low-dose end and
    approaches the origin as ``d -> 0`` (exactly 0 in the limit; for small
    ``b`` the shifted-dose floor of ``1e-6`` leaves a small non-zero
    intercept, the same floor the parent Hill/exp families use, which the
    GP kernel residual absorbs).  Parameter names (``pow_a`` / ``pow_b``)
    match the PARAM_SPECS convention used by the analysis notebooks.
    """
    d = _shifted_dose(x)
    return p["pow_a"] * d ** p["pow_b"]


def power_law_mean_fn_prior():
    return {
        "pow_a": numpyro.sample("pow_a", dist.HalfNormal(2.0)),
        "pow_b": numpyro.sample("pow_b", dist.Uniform(0.1, 2.0)),
    }


# Default v4 candidate sets.
#   peak (primary): cooperative-saturating (Hill, in-range), hyperbolic-
#       saturating (exponential), non-saturating (power_law).
#   frac (secondary): saturating only -- a recruited fraction must plateau.
V4_CANDIDATES_PEAK: dict[str, tuple[Callable, Callable]] = {
    "hill": (hill_mean_fn, hill_inrange_mean_fn_prior),
    "exponential": (exponential_mean_fn, exponential_mean_fn_prior),
    "power_law": (power_law_mean_fn, power_law_mean_fn_prior),
}

V4_CANDIDATES_FRAC: dict[str, tuple[Callable, Callable]] = {
    "hill": (hill_mean_fn, hill_inrange_mean_fn_prior),
    "exponential": (exponential_mean_fn, exponential_mean_fn_prior),
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TwoTaskDoseResponseV4BO(TwoTaskDoseResponseBO):
    """v4 dose-response BO: peak-amplitude primary readout + honest model selection.

    See module docstring for the full rationale.  New args (everything else
    is inherited from :class:`TwoTaskDoseResponseBO`):

    Args:
        min_valid_cells: reject an FOV unless at least this many cells
            survive the per-cell filters (track length, ``max_baseline_cnr``).
            Default 20 (v3 used an effective floor of 3).
        uniform_expression_grid: if True (default), acquisition marginalises
            over a uniform low->high grid of ``expression_covariate`` (its
            observed 5-95 percentile range) instead of the empirical sample,
            so the chosen doses reduce uncertainty evenly across expression.
        expression_covariate: name of the optoRTK-expression covariate.
            Default ``"optortk_expression"``.
        stop_rms_tol: peak-curve relative-RMS-change threshold for the
            convergence diagnostic.  Default 0.05 (5%).
        stop_patience: number of consecutive phases that must satisfy the
            stability threshold (with a stable selected family) before
            :attr:`is_converged` flips True.  Default 2.
        candidate_hypotheses_auc / candidate_hypotheses_frac: override the
            default v4 candidate sets if given.
        frac_acq_weight: default lowered to 0.15.
    """

    def __init__(
        self,
        *,
        min_valid_cells: int = 20,
        uniform_expression_grid: bool = True,
        expression_covariate: str = "optortk_expression",
        stop_rms_tol: float = 0.05,
        stop_patience: int = 2,
        candidate_hypotheses_auc: Optional[dict] = None,
        candidate_hypotheses_frac: Optional[dict] = None,
        frac_acq_weight: float = 0.15,
        **kwargs,
    ):
        if candidate_hypotheses_auc is None:
            candidate_hypotheses_auc = dict(V4_CANDIDATES_PEAK)
        if candidate_hypotheses_frac is None:
            candidate_hypotheses_frac = dict(V4_CANDIDATES_FRAC)

        super().__init__(
            candidate_hypotheses_auc=candidate_hypotheses_auc,
            candidate_hypotheses_frac=candidate_hypotheses_frac,
            frac_acq_weight=frac_acq_weight,
            **kwargs,
        )

        # Repurpose the primary task key "auc" to carry the peak readout.
        self.task_columns["auc"] = "mean_peak_amp"

        self.min_valid_cells = int(min_valid_cells)
        self.uniform_expression_grid = bool(uniform_expression_grid)
        self.expression_covariate = str(expression_covariate)

        self.stop_rms_tol = float(stop_rms_tol)
        self.stop_patience = int(stop_patience)

        # Convergence history (filled in _on_phase_complete).
        self._peak_curve_history: list[np.ndarray] = []
        self._peak_rms_history: list[float] = []
        self._frac_halfdose_history: list[float] = []
        self._selected_family_history: list[str] = []
        self.is_converged: bool = False

    # ------------------------------------------------------------------
    # Per-FOV extraction: add mean_peak_amp + cell-count floor
    # ------------------------------------------------------------------

    def _preprocess_results(self, fov_tracks):
        """As :meth:`TwoTaskDoseResponseBO._preprocess_results`, but also emit
        ``mean_peak_amp`` and enforce a ``min_valid_cells`` floor.

        ``mean_peak_amp`` = mean over surviving cells of ``max(cnr - baseline)``
        over the observation window, computed from the *same* surviving cells
        as ``mean_delta_auc`` and ``frac_peak_responders``.
        """
        import pandas as pd

        phase_id = self._current_phase_id
        results = []
        n_optortk_retried = 0
        n_optortk_recovered = 0

        for fov_idx, df_tracks in fov_tracks.items():
            if df_tracks.empty or "particle" not in df_tracks.columns:
                continue
            if "phase_id" in df_tracks.columns:
                df_phase = df_tracks[df_tracks["phase_id"] == phase_id]
                if df_phase.empty:
                    continue
            else:
                df_phase = df_tracks

            params = self._current_condition_map.get(fov_idx)
            if params is None:
                print(f"  Warning: no condition mapping for FOV {fov_idx}, skipping")
                continue

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

            min_frames = int(self.min_track_fraction * self.n_frames)
            frames_per_cell = df_phase.groupby("particle")["fov_timestep"].nunique()

            baseline_df = df_phase[df_phase["fov_timestep"] < self.n_frames_baseline]
            per_cell_baseline = pd.Series(dtype=float)
            if not baseline_df.empty:
                per_cell_baseline = (
                    baseline_df.groupby("particle")[cnr_col].mean().dropna()
                )
            baseline_cnr_fov = (
                float(per_cell_baseline.mean()) if len(per_cell_baseline) > 0 else 0.0
            )

            optortk_vals = _extract_optortk_vals(df_phase)

            valid_particles = set(all_particles)
            valid_particles &= set(frames_per_cell[frames_per_cell >= min_frames].index)
            if self.max_baseline_cnr is not None and len(per_cell_baseline) > 0:
                valid_particles &= set(
                    per_cell_baseline[per_cell_baseline < self.max_baseline_cnr].index
                )

            n_cells = len(valid_particles)
            n_responding = 0
            cell_aucs = []
            cell_peaks = []

            obs_start = self.stim_frame + 1
            # Analysis window for peak amplitude + responder calls. Defaults to
            # the full observation tail; `response_window_frames` restricts it to
            # the early transient. The ERK-KTR response peaks ~7-9 min post-stim,
            # so the long decay tail mainly adds noise-driven responder calls
            # (low-dose false positives ~doubled by the full 25-frame window).
            if getattr(self, "response_window_frames", None):
                obs_end = min(
                    obs_start + int(self.response_window_frames), self.n_frames
                )
            else:
                obs_end = self.n_frames

            for particle, grp in df_phase.groupby("particle"):
                if particle not in valid_particles:
                    continue
                grp = grp.sort_values("fov_timestep")
                bl = grp[grp["fov_timestep"] < self.n_frames_baseline]
                if bl.empty:
                    continue
                baseline_val = float(bl[cnr_col].mean())
                obs = grp[
                    (grp["fov_timestep"] >= obs_start) & (grp["fov_timestep"] < obs_end)
                ]
                if len(obs) < 3:
                    continue

                t_vals = obs["fov_timestep"].values.astype(float)
                obs_cnr = obs[cnr_col].values.astype(float)
                delta = obs_cnr - baseline_val

                cell_aucs.append(float(np.trapezoid(delta, t_vals)))
                cell_peaks.append(float(np.max(delta)))

                if self._is_peak_responder(baseline_val, obs_cnr):
                    n_responding += 1

            # v4 cell-count floor: reject thin FOVs whose means are noise.
            if n_cells < self.min_valid_cells or len(cell_aucs) == 0:
                print(
                    f"  Warning: FOV {fov_idx} has only {n_cells} valid cells "
                    f"(of {n_cells_total} total; floor={self.min_valid_cells}), "
                    f"skipping"
                )
                continue

            mean_delta_auc = float(np.mean(cell_aucs))
            mean_peak_amp = float(np.mean(cell_peaks))
            frac_peak_responders = n_responding / n_cells
            optortk_expression = (
                float(np.mean(optortk_vals)) if len(optortk_vals) > 0 else 0.0
            )

            if not np.isfinite(optortk_expression) or optortk_expression <= 0.0:
                n_optortk_retried += 1
                for _delay in (0.5, 1.5, 3.0):
                    time.sleep(_delay)
                    try:
                        df_retry = self.read_tracks(fov_idx, phase_id=phase_id)
                    except Exception:
                        df_retry = pd.DataFrame()
                    if "phase_id" in df_retry.columns:
                        df_retry = df_retry[df_retry["phase_id"] == phase_id]
                    optortk_vals_retry = _extract_optortk_vals(df_retry)
                    if len(optortk_vals_retry) > 0:
                        optortk_expression = float(np.mean(optortk_vals_retry))
                        n_optortk_recovered += 1
                        break

            results.append(
                {
                    **params,
                    "n_cells": float(n_cells),
                    "baseline_cnr": baseline_cnr_fov,
                    "optortk_expression": optortk_expression,
                    "mean_peak_amp": mean_peak_amp,
                    "mean_delta_auc": mean_delta_auc,  # logged diagnostic, not fitted
                    "frac_peak_responders": frac_peak_responders,
                }
            )

        if n_optortk_retried > 0:
            print(
                f"  optoRTK retry summary: "
                f"{n_optortk_recovered}/{n_optortk_retried} recovered after re-read"
            )

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Acquisition: marginalise over a UNIFORM expression grid
    # ------------------------------------------------------------------

    def _sample_covariate_grid(self, df_results):
        """As parent, but optionally sweep ``expression_covariate`` uniformly.

        The base class jointly resamples covariates from observed FOVs.  When
        ``uniform_expression_grid`` is on, we keep that joint sample but
        overwrite the expression column with a uniform sweep across its
        observed 5-95 percentile range, so MaxVar reduces uncertainty evenly
        across low/med/high expression instead of only the densely-sampled
        middle.  ``baseline_cnr`` (and any other covariate) stays empirically
        resampled.
        """
        grid = super()._sample_covariate_grid(df_results)
        if grid is None or not self.uniform_expression_grid:
            return grid

        names = [c.name for c in self.bo_covariates]
        if self.expression_covariate not in names:
            return grid
        if df_results is None or len(df_results) < 2:
            return grid

        j = names.index(self.expression_covariate)
        vals = np.asarray(df_results[self.expression_covariate].values, dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0)]
        if vals.size < 2:
            return grid

        lo, hi = np.percentile(vals, [5, 95])
        if not (hi > lo):
            return grid

        grid = np.array(grid, dtype=float, copy=True)
        grid[:, j] = np.linspace(lo, hi, grid.shape[0])
        return grid

    # ------------------------------------------------------------------
    # Live plot: correct labels (primary curve is peak amplitude)
    # ------------------------------------------------------------------

    def _plot_1d_landscape(self, ctx, df_results, ax_mean, ax_acq, fig):
        """Two-task 1-D landscape with correct (peak-amplitude) labels.

        Same structure as the parent two-task plot, but reads the y-axis
        label from ``self.task_columns`` so the primary axis says
        ``mean_peak_amp`` rather than the inherited literal ``mean_delta_auc``.
        """
        gp_models = ctx.get("gp_models_by_task")
        y_scalers = ctx.get("y_scalers_by_task")
        if gp_models is None or y_scalers is None:
            return TwoTaskDoseResponseBO._plot_1d_landscape(
                self, ctx, df_results, ax_mean, ax_acq, fig
            )

        x_scaler = ctx["x_scaler"]
        rng_key_predict = ctx["rng_key_predict"]
        acq_values_total = ctx["acq_values_total"]
        x_unmeasured = ctx["x_unmeasured_at_computation"]

        x_total_ctrl = self.x_total_linespace.copy()
        unique_doses = np.sort(np.unique(x_total_ctrl[:, 0]))

        if len(self.bo_covariates) > 0:
            cov_grid = self._sample_covariate_grid(df_results)
            cov_med = np.median(np.asarray(cov_grid), axis=0).reshape(1, -1)
            x_pred = np.column_stack(
                [unique_doses.reshape(-1, 1), np.tile(cov_med, (len(unique_doses), 1))]
            )
        else:
            x_pred = unique_doses.reshape(-1, 1)
        x_pred_scaled = x_scaler.transform(x_pred)

        ax2 = ax_mean.twinx()
        for task, ax_y, color in [
            ("auc", ax_mean, "C0"),  # key "auc" carries mean_peak_amp in v4
            ("frac", ax2, "C2"),
        ]:
            gp = gp_models[task]
            y_scaler = y_scalers[task]
            label = self.task_columns[task]
            y_pred_scaled, _ = gp.predict_in_batches(
                rng_key_predict,
                x_pred_scaled,
                batch_size=_GPAX_PREDICT_SAFE_BATCH,
                n=4,
                noiseless=True,
            )
            y_pred_orig = y_scaler.inverse_transform(
                np.asarray(y_pred_scaled).reshape(-1, 1)
            ).flatten()
            if self.task_logit[task]:
                y_pred_orig = _inv_logit(y_pred_orig)
            ax_y.plot(unique_doses, y_pred_orig, color=color, lw=1.6, label=label)
            ax_y.scatter(
                df_results["pulse_duration"].values,
                df_results[label].values,
                color=color,
                s=16,
                alpha=0.5,
            )
            ax_y.set_ylabel(label, color=color)
            ax_y.tick_params(axis="y", labelcolor=color)

        ax_mean.set_xlabel("pulse_duration (ms)")
        ax_mean.set_title("peak-amplitude (primary) + frac dose-response")
        ax_acq.plot(x_unmeasured[:, 0], acq_values_total, color="C3", lw=1.0)
        ax_acq.set_xlabel("pulse_duration (ms)")
        ax_acq.set_ylabel(f"acq (z(peak) + {self.frac_acq_weight}*z(frac))")

    # ------------------------------------------------------------------
    # Convergence diagnostic (peak-curve stability + frac half-recruitment)
    # ------------------------------------------------------------------

    def _predict_task_curve(self, task, doses, apply_inv_logit=False):
        """Posterior-mean curve for ``task`` over ``doses`` (original units),
        covariates fixed at their median.  Returns ``None`` if the task GP is
        not fit yet."""
        gp = self._gp_models_by_task.get(task)
        y_scaler = self._y_scalers_by_task.get(task)
        if gp is None or y_scaler is None or self.df_results is None:
            return None
        doses = np.asarray(doses, dtype=float)
        if len(self.bo_covariates) > 0:
            cov_med = np.median(
                self.df_results[[c.name for c in self.bo_covariates]].to_numpy(float),
                axis=0,
            ).reshape(1, -1)
            x_raw = np.column_stack(
                [doses.reshape(-1, 1), np.tile(cov_med, (len(doses), 1))]
            )
        else:
            x_raw = doses.reshape(-1, 1)
        x_scaled = self._x_scaler.transform(x_raw)
        mean_s, _ = gp.predict_in_batches(
            self._rng_key_predict,
            jnp.asarray(x_scaled),
            batch_size=_GPAX_PREDICT_SAFE_BATCH,
            n=4,
            noiseless=True,
        )
        curve = y_scaler.inverse_transform(np.asarray(mean_s).reshape(-1, 1)).ravel()
        if apply_inv_logit:
            curve = _inv_logit(curve)
        return curve

    def _on_phase_complete(self, df_new, phase_id):
        super()._on_phase_complete(df_new, phase_id)
        try:
            self._update_convergence()
        except Exception as exc:  # never let a diagnostic break the run
            print(f"  [v4 convergence diagnostic skipped: {type(exc).__name__}: {exc}]")

    def _update_convergence(self):
        """Record the peak curve + frac half-recruitment dose, update
        :attr:`is_converged`.

        Note: the GP read here was fit at the start of this phase (to choose
        this phase's doses), i.e. on data through the *previous* phase -- the
        just-measured FOVs are not yet in it.  The convergence trace is thus
        one phase stale relative to the phase label; since it is informational
        only (the run never halts early), this is a labelling nuance, not a
        correctness issue.
        """
        if "auc" not in self._gp_models_by_task:
            return
        doses = np.sort(np.unique(self.x_total_linespace[:, 0]))

        peak_curve = self._predict_task_curve("auc", doses)
        if peak_curve is None or not np.all(np.isfinite(peak_curve)):
            return
        self._peak_curve_history.append(peak_curve)

        # Peak-curve relative RMS change vs the previous phase.
        rms = np.nan
        if len(self._peak_curve_history) >= 2:
            prev = self._peak_curve_history[-2]
            amp = np.ptp(peak_curve)  # max - min
            if amp > 0:
                rms = float(np.sqrt(np.mean((peak_curve - prev) ** 2)) / amp)
        self._peak_rms_history.append(rms)

        # frac half-recruitment dose (first dose where predicted frac >= 0.5).
        half_dose = np.nan
        frac_curve = self._predict_task_curve("frac", doses, apply_inv_logit=True)
        if frac_curve is not None and np.any(frac_curve >= 0.5):
            half_dose = float(doses[np.argmax(frac_curve >= 0.5)])
        self._frac_halfdose_history.append(half_dose)

        fam = self._task_state["auc"].get("selected_hypothesis")
        self._selected_family_history.append(fam)

        # Converged iff the last `stop_patience` phases all had small RMS
        # change AND the selected peak family was stable across them.
        recent_rms = self._peak_rms_history[-self.stop_patience :]
        recent_fam = self._selected_family_history[-self.stop_patience :]
        stable_curve = len(recent_rms) >= self.stop_patience and all(
            np.isfinite(r) and r < self.stop_rms_tol for r in recent_rms
        )
        stable_family = len(set(recent_fam)) == 1 and recent_fam[0] is not None
        self.is_converged = bool(stable_curve and stable_family)

        rms_str = "n/a" if not np.isfinite(rms) else f"{rms:.3f}"
        hd_str = "n/a" if not np.isfinite(half_dose) else f"{half_dose:.0f} ms"
        print(
            f"  [v4 convergence] peak-curve RMS change={rms_str} "
            f"(tol {self.stop_rms_tol}), family={fam!r}, "
            f"frac half-recruitment dose={hd_str}, converged={self.is_converged}"
        )

    def convergence_report(self) -> dict:
        """Return the convergence history as a dict (for notebook display)."""
        return {
            "peak_rms_history": list(self._peak_rms_history),
            "frac_halfdose_history": list(self._frac_halfdose_history),
            "selected_family_history": list(self._selected_family_history),
            "is_converged": self.is_converged,
            "stop_rms_tol": self.stop_rms_tol,
            "stop_patience": self.stop_patience,
        }
