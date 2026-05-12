"""Two-task dose-response BO with canonical gpax.hypo eps-greedy hypothesis learning.

Two tasks fit independently, vote jointly:

* ``auc``  -> ``mean_delta_auc``           (unbounded continuous, primary)
* ``frac`` -> ``frac_peak_responders``     (in [0, 1], logit-transformed, secondary)

Each task has its own structured GP (with a parametric mean function chosen
from a candidate set: Hill / linear / exponential) and its own hypothesis
selection state. Every phase, one candidate per task is picked; during a
brief warmup ALL candidates are fit and the winner gets +1 (canonical
gpax_hypo idiom), then eps-greedy on the running mean reward takes over.
No model is ever permanently locked in -- the algorithm rotates away from
a candidate whose variance landscape stalls picks (e.g., Hill saturating
at the upper boundary).

Acquisition is ``z(var_auc) + frac_acq_weight * z(var_frac)`` over the
unmeasured dose grid, with inverse-distance penalty against all
previously-measured doses (cross-phase repulsion).

Implements the gpax.hypo pattern from arXiv:2112.06649. See cell `hypo-md`
in `experiments/31_bo_optimisation/bo_erk_dose_response_v1.ipynb` for
the design rationale.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from numpyro.infer.util import log_likelihood as _numpyro_log_likelihood

import gpax
import gpax.hypo
import gpax.utils

# Enable float64 at module import time. gpax 0.1.9 has a Cholesky-instability
# bug in float32 that causes ~99% of GP predict() outputs to be NaN for
# certain test-X arrangements (covariate-marginalised dose grids especially).
# Float64 fully eliminates the bug. Must run BEFORE any gpax.ExactGP.fit().
gpax.utils.enable_x64()

from faro.agents.bo_dose_response import DoseResponseBO
from faro.agents.bo_optimization import StandardScalerBounds


# ---------------------------------------------------------------------------
# Candidate parametric mean functions
# ---------------------------------------------------------------------------

DOSE_COL = 0  # pulse_duration is the only controllable parameter


def _shifted_dose(x):
    """Map scaled dose from ~[-1, 1] to [0, 2] (clipped) so monotonic
    parametric forms are well-defined on the GP's working scale."""
    return jnp.clip(x[:, DOSE_COL] + 1.0, 1e-6, None)


def hill_mean_fn(x, p):
    """Hill: V_max * d^n / (K^n + d^n)  -- saturating sigmoidal."""
    d = _shifted_dose(x)
    return (
        p["hill_vmax"]
        * d ** p["hill_n"]
        / (p["hill_k"] ** p["hill_n"] + d ** p["hill_n"])
    )


def hill_mean_fn_prior():
    return {
        "hill_vmax": numpyro.sample("hill_vmax", dist.HalfNormal(2.0)),
        "hill_k": numpyro.sample("hill_k", dist.Uniform(0.05, 3.0)),
        "hill_n": numpyro.sample("hill_n", dist.Uniform(0.5, 4.0)),
    }


def linear_mean_fn(x, p):
    """Linear: a * d + b -- baseline / sanity check."""
    d = _shifted_dose(x)
    return p["lin_a"] * d + p["lin_b"]


def linear_mean_fn_prior():
    return {
        "lin_a": numpyro.sample("lin_a", dist.Normal(0.0, 2.0)),
        "lin_b": numpyro.sample("lin_b", dist.Normal(0.0, 1.0)),
    }


def exponential_mean_fn(x, p):
    """Exponential saturation: A * (1 - exp(-d / tau)) -- softer alt to Hill."""
    d = _shifted_dose(x)
    return p["exp_amp"] * (1.0 - jnp.exp(-d / p["exp_tau"]))


def exponential_mean_fn_prior():
    return {
        "exp_amp": numpyro.sample("exp_amp", dist.HalfNormal(2.0)),
        "exp_tau": numpyro.sample("exp_tau", dist.Uniform(0.1, 3.0)),
    }


# Default candidate set used by both tasks if no override is provided.
# Keys are the candidate names; values are (mean_fn, mean_fn_prior) tuples.
DEFAULT_CANDIDATE_HYPOTHESES: dict[str, tuple[Callable, Callable]] = {
    "hill": (hill_mean_fn, hill_mean_fn_prior),
    "linear": (linear_mean_fn, linear_mean_fn_prior),
    "exponential": (exponential_mean_fn, exponential_mean_fn_prior),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _logit(p, eps=1e-4):
    """Logit transform with clipping to keep +/- inf out of the GP."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _inv_logit(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def _extract_optortk_vals(df_phase):
    """Per-cell ref_mean_intensity, NaN-dropped and zero-filtered."""
    if "ref_mean_intensity" not in df_phase.columns:
        return np.array([])
    s = df_phase.groupby("particle")["ref_mean_intensity"].first().dropna()
    return s[s > 0].values


def _log_marginal_likelihood(gp_model, x, y, max_samples=200):
    """Diagnostic log-marginal-likelihood. Not used for selection any more."""
    samples = gp_model.mcmc.get_samples()
    n_total = next(iter(samples.values())).shape[0]
    n_use = min(int(max_samples), int(n_total))
    sub = {k: v[:n_use] for k, v in samples.items()}
    ll = _numpyro_log_likelihood(gp_model.model, sub, x, y)["y"]
    return float(jax.scipy.special.logsumexp(ll) - jnp.log(ll.shape[0]))


# Maximum rows per gpax.predict() call. gpax 0.1.9 has a bug where predict()
# produces NaN samples for X_new shape (>~50, *): 50 -> 0% NaN, 100 -> 0.5%,
# 1000 -> 100%. Keep batches at 50 to stay in the safe regime.
_GPAX_PREDICT_SAFE_BATCH = 50


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class TwoTaskDoseResponseBO(DoseResponseBO):
    """Two-task dose-response BO with gpax.hypo eps-greedy model selection.

    Args:
        candidate_hypotheses_auc: ``{name: (mean_fn, mean_fn_prior)}`` for AUC.
            Defaults to ``DEFAULT_CANDIDATE_HYPOTHESES`` (hill / linear / exp).
        candidate_hypotheses_frac: same for ``frac_peak_responders``.
        eps_greedy_eps: probability of exploring a random candidate after
            warmup. Default 0.3.
        n_warmup_phases: number of phases during which **all** candidates are
            fit and the lowest-uncertainty winner gets +1 reward. Match the
            canonical ``gpax_hypo.ipynb`` warmup. Default 2.
        frac_acq_weight: weight of the frac task in the joint acquisition
            ``z(var_auc) + frac_acq_weight * z(var_frac)``. Lower values
            make AUC dominate. Default 0.3.
        lengthscale_prior_dist: bounded ARD lengthscale prior, e.g.
            ``dist.Uniform(0.1, 1.0)`` (canonical gpax_hypo idiom). ``None``
            uses gpax default ``LogNormal(0, 1)``.
        peak_ratio: a cell counts as a peak-responder iff its post-stim
            ``cnr`` exceeds ``peak_ratio * baseline_cnr`` for at least
            ``peak_min_consecutive_frames`` consecutive observation frames.
            Default 1.5 (50% above baseline). Multiplicative -- replaces v1's
            additive ``peak_threshold`` which was misleadingly named.
        peak_min_consecutive_frames: min consecutive frames above ratio for
            a cell to count as a responder. Default 2.
        gp_kernel: passed to ``gpax.ExactGP``.
        gp_num_warmup / gp_num_samples: HMC budget per fit (per task per
            candidate, so warmup phases cost ``n_candidates *
            n_tasks * num_samples`` MCMC samples).
        rng_seed: base RNG seed. Per-phase keys are derived as
            ``rng_seed + phase_id`` so HMC traces are reproducible
            phase-by-phase.
    """

    TASKS = ("auc", "frac")

    def __init__(
        self,
        *,
        candidate_hypotheses_auc: Optional[dict] = None,
        candidate_hypotheses_frac: Optional[dict] = None,
        eps_greedy_eps: float = 0.3,
        n_warmup_phases: int = 2,
        frac_acq_weight: float = 0.3,
        lengthscale_prior_dist: Optional[dist.Distribution] = None,
        peak_ratio: float = 1.5,
        peak_min_consecutive_frames: int = 2,
        gp_kernel: str = "Matern",
        gp_num_warmup: int = 400,
        gp_num_samples: int = 800,
        rng_seed: int = 42,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if candidate_hypotheses_auc is None:
            candidate_hypotheses_auc = dict(DEFAULT_CANDIDATE_HYPOTHESES)
        if candidate_hypotheses_frac is None:
            candidate_hypotheses_frac = dict(DEFAULT_CANDIDATE_HYPOTHESES)
        if not candidate_hypotheses_auc or not candidate_hypotheses_frac:
            raise ValueError("candidate_hypotheses_{auc,frac} must be non-empty")

        self.peak_ratio = float(peak_ratio)
        self.peak_min_consecutive_frames = int(peak_min_consecutive_frames)
        self.gp_kernel = gp_kernel
        self.gp_num_warmup = int(gp_num_warmup)
        self.gp_num_samples = int(gp_num_samples)
        self.eps_greedy_eps = float(eps_greedy_eps)
        self.n_warmup_phases = int(n_warmup_phases)
        self.frac_acq_weight = float(frac_acq_weight)
        self.lengthscale_prior_dist = lengthscale_prior_dist
        self.rng_seed = int(rng_seed)

        self.task_columns = {
            "auc": "mean_delta_auc",
            "frac": "frac_peak_responders",
        }
        self.task_logit = {"auc": False, "frac": True}
        self.candidate_hypotheses_by_task = {
            "auc": dict(candidate_hypotheses_auc),
            "frac": dict(candidate_hypotheses_frac),
        }
        # Per-task hypothesis-learning state.
        # ``record[i] = (visit_count, mean_reward)`` -- gpax.hypo convention.
        # During warmup ``record[:, 1]`` accumulates raw +1 win counts; after
        # warmup it's normalised to win frequency, then updated by
        # ``gpax.hypo.update_record`` (running mean of +/-1 rewards).
        self._task_state: dict[str, dict] = {
            t: {
                "record": np.zeros(
                    (len(self.candidate_hypotheses_by_task[t]), 2), dtype=float
                ),
                "obj_history": [],
                "hypothesis_log": [],  # list of (phase_id, name, log_evidence)
                "n_warmup_phases_done": 0,
                "selected_hypothesis": None,
            }
            for t in self.TASKS
        }
        self._gp_models_by_task: dict[str, Any] = {}
        self._y_scalers_by_task: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Per-FOV result extraction with peak-based responder counting
    # ------------------------------------------------------------------

    def _is_peak_responder(
        self, baseline_val: float, post_cnr_values: np.ndarray
    ) -> bool:
        """Return True iff ``post_cnr / baseline_val > peak_ratio`` for at
        least ``peak_min_consecutive_frames`` consecutive frames.

        Multiplicative threshold -- replaces v1's additive cutoff.
        """
        if baseline_val <= 0 or not np.isfinite(baseline_val):
            return False
        above = (post_cnr_values / baseline_val) > self.peak_ratio
        if self.peak_min_consecutive_frames <= 1:
            return bool(np.any(above))
        consec = 0
        for v in above:
            if v:
                consec += 1
                if consec >= self.peak_min_consecutive_frames:
                    return True
            else:
                consec = 0
        return False

    def _preprocess_results(self, fov_tracks):
        """Compute per-FOV ``mean_delta_auc`` and ``frac_peak_responders``."""
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

            obs_start = self.stim_frame + 1
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

                if self._is_peak_responder(baseline_val, obs_cnr):
                    n_responding += 1

            if n_cells < 3 or len(cell_aucs) == 0:
                print(
                    f"  Warning: FOV {fov_idx} has only {n_cells} valid "
                    f"cells (of {n_cells_total} total), skipping"
                )
                continue

            mean_delta_auc = float(np.mean(cell_aucs))
            frac_peak_responders = n_responding / n_cells
            optortk_expression = (
                float(np.mean(optortk_vals)) if len(optortk_vals) > 0 else 0.0
            )

            # optoRTK race-condition retry: ref-frame parquet write may not
            # be flushed when the agent first reads. Re-read with backoff.
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
                    "mean_delta_auc": mean_delta_auc,
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
    # Hypothesis fitting + selection (canonical gpax.hypo eps-greedy)
    # ------------------------------------------------------------------

    def _fit_candidate(self, mean_fn, mean_fn_prior, x, y, rng_key):
        """Fit one candidate's structured GP.

        Lengthscale prior taken from ``self.lengthscale_prior_dist`` if set;
        otherwise gpax default ``LogNormal(0, 1)``.
        """
        kwargs = dict(
            input_dim=x.shape[1],
            kernel=self.gp_kernel,
            mean_fn=mean_fn,
            mean_fn_prior=mean_fn_prior,
        )
        if self.lengthscale_prior_dist is not None:
            kwargs["lengthscale_prior_dist"] = self.lengthscale_prior_dist
        gp_model = gpax.ExactGP(**kwargs)
        gp_model.fit(
            rng_key,
            X=x,
            y=y,
            progress_bar=False,
            num_warmup=self.gp_num_warmup,
            num_samples=self.gp_num_samples,
            print_summary=False,
            jitter=1e-4,
        )
        log_evidence = _log_marginal_likelihood(gp_model, x, y)
        return gp_model, log_evidence

    def _median_predictive_variance(self, gp_model, x, rng_key):
        """Reward signal: median predictive variance on the training inputs.

        Same convention as canonical ``get_reward(obj_history)`` in
        gpax_hypo.ipynb. Uses plain ``predict()`` because gpax 0.1.9's
        ``split_in_batches`` raises ``UnboundLocalError`` when
        ``batch_size > X.shape[0]`` (training set is typically << 100).
        Training X is small enough (<= ~200 rows) to stay below the
        gpax predict() NaN threshold.
        """
        _, samples = gp_model.predict(
            rng_key,
            jnp.asarray(x),
            n=4,
            noiseless=True,
        )
        arr = jnp.asarray(samples).reshape(-1, samples.shape[-1])
        med = float(jnp.nanmedian(arr.var(0)))
        return med

    def _select_or_fit_for_task(self, task, x, y, rng_key):
        """Pick + fit one candidate per task, update reward record.

        Two regimes (canonical gpax_hypo idiom):

        * **Warmup** (``n_warmup_phases_done < n_warmup_phases``):
          Fit ALL candidates on the same data. The lowest-median-uncertainty
          winner gets +1 (raw count) on its reward slot. The winner's GP is
          returned for the joint acquisition. After the last warmup phase,
          ``record[:, 1]`` is normalised by ``n_warmup_phases`` so it
          becomes win frequency in [0, 1] -- the initial mean reward for
          eps-greedy.

        * **Post-warmup**: ``gpax.hypo.sample_next(method="eps-greedy")``
          picks one candidate. Its reward is +1 if median predictive
          variance dropped vs the previous phase, -1 otherwise.

        No model is ever permanently locked in.
        """
        state = self._task_state[task]
        candidates = list(self.candidate_hypotheses_by_task[task].items())
        n_models = len(candidates)
        n_warmup_done = state["n_warmup_phases_done"]

        if n_warmup_done < self.n_warmup_phases:
            return self._warmup_phase(task, state, candidates, x, y, rng_key)
        return self._eps_greedy_phase(task, state, candidates, x, y, rng_key)

    def _warmup_phase(self, task, state, candidates, x, y, rng_key):
        n_models = len(candidates)
        print(
            f"    [{task}: warmup phase {state['n_warmup_phases_done'] + 1}"
            f"/{self.n_warmup_phases} -- fitting all {n_models} candidates]"
        )

        fits = {}
        obj_per_model = {}
        for i, (name, (mfn, mfp)) in enumerate(candidates):
            try:
                gp_model, log_evidence = self._fit_candidate(mfn, mfp, x, y, rng_key)
                obj = self._median_predictive_variance(gp_model, x, rng_key)
            except Exception as exc:
                print(f"      {name}: FIT FAILED ({type(exc).__name__}: {exc})")
                continue
            fits[name] = (gp_model, log_evidence)
            obj_per_model[name] = obj
            state["record"][i, 0] += 1
            print(f"      {name}: obj={obj:.4f}, log_evidence={log_evidence:+.3f}")
            state["hypothesis_log"].append(
                (self._current_phase_id, name, float(log_evidence))
            )

        valid = {k: v for k, v in obj_per_model.items() if np.isfinite(v)}
        if valid:
            winner_name = min(valid, key=valid.get)
            winner_idx = next(
                i for i, (n, _) in enumerate(candidates) if n == winner_name
            )
            state["record"][winner_idx, 1] += 1.0  # raw +1 (canonical convention)
            state["obj_history"].append(obj_per_model[winner_name])
            print(
                f"    [{task}: warmup winner={winner_name!r}, obj={obj_per_model[winner_name]:.4f}]"
            )
        elif fits:
            # All NaN obj but at least one fit succeeded -- fall back to first.
            winner_name = next(iter(fits))
            state["obj_history"].append(np.nan)
            print(
                f"    [{task}: ALL candidates produced NaN obj; falling back to {winner_name!r}]"
            )
        else:
            raise RuntimeError(f"No candidate could be fit for task {task!r}")

        state["n_warmup_phases_done"] += 1
        if state["n_warmup_phases_done"] == self.n_warmup_phases:
            # Normalise raw win counts to win frequency (in [0, 1]).
            state["record"][:, 1] = state["record"][:, 1] / self.n_warmup_phases
            freqs = ", ".join(
                f"{candidates[i][0]}={state['record'][i, 1]:.2f}"
                for i in range(len(candidates))
            )
            print(f"    [{task}: WARMUP COMPLETE -- win frequencies: {freqs}]")

        state["selected_hypothesis"] = winner_name
        return fits[winner_name][0]

    def _eps_greedy_phase(self, task, state, candidates, x, y, rng_key):
        idx = int(
            gpax.hypo.sample_next(
                state["record"][:, 1],
                method="eps-greedy",
                eps=self.eps_greedy_eps,
            )
        )
        rewards_str = ", ".join(
            f"{candidates[i][0]}={state['record'][i, 1]:+.2f}(n={int(state['record'][i, 0])})"
            for i in range(len(candidates))
        )
        print(f"    [{task}: eps-greedy -> {candidates[idx][0]!r}  ({rewards_str})]")

        name, (mfn, mfp) = candidates[idx]
        gp_model, log_evidence = self._fit_candidate(mfn, mfp, x, y, rng_key)

        median_var = self._median_predictive_variance(gp_model, x, rng_key)
        state["obj_history"].append(median_var)

        prev_finite = next(
            (v for v in reversed(state["obj_history"][:-1]) if np.isfinite(v)),
            None,
        )
        if prev_finite is not None and np.isfinite(median_var):
            r = 1.0 if median_var < prev_finite else -1.0
            state["record"] = gpax.hypo.update_record(state["record"], idx, r)
            print(
                f"    [{task}: obj={median_var:.4f} "
                f"(prev_finite={prev_finite:.4f}) -> reward={r:+.0f}]"
            )
        else:
            state["record"][idx, 0] += 1
            note = (
                "first finite obj"
                if prev_finite is None
                else "NaN obj; skipping reward"
            )
            print(f"    [{task}: obj={median_var}  ({note})]")

        state["hypothesis_log"].append(
            (self._current_phase_id, name, float(log_evidence))
        )
        # Update favourite (current eps-greedy argmax) for visualisation.
        state["selected_hypothesis"] = candidates[int(state["record"][:, 1].argmax())][
            0
        ]
        return gp_model

    def _build_y_for_task(self, task, df_results, y_scaler):
        col = self.task_columns[task]
        y_raw = np.asarray(df_results[col].values, dtype=float).reshape(-1, 1)
        if self.task_logit[task]:
            y_raw = _logit(y_raw)
        return y_scaler.fit_transform(y_raw, bounds=None, log_scale=[False])

    # ------------------------------------------------------------------
    # Acquisition: AUC-primary joint pure-exploration
    # ------------------------------------------------------------------

    def _compute_marginal_variance(
        self, rng_key, gp_model, x_grid, c_samples, x_scaler
    ):
        n_grid = x_grid.shape[0]
        if c_samples is not None:
            n_mc = c_samples.shape[0] if c_samples.ndim > 1 else len(c_samples)
            if c_samples.ndim == 1:
                c_samples = c_samples.reshape(-1, 1)
            x_repeated = jnp.repeat(x_grid, n_mc, axis=0)
            c_tiled = jnp.tile(c_samples, (n_grid, 1))
            x_full = jnp.hstack([x_repeated, c_tiled])
        else:
            n_mc = 1
            x_full = jnp.asarray(x_grid)
        x_full_scaled = x_scaler.transform(x_full)

        # batch_size=50 to stay below gpax 0.1.9's predict() NaN threshold.
        # Default batch_size=100 produces ~0.5% NaN; batch_size=1000 produces 100%.
        mean_pred, y_sampled = gp_model.predict_in_batches(
            rng_key,
            x_full_scaled,
            batch_size=_GPAX_PREDICT_SAFE_BATCH,
            n=self.ei_num_samples,
            noiseless=self.ei_noiseless,
        )
        var_full = jnp.asarray(y_sampled).reshape(-1, x_full_scaled.shape[0]).var(0)
        var_per_grid = var_full.reshape(n_grid, n_mc).mean(axis=1)
        # Defensive NaN guard: replace any remaining NaN with 0 so they
        # contribute no acquisition weight rather than poisoning argmax.
        n_nan = int(jnp.isnan(var_per_grid).sum())
        if n_nan > 0:
            print(
                f"    Warning: {n_nan}/{n_grid} grid points had NaN posterior "
                f"variance; replacing with 0"
            )
            var_per_grid = jnp.where(jnp.isnan(var_per_grid), 0.0, var_per_grid)
        return var_per_grid

    def _phase_rng_keys(self):
        """Per-phase reproducible RNG keys derived from ``self.rng_seed``.

        Returns ``(fit_key, predict_key)``. Same call within a single phase
        returns the same keys, so the reward signal is reproducible.
        """
        seed = self.rng_seed + int(self._current_phase_id)
        return gpax.utils.get_keys(seed)

    def _determine_next_parameters(self, df_results, verbose=False):
        cache = getattr(self, "_cached_gp_fit", None)
        is_first_call_in_batch = cache is None

        if cache is not None:
            x_scaler = cache["x_scaler"]
            x = cache["x"]
            rng_key_predict = cache["rng_key_predict"]
            gp_models_by_task = cache["gp_models_by_task"]
            y_scalers_by_task = cache["y_scalers_by_task"]
            print("  [reusing cached two-task GP fits from earlier in this batch]")
        else:
            x_scaler = StandardScalerBounds()
            bounds, log_scale = self._get_bounds_and_log_scale()
            x_raw = self._extract_x_from_df(df_results)
            x = x_scaler.fit_transform(x_raw, bounds=bounds, log_scale=log_scale)

            rng_key, rng_key_predict = self._phase_rng_keys()

            gp_models_by_task = {}
            y_scalers_by_task = {}
            for task in self.TASKS:
                print(
                    f"  Fitting task = {task!r} "
                    f"(column = {self.task_columns[task]!r}, "
                    f"logit={self.task_logit[task]})"
                )
                y_scaler = StandardScalerBounds()
                y = self._build_y_for_task(task, df_results, y_scaler)
                gp_models_by_task[task] = self._select_or_fit_for_task(
                    task, x, y, rng_key
                )
                y_scalers_by_task[task] = y_scaler

            self._gp_models_by_task = gp_models_by_task
            self._y_scalers_by_task = y_scalers_by_task
            # Framework-compat shims: viz/save_model expect a single
            # `self.model` -- expose the AUC GP.
            self.model = gp_models_by_task["auc"]
            self._x_scaler = x_scaler
            self._y_scaler = y_scalers_by_task["auc"]
            self._rng_key_predict = rng_key_predict

            if getattr(self, "_batch_fit_reuse", False):
                self._cached_gp_fit = dict(
                    gp_models_by_task=gp_models_by_task,
                    y_scalers_by_task=y_scalers_by_task,
                    x_scaler=x_scaler,
                    rng_key_predict=rng_key_predict,
                    x=x,
                )

        # ----- joint AUC-primary acquisition -----
        x_grid_ctrl = self.x_unmeasured.copy()
        cov_grid = (
            self._sample_covariate_grid(df_results)
            if len(self.bo_covariates) > 0
            else None
        )
        per_task_var_z = {}
        per_task_var_raw = {}
        for task in self.TASKS:
            v = self._compute_marginal_variance(
                rng_key_predict,
                gp_models_by_task[task],
                x_grid_ctrl,
                cov_grid,
                x_scaler,
            )
            per_task_var_raw[task] = v
            v_std = jnp.std(v)
            v_std = jnp.where(v_std < 1e-12, 1.0, v_std)
            per_task_var_z[task] = (v - jnp.mean(v)) / v_std

        # AUC-primary weighting: z(var_auc) + frac_acq_weight * z(var_frac)
        acq_values_total = (
            per_task_var_z["auc"] + self.frac_acq_weight * per_task_var_z["frac"]
        )

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

            x_grid_scaled = _scale_ctrl(x_grid_ctrl)
            recent_ctrl_scaled = _scale_ctrl(jnp.asarray(recent_ctrl_raw))
            acq_values_total = self._apply_penalty(
                acq_values_total,
                x_grid_scaled,
                np.asarray(recent_ctrl_scaled),
                self.penalty,
                self.penalty_factor,
            )

        self._acquisition_used_this_round = "explore_two_task_auc_primary"
        print(
            f"  AUC-primary acq stats (frac_w={self.frac_acq_weight}): "
            f"min={float(jnp.min(acq_values_total)):.4f}, "
            f"max={float(jnp.max(acq_values_total)):.4f}"
        )

        next_measurement_idx = jnp.argmax(acq_values_total)
        next_parameters = np.asarray(x_grid_ctrl[int(next_measurement_idx)])
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

        if is_first_call_in_batch:
            self._last_plot_context = dict(
                df_results=df_results,
                x_scaler=x_scaler,
                rng_key_predict=rng_key_predict,
                gp_models_by_task=gp_models_by_task,
                y_scalers_by_task=y_scalers_by_task,
                x_unmeasured_at_computation=x_grid_ctrl.copy(),
                acq_values_total=np.asarray(acq_values_total),
                per_task_var_z={k: np.asarray(v) for k, v in per_task_var_z.items()},
                per_task_var_raw={
                    k: np.asarray(v) for k, v in per_task_var_raw.items()
                },
                # Back-compat keys for the base class plot helper.
                gp_model=gp_models_by_task["auc"],
                y_scaler=y_scalers_by_task["auc"],
                acquisition_used=self._acquisition_used_this_round,
                current_xi=0.0,
                y=np.zeros((1, 1)),
                cov_grid=cov_grid,
            )
        return next_parameters_dict

    # ------------------------------------------------------------------
    # Plot override: two-task 1-D dose-response landscape
    # ------------------------------------------------------------------

    def _plot_1d_landscape(self, ctx, df_results, ax_mean, ax_acq, fig):
        gp_models = ctx.get("gp_models_by_task")
        y_scalers = ctx.get("y_scalers_by_task")
        if gp_models is None or y_scalers is None:
            return DoseResponseBO._plot_1d_landscape(
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
        for task, ax_y, color, label in [
            ("auc", ax_mean, "C0", "mean_delta_auc"),
            ("frac", ax2, "C2", "frac_peak_responders"),
        ]:
            gp = gp_models[task]
            y_scaler = y_scalers[task]
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
            obs_x = df_results["pulse_duration"].values
            obs_y = df_results[self.task_columns[task]].values
            ax_y.scatter(obs_x, obs_y, color=color, s=16, alpha=0.5)
            ax_y.set_ylabel(label, color=color)
            ax_y.tick_params(axis="y", labelcolor=color)

        ax_mean.set_xlabel("pulse_duration (ms)")
        ax_acq.plot(x_unmeasured[:, 0], acq_values_total, color="C3", lw=1.0)
        ax_acq.set_xlabel("pulse_duration (ms)")
        ax_acq.set_ylabel(f"acq (z(auc) + {self.frac_acq_weight}*z(frac))")

    # ------------------------------------------------------------------
    # Checkpoint extension: also persist the per-task hypothesis state
    # ------------------------------------------------------------------

    def save_model(self, path: str | None = None) -> str | None:
        """Save the AUC GP plus the per-task hypothesis-learning state.

        Extends ``BOptGPAX.save_model`` so a crashed run can recover not
        just the dataframe but also each task's reward record + obj_history,
        which would otherwise need to be rebuilt by replaying every phase.
        """
        path = super().save_model(path)
        if path is None:
            return None
        try:
            import joblib

            payload = joblib.load(path)

            # Serialise _task_state. record / obj_history are numpy / scalars;
            # hypothesis_log is a list of plain tuples. selected_hypothesis is
            # a string. n_warmup_phases_done is an int. All picklable.
            payload["task_state"] = {
                t: {
                    "record": np.asarray(s["record"]).copy(),
                    "obj_history": list(s["obj_history"]),
                    "hypothesis_log": list(s["hypothesis_log"]),
                    "n_warmup_phases_done": int(s["n_warmup_phases_done"]),
                    "selected_hypothesis": s["selected_hypothesis"],
                }
                for t, s in self._task_state.items()
            }
            payload["task_columns"] = dict(self.task_columns)
            payload["task_logit"] = dict(self.task_logit)
            payload["candidate_names_by_task"] = {
                t: list(d) for t, d in self.candidate_hypotheses_by_task.items()
            }
            payload["frac_acq_weight"] = self.frac_acq_weight
            payload["peak_ratio"] = self.peak_ratio
            payload["eps_greedy_eps"] = self.eps_greedy_eps
            payload["n_warmup_phases"] = self.n_warmup_phases
            payload["rng_seed"] = self.rng_seed

            joblib.dump(payload, path)
            return path
        except Exception as exc:  # pragma: no cover -- best-effort
            print(f"  Warning: could not extend checkpoint with task_state: {exc}")
            return path
