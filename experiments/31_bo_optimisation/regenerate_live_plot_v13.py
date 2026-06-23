"""Regenerate the v13 BO "live plot" (3 columns) from a saved run.

The live plot inside ``bo_erk_oscillation_testv13_auc_fixed_budget.ipynb``
(``OscillationBO._plot_live`` -> ``_plot_landscape_and_acq_from_context``)
relies on an in-memory ``_last_plot_context`` that only exists while the
experiment is running.  This script rebuilds the same three panels from the
artifacts on disk:

    column 1  -- scatter of measured FOVs (ramp_fraction vs pulse_interval),
                 coloured by the BO objective (auc_norm)
    column 2  -- GP-predicted objective landscape, marginalised over the
                 observed covariate distribution
    column 3  -- acquisition surface (Expected Improvement), marginalised
                 over covariates

It does NOT reimplement any GP/acquisition math.  It reconstructs the *real*
``OscillationBO`` agent, re-attaches the trained ``gpax.ExactGP`` that
``BOptGPAX.save_model`` pickled (as posterior ``samples`` + scaled
``X_train``/``y_train``), and calls the agent's own methods:

    * ``agent.model.predict_in_batches``  -> GP landscape (column 2)
    * ``agent._sample_covariate_grid``     -> covariate marginalisation samples
    * ``agent._current_ei_xi``             -> EI exploration offset
    * ``agent._compute_robust_acq``        -> acquisition (column 3)

so the panels match exactly what the live BO computed (the log shows it stayed
on EI throughout, with ``use_closed_form_predict=True``).  Only the plotting is
done here, with the v13-correct ``ramp_fraction`` / ``pulse_interval`` axes
(the inherited ``_plot_landscape_and_acq_from_context`` mislabels them as
``stim_exposure`` / ``ramp``, which is why the live panels looked wrong).

Two gpax/numpyro quirks are handled the same way the notebook does:
  * the haiku-shim (numpyro >= 0.20 removed ``random_haiku_module``, which
    gpax 0.1.9 imports eagerly via viDKL) -- installed before ``import gpax``;
  * float64 is enabled (``gpax.utils.enable_x64`` -- gpax 0.1.9 is NaN-prone
    in float32; ``BOptGPAX.__init__`` does this too).

Output is one figure per BO phase (initial-spread phases are scatter-only;
GP-fitted phases get all three panels), written to
``{run}/plots/regenerated/phase_NNN.png``.

Usage
-----
    # all phases (default)
    ./.venv/Scripts/python.exe experiments/31_bo_optimisation/regenerate_live_plot_v13.py [RUN_DIR]
    # a single phase's model
    ./.venv/Scripts/python.exe .../regenerate_live_plot_v13.py [RUN_DIR] --model .../bo_model_iter_004.joblib
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import types

# --- gpax/numpyro haiku shim (must run before `import gpax`) ----------------
import numpyro.contrib.module as _ncm


def _haiku_unavailable(*_args, **_kwargs):
    raise NotImplementedError(
        "haiku support was removed from numpyro >= 0.20; viDKL is unavailable."
    )


for _name in ("random_haiku_module", "haiku_module"):
    if not hasattr(_ncm, _name):
        setattr(_ncm, _name, _haiku_unavailable)
# ---------------------------------------------------------------------------

import joblib
import numpy as np
import pandas as pd
import jax.numpy as jnp
import matplotlib.pyplot as plt

import gpax
import gpax.utils

gpax.utils.enable_x64()  # gpax 0.1.9 float32 Cholesky -> NaN; match the agent.

from faro.agents.bo_oscillation import OscillationBO
from faro.agents.bo_optimization import BO_Parameter, BO_Covariate, BO_Objective
from faro.agents.bo_optimization_sparse import _safe_batch_size

DEFAULT_RUN_DIR = r"E:\Alex\2026-05-20_bo_erk_auc_v13_fixed_budget_4s"

# Acquisition hyper-params not stored by save_model -- taken from the notebook
# (cell 14).  They only affect the EI xi schedule + MC sample count; with
# use_closed_form_predict=True the acquisition mean/var are exact.
EI_XI = 0.1
EI_XI_FINAL = 0.01
EI_XI_DECAY_FRACTION = 0.7
EI_NUM_SAMPLES = 4


def find_latest_model(run_dir):
    cands = sorted(glob.glob(os.path.join(run_dir, "models", "bo_model_iter_*.joblib")))
    if not cands:
        raise FileNotFoundError(f"No models/bo_model_iter_*.joblib in {run_dir}")
    return cands[-1]


def rebuild_model(model_state):
    """Re-attach the trained gpax.ExactGP from the pickled posterior samples.

    save_model can't pickle the live gpax object (JAX-traced kernel), so it
    stores the MCMC ``samples`` dict + scaled training arrays.  ``get_samples``
    reads ``self.mcmc.get_samples(...)``; we wire the saved samples through a
    stub so the agent's closed-form predict path works unchanged.
    """
    samples = {k: jnp.asarray(v) for k, v in model_state["samples"].items()}
    kernel_name = (
        "Matern" if "Matern" in str(model_state.get("kernel", "Matern")) else "RBF"
    )
    gp = gpax.ExactGP(
        input_dim=int(model_state["X_train"].shape[1]), kernel=kernel_name
    )
    gp.X_train = jnp.asarray(model_state["X_train"])
    gp.y_train = jnp.asarray(model_state["y_train"])
    gp.mcmc = types.SimpleNamespace(
        get_samples=lambda group_by_chain=False, _s=samples: _s
    )
    return gp


def infer_param(name, bounds, log_scale, df):
    """Rebuild a BO_Parameter (spacing/type aren't saved -> infer from data).

    Rounds to kill float drift (measured values like 0.6000000000000001
    otherwise yield a sub-0.1 min-diff and an extra grid column).
    """
    vals = np.unique(np.round(df[name].to_numpy(dtype=float), 6))
    step = None
    if len(vals) >= 2:
        d = np.diff(vals)
        d = d[d > 1e-9]
        if len(d):
            step = round(float(np.min(d)), 6)
    if step is None or step <= 0:
        step = round((bounds[1] - bounds[0]) / 20.0, 6)
    integerish = (
        np.allclose(vals, np.round(vals))
        and abs(step - round(step)) < 1e-9
        and round(step) >= 1
    )
    ptype = "int" if integerish else "float"
    if integerish:
        step = float(round(step))
    return BO_Parameter(
        name=name,
        bounds=tuple(bounds),
        param_type=ptype,
        spacing=step,
        log_scale=bool(log_scale),
    )


def build_agent(p, df, run_dir):
    """Reconstruct the real OscillationBO with the run's BO configuration."""
    names = p["parameter_names"]
    params = [
        infer_param(n, b, lg, df)
        for n, b, lg in zip(names, p["parameter_bounds"], p["parameter_log_scale"])
    ]
    # covariate log-scale isn't saved separately; recover it from the scaler
    # (covariates are the trailing entries after the control params).
    xlog = (
        list(p["x_scaler"].log_scale) if p["x_scaler"].log_scale is not None else None
    )
    n_ctrl = len(names)
    covs = []
    for i, (cn, cb) in enumerate(zip(p["covariate_names"], p["covariate_bounds"])):
        lg = bool(xlog[n_ctrl + i]) if xlog is not None else False
        covs.append(
            BO_Covariate(name=cn, bounds=(tuple(cb) if cb else None), log_scale=lg)
        )

    agent = OscillationBO(
        storage_path=run_dir,
        parameters_to_optimize=params,
        objective_metric=BO_Objective(
            name=p["objective_name"], goal=p["objective_goal"]
        ),
        bo_covariates=covs,
        n_iterations=int(p["n_iterations"]),
        acquisition_function=p["acquisition_function"],
        n_cov_samples=int(p["n_cov_samples"]),
        cov_marginalization_mode=p["cov_marginalization_mode"],
        use_closed_form_predict=True,
        ei_xi=EI_XI,
        ei_xi_final=EI_XI_FINAL,
        ei_xi_decay_fraction=EI_XI_DECAY_FRACTION,
        ei_num_samples=EI_NUM_SAMPLES,
        n_conditions_per_iter=int(p["n_conditions_per_iter"]),
        n_initial_phases=int(p["n_initial_phases"]),
        # OscillationBO-required args -- unused by the GP/acquisition/plot path.
        n_frames=90,
        first_frame_stim=10,
        last_frame_stim=70,
        time_between_timesteps=60,
        imaging_channels=None,
        stim_channel=None,
        optocheck_channel=None,
        osc_clf=None,
        osc_scaler=None,
        osc_feature_cols=None,
        osc_cfg=None,
        osc_predict_fn=None,
        plot_live=False,
        save_checkpoints=False,
    )
    # Inject the saved fit + state the acquisition methods read.
    agent.iteration = int(p["iteration"])
    agent.model = rebuild_model(p["model_state"])
    agent._x_scaler = p["x_scaler"]
    agent._y_scaler = p["y_scaler"]
    agent.df_results = df
    agent.x_performed_experiments = None  # penalty is None -> unused
    agent._cached_predictions = None  # force a fresh predict
    agent._cached_predictions_active_mask = None
    return agent


def gp_landscape(agent, ctrl_grid, n_cov, rng):
    """GP-predicted objective over ctrl_grid, marginalised over covariates.

    Same recipe as the agent's _plot_landscape_and_acq_from_context: real
    ``agent.model.predict_in_batches`` over (ctrl x cov), marginalise the mean,
    inverse the y-scaler.
    """
    cov_cols = [c.name for c in agent.bo_covariates]
    cov_full = agent.df_results[cov_cols].to_numpy(dtype=float)
    cov_samples = cov_full[rng.integers(0, cov_full.shape[0], n_cov)]
    n_grid = ctrl_grid.shape[0]
    x_full = np.hstack(
        [
            np.repeat(ctrl_grid, n_cov, axis=0),
            np.tile(cov_samples, (n_grid, 1)),
        ]
    )
    x_scaled = agent._x_scaler.transform(jnp.asarray(x_full))
    _, rng_key = gpax.utils.get_keys()
    bs = _safe_batch_size(int(np.asarray(x_scaled).shape[0]), 1000)
    y_pred_scaled, _ = agent.model.predict_in_batches(
        rng_key,
        x_scaled,
        batch_size=bs,
        n=agent.ei_num_samples,
        noiseless=True,
    )
    y_pred = np.asarray(
        agent._y_scaler.inverse_transform(
            jnp.asarray(np.asarray(y_pred_scaled).reshape(-1, 1))
        )
    ).ravel()
    return y_pred.reshape(n_grid, n_cov).mean(axis=1)


XLIM = (-0.03, 1.03)  # ramp_fraction axis range, shared across panels
YLIM = (
    0.5,
    20.5,
)  # pulse_interval axis range (small pad so pi=1/20 aren't on the edge)


def compute_panels(agent, n_cov, rng):
    """Run the real GP landscape + acquisition for one fitted agent."""
    obj_name = agent.objective_metric.name
    maximize = agent.objective_metric.goal == "maximize"
    ctrl_grid = np.asarray(agent.x_total_linespace, dtype=float)
    rf_vals = np.unique(ctrl_grid[:, 0])
    pi_vals = np.unique(ctrl_grid[:, 1])
    n_r, n_p = len(rf_vals), len(pi_vals)

    print(f"  GP landscape: predict over {len(ctrl_grid)} grid x {n_cov} cov ...")
    Y_mean = gp_landscape(agent, ctrl_grid, n_cov, rng).reshape(n_r, n_p)

    cov_samples = agent._sample_covariate_grid(agent.df_results)  # joint, n_cov_samples
    xi = agent._current_ei_xi()
    y_scaled = np.asarray(
        agent._y_scaler.transform(
            jnp.asarray(agent.df_results[obj_name].to_numpy(float).reshape(-1, 1))
        )
    )
    _, rng_key = gpax.utils.get_keys()
    print(
        f"  Acquisition: real _compute_robust_acq over {len(ctrl_grid)} grid "
        f"x {cov_samples.shape[0]} cov (xi={xi:.4f}) ..."
    )
    acq = np.asarray(
        agent._compute_robust_acq(
            rng_key,
            agent.model,
            ctrl_grid,
            cov_samples,
            agent._x_scaler,
            y_scaled,
            xi=xi,
        )
    )
    Y_acq = acq.reshape(n_r, n_p)

    ri, pii = np.unravel_index(
        int(np.argmax(Y_mean if maximize else -Y_mean)), (n_r, n_p)
    )

    # Next measurement points: the conditions this phase's acquisition actually
    # picked, read straight from the run (rows of the LAST phase in df_results),
    # NOT re-inferred.  n_conditions_per_iter conditions per phase.
    p0, p1 = [pp.name for pp in agent.parameters_to_optimize]
    df = agent.df_results
    if "phase_id" in df.columns:
        last = df[df["phase_id"] == int(df["phase_id"].max())]
    else:
        last = df.iloc[0:0]
    next_pts = last[[p0, p1]].drop_duplicates().to_numpy(dtype=float)

    return dict(
        rf_vals=rf_vals,
        pi_vals=pi_vals,
        Y_mean=Y_mean,
        Y_acq=Y_acq,
        xi=xi,
        acq_used=agent.acquisition_function.upper(),
        opt=(rf_vals[ri], pi_vals[pii], Y_mean[ri, pii]),
        next_pts=next_pts,
    )


def make_figure(df, obj_name, p0, p1, phase_id, panels, run_dir, out_path, ranges):
    """3-column figure for one phase; ``panels=None`` => GP not fit (initial).

    ``ranges`` = {"meas": (lo, hi), "pred": (lo, hi), "acq": (lo, hi)} -- fixed
    colour scales so every phase figure is directly comparable.  Measured
    (col 1) and predicted (col 2) deliberately use SEPARATE ranges (the GP
    predictions span a narrower band than the raw FOV measurements, so a shared
    range washes the landscape out).
    """
    m_lo, m_hi = ranges["meas"]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.4))
    stage = "initial spread" if panels is None else f"iter={phase_id}"
    fig.suptitle(
        f"v13 phase {phase_id} ({stage})  --  {os.path.basename(run_dir)}\n"
        f"{len(df)} FOVs, objective={obj_name} (marginalised over covariate samples)",
        fontsize=12,
        fontweight="bold",
    )

    # Column 1: measured scatter (ramp_fraction vs pulse_interval).
    ax = axes[0]
    jx = rng.normal(0, 0.012, size=len(df))
    jy = rng.normal(0, 0.15, size=len(df))
    sc = ax.scatter(
        df[p0] + jx,
        df[p1] + jy,
        c=df[obj_name],
        cmap="viridis",
        s=45,
        edgecolors="k",
        linewidths=0.4,
        vmin=m_lo,
        vmax=m_hi,
    )
    fig.colorbar(sc, ax=ax, label=obj_name)
    ax.set_title(f"Measured {obj_name}  (n_FOVs={len(df)})")

    if panels is None:
        for ax, ttl in zip(axes[1:], ["GP predicted landscape", "Acquisition"]):
            ax.text(
                0.5,
                0.5,
                "GP not fit yet\n(initial spread phase)",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
                color="gray",
            )
            ax.set_title(ttl)
    else:
        rf_vals, pi_vals = panels["rf_vals"], panels["pi_vals"]
        opt_rf, opt_pi, opt_val = panels["opt"]
        next_pts = panels["next_pts"]
        acq_used, xi = panels["acq_used"], panels["xi"]

        # Column 2: GP landscape -- its OWN range (cividis), fixed across phases.
        ax = axes[1]
        p_lo, p_hi = ranges["pred"]
        cf = ax.contourf(
            rf_vals,
            pi_vals,
            panels["Y_mean"].T,
            levels=np.linspace(p_lo, p_hi, 21),
            cmap="cividis",
            vmin=p_lo,
            vmax=p_hi,
            extend="both",
        )
        fig.colorbar(cf, ax=ax, label=f"predicted {obj_name}")
        ax.scatter(
            df[p0], df[p1], c="white", s=14, alpha=0.6, marker="x", linewidths=0.8
        )
        ax.scatter(
            opt_rf,
            opt_pi,
            c="red",
            s=240,
            marker="*",
            edgecolors="black",
            linewidths=1.5,
            zorder=10,
            label=f"opt {p0}={opt_rf:.2f}, {p1}={int(opt_pi)}\npred={opt_val:.3f}",
        )
        ax.set_title("GP predicted landscape (marginalised)")
        ax.legend(loc="best", fontsize=8)

        # Column 3: acquisition + the real next-phase measurement conditions.
        ax = axes[2]
        a_lo, a_hi = ranges["acq"]
        cf = ax.contourf(
            rf_vals,
            pi_vals,
            panels["Y_acq"].T,
            levels=np.linspace(a_lo, a_hi, 21),
            cmap="inferno",
            vmin=a_lo,
            vmax=a_hi,
            extend="both",
        )
        fig.colorbar(cf, ax=ax, label=f"{acq_used} acquisition (scaled y)")
        ax.scatter(
            df[p0], df[p1], c="white", s=14, alpha=0.6, marker="x", linewidths=0.8
        )
        if len(next_pts):
            ax.scatter(
                next_pts[:, 0],
                next_pts[:, 1],
                c="cyan",
                s=200,
                marker="X",
                edgecolors="black",
                linewidths=1.2,
                zorder=10,
                label=f"next conditions (n={len(next_pts)})",
            )
        ax.set_title(f"Acquisition {acq_used} (marginalised, xi={xi:.3f})")
        ax.legend(loc="best", fontsize=8)

    for ax in axes:
        ax.set_xlabel(p0)
        ax.set_ylabel(f"{p1} (frames)")
        ax.set_xlim(*XLIM)
        ax.set_ylim(*YLIM)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    return fig


def _collect_phase(model_path, run_dir, n_cov):
    """Build the agent + compute panels for one GP-fitted phase (slow pass)."""
    p = joblib.load(model_path)
    if p.get("model_state") is None or p["model_state"].get("backend") != "ExactGP":
        raise RuntimeError(f"{model_path}: no usable ExactGP model_state.")
    df = p["df_results"]
    if df is None or df.empty:
        raise RuntimeError(f"{model_path}: empty df_results.")
    phase_id = int(p["iteration"])
    print(f"Phase {phase_id}: {os.path.basename(model_path)}  ({len(df)} FOVs)")
    agent = build_agent(p, df, run_dir)
    panels = compute_panels(agent, n_cov, np.random.default_rng(0))
    o = panels["opt"]
    print(
        f"  opt: ramp_fraction={o[0]:.2f}, pulse_interval={int(o[1])}, pred={o[2]:.3f}"
    )
    return dict(
        phase_id=phase_id,
        df=df,
        panels=panels,
        obj_name=agent.objective_metric.name,
        p0=agent.parameters_to_optimize[0].name,
        p1=agent.parameters_to_optimize[1].name,
    )


def _global_ranges(jobs):
    """Fixed colour ranges across all phases so figures are comparable.

    Measured (col 1) and predicted (col 2) get SEPARATE ranges on purpose.
    """
    obj = jobs[0]["obj_name"]
    final_df = max((j["df"] for j in jobs), key=len)
    meas = (float(final_df[obj].min()), float(final_df[obj].max()))
    gp = [j["panels"] for j in jobs if j["panels"] is not None]
    if gp:
        pred = (
            min(float(p["Y_mean"].min()) for p in gp),
            max(float(p["Y_mean"].max()) for p in gp),
        )
        acq = (
            min(float(p["Y_acq"].min()) for p in gp),
            max(float(p["Y_acq"].max()) for p in gp),
        )
    else:
        pred, acq = meas, (0.0, 1.0)
    return dict(meas=meas, pred=pred, acq=acq)


def regenerate_phase_from_model(model_path, run_dir, n_cov, out_dir):
    """Single GP phase (--model): per-phase colour ranges."""
    job = _collect_phase(model_path, run_dir, n_cov)
    ranges = _global_ranges([job])
    out = os.path.join(out_dir, f"phase_{job['phase_id']:03d}.png")
    make_figure(
        job["df"],
        job["obj_name"],
        job["p0"],
        job["p1"],
        job["phase_id"],
        job["panels"],
        run_dir,
        out,
        ranges,
    )
    return job["phase_id"]


def regenerate_all(run_dir, n_cov=50, out_dir=None):
    """One figure per phase, with colour ranges fixed across all phases.

    Two passes: (1) compute every phase's GP landscape + acquisition, (2) derive
    the shared colour ranges and render.
    """
    out_dir = out_dir or os.path.join(run_dir, "plots", "regenerated")
    models = sorted(
        glob.glob(os.path.join(run_dir, "models", "bo_model_iter_*.joblib"))
    )
    model_phases = {int(re.search(r"iter_(\d+)", m).group(1)): m for m in models}
    first_gp = min(model_phases) if model_phases else 0

    # Pass 1 -- compute.
    jobs = []
    for ph in range(first_gp):  # initial-spread phases (no model)
        ckpt = os.path.join(
            run_dir, "checkpoints", f"bo_results_phase_{ph:03d}.parquet"
        )
        if not os.path.exists(ckpt):
            continue
        df = pd.read_parquet(ckpt)
        obj = "auc_norm" if "auc_norm" in df.columns else df.columns[-1]
        print(f"Phase {ph}: initial spread (no model)  ({len(df)} FOVs)")
        jobs.append(
            dict(
                phase_id=ph,
                df=df,
                panels=None,
                obj_name=obj,
                p0="ramp_fraction",
                p1="pulse_interval",
            )
        )
    for ph in sorted(model_phases):  # GP-fitted phases
        jobs.append(_collect_phase(model_phases[ph], run_dir, n_cov))

    if not jobs:
        print(f"No phases found in {run_dir}")
        return

    # Pass 2 -- shared ranges + render.
    ranges = _global_ranges(jobs)
    print(
        f"Colour ranges (fixed across phases): measured={ranges['meas']}, "
        f"predicted={ranges['pred']}, acq={ranges['acq']}"
    )
    for j in jobs:
        out = os.path.join(out_dir, f"phase_{j['phase_id']:03d}.png")
        make_figure(
            j["df"],
            j["obj_name"],
            j["p0"],
            j["p1"],
            j["phase_id"],
            j["panels"],
            run_dir,
            out,
            ranges,
        )
    print(f"\nDone: {len(jobs)} phase figures in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_dir", nargs="?", default=DEFAULT_RUN_DIR)
    ap.add_argument(
        "--n-cov",
        type=int,
        default=50,
        help="covariate samples for the GP landscape (default 50)",
    )
    ap.add_argument(
        "--out-dir", default=None, help="output dir (default {run}/plots/regenerated)"
    )
    ap.add_argument(
        "--model", default=None, help="render only this single bo_model_iter_*.joblib"
    )
    ap.add_argument("--show", action="store_true", help="plt.show() at the end")
    args = ap.parse_args()

    if args.model:
        out_dir = args.out_dir or os.path.join(args.run_dir, "plots", "regenerated")
        regenerate_phase_from_model(args.model, args.run_dir, args.n_cov, out_dir)
    else:
        regenerate_all(args.run_dir, n_cov=args.n_cov, out_dir=args.out_dir)
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
