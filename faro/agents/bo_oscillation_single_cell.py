"""Single-cell batch BO agent for ERK oscillation optimisation.

Same ERK oscillation experiment as :class:`OscillationBO`, but the
objective is computed **per cell** rather than per FOV.  Each qualifying
cell becomes one observation fed to the sparse GP, so per-phase the BO
sees hundreds of (x, y, covariates) triples instead of ~18.

Design decisions (confirmed with PI):

- **Objective**: ``mean_osc_probability`` — the mean of the classifier
  ``osc_probability`` over all sliding windows that fall within the
  scoring range ``[first_frame_stim, last_frame_stim]``.  Continuous in
  [0, 1].  Robust to damped / irregular oscillations because the
  classifier was trained to detect those directly (unlike FFT, which
  assumes regularity).

- **Per-cell covariates**:
    * ``baseline_cnr`` — mean CNR over the baseline window for this cell
    * ``optortk_expression`` — ``ref_mean_intensity`` for this cell
      (optoRTK reporter at the optocheck timepoint)
    * ``mean_dist_k_nearest`` — mean distance (µm, or pixels depending
      on the feature extractor) to the ``k=5`` nearest neighbours at
      baseline — a local-density proxy that replaces the FOV-level
      ``n_cells`` covariate with something meaningful per cell.

- **Quality gates**: same as :class:`OscillationBO` (``min_track_fraction``
  and ``max_baseline_cnr``).  Applied per cell *before* any observation
  reaches the BO.

- **GP backend**: combined with :class:`BOptGPAXSparse` (variational
  sparse GP) via multiple inheritance — ExactGP scales as ``O(N^3)``
  and becomes too slow once you have a few thousand cells.

The class is thin: it only overrides :meth:`_preprocess_results` (to
yield per-cell rows instead of per-FOV rows).  Everything else -- event
creation, plotting, oscillation scoring helpers, checkpointing, etc. --
is inherited unchanged from :class:`OscillationBO`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from faro.agents.bo_optimization_sparse import BOptGPAXSparse
from faro.agents.bo_oscillation import OscillationBO


class OscillationBOSingleCell(OscillationBO, BOptGPAXSparse):
    """Single-cell batch BO for ERK oscillation.

    MRO: ``OscillationBOSingleCell -> OscillationBO -> BOptGPAXSparse
    -> BOptGPAX`` so method resolution picks up:

    * event creation, oscillation scoring helpers, plotting from
      :class:`OscillationBO`,
    * sparse GP acquisition from :class:`BOptGPAXSparse`,
    * run loop / batch management from :class:`BOptGPAX`.

    Only :meth:`_preprocess_results` is overridden.

    Args:
        density_k_neighbours: How many nearest neighbours to average
            over when computing ``mean_dist_k_nearest`` per cell.
            Default ``5``.  Distances are computed on cell centroids
            at baseline frames (``fov_timestep < n_baseline_frames``).
        **kwargs: Everything else accepted by :class:`OscillationBO` and
            :class:`BOptGPAXSparse` (``gp_backend``,
            ``inducing_points_ratio``, ``num_svi_steps``, etc.).
    """

    def __init__(self, *, density_k_neighbours: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.density_k_neighbours = int(density_k_neighbours)

    # ------------------------------------------------------------------
    # Per-cell result preprocessing
    # ------------------------------------------------------------------

    def _preprocess_results(self, fov_tracks: dict) -> pd.DataFrame:
        """Return one row per qualifying cell.

        Columns:
            stim_exposure, ramp                   -- from the condition map
            baseline_cnr                          -- per-cell mean baseline CNR
            optortk_expression                    -- per-cell ref_mean_intensity
            mean_dist_k_nearest                   -- per-cell local density proxy
            mean_osc_probability                  -- per-cell objective

        Quality gates (``min_track_fraction`` and ``max_baseline_cnr``)
        are applied per cell -- cells that fail either gate are dropped
        entirely (neither numerator nor denominator, since at single-cell
        level every row is its own numerator).
        """
        phase_id = self._current_phase_id
        results: list[dict] = []

        min_frames = int(self.min_track_fraction * self.n_frames)

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
                continue

            # --- Per-cell baseline CNR ---------------------------------
            baseline_df = df_phase[df_phase["fov_timestep"] < self.n_baseline_frames]
            per_cell_baseline = (
                baseline_df.groupby("particle")[cnr_col].mean().dropna()
                if not baseline_df.empty
                else pd.Series(dtype=float)
            )

            # --- Per-cell track length ---------------------------------
            frames_per_cell = df_phase.groupby("particle")["fov_timestep"].nunique()

            # --- Per-cell optoRTK expression ---------------------------
            if "ref_mean_intensity" in df_phase.columns:
                per_cell_optortk = (
                    df_phase.groupby("particle")["ref_mean_intensity"].first().dropna()
                )
            else:
                per_cell_optortk = pd.Series(dtype=float)

            # --- Per-cell mean distance to k nearest neighbours --------
            # Uses centroids at baseline.  The feature extractor may
            # expose (x, y), (centroid-0, centroid-1), or (centroid_x,
            # centroid_y); probe in that order.
            per_cell_density = self._compute_per_cell_density(
                baseline_df if not baseline_df.empty else df_phase,
                k=self.density_k_neighbours,
            )

            # --- Iterate over cells and classify -----------------------
            for particle, grp in df_phase.groupby("particle"):
                # Gate 1: tracking length
                if frames_per_cell.get(particle, 0) < min_frames:
                    continue
                # Gate 2: baseline CNR
                if (
                    self.max_baseline_cnr is not None
                    and particle in per_cell_baseline.index
                    and per_cell_baseline.loc[particle] >= self.max_baseline_cnr
                ):
                    continue

                grp_clf = grp.sort_values("fov_timestep")
                grp_clf = grp_clf[
                    (grp_clf["fov_timestep"] >= self.classifier_first_frame)
                    & (grp_clf["fov_timestep"] < self.classifier_last_frame)
                ]
                if len(grp_clf) < 20:
                    continue

                x = grp_clf["fov_timestep"].values.astype(float)
                y = grp_clf[cnr_col].values.astype(float)
                windows = self.osc_predict_fn(
                    x,
                    y,
                    self.osc_clf,
                    self.osc_scaler,
                    self.osc_feature_cols,
                    self.osc_cfg,
                )
                # Filter to scoring window (same gate _is_cell_oscillating uses)
                scored_windows = self._filter_windows(windows)
                if not scored_windows:
                    continue

                mean_prob = float(
                    np.mean([w["osc_probability"] for w in scored_windows])
                )

                results.append(
                    {
                        "stim_exposure": params["stim_exposure"],
                        "ramp": params["ramp"],
                        "baseline_cnr": float(per_cell_baseline.get(particle, 0.0)),
                        "optortk_expression": float(
                            per_cell_optortk.get(particle, 0.0)
                        ),
                        "mean_dist_k_nearest": float(
                            per_cell_density.get(particle, 0.0)
                        ),
                        "mean_osc_probability": mean_prob,
                        # Bookkeeping columns (not used as BO features)
                        "fov": int(fov_idx),
                        "particle": int(particle),
                        "phase_id": int(phase_id),
                    }
                )

        if not results:
            print(f"Warning: no valid cells in phase {phase_id}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        print(
            f"  Phase {phase_id}: {len(df)} cells (from "
            f"{df['fov'].nunique()} FOVs), "
            f"mean mean_osc_probability={df['mean_osc_probability'].mean():.4f}, "
            f"max={df['mean_osc_probability'].max():.4f}"
        )
        return df

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_per_cell_density(df: pd.DataFrame, *, k: int) -> pd.Series:
        """Mean distance to the *k* nearest neighbours per particle.

        Uses the first timepoint of each particle's centroid.  Returns a
        Series indexed by particle id; missing entries (isolated FOVs,
        missing columns) default to 0.0 at the call site.
        """
        # Probe for centroid columns
        if {"x", "y"}.issubset(df.columns):
            x_col, y_col = "x", "y"
        elif {"centroid_x", "centroid_y"}.issubset(df.columns):
            x_col, y_col = "centroid_x", "centroid_y"
        elif {"centroid-0", "centroid-1"}.issubset(df.columns):
            x_col, y_col = "centroid-1", "centroid-0"
        else:
            return pd.Series(dtype=float)

        # First observation per particle
        first = df.groupby("particle")[[x_col, y_col]].first().dropna()
        if len(first) < k + 1:
            # Not enough cells to compute a k-NN; fall back to 0
            return pd.Series(np.zeros(len(first)), index=first.index, dtype=float)

        centroids = first[[x_col, y_col]].to_numpy(dtype=float)

        try:
            from scipy.spatial import cKDTree

            tree = cKDTree(centroids)
            # k+1 because nearest neighbour is the point itself (dist 0)
            dists, _ = tree.query(centroids, k=k + 1)
            mean_dist = dists[:, 1:].mean(axis=1)
        except ImportError:
            # Fallback: brute-force pairwise distances (O(N^2))
            N = centroids.shape[0]
            D = np.linalg.norm(centroids[:, None, :] - centroids[None, :, :], axis=-1)
            D[np.eye(N, dtype=bool)] = np.inf
            D_sorted = np.sort(D, axis=1)
            mean_dist = D_sorted[:, :k].mean(axis=1)

        return pd.Series(mean_dist, index=first.index, dtype=float)
