"""Density-aware FOV scoring and window selection.

This module holds the *hardware-agnostic* heart of the grid-overview FOV
finder (:class:`faro.agents.grid_fov_finder.GridFOVFinderAgent`).  It works
purely on **cell centroids in µm** in some global frame, so the same code can
be exercised offline on saved TIFFs (see the notebook
``experiments/32_fov_finder/grid_density_fov_finder_demo.ipynb``) and online on
a freshly-acquired grid scan.

The idea
--------
Instead of imaging scattered random candidates and *returning those same
points* (what :class:`~faro.agents.fov_finder.FOVFinderAgent` does), the grid
finder scans a contiguous region, reconstructs where the cells actually are
(a centroid cloud), and then chooses final FOV positions **anywhere** in that
region — not just at the tile centres.  A candidate FOV is scored on:

* **count** — enough cells, but not too many (``min_cells`` / ``max_cells``);
* **clumping** — cells must not be packed on top of each other.  Quantified
  by the per-cell nearest-neighbour (NN) distance: a cell whose nearest
  neighbour is closer than ``clump_distance_um`` is counted as *clumped*, and
  a window is rejected once more than ``max_clumped_fraction`` of its cells
  are clumped;
* **spread band** — optionally the *median* NN distance must sit inside
  ``[min_nn_um, max_nn_um]`` so the field is neither a dense carpet
  (median NN too small) nor a few lonely cells (median NN too large).

Recentering ("move the FOV onto the nice cells")
-------------------------------------------------
Candidate window centres are seeded at the cells themselves and then
**mean-shifted**: a few iterations of "take the cells currently inside the
window, recompute their centroid, move the window there".  So if the overview
scan caught a beautiful cluster sitting at the *edge* of a tile (rest of the
tile empty), the window slides over until that cluster is centred — pulling in
any neighbouring cells that were just out of frame.  This is the key win over
discrete tile-centre selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from faro.agents.fov_finder import FOVCondition


# ----------------------------------------------------------------------
# Geometry / neighbour helpers
# ----------------------------------------------------------------------


def nearest_neighbor_distances(xy: np.ndarray) -> np.ndarray:
    """Return the distance from each point to its nearest other point.

    Args:
        xy: ``(N, 2)`` array of (x, y) coordinates (any consistent unit).

    Returns:
        ``(N,)`` array of nearest-neighbour distances.  Points with no
        neighbour (``N < 2``) get ``np.inf`` — i.e. an isolated cell is
        treated as "maximally un-clumped".
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(xy)
    if n < 2:
        return np.full(n, np.inf)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(xy)
        # k=2: nearest is the point itself (dist 0), second is the true NN.
        dist, _ = tree.query(xy, k=2)
        return dist[:, 1]
    except Exception:
        diff = xy[:, None, :] - xy[None, :, :]
        dmat = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(dmat, np.inf)
        return dmat.min(axis=1)


def label_centroids_um(
    label_img: np.ndarray,
    pixel_size_um: float = 1.0,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    *,
    flip_x: bool = False,
    flip_y: bool = False,
    return_labels: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Centroids of a label image as ``(N, 2)`` (x, y) coordinates in µm.

    ``regionprops`` returns ``centroid-0 = row`` (image y) and
    ``centroid-1 = col`` (image x); we map col→x and row→y so the returned
    array is in conventional (x, y) order.

    Args:
        label_img: Integer label image (0 = background).
        pixel_size_um: Camera pixel size (µm / px).
        origin_xy: Stage (x, y) of the image's *centre* in global µm — the
            tile centre when stitching a grid scan.  Pixel offsets are taken
            relative to the image centre so a cell dead-centre in the tile
            maps to ``origin_xy``.
        flip_x / flip_y: Negate the in-image offset along that axis to match
            the camera→stage orientation (e.g. a 180° plate rotation is
            ``flip_x=flip_y=True``).
        return_labels: If ``True``, also return the integer label id of each
            centroid (same order as the rows), so per-cell features computed
            elsewhere (e.g. ERK CNR from a feature extractor) can be joined
            onto the cloud by label.

    Returns:
        ``(N, 2)`` float array of global (x, y) centroids in µm, or
        ``((N, 2), (N,))`` when *return_labels* is set.  Empty arrays when the
        image has no objects.
    """
    from skimage.measure import regionprops_table

    if label_img is None or label_img.size == 0 or int(label_img.max()) == 0:
        empty_xy = np.empty((0, 2), dtype=float)
        if return_labels:
            return empty_xy, np.empty((0,), dtype=int)
        return empty_xy
    props = ["centroid", "label"] if return_labels else ["centroid"]
    table = regionprops_table(label_img, properties=props)
    rows = np.asarray(table["centroid-0"], dtype=float)
    cols = np.asarray(table["centroid-1"], dtype=float)
    h, w = label_img.shape[:2]
    dx = (cols - w / 2.0) * pixel_size_um
    dy = (rows - h / 2.0) * pixel_size_um
    if flip_x:
        dx = -dx
    if flip_y:
        dy = -dy
    xy = np.column_stack([origin_xy[0] + dx, origin_xy[1] + dy])
    if return_labels:
        return xy, np.asarray(table["label"], dtype=int)
    return xy


def build_stage_montage(
    tile_centers_um: np.ndarray,
    tile_images: list,
    pixel_size_um: float,
    *,
    flip_x: bool = False,
    flip_y: bool = False,
    reduce: str = "max",
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Stitch grid tiles into one image laid out in **stage µm coordinates**.

    Uses the *same* pixel→stage convention as :func:`label_centroids_um`
    (col→x, row→y, with optional ``flip_x`` / ``flip_y``), so the montage is a
    faithful picture of where the agent *thinks* each pixel sits in stage
    space.  Overlay the detected centroids or selected FOV boxes on it (in
    stage coordinates) to check that the transform is self-consistent: if a
    flip is wrong the overlapping tile seams won't line up.

    Args:
        tile_centers_um: ``(T, 2)`` stage (x, y) centres of each tile (µm).
        tile_images: list of ``T`` 2D arrays (same H×W), aligned to
            *tile_centers_um*.
        pixel_size_um: Camera pixel size (µm / px).
        flip_x / flip_y: Camera→stage axis flips — must match what was passed
            to the finder.
        reduce: How to combine overlapping tile pixels — ``"max"`` (default,
            robust to seams) or ``"last"`` (later tile overwrites).

    Returns:
        ``(montage, (x_min, x_max, y_min, y_max))``.  Display with
        ``imshow(montage, origin="lower", extent=(x_min, x_max, y_min, y_max))``
        so stage coordinates map directly onto the axes.
    """
    imgs = [np.asarray(im, dtype=float) for im in tile_images]
    if not imgs:
        return np.zeros((1, 1)), (0.0, 0.0, 0.0, 0.0)
    h, w = imgs[0].shape[:2]
    half_w = (w / 2.0) * pixel_size_um
    half_h = (h / 2.0) * pixel_size_um
    centers = np.asarray(tile_centers_um, dtype=float).reshape(-1, 2)
    x_min = float((centers[:, 0] - half_w).min())
    x_max = float((centers[:, 0] + half_w).max())
    y_min = float((centers[:, 1] - half_h).min())
    y_max = float((centers[:, 1] + half_h).max())
    W = max(1, int(round((x_max - x_min) / pixel_size_um)))
    H = max(1, int(round((y_max - y_min) / pixel_size_um)))
    montage = np.zeros((H, W), dtype=float)

    for (xc, yc), im in zip(centers, imgs):
        f = im
        # Reorient so increasing col -> increasing stage_x and increasing row
        # -> increasing stage_y, matching label_centroids_um's mapping.
        if flip_x:
            f = f[:, ::-1]
        if flip_y:
            f = f[::-1, :]
        col0 = int(round((xc - half_w - x_min) / pixel_size_um))
        row0 = int(round((yc - half_h - y_min) / pixel_size_um))
        rr0, cc0 = max(row0, 0), max(col0, 0)
        rr1, cc1 = min(row0 + h, H), min(col0 + w, W)
        if rr1 <= rr0 or cc1 <= cc0:
            continue
        sub = f[rr0 - row0 : rr1 - row0, cc0 - col0 : cc1 - col0]
        if reduce == "max":
            montage[rr0:rr1, cc0:cc1] = np.maximum(montage[rr0:rr1, cc0:cc1], sub)
        else:
            montage[rr0:rr1, cc0:cc1] = sub
    return montage, (x_min, x_max, y_min, y_max)


def cells_in_window(
    xy: np.ndarray, center: np.ndarray, fov_w_um: float, fov_h_um: float
) -> np.ndarray:
    """Boolean mask of which points fall inside an axis-aligned FOV window."""
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)
    cx, cy = float(center[0]), float(center[1])
    return (np.abs(xy[:, 0] - cx) <= fov_w_um / 2.0) & (
        np.abs(xy[:, 1] - cy) <= fov_h_um / 2.0
    )


# ----------------------------------------------------------------------
# Scorer
# ----------------------------------------------------------------------


@dataclass
class FovDensityScorer:
    """Score an FOV-sized window of cells by count + spatial distribution.

    All distances are in µm.  ``metrics`` is a pure function of the cells
    inside the window, so it can be tuned offline on saved images.

    Args:
        fov_w_um / fov_h_um: Size of the FOV window (camera field of view).
        min_cells: Reject windows with fewer cells than this.
        max_cells: Reject windows with more cells than this (``None`` = no
            upper bound).
        clump_distance_um: A cell whose nearest neighbour is closer than this
            is counted as *clumped*.  This is the main "dense" knob.
        max_clumped_fraction: Reject the window when the fraction of clumped
            cells exceeds this (``1.0`` = never reject on clumping).
        min_nn_um: Optional lower bound on the *median* NN distance — reject
            fields that are too densely packed overall.  ``None`` disables.
        max_nn_um: Optional upper bound on the *median* NN distance — reject
            fields where cells sit too far apart (sparse / scattered).
            ``None`` disables.
        fov_conditions: Optional list of
            :class:`~faro.agents.fov_finder.FOVCondition` evaluated on
            **per-cell features** of the cells inside the window (e.g. ERK
            CNR).  A window is rejected unless every condition passes.  The
            features are supplied per window via ``feat_window`` (see
            :meth:`metrics`); when this list is non-empty the window finder
            must be given a ``feat`` mapping.  This is what lets the grid
            finder *select for ERK activity* on top of pure cell density.
    """

    fov_w_um: float
    fov_h_um: float
    min_cells: int = 10
    max_cells: int | None = None
    clump_distance_um: float = 15.0
    max_clumped_fraction: float = 0.3
    min_nn_um: float | None = None
    max_nn_um: float | None = None
    fov_conditions: list["FOVCondition"] = field(default_factory=list)

    def metrics(
        self, xy_window: np.ndarray, feat_window: dict[str, np.ndarray] | None = None
    ) -> dict:
        """Compute count / clumping / spread metrics for cells in a window.

        Args:
            xy_window: ``(M, 2)`` centroids of the cells already known to be
                inside the window.
            feat_window: Optional ``{feature_name: (M,) array}`` of per-cell
                features for those same cells (aligned to *xy_window*).
                Required when :attr:`fov_conditions` is set; each condition is
                evaluated against it and its satisfied-fraction recorded.

        Returns:
            Dict with ``n_cells``, ``clumped_fraction``, ``median_nn_um``,
            ``min_nn_um`` (smallest NN distance) and ``density_per_mm2``.  When
            :attr:`fov_conditions` is set it also carries
            ``conditions_pass`` (bool) and one
            ``cond_<feature>_<operator>_frac`` column per condition.
        """
        xy = np.asarray(xy_window, dtype=float).reshape(-1, 2)
        n = len(xy)
        area_mm2 = (self.fov_w_um * self.fov_h_um) / 1e6
        if n == 0:
            m = {
                "n_cells": 0,
                "clumped_fraction": 0.0,
                "median_nn_um": np.inf,
                "min_nn_um": np.inf,
                "density_per_mm2": 0.0,
            }
        else:
            nn = nearest_neighbor_distances(xy)
            finite = nn[np.isfinite(nn)]
            clumped_fraction = (
                float(np.mean(nn < self.clump_distance_um)) if n >= 2 else 0.0
            )
            m = {
                "n_cells": int(n),
                "clumped_fraction": clumped_fraction,
                "median_nn_um": float(np.median(finite)) if finite.size else np.inf,
                "min_nn_um": float(finite.min()) if finite.size else np.inf,
                "density_per_mm2": float(n / area_mm2) if area_mm2 > 0 else 0.0,
            }

        # Per-cell feature conditions (e.g. ERK CNR).  Build a one-row-per-cell
        # DataFrame from the window's features and reuse FOVCondition.check,
        # so the gate is identical to FOVFinderAgent's.
        if self.fov_conditions:
            df = pd.DataFrame(feat_window or {})
            all_pass = True
            for cond in self.fov_conditions:
                passed, fraction = cond.check(df)
                m[f"cond_{cond.feature}_{cond.operator}_frac"] = fraction
                all_pass = all_pass and passed
            m["conditions_pass"] = bool(all_pass)
        return m

    def passes(self, m: dict) -> tuple[bool, str]:
        """Return ``(ok, reason)``; ``reason`` is the first failing check."""
        if m["n_cells"] < self.min_cells:
            return False, "below_min_cells"
        if self.max_cells is not None and m["n_cells"] > self.max_cells:
            return False, "above_max_cells"
        if m["clumped_fraction"] > self.max_clumped_fraction:
            return False, "too_clumped"
        if self.min_nn_um is not None and m["median_nn_um"] < self.min_nn_um:
            return False, "too_dense"
        if self.max_nn_um is not None and m["median_nn_um"] > self.max_nn_um:
            return False, "too_sparse"
        # Feature gate last: a field can have great density yet the wrong ERK
        # state, so reject it only after the cheap geometric checks pass.
        if self.fov_conditions and not m.get("conditions_pass", True):
            return False, "condition_failed"
        return True, ""

    def score(self, m: dict) -> float:
        """Scalar quality (higher = better) for ranking passing windows.

        Rewards cell count and penalises clumping:
        ``n_cells * (1 - clumped_fraction)``.  Selection only ever ranks
        windows that already ``passes()``, so this just orders the good ones.
        """
        return float(m["n_cells"]) * (1.0 - float(m["clumped_fraction"]))


# ----------------------------------------------------------------------
# Window finding
# ----------------------------------------------------------------------


def _nonoverlap_ok(
    p: np.ndarray,
    selected_xy: list[np.ndarray],
    fov_w_um: float,
    fov_h_um: float,
    min_separation_um: float | None,
) -> bool:
    """True if an FOV at *p* neither overlaps nor is too close to selected ones.

    Two axis-aligned FOVs overlap iff ``|dx| < fov_w`` **and** ``|dy| < fov_h``;
    this is the exact rectangle test (a Euclidean centre-distance threshold is
    *not* sufficient — two windows 1.1·side apart can still overlap on the
    diagonal).  ``min_separation_um``, when set, adds an extra centre-to-centre
    spacing requirement on top of non-overlap.
    """
    for q in selected_xy:
        dx = abs(float(p[0]) - float(q[0]))
        dy = abs(float(p[1]) - float(q[1]))
        if dx < fov_w_um and dy < fov_h_um:
            return False  # rectangles overlap
        if min_separation_um is not None and np.hypot(dx, dy) < min_separation_um:
            return False
    return True


def _select_nonoverlapping(
    df: pd.DataFrame,
    order: list[int],
    pts: np.ndarray,
    n_select: int,
    fov_w_um: float,
    fov_h_um: float,
    min_separation_um: float | None,
    selected: list[int],
    sel_xy: list[np.ndarray],
    mode: str = "spread",
) -> None:
    """Greedily add non-overlapping windows; mutates *selected* / *sel_xy*.

    ``order`` is score-descending.  ``mode`` controls how each next window is
    chosen among the non-overlapping candidates:

    * ``"spread"`` — seed on the highest-score window, then take the window
      **farthest** from those already selected (farthest-point sampling).
      Use for *good* FOVs you want pushed as far apart as possible.
    * ``"score"`` — always take the highest-score eligible window.  Use when
      topping up with below-threshold fills: it picks the *best available*
      leftover region (which tends to sit near the cells) rather than an empty
      far corner.

    Either way no two selected FOVs overlap.
    """
    while len(selected) < n_select:
        best_i = -1
        best_key = None
        for i in order:
            if i in selected:
                continue
            p = pts[i]
            if not _nonoverlap_ok(p, sel_xy, fov_w_um, fov_h_um, min_separation_um):
                continue
            if mode == "score" or not sel_xy:
                best_i = i  # order is score-desc -> first eligible = best score
                break
            min_d = min(np.hypot(p[0] - q[0], p[1] - q[1]) for q in sel_xy)
            if best_key is None or min_d > best_key:
                best_key = min_d
                best_i = i
        if best_i == -1:
            break  # nothing left that fits without overlapping
        selected.append(best_i)
        sel_xy.append(pts[best_i])


def _mean_shift_center(
    xy: np.ndarray,
    center: np.ndarray,
    fov_w_um: float,
    fov_h_um: float,
    iters: int,
) -> np.ndarray:
    """Slide a window onto its local cluster of cells (mean-shift)."""
    c = np.asarray(center, dtype=float).copy()
    for _ in range(max(0, iters)):
        mask = cells_in_window(xy, c, fov_w_um, fov_h_um)
        if not mask.any():
            break
        new_c = xy[mask].mean(axis=0)
        if np.allclose(new_c, c):
            break
        c = new_c
    return c


def find_fov_windows(
    xy: np.ndarray,
    scorer: FovDensityScorer,
    *,
    n_select: int = 3,
    recenter: bool = True,
    recenter_iters: int = 3,
    min_separation_um: float | None = None,
    fill_invalid: bool = False,
    fill_prefers_conditions: bool = True,
    candidate_centers: np.ndarray | None = None,
    feat: dict[str, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pick ``n_select`` non-overlapping, maximally-spread FOV windows.

    Selection guarantees and goals:

    * **Non-overlap (hard).** No two returned FOVs overlap — checked with the
      exact axis-aligned-rectangle test for ``scorer.fov_w_um`` ×
      ``scorer.fov_h_um``, so the handed positions can always be imaged side by
      side without sharing cells.
    * **Maximal spacing.** After seeding on the best-scoring window, every
      further pick is the eligible window *farthest* from those already chosen
      (farthest-point sampling), so the FOVs are spread as far apart as the
      cloud permits.
    * **Quality first.** Density-passing (``valid``) windows are exhausted
      before any fallback.

    Args:
        xy: ``(N, 2)`` global cell centroids in µm (the overview-scan cloud).
        scorer: Configured :class:`FovDensityScorer`.
        n_select: How many windows to return.
        recenter: If ``True``, mean-shift each candidate centre onto its local
            cluster before scoring (the "move onto the nice cells" behaviour).
        recenter_iters: Mean-shift iterations.
        min_separation_um: Optional *extra* minimum centre-to-centre spacing
            (µm) on top of the always-enforced non-overlap.  ``None`` (default)
            relies on non-overlap alone, i.e. FOVs may sit edge-to-edge.
        fill_invalid: If ``True`` and fewer than ``n_select`` valid windows can
            be placed without overlapping, top up from the **invalid** windows,
            picking the *best-scoring* (densest, least-clumped) non-overlapping
            leftover rather than the farthest one — a good nearby FOV beats an
            empty far corner.  Use this when a downstream agent needs exactly
            ``n_select`` FOVs.  When ``False`` the result may contain fewer
            than ``n_select`` (and you avoid imaging junk fields entirely).
        fill_prefers_conditions: When topping up invalid windows (see
            *fill_invalid*) and the *scorer* carries ``fov_conditions``, rank the
            fill by ``(cell-count band, feature gate, density score)`` — i.e.
            put the feature gate (e.g. ERK CNR) **above clumping** but keep the
            cell-count band on top.  So an ERK-ready field that was rejected only
            for being *too clumped* beats a clean but ERK-dead one, while a field
            that fails ``min_cells`` / ``max_cells`` is still not preferred.
            ``True`` (default) is a no-op without ``fov_conditions``; set
            ``False`` to rank the fill purely by density score.
        candidate_centers: ``(K, 2)`` seed centres to evaluate.  Defaults to
            the cell positions themselves (so every cluster is a candidate).
        feat: Optional ``{feature_name: (N,) array}`` of per-cell features
            aligned to *xy* (e.g. ``{"cnr": ...}``).  Required when the
            *scorer* carries ``fov_conditions``; each window is gated on the
            features of the cells it contains, so the finder can *select for
            ERK activity* and not only cell density.

    Returns:
        ``(df_all, df_selected)`` DataFrames.  ``df_all`` has one row per
        (deduplicated) candidate window with columns ``x, y, n_cells,
        clumped_fraction, median_nn_um, min_nn_um, density_per_mm2, valid,
        reason, score``.  ``df_selected`` is the chosen subset (≤ ``n_select``);
        rows with ``valid == False`` are non-overlapping fallback fills.
    """
    xy = np.asarray(xy, dtype=float).reshape(-1, 2)

    if candidate_centers is None:
        centers = xy.copy()
    else:
        centers = np.asarray(candidate_centers, dtype=float).reshape(-1, 2)

    if len(xy) == 0 or len(centers) == 0:
        cols = [
            "x",
            "y",
            "n_cells",
            "clumped_fraction",
            "median_nn_um",
            "min_nn_um",
            "density_per_mm2",
            "valid",
            "reason",
            "score",
        ]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    if recenter:
        centers = np.array(
            [
                _mean_shift_center(
                    xy, c, scorer.fov_w_um, scorer.fov_h_um, recenter_iters
                )
                for c in centers
            ]
        )

    # Deduplicate near-identical centres (mean-shift collapses clusters onto
    # the same point); round to 1 µm so we don't score the same window twice.
    _, uniq_idx = np.unique(np.round(centers, 0), axis=0, return_index=True)
    centers = centers[np.sort(uniq_idx)]

    rows: list[dict] = []
    for c in centers:
        mask = cells_in_window(xy, c, scorer.fov_w_um, scorer.fov_h_um)
        feat_window = (
            {k: np.asarray(v)[mask] for k, v in feat.items()}
            if feat is not None
            else None
        )
        m = scorer.metrics(xy[mask], feat_window)
        ok, reason = scorer.passes(m)
        rows.append(
            {
                "x": float(c[0]),
                "y": float(c[1]),
                **m,
                "valid": ok,
                "reason": reason,
                "score": scorer.score(m),
            }
        )
    df_all = pd.DataFrame(rows).reset_index(drop=True)

    # Selection: spread non-overlapping windows, valid (quality) first, then
    # optionally top up from invalid windows — always non-overlapping.
    pts = df_all[["x", "y"]].to_numpy(dtype=float)
    valid_order = sorted(
        df_all.index[df_all["valid"]].tolist(),
        key=lambda i: -float(df_all.at[i, "score"]),
    )
    selected: list[int] = []
    sel_xy: list[np.ndarray] = []
    # Good (valid) FOVs are spread as far apart as possible.
    _select_nonoverlapping(
        df_all,
        valid_order,
        pts,
        n_select,
        scorer.fov_w_um,
        scorer.fov_h_um,
        min_separation_um,
        selected,
        sel_xy,
        mode="spread",
    )
    if fill_invalid and len(selected) < n_select:
        invalid_idx = df_all.index[~df_all["valid"]].tolist()
        if fill_prefers_conditions and "conditions_pass" in df_all.columns:
            # Fill priority: (cell-count band, feature gate, density score).
            # Cell count keeps top priority, then the feature gate (e.g. ERK CNR)
            # ranks ABOVE clumping (clumping only enters via the score), so an
            # ERK-ready but clumped field beats a clean but ERK-dead one while a
            # field outside the cell-count band is still not preferred.
            def _fill_key(i: int) -> tuple[bool, bool, float]:
                n = float(df_all.at[i, "n_cells"])
                count_ok = n >= scorer.min_cells and (
                    scorer.max_cells is None or n <= scorer.max_cells
                )
                return (
                    count_ok,
                    bool(df_all.at[i, "conditions_pass"]),
                    float(df_all.at[i, "score"]),
                )

            invalid_order = sorted(invalid_idx, key=_fill_key, reverse=True)
        else:
            invalid_order = sorted(
                invalid_idx, key=lambda i: -float(df_all.at[i, "score"])
            )
        # Fills pick the *best available* leftover window rather than the
        # farthest — a good nearby FOV beats an empty far corner.
        _select_nonoverlapping(
            df_all,
            invalid_order,
            pts,
            n_select,
            scorer.fov_w_um,
            scorer.fov_h_um,
            min_separation_um,
            selected,
            sel_xy,
            mode="score",
        )

    df_selected = df_all.loc[selected].copy()
    return df_all, df_selected
