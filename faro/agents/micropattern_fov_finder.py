"""FOV finder for micropatterned wells (one cell per grid node).

Companion to :class:`faro.agents.fov_finder.FOVFinderAgent`.  Where the
plain FOV finder picks random positions and filters by cell count, this
agent assumes the well carries a known **square micropattern** (period
fixed at fabrication, e.g. 100 um) and looks for FOVs where the cell
distribution best matches the lattice -- one nucleus on every grid
node, no doublets, no off-lattice contaminants.

Pipeline (per phase, per well):

1. Move to well centre.
2. Run a coarse mosaic scan across the well, segmenting H2B nuclei at
   every tile and projecting their centroids back into stage frame.
3. Estimate the lattice rotation from the centroids (period is known,
   rotation is unknown -- the plate is rarely axis-aligned to the
   stage).  Phase falls out of the same fit.
4. Score every candidate FOV by "fewest defects" (empty nodes,
   doublets, off-lattice cells), optionally snapping the FOV centre to
   a lattice node so nuclei don't get clipped at the edge.
5. Return the top-K positions (greedy spaced).

The algorithm helpers (``estimate_lattice``, ``score_fov``,
``find_best_fovs``) are public so the simulation script in
``experiments/32_fov_finder/micropattern/`` can drive them headlessly
against synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from faro.agents.base import PreExperimentAgent
from faro.agents.fov_finder import FOVCondition
from faro.core.data_structures import Channel, RTMSequence
from faro.core.utils import FovPosition

if TYPE_CHECKING:
    from useq import WellPlate, WellPlatePlan

    from faro.feature_extraction.base import FeatureExtractor
    from faro.microscope.base import AbstractMicroscope
    from faro.segmentation.base import Segmentator


# ----------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------


def _rotate(points: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate ``(N, 2)`` points by ``angle_deg`` around the origin."""
    a = np.deg2rad(angle_deg)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return np.asarray(points) @ R.T


# ----------------------------------------------------------------------
# Lattice fitting (rotation + phase from a noisy point cloud)
# ----------------------------------------------------------------------


@dataclass
class LatticeFit:
    """Estimated micropattern lattice parameters.

    Attributes:
        rotation_deg: Rotation of the lattice in stage frame, in degrees.
        phase_um: ``(phase_x, phase_y)`` in lattice frame -- where node
            ``(0, 0)`` sits relative to the world origin after un-rotating.
        sharpness: Diagnostic.  Bigger == cleaner fit.  Below ~5 the fit
            should be considered untrustworthy (too few cells, wrong
            period, or no actual lattice).
    """

    rotation_deg: float
    phase_um: tuple[float, float]
    sharpness: float

    def predict_nodes(
        self,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
        period: float,
    ) -> np.ndarray:
        """Lattice nodes inside a stage-frame bounding box."""
        ks_x = np.arange(
            int(np.floor((x_min - 2 * period) / period)),
            int(np.ceil((x_max + 2 * period) / period)) + 1,
        )
        ks_y = np.arange(
            int(np.floor((y_min - 2 * period) / period)),
            int(np.ceil((y_max + 2 * period) / period)) + 1,
        )
        gx, gy = np.meshgrid(
            self.phase_um[0] + ks_x * period,
            self.phase_um[1] + ks_y * period,
        )
        nodes_lat = np.stack([gx.ravel(), gy.ravel()], axis=1)
        nodes_stage = _rotate(nodes_lat, self.rotation_deg)
        m = (
            (nodes_stage[:, 0] >= x_min)
            & (nodes_stage[:, 0] <= x_max)
            & (nodes_stage[:, 1] >= y_min)
            & (nodes_stage[:, 1] <= y_max)
        )
        return nodes_stage[m]


def estimate_lattice(
    cells_xy: np.ndarray,
    period: float,
    *,
    rotation_search_deg: tuple[float, float] = (-45.0, 45.0),
    rotation_steps: int = 901,
    bins_per_axis: int = 64,
) -> LatticeFit:
    """Recover lattice rotation + phase from a noisy 2D point cloud.

    Period is assumed known.  Algorithm: brute-force rotation sweep +
    unit-cell folding.  For each candidate rotation theta, we rotate
    the cell centroids by ``-theta`` so the lattice would become
    axis-aligned, then fold via modulo into a single unit cell
    ``[0, period)^2``.  A correctly rotated lattice produces a tight
    cluster (cells stacking on the same fractional offset), so the
    *peakiness* of the folded distribution is the fit score.

    Square lattices have 4-fold rotational symmetry; the search range
    is therefore bounded to a 90 deg span (default centred on 0).

    Args:
        cells_xy: ``(N, 2)`` cell centroids in stage frame, in um.
        period: Known lattice spacing in um.
        rotation_search_deg: ``(theta_min, theta_max)`` in degrees.
            Default ``(-45, 45)`` covers the full 4-fold-symmetric range.
        rotation_steps: Number of rotation candidates.  Default 901
            gives 0.1 deg resolution over 90 deg, which is finer than
            real plate placement uncertainty.  Drop to ~91 for a quick
            look (1 deg resolution).
        bins_per_axis: Histogram resolution for the folded unit cell.
            64 is a good default for ~100 um period (~1.5 um/bin).

    Returns:
        :class:`LatticeFit` with rotation, phase, and sharpness.

    Raises:
        ValueError: If fewer than 4 cells are supplied -- the fit is
            meaningless below this.
    """
    if len(cells_xy) < 4:
        raise ValueError("Need at least 4 cells to fit a lattice.")
    cells_xy = np.asarray(cells_xy, dtype=float)
    thetas = np.linspace(rotation_search_deg[0], rotation_search_deg[1], rotation_steps)
    bin_edges = np.linspace(0, period, bins_per_axis + 1)

    best_score = -np.inf
    best_theta = 0.0
    best_phase = (0.0, 0.0)
    for theta in thetas:
        rotated = _rotate(cells_xy, -theta)
        folded = np.mod(rotated, period)
        H, _, _ = np.histogram2d(
            folded[:, 0], folded[:, 1], bins=[bin_edges, bin_edges]
        )
        peak = H.max()
        mean = H.mean() if H.mean() > 0 else 1.0
        score = peak / mean
        if score > best_score:
            best_score = score
            best_theta = float(theta)
            iy, ix = np.unravel_index(np.argmax(H), H.shape)
            x_lo, x_hi = bin_edges[iy], bin_edges[iy + 1]
            y_lo, y_hi = bin_edges[ix], bin_edges[ix + 1]
            in_peak = (
                (folded[:, 0] >= x_lo)
                & (folded[:, 0] < x_hi)
                & (folded[:, 1] >= y_lo)
                & (folded[:, 1] < y_hi)
            )
            if in_peak.sum() > 0:
                best_phase = (
                    float(folded[in_peak, 0].mean()),
                    float(folded[in_peak, 1].mean()),
                )
            else:
                best_phase = (
                    float((x_lo + x_hi) / 2),
                    float((y_lo + y_hi) / 2),
                )
    return LatticeFit(
        rotation_deg=best_theta,
        phase_um=best_phase,
        sharpness=float(best_score),
    )


# ----------------------------------------------------------------------
# FOV scoring
# ----------------------------------------------------------------------


@dataclass
class FOVScore:
    """Result of evaluating a single FOV against the fitted lattice."""

    x: float
    y: float
    n_nodes: int
    n_occupied: int
    n_empty: int
    n_doublets: int
    n_off_lattice: int
    n_defects: int  # n_empty + n_doublets + n_off_lattice (raw)
    score: float  # weighted "occupied minus defects"; bigger is better
    skip: bool
    n_feature_failed: int = 0  # number of fov_conditions that failed
    reason: str = ""  # short description of why skip is True (if at all)
    feature_summary: dict[str, float] = field(default_factory=dict)
    """Per-condition diagnostics: ``{feature: actual_fraction_satisfying}``.
    Only populated when ``fov_conditions`` was supplied."""


def score_fov(
    cells_xy: np.ndarray,
    fit: LatticeFit,
    *,
    x: float,
    y: float,
    fov_w: float,
    fov_h: float,
    period: float,
    node_tol_um: float = 30.0,
    w_empty: float = 1.0,
    w_doublet: float = 2.0,
    w_off_lattice: float = 0.5,
    max_doublet_fraction: float = 0.4,
    cell_features: pd.DataFrame | None = None,
    fov_conditions: list[FOVCondition] | None = None,
) -> FOVScore:
    """Score one candidate FOV at stage-frame centre ``(x, y)``.

    Objective: minimise defects.  A defect is anything that breaks the
    "exactly one cell on every grid node" goal -- empty nodes, doublet
    nodes, and cells sitting off the lattice.

    ::

        score = n_occupied
              - w_empty   * n_empty
              - w_doublet * n_doublets
              - w_off_lattice * n_off_lattice

    A defect-free FOV with all ``n_nodes`` nodes occupied scores
    ``n_nodes``, which is the upper bound for a given FOV size.

    Args:
        cells_xy: ``(N, 2)`` candidate cell centroids in stage frame, um.
        fit: Lattice fit (use :func:`estimate_lattice`).
        x, y: FOV centre in stage frame, um.
        fov_w, fov_h: FOV dimensions in um.
        period: Lattice period in um.
        node_tol_um: A cell counts as "on" a node if it falls within
            this radius.  Default ~30 um works for 100 um pitch -- well
            inside the half-period so two nodes can't both claim one
            cell, but big enough to absorb typical jitter (~5 um).
        w_empty, w_doublet, w_off_lattice: Defect weights.  Doublets
            weigh heaviest by default -- they are harder to interpret
            downstream than missing cells.
        max_doublet_fraction: Hard threshold; FOVs whose doublets
            exceed this fraction of (occupied + doublet) nodes get
            their ``skip`` flag set so callers can drop them.
        cell_features: Optional per-cell feature DataFrame, aligned
            row-by-row with ``cells_xy`` (row i describes cell i).  Each
            condition in ``fov_conditions`` is evaluated on the subset
            of rows whose cells fall inside the FOV bbox.  When omitted
            (or ``fov_conditions`` is empty) feature filtering is skipped
            and the score is purely lattice-based.
        fov_conditions: Optional list of :class:`FOVCondition` objects
            (same class as :class:`FOVFinderAgent`).  An FOV that fails
            any condition gets its ``skip`` flag set with a ``reason``
            explaining which feature failed.
    """
    cells_xy = np.asarray(cells_xy, dtype=float)
    x_min, x_max = x - fov_w / 2, x + fov_w / 2
    y_min, y_max = y - fov_h / 2, y + fov_h / 2
    nodes = fit.predict_nodes(x_min, y_min, x_max, y_max, period)

    in_fov = (
        (cells_xy[:, 0] >= x_min)
        & (cells_xy[:, 0] <= x_max)
        & (cells_xy[:, 1] >= y_min)
        & (cells_xy[:, 1] <= y_max)
    )
    cells_in = cells_xy[in_fov]

    if len(nodes) == 0:
        return FOVScore(
            float(x),
            float(y),
            0,
            0,
            0,
            0,
            len(cells_in),
            len(cells_in),
            -np.inf,
            True,
            reason="no_nodes_in_fov",
        )

    if len(cells_in) > 0:
        dx = nodes[:, None, 0] - cells_in[None, :, 0]
        dy = nodes[:, None, 1] - cells_in[None, :, 1]
        dist = np.hypot(dx, dy)
        cells_per_node = (dist <= node_tol_um).sum(axis=1)
        nearest_node_dist = dist.min(axis=0)
        n_off = int((nearest_node_dist > node_tol_um).sum())
    else:
        cells_per_node = np.zeros(len(nodes), dtype=int)
        n_off = 0

    n_occupied = int((cells_per_node == 1).sum())
    n_doublets = int((cells_per_node >= 2).sum())
    n_empty = int((cells_per_node == 0).sum())

    score = (
        n_occupied - w_empty * n_empty - w_doublet * n_doublets - w_off_lattice * n_off
    )

    reason_parts: list[str] = []
    skip = n_doublets / max(1, n_occupied + n_doublets) > max_doublet_fraction
    if skip:
        reason_parts.append(f"doublets:{n_doublets}/{n_occupied + n_doublets}")

    # Per-cell feature condition checks (optional, only run when both
    # cell_features and fov_conditions are supplied).  Slice the
    # feature DataFrame down to rows whose cells lie inside the FOV
    # bbox, then evaluate every condition on that subset.
    n_feature_failed = 0
    feature_summary: dict[str, float] = {}
    if cell_features is not None and fov_conditions:
        in_fov_idx = np.where(in_fov)[0]
        df_in = (
            cell_features.iloc[in_fov_idx]
            if len(in_fov_idx)
            else cell_features.iloc[0:0]
        )
        for cond in fov_conditions:
            passed, fraction = cond.check(df_in)
            feature_summary[cond.feature] = float(fraction)
            if not passed:
                n_feature_failed += 1
                reason_parts.append(
                    f"{cond.feature}_{cond.operator}_{cond.threshold:g}"
                    f"@{fraction:.2f}<{cond.min_fraction:.2f}"
                )
        if n_feature_failed > 0:
            skip = True

    return FOVScore(
        float(x),
        float(y),
        n_nodes=len(nodes),
        n_occupied=n_occupied,
        n_empty=n_empty,
        n_doublets=n_doublets,
        n_off_lattice=n_off,
        n_defects=int(n_empty + n_doublets + n_off),
        score=float(score),
        skip=bool(skip),
        n_feature_failed=int(n_feature_failed),
        reason=" | ".join(reason_parts),
        feature_summary=feature_summary,
    )


def _snap_to_lattice(
    x: float, y: float, fit: LatticeFit, period: float
) -> tuple[float, float]:
    """Snap stage-frame point onto the nearest lattice node."""
    p = _rotate(np.array([[x, y]]), -fit.rotation_deg)[0]
    p_lat = np.round((p - np.array(fit.phase_um)) / period) * period + np.array(
        fit.phase_um
    )
    out = _rotate(p_lat[None, :], fit.rotation_deg)[0]
    return float(out[0]), float(out[1])


def find_best_fovs(
    cells_xy: np.ndarray,
    fit: LatticeFit,
    *,
    fov_w: float,
    fov_h: float,
    period: float,
    well_radius: float,
    candidate_step: float = 50.0,
    snap_to_lattice: bool = True,
    k: int = 5,
    min_separation_um: float | None = None,
    node_tol_um: float = 30.0,
    skip_invalid_fovs: bool = True,
    well_centre: tuple[float, float] = (0.0, 0.0),
    cell_features: pd.DataFrame | None = None,
    fov_conditions: list[FOVCondition] | None = None,
    **score_kwargs,
) -> list[FOVScore]:
    """Sweep the well, score every candidate FOV, return the top-K spaced apart.

    Args:
        cells_xy: Cell centroids in stage frame.
        fit: Lattice fit to score against.
        fov_w, fov_h: FOV dimensions in um.
        period: Lattice period in um.
        well_radius: Used to bound the candidate sweep.  Centre defaults
            to (0, 0); pass ``well_centre`` for off-origin wells.
        candidate_step: Stride of the candidate sweep (um).  Smaller =
            finer sweep, more compute.
        snap_to_lattice: Snap each FOV centre to the nearest node so
            the FOV is phase-aligned -- nuclei sit roughly in cell
            centres rather than being clipped at the FOV boundary.
        k: Number of FOVs to return.
        min_separation_um: Required centre-to-centre spacing between
            picks.  Defaults to ``max(fov_w, fov_h)`` so picks don't
            overlap.
        skip_invalid_fovs: Drop FOVs whose ``skip`` flag is set.  Covers
            both doublet-heavy FOVs and FOVs that failed any
            ``fov_conditions`` feature check.
        well_centre: ``(cx, cy)`` of the well in stage frame, um.
        cell_features: Optional per-cell feature DataFrame (one row per
            cell in ``cells_xy``).  Forwarded to :func:`score_fov`.
        fov_conditions: Optional list of :class:`FOVCondition` objects
            evaluated per FOV on the in-FOV cell subset.
        **score_kwargs: Forwarded to :func:`score_fov` (weights etc).
    """
    if min_separation_um is None:
        min_separation_um = max(fov_w, fov_h)
    cx, cy = well_centre

    xs = np.arange(cx - well_radius, cx + well_radius + candidate_step, candidate_step)
    ys = np.arange(cy - well_radius, cy + well_radius + candidate_step, candidate_step)
    candidates: list[FOVScore] = []
    for x in xs:
        for y in ys:
            if np.hypot(x - cx, y - cy) > well_radius - max(fov_w, fov_h) / 2:
                continue
            s = score_fov(
                cells_xy,
                fit,
                x=x,
                y=y,
                fov_w=fov_w,
                fov_h=fov_h,
                period=period,
                node_tol_um=node_tol_um,
                cell_features=cell_features,
                fov_conditions=fov_conditions,
                **score_kwargs,
            )
            if skip_invalid_fovs and s.skip:
                continue
            if snap_to_lattice:
                snapped = _snap_to_lattice(x, y, fit, period)
                s = score_fov(
                    cells_xy,
                    fit,
                    x=snapped[0],
                    y=snapped[1],
                    fov_w=fov_w,
                    fov_h=fov_h,
                    period=period,
                    node_tol_um=node_tol_um,
                    cell_features=cell_features,
                    fov_conditions=fov_conditions,
                    **score_kwargs,
                )
                if skip_invalid_fovs and s.skip:
                    continue
            candidates.append(s)
    if not candidates:
        return []

    candidates.sort(key=lambda s: s.score, reverse=True)
    picked: list[FOVScore] = []
    for cand in candidates:
        if all(
            np.hypot(cand.x - p.x, cand.y - p.y) >= min_separation_um for p in picked
        ):
            picked.append(cand)
        if len(picked) >= k:
            break
    return picked


# ----------------------------------------------------------------------
# Agent
# ----------------------------------------------------------------------


class MicroPatternedFOVFinderAgent(PreExperimentAgent):
    """FOV finder that exploits a known micropattern grid prior.

    Each :meth:`run` call:

    1. Pops a chunk of ``wells_per_phase`` wells from ``wells``.
    2. For each well, runs a coarse mosaic scan via the microscope --
       a grid of stage moves covering ``mosaic_size_um`` x ``mosaic_size_um``
       around the well centre, with one tile every ``mosaic_step_um``.
       The full mosaic is always scanned; picks are global to the well.
    3. Segments H2B at every tile and projects nuclei centroids back
       into stage frame.
    4. Estimates the lattice rotation + phase (period is fixed at
       ``grid_period_um``) from the centroids, unless
       ``grid_rotation_deg`` was supplied explicitly.
    5. Picks ``fovs_per_well`` low-defect, non-overlapping FOVs via
       :func:`find_best_fovs`.  When ``fov_conditions`` are configured,
       FOVs whose cells fail any condition (e.g. CNR out of range) are
       dropped before picking.
    6. Returns all picked positions as a flat ``list[FovPosition]``.

    This is a skeleton -- the mosaic-scan / centroid extraction step is
    written against the same ``mic.run_mda`` + ``connect_frame`` plumbing
    used by :class:`faro.agents.fov_finder.FOVFinderAgent`, but has not
    yet been exercised against a real microscope.  Use the simulation
    in ``experiments/32_fov_finder/micropattern/sim_micropattern.py`` to
    validate the algorithm path; once that is happy, hook this class
    onto a real ``WellPlatePlan`` and run it on a calibration plate.
    """

    def __init__(
        self,
        microscope: AbstractMicroscope,
        *,
        well_plate_plan: str | "WellPlatePlan",
        wells: list[str],
        wells_per_phase: int | None = None,
        fovs_per_well: int = 3,
        fov_size_um: tuple[float, float],
        grid_period_um: float = 100.0,
        grid_rotation_deg: float | None = None,
        node_tol_um: float = 30.0,
        mosaic_size_um: float = 3000.0,
        mosaic_step_um: float = 400.0,
        candidate_step_um: float = 50.0,
        snap_to_lattice: bool = True,
        skip_invalid_fovs: bool = True,
        w_empty: float = 1.0,
        w_doublet: float = 2.0,
        w_off_lattice: float = 0.5,
        imaging_channels: tuple[Channel, ...],
        segmentator: Segmentator,
        seg_channel_index: int = 0,
        feature_extractor: "FeatureExtractor | None" = None,
        fov_conditions: list[FOVCondition] | None = None,
        z: float | None = None,
        name_prefix: str = "mp",
        verbose: bool = False,
    ):
        super().__init__(microscope)
        if not imaging_channels:
            raise ValueError("imaging_channels must contain at least one channel")
        if fovs_per_well <= 0:
            raise ValueError("fovs_per_well must be positive")
        if fov_size_um[0] <= 0 or fov_size_um[1] <= 0:
            raise ValueError("fov_size_um must be positive")
        if fov_conditions and feature_extractor is None:
            raise ValueError(
                "fov_conditions requires a feature_extractor that produces "
                "the referenced feature columns (e.g. FE_ErkKtr for 'cnr')."
            )

        self.well_plate_plan_input = well_plate_plan
        self.wells_per_phase = (
            len(wells) if wells_per_phase is None else int(wells_per_phase)
        )
        self.fovs_per_well = int(fovs_per_well)
        self.fov_w, self.fov_h = float(fov_size_um[0]), float(fov_size_um[1])
        self.grid_period_um = float(grid_period_um)
        self.grid_rotation_deg = grid_rotation_deg
        self.node_tol_um = float(node_tol_um)
        self.mosaic_size_um = float(mosaic_size_um)
        self.mosaic_step_um = float(mosaic_step_um)
        self.candidate_step_um = float(candidate_step_um)
        self.snap_to_lattice = bool(snap_to_lattice)
        self.skip_invalid_fovs = bool(skip_invalid_fovs)
        self.w_empty = float(w_empty)
        self.w_doublet = float(w_doublet)
        self.w_off_lattice = float(w_off_lattice)
        self.imaging_channels = tuple(imaging_channels)
        self.segmentator = segmentator
        self.seg_channel_index = int(seg_channel_index)
        self.feature_extractor = feature_extractor
        self.fov_conditions: list[FOVCondition] = list(fov_conditions or [])
        self.z = z
        self.name_prefix = name_prefix
        self.verbose = bool(verbose)

        self._plan = self._load_plan(well_plate_plan)
        self._plate: WellPlate = self._plan.plate
        self._a1_center_xy = self._plan.a1_center_xy
        self._rotation = self._plan.rotation
        self._wells_source: list[str] = list(wells)
        self._remaining_wells: list[str] = list(wells)
        self._phase_index = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Calibration loading (same as FOVFinderAgent)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_plan(source: str | "WellPlatePlan") -> "WellPlatePlan":
        from useq import WellPlatePlan

        if isinstance(source, WellPlatePlan):
            return source
        return WellPlatePlan.from_file(str(source))

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _well_center_um(self, well_name: str) -> tuple[float, float]:
        from useq import WellPlatePlan

        from faro.agents.fov_finder import _well_name_to_index

        row, col = _well_name_to_index(well_name)
        single = WellPlatePlan(
            plate=self._plate,
            a1_center_xy=self._a1_center_xy,
            rotation=self._rotation,
            selected_wells=([row], [col]),
        )
        pos = single.selected_well_positions[0]
        return float(pos.x), float(pos.y)

    def _mosaic_positions(
        self, cx: float, cy: float, z_value: float | None
    ) -> list[FovPosition]:
        """Build a square mosaic of stage positions around ``(cx, cy)``."""
        n = int(np.ceil(self.mosaic_size_um / self.mosaic_step_um))
        half = (n - 1) * self.mosaic_step_um / 2.0
        positions: list[FovPosition] = []
        for i in range(n):
            for j in range(n):
                x = cx - half + i * self.mosaic_step_um
                y = cy - half + j * self.mosaic_step_um
                positions.append(
                    FovPosition(
                        x=x,
                        y=y,
                        z=z_value,
                        name=f"{self.name_prefix}_p{self._phase_index}_tile_{i}_{j}",
                    )
                )
        return positions

    # ------------------------------------------------------------------
    # Acquisition + per-tile centroid extraction
    # ------------------------------------------------------------------

    def _acquire_frames(
        self, positions: list[FovPosition]
    ) -> dict[tuple[int, int], np.ndarray]:
        """Run an MDA over *positions* and collect frames keyed by (p, c).

        Lifted from :class:`FOVFinderAgent`; same callback shape.
        """
        seq = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=positions,
            channels=self.imaging_channels,
            rtm_metadata={"micropattern_phase": self._phase_index},
        )
        rtm_events = list(seq)
        mda_events: list = []
        for ev in rtm_events:
            mda_events.extend(
                ev.to_mda_events(
                    resolve_group=self.microscope.resolve_group,
                    resolve_power=self.microscope.resolve_power,
                )
            )

        frames: dict[tuple[int, int], np.ndarray] = {}

        def _on_frame(img, event) -> None:
            p = event.index.get("p", 0)
            c = event.index.get("c", 0)
            frames[(p, c)] = np.asarray(img).copy()

        self.microscope.connect_frame(_on_frame)
        try:
            thread = self.microscope.run_mda(iter(mda_events))
            if thread is not None and hasattr(thread, "join"):
                thread.join()
        finally:
            try:
                self.microscope.disconnect_frame(_on_frame)
            except Exception:
                pass
        return frames

    def _extract_centroids(
        self,
        positions: list[FovPosition],
        frames: dict[tuple[int, int], np.ndarray],
        pixel_size_um: float,
    ) -> tuple[np.ndarray, pd.DataFrame | None]:
        """Segment each tile, project centroids, optionally extract features.

        Each tile reports its centroids in pixel coordinates relative to
        the tile.  We convert pixels to um and add the tile centre to
        get stage-frame coordinates.

        When ``self.feature_extractor`` is configured, also runs
        ``extract_features({mask: label}, img_stack)`` per tile and
        accumulates the per-cell feature DataFrame across the well so
        downstream condition checks (CNR etc.) see one row per cell.
        Cells whose label disappears between segmentation and feature
        extraction (e.g. dropped by a min-area filter) are kept in the
        position array but their feature row is filled with NaN.

        Assumes the camera and stage axes are aligned (no rotation
        between camera and stage).  If your scope has a non-trivial
        camera-to-stage transform this is the place to apply it.

        Returns:
            ``(cells_xy, cell_features)`` -- a ``(N, 2)`` array of
            stage-frame centroids and an optional DataFrame aligned
            row-by-row with it.  ``cell_features`` is ``None`` when no
            feature extractor is configured.
        """
        all_xy: list[np.ndarray] = []
        all_feature_dfs: list[pd.DataFrame] = []
        n_channels = len(self.imaging_channels)

        for p_idx, fp in enumerate(positions):
            seg_img = frames.get((p_idx, self.seg_channel_index))
            if seg_img is None:
                continue
            label = self.segmentator.segment(seg_img)
            if label.size == 0 or label.max() == 0:
                continue
            n_labels = int(label.max())
            ys, xs = np.indices(label.shape)
            h, w = label.shape

            tile_xy: list[np.ndarray] = []
            tile_labels: list[int] = []
            for lab in range(1, n_labels + 1):
                m = label == lab
                if not m.any():
                    continue
                cy = ys[m].mean() * pixel_size_um - h * pixel_size_um / 2
                cx = xs[m].mean() * pixel_size_um - w * pixel_size_um / 2
                tile_xy.append(np.array([fp.x + cx, fp.y + cy]))
                tile_labels.append(lab)
            if not tile_xy:
                continue

            # Optional per-cell feature extraction.  We must run it on
            # the *full* imaging-channel stack so feature extractors
            # that need ratios across channels (e.g. CNR) work.
            tile_features: pd.DataFrame | None = None
            if self.feature_extractor is not None:
                channel_imgs: list[np.ndarray] | None = []
                for c in range(n_channels):
                    cimg = frames.get((p_idx, c))
                    if cimg is None:
                        channel_imgs = None
                        break
                    channel_imgs.append(cimg)
                if channel_imgs is not None:
                    img_stack = np.stack(channel_imgs, axis=0)
                    used_mask = getattr(self.feature_extractor, "used_mask", "labels")
                    try:
                        fe_result = self.feature_extractor.extract_features(
                            {used_mask: label}, img_stack
                        )
                    except Exception as e:  # pragma: no cover - user FE
                        if self.verbose:
                            print(
                                f"[micropattern] feature extraction failed at "
                                f"tile {p_idx}: {type(e).__name__}: {e}"
                            )
                        fe_result = None
                    df_features = (
                        fe_result[0]
                        if isinstance(fe_result, tuple) and fe_result
                        else fe_result
                    )
                    if df_features is not None and not df_features.empty:
                        # Align to tile_labels: keep the rows in label order so
                        # they line up with the centroids we just computed.
                        if "label" in df_features.columns:
                            df_features = (
                                df_features.set_index("label")
                                .reindex(tile_labels)
                                .reset_index()
                            )
                        tile_features = df_features

            if self.feature_extractor is not None and tile_features is None:
                # Centroids computed but features unavailable -- emit a
                # NaN row per cell so the row alignment with cells_xy
                # is preserved.  Conditions evaluated on NaNs will fail
                # naturally (cond.check returns False on empty series).
                tile_features = pd.DataFrame({"label": tile_labels})

            all_xy.extend(tile_xy)
            if tile_features is not None:
                all_feature_dfs.append(tile_features)

        if not all_xy:
            return np.empty((0, 2)), None
        cells_xy = np.stack(all_xy)
        cell_features = (
            pd.concat(all_feature_dfs, ignore_index=True) if all_feature_dfs else None
        )
        return cells_xy, cell_features

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def pick_next_wells(self, n: int | None = None) -> list[str]:
        n = self.wells_per_phase if n is None else int(n)
        if not self._remaining_wells:
            raise RuntimeError("MicroPatternedFOVFinderAgent: out of wells")
        chosen = self._remaining_wells[:n]
        self._remaining_wells = self._remaining_wells[n:]
        return chosen

    def run(self) -> list[FovPosition]:
        """Find micropattern-aligned FOVs for one phase."""
        # Pixel size of the imaging channel (best-effort -- agents that
        # need it should expose it via the microscope; here we let the
        # user override per-call by setting ``self.pixel_size_um``).
        pixel_size_um = getattr(self, "pixel_size_um", None)
        if pixel_size_um is None:
            mmc = getattr(self.microscope, "mmc", None)
            pixel_size_um = float(mmc.getPixelSizeUm()) if mmc is not None else 1.0

        wells = self.pick_next_wells()
        phase = self._phase_index
        z_value = self.microscope.get_focus() if self.z == "current" else self.z

        if self.verbose:
            print(
                f"[MicroPatternedFOVFinderAgent] Phase {phase}: scanning "
                f"{len(wells)} wells: {wells}"
            )

        all_picks: list[FovPosition] = []
        per_well: dict[str, dict[str, Any]] = {}

        for well in wells:
            cx, cy = self._well_center_um(well)
            mosaic = self._mosaic_positions(cx, cy, z_value)
            frames = self._acquire_frames(mosaic)
            cells_xy, cell_features = self._extract_centroids(
                mosaic, frames, pixel_size_um
            )
            if self.verbose:
                feat_note = (
                    f", features {len(cell_features)} rows"
                    if cell_features is not None
                    else ""
                )
                print(
                    f"[MicroPatternedFOVFinderAgent] well {well}: "
                    f"{len(cells_xy)} centroids from {len(mosaic)} tiles"
                    f"{feat_note}"
                )
            if len(cells_xy) < 4:
                if self.verbose:
                    print(
                        f"[MicroPatternedFOVFinderAgent] WARN: well {well} "
                        f"too few centroids; skipping."
                    )
                continue

            if self.grid_rotation_deg is None:
                fit = estimate_lattice(cells_xy, self.grid_period_um)
            else:
                # User-supplied rotation; phase still needs estimating
                # via a 1-step "sweep".
                fit = estimate_lattice(
                    cells_xy,
                    self.grid_period_um,
                    rotation_search_deg=(
                        self.grid_rotation_deg,
                        self.grid_rotation_deg,
                    ),
                    rotation_steps=1,
                )

            picks = find_best_fovs(
                cells_xy,
                fit,
                fov_w=self.fov_w,
                fov_h=self.fov_h,
                period=self.grid_period_um,
                well_radius=self.mosaic_size_um / 2,
                well_centre=(cx, cy),
                candidate_step=self.candidate_step_um,
                snap_to_lattice=self.snap_to_lattice,
                k=self.fovs_per_well,
                node_tol_um=self.node_tol_um,
                skip_invalid_fovs=self.skip_invalid_fovs,
                cell_features=cell_features,
                fov_conditions=self.fov_conditions,
                w_empty=self.w_empty,
                w_doublet=self.w_doublet,
                w_off_lattice=self.w_off_lattice,
            )
            for k, p in enumerate(picks):
                all_picks.append(
                    FovPosition(
                        x=p.x,
                        y=p.y,
                        z=z_value,
                        name=f"{well}_{k:04d}",
                    )
                )
            per_well[well] = {
                "centroids": cells_xy,
                "cell_features": cell_features,
                "fit": fit,
                "picks": picks,
            }
            if self.verbose:
                for p in picks:
                    feat_str = (
                        " "
                        + " ".join(f"{k}={v:.2f}" for k, v in p.feature_summary.items())
                        if p.feature_summary
                        else ""
                    )
                    print(
                        f"  ({p.x:7.1f}, {p.y:7.1f}) score={p.score:.2f} "
                        f"occ={p.n_occupied}/{p.n_nodes} "
                        f"d={p.n_doublets} off={p.n_off_lattice}"
                        f"{feat_str}"
                    )

        self.last_run = {
            "positions": all_picks,
            "per_well": per_well,
            "phase": phase,
        }
        self.history.append({"phase": phase, "n_selected": len(all_picks)})
        self._phase_index += 1
        return all_picks
