"""FOV finder agent: pick good positions inside wells from a plate calibration.

Loads a ``WellPlatePlan`` calibration file (saved by the pymmcore-widgets MDA
plate widget — i.e. plate definition + ``a1_center_xy`` + ``rotation``) and,
for each phase, picks a chunk of wells from a user-supplied ordered list,
generates random candidate positions inside each well (kept ``border_um``
away from the well edge), images them via ``mic.run_mda`` using the user's
imaging channels, segments the result to count cells, and returns a number
of valid positions per well chosen by **farthest-point sampling** so they
are maximally spread out (minimising overlap).

Designed to be plugged into :class:`faro.agents.composed.ComposedAgent`
together with any :class:`InterPhaseAgent` (e.g. :class:`BOptGPAX`) to drive
multi-phase experiments where each phase visits a fresh batch of wells.
"""

from __future__ import annotations

import operator as _operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from faro.agents.base import PreExperimentAgent
from faro.core.data_structures import Channel, RTMSequence
from faro.core.utils import FovPosition

_FOV_OPERATORS: dict[str, Any] = {
    "below": _operator.lt,
    "above": _operator.gt,
    "below_or_equal": _operator.le,
    "above_or_equal": _operator.ge,
    "equal": _operator.eq,
}


@dataclass
class FOVCondition:
    """Condition for filtering FOVs based on per-cell features.

    Evaluates whether a sufficient fraction of cells in an FOV satisfy a
    threshold on a given feature.  This allows feature-aware FOV selection
    beyond simple cell-count filtering — e.g. selecting only FOVs where
    the biosensor signal is in a usable range.

    Requires a :class:`~faro.feature_extraction.base.FeatureExtractor`
    that produces the referenced feature column (e.g.
    :class:`~faro.feature_extraction.erk_ktr.FE_ErkKtr` for ``"cnr"``).

    Args:
        feature: Column name in the per-cell feature table produced by
            ``FeatureExtractor.extract_features()`` (e.g. ``"cnr"``).
        operator: Comparison — ``"below"``, ``"above"``,
            ``"below_or_equal"``, ``"above_or_equal"``, ``"equal"``.
        threshold: Value to compare against.
        min_fraction: Minimum fraction of cells that must satisfy the
            condition for the FOV to be accepted.  Default ``1.0``
            (every cell must satisfy).

    Example::

        # Accept FOVs where at least 80 % of cells have CNR < 0.6
        FOVCondition("cnr", "below", 0.6, min_fraction=0.8)
    """

    feature: str
    operator: str
    threshold: float
    min_fraction: float = 1.0

    def __post_init__(self) -> None:
        if self.operator not in _FOV_OPERATORS:
            raise ValueError(
                f"Unknown operator {self.operator!r}; "
                f"choose from {list(_FOV_OPERATORS)}"
            )
        if not 0.0 <= self.min_fraction <= 1.0:
            raise ValueError(f"min_fraction must be in [0, 1], got {self.min_fraction}")

    def check(self, df_cells: pd.DataFrame) -> tuple[bool, float]:
        """Check whether the condition is met on per-cell data.

        Args:
            df_cells: DataFrame with one row per cell, as returned by
                ``FeatureExtractor.extract_features()``.

        Returns:
            ``(passed, actual_fraction)`` — whether the condition passed
            and the actual fraction of cells that satisfied it.
        """
        if df_cells.empty or self.feature not in df_cells.columns:
            return False, 0.0
        series = df_cells[self.feature].dropna()
        if series.empty:
            return False, 0.0
        op_fn = _FOV_OPERATORS[self.operator]
        n_satisfying = op_fn(series, self.threshold).sum()
        fraction = float(n_satisfying) / len(series)
        return fraction >= self.min_fraction, fraction


if TYPE_CHECKING:
    from useq import WellPlate, WellPlatePlan

    from faro.feature_extraction.base import FeatureExtractor
    from faro.microscope.base import AbstractMicroscope
    from faro.segmentation.base import Segmentator


_WELL_NAME_RE = re.compile(r"^\s*([A-Za-z]+)\s*(\d+)\s*$")


def _well_name_to_index(well_name: str) -> tuple[int, int]:
    """Convert a well name like ``"B2"`` to a zero-indexed ``(row, col)``.

    Supports multi-letter rows (``"AA1"``) for plates with > 26 rows.
    """
    m = _WELL_NAME_RE.match(well_name)
    if not m:
        raise ValueError(
            f"Invalid well name {well_name!r}; expected e.g. 'B2', 'AA12'."
        )
    letters, digits = m.groups()
    row = 0
    for ch in letters.upper():
        row = row * 26 + (ord(ch) - ord("A") + 1)
    return row - 1, int(digits) - 1


def _farthest_point_sampling(
    points: np.ndarray, k: int, *, seed: int | None = None
) -> list[int]:
    """Greedy max-min farthest-point sampling.

    Args:
        points: ``(N, 2)`` array of candidate (x, y) positions in µm.
        k: Number of points to select. If ``k >= N`` returns all indices.
        seed: Optional RNG seed for reproducible first-point selection.

    Returns:
        List of indices into *points* in selection order.
    """
    n = len(points)
    if k <= 0 or n == 0:
        return []
    if k >= n:
        return list(range(n))
    rng = np.random.default_rng(seed)
    first = int(rng.integers(n))
    selected = [first]
    dist = np.linalg.norm(points - points[first], axis=1)
    for _ in range(1, k):
        next_idx = int(np.argmax(dist))
        selected.append(next_idx)
        new_dist = np.linalg.norm(points - points[next_idx], axis=1)
        dist = np.minimum(dist, new_dist)
    return selected


def _extreme_sampling(
    values: np.ndarray,
    k: int,
    *,
    points: np.ndarray | None = None,
) -> list[int]:
    """Pick ``k`` indices spanning the extremes of ``values``.

    Primary criterion: feature value at evenly-spaced quantile targets
    along the sorted axis, so ``k=2`` picks the min and max, ``k=3``
    picks min/median/max, etc.  Secondary criterion: when ``points`` is
    supplied, feature-value ties and near-ties are broken in favour of
    candidates **farthest** from already-picked FOVs (max-min spatial
    distance).  This preserves the "extremes first, spread second"
    contract so that e.g. two candidates with equal cell counts don't
    end up clustered in one corner of the well.

    Examples:
        ``values=[36, 135, 136, 200]``, ``k=2`` -> indices of ``36`` and
        ``200`` (unique values, no tiebreak needed).
        ``values=[40, 40, 200, 200]``, ``k=2`` with ``points`` -> the
        ``40`` and ``200`` candidates that are furthest apart in (x, y).

    Args:
        values: ``(N,)`` scalar feature values (e.g. cell counts).
        k: Number of points to select. If ``k >= N`` returns all indices
            in feature-sorted order.
        points: Optional ``(N, 2)`` spatial coordinates.  When given,
            ties on the feature axis are broken by maximising minimum
            spatial distance to previously-picked FOVs.  When omitted
            the selector falls back to rank-based picking with stable
            tiebreaking.

    Returns:
        List of indices into *values*, ordered by selection (first is
        the min-feature slot, last is the max-feature slot).
    """
    n = len(values)
    if k <= 0 or n == 0:
        return []
    vals = np.asarray(values)
    order = np.argsort(vals, kind="stable")
    if k >= n:
        return [int(i) for i in order]
    if k == 1:
        return [int(order[n // 2])]

    target_ranks = np.linspace(0, n - 1, k).round().astype(int)

    if points is None:
        # Legacy fast path: pick directly at the integer target ranks,
        # deduping defensively (rounding can collide for small n).
        seen: set[int] = set()
        out: list[int] = []
        for rank in target_ranks:
            idx = int(order[int(rank)])
            if idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    pts = np.asarray(points, dtype=float)
    target_values = vals[order][target_ranks]

    picked: list[int] = []
    picked_set: set[int] = set()
    for target_val in target_values:
        unpicked = [i for i in range(n) if i not in picked_set]
        if not unpicked:
            break
        unpicked_arr = np.asarray(unpicked, dtype=int)
        feat_dist = np.abs(vals[unpicked_arr] - target_val)
        if picked:
            picked_pts = pts[picked]
            # Max-min distance: each candidate scored by its nearest
            # already-picked neighbour; we then maximise that score.
            diffs = pts[unpicked_arr][:, None, :] - picked_pts[None, :, :]
            spatial_dist = np.linalg.norm(diffs, axis=-1).min(axis=1)
        else:
            spatial_dist = np.zeros(len(unpicked_arr))
        # Lexsort's last key is primary — feature-distance wins, ties
        # broken by *maximising* spatial distance (negated for argsort).
        rank = np.lexsort((-spatial_dist, feat_dist))
        best = int(unpicked_arr[int(rank[0])])
        picked.append(best)
        picked_set.add(best)
    return picked


class FOVFinderAgent(PreExperimentAgent):
    """Pick good FOV positions inside wells using a saved plate calibration.

    Each call to :meth:`run` consumes ``wells_per_phase`` wells from the
    user-supplied ordered ``wells`` list, generates
    ``n_candidates_per_well`` random candidate positions inside each well
    (clipped by ``border_um`` to keep them away from the well edge), images
    them via ``mic.run_mda`` with the user's ``imaging_channels``, segments
    each image, filters out positions with ``< min_cells`` cells, and
    returns ``fovs_per_well`` valid positions per well selected by
    farthest-point sampling so the FOVs do not overlap.

    Designed to be plugged into :class:`faro.agents.composed.ComposedAgent`
    so that an inter-phase agent (e.g. Bayesian Optimisation) gets a fresh
    batch of FOVs each phase.

    Args:
        microscope: Microscope instance.  ``microscope.run_mda``,
            ``connect_frame``, ``disconnect_frame``, and ``mmc`` are used.
        well_plate_plan: A ``useq.WellPlatePlan`` instance OR a path to a
            JSON file produced by the pymmcore-widgets MDA plate widget
            (it stores ``plate``, ``a1_center_xy``, ``rotation``).
        wells: Ordered list of well names (e.g. ``["B2", "B3", ...]``)
            consumed in chunks of ``wells_per_phase`` per call to
            :meth:`run`.
        wells_per_phase: How many wells to consume per :meth:`run` call.
            ``None`` (default) consumes the entire ``wells`` queue in
            one phase — the right choice when you don't care about
            multi-phase scheduling and just want all FOVs at once.
        fovs_per_well: How many valid FOV positions to return per well
            (after cell-count filtering and farthest-point selection).
        n_candidates_per_well: How many random candidate positions to scan
            per well.  Should be larger than *fovs_per_well* to give the
            farthest-point selection something to choose from.
        border_um: Minimum distance (µm) to keep candidate positions away
            from the well edge.  Random candidates are generated inside a
            shape ``(well_size_x - 2*border_um, well_size_y - 2*border_um)``.
        min_distance_um: Minimum centre-to-centre spacing (µm) between
            random candidates within the same well.  Implemented by
            passing ``fov_width = fov_height = min_distance_um`` and
            ``allow_overlap=False`` to :class:`useq.RandomPoints`, which
            then rejection-samples until non-overlapping FOV bounding
            boxes can be placed.  Default ``1500`` µm — large enough to
            give a visibly spread scan across a 96-well (6.4 mm) well
            even when ``n_candidates_per_well`` is small.  Set to ``0``
            to disable the constraint (overlap allowed, fastest), or
            lower it for smaller wells / denser scans.
        min_cells: Minimum cell count for a candidate to be considered
            valid.  Positions with fewer cells are dropped before
            farthest-point sampling.
        max_cells: Optional upper bound on cell count.  When set,
            candidates with ``n_cells > max_cells`` are also marked
            invalid (useful to skip overly confluent or
            segmentation-artefact fields).  ``None`` (default) imposes
            no upper limit.
        imaging_channels: Tuple of :class:`Channel` /
            :class:`PowerChannel` to acquire at each candidate position.
            Stim and reference channels are intentionally not used here.
        segmentator: A :class:`Segmentator` used to count cells.
        seg_channel_index: Which imaging channel to segment.  Defaults to
            ``0`` (first channel).
        feature_extractor: Optional :class:`FeatureExtractor`.  If
            provided, ``extract_positions({"labels": label_image})`` is
            called per FOV and the resulting DataFrame is attached to the
            ``"all_candidates"`` output for downstream inspection.  This
            does not affect selection unless *fov_conditions* is also
            set — in which case
            ``extract_features(labels, img_stack)`` is called instead
            to produce per-cell features (e.g. CNR) that the conditions
            can filter on.
        fov_conditions: Optional list of :class:`FOVCondition` objects
            applied per FOV to filter by per-cell features.  An FOV is
            accepted only if it passes the cell-count check **and**
            every condition in the list.  Requires a
            ``feature_extractor`` that produces the referenced columns.
            The per-cell feature table produced for each FOV is stored
            on ``self.last_run["fov_features"]`` for inspection (keyed
            by candidate index ``p``).  Example::

                fov_conditions=[
                    FOVCondition("cnr", "below", 0.6, min_fraction=0.8),
                ]
        selection_mode: How to pick ``fovs_per_well`` positions from the
            valid candidates of each well:

            * ``"farthest_point"`` (default) — greedy max-min sampling on
                (x, y) so the picked FOVs are maximally spread in space.
                Use this when you want low spatial overlap and don't care
                which cells you land on.
            * ``"extremes"`` — sort the valid candidates by
                ``selection_feature`` and pick ``fovs_per_well`` evenly-
                spaced positions spanning the min and max.  For
                ``fovs_per_well=2`` this returns the FOVs with the
                lowest and highest feature value; for ``=3`` it adds the
                median.  Useful for picking a contrast pair (e.g. the
                sparsest and densest FOV inside the valid cell-count
                window) instead of a spatially spread set.  Ties on the
                feature axis are broken by **spatial distance** —
                candidates farther from already-picked FOVs win, so two
                equally-extreme candidates won't end up clustered.
        selection_feature: Column name in the per-candidate scan
            DataFrame used as the axis for ``selection_mode="extremes"``.
            Defaults to ``"n_cells"``.  Any column produced by the scan
            works (e.g. ``"fe_cnr_mean"`` when a ``feature_extractor`` is
            configured).  Ignored in ``"farthest_point"`` mode.
        z: Z handling for the candidate positions.  Three modes:

            * ``None`` (default) — the agent does **not** drive focus.
              Candidate :class:`FovPosition`\\ s carry ``z=None``, so the
              resulting useq ``MDAEvent``\\ s have ``z_pos=None`` and the
              microscope engine leaves the Z stage untouched.  This is
              the right mode whenever focus is held by the user
              (pre-focused before starting) or by hardware autofocus /
              PFS — e.g. on the Jungfrau microscope where
              ``USE_ONLY_PFS = True``.
            * ``"current"`` — snapshot the current focus once at the
              start of each phase via
              :meth:`AbstractMicroscope.get_focus` and pin every
              candidate in that phase to it.  Use this on stages without
              PFS when you want explicit, repeatable Z values written
              into the events without manually typing one in.
            * ``float`` — pin every candidate to this absolute Z (µm).
              The agent will then actively command the Z stage to this
              value via the standard useq event path.
        random_seed: Optional RNG seed.  Each phase derives a deterministic
            sub-seed from this so candidate generation and farthest-point
            tie-breaking are reproducible across runs.
        name_prefix: Prefix used to name returned :class:`FovPosition`
            tuples (``"<prefix>_<phase>_<wellname>_<i>"``).
        return_json: Controls the return type of :meth:`run`.

            * ``None`` (default) — return the canonical
              ``list[FovPosition]`` (plain namedtuples; matches what
              :func:`faro.core.utils.generate_fov_positions` produces).
            * ``True`` — return ``list[useq.Position]`` instead.  These
              are pydantic models with the full useq schema (``x``, ``y``,
              ``z``, ``name``, ``sequence``, ``properties``,
              ``plate_row``, ``plate_col``, ``grid_row``, ``grid_col``)
              and serialise directly to the JSON shape that ``useq``-aware
              tools (e.g. the pymmcore-widgets MDA position list) expect.
        verbose: If ``True``, every :meth:`run` call shows debug plots
            via matplotlib: one ``raw | segmentation`` figure per scanned
            candidate (title carries the position name and detected cell
            count) and a final per-well scatter showing the candidates
            colour-coded by cell count with the FPS-selected positions
            highlighted.  Off by default.
    """

    def __init__(
        self,
        microscope: AbstractMicroscope,
        *,
        well_plate_plan: str | Path | "WellPlatePlan",
        wells: list[str],
        wells_per_phase: int | None = None,
        fovs_per_well: int,
        n_candidates_per_well: int,
        border_um: float,
        min_distance_um: float = 1500.0,
        min_cells: int,
        max_cells: int | None = None,
        imaging_channels: tuple[Channel, ...],
        segmentator: Segmentator,
        seg_channel_index: int = 0,
        feature_extractor: FeatureExtractor | None = None,
        fov_conditions: list[FOVCondition] | None = None,
        selection_mode: Literal["farthest_point", "extremes"] = "farthest_point",
        selection_feature: str = "n_cells",
        z: float | None | Literal["current"] = None,
        random_seed: int | None = None,
        name_prefix: str = "fov",
        strict_count: bool = True,
        return_json: bool | None = None,
        verbose: bool = False,
    ):
        """See class docstring.

        Args:
            strict_count: If ``True`` (default), guarantee exactly
                ``wells_per_phase * fovs_per_well`` positions per phase by
                padding wells whose valid candidates fall short with the
                highest-cell-count candidates from that well (even if they
                are below ``min_cells``).  Required when downstream agents
                expect a fixed FOV count per phase (e.g. ``OscillationBO``
                splits the FOVs into ``n_conditions_per_iter`` equal
                groups).  If ``False`` the run() result may contain fewer
                positions than the nominal target — the user is expected
                to handle that.
        """
        super().__init__(microscope)
        # Resolve wells_per_phase: None -> consume the entire queue.
        if wells_per_phase is None:
            wells_per_phase = len(wells)
        if wells_per_phase <= 0:
            raise ValueError("wells_per_phase must be positive")
        if max_cells is not None and max_cells < min_cells:
            raise ValueError(
                f"max_cells ({max_cells}) must be >= min_cells ({min_cells})"
            )
        if fovs_per_well <= 0:
            raise ValueError("fovs_per_well must be positive")
        if n_candidates_per_well < fovs_per_well:
            raise ValueError(
                f"n_candidates_per_well ({n_candidates_per_well}) must be "
                f">= fovs_per_well ({fovs_per_well})"
            )
        if border_um < 0:
            raise ValueError("border_um must be non-negative")
        if min_distance_um < 0:
            raise ValueError("min_distance_um must be non-negative")
        if not imaging_channels:
            raise ValueError("imaging_channels must contain at least one channel")
        if not (z is None or z == "current" or isinstance(z, (int, float))):
            raise ValueError(f"z must be None, 'current', or a float; got {z!r}")
        if fov_conditions and feature_extractor is None:
            raise ValueError(
                "fov_conditions requires a feature_extractor that produces "
                "the referenced feature columns (e.g. FE_ErkKtr for 'cnr')."
            )
        if selection_mode not in ("farthest_point", "extremes"):
            raise ValueError(
                f"selection_mode must be 'farthest_point' or 'extremes'; "
                f"got {selection_mode!r}"
            )

        self.well_plate_plan_input = well_plate_plan
        self.wells_per_phase = int(wells_per_phase)
        self.fovs_per_well = int(fovs_per_well)
        self.n_candidates_per_well = int(n_candidates_per_well)
        self.border_um = float(border_um)
        self.min_distance_um = float(min_distance_um)
        self.min_cells = int(min_cells)
        self.max_cells = None if max_cells is None else int(max_cells)
        self.imaging_channels = tuple(imaging_channels)
        self.segmentator = segmentator
        self.seg_channel_index = int(seg_channel_index)
        self.feature_extractor = feature_extractor
        self.fov_conditions: list[FOVCondition] = list(fov_conditions or [])
        self.selection_mode = selection_mode
        self.selection_feature = str(selection_feature)
        self.z = z
        self.random_seed = random_seed
        self.name_prefix = name_prefix
        self.strict_count = bool(strict_count)
        self.return_json = bool(return_json) if return_json is not None else False
        self.verbose = bool(verbose)

        self._plan = self._load_plan(well_plate_plan)
        self._plate: WellPlate = self._plan.plate
        self._a1_center_xy = self._plan.a1_center_xy
        self._rotation = self._plan.rotation

        # Validate well names up front so we fail fast.
        for w in wells:
            r, c = _well_name_to_index(w)
            if r >= self._plate.rows or c >= self._plate.columns:
                raise ValueError(
                    f"Well {w!r} is outside the plate "
                    f"({self._plate.rows} rows x {self._plate.columns} cols)"
                )
        self._remaining_wells: list[str] = list(wells)

        self._phase_index = 0
        self.history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Calibration loading
    # ------------------------------------------------------------------

    @staticmethod
    def _load_plan(source: str | Path | "WellPlatePlan") -> "WellPlatePlan":
        """Accept a ``WellPlatePlan`` instance or a path to a JSON file."""
        from useq import WellPlatePlan

        if isinstance(source, WellPlatePlan):
            return source
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Plate calibration file not found: {path}")
        return WellPlatePlan.from_file(str(path))

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _well_center_um(self, well_name: str) -> tuple[float, float]:
        """Stage coordinates (µm) of the centre of *well_name*."""
        from useq import WellPlatePlan

        row, col = _well_name_to_index(well_name)
        single = WellPlatePlan(
            plate=self._plate,
            a1_center_xy=self._a1_center_xy,
            rotation=self._rotation,
            selected_wells=([row], [col]),
        )
        pos = single.selected_well_positions[0]
        return float(pos.x), float(pos.y)

    def _generate_candidate_offsets(self, n_points: int, *, seed: int) -> np.ndarray:
        """Generate ``n_points`` random offsets relative to the well centre.

        Uses :class:`useq.RandomPoints` so we get the same shape semantics
        (ELLIPSE for circular wells, RECTANGLE otherwise) as the plate
        widget itself.
        """
        from useq import RandomPoints, Shape

        well_size_x_um = float(self._plate.well_size[0]) * 1000.0
        well_size_y_um = float(self._plate.well_size[1]) * 1000.0
        max_w = max(well_size_x_um - 2.0 * self.border_um, 1.0)
        max_h = max(well_size_y_um - 2.0 * self.border_um, 1.0)
        # If the user requested a minimum spacing, use useq's non-overlap
        # rejection sampler with FOV-size = min_distance.  Otherwise let
        # candidates fall freely (faster, but may bunch up).
        rp_kwargs: dict[str, Any] = dict(
            num_points=n_points,
            max_width=max_w,
            max_height=max_h,
            shape=Shape.ELLIPSE if self._plate.circular_wells else Shape.RECTANGLE,
            random_seed=seed,
        )
        if self.min_distance_um > 0:
            rp_kwargs["fov_width"] = self.min_distance_um
            rp_kwargs["fov_height"] = self.min_distance_um
            rp_kwargs["allow_overlap"] = False
        else:
            rp_kwargs["allow_overlap"] = True
        rp = RandomPoints(**rp_kwargs)
        return np.array([(p.x, p.y) for p in rp], dtype=float)

    # ------------------------------------------------------------------
    # Well selection
    # ------------------------------------------------------------------

    def pick_next_wells(self, n: int | None = None) -> list[str]:
        """Pop and return the next chunk of wells from the queue.

        Args:
            n: How many wells to pop.  Defaults to ``self.wells_per_phase``.

        Raises:
            RuntimeError: If the well queue is empty.
        """
        n = self.wells_per_phase if n is None else int(n)
        if not self._remaining_wells:
            raise RuntimeError("FOVFinderAgent: out of wells")
        chosen = self._remaining_wells[:n]
        self._remaining_wells = self._remaining_wells[n:]
        return chosen

    @property
    def remaining_wells(self) -> list[str]:
        """Wells that have not yet been consumed."""
        return list(self._remaining_wells)

    @property
    def n_remaining_phases(self) -> int:
        """How many full phases worth of wells are still available."""
        return len(self._remaining_wells) // self.wells_per_phase

    # ------------------------------------------------------------------
    # Acquisition + segmentation
    # ------------------------------------------------------------------

    def _acquire_frames(
        self, positions: list[FovPosition]
    ) -> dict[tuple[int, int], np.ndarray]:
        """Image *positions* with :attr:`imaging_channels`.

        Builds a single-timepoint :class:`RTMSequence`, runs it through
        ``mic.run_mda``, and collects the resulting frames into a
        ``{(p, c): img}`` dict via a ``frameReady`` callback.
        """
        seq = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=positions,
            channels=self.imaging_channels,
            rtm_metadata={"fov_finder_phase": self._phase_index},
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

    def _segment_and_score(
        self,
        positions: list[FovPosition],
        frames: dict[tuple[int, int], np.ndarray],
    ) -> pd.DataFrame:
        """Run segmentation per FOV and return a DataFrame of cell counts.

        Optionally also runs the configured ``feature_extractor`` per FOV.
        When :attr:`fov_conditions` is set, calls the extractor's
        ``extract_features`` (instead of ``extract_positions``) to obtain
        per-cell features (e.g. CNR), evaluates every condition, and marks
        the FOV invalid if any condition fails.  The per-cell feature
        DataFrames are stashed on ``self._last_fov_features`` keyed by ``p``.
        """
        n_channels = len(self.imaging_channels)
        rows: list[dict[str, Any]] = []
        # Reset per-call storage of per-cell feature tables (one per FOV
        # that actually had extract_features called on it).
        self._last_fov_features: dict[int, pd.DataFrame] = {}

        for p_idx, fp in enumerate(positions):
            channel_imgs = []
            for c in range(n_channels):
                img = frames.get((p_idx, c))
                if img is None:
                    channel_imgs = None
                    break
                channel_imgs.append(img)
            if channel_imgs is None:
                rows.append(
                    {
                        "p": p_idx,
                        "x": fp.x,
                        "y": fp.y,
                        "z": fp.z,
                        "n_cells": 0,
                        "valid": False,
                        "reason": "missing_frame",
                    }
                )
                continue

            img_stack = np.stack(channel_imgs, axis=0)  # (C, H, W)
            seg_input = img_stack[self.seg_channel_index]
            label_img = self.segmentator.segment(seg_input)
            label_max = int(label_img.max()) if label_img.size > 0 else 0
            n_cells = label_max if label_max > 0 else 0

            if self.verbose:
                self._debug_show_candidate(fp, seg_input, label_img, n_cells)

            below = n_cells < self.min_cells
            above = self.max_cells is not None and n_cells > self.max_cells
            valid = not (below or above)
            if below:
                reason = "below_min_cells"
            elif above:
                reason = "above_max_cells"
            else:
                reason = ""

            row: dict[str, Any] = {
                "p": p_idx,
                "x": fp.x,
                "y": fp.y,
                "z": fp.z,
                # "well" is overwritten in run() from the parallel
                # candidate_well list (the source of truth).
                "n_cells": n_cells,
                "valid": valid,
                "reason": reason,
            }

            if self.feature_extractor is not None:
                # Two paths:
                #   * fov_conditions set -> call extract_features() so we
                #     have per-cell features (e.g. CNR) to filter on.
                #   * no fov_conditions  -> keep the original cheap
                #     extract_positions() path (label, x, y aggregates).
                if self.fov_conditions and valid:
                    used_mask = getattr(self.feature_extractor, "used_mask", "labels")
                    try:
                        fe_result = self.feature_extractor.extract_features(
                            {used_mask: label_img}, img_stack
                        )
                    except Exception as e:  # pragma: no cover - user-supplied FE
                        print(
                            f"[FOVFinderAgent] extract_features failed at FOV "
                            f"{p_idx}: {type(e).__name__}: {e}"
                        )
                        fe_result = None

                    # extract_features may return (df, extra_masks) or just df.
                    if isinstance(fe_result, tuple) and fe_result:
                        df_features = fe_result[0]
                    else:
                        df_features = fe_result

                    if df_features is None or df_features.empty:
                        valid = False
                        reason = "feature_extraction_empty"
                    else:
                        self._last_fov_features[p_idx] = df_features
                        row["n_cells_features"] = int(len(df_features))
                        # Summary stats for inspection
                        for col in df_features.columns:
                            if col == "label":
                                continue
                            try:
                                row[f"fe_{col}_mean"] = float(df_features[col].mean())
                                row[f"fe_{col}_median"] = float(
                                    df_features[col].median()
                                )
                            except (TypeError, ValueError):
                                pass
                        # Evaluate every FOV condition; first failure wins.
                        for cond in self.fov_conditions:
                            passed, fraction = cond.check(df_features)
                            row[f"cond_{cond.feature}_{cond.operator}_frac"] = fraction
                            if not passed:
                                valid = False
                                reason = (
                                    f"condition_failed:{cond.feature}"
                                    f"_{cond.operator}_{cond.threshold}"
                                    f"@{cond.min_fraction:.2f}"
                                )
                                break
                    # Update the row's valid/reason (they may have changed).
                    row["valid"] = valid
                    row["reason"] = reason
                else:
                    try:
                        df_features = self.feature_extractor.extract_positions(
                            {"labels": label_img}
                        )
                        if df_features is not None and not df_features.empty:
                            for col in df_features.columns:
                                if col == "label":
                                    continue
                                try:
                                    row[f"fe_{col}_mean"] = float(
                                        df_features[col].mean()
                                    )
                                except (TypeError, ValueError):
                                    pass
                    except Exception as e:  # pragma: no cover - user-supplied FE
                        print(
                            f"[FOVFinderAgent] feature_extractor failed at FOV "
                            f"{p_idx}: {type(e).__name__}: {e}"
                        )

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> list[FovPosition] | list[Any]:
        """Find FOV positions for one phase.

        Returns:
            By default a ``list[FovPosition]`` with the FPS-selected
            positions (length ``wells_per_phase * fovs_per_well`` when
            all wells yielded enough valid candidates).  Each position
            name is ``"<well>_<i:04d>"`` (e.g. ``"B3_0000"``) and ``z``
            is left as ``None`` when the agent is not driving focus
            (PFS-friendly default).

            If the agent was constructed with ``return_json=True`` the
            same positions are returned as ``list[useq.Position]``
            instead, which serialise directly to the JSON shape that
            ``useq``-aware tools expect.

            Per-phase debug data — wells consumed, the full
            ``all_candidates`` DataFrame, and the parallel
            ``wells_for_positions`` list — is stashed on
            :attr:`last_run` after every call (overwritten each phase).
        """
        wells = self.pick_next_wells()
        phase = self._phase_index
        if self.verbose:
            print(
                f"[FOVFinderAgent] Phase {phase}: scanning "
                f"{self.n_candidates_per_well} candidates in "
                f"{len(wells)} wells: {wells}"
            )

        # Resolve the Z value to write into this phase's candidate events.
        #   - None     -> leave Z untouched (PFS / pre-focused; no z_pos)
        #   - "current"-> snapshot focus once via the mic-agnostic accessor
        #   - float    -> pin every candidate to this absolute Z
        if self.z == "current":
            z_value = self.microscope.get_focus()
        else:
            z_value = self.z

        # 1. Generate candidates for every well.
        all_candidates: list[FovPosition] = []
        candidate_well: list[str] = []
        for w_idx, well in enumerate(wells):
            cx, cy = self._well_center_um(well)
            seed_offset = (
                None
                if self.random_seed is None
                else self.random_seed + 1000 * phase + w_idx
            )
            offsets = self._generate_candidate_offsets(
                self.n_candidates_per_well, seed=seed_offset
            )
            for i, (dx, dy) in enumerate(offsets):
                name = f"{self.name_prefix}_p{phase}_{well}__cand{i:03d}"
                all_candidates.append(
                    FovPosition(x=cx + dx, y=cy + dy, z=z_value, name=name)
                )
                candidate_well.append(well)

        # 2. Acquire and segment.
        frames = self._acquire_frames(all_candidates)
        df_scan = self._segment_and_score(all_candidates, frames)
        df_scan["well"] = candidate_well

        # 3. For each well, farthest-point pick fovs_per_well valid points.
        #    Track wells parallel to selected positions so callers don't
        #    have to parse position names.
        selected: list[FovPosition] = []
        wells_for_positions: list[str] = []

        for w_idx, well in enumerate(wells):
            df_w = df_scan[df_scan["well"] == well]
            df_valid = df_w[df_w["valid"]]

            n_valid = len(df_valid)
            if n_valid == 0 and not self.strict_count:
                if self.verbose:
                    print(
                        f"[FOVFinderAgent] WARNING: well {well} has no valid "
                        f"positions (need >= {self.min_cells} cells); skipping."
                    )
                continue

            if n_valid > 0:
                if self.selection_mode == "extremes":
                    if self.selection_feature not in df_valid.columns:
                        raise RuntimeError(
                            f"selection_feature {self.selection_feature!r} not "
                            f"found in scan columns {list(df_valid.columns)}. "
                            f"When using selection_mode='extremes' with a "
                            f"feature-extractor column, make sure the "
                            f"feature_extractor produces it."
                        )
                    values = df_valid[self.selection_feature].to_numpy(dtype=float)
                    xy = df_valid[["x", "y"]].to_numpy(dtype=float)
                    picked_local = _extreme_sampling(
                        values, self.fovs_per_well, points=xy
                    )
                else:
                    xy = df_valid[["x", "y"]].to_numpy()
                    seed_offset = (
                        None
                        if self.random_seed is None
                        else self.random_seed + 7919 * phase + w_idx
                    )
                    picked_local = _farthest_point_sampling(
                        xy, self.fovs_per_well, seed=seed_offset
                    )
                df_picked = df_valid.iloc[picked_local]
            else:
                df_picked = df_w.iloc[0:0]  # empty frame with same columns

            n_picked = len(df_picked)

            # Pad with the highest-cell-count *invalid* candidates so the
            # phase has exactly fovs_per_well positions per well — required
            # by downstream agents (e.g. OscillationBO) that split the FOVs
            # into equal-size condition groups.
            if n_picked < self.fovs_per_well and self.strict_count:
                needed = self.fovs_per_well - n_picked
                df_unused = df_w[~df_w.index.isin(df_picked.index)]
                df_pad = df_unused.nlargest(needed, "n_cells")
                if len(df_pad) > 0:
                    df_picked = pd.concat([df_picked, df_pad], ignore_index=False)
                    if self.verbose:
                        print(
                            f"[FOVFinderAgent] WARNING: well {well} only had "
                            f"{n_picked}/{self.fovs_per_well} valid positions; "
                            f"padded with {len(df_pad)} below-threshold "
                            f"candidates (best n_cells={int(df_pad['n_cells'].max())})."
                        )
                elif self.verbose:
                    print(
                        f"[FOVFinderAgent] WARNING: well {well} has no "
                        f"candidates at all to pad with; phase will have "
                        f"fewer FOVs than expected."
                    )

            if not self.strict_count and n_picked < self.fovs_per_well and self.verbose:
                print(
                    f"[FOVFinderAgent] WARNING: well {well} only yielded "
                    f"{n_picked}/{self.fovs_per_well} valid positions."
                )

            for k, (_, row) in enumerate(df_picked.iterrows()):
                name = f"{well}_{k:04d}"
                selected.append(
                    FovPosition(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=z_value,
                        name=name,
                    )
                )
                wells_for_positions.append(well)

        if self.verbose:
            self._debug_show_well_summary(
                wells, df_scan, selected, wells_for_positions, phase
            )

        # Stash phase-level debug data so notebooks / inspectors can
        # still reach the wells, the candidate dataframe, etc. without
        # cluttering the public return value.
        self.last_run: dict[str, Any] = {
            "positions": selected,
            "wells_for_positions": wells_for_positions,
            "wells_used": list(wells),
            "all_candidates": df_scan,
            "phase": phase,
            # Per-cell feature DataFrames keyed by candidate index p.
            # Empty when fov_conditions is not set.
            "fov_features": getattr(self, "_last_fov_features", {}),
        }
        self.history.append(
            {
                "phase": phase,
                "wells_used": list(wells),
                "n_selected": len(selected),
                "n_candidates": len(all_candidates),
            }
        )
        self._phase_index += 1
        if self.verbose:
            print(
                f"[FOVFinderAgent] Phase {phase}: selected "
                f"{len(selected)} FOVs across {len(wells)} wells "
                f"(median valid candidates/well: "
                f"{int(df_scan.groupby('well')['valid'].sum().median()) if not df_scan.empty else 0})."
            )

        if self.return_json:
            from useq import Position

            return [Position(x=fp.x, y=fp.y, z=fp.z, name=fp.name) for fp in selected]
        return selected

    # ------------------------------------------------------------------
    # Debug plotting (verbose=True)
    # ------------------------------------------------------------------

    def _debug_show_candidate(
        self,
        fp: FovPosition,
        seg_input: np.ndarray,
        label_img: np.ndarray,
        n_cells: int,
    ) -> None:
        """Show ``raw | segmentation`` for a single scanned candidate."""
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(seg_input, cmap="gray")
        axes[0].set_title(f"{fp.name}\nx={fp.x:.0f}  y={fp.y:.0f}")
        axes[0].axis("off")
        axes[1].imshow(label_img, cmap="nipy_spectral")
        axes[1].set_title(f"segmentation — {n_cells} cells")
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()

    def _debug_show_well_summary(
        self,
        wells: list[str],
        df_scan: pd.DataFrame,
        selected: list[FovPosition],
        wells_for_positions: list[str],
        phase: int,
    ) -> None:
        """Per-well scatter of candidates with FPS-selected points highlighted."""
        import matplotlib.pyplot as plt

        selected_by_well: dict[str, list[FovPosition]] = {}
        for fp, w in zip(selected, wells_for_positions):
            selected_by_well.setdefault(w, []).append(fp)

        fig, axes = plt.subplots(
            1, len(wells), figsize=(4 * len(wells), 4), squeeze=False
        )
        for ax, well in zip(axes[0], wells):
            df_w = df_scan[df_scan["well"] == well]
            sc = ax.scatter(
                df_w["x"],
                df_w["y"],
                c=df_w["n_cells"],
                cmap="viridis",
                s=80,
                edgecolors="k",
                linewidths=0.5,
            )
            sel = selected_by_well.get(well, [])
            if sel:
                ax.scatter(
                    [fp.x for fp in sel],
                    [fp.y for fp in sel],
                    marker="x",
                    c="red",
                    s=200,
                    linewidths=3,
                    label="selected",
                )
                ax.legend(loc="best")
            ax.set_title(f"phase {phase} — well {well}")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_aspect("equal")
            fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="n_cells")
        plt.tight_layout()
        plt.show()
