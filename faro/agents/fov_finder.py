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

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from faro.agents.base import PreExperimentAgent
from faro.core.data_structures import Channel, RTMSequence
from faro.core.utils import FovPosition

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
        wells_per_phase: How many wells to consume per phase.
        fovs_per_well: How many valid FOV positions to return per well
            (after cell-count filtering and farthest-point selection).
        n_candidates_per_well: How many random candidate positions to scan
            per well.  Should be larger than *fovs_per_well* to give the
            farthest-point selection something to choose from.
        border_um: Minimum distance (µm) to keep candidate positions away
            from the well edge.  Random candidates are generated inside a
            shape ``(well_size_x - 2*border_um, well_size_y - 2*border_um)``.
        min_cells: Minimum cell count for a candidate to be considered
            valid.  Positions with fewer cells are dropped before
            farthest-point sampling.
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
            does not affect selection.
        z: Optional fixed Z position for all candidates.  When ``None``
            (default) the current focus reported by ``mmc.getPosition()``
            is used at the moment :meth:`run` is invoked.
        random_seed: Optional RNG seed.  Each phase derives a deterministic
            sub-seed from this so candidate generation and farthest-point
            tie-breaking are reproducible across runs.
        name_prefix: Prefix used to name returned :class:`FovPosition`
            tuples (``"<prefix>_<phase>_<wellname>_<i>"``).
    """

    def __init__(
        self,
        microscope: AbstractMicroscope,
        *,
        well_plate_plan: str | Path | "WellPlatePlan",
        wells: list[str],
        wells_per_phase: int,
        fovs_per_well: int,
        n_candidates_per_well: int,
        border_um: float,
        min_cells: int,
        imaging_channels: tuple[Channel, ...],
        segmentator: Segmentator,
        seg_channel_index: int = 0,
        feature_extractor: FeatureExtractor | None = None,
        z: float | None = None,
        random_seed: int | None = None,
        name_prefix: str = "fov",
        strict_count: bool = True,
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
        if wells_per_phase <= 0:
            raise ValueError("wells_per_phase must be positive")
        if fovs_per_well <= 0:
            raise ValueError("fovs_per_well must be positive")
        if n_candidates_per_well < fovs_per_well:
            raise ValueError(
                f"n_candidates_per_well ({n_candidates_per_well}) must be "
                f">= fovs_per_well ({fovs_per_well})"
            )
        if border_um < 0:
            raise ValueError("border_um must be non-negative")
        if not imaging_channels:
            raise ValueError("imaging_channels must contain at least one channel")

        self.well_plate_plan_input = well_plate_plan
        self.wells_per_phase = int(wells_per_phase)
        self.fovs_per_well = int(fovs_per_well)
        self.n_candidates_per_well = int(n_candidates_per_well)
        self.border_um = float(border_um)
        self.min_cells = int(min_cells)
        self.imaging_channels = tuple(imaging_channels)
        self.segmentator = segmentator
        self.seg_channel_index = int(seg_channel_index)
        self.feature_extractor = feature_extractor
        self.z = z
        self.random_seed = random_seed
        self.name_prefix = name_prefix
        self.strict_count = bool(strict_count)

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
        rp = RandomPoints(
            num_points=n_points,
            max_width=max_w,
            max_height=max_h,
            shape=Shape.ELLIPSE if self._plate.circular_wells else Shape.RECTANGLE,
            random_seed=seed,
            allow_overlap=True,
        )
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
        """
        n_channels = len(self.imaging_channels)
        rows: list[dict[str, Any]] = []

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

            row: dict[str, Any] = {
                "p": p_idx,
                "x": fp.x,
                "y": fp.y,
                "z": fp.z,
                # "well" is overwritten in run() from the parallel
                # candidate_well list (the source of truth).
                "n_cells": n_cells,
                "valid": n_cells >= self.min_cells,
                "reason": "" if n_cells >= self.min_cells else "below_min_cells",
            }

            if self.feature_extractor is not None:
                try:
                    df_features = self.feature_extractor.extract_positions(
                        {"labels": label_img}
                    )
                    if df_features is not None and not df_features.empty:
                        for col in df_features.columns:
                            if col == "label":
                                continue
                            try:
                                row[f"fe_{col}_mean"] = float(df_features[col].mean())
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

    def run(self) -> dict[str, Any]:
        """Find FOV positions for one phase.

        Returns:
            Dict with keys:

            * ``"positions"`` – list of selected :class:`FovPosition`
              tuples (length ``wells_per_phase * fovs_per_well`` when all
              wells yielded enough valid candidates).
            * ``"wells_used"`` – list of well names consumed this phase.
            * ``"all_candidates"`` – DataFrame of every scanned candidate
              with cell counts (and optional feature-extractor outputs).
            * ``"phase"`` – zero-based phase index for this call.
        """
        wells = self.pick_next_wells()
        phase = self._phase_index
        print(
            f"[FOVFinderAgent] Phase {phase}: scanning "
            f"{self.n_candidates_per_well} candidates in "
            f"{len(wells)} wells: {wells}"
        )

        # Resolve a Z value for all candidates if the user did not pin one.
        z_value = self.z
        if z_value is None:
            try:
                z_value = float(self.microscope.mmc.getPosition())
            except Exception:
                z_value = None

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
                print(
                    f"[FOVFinderAgent] WARNING: well {well} has no valid "
                    f"positions (need >= {self.min_cells} cells); skipping."
                )
                continue

            if n_valid > 0:
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
                    print(
                        f"[FOVFinderAgent] WARNING: well {well} only had "
                        f"{n_picked}/{self.fovs_per_well} valid positions; "
                        f"padded with {len(df_pad)} below-threshold "
                        f"candidates (best n_cells={int(df_pad['n_cells'].max())})."
                    )
                else:
                    print(
                        f"[FOVFinderAgent] WARNING: well {well} has no "
                        f"candidates at all to pad with; phase will have "
                        f"fewer FOVs than expected."
                    )

            if not self.strict_count and n_picked < self.fovs_per_well:
                print(
                    f"[FOVFinderAgent] WARNING: well {well} only yielded "
                    f"{n_picked}/{self.fovs_per_well} valid positions."
                )

            for k, (_, row) in enumerate(df_picked.iterrows()):
                name = f"{self.name_prefix}_p{phase}_{well}_{k}"
                selected.append(
                    FovPosition(
                        x=float(row["x"]),
                        y=float(row["y"]),
                        z=z_value,
                        name=name,
                    )
                )
                wells_for_positions.append(well)

        result: dict[str, Any] = {
            "positions": selected,
            "wells_for_positions": wells_for_positions,
            "wells_used": list(wells),
            "all_candidates": df_scan,
            "phase": phase,
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
        print(
            f"[FOVFinderAgent] Phase {phase}: selected "
            f"{len(selected)} FOVs across {len(wells)} wells "
            f"(median valid candidates/well: "
            f"{int(df_scan.groupby('well')['valid'].sum().median()) if not df_scan.empty else 0})."
        )
        return result
