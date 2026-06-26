"""Grid-overview FOV finder: scan a contiguous region, then place FOVs on cells.

Where :class:`~faro.agents.fov_finder.FOVFinderAgent` images *scattered random
candidates* and returns those same points, :class:`GridFOVFinderAgent` images a
small contiguous **grid** centred on each well (e.g. 5×5 tiles), reconstructs a
global cloud of cell centroids from the segmented tiles, and then chooses final
FOV positions **anywhere** in that scanned region — recentred onto good cell
clusters and filtered by density (count + nearest-neighbour clumping).

The selection logic lives in :mod:`faro.agents.fov_density` so it is
hardware-agnostic and tunable offline on saved images.  This class is the thin
acquisition wrapper: it reuses the plate calibration, geometry and acquisition
machinery of :class:`FOVFinderAgent` and only swaps *how candidates are imaged*
(grid instead of random) and *how positions are picked* (density-scored window
search instead of farthest-point sampling).

Coordinate caveat
------------------
Per-tile pixel→stage mapping assumes the camera axes are aligned with the stage
axes up to optional ``flip_x`` / ``flip_y`` (a ~180° plate rotation, common
here, is ``flip_x=flip_y=True``).  The grid itself is axis-aligned around each
well centre.  For a setup with an arbitrary camera-stage rotation you would
need a full affine; the offline scorer in :mod:`faro.agents.fov_density` is
unaffected as it works in whatever frame you hand it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from faro.agents.base import PreExperimentAgent
from faro.agents.fov_density import (
    FovDensityScorer,
    find_fov_windows,
    label_centroids_um,
)
from faro.agents.fov_finder import FOVFinderAgent
from faro.core.utils import FovPosition

if TYPE_CHECKING:
    from useq import WellPlate, WellPlatePlan

    from faro.agents.fov_finder import FOVCondition
    from faro.core.data_structures import Channel
    from faro.feature_extraction.base import FeatureExtractor
    from faro.microscope.base import AbstractMicroscope
    from faro.segmentation.base import Segmentator


class GridFOVFinderAgent(PreExperimentAgent):
    """Pick FOVs from a contiguous grid overview scan, scored by cell density.

    Each :meth:`run` consumes ``wells_per_phase`` wells, and for each well:

    1. images a ``grid_rows`` × ``grid_cols`` grid of tiles centred on the
       well (spacing derived from the camera FOV and ``grid_overlap``),
    2. segments every tile and converts cell centroids into one global
       (x, y) cloud in µm,
    3. runs :func:`~faro.agents.fov_density.find_fov_windows` to place
       ``fovs_per_well`` non-overlapping FOV windows on good clusters —
       recentred onto the cells and filtered by count + NN clumping.

    Args mirror :class:`FOVFinderAgent` where they overlap (plate plan, wells,
    phasing, channels, segmentator, z, naming, return type).  Density-specific
    args:

        grid_rows / grid_cols: Overview-scan grid size (default 5×5).
        grid_overlap: Fractional overlap between adjacent tiles in
            ``[0, 1)``.  Tile spacing = ``fov_size * (1 - grid_overlap)``.
            A little overlap (e.g. 0.05) avoids blind seams between tiles.
        fov_size_um: ``(width, height)`` of one FOV in µm.  ``None`` (default)
            derives it from ``pixel_size_um`` × camera image dimensions.
        pixel_size_um: Camera pixel size (µm/px).  ``None`` reads
            ``mmc.getPixelSizeUm()`` at run time.
        min_cells / max_cells: Cell-count band for an accepted FOV window.
        clump_distance_um / max_clumped_fraction: "Not clumped" definition —
            a cell whose nearest neighbour is within ``clump_distance_um`` is
            clumped; reject windows with more than ``max_clumped_fraction``
            clumped cells.
        min_nn_um / max_nn_um: Optional band on the median nearest-neighbour
            distance ("not too dense" / "not too sparse").  ``None`` disables.
        feature_extractor / fov_conditions: Optional per-cell feature gate,
            mirroring :class:`~faro.agents.fov_finder.FOVFinderAgent`.  When
            ``fov_conditions`` is set, every grid tile is also run through
            ``feature_extractor.extract_features`` to get per-cell features
            (e.g. ERK CNR via :class:`~faro.feature_extraction.erk_ktr.FE_ErkKtr`);
            those are joined onto the centroid cloud by label and each
            candidate FOV window is accepted only if it passes the density
            checks **and** every :class:`~faro.agents.fov_finder.FOVCondition`.
            This is how the grid finder *selects for ERK activity* on top of
            cell density.  Requires the imaging channels the extractor needs
            (e.g. the mScarlet3 reporter at channel index 1 for ``FE_ErkKtr``).
            Scope: count and clumping are already handled **geometrically** by
            the density knobs above, so ``fov_conditions`` here are for per-cell
            *biology* the geometry can't see (e.g. ``cnr``) — don't add a
            ``SpatialFE`` ``nn_dist`` condition to re-express clumping; use
            ``clump_distance_um`` / ``min_nn_um`` / ``max_nn_um`` instead.
        recenter / recenter_iters: Mean-shift the chosen windows onto their
            local cluster (the "move the FOV onto the nice cells" behaviour).
        min_separation_um: Minimum spacing between selected FOVs (``None`` ->
            ~one FOV side, i.e. non-overlapping).
        flip_x / flip_y: Camera→stage axis flips (see module caveat).
        strict_count: If ``True`` (default), always return exactly
            ``fovs_per_well`` per well, padding with the best-scoring windows
            even if they failed the density filter (downstream agents such as
            OscillationBO expect a fixed count).
        store_tiles: If ``True``, keep each well's raw segmentation-channel
            grid tiles on ``last_run["tile_images_by_well"]`` so the scan can
            be stitched into a stage-coordinate montage (via
            :func:`~faro.agents.fov_density.build_stage_montage`) to validate
            that the pixel→stage transform / repositioning is correct.  Costs
            ~``grid_rows*grid_cols`` images of RAM per well; default ``False``.
    """

    def __init__(
        self,
        microscope: AbstractMicroscope,
        *,
        well_plate_plan: str | Path | "WellPlatePlan",
        wells: list[str],
        wells_per_phase: int | None = None,
        fovs_per_well: int,
        grid_rows: int = 5,
        grid_cols: int = 5,
        grid_overlap: float = 0.0,
        fov_size_um: tuple[float, float] | None = None,
        pixel_size_um: float | None = None,
        min_cells: int = 10,
        max_cells: int | None = None,
        clump_distance_um: float = 15.0,
        max_clumped_fraction: float = 0.3,
        min_nn_um: float | None = None,
        max_nn_um: float | None = None,
        recenter: bool = True,
        recenter_iters: int = 3,
        min_separation_um: float | None = None,
        imaging_channels: tuple[Channel, ...],
        segmentator: Segmentator,
        seg_channel_index: int = 0,
        feature_extractor: "FeatureExtractor | None" = None,
        fov_conditions: list["FOVCondition"] | None = None,
        z: float | None | Literal["current"] = None,
        flip_x: bool = False,
        flip_y: bool = False,
        random_seed: int | None = None,
        name_prefix: str = "fov",
        strict_count: bool = True,
        return_json: bool | None = None,
        cycle_wells: bool = False,
        store_tiles: bool = False,
        verbose: bool = False,
    ):
        super().__init__(microscope)

        if wells_per_phase is None:
            wells_per_phase = len(wells)
        if wells_per_phase <= 0:
            raise ValueError("wells_per_phase must be positive")
        if fovs_per_well <= 0:
            raise ValueError("fovs_per_well must be positive")
        if grid_rows <= 0 or grid_cols <= 0:
            raise ValueError("grid_rows and grid_cols must be positive")
        if not 0.0 <= grid_overlap < 1.0:
            raise ValueError("grid_overlap must be in [0, 1)")
        if max_cells is not None and max_cells < min_cells:
            raise ValueError(
                f"max_cells ({max_cells}) must be >= min_cells ({min_cells})"
            )
        if not imaging_channels:
            raise ValueError("imaging_channels must contain at least one channel")
        if not (z is None or z == "current" or isinstance(z, (int, float))):
            raise ValueError(f"z must be None, 'current', or a float; got {z!r}")
        if fov_conditions and feature_extractor is None:
            raise ValueError(
                "fov_conditions requires a feature_extractor that produces "
                "the referenced feature columns (e.g. FE_ErkKtr for 'cnr')."
            )

        self.wells_per_phase = int(wells_per_phase)
        self.fovs_per_well = int(fovs_per_well)
        self.grid_rows = int(grid_rows)
        self.grid_cols = int(grid_cols)
        self.grid_overlap = float(grid_overlap)
        self.fov_size_um = fov_size_um
        self.pixel_size_um = pixel_size_um
        self.min_cells = int(min_cells)
        self.max_cells = None if max_cells is None else int(max_cells)
        self.clump_distance_um = float(clump_distance_um)
        self.max_clumped_fraction = float(max_clumped_fraction)
        self.min_nn_um = None if min_nn_um is None else float(min_nn_um)
        self.max_nn_um = None if max_nn_um is None else float(max_nn_um)
        self.recenter = bool(recenter)
        self.recenter_iters = int(recenter_iters)
        self.min_separation_um = min_separation_um
        self.imaging_channels = tuple(imaging_channels)
        self.segmentator = segmentator
        self.seg_channel_index = int(seg_channel_index)
        self.feature_extractor = feature_extractor
        self.fov_conditions: list[FOVCondition] = list(fov_conditions or [])
        self.z = z
        self.flip_x = bool(flip_x)
        self.flip_y = bool(flip_y)
        self.random_seed = random_seed
        self.name_prefix = name_prefix
        self.strict_count = bool(strict_count)
        self.return_json = bool(return_json) if return_json is not None else False
        self.cycle_wells = bool(cycle_wells)
        self.store_tiles = bool(store_tiles)
        self.verbose = bool(verbose)

        # Reuse FOVFinderAgent's calibration loading + geometry. We only need
        # the plate; the rest of FOVFinderAgent's __init__ (random-candidate
        # knobs) is not relevant here, so we set up the minimal state its
        # reused methods (_well_center_um, _acquire_frames, pick_next_wells)
        # depend on.
        self._plan = FOVFinderAgent._load_plan(well_plate_plan)
        self._plate: WellPlate = self._plan.plate
        self._a1_center_xy = self._plan.a1_center_xy
        self._rotation = self._plan.rotation

        from faro.agents.fov_finder import _well_name_to_index

        for w in wells:
            r, c = _well_name_to_index(w)
            if r >= self._plate.rows or c >= self._plate.columns:
                raise ValueError(
                    f"Well {w!r} is outside the plate "
                    f"({self._plate.rows} rows x {self._plate.columns} cols)"
                )
        self._wells_source: list[str] = list(wells)
        self._remaining_wells: list[str] = list(wells)
        self._phase_index = 0
        self.history: list[dict[str, Any]] = []

    # Borrow the parent's geometry / acquisition / queue helpers so we don't
    # duplicate the WellPlatePlan math or the run_mda frame-collection logic.
    _well_center_um = FOVFinderAgent._well_center_um
    _acquire_frames = FOVFinderAgent._acquire_frames
    pick_next_wells = FOVFinderAgent.pick_next_wells
    remaining_wells = FOVFinderAgent.remaining_wells
    n_remaining_phases = FOVFinderAgent.n_remaining_phases

    # ------------------------------------------------------------------
    # FOV geometry
    # ------------------------------------------------------------------

    def _resolve_fov_geometry(self) -> tuple[float, float, float]:
        """Return ``(pixel_size_um, fov_w_um, fov_h_um)`` for this run."""
        mmc = self.microscope.mmc
        px = self.pixel_size_um
        if px is None:
            px = float(mmc.getPixelSizeUm())
            if px <= 0:
                raise RuntimeError(
                    "Camera pixel size is 0/undefined; set pixel_size_um "
                    "explicitly on GridFOVFinderAgent."
                )
        if self.fov_size_um is not None:
            fov_w, fov_h = float(self.fov_size_um[0]), float(self.fov_size_um[1])
        else:
            w_px = int(mmc.getImageWidth())
            h_px = int(mmc.getImageHeight())
            fov_w, fov_h = w_px * px, h_px * px
        return px, fov_w, fov_h

    def _grid_offsets(self, fov_w_um: float, fov_h_um: float) -> np.ndarray:
        """Axis-aligned ``(grid_rows*grid_cols, 2)`` tile-centre offsets (µm)."""
        step_x = fov_w_um * (1.0 - self.grid_overlap)
        step_y = fov_h_um * (1.0 - self.grid_overlap)
        cols = (np.arange(self.grid_cols) - (self.grid_cols - 1) / 2.0) * step_x
        rows = (np.arange(self.grid_rows) - (self.grid_rows - 1) / 2.0) * step_y
        gx, gy = np.meshgrid(cols, rows)
        return np.column_stack([gx.ravel(), gy.ravel()])

    # ------------------------------------------------------------------
    # Per-well overview scan -> global centroid cloud
    # ------------------------------------------------------------------

    def _tile_features(
        self,
        t_idx: int,
        frames: dict[tuple[int, int], np.ndarray],
        label_img: np.ndarray,
        labels: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Per-cell feature arrays for one tile, aligned to *labels*.

        Runs :attr:`feature_extractor` on the tile's full multi-channel stack
        and joins the requested condition features (e.g. ``"cnr"``) onto the
        centroid labels.  Cells the extractor dropped (or a failed extraction)
        get ``NaN`` so the arrays always line up 1:1 with *labels* — keeping the
        well-level feature cloud aligned with the centroid cloud.
        """
        needed = [c.feature for c in self.fov_conditions]
        nan_fill = {f: np.full(len(labels), np.nan, dtype=float) for f in needed}
        n_channels = len(self.imaging_channels)
        channel_imgs: list[np.ndarray] | None = []
        for c in range(n_channels):
            im = frames.get((t_idx, c))
            if im is None:
                channel_imgs = None
                break
            channel_imgs.append(np.asarray(im))
        if channel_imgs is None:
            return nan_fill

        img_stack = np.stack(channel_imgs, axis=0)  # (C, H, W) == [C, X, Y]
        used_mask = getattr(self.feature_extractor, "used_mask", "labels")
        try:
            fe_result = self.feature_extractor.extract_features(
                {used_mask: label_img}, img_stack
            )
        except Exception as e:  # pragma: no cover - user-supplied FE
            print(
                f"[GridFOVFinderAgent] extract_features failed on tile "
                f"{t_idx}: {type(e).__name__}: {e}"
            )
            return nan_fill

        df = fe_result[0] if isinstance(fe_result, tuple) and fe_result else fe_result
        if df is None or getattr(df, "empty", True) or "label" not in df.columns:
            return nan_fill

        out: dict[str, np.ndarray] = {}
        warned = self.__dict__.setdefault("_warned_missing_feats", set())
        for f in needed:
            if f in df.columns:
                by_label = dict(zip(df["label"].astype(int), df[f].astype(float)))
                out[f] = np.array(
                    [by_label.get(int(lbl), np.nan) for lbl in labels], dtype=float
                )
            else:
                # The condition names a feature this extractor doesn't produce.
                # Don't silently reject every window -- say so loudly, once, so a
                # mismatched extractor/condition (e.g. FE_ErkKtr + an "nn_dist"
                # condition) is caught instead of quietly padding by density.
                if f not in warned:
                    warned.add(f)
                    fe_name = type(self.feature_extractor).__name__
                    print(
                        f"[GridFOVFinderAgent] WARNING: FOVCondition references "
                        f"feature {f!r}, but {fe_name}.extract_features produced "
                        f"{sorted(c for c in df.columns if c != 'label')}. Every "
                        f"window will FAIL this condition (FOVs then filled by "
                        f"density alone). Use a feature_extractor that emits "
                        f"{f!r} (e.g. SpatialFE for 'nn_dist'/'area_um2', "
                        f"FE_ErkKtr for 'cnr')."
                    )
                out[f] = np.full(len(labels), np.nan, dtype=float)
        return out

    def _scan_well_cloud(
        self,
        well: str,
        phase: int,
        z_value: float | None,
        pixel_size_um: float,
        fov_w_um: float,
        fov_h_um: float,
    ) -> tuple[
        np.ndarray,
        list[FovPosition],
        list[np.ndarray | None],
        dict[str, np.ndarray] | None,
    ]:
        """Image the grid for one well.

        Returns ``(centroid cloud µm, tile positions, tile seg-channel images,
        feature cloud)``.  The tile images are returned only when
        :attr:`store_tiles` is set (otherwise the list holds ``None``
        placeholders to save memory) so the scan can be stitched into a
        stage-coordinate montage for validation.  The feature cloud is a
        ``{feature_name: (N,) array}`` aligned to the centroid cloud when
        :attr:`fov_conditions` is set (so windows can be gated on ERK activity),
        otherwise ``None``.
        """
        cx, cy = self._well_center_um(well)
        offsets = self._grid_offsets(fov_w_um, fov_h_um)
        tile_positions = [
            FovPosition(
                x=cx + float(dx),
                y=cy + float(dy),
                z=z_value,
                name=f"{self.name_prefix}_p{phase}_{well}__tile{i:03d}",
            )
            for i, (dx, dy) in enumerate(offsets)
        ]

        frames = self._acquire_frames(tile_positions)

        want_feat = bool(self.fov_conditions)
        clouds: list[np.ndarray] = []
        tile_images: list[np.ndarray | None] = []
        feat_cols: dict[str, list[np.ndarray]] = {}
        for t_idx, tile in enumerate(tile_positions):
            img = frames.get((t_idx, self.seg_channel_index))
            if img is None:
                tile_images.append(None)
                continue
            img = np.asarray(img)
            tile_images.append(img.copy() if self.store_tiles else None)
            label_img = self.segmentator.segment(img)
            if want_feat:
                pts, labels = label_centroids_um(
                    label_img,
                    pixel_size_um=pixel_size_um,
                    origin_xy=(tile.x, tile.y),
                    flip_x=self.flip_x,
                    flip_y=self.flip_y,
                    return_labels=True,
                )
            else:
                pts = label_centroids_um(
                    label_img,
                    pixel_size_um=pixel_size_um,
                    origin_xy=(tile.x, tile.y),
                    flip_x=self.flip_x,
                    flip_y=self.flip_y,
                )
            if not len(pts):
                continue
            clouds.append(pts)
            if want_feat:
                # One array per condition feature, aligned to this tile's pts,
                # appended in lockstep with `clouds` so the well-level concat
                # stays aligned with the centroid cloud.
                tile_feat = self._tile_features(t_idx, frames, label_img, labels)
                for k, arr in tile_feat.items():
                    feat_cols.setdefault(k, []).append(arr)

        cloud = (
            np.concatenate(clouds, axis=0) if clouds else np.empty((0, 2), dtype=float)
        )
        feat = None
        if want_feat:
            feat = {
                k: (np.concatenate(v) if v else np.empty((0,), dtype=float))
                for k, v in feat_cols.items()
            }
        return cloud, tile_positions, tile_images, feat

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> list[FovPosition] | list[Any]:
        """Run the grid-overview FOV finder for one phase.

        Returns the selected positions (``list[FovPosition]`` by default, or
        ``list[useq.Position]`` when ``return_json=True``).  Per-phase debug
        data — the centroid clouds, candidate-window and selected-window
        DataFrames, and tile positions per well — is stashed on
        :attr:`last_run`.
        """
        phase = self._phase_index
        z_value = self.microscope.get_focus() if self.z == "current" else self.z
        pixel_size_um, fov_w_um, fov_h_um = self._resolve_fov_geometry()

        wells = self.pick_next_wells()
        if self.verbose:
            print(
                f"[GridFOVFinderAgent] Phase {phase}: {self.grid_rows}x"
                f"{self.grid_cols} grid scan in {len(wells)} well(s): {wells} "
                f"(fov={fov_w_um:.0f}x{fov_h_um:.0f} µm, px={pixel_size_um:.3f})"
            )

        scorer = FovDensityScorer(
            fov_w_um=fov_w_um,
            fov_h_um=fov_h_um,
            min_cells=self.min_cells,
            max_cells=self.max_cells,
            clump_distance_um=self.clump_distance_um,
            max_clumped_fraction=self.max_clumped_fraction,
            min_nn_um=self.min_nn_um,
            max_nn_um=self.max_nn_um,
            fov_conditions=self.fov_conditions,
        )

        selected: list[FovPosition] = []
        wells_for_positions: list[str] = []
        clouds_by_well: dict[str, np.ndarray] = {}
        candidates_by_well: dict[str, pd.DataFrame] = {}
        tiles_by_well: dict[str, list[FovPosition]] = {}
        tile_images_by_well: dict[str, list[np.ndarray | None]] = {}
        feats_by_well: dict[str, dict[str, np.ndarray]] = {}

        for well in wells:
            cloud, tiles, tile_images, feat = self._scan_well_cloud(
                well, phase, z_value, pixel_size_um, fov_w_um, fov_h_um
            )
            clouds_by_well[well] = cloud
            tiles_by_well[well] = tiles
            if self.store_tiles:
                tile_images_by_well[well] = tile_images
            if feat is not None:
                feats_by_well[well] = feat

            df_all, df_sel = find_fov_windows(
                cloud,
                scorer,
                n_select=self.fovs_per_well,
                recenter=self.recenter,
                recenter_iters=self.recenter_iters,
                min_separation_um=self.min_separation_um,
                # strict_count -> top up to fovs_per_well from the best
                # below-threshold windows, but still non-overlapping & spread.
                fill_invalid=self.strict_count,
                feat=feat,  # per-cell ERK features -> gate windows on activity
            )
            candidates_by_well[well] = df_all

            if len(df_sel) < self.fovs_per_well and self.verbose:
                n_pad = int((~df_sel["valid"]).sum()) if len(df_sel) else 0
                print(
                    f"[GridFOVFinderAgent] well {well}: only {len(df_sel)}/"
                    f"{self.fovs_per_well} non-overlapping FOVs could be placed "
                    f"({n_pad} below-threshold fill); the well's good cells may "
                    f"not span enough non-overlapping fields."
                )

            for k, (_, r) in enumerate(df_sel.iterrows()):
                selected.append(
                    FovPosition(
                        x=float(r["x"]),
                        y=float(r["y"]),
                        z=z_value,
                        name=f"{well}_{k:04d}",
                    )
                )
                wells_for_positions.append(well)

            if self.verbose:
                n_found = int(cloud.shape[0])
                n_valid = int(df_all["valid"].sum()) if len(df_all) else 0
                print(
                    f"[GridFOVFinderAgent] well {well}: {n_found} cells in scan, "
                    f"{n_valid} passing window(s), selected {len(df_sel)}."
                )

        if self.verbose:
            self._debug_show_wells(
                wells,
                clouds_by_well,
                candidates_by_well,
                selected,
                wells_for_positions,
                fov_w_um,
                fov_h_um,
                phase,
            )

        print(f"[GridFOVFinderAgent] Phase {phase} — selected FOVs:")
        for fp, w in zip(selected, wells_for_positions):
            df_all = candidates_by_well.get(w)
            tag = ""
            if df_all is not None and len(df_all):
                row = df_all[
                    np.isclose(df_all["x"], fp.x) & np.isclose(df_all["y"], fp.y)
                ]
                if not row.empty:
                    r0 = row.iloc[0]
                    tag = (
                        f": {int(r0['n_cells'])} cells, "
                        f"clumped={r0['clumped_fraction']:.2f}, "
                        f"med_nn={r0['median_nn_um']:.1f}µm"
                    )
                    if not bool(r0["valid"]):
                        tag += f" (padded: {r0['reason']})"
            print(f"    {fp.name}{tag}")

        self.last_run: dict[str, Any] = {
            "positions": selected,
            "wells_for_positions": wells_for_positions,
            "wells_used": list(wells),
            "clouds_by_well": clouds_by_well,
            "candidates_by_well": candidates_by_well,
            "feats_by_well": feats_by_well,
            "tiles_by_well": tiles_by_well,
            "tile_images_by_well": tile_images_by_well,
            "phase": phase,
            "fov_size_um": (fov_w_um, fov_h_um),
            "pixel_size_um": pixel_size_um,
            "flip_x": self.flip_x,
            "flip_y": self.flip_y,
        }
        self.history.append(
            {
                "phase": phase,
                "wells_used": list(wells),
                "n_selected": len(selected),
            }
        )
        self._phase_index += 1

        if self.return_json:
            from useq import Position

            return [Position(x=fp.x, y=fp.y, z=fp.z, name=fp.name) for fp in selected]
        return selected

    # ------------------------------------------------------------------
    # Debug plotting
    # ------------------------------------------------------------------

    def _debug_show_wells(
        self,
        wells: list[str],
        clouds_by_well: dict[str, np.ndarray],
        candidates_by_well: dict[str, pd.DataFrame],
        selected: list[FovPosition],
        wells_for_positions: list[str],
        fov_w_um: float,
        fov_h_um: float,
        phase: int,
    ) -> None:
        """Per-well scatter of the centroid cloud + selected FOV footprints."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        sel_by_well: dict[str, list[FovPosition]] = {}
        for fp, w in zip(selected, wells_for_positions):
            sel_by_well.setdefault(w, []).append(fp)

        fig, axes = plt.subplots(
            1, len(wells), figsize=(5 * len(wells), 5), squeeze=False
        )
        for ax, well in zip(axes[0], wells):
            cloud = clouds_by_well.get(well, np.empty((0, 2)))
            if len(cloud):
                ax.scatter(cloud[:, 0], cloud[:, 1], s=10, c="0.6", label="cells")
            for fp in sel_by_well.get(well, []):
                ax.add_patch(
                    Rectangle(
                        (fp.x - fov_w_um / 2, fp.y - fov_h_um / 2),
                        fov_w_um,
                        fov_h_um,
                        fill=False,
                        edgecolor="red",
                        linewidth=2,
                    )
                )
                ax.plot(fp.x, fp.y, "rx", markersize=10)
            ax.set_title(f"phase {phase} — well {well}")
            ax.set_xlabel("x (µm)")
            ax.set_ylabel("y (µm)")
            ax.set_aspect("equal")
        plt.tight_layout()
        plt.show()
