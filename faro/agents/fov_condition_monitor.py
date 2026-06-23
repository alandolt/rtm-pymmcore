"""Periodic-monitor agent that gates a downstream experiment on cell-feature
conditions (FOV-level features pooled across monitored FOVs).

The agent images a chosen scout source on a fixed cadence, segments,
extracts per-cell features, and tests a user-supplied list of
:class:`FOVCondition`\\ s against the pooled cells.  It returns as soon as
**every** condition passes (AND-combined), or a timeout elapses.

Use it whenever you want to wait for the cells to reach a defined state
before launching the real experiment.  Examples:

* **Starvation gate.**  After moving the plate to the microscope, ERK is
  transiently active from handling stress (``cnr`` ~ 1.5-2) and decays
  over hours in starvation medium.  Wait until ``cnr < 1.0`` in 75 % of
  cells before starting::

      FOVCondition("cnr", "below", 1.0, min_fraction=0.75)

* **Growth gate.**  Wait until cells have grown to a target size, e.g.
  until 80 % of cells have nuclear area above 650 µm²::

      FOVCondition("area_nuc", "above", 650.0, min_fraction=0.8)

* **Density gate.**  Wait until the FOV has enough cells, or fewer than
  some upper bound.  Combine multiple to AND them together::

      [FOVCondition("cnr",      "below", 1.0, min_fraction=0.75),
       FOVCondition("area_nuc", "above", 650, min_fraction=0.7)]

The ``fov_finder`` argument is polymorphic — it picks the scouting
strategy:

* :class:`FOVFinderAgent` — fresh FOVs in a sacrificial well each round
  (well-plate experiments; the finder must have ``cycle_wells=True`` and
  a non-empty ``fov_conditions`` list so per-cell features end up in
  ``last_run["fov_features"]``).
* :class:`FovPosition` — image this single spot every round.
* ``list[FovPosition]`` — image these spots every round.
* ``None`` (default) — image at the **current** stage position, fetched
  once at :meth:`run` start via ``microscope.get_position()`` (the
  backend must implement it).

A :class:`FOVFinderAgent` already bundles the microscope, imaging
channels, segmentator, and feature extractor.  For the three other
forms, pass them explicitly via ``microscope``, ``imaging_channels``,
``segmentator``, ``feature_extractor``, and optionally
``seg_channel_index``.

Each monitoring round writes one row to a trajectory DataFrame; columns
are ``round``, ``elapsed_min``, ``n_cells_pooled``, ``triggered``
(overall AND), plus per condition ``c{i}_pass`` (bool), ``c{i}_frac``
(fraction of cells satisfying that condition) and ``c{i}_median``
(median of that condition's feature across pooled cells).  If
``storage_path`` is set the trajectory is written to
``<storage_path>/condition_monitor.parquet``.

Well-plate example (starvation gate)::

    scout = FOVFinderAgent(
        microscope=mic, well_plate_plan=PLATE_CALIBRATION_PATH,
        wells=[SACRIFICIAL_WELL], wells_per_phase=1,
        fovs_per_well=4, n_candidates_per_well=4,
        ..., fov_conditions=[FOVCondition("cnr", "below", 1.0, min_fraction=0.75)],
        cycle_wells=True, strict_count=False,
    )
    monitor = FOVConditionMonitorAgent(
        fov_finder=scout,
        fov_conditions=[FOVCondition("cnr", "below", 1.0, min_fraction=0.75)],
        check_interval_s=20 * 60, timeout_s=20 * 60 * 60,
        storage_path=path,
    )

Bench example — image at the current stage position (growth gate)::

    monitor = FOVConditionMonitorAgent(
        fov_finder=None,                          # current stage position
        microscope=mic,
        imaging_channels=(mirfp_channel, mscarlet3_channel),
        segmentator=segmentator,
        feature_extractor=FE_ErkKtr("labels"),
        fov_conditions=[
            FOVCondition("area_nuc", "above", 650.0, min_fraction=0.8),
        ],
        check_interval_s=20 * 60, timeout_s=20 * 60 * 60,
        storage_path=path,
    )
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from faro.agents.base import PreExperimentAgent
from faro.agents.fov_finder import FOVCondition, FOVFinderAgent
from faro.core.data_structures import RTMSequence
from faro.core.utils import FovPosition

if TYPE_CHECKING:
    from faro.core.data_structures import Channel
    from faro.feature_extraction.base import FeatureExtractor
    from faro.microscope.base import AbstractMicroscope
    from faro.segmentation.base import Segmentator


class FOVConditionMonitorAgent(PreExperimentAgent):
    """Poll cell-feature conditions until they all pass, or time out.

    See the module docstring for the polymorphic ``fov_finder`` argument
    and full examples.

    Args:
        fov_conditions: A non-empty list of :class:`FOVCondition`\\ s
            AND-combined into the readiness gate.  Each round each
            condition's ``check(pooled)`` is evaluated on the per-cell
            DataFrame pooled across monitored FOVs; the agent triggers
            when *all* conditions pass.  The list mirrors the
            :class:`FOVFinderAgent` API — pass one or more conditions
            with the same semantics.
        check_interval_s: Seconds to wait between monitoring rounds.
        timeout_s: Maximum total seconds to monitor before giving up.  On
            timeout the agent returns ``triggered=False`` — whether to
            start the downstream experiment anyway is the caller's call.
        storage_path: If set, the per-round trajectory is written to
            ``<storage_path>/condition_monitor.parquet``.
        fov_finder: Polymorphic — picks the scouting strategy (see
            module docstring).  Accepts :class:`FOVFinderAgent`,
            :class:`FovPosition`, ``list[FovPosition]``, or ``None``
            (current stage position).
        microscope: Required when ``fov_finder`` is not a
            :class:`FOVFinderAgent`.
        imaging_channels: Required when ``fov_finder`` is not a
            :class:`FOVFinderAgent`.
        segmentator: Required when ``fov_finder`` is not a
            :class:`FOVFinderAgent`.
        feature_extractor: Required when ``fov_finder`` is not a
            :class:`FOVFinderAgent`.  Must produce a per-cell DataFrame
            carrying every feature referenced in *fov_conditions*.
        seg_channel_index: Which imaging channel to segment.  Defaults to
            ``0``.  Ignored when ``fov_finder`` is a
            :class:`FOVFinderAgent`.

    The :meth:`run` result dict has keys ``triggered`` (bool),
    ``n_rounds``, ``elapsed_min``, and ``trajectory`` (the per-round
    DataFrame described above).
    """

    def __init__(
        self,
        *,
        fov_conditions: list[FOVCondition],
        check_interval_s: float,
        timeout_s: float,
        storage_path: str = "",
        fov_finder: FOVFinderAgent | FovPosition | list[FovPosition] | None = None,
        microscope: "AbstractMicroscope | None" = None,
        imaging_channels: "tuple[Channel, ...] | None" = None,
        segmentator: "Segmentator | None" = None,
        feature_extractor: "FeatureExtractor | None" = None,
        seg_channel_index: int = 0,
    ):
        # --- fov_conditions: non-empty list of FOVCondition ---
        if not fov_conditions:
            raise ValueError("fov_conditions must contain at least one FOVCondition")
        for c in fov_conditions:
            if not isinstance(c, FOVCondition):
                raise TypeError(
                    "fov_conditions must contain FOVCondition instances; "
                    f"got {type(c).__name__}"
                )
        self.fov_conditions: list[FOVCondition] = list(fov_conditions)

        if check_interval_s <= 0 or timeout_s <= 0:
            raise ValueError("check_interval_s and timeout_s must be positive")
        if timeout_s < check_interval_s:
            raise ValueError("timeout_s must be >= check_interval_s")

        # --- Dispatch on the polymorphic fov_finder argument -----------------
        if isinstance(fov_finder, FOVFinderAgent):
            self.mode = "fov_finder"
            if not fov_finder.cycle_wells:
                raise ValueError(
                    "fov_finder (FOVFinderAgent) must be constructed with "
                    "cycle_wells=True so the sacrificial well can be "
                    "re-imaged every round."
                )
            if fov_finder.feature_extractor is None or not fov_finder.fov_conditions:
                raise ValueError(
                    "fov_finder (FOVFinderAgent) must have a "
                    "feature_extractor AND a non-empty fov_conditions list, "
                    "so per-cell features are extracted and exposed via "
                    "last_run['fov_features']."
                )
            self.fov_finder = fov_finder
            self._positions: list[FovPosition] | None = None
            super().__init__(fov_finder.microscope)
        else:
            self.mode = "positions"
            if isinstance(fov_finder, FovPosition):
                self._positions = [fov_finder]
            elif isinstance(fov_finder, list):
                if not fov_finder:
                    raise ValueError("fov_finder list is empty")
                for p in fov_finder:
                    if not isinstance(p, FovPosition):
                        raise TypeError(
                            "fov_finder list must contain FovPosition "
                            f"instances; got {type(p).__name__}"
                        )
                self._positions = list(fov_finder)
            elif fov_finder is None:
                self._positions = None  # resolve at run() via get_position()
            else:
                raise TypeError(
                    "fov_finder must be a FOVFinderAgent, a FovPosition, a "
                    "list[FovPosition], or None (current stage position); "
                    f"got {type(fov_finder).__name__}"
                )

            missing = [
                name
                for name, val in zip(
                    (
                        "microscope",
                        "imaging_channels",
                        "segmentator",
                        "feature_extractor",
                    ),
                    (microscope, imaging_channels, segmentator, feature_extractor),
                )
                if val is None
            ]
            if missing:
                raise ValueError(
                    "When fov_finder is not a FOVFinderAgent (positional "
                    "scouting), these args are required: "
                    f"{', '.join(missing)}."
                )
            if not imaging_channels:
                raise ValueError("imaging_channels must contain at least one channel")
            if not (0 <= int(seg_channel_index) < len(imaging_channels)):
                raise ValueError(
                    f"seg_channel_index={seg_channel_index} is outside the "
                    f"imaging_channels range [0, {len(imaging_channels)})"
                )

            self.fov_finder = None
            self._imaging_channels = tuple(imaging_channels)
            self._segmentator = segmentator
            self._feature_extractor = feature_extractor
            self._seg_channel_index = int(seg_channel_index)
            super().__init__(microscope)

        self.check_interval_s = float(check_interval_s)
        self.timeout_s = float(timeout_s)
        self.storage_path = str(storage_path)
        self._stop = False
        self._round_counter = 0

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict:
        """Monitor until all conditions pass, or time out (blocking)."""
        print(
            f"[FOVConditionMonitorAgent] mode={self.mode}  "
            f"{len(self.fov_conditions)} condition(s) AND-combined; "
            f"timeout {self.timeout_s / 3600:.1f} h; "
            f"check every {self.check_interval_s / 60:.1f} min."
        )
        for i, c in enumerate(self.fov_conditions):
            print(
                f"  c{i}: {c.feature} {c.operator} {c.threshold:g} "
                f"in >= {c.min_fraction * 100:.0f}% of cells"
            )

        # Mode B with positions=None: resolve current stage position once.
        if self.mode == "positions" and self._positions is None:
            xy = self.microscope.get_position()
            if xy is None:
                raise RuntimeError(
                    "fov_finder=None requires the microscope backend to "
                    "implement get_position(); this one returned None. "
                    "Pass an explicit FovPosition instead."
                )
            x, y = xy
            z = self.microscope.get_focus()
            self._positions = [FovPosition(x=float(x), y=float(y), z=z, name="current")]
            print(
                f"[FOVConditionMonitorAgent] resolved current stage position: "
                f"x={x:.1f} y={y:.1f} z={'N/A' if z is None else f'{z:.1f}'}"
            )

        t_start = time.monotonic()
        trajectory: list[dict] = []
        triggered = False
        rnd = 0

        while not self._stop:
            elapsed = time.monotonic() - t_start

            if self.mode == "fov_finder":
                self.fov_finder.run()
                features = self.fov_finder.last_run.get("fov_features", {})
            else:
                features = self._scout_positions()

            pooled = (
                pd.concat(features.values(), ignore_index=True)
                if features
                else pd.DataFrame()
            )
            n_cells = int(len(pooled))
            n_fovs_in_pool = len(features)

            row: dict = {
                "round": rnd,
                "elapsed_min": elapsed / 60.0,
                "n_cells_pooled": n_cells,
                "n_fovs_in_pool": int(n_fovs_in_pool),
            }
            all_passed = True

            # --- Per-FOV breakdown: clarifies *why* the pooled count is what
            # it is (e.g. segmentation said 42 but feature-extractor produced
            # only 12 cells, or some candidates failed min_cells / fov_conditions).
            self._log_round_detail(features, rnd, elapsed)
            print(
                f"[FOVConditionMonitorAgent] round {rnd} "
                f"t={elapsed / 60:.0f} min: pool = {n_cells} cells "
                f"from {n_fovs_in_pool} FOV(s)"
            )
            if n_cells == 0:
                print(
                    "[FOVConditionMonitorAgent] WARNING: no cells extracted "
                    "this round — check segmentation channel / feature "
                    "extractor."
                )
            for i, c in enumerate(self.fov_conditions):
                passed_i, frac_i = c.check(pooled)
                median_i = (
                    float(pooled[c.feature].median())
                    if n_cells and c.feature in pooled.columns
                    else float("nan")
                )
                row[f"c{i}_pass"] = bool(passed_i)
                row[f"c{i}_frac"] = float(frac_i)
                row[f"c{i}_median"] = median_i
                all_passed = all_passed and passed_i
                flag = "OK " if passed_i else "..."
                print(
                    f"  [{flag}] c{i} {c.feature} {c.operator} "
                    f"{c.threshold:g}: median={median_i:.3f}, "
                    f"{frac_i * 100:.0f}% pass (target "
                    f"{c.min_fraction * 100:.0f}%)"
                )

            row["triggered"] = bool(all_passed)
            trajectory.append(row)

            if all_passed:
                triggered = True
                print(
                    f"[FOVConditionMonitorAgent] TRIGGERED after "
                    f"{elapsed / 60:.0f} min ({rnd + 1} rounds): all "
                    f"{len(self.fov_conditions)} conditions met."
                )
                break

            if elapsed + self.check_interval_s >= self.timeout_s:
                print(
                    f"[FOVConditionMonitorAgent] TIMEOUT after "
                    f"{elapsed / 60:.0f} min — at least one condition "
                    f"never met. Returning triggered=False; starting "
                    f"anyway is the caller's call."
                )
                break

            rnd += 1
            self._sleep(self.check_interval_s)

        df_traj = pd.DataFrame(trajectory)
        self._save_trajectory(df_traj)

        return {
            "triggered": triggered,
            "n_rounds": len(trajectory),
            "elapsed_min": (time.monotonic() - t_start) / 60.0,
            "trajectory": df_traj,
        }

    def stop(self) -> None:
        """Request a graceful stop before the next round / during a sleep."""
        self._stop = True

    # ------------------------------------------------------------------
    # Per-round diagnostic: which FOVs contributed (and which didn't)
    # ------------------------------------------------------------------

    def _log_round_detail(
        self,
        features: dict[int, pd.DataFrame],
        rnd: int,
        elapsed: float,
    ) -> None:
        """Print one line per scanned FOV: seg count, FE count, pool status.

        Clarifies pooling discrepancies — e.g. segmentation says 42 cells
        but only 12 reach the pool, or some candidates failed ``min_cells``
        and never produced features at all.
        """
        if self.mode == "fov_finder":
            df_scan = self.fov_finder.last_run.get("all_candidates", None)
            if df_scan is None or df_scan.empty:
                return
            print(
                f"[FOVConditionMonitorAgent] round {rnd} "
                f"t={elapsed / 60:.0f} min — per-FOV detail:"
            )
            for _, row in df_scan.iterrows():
                p_idx = int(row["p"])
                seg_n = int(row.get("n_cells", 0))
                fe_n = (
                    int(row["n_cells_features"])
                    if "n_cells_features" in row and pd.notna(row["n_cells_features"])
                    else None
                )
                valid = bool(row.get("valid", False))
                reason = str(row.get("reason", "") or "")
                pool_n = int(len(features[p_idx])) if p_idx in features else 0
                flag = "OK" if (pool_n > 0) else ".."
                fe_str = f"fe={fe_n:>3d}" if fe_n is not None else "fe=  ?"
                tail = f"  pool={pool_n:>3d}" + (f"  [{reason}]" if reason else "")
                print(f"  [{flag}] p{p_idx:<2d}  seg={seg_n:>3d}, {fe_str}{tail}")
        else:
            # Mode B: fixed positions, no df_scan; report what came back.
            positions = self._positions or []
            if not positions:
                return
            print(
                f"[FOVConditionMonitorAgent] round {rnd} "
                f"t={elapsed / 60:.0f} min — per-FOV detail:"
            )
            for p_idx, pos in enumerate(positions):
                pool_n = int(len(features[p_idx])) if p_idx in features else 0
                flag = "OK" if (pool_n > 0) else ".."
                name = getattr(pos, "name", f"p{p_idx}") or f"p{p_idx}"
                print(f"  [{flag}] p{p_idx:<2d}  {name:<20s}  pool={pool_n:>3d}")

    # ------------------------------------------------------------------
    # Position-mode scouting: image fixed positions, segment, extract features
    # ------------------------------------------------------------------

    def _scout_positions(self) -> dict[int, pd.DataFrame]:
        """Image self._positions, segment, run the FE, return per-FOV per-cell DFs.

        Mirrors the relevant subset of :meth:`FOVFinderAgent._acquire_frames`
        and the feature-extraction branch of ``_segment_and_score``, with no
        candidate generation / FPS selection / cell-count filtering.
        """
        positions = self._positions
        assert positions is not None  # resolved at run() start
        n_channels = len(self._imaging_channels)

        seq = RTMSequence(
            time_plan={"interval": 0, "loops": 1},
            stage_positions=positions,
            channels=self._imaging_channels,
            rtm_metadata={"condition_monitor_round": self._round_counter},
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
            except Exception:  # pragma: no cover - backend cleanup
                pass

        self._round_counter += 1

        used_mask = getattr(self._feature_extractor, "used_mask", "labels")
        out: dict[int, pd.DataFrame] = {}
        for p_idx in range(len(positions)):
            channel_imgs: list[np.ndarray] = []
            missing = False
            for c in range(n_channels):
                img = frames.get((p_idx, c))
                if img is None:
                    missing = True
                    break
                channel_imgs.append(img)
            if missing:
                continue
            img_stack = np.stack(channel_imgs, axis=0)
            label_img = self._segmentator.segment(img_stack[self._seg_channel_index])
            try:
                fe_result = self._feature_extractor.extract_features(
                    {used_mask: label_img}, img_stack
                )
            except Exception as e:  # pragma: no cover - user-supplied FE
                print(
                    f"[FOVConditionMonitorAgent] extract_features failed at "
                    f"position {p_idx}: {type(e).__name__}: {e}"
                )
                continue
            df_features = (
                fe_result[0]
                if isinstance(fe_result, tuple) and fe_result
                else fe_result
            )
            if df_features is None or df_features.empty:
                continue
            out[p_idx] = df_features
        return out

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _sleep(self, seconds: float) -> None:
        """Sleep in short chunks so :meth:`stop` stays responsive."""
        deadline = time.monotonic() + seconds
        while not self._stop:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(10.0, remaining))

    def _save_trajectory(self, df_traj: pd.DataFrame) -> None:
        """Write the per-round trajectory to parquet for post-hoc plotting."""
        if not self.storage_path or df_traj.empty:
            return
        out = os.path.join(self.storage_path, "condition_monitor.parquet")
        try:
            os.makedirs(self.storage_path, exist_ok=True)
            df_traj.to_parquet(out)
            print(f"[FOVConditionMonitorAgent] trajectory saved -> {out}")
        except Exception as e:  # pragma: no cover - best-effort I/O
            print(f"[FOVConditionMonitorAgent] could not save trajectory: {e}")
