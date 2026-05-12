"""Batch BO agent for ERK oscillation frequency optimisation.

Subclasses :class:`faro.agents.bo_optimization.BOptGPAX` to specialise it for
the optogenetic ERK-KTR oscillation experiment:

* Each phase tests ``n_conditions_per_iter`` distinct (``stim_exposure``,
  ``ramp``) parameter sets simultaneously, splitting the configured FOVs
  evenly between them.
* Per-condition stimulation events apply a per-frame ramped exposure
  (``base_exposure + ramp * frame_index``) instead of a flat exposure.
* :meth:`_preprocess_results` runs a pre-trained sliding-window
  oscillation classifier on each cell's ``cnr`` trace and returns
  the per-FOV oscillating fraction.

All batch-BO machinery (sequential greedy acquisition, FOV-index offsets
across phases, ``run_one_phase`` integration with :class:`ComposedAgent`)
is inherited unchanged from :class:`BOptGPAX`.

The class is generic over the GP backend: pair it with
:class:`faro.agents.bo_optimization_sparse.BOptGPAXSparse` via multiple
inheritance to use the variational sparse GP backend instead of ExactGP.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

import faro.core.utils as utils
from faro.agents.bo_optimization import BOptGPAX
from faro.core.data_structures import RTMSequence

if TYPE_CHECKING:
    import matplotlib.pyplot as plt  # noqa: F401


class OscillationBO(BOptGPAX):
    """Batch BO agent for ERK oscillation optimisation.

    Each BO phase tests ``n_conditions_per_iter`` conditions
    simultaneously across the configured FOVs (FOVs are split evenly
    between conditions).  When driven by :class:`ComposedAgent` the FOVs
    are *fresh* every phase (chosen by ``FOVFinderAgent``); the GP
    accumulates observations across all phases.

    Args:
        n_frames: Total number of timepoints per FOV (baseline + stim + optocheck).
        first_frame_stim: First frame index that receives stimulation.
        last_frame_stim: First frame *after* the stim window (Python slice end).
        time_between_timesteps: Seconds between consecutive frames.
        imaging_channels: Tuple of imaging channels acquired every frame.
        stim_channel: Stimulation channel (exposure overridden per condition).
        optocheck_channel: Channel acquired only on the final frame for
            mCitrine optoRTK expression readout.
        osc_clf, osc_scaler, osc_feature_cols, osc_cfg: Pre-loaded
            oscillation classifier components.
        osc_predict_fn: Sliding-window predict function returning a list
            of ``{"osc_probability": ..., "predicted_osc": ...}`` dicts
            per cell trace.
        min_osc_probability: Threshold (``>=``) on max window
            probability for a cell to be classified oscillating.
        min_consecutive_windows: Threshold (``>=``) on the longest
            run of consecutive ``predicted_osc == 1`` windows.
        min_fft_amplitude: Threshold (``>=``) on the FFT amplitude
            score (``longest_osc_frac * fft_peak_amplitude``).  Set to
            ``0.0`` to disable the FFT gate.
        osc_score_window: ``(start_frame, end_frame)`` half-open interval
            restricting which *classifier windows* contribute to the
            per-cell oscillation score.  A window is included when
            ``window_start >= start_frame`` and ``window_end <= end_frame``.
            Defaults to ``(first_frame_stim, last_frame_stim)``—i.e. only
            windows that fall entirely within the stimulation period.
        fft_period_band: ``(min_period, max_period)`` in timesteps.  The
            FFT score only considers spectral peaks whose period lies in
            this range.  Default ``(4, 40)``.
        n_baseline_frames: Number of frames at the start of the trace
            (timesteps ``[0, n_baseline_frames)``) used to compute the
            ``baseline_cnr`` covariate.  Defaults to ``first_frame_stim``
            (i.e. all frames before stimulation begins) which is the
            biologically natural choice.  Override only if you want to
            include or exclude specific pre-stim frames.
        min_track_fraction: Minimum fraction of the experiment a cell
            must be tracked for (by ``fov_timestep`` count) to be
            included in the oscillation classification.  Cells tracked
            for fewer than ``min_track_fraction * n_frames`` timepoints
            are excluded from both the numerator and denominator of
            ``frac_oscillating``.  Default ``0.9`` (90%).
        max_baseline_cnr: Per-cell mean baseline CNR threshold.  Cells
            whose mean ``cnr`` over the first ``n_baseline_frames`` is
            **above** this value are excluded from the oscillation
            classification (both numerator and denominator).  Filters
            out cells that are already highly active before stimulation
            or that are segmentation artefacts.  Default ``0.8``.
            Set to ``None`` to disable this filter.
        classifier_window: ``(start_frame, end_frame)`` half-open
            interval restricting which timesteps the oscillation
            classifier sees per cell.  Defaults to
            ``(first_frame_stim, last_frame_stim)`` so the classifier
            only sees the **stimulation window** and ignores any
            baseline or recovery frames either side — biologically
            correct (no stim → no oscillations to find), and lets you
            add a post-stim recovery period to the experiment without
            polluting the classifier with all-zero traces.  Pass
            ``(0, n_frames)`` to look at the full trace (legacy
            behaviour).  The slice is applied after sorting by
            ``timestep`` and before the per-cell length check.
        save_checkpoints: If True, write ``df_results`` to a parquet
            checkpoint after every phase.
        plot_live: If True, plot per-phase progress with matplotlib.
        **kwargs: Forwarded to :class:`BOptGPAX` (must include
            ``n_conditions_per_iter`` for batch BO).
    """

    def __init__(
        self,
        *,
        n_frames: int,
        first_frame_stim: int,
        last_frame_stim: int,
        time_between_timesteps: float,
        imaging_channels,
        stim_channel,
        optocheck_channel,
        osc_clf,
        osc_scaler,
        osc_feature_cols,
        osc_cfg,
        osc_predict_fn,
        min_osc_probability: float = 0.95,
        min_consecutive_windows: int = 3,
        min_fft_amplitude: float = 0.3,
        osc_score_window: tuple[int, int] | None = None,
        fft_period_band: tuple[float, float] = (4.0, 40.0),
        n_baseline_frames: int | None = None,
        min_track_fraction: float = 0.9,
        max_baseline_cnr: float | None = 0.8,
        classifier_window: tuple[int, int] | None = None,
        frac_responder_threshold: float = 0.75,
        save_checkpoints: bool = True,
        plot_live: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.n_frames = n_frames
        self.first_frame_stim = first_frame_stim
        self.last_frame_stim = last_frame_stim
        self.time_between_timesteps = time_between_timesteps
        self.imaging_channels = imaging_channels
        self.stim_channel = stim_channel
        self.optocheck_channel = optocheck_channel

        self.osc_clf = osc_clf
        self.osc_scaler = osc_scaler
        self.osc_feature_cols = osc_feature_cols
        self.osc_cfg = osc_cfg
        self.osc_predict_fn = osc_predict_fn
        self.min_osc_probability = float(min_osc_probability)
        self.min_consecutive_windows = int(min_consecutive_windows)
        self.min_fft_amplitude = float(min_fft_amplitude)
        self.fft_period_band = (float(fft_period_band[0]), float(fft_period_band[1]))

        # Default baseline window = all pre-stim frames (the biologically
        # natural choice).  In the test notebook with first_frame_stim=1
        # this gives a 1-frame baseline; in a real 60-frame experiment
        # with first_frame_stim=10 this gives a 10-frame baseline.
        if n_baseline_frames is None:
            n_baseline_frames = first_frame_stim
        if n_baseline_frames < 1:
            raise ValueError(f"n_baseline_frames must be >= 1, got {n_baseline_frames}")
        self.n_baseline_frames = int(n_baseline_frames)
        self.min_track_fraction = float(min_track_fraction)
        self.max_baseline_cnr = (
            float(max_baseline_cnr) if max_baseline_cnr is not None else None
        )
        self.frac_responder_threshold = float(frac_responder_threshold)

        # Default classifier window = the stim window itself.  Lets us
        # add a recovery period (frames > last_frame_stim) without
        # polluting the classifier with post-stim frames where, by
        # design, the optogenetic input has been switched off and any
        # remaining oscillations are decay artefacts of the prior
        # stimulation rather than the BO-relevant signal.
        if classifier_window is None:
            classifier_window = (first_frame_stim, last_frame_stim)
        clf_start, clf_end = classifier_window
        if clf_start < 0 or clf_end <= clf_start:
            raise ValueError(
                f"classifier_window must be a half-open interval with "
                f"0 <= start < end, got {classifier_window}"
            )
        if clf_end > n_frames:
            raise ValueError(
                f"classifier_window end ({clf_end}) exceeds n_frames " f"({n_frames})"
            )
        self.classifier_first_frame = int(clf_start)
        self.classifier_last_frame = int(clf_end)

        # Scoring window: which classifier windows contribute to the
        # per-cell oscillation metrics.  Defaults to the stim window.
        if osc_score_window is None:
            osc_score_window = (first_frame_stim, last_frame_stim)
        self.osc_score_window = (int(osc_score_window[0]), int(osc_score_window[1]))

        self.save_checkpoints = bool(save_checkpoints)
        self.plot_live = bool(plot_live)

    @property
    def fovs_per_condition(self) -> int:
        """Number of FOVs assigned to each condition in the current batch."""
        if not self.fovs:
            raise RuntimeError(
                "fovs_per_condition is undefined until FOVs have been "
                "configured (call run_one_phase with fov_positions=...)."
            )
        return len(self.fovs) // self.n_conditions_per_iter

    # ------------------------------------------------------------------
    # Event creation
    # ------------------------------------------------------------------

    def _create_events_for_cycle(self, parameters: dict) -> list:
        raise NotImplementedError(
            "OscillationBO is batch-only -- use _create_events_for_batch."
        )

    def _create_events_for_batch(self, param_list: list[dict]) -> list:
        """Build RTMEvents for all conditions in one batch iteration.

        Creates one RTMSequence per condition (each with
        :attr:`fovs_per_condition` FOVs), remaps FOV indices to globally
        unique values using :attr:`_fov_index_offset`, populates
        :attr:`_current_condition_map` so :meth:`_preprocess_results` can
        recover the originating parameters per FOV, and merges /
        FOV-batches the events.
        """
        phase_id = self._phase_counter
        all_events = []
        fovs_per_condition = self.fovs_per_condition

        self._current_condition_map = {}

        for cond_idx, params in enumerate(param_list):
            start = cond_idx * fovs_per_condition
            end = start + fovs_per_condition
            cond_positions = self.fov_positions[start:end]

            # _current_condition_map is keyed by GLOBAL fov index so that
            # _preprocess_results (which iterates self.fovs == globals)
            # can look up the right BO params for each FOV.
            for fov_idx in range(start, end):
                self._current_condition_map[fov_idx + self._fov_index_offset] = params

            base_exp = float(params["stim_exposure"])
            ramp = float(params["ramp"])
            # Optional per-condition pulse interval (frames between pulses).
            # Defaults to 1 (= every frame, i.e. every minute at 1 fr/min,
            # the original behaviour).  The BO can treat this as an
            # additional control axis by declaring
            # ``BO_Parameter(name="pulse_interval", ...)`` in the notebook.
            pulse_interval = max(1, int(params.get("pulse_interval", 1)))
            stim_frames = range(
                self.first_frame_stim,
                self.last_frame_stim,
                pulse_interval,
            )
            n_stim_frames = len(stim_frames)
            exposures = tuple(base_exp + ramp * i for i in range(n_stim_frames))

            acq = RTMSequence(
                time_plan={
                    "interval": self.time_between_timesteps,
                    "loops": self.n_frames,
                },
                stage_positions=cond_positions,
                channels=self.imaging_channels,
                stim_channels=(self.stim_channel,),
                stim_frames=stim_frames,
                stim_exposure=exposures,
                ref_channels=(self.optocheck_channel,),
                ref_frames=frozenset({self.n_frames - 1}),
                rtm_metadata={
                    "phase_name": f"BO_iter_{phase_id}_cond_{cond_idx}",
                    "phase_id": phase_id,
                    "condition_idx": cond_idx,
                    "stim_exposure": base_exp,
                    "ramp": ramp,
                    "pulse_interval": pulse_interval,
                },
            )

            # Remap local p indices (0..fovs_per_condition-1) to GLOBAL FOV ids:
            #   condition offset:  cond_idx * fovs_per_condition
            #   phase     offset:  self._fov_index_offset
            p_offset = cond_idx * fovs_per_condition + self._fov_index_offset
            for ev in acq:
                local_p = ev.index.get("p", 0)
                all_events.append(
                    ev.model_copy(
                        update={"index": {**dict(ev.index), "p": local_p + p_offset}}
                    )
                )

        all_events = utils.apply_fov_batching(all_events, time_per_fov=2.0)

        print(f"  Created {len(all_events)} events for phase {phase_id}")
        for i, p in enumerate(param_list):
            exp_start = p["stim_exposure"]
            pi = max(1, int(p.get("pulse_interval", 1)))
            n_pulses_i = len(range(self.first_frame_stim, self.last_frame_stim, pi))
            exp_end = exp_start + p["ramp"] * (n_pulses_i - 1)
            fov_start = i * fovs_per_condition + self._fov_index_offset
            fov_end = fov_start + fovs_per_condition - 1
            print(
                f"    Cond {i} (FOVs {fov_start}-{fov_end}): "
                f"stim_exposure={exp_start:.0f}ms, ramp={p['ramp']:.0f}ms/frame, "
                f"pulse_interval={pi}min, n_pulses={n_pulses_i} "
                f"(final exposure: {exp_end:.0f}ms)"
            )
        return all_events

    # ------------------------------------------------------------------
    # Oscillation classification
    # ------------------------------------------------------------------

    def _filter_windows(self, windows: list[dict]) -> list[dict]:
        """Return only windows within :attr:`osc_score_window`."""
        w_lo, w_hi = self.osc_score_window
        return [
            w for w in windows if w["window_start"] >= w_lo and w["window_end"] <= w_hi
        ]

    @staticmethod
    def _score_max_prob(windows: list[dict]) -> float:
        """Max ``osc_probability`` over the given windows."""
        if not windows:
            return 0.0
        return max(w["osc_probability"] for w in windows)

    @staticmethod
    def _score_max_consec(windows: list[dict]) -> int:
        """Longest consecutive run of ``predicted_osc == 1``.

        Uses the ``np.diff([0, ..., 0])`` padding trick from
        ``OscScoreEngine.query_basic``.
        """
        if not windows:
            return 0
        osc_seq = [
            w["predicted_osc"] for w in sorted(windows, key=lambda w: w["window_start"])
        ]
        padded = np.array([0] + osc_seq + [0])
        d = np.diff(padded)
        starts = np.where(d == 1)[0]
        ends = np.where(d == -1)[0]
        if len(starts) == 0:
            return 0
        return int(np.max(ends - starts))

    def _merge_osc_regions(self, windows: list[dict]) -> list[tuple[float, float]]:
        """Merge consecutive windows with ``predicted_osc == 1``.

        Returns a list of ``(start, end)`` time segments.  Overlapping
        windows (from stride < window_size) are merged into contiguous
        segments.
        """
        sorted_wins = sorted(windows, key=lambda w: w["window_start"])
        regions: list[tuple[float, float]] = []
        cur_start = cur_end = None
        for w in sorted_wins:
            if w["predicted_osc"] != 1:
                if cur_start is not None:
                    regions.append((cur_start, cur_end))
                    cur_start = cur_end = None
                continue
            if cur_start is None:
                cur_start, cur_end = w["window_start"], w["window_end"]
            else:
                cur_end = max(cur_end, w["window_end"])
        if cur_start is not None:
            regions.append((cur_start, cur_end))
        return regions

    def _score_fft(
        self,
        windows: list[dict],
        x: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """FFT amplitude score: ``longest_osc_frac * fft_peak_amplitude``.

        Matches ``OscScoreEngine.query(metric="fft")``:
        1. Merge consecutive oscillation windows into contiguous regions.
        2. ``longest_frac`` = duration of longest region / trace duration.
        3. Extract signal values from *all* merged regions and compute FFT.
        4. Peak amplitude in the period band :attr:`fft_period_band`.
        5. Score = ``longest_frac * peak_amplitude``.
        """
        regions = self._merge_osc_regions(windows)
        if not regions:
            return 0.0

        w_lo, w_hi = self.osc_score_window
        trace_duration = float(w_hi - w_lo)
        if trace_duration <= 0:
            return 0.0

        durations = [end - start for start, end in regions]
        longest_frac = max(durations) / trace_duration

        # Extract signal from all oscillation regions
        osc_mask = np.zeros(len(x), dtype=bool)
        for start, end in regions:
            osc_mask |= (x >= start) & (x < end)
        y_osc = y[osc_mask]
        if len(y_osc) < 4:
            return 0.0

        # FFT: Hanning window, find peak amplitude in period band
        n = len(y_osc)
        w = np.hanning(n)
        fft_vals = np.fft.rfft(y_osc * w)
        freqs = np.fft.rfftfreq(n, d=1.0)

        min_period, max_period = self.fft_period_band
        freq_lo = 1.0 / max_period
        freq_hi = 1.0 / min_period
        band_mask = (freqs >= freq_lo) & (freqs <= freq_hi)
        if not np.any(band_mask):
            return 0.0

        amplitudes = 2.0 * np.abs(fft_vals[band_mask]) / n
        peak_amplitude = float(np.max(amplitudes))

        return longest_frac * peak_amplitude

    def _is_cell_oscillating(
        self,
        windows: list[dict],
        x: np.ndarray | None = None,
        y: np.ndarray | None = None,
    ) -> bool:
        """Classify a cell as oscillating using three metrics.

        A cell is oscillating iff **all** of the following hold (over
        windows within :attr:`osc_score_window`):

        1. ``max(osc_probability) >= min_osc_probability``
        2. ``max_consecutive_osc_windows >= min_consecutive_windows``
        3. ``fft_amplitude_score >= min_fft_amplitude``

        Args:
            windows: Per-window classifier output (full trace).
            x: Timestep array for the cell (needed for FFT score).
            y: Signal array (``cnr`` or ``cnr_median``) for the cell.
        """
        scored = self._filter_windows(windows)
        if not scored:
            return False

        if self._score_max_prob(scored) < self.min_osc_probability:
            return False
        if self._score_max_consec(scored) < self.min_consecutive_windows:
            return False

        if self.min_fft_amplitude > 0 and x is not None and y is not None:
            if self._score_fft(scored, x, y) < self.min_fft_amplitude:
                return False

        return True

    # ------------------------------------------------------------------
    # Result preprocessing
    # ------------------------------------------------------------------

    def _preprocess_results(self, fov_tracks: dict) -> pd.DataFrame:
        """Classify oscillations and compute per-FOV metrics.

        Reads :attr:`_current_phase_id` and :attr:`_current_condition_map`
        from instance state -- both are populated by
        :meth:`BOptGPAX._run_one_phase_batch` (and
        :meth:`_create_events_for_batch`) before this is called.
        """
        phase_id = self._current_phase_id
        results = []
        for fov_idx, df_tracks in fov_tracks.items():
            if df_tracks.empty or "particle" not in df_tracks.columns:
                continue

            # --- Filter to current phase only ---
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

            # --- Per-cell tracking length + baseline CNR -----------------
            # Compute these upfront so we can filter cells before the
            # oscillation classification loop.  Both the numerator
            # (n_oscillating) and denominator (n_cells) of
            # frac_oscillating should only count cells that pass the
            # quality gates: (1) tracked for >= min_track_fraction of
            # the experiment, and (2) baseline CNR below max_baseline_cnr
            # (filters out already-active / artefact cells).
            #
            # The FOV-level covariates (n_cells, optortk_expression,
            # baseline_cnr) are computed on ALL cells in the FOV
            # (before filtering), because they describe the FOV context
            # as a whole, not just the classifiable subset.

            all_particles = df_phase["particle"].unique()
            n_cells_total = len(all_particles)

            # Per-cell frame count (for tracking-length filter)
            min_frames = int(self.min_track_fraction * self.n_frames)
            frames_per_cell = df_phase.groupby("particle")["fov_timestep"].nunique()

            # Per-cell baseline CNR (support both "cnr" and "cnr_median")
            per_cell_baseline = pd.Series(dtype=float)
            cnr_col = (
                "cnr"
                if "cnr" in df_phase.columns
                else "cnr_median" if "cnr_median" in df_phase.columns else None
            )
            if cnr_col is not None:
                baseline_df = df_phase[
                    df_phase["fov_timestep"] < self.n_baseline_frames
                ]
                if not baseline_df.empty:
                    per_cell_baseline = (
                        baseline_df.groupby("particle")[cnr_col].mean().dropna()
                    )

            # FOV-level baseline_cnr covariate (all cells, before filter)
            baseline_cnr = (
                float(per_cell_baseline.mean()) if len(per_cell_baseline) > 0 else 0.0
            )

            # optoRTK expression covariate (all cells, before filter).
            # Drop NaN rows BEFORE the groupby so .first() can't return
            # NaN from a pre-ref-frame row (which happens on every
            # timestep except the ref frame itself).  Depending on the
            # pandas version `.first()` may or may not skip NaN by
            # default — earlier code relied on that behaviour and
            # silently produced an empty optortk_vals array on older
            # pandas, causing the FOV to be discarded even when the ref
            # measurement had succeeded.
            optortk_vals = []
            if "ref_mean_intensity" in df_phase.columns:
                ref_rows = df_phase.dropna(subset=["ref_mean_intensity"])
                if not ref_rows.empty:
                    optortk_vals = (
                        ref_rows.groupby("particle")["ref_mean_intensity"]
                        .first()
                        .values
                    )

            # --- Build set of cells that pass quality gates ------
            valid_particles = set(all_particles)

            # Gate 1: tracked for >= min_track_fraction of n_frames
            valid_particles &= set(frames_per_cell[frames_per_cell >= min_frames].index)

            # Gate 2: baseline CNR below threshold
            if self.max_baseline_cnr is not None and len(per_cell_baseline) > 0:
                valid_particles &= set(
                    per_cell_baseline[per_cell_baseline < self.max_baseline_cnr].index
                )

            n_cells = len(valid_particles)
            n_oscillating = 0

            # --- Oscillation classification (filtered cells only) ---
            # Collect per-cell scores so we can compute BOTH the binary
            # gated `frac_oscillating` (original objective, 65%-zero on
            # sparse data) AND continuous aggregates that give the GP a
            # gradient everywhere: `mean_osc_probability`,
            # `mean_max_consecutive`, `mean_fft_amplitude`.  Switch the
            # BO to any of them by changing BO_Objective(name=...)
            # in the notebook — default stays `frac_oscillating` for
            # backward compatibility.
            per_cell_max_prob = []
            per_cell_mean_prob = []
            per_cell_max_consec = []
            per_cell_fft_score = []
            for particle, grp in df_phase.groupby("particle"):
                if particle not in valid_particles:
                    continue
                grp = grp.sort_values("fov_timestep")
                if "cnr" not in grp.columns and "cnr_median" not in grp.columns:
                    continue
                # Restrict to the classifier window (defaults to the
                # stim window).
                grp = grp[
                    (grp["fov_timestep"] >= self.classifier_first_frame)
                    & (grp["fov_timestep"] < self.classifier_last_frame)
                ]
                if len(grp) < 20:
                    continue
                x = grp["fov_timestep"].values.astype(float)
                cnr_col = "cnr" if "cnr" in grp.columns else "cnr_median"
                y = grp[cnr_col].values.astype(float)
                windows = self.osc_predict_fn(
                    x,
                    y,
                    self.osc_clf,
                    self.osc_scaler,
                    self.osc_feature_cols,
                    self.osc_cfg,
                )
                scored = self._filter_windows(windows)
                per_cell_max_prob.append(self._score_max_prob(scored))
                # Mean osc_probability over windows — rewards duration
                # AND strength jointly (unlike max, which only captures
                # peak strength).  A cell oscillating throughout the
                # stim window scores ~4x higher than one with a single
                # brief high-prob burst, given window_size=40/stride=10.
                per_cell_mean_prob.append(
                    float(np.mean([w["osc_probability"] for w in scored]))
                    if scored
                    else 0.0
                )
                per_cell_max_consec.append(self._score_max_consec(scored))
                per_cell_fft_score.append(self._score_fft(scored, x, y))
                if self._is_cell_oscillating(windows, x=x, y=y):
                    n_oscillating += 1

            if n_cells < 5:
                print(
                    f"  Warning: FOV {fov_idx} has only {n_cells} valid "
                    f"cells (of {n_cells_total} total), skipping"
                )
                continue

            frac_oscillating = n_oscillating / n_cells
            mean_osc_probability = (  # outer mean of per-cell max osc_prob
                float(np.mean(per_cell_max_prob)) if per_cell_max_prob else 0.0
            )
            # Outer **median** of per-cell mean osc_prob across the FOV.
            # Column name stays `mean_osc_probability_avg` for back-compat
            # with existing notebooks and checkpoints.  The per-cell
            # baseline_cnr filter is already applied upstream via the
            # `max_baseline_cnr` gate that builds `valid_particles`, so
            # no extra filtering is needed here.  Median (not mean)
            # makes the metric robust to single-cell outliers.
            mean_osc_probability_avg = (
                float(np.median(per_cell_mean_prob)) if per_cell_mean_prob else 0.0
            )
            mean_max_consecutive = (
                float(np.mean(per_cell_max_consec)) if per_cell_max_consec else 0.0
            )
            mean_fft_amplitude = (
                float(np.mean(per_cell_fft_score)) if per_cell_fft_score else 0.0
            )
            # Fraction of cells with per-cell mean osc_probability above
            # frac_responder_threshold.  Unlike `frac_oscillating` (which
            # requires the full 3-gate rule and tends to saturate at 0 or 1),
            # this captures *how many cells the classifier considers strong
            # responders on average over the stim window* — a smoother BO
            # target bounded in [0, 1].
            frac_responders = (
                float(
                    np.mean(
                        np.asarray(per_cell_mean_prob) >= self.frac_responder_threshold
                    )
                )
                if per_cell_mean_prob
                else 0.0
            )
            # Use mean (not median) so the covariate responds to atypical
            # cells: the BO needs to know when a FOV has unusually high
            # or low optoRTK expression, not just the typical value.
            optortk_expression = (
                float(np.mean(optortk_vals)) if len(optortk_vals) > 0 else 0.0
            )
            # Skip FOVs without a positive optoRTK reading: if the user
            # declares `BO_Covariate(..., log_scale=True)` the GP scaler
            # takes log() of this column, and log(0) -> -inf corrupts
            # the covariance matrix during MCMC (manifests as
            # "MultivariateNormal got invalid covariance_matrix").
            if not np.isfinite(optortk_expression) or optortk_expression <= 0.0:
                n_ref_cells = len(optortk_vals)
                if n_ref_cells == 0:
                    reason = "no cell had a valid ref_mean_intensity reading"
                elif np.all(np.asarray(optortk_vals) == 0.0):
                    reason = f"all {n_ref_cells} ref readings were exactly 0"
                else:
                    reason = (
                        f"mean {optortk_expression!r} over {n_ref_cells} cells "
                        f"(min={float(np.min(optortk_vals)):.3f})"
                    )
                print(
                    f"  Warning: FOV {fov_idx} has no valid optoRTK "
                    f"expression ({reason}); skipping"
                )
                continue

            results.append(
                {
                    "fov": int(fov_idx),
                    "phase_id": int(phase_id),
                    "stim_exposure": params["stim_exposure"],
                    "ramp": params["ramp"],
                    "pulse_interval": int(params.get("pulse_interval", 1)),
                    "n_cells": float(n_cells),
                    "optortk_expression": optortk_expression,
                    "baseline_cnr": baseline_cnr,
                    "frac_oscillating": frac_oscillating,
                    "mean_osc_probability": mean_osc_probability,
                    "mean_osc_probability_avg": mean_osc_probability_avg,
                    "mean_max_consecutive": mean_max_consecutive,
                    "mean_fft_amplitude": mean_fft_amplitude,
                    "frac_responders": frac_responders,
                }
            )

        if not results:
            print(f"Warning: no valid FOVs in phase {phase_id}")
            return pd.DataFrame()

        df = pd.DataFrame(results)
        print(
            f"  Phase {phase_id}: {len(df)} FOVs, "
            f"mean frac_oscillating={df['frac_oscillating'].mean():.4f}, "
            f"max={df['frac_oscillating'].max():.4f}"
        )
        return df

    # ------------------------------------------------------------------
    # Per-phase hook (live plot + checkpoint)
    # ------------------------------------------------------------------

    def _on_phase_complete(self, df_new: pd.DataFrame, phase_id: int) -> None:
        if self.save_checkpoints:
            self._save_checkpoint(self.df_results)
            # Persist the trained GP + scalers + metadata to
            # ``{storage_path}/bo_model.joblib`` after every phase.
            # Overwrites in place so the file always reflects the latest
            # fit — even a crash mid-experiment leaves the most recent
            # model on disk for post-hoc analysis.  No-op when called
            # during initial-spread phases (no model yet).
            self.save_model()
        if self.plot_live:
            self._plot_live(self.df_results, f"After phase {phase_id + 1}")

    def _save_checkpoint(self, df_results: pd.DataFrame) -> None:
        # Cumulative (latest state, overwritten each phase)
        ckpt_path = os.path.join(self.storage_path, "bo_results_checkpoint.parquet")
        df_results.to_parquet(ckpt_path, index=False)
        # Per-phase snapshot
        ckpt_dir = os.path.join(self.storage_path, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        df_results.to_parquet(
            os.path.join(
                ckpt_dir, f"bo_results_phase_{self._current_phase_id:03d}.parquet"
            ),
            index=False,
        )

    def _plot_bo_landscape(self, *args, **kwargs) -> None:
        """Suppress the base-class 2x2 plot — replaced by _plot_live."""
        pass

    def _plot_live(
        self,
        df_results: pd.DataFrame,
        iteration_label: str,
        save_subdir: str = "after",
    ) -> None:
        """1x3 plot: measured points, GP landscape, acquisition function.

        Reuses the pre-computed data from ``_last_plot_context`` (stashed
        by the base class during ``_determine_next_parameters``) so no
        GP re-prediction is needed — exactly the same data the base
        class ``_plot_bo_landscape`` would have used.
        """
        import jax.numpy as jnp
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
        fig.suptitle(iteration_label, fontsize=13, fontweight="bold")

        # --- Subplot 1: measured objective (with jitter) ---
        # Use the configured objective name so subclasses (e.g. single-cell
        # with "mean_osc_probability") plot the right column.
        obj_name = self.objective_metric.name
        jitter_x = self._rng.normal(0, 3.0, size=len(df_results))
        jitter_y = self._rng.normal(0, 0.5, size=len(df_results))
        sc = axes[0].scatter(
            df_results["stim_exposure"].values + jitter_x,
            df_results["ramp"].values + jitter_y,
            c=df_results[obj_name],
            cmap="viridis",
            s=30,
            edgecolors="k",
            linewidths=0.3,
            alpha=0.8,
        )
        axes[0].set_xlabel("stim_exposure (ms)")
        axes[0].set_ylabel("ramp (ms/frame)")
        axes[0].set_title(f"Measured {obj_name}")
        fig.colorbar(sc, ax=axes[0], label=obj_name)

        # --- Subplots 2 & 3: GP landscape + acquisition ----------------
        ctx = getattr(self, "_last_plot_context", None)
        if ctx is None:
            for ax in axes[1:]:
                ax.text(
                    0.5,
                    0.5,
                    "GP not fit yet\n(initial batch)",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=11,
                    color="gray",
                )
                ax.set_xlabel("stim_exposure (ms)")
                ax.set_ylabel("ramp (ms/frame)")
            axes[1].set_title("GP predicted landscape")
            axes[2].set_title("Acquisition")
        else:
            try:
                self._plot_landscape_and_acq_from_context(
                    ctx,
                    df_results,
                    axes[1],
                    axes[2],
                    fig,
                )
            except Exception as e:
                for ax in axes[1:]:
                    ax.text(
                        0.5,
                        0.5,
                        f"Plot failed:\n{type(e).__name__}: {e}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                        fontsize=9,
                        color="red",
                    )

        plt.tight_layout()

        # Save to disk
        plots_dir = os.path.join(self.storage_path, "plots", save_subdir)
        os.makedirs(plots_dir, exist_ok=True)
        fig.savefig(
            os.path.join(plots_dir, f"phase_{self._current_phase_id:03d}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            os.path.join(plots_dir, f"phase_{self._current_phase_id:03d}.svg"),
            bbox_inches="tight",
        )
        plt.show()

    # ------------------------------------------------------------------
    # Shared helper: render GP landscape + acquisition from cached context
    # ------------------------------------------------------------------

    def _plot_landscape_and_acq_from_context(
        self,
        ctx: dict,
        df_results,
        ax_mean,
        ax_acq,
        fig,
    ) -> None:
        """Build the GP landscape + acquisition heatmaps.

        Uses the same prediction approach as ``_plot_bo_landscape`` in
        the base class: ``gp_model.predict_in_batches`` on the full
        (ctrl x cov) grid, then marginalise.
        """
        import jax.numpy as jnp

        gp_model = ctx["gp_model"]
        x_scaler = ctx["x_scaler"]
        y_scaler = ctx["y_scaler"]
        rng_key_predict = ctx["rng_key_predict"]
        acq_values_total = ctx["acq_values_total"]
        acquisition_used = ctx["acquisition_used"]
        x_unmeasured = ctx["x_unmeasured_at_computation"]

        x_total_ctrl = self.x_total_linespace.copy()
        unique_x1 = np.unique(x_total_ctrl[:, 0])
        unique_x2 = np.unique(x_total_ctrl[:, 1])
        n_ctrl = len(unique_x1) * len(unique_x2)

        # Covariate samples for marginalisation (same approach as base class)
        if len(self.bo_covariates) > 0 and not df_results.empty:
            cov_cols = [c.name for c in self.bo_covariates]
            cov_vals_full = np.asarray(df_results[cov_cols].to_numpy(), dtype=float)
            n_cov_samples = 50
            _plot_rng = np.random.default_rng(0)
            row_idx = _plot_rng.integers(0, cov_vals_full.shape[0], size=n_cov_samples)
            cov_samples_joint = cov_vals_full[row_idx]
        else:
            n_cov_samples = 1
            cov_samples_joint = None

        ctrl_grid = np.array([[x1, x2] for x1 in unique_x1 for x2 in unique_x2])

        if cov_samples_joint is not None:
            x_grid_full = np.hstack(
                [
                    np.repeat(ctrl_grid, n_cov_samples, axis=0),
                    np.tile(cov_samples_joint, (n_ctrl, 1)),
                ]
            )
        else:
            x_grid_full = ctrl_grid

        x_grid_scaled = x_scaler.transform(x_grid_full)

        # Batched prediction (same as base class _plot_bo_landscape).
        # Use the shared safe-batch-size helper to dodge gpax's two
        # split_in_batches bugs (UnboundLocalError when batch_size >= n,
        # scalar concat failure when remainder is 1).
        from faro.agents.bo_optimization_sparse import _safe_batch_size

        n_rows = np.asarray(x_grid_scaled).shape[0]
        _bs = _safe_batch_size(n_rows, 1000)
        y_pred_scaled, y_samples_scaled = gp_model.predict_in_batches(
            rng_key_predict,
            x_grid_scaled,
            batch_size=_bs,
            n=self.ei_num_samples,
            noiseless=True,
        )
        y_pred = y_scaler.inverse_transform(
            np.asarray(y_pred_scaled).reshape(-1, 1)
        ).flatten()

        # Marginalise GP mean
        if len(self.bo_covariates) > 0:
            y_pred_marg = y_pred.reshape(n_ctrl, n_cov_samples).mean(axis=1)
        else:
            y_pred_marg = y_pred

        X_mesh, Y_mesh = np.meshgrid(unique_x1, unique_x2, indexing="ij")
        y_pred_2d = y_pred_marg.reshape(len(unique_x1), len(unique_x2))

        # --- Subplot: GP predicted landscape ---
        im1 = ax_mean.pcolormesh(
            X_mesh,
            Y_mesh,
            y_pred_2d,
            cmap="viridis",
            shading="auto",
        )
        fig.colorbar(im1, ax=ax_mean, label=f"predicted {self.objective_metric.name}")
        ax_mean.scatter(
            df_results["stim_exposure"],
            df_results["ramp"],
            c="white",
            s=15,
            alpha=0.6,
            marker="x",
            linewidths=0.8,
        )
        ax_mean.set_xlabel("stim_exposure (ms)")
        ax_mean.set_ylabel("ramp (ms/frame)")
        ax_mean.set_title("GP predicted landscape (marginalised)")

        # --- Subplot: Acquisition function ---
        # Use pre-computed acq values; pad to full grid if needed
        acq_marg = np.asarray(acq_values_total)
        if len(acq_marg) != n_ctrl:
            acq_full = np.zeros(n_ctrl)
            x_full = self.x_total_linespace
            for j, pt in enumerate(x_unmeasured):
                diffs = np.abs(x_full - pt).sum(axis=1)
                idx = np.argmin(diffs)
                if j < len(acq_marg):
                    acq_full[idx] = float(acq_marg[j])
            acq_marg = acq_full

        acq_2d = acq_marg.reshape(len(unique_x1), len(unique_x2))
        im2 = ax_acq.pcolormesh(
            X_mesh,
            Y_mesh,
            acq_2d,
            cmap="inferno",
            shading="auto",
        )
        acq_label = acquisition_used.upper()
        fig.colorbar(im2, ax=ax_acq, label=f"{acq_label} acquisition")
        ax_acq.scatter(
            df_results["stim_exposure"],
            df_results["ramp"],
            c="white",
            s=15,
            alpha=0.6,
            marker="x",
            linewidths=0.8,
        )

        # Overlay next measurement positions (cyan crosses)
        picks = getattr(self, "_current_batch_picks", None)
        if picks:
            picks_arr = np.array(picks, dtype=float)
            ax_acq.scatter(
                picks_arr[:, 0],
                picks_arr[:, 1],
                c="cyan",
                s=120,
                marker="X",
                edgecolors="k",
                linewidths=1.0,
                zorder=10,
                label="next conditions",
            )
            ax_acq.legend(loc="upper right", fontsize=7)

        ax_acq.set_xlabel("stim_exposure (ms)")
        ax_acq.set_ylabel("ramp (ms/frame)")
        ax_acq.set_title(f"Acquisition {acq_label} (marginalised)")
