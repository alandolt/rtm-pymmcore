"""Tests for agent mode: the agent framework + controller integration.

Covers the ``feat/agent_mode`` features the rest of the suite does not, and
that were reconciled against the async-controller rework during the rebase
onto main.  The emphasis here is on *failure and recovery* behaviour:

* :class:`FOVCondition` degrades gracefully on empty / NaN / missing-column
  data instead of raising (the monitor polls noisy live data).
* ``get_position`` recovers to ``None`` when the backend can't report XY,
  rather than letting an ``AttributeError`` escape into the monitor.
* Zero-dose control: a ``stim_exposure <= 0`` request yields *no* stim pulse
  at all — no SLM event, no hardware call, no camera-minimum validation —
  while a real exposure preserves a ``PowerChannel`` (power not dropped).
* Per-phase event snapshots survive the cumulative ``events.json.gz`` being
  overwritten every phase (offline-replay provenance).
* ``offset_timepoints=False`` keeps fresh-FOV phases on 0-based ``t`` instead
  of shifting them to nonsensical frame numbers.

Run-based tests use the *non-blocking* controller API correctly
(``handle = ctrl.run_experiment(...); handle.wait()``) — unlike the legacy
``run_and_wait`` fixture, which touches ``ctrl._analyzer`` before the worker
thread has created it.
"""

from __future__ import annotations

import gzip
import json
import os
import time

import pandas as pd
import pytest

from faro.agents.base import Agent, InterPhaseAgent
from faro.agents.fov_finder import FOVCondition
from faro.core.controller import Controller
from faro.core.conversion import load_events_json
from faro.core.data_structures import PowerChannel, RTMSequence
from faro.core.writers import TiffWriter
from faro.microscope.base import AbstractMicroscope
from faro.microscope.pymmcore import PyMMCoreMicroscope
from faro.tracking.trackpy import TrackerTrackpy

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline

# ==========================================================================
# FOVCondition — the cell-feature gate (graceful degradation on bad data)
# ==========================================================================


def _cells(values) -> pd.DataFrame:
    return pd.DataFrame({"cnr": list(values)})


class TestFOVCondition:
    @pytest.mark.parametrize(
        "operator, threshold, expect_frac",
        [
            ("below", 1.0, 0.5),  # 2 of 4 below 1.0
            ("below_or_equal", 1.0, 0.75),  # 3 of 4 <= 1.0
            ("above", 1.0, 0.25),  # 1 of 4 above 1.0
            ("above_or_equal", 1.0, 0.5),  # 2 of 4 >= 1.0
            ("equal", 1.0, 0.25),  # 1 of 4 == 1.0
        ],
    )
    def test_operators_fraction(self, operator, threshold, expect_frac):
        cond = FOVCondition("cnr", operator, threshold, min_fraction=0.0)
        passed, frac = cond.check(_cells([0.5, 0.8, 1.0, 1.5]))
        assert passed is True  # min_fraction=0 always passes
        assert frac == pytest.approx(expect_frac)

    def test_min_fraction_gates_pass(self):
        cells = _cells([0.5, 0.8, 1.0, 1.5])  # 2/4 = 0.5 below 1.0
        assert FOVCondition("cnr", "below", 1.0, min_fraction=0.5).check(cells)[0]
        assert not FOVCondition("cnr", "below", 1.0, min_fraction=0.75).check(cells)[0]

    # --- recovery: bad data must not raise, just report "not met" ---------

    def test_empty_dataframe_fails_gracefully(self):
        passed, frac = FOVCondition("cnr", "below", 1.0).check(pd.DataFrame())
        assert passed is False and frac == 0.0

    def test_missing_column_fails_gracefully(self):
        passed, frac = FOVCondition("cnr", "below", 1.0).check(
            pd.DataFrame({"area_nuc": [100, 200]})
        )
        assert passed is False and frac == 0.0

    def test_all_nan_fails_gracefully(self):
        passed, frac = FOVCondition("cnr", "below", 1.0).check(
            _cells([float("nan"), float("nan")])
        )
        assert passed is False and frac == 0.0

    def test_nan_rows_excluded_from_fraction(self):
        # 1 valid below, 1 valid above, 2 NaN -> fraction over the 2 non-NaN
        _, frac = FOVCondition("cnr", "below", 1.0, min_fraction=0.0).check(
            _cells([0.5, 1.5, float("nan"), float("nan")])
        )
        assert frac == pytest.approx(0.5)

    def test_invalid_operator_raises(self):
        with pytest.raises(ValueError):
            FOVCondition("cnr", "nope", 1.0)

    @pytest.mark.parametrize("bad", [-0.1, 1.1])
    def test_min_fraction_out_of_range_raises(self, bad):
        with pytest.raises(ValueError):
            FOVCondition("cnr", "below", 1.0, min_fraction=bad)


# ==========================================================================
# get_position — microscope-agnostic XY readout, with error recovery
# ==========================================================================


class TestGetPosition:
    def test_base_returns_none_when_unsupported(self):
        class _Bare(AbstractMicroscope):
            pass

        assert _Bare().get_position() is None

    def test_subclass_override_returns_tuple(self):
        class _WithXY(AbstractMicroscope):
            def get_position(self):
                return (12.5, -7.0)

        assert _WithXY().get_position() == (12.5, -7.0)

    def _bare_pymmcore(self, mmc):
        # Bypass __init__ (atexit hook / hardware) — we only exercise read-out.
        scope = PyMMCoreMicroscope.__new__(PyMMCoreMicroscope)
        scope.mmc = mmc
        return scope

    def test_pymmcore_reads_and_coerces_to_float(self):
        class _MMC:
            def getXYPosition(self):
                return (10, 20)  # ints from the core

        x, y = self._bare_pymmcore(_MMC()).get_position()
        assert (x, y) == (10.0, 20.0)
        assert isinstance(x, float) and isinstance(y, float)

    def test_pymmcore_none_core_returns_none(self):
        assert self._bare_pymmcore(None).get_position() is None

    def test_pymmcore_recovers_when_backend_raises(self):
        # A stage that errors (disconnected, busy, no XY device) must NOT
        # propagate — the monitor relies on None to mean "can't report XY".
        class _Flaky:
            def getXYPosition(self):
                raise RuntimeError("stage not responding")

        assert self._bare_pymmcore(_Flaky()).get_position() is None


# ==========================================================================
# Zero-dose control on RTMSequence (a 0 ms "dose" must fire no pulse)
# ==========================================================================


class TestZeroDoseControl:
    """``stim_exposure <= 0`` => no stim channels on stim frames.

    Failure mode this guards: a 0 ms condition (a genuine negative control
    in a BO dose-response) would otherwise rebuild a stim channel at 0 ms,
    hit the camera-minimum-exposure validation, and try to fire an SLM
    pulse for "no light".  The dose is still recorded via metadata so the
    optimiser sees a real 0-dose point.
    """

    def _seq(self, stim_exposure):
        return RTMSequence(
            time_plan={"interval": 1.0, "loops": 2},
            stage_positions=[(0, 0, 0)],
            channels=[
                PowerChannel(config="mScarlet3", exposure=250, group="g", power=20)
            ],
            stim_channels=(
                PowerChannel(config="CyanStim", exposure=200, group="g", power=25),
            ),
            stim_frames={1},
            stim_exposure=stim_exposure,
        )

    def _stim_frame(self, seq):
        return next(e for e in seq if e.index["t"] == 1)

    @pytest.mark.parametrize("zero", [0, 0.0, (0,), -5])
    def test_zero_dose_emits_no_stim(self, zero):
        ev = self._stim_frame(self._seq(zero))
        assert ev.stim_channels == ()

    def test_positive_dose_fires_and_preserves_powerchannel(self):
        ev = self._stim_frame(self._seq(123))
        assert len(ev.stim_channels) == 1
        ch = ev.stim_channels[0]
        # rebuilt with the new exposure but NOT downgraded to a bare Channel
        assert type(ch).__name__ == "PowerChannel"
        assert (ch.exposure, ch.power, ch.group) == (123, 25, "g")

    def test_per_frame_zero_mixed_with_nonzero(self):
        # Two stim frames, exposures (0, 250): frame 1 dark, frame 2 fires.
        seq = RTMSequence(
            time_plan={"interval": 1.0, "loops": 3},
            stage_positions=[(0, 0, 0)],
            channels=[PowerChannel(config="img", exposure=100, group="g", power=5)],
            stim_channels=(
                PowerChannel(config="CyanStim", exposure=200, group="g", power=25),
            ),
            stim_frames={1, 2},
            stim_exposure=(0, 250),
        )
        by_t = {e.index["t"]: e for e in seq}
        assert by_t[1].stim_channels == ()  # 0 ms -> control, no pulse
        assert (
            len(by_t[2].stim_channels) == 1 and by_t[2].stim_channels[0].exposure == 250
        )


# ==========================================================================
# Controller agent wiring + timepoint offsetting (no acquisition needed)
# ==========================================================================


class _DummyAgent(Agent):
    def run(self):  # the one abstractmethod on Agent
        return None


def _make_controller(tmp_path, *, agent=None, writer=None):
    mic = FakeMicroscope(CircleScene())
    pipeline = make_pipeline(str(tmp_path), tracker=TrackerTrackpy())
    return Controller(mic, pipeline, writer=writer, agent=agent)


class TestControllerAgentWiring:
    def test_agent_is_back_referenced(self, tmp_path):
        agent = _DummyAgent()
        ctrl = _make_controller(tmp_path, agent=agent)
        assert ctrl._agent is agent
        assert agent.controller is ctrl  # so the agent can call back in

    def test_no_agent_by_default(self, tmp_path):
        assert _make_controller(tmp_path)._agent is None


class TestOffsetEvents:
    def test_offset_shifts_t_and_stamps_time_offset(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        ctrl._t_offset = 5
        ctrl._time_offset = 1.25
        events = make_events(3)
        offset = ctrl._offset_events(events)

        assert [e.index["t"] for e in offset] == [5, 6, 7]
        assert all(e.metadata["time_offset"] == 1.25 for e in offset)
        # original events untouched (model_copy, not in-place mutation)
        assert [e.index["t"] for e in events] == [0, 1, 2]


# ==========================================================================
# Run-worker behaviour: per-phase snapshots + offset_timepoints flag.
# Tiny 2-frame acquisitions on the FakeMicroscope, blocking on the handle.
# ==========================================================================


def _phase_events(n, *, phase_id):
    return [
        e.model_copy(update={"metadata": {"phase_id": phase_id}})
        for e in make_events(n)
    ]


def _snapshot_path(storage_path, phase_id):
    events_dir = os.path.join(storage_path, "events")
    for ext in (".json.gz", ".json"):
        p = os.path.join(events_dir, f"events_phase_{phase_id:03d}{ext}")
        if os.path.exists(p):
            return p
    return None


def _read_snapshot(path):
    """Load a (gzipped) per-phase snapshot as a list of event dicts."""
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def _run(ctrl, events, **kw):
    handle = ctrl.run_experiment(events, validate=False, **kw)
    status = handle.wait(timeout=60)
    assert not handle.is_running()
    assert getattr(status, "fatal_error", None) is None, status
    return handle


def _continue(ctrl, events, **kw):
    handle = ctrl.continue_experiment(events, validate=False, **kw)
    status = handle.wait(timeout=60)
    assert not handle.is_running()
    assert getattr(status, "fatal_error", None) is None, status
    return handle


class TestRunWorkerEventPersistence:
    def test_per_phase_snapshots_survive_cumulative_overwrite(self, tmp_path):
        """Each phase's events are preserved even though the cumulative
        ``events.json.gz`` is rewritten every phase (offline-replay provenance).
        """
        writer = TiffWriter(str(tmp_path))
        ctrl = _make_controller(tmp_path, writer=writer)

        _run(ctrl, _phase_events(2, phase_id=0))
        _continue(ctrl, _phase_events(2, phase_id=1), offset_timepoints=True)
        _continue(ctrl, _phase_events(2, phase_id=2), offset_timepoints=True)

        # 1) every phase left its own snapshot, and they are distinct files
        snaps = [_snapshot_path(str(tmp_path), p) for p in (0, 1, 2)]
        assert all(snaps), f"missing per-phase snapshot(s): {snaps}"
        assert len(set(snaps)) == 3

        # 2) each snapshot holds ONLY that phase's two events
        for phase_id, snap in enumerate(snaps):
            rows = _read_snapshot(snap)
            assert len(rows) == 2
            assert {r["metadata"]["phase_id"] for r in rows} == {phase_id}

        # 3) the cumulative file (overwritten each phase) ends with all 6,
        #    and shift mode produced contiguous 0..5 timepoints
        cumulative = load_events_json(str(tmp_path))
        assert len(cumulative) == 6
        assert sorted(e.index["t"] for e in cumulative) == [0, 1, 2, 3, 4, 5]

        ctrl._analyzer.shutdown(wait=True)

    def test_continue_offset_timepoints_false_keeps_t(self, tmp_path):
        writer = TiffWriter(str(tmp_path))
        ctrl = _make_controller(tmp_path, writer=writer)

        _run(ctrl, _phase_events(2, phase_id=0))
        # Fresh-FOV phase: keep original 0-based t instead of shifting.
        _continue(ctrl, _phase_events(2, phase_id=1), offset_timepoints=False)

        phase1 = [e for e in ctrl._all_events if e.metadata.get("phase_id") == 1]
        assert sorted(e.index["t"] for e in phase1) == [0, 1]

        # the persisted snapshot agrees, and still records a time_offset
        rows = _read_snapshot(_snapshot_path(str(tmp_path), 1))
        assert sorted(r["index"]["t"] for r in rows) == [0, 1]
        assert all("time_offset" in r["metadata"] for r in rows)

        ctrl._analyzer.shutdown(wait=True)

    def test_continue_offset_timepoints_true_shifts_t(self, tmp_path):
        ctrl = _make_controller(tmp_path)
        _run(ctrl, _phase_events(2, phase_id=0))
        _continue(ctrl, _phase_events(2, phase_id=1), offset_timepoints=True)

        phase1 = [e for e in ctrl._all_events if e.metadata.get("phase_id") == 1]
        # phase 0 had t in {0, 1}; _t_offset -> 2, so phase 1 shifts to {2, 3}
        assert sorted(e.index["t"] for e in phase1) == [2, 3]

        ctrl._analyzer.shutdown(wait=True)


class TestInterPhaseWaitsForRun:
    """``InterPhaseAgent._wait_for_pipeline`` must block on the run handle.

    Regression guarded: ``run_experiment`` / ``continue_experiment`` became
    *non-blocking* on the async controller (they return a ``RunHandle`` and
    spawn a worker thread).  The BO agents call ``run_experiment(...)`` then
    ``_wait_for_pipeline()`` and read the phase's tracks afterwards.  If
    ``_wait_for_pipeline`` doesn't wait on the handle it returns while the
    worker is still feeding the engine (or before ``_analyzer`` even exists),
    so the agent reads an empty / partial phase.
    """

    def test_wait_for_pipeline_blocks_until_run_finishes(self, tmp_path):
        ctrl = _make_controller(tmp_path, writer=TiffWriter(str(tmp_path)))

        class _Probe(InterPhaseAgent):
            def run(self):  # the one abstractmethod
                raise NotImplementedError

        agent = _Probe(storage_path=str(tmp_path))
        agent.controller = ctrl

        # Hold the worker thread *inside* the run (the pre-feed-loop testing
        # hook runs on the worker), so a non-waiting _wait_for_pipeline would
        # provably return while the run is still in progress.
        ctrl._pre_loop_hook = lambda: time.sleep(0.4)

        ctrl.run_experiment(_phase_events(3, phase_id=0), validate=False)
        agent._wait_for_pipeline(timeout=60)

        # If it truly waited, the run has reached a terminal state.
        assert ctrl._current_handle is not None
        assert not ctrl._current_handle.is_running()
        ctrl._analyzer.shutdown(wait=True)
