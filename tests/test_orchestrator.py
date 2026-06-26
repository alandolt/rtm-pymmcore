"""Tests for the async multi-run orchestrator layer.

Covers the three pieces added on top of ``run_experiment``:

* :class:`faro.core.run_status.OrchestratorHandle` — lifecycle, error
  re-raise, cooperative cancel, progress reporting, and the ``current_run``
  drill-down (+ ``currentRunChanged``).
* :meth:`faro.core.controller.Controller.run_orchestrator_async` — runs a
  blocking orchestrator on a worker thread, injects cooperation by
  introspection (``progress`` handle preferred, ``cancel_event`` fallback),
  keeps ``current_run`` pointed at the live acquisition via the controller's
  ``runStarted`` signal, and disconnects that subscription when done.
* The two first consumers — :func:`faro.agents.run_well_patterns` and
  :meth:`faro.agents.ComposedAgent.run` — honour ``progress`` (cancel +
  step reporting).

The controller-integration tests use the real :class:`Controller` +
:class:`tests.fake_microscope.FakeMicroscope`, so ``current_run`` is driven by
genuine ``runStarted`` emissions from ``run_experiment`` /
``continue_experiment`` — not a stub.
"""

from __future__ import annotations

import os
import threading
import time

import pytest

from faro.agents import WellPattern, run_well_patterns
from faro.agents.base import InterPhaseAgent, PreExperimentAgent
from faro.agents.composed import ComposedAgent
from faro.core.controller import Controller
from faro.core.data_structures import Channel, RTMSequence
from faro.core.run_status import OrchestratorHandle, RunHandle
from faro.core.utils import FovPosition

from tests.fake_microscope import FakeMicroscope
from tests.fixtures import CircleScene, make_events, make_pipeline
from faro.tracking.trackpy import TrackerTrackpy


def _wait_until(predicate, *, timeout=5.0, poll=0.005):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll)
    return False


def _spaced(events, dt):
    return [
        e.model_copy(update={"min_start_time": i * dt}) for i, e in enumerate(events)
    ]


def _shift_p(events, p):
    return [e.model_copy(update={"index": {**dict(e.index), "p": p}}) for e in events]


# ===========================================================================
# OrchestratorHandle — pure unit tests (no controller)
# ===========================================================================


class TestOrchestratorHandle:
    def test_initial_state_pending(self):
        h = OrchestratorHandle()
        assert h.status().state == "pending"
        assert h.current_run is None
        assert not h.cancelled

    def test_update_emits_statuschanged(self):
        h = OrchestratorHandle()
        seen = []
        h.statusChanged.connect(seen.append)
        h.update(state="running", message="hi")
        assert seen and seen[-1].state == "running" and seen[-1].message == "hi"

    def test_report_progress_sets_step_fields(self):
        h = OrchestratorHandle()
        h.report_progress(2, 10, "batch 3/10")
        s = h.status()
        assert (s.step, s.n_steps, s.message) == (2, 10, "batch 3/10")
        # state untouched — lifecycle is the controller's job.
        assert s.state == "pending"

    def test_cancel_sets_flag_and_runs_hook_once(self):
        calls = []
        h = OrchestratorHandle(on_cancel=lambda: calls.append(1))
        h.update(state="running")
        h.cancel()
        h.cancel()  # idempotent
        assert h.cancelled
        assert h.status().state == "cancelling"
        assert calls == [1]  # hook ran exactly once

    def test_set_current_run_emits_and_is_idempotent(self):
        h = OrchestratorHandle()
        seen = []
        h.currentRunChanged.connect(seen.append)
        rh1, rh2 = RunHandle(), RunHandle()
        h._set_current_run(rh1)
        h._set_current_run(rh1)  # same -> no emit
        h._set_current_run(rh2)
        assert h.current_run is rh2
        assert seen == [rh1, rh2]

    def test_wait_reraises_error(self):
        h = OrchestratorHandle()

        def worker():
            try:
                raise ValueError("boom")
            except ValueError as e:
                h.update(state="error", error=e)

        h._thread = threading.Thread(target=worker)
        h._thread.start()
        with pytest.raises(ValueError, match="boom"):
            h.wait(timeout=5)


# ===========================================================================
# Controller.run_orchestrator_async — injection, lifecycle, errors
# ===========================================================================


def _make_ctrl(tmp_path):
    pipeline = make_pipeline(str(tmp_path), tracker=TrackerTrackpy(search_range=50))
    return Controller(FakeMicroscope(CircleScene()), pipeline)


class TestRunOrchestratorAsyncContract:
    def test_injects_progress_handle_when_accepted(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        received = {}

        def orch(progress=None):
            received["progress"] = progress

        ctrl.run_orchestrator_async(orch).wait(timeout=5)
        assert isinstance(received["progress"], OrchestratorHandle)

    def test_injects_cancel_event_fallback(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        received = {}

        def orch(cancel_event=None):
            received["cancel_event"] = cancel_event

        ctrl.run_orchestrator_async(orch).wait(timeout=5)
        assert isinstance(received["cancel_event"], threading.Event)

    def test_prefers_progress_over_cancel_event(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        received = {}

        def orch(progress=None, cancel_event=None):
            received["progress"] = progress
            received["cancel_event"] = cancel_event

        ctrl.run_orchestrator_async(orch).wait(timeout=5)
        assert isinstance(received["progress"], OrchestratorHandle)
        assert received["cancel_event"] is None  # not injected when progress present

    def test_no_injection_when_neither_accepted(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        ran = []
        ctrl.run_orchestrator_async(lambda: ran.append(1)).wait(timeout=5)
        assert ran == [1]

    def test_forwards_positional_and_keyword_args(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        got = {}

        def orch(a, b, *, c, progress=None):
            got.update(a=a, b=b, c=c)

        ctrl.run_orchestrator_async(orch, 1, 2, c=3).wait(timeout=5)
        assert got == {"a": 1, "b": 2, "c": 3}

    def test_error_is_reraised_by_wait(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)

        def orch(progress=None):
            raise RuntimeError("orchestrator failed")

        handle = ctrl.run_orchestrator_async(orch)
        with pytest.raises(RuntimeError, match="orchestrator failed"):
            handle.wait(timeout=5)
        assert handle.status().state == "error"

    def test_concurrent_orchestrator_rejected(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        release = threading.Event()

        def slow(progress=None):
            release.wait(timeout=5)

        h1 = ctrl.run_orchestrator_async(slow)
        assert _wait_until(lambda: h1.status().state == "running")
        with pytest.raises(RuntimeError, match="already running"):
            ctrl.run_orchestrator_async(lambda: None)
        release.set()
        h1.wait(timeout=5)
        # once finished, a new orchestrator is allowed again
        ctrl.run_orchestrator_async(lambda: None).wait(timeout=5)

    def test_runstarted_subscription_disconnected_after_finish(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        before = len(ctrl.runStarted)
        ctrl.run_orchestrator_async(lambda progress=None: None).wait(timeout=5)
        assert _wait_until(lambda: len(ctrl.runStarted) == before)

    def test_stop_run_cancels_active_orchestrator(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        log = []

        def orch(progress=None):
            for i in range(500):
                if progress.cancelled:
                    log.append(("cancelled", i))
                    return
                time.sleep(0.01)
            log.append(("ran_to_end",))

        h = ctrl.run_orchestrator_async(orch)
        assert _wait_until(lambda: h.status().state == "running")
        time.sleep(0.05)
        ctrl.stop_run()  # legacy entry point must now cancel the orchestrator
        h.wait(timeout=5)
        assert h.status().state == "done"
        assert log and log[0][0] == "cancelled"


# ===========================================================================
# Controller integration — current_run driven by REAL runStarted
# ===========================================================================


class TestCurrentRunDrilldown:
    def _batches(self, n_batches, n_t=2, dt=0.15):
        # Each batch is a tiny spaced single-FOV run at a distinct position.
        return [_spaced(_shift_p(make_events(n_t), p=i), dt) for i in range(n_batches)]

    def _orchestrator(self, ctrl, batches, *, progress=None):
        for i, events in enumerate(batches):
            if progress is not None and progress.cancelled:
                break
            if progress is not None:
                progress.report_progress(
                    i, len(batches), f"batch {i + 1}/{len(batches)}"
                )
            if i == 0:
                h = ctrl.run_experiment(events, validate=False)
            else:
                h = ctrl.continue_experiment(
                    events, validate=False, offset_timepoints=False
                )
            h.wait(timeout=20)
        ctrl.finish_experiment()

    def test_current_run_tracks_each_batch(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        batches = self._batches(3)

        run_count = []
        ctrl.runStarted.connect(lambda h: run_count.append(h))

        handle = ctrl.run_orchestrator_async(self._orchestrator, ctrl, batches)
        seen_runs = []
        handle.currentRunChanged.connect(seen_runs.append)

        handle.wait(timeout=30)
        assert handle.status().state == "done"
        # Real runStarted fired once per batch.
        assert len(run_count) == 3
        # Orchestrator handle saw the live run advance (>=2 robust against the
        # tiny connect-after-launch window).
        assert len([r for r in seen_runs if r is not None]) >= 2
        # current_run ends pointing at the last batch's RunHandle.
        assert handle.current_run is not None
        assert handle.current_run is run_count[-1]
        # Progress counter advanced to the last batch.
        assert handle.status().step == 2
        assert handle.status().n_steps == 3

    def test_cancel_aborts_inflight_run_and_stops_loop(self, tmp_path):
        ctrl = _make_ctrl(tmp_path)
        # Long, well-spaced batches so the run is reliably mid-flight on cancel.
        batches = self._batches(4, n_t=6, dt=0.4)

        handle = ctrl.run_orchestrator_async(self._orchestrator, ctrl, batches)
        # Wait until the first batch's acquisition is actually running.
        assert _wait_until(
            lambda: handle.current_run is not None
            and handle.current_run.status().n_events_consumed >= 1,
            timeout=10,
        )
        inflight = handle.current_run
        handle.cancel()
        handle.wait(timeout=20)

        assert handle.status().state == "done"
        assert handle.status().message == "cancelled"
        # The in-flight batch's run handle was cancelled (not left running).
        assert inflight.cancel_event.is_set()
        # It stopped early: fewer than all 4 batches ran.
        assert handle.status().step < 3


# ===========================================================================
# run_well_patterns honours `progress` (cancel + reporting)
# ===========================================================================


class _FakeFinder:
    def __init__(self, well):
        self.well = well

    def run(self):
        return [
            FovPosition(x=10.0 * i, y=20.0, z=None, name=f"{self.well}_{i:04d}")
            for i in range(2)
        ]


class _InstanceFinder:
    """A configured finder, scoped to ``_remaining_wells`` like the real ones.

    Models the attributes :func:`resolve_well_patterns` re-scopes when handed a
    single ``finder=`` instance, so we can assert the per-well copies are
    independent and the original is left untouched.
    """

    def __init__(self, wells):
        self.wells_per_phase = len(wells)
        self._wells_source = list(wells)
        self._remaining_wells = list(wells)
        self._phase_index = 0
        self.history = []
        self.fovs_per_well = 2

    def run(self):
        well = self._remaining_wells[0]
        self.history.append(well)
        self._phase_index += 1
        return [
            FovPosition(x=1.0 * i, y=2.0, z=None, name=f"{well}_{i:04d}")
            for i in range(self.fovs_per_well)
        ]


def test_resolve_with_finder_instance_rescopes_per_well():
    """`finder=` re-scopes ONE configured instance per well via cheap copies."""
    from faro.agents import resolve_well_patterns

    base = _InstanceFinder(["A1", "A2", "A3"])
    resolved = resolve_well_patterns(
        "MIC", _patterns(3), finder=base, apply_batching=False, verbose=False
    )
    # FOVs named by each pattern's OWN well -> re-scoping worked.
    names = [fp.name for fovs in resolved.fovs for fp in fovs]
    assert names == [
        "A1_0000",
        "A1_0001",
        "A2_0000",
        "A2_0001",
        "A3_0000",
        "A3_0001",
    ]
    # The original instance is NOT consumed/mutated (copies were used).
    assert base._remaining_wells == ["A1", "A2", "A3"]
    assert base.history == []
    # Each stored finder is an independent copy scoped to its own well.
    assert [f._remaining_wells for f in resolved.finders] == [["A1"], ["A2"], ["A3"]]
    assert all(f.wells_per_phase == 1 for f in resolved.finders)


def test_resolve_rejects_multiple_finder_specs():
    from faro.agents import resolve_well_patterns

    with pytest.raises(ValueError, match="exactly one of finder"):
        resolve_well_patterns(
            "MIC",
            _patterns(1),
            finder=_InstanceFinder(["A1"]),
            finder_factory=lambda w: _FakeFinder(w),
            apply_batching=False,
        )


class _RecordingCtrl(Controller):
    """Minimal controller stand-in that records run/continue/finish calls and
    fakes a quick async per-batch RunHandle."""

    def __init__(self):
        self._current_handle = None
        self._orchestrator_handle = None
        self.runs = []

    def _fake_run(self, events, **kw):
        h = RunHandle(n_events_total=len(events))
        h._thread = threading.Thread(target=lambda: time.sleep(0.05), daemon=True)
        h._thread.start()
        self._current_handle = h
        self.runStarted.emit(h)
        self.runs.append(sorted({e.index.get("p", 0) for e in events}))
        return h

    def run_experiment(self, events, **kw):
        return self._fake_run(events, **kw)

    def continue_experiment(self, events, **kw):
        return self._fake_run(events, **kw)

    def finish_experiment(self, **kw):
        self.runs.append("finish")


def _patterns(n):
    ch = (Channel(config="phase-contrast", exposure=50, group="Channel"),)
    stim = (Channel(config="stim-405", exposure=100, group="Channel"),)

    def make(well):
        def build(fovs):
            return RTMSequence(
                time_plan={"interval": 1.0, "loops": 3},
                stage_positions=fovs,
                channels=ch,
                stim_channels=stim,
                stim_frames=frozenset({1}),
                rtm_metadata={"well": well},
            )

        return WellPattern(well=well, build_sequence=build)

    return [make(f"A{i}") for i in range(1, n + 1)]


class TestRunWellPatternsProgress:
    def test_reports_progress_and_runs_all_batches(self, tmp_path):
        ctrl = _RecordingCtrl()
        handle = ctrl.run_orchestrator_async(
            run_well_patterns,
            ctrl,
            "MIC",
            _patterns(6),
            wells_per_batch=2,
            finder_factory=lambda w: _FakeFinder(w),
            time_per_fov=1.0,
            n_parallel=4,
            verbose=False,
        )
        handle.wait(timeout=20)
        assert handle.status().state == "done"
        # 3 batches of 2 wells + a finish call.
        runs = [r for r in ctrl.runs if r != "finish"]
        assert len(runs) == 3
        assert "finish" in ctrl.runs
        # Globally-unique p indices across batches (2 wells x 2 FOVs = 4 each).
        assert runs[0] == [0, 1, 2, 3]
        assert runs[1] == [4, 5, 6, 7]
        assert runs[2] == [8, 9, 10, 11]
        # Progress reported "batch i/n".
        assert handle.status().n_steps == 3

    def test_cancel_during_scan_does_not_run_that_batch(self, tmp_path):
        # The hard case: user cancels while the finder is still scanning a
        # batch's wells (the long phase). The scan must stop early and the
        # batch must NOT be run.
        ctrl = _RecordingCtrl()
        n_runs = {"finder": 0}

        def factory(well):
            f = _FakeFinder(well)
            orig = f.run

            def run():
                n_runs["finder"] += 1
                if n_runs["finder"] == 1:
                    # Simulate the user hitting cancel mid-scan (first well).
                    ctrl._orchestrator_handle.cancel()
                return orig()

            f.run = run
            return f

        handle = ctrl.run_orchestrator_async(
            run_well_patterns,
            ctrl,
            "MIC",
            _patterns(6),
            wells_per_batch=2,
            finder_factory=factory,
            time_per_fov=1.0,
            n_parallel=4,
            verbose=False,
        )
        handle.wait(timeout=20)
        # No acquisition ran (cancel landed during batch 0's scan), and the
        # scan stopped after the first well rather than scanning all of them.
        assert [r for r in ctrl.runs if r != "finish"] == []
        assert n_runs["finder"] == 1  # stopped after the first well, not all 2
        assert "finish" in ctrl.runs  # store still closed cleanly

    def test_cancel_stops_between_batches_but_finishes_store(self, tmp_path):
        ctrl = _RecordingCtrl()

        # Cancel as soon as the first batch's run starts.
        def _cancel_on_first_run(_h):
            ctrl._orchestrator_handle.cancel()

        ctrl.runStarted.connect(_cancel_on_first_run)

        handle = ctrl.run_orchestrator_async(
            run_well_patterns,
            ctrl,
            "MIC",
            _patterns(8),
            wells_per_batch=2,
            finder_factory=lambda w: _FakeFinder(w),
            time_per_fov=1.0,
            n_parallel=4,
            verbose=False,
        )
        handle.wait(timeout=20)
        runs = [r for r in ctrl.runs if r != "finish"]
        assert len(runs) < 4  # stopped early (8 wells / 2 = 4 batches max)
        assert "finish" in ctrl.runs  # store still closed cleanly


# ===========================================================================
# ComposedAgent.run honours `progress` (cancel + phase reporting)
# ===========================================================================


class _StubController:
    def finish_experiment(self):
        self.finished = True


class _InnerAgent(InterPhaseAgent):
    def __init__(self, cancel_at=None, handle=None):
        super().__init__(storage_path="")
        self.phases = []
        self._cancel_at = cancel_at
        self._handle = handle

    def run(self):  # required abstract
        pass

    def run_one_phase(self, phase_id, fov_positions=None, fovs=None):
        self.phases.append(phase_id)
        if self._cancel_at is not None and phase_id == self._cancel_at:
            self._handle.cancel()
        return {"phase_id": phase_id}


class _PreAgent(PreExperimentAgent):
    def __init__(self):
        super().__init__(microscope=None)

    def run(self):
        return []  # bare positions list


class TestComposedAgentProgress:
    def _composed(self, inner, n_phases):
        composed = ComposedAgent(
            inner_agent=inner,
            pre_phase_agents=[_PreAgent()],
            n_phases=n_phases,
            finish_experiment=False,
        )
        composed.controller = _StubController()  # delegates to inner.controller
        return composed

    def test_reports_phase_progress(self):
        inner = _InnerAgent()
        composed = self._composed(inner, n_phases=4)
        handle = OrchestratorHandle()
        msgs = []
        handle.statusChanged.connect(lambda s: msgs.append((s.step, s.n_steps)))
        composed.run(progress=handle)
        assert inner.phases == [0, 1, 2, 3]
        assert (handle.status().step, handle.status().n_steps) == (3, 4)
        assert (3, 4) in msgs

    def test_cancel_stops_between_phases(self):
        handle = OrchestratorHandle()
        inner = _InnerAgent(cancel_at=1, handle=handle)
        composed = self._composed(inner, n_phases=5)
        composed.run(progress=handle)
        # phase 0 and 1 ran; cancel during phase 1 -> phase 2 never starts.
        assert inner.phases == [0, 1]

    def test_runs_async_via_orchestrator_with_injected_progress(self, tmp_path):
        # The BO path: `ctrl.run_orchestrator_async(composed.run)` must
        # introspect composed.run, inject the OrchestratorHandle as `progress`,
        # and the agent reports phase progress -- exactly how the BO notebooks
        # will launch (composed_agent.run -> run_orchestrator_async).
        ctrl = _make_ctrl(tmp_path)
        inner = _InnerAgent()
        composed = ComposedAgent(
            inner_agent=inner,
            pre_phase_agents=[_PreAgent()],
            n_phases=3,
            finish_experiment=False,
        )
        composed.controller = ctrl  # wire to the real controller

        handle = ctrl.run_orchestrator_async(composed.run)
        handle.wait(timeout=10)
        assert handle.status().state == "done"
        assert inner.phases == [0, 1, 2]
        assert (handle.status().step, handle.status().n_steps) == (2, 3)
