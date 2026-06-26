"""Bind stimulation patterns to wells whose FOVs are located at runtime.

A *pattern* here is a full :class:`~faro.core.data_structures.RTMSequence`
recipe (time plan + imaging channels + stim schedule) that you author once
per well — e.g. one stim waveform per well from a CSV.  The catch is that the
**FOV positions inside each well are not known up front**: they should be
chosen by a :class:`~faro.agents.fov_finder.FOVFinderAgent` scoped to that
well, at the moment the experiment runs.

This module provides the thin orchestration layer that closes that gap
*without* changing ``RTMSequence`` (whose ``stage_positions`` must be concrete
to materialise events):

1. :class:`WellPattern` pairs a well name with a ``build_sequence`` callback
   that turns the found FOVs into a concrete ``RTMSequence``.  The callback is
   *your* domain logic — e.g. map a CSV stim waveform onto ``stim_frames``.
2. :func:`resolve_well_patterns` runs a well-scoped finder for every pattern,
   builds each ``RTMSequence`` with the located FOVs, combines them along the
   position axis (so each well keeps its own stim schedule) and time-multiplexes
   them with :func:`~faro.core.utils.apply_fov_batching`.  Use this when you
   want **all** FOVs found up front and run as one combined acquisition.
3. :func:`run_well_patterns` processes the patterns in **batches**: it finds
   one batch of wells, runs that batch's time-lapse to completion, then finds
   and runs the next batch.  Use this when you don't want to scan every well
   before any imaging starts.  Each batch's FOVs get globally-unique position
   indices and the whole run accumulates into a single store — every FOV is a
   distinct position, so no per-phase bookkeeping is needed.

The result of :func:`resolve_well_patterns` is the flat ``list[RTMEvent]`` you
pass straight to ``Controller.run_experiment`` — the FOV finding happens
automatically inside the run script, not as a manual pre-step.

Typical use::

    from faro.agents import WellPattern, resolve_well_patterns

    def build_pattern(uid, well):
        sub = patterns_to_test[patterns_to_test.uid == uid].sort_values("time")
        stim_frames = frozenset(sub.loc[sub["value"] > 0, "time"].astype(int))

        def build(fovs):  # fovs located at runtime by the FOV finder
            return RTMSequence(
                time_plan={"interval": 10.0, "loops": len(sub)},
                stage_positions=fovs,
                channels=imaging_channels,
                stim_channels=(stim_channel,),
                stim_frames=stim_frames,
                rtm_metadata={"uid": int(uid), "well": well},
            )

        return WellPattern(well=well, build_sequence=build)

    patterns = [build_pattern(uid, well) for uid, well in zip(uids, wells)]

    resolved = resolve_well_patterns(
        mic,
        patterns,
        finder_kwargs=dict(
            well_plate_plan=PLATE_CALIBRATION_PATH,
            fovs_per_well=3,
            n_candidates_per_well=12,
            border_um=300,
            min_cells=20,
            imaging_channels=imaging_channels,
            segmentator=segmentator,
            z=None,
        ),
        time_per_fov=2.0,
        n_parallel=18,  # 6 wells x 3 FOVs imaged per interval
    )

    ctrl.validate_events(resolved.events)
    ctrl.run_experiment(resolved.events)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from faro.agents.fov_finder import FOVFinderAgent
from faro.core.data_structures import RTMSequence, combine
from faro.core.utils import FovPosition, apply_fov_batching

if TYPE_CHECKING:
    from faro.core.controller import Controller
    from faro.core.data_structures import RTMEvent
    from faro.core.run_status import OrchestratorHandle
    from faro.microscope.base import AbstractMicroscope


@runtime_checkable
class FOVFinder(Protocol):
    """The only interface a FOV finder must satisfy to drive well-patterns.

    A finder is scoped to a single well and returns that well's FOVs from
    :meth:`run`.  Both :class:`~faro.agents.fov_finder.FOVFinderAgent`
    (random-candidate + farthest-point) and
    :class:`~faro.agents.grid_fov_finder.GridFOVFinderAgent` (grid overview +
    density scoring) satisfy it, as does any custom finder you write — so any
    of them can be dropped into :func:`resolve_well_patterns` /
    :func:`run_well_patterns` interchangeably, either up front or per batch
    during the run.

    To be buildable through the easy ``finder_class`` + ``finder_kwargs`` path
    (rather than a hand-written ``finder_factory``), the constructor must also
    accept ``microscope`` positionally plus ``wells=[well]`` and
    ``wells_per_phase=1`` as keywords — the convention both built-in finders
    follow.
    """

    def run(self) -> list[FovPosition]: ...


@dataclass
class WellPattern:
    """A stimulation pattern bound to a well whose FOVs are found at runtime.

    Args:
        well: Well name (e.g. ``"A1"``) the pattern is applied to.  The
            :class:`~faro.agents.fov_finder.FOVFinderAgent` is scoped to
            this single well to locate the FOVs.
        build_sequence: Callback that receives the located FOVs
            (``list[FovPosition]``) and returns the concrete
            :class:`~faro.core.data_structures.RTMSequence` for this well.
            This is your domain logic — e.g. map a CSV stim waveform onto
            ``stim_frames`` (see the module docstring for an example).
    """

    well: str
    build_sequence: Callable[[list[FovPosition]], RTMSequence]


@dataclass
class ResolvedWellPatterns:
    """Result of :func:`resolve_well_patterns`.

    Attributes:
        events: Flat ``list[RTMEvent]`` ready for
            ``Controller.run_experiment`` (combined across wells along the
            position axis and time-multiplexed via ``apply_fov_batching``
            when batching is enabled).
        sequences: The per-well :class:`RTMSequence` objects, in pattern
            order.
        fovs: The located FOVs per pattern, parallel to *sequences*.
        wells: Well name per pattern, parallel to *sequences*.
        finders: The :class:`FOVFinder` instances used (whatever
            ``finder_class`` / ``finder_factory`` produced), for inspection
            (e.g. ``finders[i].last_run`` holds the scan DataFrame).
    """

    events: list["RTMEvent"]
    sequences: list[RTMSequence] = field(default_factory=list)
    fovs: list[list[FovPosition]] = field(default_factory=list)
    wells: list[str] = field(default_factory=list)
    finders: list[FOVFinder] = field(default_factory=list)


def resolve_well_patterns(
    microscope: "AbstractMicroscope",
    well_patterns: Sequence[WellPattern],
    *,
    finder: FOVFinder | None = None,
    finder_factory: Callable[[str], FOVFinder] | None = None,
    finder_class: type[FOVFinder] = FOVFinderAgent,
    finder_kwargs: dict[str, Any] | None = None,
    time_per_fov: float | None = None,
    n_parallel: int | None = None,
    apply_batching: bool = True,
    axis: str = "p",
    progress: "OrchestratorHandle | None" = None,
    verbose: bool = True,
) -> ResolvedWellPatterns:
    """Locate each pattern's FOVs at runtime, then combine and batch.

    For every :class:`WellPattern` a :class:`FOVFinderAgent` scoped to that
    pattern's well is run; the located FOVs are passed to the pattern's
    ``build_sequence`` to produce a concrete ``RTMSequence``.  All sequences
    are combined along *axis* (``"p"`` by default, so each well keeps its own
    stim schedule and they share the wall clock) and, when *apply_batching*
    is set, time-multiplexed with
    :func:`~faro.core.utils.apply_fov_batching`.

    Exactly one of *finder* / *finder_factory* / *finder_kwargs* must be given:

    * ``finder`` — **the easy path.**  A single, fully-configured finder
      instance (e.g. the very ``GridFOVFinderAgent`` you built and tried in
      ``grid_fov_finder_demo.ipynb``).  It is re-scoped to one well at a time
      internally (a cheap shallow copy per well, sharing the loaded plate plan /
      segmentator / feature extractor), so its ``wells`` / ``wells_per_phase``
      are **managed here** — set them to anything (they are overridden); every
      other knob (grid size, density band, ERK ``fov_conditions``, flips, …) is
      taken from the instance.  This avoids the ``finder_class`` +
      ``finder_kwargs`` split.
    * ``finder_kwargs`` — a dict of *finder_class* constructor arguments
      **without** ``wells`` / ``wells_per_phase`` (managed here).  One finder
      is built per well as
      ``finder_class(microscope, wells=[well], wells_per_phase=1, **finder_kwargs)``.
      Set ``finder_class`` to pick which finder runs (default
      :class:`FOVFinderAgent`).
    * ``finder_factory`` — full control: a callable ``well -> FOVFinder`` for
      when you need per-well customisation the other paths can't express.

    Args:
        microscope: Microscope instance (forwarded to the finders).
        well_patterns: Patterns to resolve, in the order their FOVs should
            occupy the position axis.
        finder: A configured :class:`FOVFinder` instance, re-scoped per well
            (the easy path; see above).
        finder_factory: Callable building a finder scoped to a single well.
        finder_class: Finder class built per well in the ``finder_kwargs``
            path (any :class:`FOVFinder`).
        finder_kwargs: Finder constructor kwargs (see above).
        time_per_fov: Seconds to image one FOV.  Required when
            *apply_batching* is True.
        n_parallel: Max FOVs imaged per batch (e.g. 18 for 6 wells x 3
            FOVs).  ``None`` lets ``apply_fov_batching`` derive it from
            *time_per_fov* and the inferred interval.
        apply_batching: Whether to time-multiplex overflow FOVs into
            sequential batches.  Set ``False`` to return the raw combined
            events (e.g. when all FOVs fit in one interval).
        axis: Combination axis; ``"p"`` (default) runs the wells in parallel
            within one time-lapse.
        verbose: Print a per-well progress line as each finder runs.

    Returns:
        A :class:`ResolvedWellPatterns` whose ``.events`` you pass to
        ``Controller.run_experiment``.
    """
    if not well_patterns:
        raise ValueError("well_patterns is empty")
    if sum(x is not None for x in (finder, finder_factory, finder_kwargs)) != 1:
        raise ValueError(
            "provide exactly one of finder / finder_factory / finder_kwargs"
        )
    if apply_batching and time_per_fov is None:
        raise ValueError("time_per_fov is required when apply_batching=True")

    if finder is not None:
        # Easy path: re-scope ONE configured instance to a single well at a
        # time.  A shallow copy keeps the (read-only) plate plan / segmentator /
        # feature extractor shared while giving each well its own run state, so
        # `resolved.finders[i].last_run` stays per-well inspectable.
        import copy as _copy

        def finder_factory(well: str, _f: FOVFinder = finder) -> FOVFinder:
            scoped = _copy.copy(_f)
            scoped.wells_per_phase = 1
            scoped._wells_source = [well]
            scoped._remaining_wells = [well]
            scoped._phase_index = 0
            scoped.history = []
            return scoped

    elif finder_factory is None:
        fk = dict(finder_kwargs)  # copy so we don't mutate the caller's dict
        for bad in ("wells", "wells_per_phase"):
            if bad in fk:
                raise ValueError(
                    f"finder_kwargs must not set {bad!r}; it is managed "
                    f"per-well (wells=[well], wells_per_phase=1)."
                )

        def finder_factory(
            well: str,
            _fk: dict[str, Any] = fk,
            _cls: type[FOVFinder] = finder_class,
        ) -> FOVFinder:
            return _cls(microscope, wells=[well], wells_per_phase=1, **_fk)

    sequences: list[RTMSequence] = []
    fovs_all: list[list[FovPosition]] = []
    wells: list[str] = []
    finders: list[FOVFinder] = []

    n = len(well_patterns)
    for i, wp in enumerate(well_patterns):
        # Stop scanning the moment a cancel is requested — otherwise a cancel
        # during a (minutes-long) multi-well scan would keep imaging every
        # remaining well before the caller's loop gets to check. Partial
        # results are returned; the caller (run_well_patterns) sees the cancel
        # and discards this batch rather than running it.
        if progress is not None and progress.cancelled:
            if verbose:
                print(
                    f"[resolve_well_patterns] cancelled mid-scan after "
                    f"{len(wells)}/{n} well(s); returning partial."
                )
            break
        if verbose:
            print(
                f"[resolve_well_patterns] {i + 1}/{n}: finding FOVs in well "
                f"{wp.well!r} ..."
            )
        well_finder = finder_factory(wp.well)
        fovs = list(well_finder.run())
        if verbose:
            print(
                f"[resolve_well_patterns]   well {wp.well!r}: {len(fovs)} FOV(s) "
                f"-> {[fp.name for fp in fovs]}"
            )
        seq = wp.build_sequence(fovs)
        sequences.append(seq)
        fovs_all.append(fovs)
        wells.append(wp.well)
        finders.append(well_finder)

    events = combine(*sequences, axis=axis)
    if apply_batching:
        events = apply_fov_batching(
            events, time_per_fov=time_per_fov, n_parallel=n_parallel
        )

    if verbose:
        n_fovs = len({e.index.get("p", 0) for e in events})
        print(
            f"[resolve_well_patterns] resolved {len(sequences)} pattern(s), "
            f"{n_fovs} FOV(s) total, {len(events)} event(s)."
        )

    return ResolvedWellPatterns(
        events=events,
        sequences=sequences,
        fovs=fovs_all,
        wells=wells,
        finders=finders,
    )


def _shift_positions(events: list["RTMEvent"], p_offset: int) -> list["RTMEvent"]:
    """Add *p_offset* to every event's position index (no-op for ``p_offset==0``)."""
    if p_offset == 0:
        return events
    shifted: list["RTMEvent"] = []
    for ev in events:
        idx = dict(ev.index)
        if "p" in idx:
            idx["p"] = idx.get("p", 0) + p_offset
            shifted.append(ev.model_copy(update={"index": idx}))
        else:  # e.g. WaitEvent — no position to shift
            shifted.append(ev)
    return shifted


def run_well_patterns(
    controller: "Controller",
    microscope: "AbstractMicroscope",
    well_patterns: Sequence[WellPattern],
    *,
    wells_per_batch: int,
    finder: FOVFinder | None = None,
    finder_factory: Callable[[str], FOVFinder] | None = None,
    finder_class: type[FOVFinder] = FOVFinderAgent,
    finder_kwargs: dict[str, Any] | None = None,
    time_per_fov: float | None = None,
    n_parallel: int | None = None,
    apply_batching: bool = True,
    stim_mode: str = "current",
    validate: bool = True,
    finish: bool = True,
    progress: "OrchestratorHandle | None" = None,
    verbose: bool = True,
) -> list[ResolvedWellPatterns]:
    """Find + run well-patterns in sequential batches of *wells_per_batch* wells.

    Unlike :func:`resolve_well_patterns` (which finds **all** FOVs up front and
    runs them as one combined acquisition), this driver processes the patterns
    in batches: it finds the FOVs for one batch of wells, runs that batch's
    stimulation time-lapse to completion, then moves on to find + run the next
    batch.  Cells in a well aren't scanned until just before that well's batch
    runs.

    Every batch's FOVs get globally-unique position indices and the whole run
    accumulates into a single store (one big tracks DataFrame) — each FOV is a
    physically distinct position, so there is no per-phase bookkeeping and the
    pattern ``RTMSequence``\\ s must **not** set ``phase_name`` / ``phase_id``.

    The first batch starts acquisition via ``controller.run_experiment``; each
    later batch extends it via
    ``controller.continue_experiment(offset_timepoints=False)`` (the fresh-FOV
    path, so timesteps restart from 0 for the new positions).  The driver
    blocks on each batch's run handle before starting the next batch's FOV scan
    (the scan needs the microscope free).

    Args:
        controller: The :class:`~faro.core.controller.Controller` driving the
            acquisition (created **without** an ``agent``).
        microscope: Microscope instance (forwarded to the FOV finders).
        well_patterns: All patterns to run, processed in this order; batched in
            chunks of *wells_per_batch*.
        wells_per_batch: Wells (= patterns) per batch.  With ``fovs_per_well``
            FOVs each, a batch images ``wells_per_batch * fovs_per_well``
            positions together (e.g. 6 wells x 3 FOVs = 18).
        finder / finder_factory / finder_class / finder_kwargs: Forwarded to
            :func:`resolve_well_patterns` per batch (exactly one of *finder* /
            *finder_factory* / *finder_kwargs* required).  The easy path is
            *finder*: hand it one configured instance (e.g. a
            :class:`~faro.agents.grid_fov_finder.GridFOVFinderAgent` with its
            density + ERK ``fov_conditions`` set), and it runs **during** the
            experiment, re-scoped per well each batch.
        time_per_fov, n_parallel, apply_batching: Forwarded to
            :func:`resolve_well_patterns` for within-batch FOV time-multiplexing.
        stim_mode: ``"current"`` / ``"previous"`` — constant across batches
            (``continue_experiment`` rejects a mid-run mode change).
        validate: Validate each batch's events before running.
        finish: Call ``controller.finish_experiment()`` after the last batch
            (also runs if cancelled, so the store is closed cleanly).
        progress: Optional :class:`~faro.core.run_status.OrchestratorHandle`;
            checked before each batch (``progress.cancelled``) to stop cleanly,
            and updated per batch (``progress.report_progress``) so a widget
            shows "batch i/n".  Injected automatically when launched via
            :meth:`~faro.core.controller.Controller.run_orchestrator_async`.
        verbose: Print per-batch progress.

    Returns:
        One :class:`ResolvedWellPatterns` per batch (in order), so the FOVs /
        sequences / finders used in each batch can be inspected afterwards.
    """
    if wells_per_batch <= 0:
        raise ValueError("wells_per_batch must be positive")
    patterns = list(well_patterns)
    if not patterns:
        raise ValueError("well_patterns is empty")

    batches = [
        patterns[i : i + wells_per_batch]
        for i in range(0, len(patterns), wells_per_batch)
    ]
    results: list[ResolvedWellPatterns] = []
    p_offset = 0
    for b_idx, batch in enumerate(batches):
        if progress is not None and progress.cancelled:
            if verbose:
                print(
                    f"[run_well_patterns] cancelled before batch {b_idx + 1}/"
                    f"{len(batches)}; stopping after {len(results)} batch(es)."
                )
            break
        if progress is not None:
            progress.report_progress(
                b_idx,
                len(batches),
                f"batch {b_idx + 1}/{len(batches)}: "
                f"{', '.join(wp.well for wp in batch)}",
            )
        if verbose:
            print(
                f"\n=== run_well_patterns: batch {b_idx + 1}/{len(batches)} "
                f"({len(batch)} wells: {[wp.well for wp in batch]}) ==="
            )
        resolved = resolve_well_patterns(
            microscope,
            batch,
            finder=finder,
            finder_factory=finder_factory,
            finder_class=finder_class,
            finder_kwargs=finder_kwargs,
            time_per_fov=time_per_fov,
            n_parallel=n_parallel,
            apply_batching=apply_batching,
            progress=progress,  # let the scan itself stop early on cancel
            verbose=verbose,
        )
        # A cancel during the scan returns partial/empty resolved events — do
        # NOT start this batch's (long) acquisition; stop here instead.
        if progress is not None and progress.cancelled:
            if verbose:
                print(
                    f"[run_well_patterns] cancelled during batch {b_idx + 1} "
                    f"scan; not running it. Stopping after {len(results)} batch(es)."
                )
            break
        n_fovs = len({e.index.get("p", 0) for e in resolved.events})
        if n_fovs == 0:
            if verbose:
                print(
                    f"[run_well_patterns] batch {b_idx + 1} produced no FOVs; skipping."
                )
            continue
        # Globally-unique position indices so batches don't collide in storage.
        events = _shift_positions(resolved.events, p_offset)

        if verbose:
            print(
                f"[run_well_patterns] batch {b_idx}: {n_fovs} FOVs "
                f"(p {p_offset}..{p_offset + n_fovs - 1}); "
                f"{'run_experiment' if b_idx == 0 else 'continue_experiment'} ..."
            )
        if b_idx == 0:
            handle = controller.run_experiment(
                events, stim_mode=stim_mode, validate=validate
            )
        else:
            handle = controller.continue_experiment(
                events,
                stim_mode=stim_mode,
                validate=validate,
                offset_timepoints=False,
            )
        handle.wait()  # block until acquisition done -> microscope free for next scan
        p_offset += n_fovs
        results.append(resolved)

    if finish:
        controller.finish_experiment()
    return results


def run_well_patterns_async(
    controller: "Controller",
    microscope: "AbstractMicroscope",
    well_patterns: Sequence[WellPattern],
    *,
    wells_per_batch: int,
    finder: FOVFinder | None = None,
    finder_factory: Callable[[str], FOVFinder] | None = None,
    finder_class: type[FOVFinder] = FOVFinderAgent,
    finder_kwargs: dict[str, Any] | None = None,
    time_per_fov: float | None = None,
    n_parallel: int | None = None,
    apply_batching: bool = True,
    stim_mode: str = "current",
    validate: bool = True,
    finish: bool = True,
    name: str = "WellPatternRun",
    verbose: bool = True,
) -> "OrchestratorHandle":
    """Launch :func:`run_well_patterns` on a controller worker thread.

    Thin convenience wrapper around
    :meth:`~faro.core.controller.Controller.run_orchestrator_async` so the
    notebook doesn't have to repeat ``controller`` or pass the orchestrator
    function by hand.  These two are equivalent::

        run_handle = ctrl.run_orchestrator_async(
            run_well_patterns, ctrl, mic, patterns,
            wells_per_batch=6, finder=finder, time_per_fov=3.3, ...,
        )

        run_handle = run_well_patterns_async(
            ctrl, mic, patterns,
            wells_per_batch=6, finder=finder, time_per_fov=3.3, ...,
        )

    The returned :class:`~faro.core.run_status.OrchestratorHandle` has the same
    ``status()`` / ``current_run`` / ``cancel()`` / ``wait()`` contract; the
    ``progress`` handle is injected automatically (so batch progress is
    reported and the run is cancellable).  All finder/run arguments are the
    same as :func:`run_well_patterns` — easiest is to pass a configured
    *finder* instance.

    Args:
        name: Worker-thread name (for debugging).
    """
    return controller.run_orchestrator_async(
        run_well_patterns,
        controller,
        microscope,
        well_patterns,
        wells_per_batch=wells_per_batch,
        finder=finder,
        finder_factory=finder_factory,
        finder_class=finder_class,
        finder_kwargs=finder_kwargs,
        time_per_fov=time_per_fov,
        n_parallel=n_parallel,
        apply_batching=apply_batching,
        stim_mode=stim_mode,
        validate=validate,
        finish=finish,
        verbose=verbose,
        name=name,
    )
