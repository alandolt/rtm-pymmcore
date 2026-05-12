"""Condition-based monitoring agent (plug-and-play).

A :class:`ConditionMonitorAgent` runs a scouting / monitoring experiment
and watches a tracked feature in real time.  When a
:class:`~faro.agents.base.Condition` is met it stops scouting early and
triggers a response — either a fixed :class:`RTMSequence` or a delegate
:class:`InterPhaseAgent`.

This is the agent-framework equivalent of plugging a segmentator into the
image-processing pipeline: the user declares *what* to watch
(:class:`Condition`) and *how* to react (a sequence or another agent),
and the framework handles all lifecycle plumbing.

Simple usage::

    from faro.agents import Condition, ConditionMonitorAgent

    monitor = ConditionMonitorAgent(
        condition=Condition("cnr", "below", 1.0),
        scouting_sequence=scouting,          # RTMSequence for monitoring
        response_sequence=fast_sequence,      # RTMSequence to run on trigger
        storage_path=path,
    )
    ctrl = Controller(mic, pipeline, writer=writer, agent=monitor)
    monitor.run()

Composed usage (with FOV finder + BO response)::

    monitor = ConditionMonitorAgent(
        condition=Condition("cnr", "below", 1.0),
        scouting_sequence=scouting,
        response_agent=bo_agent,             # any InterPhaseAgent
        storage_path=path,
    )
    composed = ComposedAgent(
        inner_agent=monitor,
        pre_phase_agents=[fov_finder],
        n_phases=10,
    )
    ctrl = Controller(mic, pipeline, writer=writer, agent=composed)
    composed.run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from faro.agents.base import Condition, InterPhaseAgent
from faro.core.data_structures import RTMSequence

if TYPE_CHECKING:
    pass


class ConditionMonitorAgent(InterPhaseAgent):
    """Monitor a feature during scouting; trigger a response when condition is met.

    Combines intra-experiment monitoring (via :meth:`on_frame_processed`)
    with inter-phase lifecycle control (via :meth:`run_one_phase`), so it
    works both standalone and inside a :class:`ComposedAgent`.

    The agent runs in two stages per phase:

    1. **Scouting** — acquire using *scouting_sequence*.  Each processed
       frame is checked against *condition*.  When the condition fires,
       remaining scouting events are dropped immediately via
       ``controller.replace_remaining_events([])``.
    2. **Response** — if triggered, either:

       * run *response_sequence* (a fixed :class:`RTMSequence`), **or**
       * delegate to *response_agent*'s ``run_one_phase()``.

       If scouting completes without triggering, the phase ends with no
       response (the result dict indicates ``"triggered": False``).

    Args:
        condition: A :class:`Condition` describing what to watch.
        scouting_sequence: :class:`RTMSequence` for the monitoring phase.
        response_sequence: Optional :class:`RTMSequence` to run on trigger.
            Mutually exclusive with *response_agent*.
        response_agent: Optional :class:`InterPhaseAgent` to delegate to
            on trigger.  Mutually exclusive with *response_sequence*.
        n_phases: Number of phases when running standalone via :meth:`run`.
        storage_path: Passed to :class:`InterPhaseAgent`.
    """

    def __init__(
        self,
        *,
        condition: Condition,
        scouting_sequence: RTMSequence,
        response_sequence: RTMSequence | None = None,
        response_agent: InterPhaseAgent | None = None,
        n_phases: int = 1,
        storage_path: str = "",
    ):
        super().__init__(storage_path=storage_path)

        if response_sequence is not None and response_agent is not None:
            raise ValueError(
                "Provide either response_sequence or response_agent, not both."
            )

        self.condition = condition
        self.scouting_sequence = scouting_sequence
        self.response_sequence = response_sequence
        self.response_agent = response_agent
        self.n_phases = n_phases

        self._triggered = False
        self._phase_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Per-frame monitoring (called from pipeline executor thread)
    # ------------------------------------------------------------------

    def on_frame_processed(self, result: dict) -> None:
        """Check the condition on every processed frame.

        When the condition fires, scouting is stopped immediately by
        replacing all remaining events with an empty list.
        """
        if self._triggered:
            return

        df = result.get("df_tracked")
        if df is None or df.empty:
            return

        if self.condition.check(df):
            self._triggered = True
            print(
                f"[ConditionMonitorAgent] Condition met: "
                f"{self.condition.feature} {self.condition.operator} "
                f"{self.condition.threshold}"
            )
            # Stop scouting immediately
            self.controller.replace_remaining_events([])

    # ------------------------------------------------------------------
    # Phase lifecycle (composable via ComposedAgent)
    # ------------------------------------------------------------------

    def run_one_phase(
        self,
        phase_id: int,
        fov_positions: list | None = None,
        fovs: list[int] | None = None,
    ) -> dict | None:
        """Run one monitor-then-respond cycle.

        1. Start scouting (``run_experiment`` or ``continue_experiment``).
        2. ``on_frame_processed`` monitors the condition.
        3. If triggered, run the response.
        4. Return a result dict.
        """
        self._triggered = False

        # --- Build scouting events ------------------------------------
        scouting_events = list(self.scouting_sequence)

        # Inject fov_positions into scouting if provided
        if fov_positions is not None:
            scouting_seq = RTMSequence(
                time_plan=self.scouting_sequence.time_plan,
                channels=[
                    {"config": ch.config, "exposure": ch.exposure}
                    for ch in self.scouting_sequence.channels
                ],
                stage_positions=fov_positions,
            )
            scouting_events = list(scouting_seq)

        # --- Run scouting acquisition ---------------------------------
        print(
            f"[ConditionMonitorAgent] Phase {phase_id}: scouting "
            f"({len(scouting_events)} events, watching "
            f"{self.condition.feature} {self.condition.operator} "
            f"{self.condition.threshold})"
        )
        if phase_id == 0:
            self.controller.run_experiment(scouting_events, validate=False)
        else:
            self.controller.continue_experiment(scouting_events, validate=False)

        self._wait_for_pipeline()

        # --- Response (only if condition was triggered) ----------------
        response_result = None

        if self._triggered:
            if self.response_agent is not None:
                # Delegate to the response agent
                self.response_agent.controller = self.controller
                response_result = self.response_agent.run_one_phase(
                    phase_id=phase_id,
                    fov_positions=fov_positions,
                    fovs=fovs,
                )
            elif self.response_sequence is not None:
                response_events = list(self.response_sequence)
                if fov_positions is not None:
                    resp_seq = RTMSequence(
                        time_plan=self.response_sequence.time_plan,
                        channels=[
                            {"config": ch.config, "exposure": ch.exposure}
                            for ch in self.response_sequence.channels
                        ],
                        stage_positions=fov_positions,
                    )
                    response_events = list(resp_seq)

                print(
                    f"[ConditionMonitorAgent] Phase {phase_id}: "
                    f"running response sequence ({len(response_events)} events)"
                )
                self.controller.continue_experiment(response_events, validate=False)
                self._wait_for_pipeline()
        else:
            print(
                f"[ConditionMonitorAgent] Phase {phase_id}: "
                f"scouting completed without trigger"
            )

        result = {
            "phase_id": phase_id,
            "triggered": self._triggered,
            "response_result": response_result,
        }
        self._phase_results.append(result)
        return result

    # ------------------------------------------------------------------
    # Standalone run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run all phases sequentially (standalone mode).

        For composed usage, :class:`ComposedAgent` calls
        :meth:`run_one_phase` directly.
        """
        if self.controller is None:
            raise RuntimeError(
                "ConditionMonitorAgent has no controller; instantiate via "
                "Controller(mic, pipeline, agent=monitor) first."
            )

        for phase in range(self.n_phases):
            print(
                f"\n{'=' * 60}\n"
                f"=== ConditionMonitorAgent: phase {phase + 1}/{self.n_phases} ===\n"
                f"{'=' * 60}"
            )
            self.run_one_phase(phase_id=phase)

        self.controller.finish_experiment()
