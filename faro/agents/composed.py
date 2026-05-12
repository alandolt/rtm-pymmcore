"""Composed inter-phase agent: drive an inner agent with per-phase pre-agents.

A :class:`ComposedAgent` wraps any :class:`InterPhaseAgent` (the *inner*
acquisition driver, e.g. :class:`faro.agents.bo_optimization.BOptGPAX`) and
runs an arbitrary list of :class:`PreExperimentAgent` instances before each
phase.  This is the generic mechanism for combining agents in the framework:

* Pre-phase agent(s) decide *where* (or how) to acquire — e.g.
  :class:`faro.agents.fov_finder.FOVFinderAgent` picks fresh FOVs from a
  fresh batch of wells, an autofocus agent could refresh Z, etc.
* The inner inter-phase agent decides *what* to acquire and processes the
  results — e.g. Bayesian Optimisation, adaptive sampling, scheduled
  multi-condition sweeps, …

Any agent that implements ``run_one_phase(phase_id, fov_positions, fovs)``
on the :class:`InterPhaseAgent` interface can be used as the inner agent,
so the composition is fully pluggable: write a new agent, drop it in.

Typical use::

    finder = FOVFinderAgent(mic, ...)
    bo = OscillationBO(...)

    composed = ComposedAgent(
        inner_agent=bo,
        pre_phase_agents=[finder],
        n_phases=10,
    )
    ctrl = Controller(mic, pipeline, writer=writer, agent=composed)
    composed.run()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from faro.agents.base import InterPhaseAgent, PreExperimentAgent

if TYPE_CHECKING:
    from faro.core.controller import Controller


class ComposedAgent(InterPhaseAgent):
    """Run pre-phase agents and an inner :class:`InterPhaseAgent` per phase.

    The composed agent owns the outer phase loop:

    1. For each phase ``k`` in ``range(n_phases)``:

       a. Each pre-phase agent's ``run()`` method is called in order.
          The result of the *last* pre-phase agent that returns a dict
          containing ``"positions"`` is used as the FOV positions for
          this phase.
       b. ``inner_agent.run_one_phase(phase_id=k, fov_positions=...)``
          is called.

    2. After all phases finish, ``inner_agent.controller.finish_experiment()``
       is called.

    The composed agent forwards its ``controller`` to the inner agent so
    that wiring it through ``Controller(mic, pipeline, agent=composed)``
    Just Works (the inner agent's ``run_one_phase`` will see the same
    controller).

    Args:
        inner_agent: An :class:`InterPhaseAgent` that implements
            ``run_one_phase``.  Typically a BO agent or any custom
            multi-phase agent.
        pre_phase_agents: List of :class:`PreExperimentAgent` instances
            that run *before* each inner phase.  Their results are passed
            on to the inner agent (the last result containing
            ``"positions"`` wins).  Pass an empty list to drive the
            inner agent without per-phase pre-work (rare; usually you
            would just call the inner agent directly in that case).
        n_phases: Total number of phases to run.
        finish_experiment: Whether to call ``controller.finish_experiment()``
            after the loop completes.  Defaults to ``True``.  Set to
            ``False`` if you want to chain another composed run.
    """

    def __init__(
        self,
        *,
        inner_agent: InterPhaseAgent,
        pre_phase_agents: list[PreExperimentAgent] | None = None,
        n_phases: int,
        finish_experiment: bool = True,
    ):
        if not isinstance(inner_agent, InterPhaseAgent):
            raise TypeError(
                f"inner_agent must be an InterPhaseAgent, got "
                f"{type(inner_agent).__name__}"
            )
        if n_phases <= 0:
            raise ValueError("n_phases must be positive")

        # Must be set BEFORE super().__init__() because the base ``Agent``
        # initialiser does ``self.controller = None``, which routes through
        # this class's ``controller`` setter and forwards to the inner agent.
        self.inner_agent = inner_agent
        super().__init__(storage_path=getattr(inner_agent, "storage_path", ""))
        self.pre_phase_agents = list(pre_phase_agents or [])
        for a in self.pre_phase_agents:
            if not isinstance(a, PreExperimentAgent):
                raise TypeError(
                    f"pre_phase_agents must contain PreExperimentAgent "
                    f"instances; got {type(a).__name__}"
                )
        self.n_phases = int(n_phases)
        self.finish_experiment = bool(finish_experiment)

        self.phase_results: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Controller propagation
    # ------------------------------------------------------------------

    @property
    def controller(self) -> Controller | None:
        return self.inner_agent.controller

    @controller.setter
    def controller(self, value: Controller | None) -> None:
        # When the user wires this composed agent into a Controller via
        # ``Controller(..., agent=composed)``, propagate the back-reference
        # to the inner agent so its ``run_one_phase`` sees the controller.
        self.inner_agent.controller = value

    def on_frame_processed(self, result: dict) -> None:
        """Forward per-frame callbacks to the inner agent.

        The Analyzer calls ``on_frame_processed`` on the top-level agent
        registered with the Controller — which is this ComposedAgent.
        Without forwarding, inner agents that rely on per-frame data
        (e.g. :class:`ConditionMonitorAgent`) would never see it.
        """
        self.inner_agent.on_frame_processed(result)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Run all phases sequentially."""
        if self.controller is None:
            raise RuntimeError(
                "ComposedAgent has no controller; instantiate via "
                "Controller(mic, pipeline, agent=composed_agent) first."
            )

        for phase in range(self.n_phases):
            print(
                f"\n{'=' * 60}\n"
                f"=== ComposedAgent: phase {phase + 1}/{self.n_phases} ===\n"
                f"{'=' * 60}"
            )
            fov_positions = None
            pre_results: list[dict[str, Any]] = []
            for pre in self.pre_phase_agents:
                result = pre.run()
                if isinstance(result, dict):
                    pre_results.append(result)
                    if "positions" in result:
                        fov_positions = result["positions"]
                elif isinstance(result, list):
                    # Bare list-of-positions return (e.g. FOVFinderAgent).
                    fov_positions = result

            if not self.pre_phase_agents:
                # No pre-phase agents — call inner without positions; it
                # must already have FOVs configured.
                inner_result = self.inner_agent.run_one_phase(phase_id=phase)
            else:
                inner_result = self.inner_agent.run_one_phase(
                    phase_id=phase,
                    fov_positions=fov_positions,
                )
            self.phase_results.append(
                {
                    "phase": phase,
                    "pre_results": pre_results,
                    "inner_result": inner_result,
                }
            )

        if self.finish_experiment and self.controller is not None:
            self.controller.finish_experiment()

    def run_one_phase(
        self,
        phase_id: int,
        fov_positions: list | None = None,
        fovs: list[int] | None = None,
    ) -> dict | None:
        """Allow nesting: run a single phase of this composed agent.

        Useful when you want to nest a ``ComposedAgent`` inside another
        ``ComposedAgent``.  Pre-phase agents are still invoked, and
        *fov_positions* / *fovs* arguments are forwarded only when no
        pre-phase agent supplied them.
        """
        pre_positions = None
        pre_results: list[dict[str, Any]] = []
        for pre in self.pre_phase_agents:
            result = pre.run()
            if isinstance(result, dict):
                pre_results.append(result)
                if "positions" in result:
                    pre_positions = result["positions"]
            elif isinstance(result, list):
                # Bare list-of-positions return (e.g. FOVFinderAgent).
                pre_positions = result

        positions = pre_positions if pre_positions is not None else fov_positions
        inner_result = self.inner_agent.run_one_phase(
            phase_id=phase_id,
            fov_positions=positions,
            fovs=fovs,
        )
        return {"pre_results": pre_results, "inner_result": inner_result}
