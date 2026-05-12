from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from faro.core.controller import Controller
    from faro.microscope.base import AbstractMicroscope


class Agent(ABC):
    """Base class for all agents.

    Agents make decisions that influence acquisition.  They are registered
    on the :class:`Controller` which sets ``self.controller`` so the agent
    can call back into the experiment (e.g. ``extend_experiment``,
    ``replace_remaining_events``).

    Subclasses must implement :meth:`run` and/or override
    :meth:`on_frame_processed` depending on the agent type.
    """

    required_metadata: set[str] = set()

    def __init__(self):
        self.controller: Controller | None = None  # set by Controller.__init__

    def on_frame_processed(self, result: dict) -> None:
        """Called by the Analyzer after each pipeline frame completes.

        Args:
            result: Dictionary with keys:

                * ``"df_tracked"`` -- tracked DataFrame for the FOV
                * ``"segmentation_results"`` -- ``{name: label_image}``
                * ``"metadata"`` -- event metadata dict

        Override for intra-experiment agents that react per-frame.
        Default is a no-op.
        """

    @abstractmethod
    def run(self) -> None:
        """Execute the agent's main logic."""
        ...

    def stop(self) -> None:
        """Request graceful stop.  Override for long-running agents."""


class PreExperimentAgent(Agent):
    """Agent that runs before acquisition to prepare experiment parameters.

    Typical use: scan plate positions, classify FOVs, select good ones.
    Does not use :meth:`on_frame_processed`.
    """

    def __init__(self, microscope: AbstractMicroscope):
        super().__init__()
        self.microscope = microscope

    @abstractmethod
    def run(self) -> dict:
        """Run pre-experiment analysis.

        Returns:
            Agent-specific result dict (e.g. selected positions, counts).
        """
        ...


class InterPhaseAgent(Agent):
    """Agent that orchestrates multi-phase experiments.

    Owns the experiment loop: calls ``controller.run_experiment()`` /
    ``continue_experiment()`` / ``finish_experiment()`` and reads
    pipeline results from parquet between phases.

    Implementations may expose :meth:`run_one_phase` so they can be driven
    externally — e.g. wrapped by :class:`ComposedAgent` to interleave
    pre-phase work (FOV finding, focus, calibration) between phases.
    Subclasses that override ``run()`` to do their own loop should also
    factor that loop body into ``run_one_phase`` so composition works.
    """

    def __init__(self, storage_path: str):
        super().__init__()
        self.storage_path = storage_path

    def read_tracks(self, fov: int, phase_id: int | None = None) -> pd.DataFrame:
        """Read the latest tracking parquet for a FOV.

        Args:
            fov: FOV index.
            phase_id: If set, reads the phase-specific parquet file.

        Returns:
            Tracked DataFrame, or empty DataFrame if file not found.
        """
        if phase_id is not None:
            fname = f"{fov}_phase_{phase_id}_latest.parquet"
        else:
            fname = f"{fov}_latest.parquet"
        path = os.path.join(self.storage_path, "tracks", fname)
        try:
            return pd.read_parquet(path)
        except FileNotFoundError:
            return pd.DataFrame()

    def run_one_phase(
        self,
        phase_id: int,
        fov_positions: list | None = None,
        fovs: list[int] | None = None,
    ) -> dict | None:
        """Run a single phase of the multi-phase experiment.

        Default implementation raises :class:`NotImplementedError`.
        Subclasses that want to be composable (driven by
        :class:`ComposedAgent`) should override this method to:

        1. Update internal FOV state from *fov_positions* / *fovs*.
        2. Decide what to acquire next (parameters, events).
        3. Call ``controller.run_experiment()`` (first phase) or
           ``controller.continue_experiment()`` (later phases).
        4. Wait for the pipeline to drain.
        5. Read tracks, update internal state, and return any result.

        Args:
            phase_id: Zero-based phase index.  Implementations should use
                this to decide between ``run_experiment`` (==0) and
                ``continue_experiment`` (>0).
            fov_positions: Optional list of stage positions for this
                phase (e.g. ``FovPosition`` namedtuples produced by a
                :class:`PreExperimentAgent`).  May be ``None`` when the
                inner agent already knows its FOVs.
            fovs: Optional list of FOV indices to use for this phase.
                If ``None``, the implementation should derive them from
                ``fov_positions`` (typically ``range(len(fov_positions))``).

        Returns:
            Optional result dict (e.g. ``{"df_new": ..., "params": ...}``)
            so callers can inspect per-phase outputs.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement run_one_phase(); "
            "override it to make this agent composable."
        )


class IntraExperimentAgent(Agent):
    """Agent that reacts during acquisition via per-frame callbacks.

    Called by the Analyzer after each pipeline frame completes (from the
    executor thread).  Can call ``controller.replace_remaining_events()``
    or ``controller.extend_experiment()`` to modify the running
    acquisition.
    """

    @abstractmethod
    def on_frame_processed(self, result: dict) -> None:
        """React to a processed frame.

        Called from the pipeline executor thread.  Must be thread-safe.
        ``controller.replace_remaining_events()`` and
        ``controller.extend_experiment()`` are safe to call from here.
        """
        ...

    def run(self) -> None:
        """Not used -- logic lives in :meth:`on_frame_processed`."""
