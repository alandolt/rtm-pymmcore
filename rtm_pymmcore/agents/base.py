from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from rtm_pymmcore.core.controller import Controller
    from rtm_pymmcore.microscope.base import AbstractMicroscope


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
