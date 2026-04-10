from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from rtm_pymmcore.agents.base import PreExperimentAgent

if TYPE_CHECKING:
    from rtm_pymmcore.microscope.base import AbstractMicroscope
    from rtm_pymmcore.segmentation.base import Segmentator


class FOVSelectorAgent(PreExperimentAgent):
    """Scan candidate positions and select FOVs with enough cells.

    Acquires a single snapshot at each candidate position, segments it,
    and returns the *n_select* positions with the highest cell count
    (filtered by *min_cells*).

    Args:
        microscope: Microscope instance (needs ``mmc`` for stage control
            and image acquisition).
        candidate_positions: List of ``(x, y, z)`` tuples to scan.
            ``z`` may be ``None`` if not relevant.
        segmentator: A :class:`Segmentator` instance used to count cells.
        n_select: Maximum number of FOVs to return.
        min_cells: Minimum cell count for a position to be considered.
        channel_config: Optional channel configuration name to set before
            snapping (e.g. ``"Phase"``).
    """

    def __init__(
        self,
        microscope: AbstractMicroscope,
        *,
        candidate_positions: list[tuple[float, float, float | None]],
        segmentator: Segmentator,
        n_select: int = 10,
        min_cells: int = 5,
        channel_config: str | None = None,
    ):
        super().__init__(microscope)
        self.candidate_positions = candidate_positions
        self.segmentator = segmentator
        self.n_select = n_select
        self.min_cells = min_cells
        self.channel_config = channel_config

    def run(self) -> dict:
        """Snap an image at each position, segment, count cells.

        Returns:
            Dictionary with keys:

            * ``"positions"`` -- list of ``(x, y, z)`` tuples for selected FOVs
            * ``"cell_counts"`` -- corresponding cell counts
            * ``"all_results"`` -- full DataFrame with all scanned positions
        """
        mmc = self.microscope.mmc
        results: list[dict] = []

        for x, y, z in self.candidate_positions:
            mmc.setXYPosition(x, y)
            if z is not None:
                mmc.setPosition(z)
            if self.channel_config:
                mmc.setConfig(self.channel_config)
            mmc.snapImage()
            img = mmc.getImage()
            labels = self.segmentator.segment(img)
            n_cells = int(labels.max()) if labels.max() > 0 else 0
            results.append({"x": x, "y": y, "z": z, "n_cells": n_cells})

        df = pd.DataFrame(results)
        df_good = df[df["n_cells"] >= self.min_cells].nlargest(self.n_select, "n_cells")
        return {
            "positions": list(
                df_good[["x", "y", "z"]].itertuples(index=False, name=None)
            ),
            "cell_counts": list(df_good["n_cells"]),
            "all_results": df,
        }
