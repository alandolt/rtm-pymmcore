"""Single-cell batch BO agent for ERK oscillation -- BoTorch backend.

Pairs the per-cell preprocessing of
:class:`faro.agents.bo_oscillation_single_cell.OscillationBOSingleCell`
with the BoTorch GP backend instead of gpax/viSparseGP.

Uses ``SingleTaskGP`` (ExactGP, O(N^3)) — plenty fast for low hundreds
of cells, starts to drag above ~3k.  Switch to the gpax-sparse variant
(:class:`OscillationBOSingleCell`) when the cumulative training set
grows beyond that.
"""

from __future__ import annotations

import pandas as pd

from faro.agents.bo_botorch import BOptBoTorch
from faro.agents.bo_botorch_oscillation import OscillationBOBoTorch
from faro.agents.bo_oscillation_single_cell import OscillationBOSingleCell


class OscillationBOSingleCellBoTorch(OscillationBOBoTorch):
    """Single-cell BO + BoTorch backend.

    Inherits:
    * per-cell ``_preprocess_results`` from
      :class:`OscillationBOSingleCell`  (via a direct call -- see below).
    * event creation, oscillation scoring, plotting from
      :class:`OscillationBO`.
    * GP fitting / acquisition / save_model from
      :class:`BOptBoTorch`.

    We re-use ``OscillationBOSingleCell._preprocess_results`` directly
    (static method call) instead of multiple-inheriting both trees,
    since mixing :class:`BOptGPAXSparse` (gpax) with :class:`BOptBoTorch`
    (BoTorch) in the MRO causes the run loop to try to fit two GPs.
    """

    def __init__(self, *, density_k_neighbours: int = 5, **kwargs):
        super().__init__(**kwargs)
        self.density_k_neighbours = int(density_k_neighbours)

    # Forward to the gpax-sparse subclass's implementations; they're
    # backend-agnostic (only touch the classifier, quality gates, and
    # feature extraction -- no GP involved).
    def _preprocess_results(self, fov_tracks: dict) -> pd.DataFrame:
        return OscillationBOSingleCell._preprocess_results(self, fov_tracks)

    # ``_preprocess_results`` calls ``self._compute_per_cell_density``
    # inside the loop; expose it on this class too.
    _compute_per_cell_density = staticmethod(
        OscillationBOSingleCell._compute_per_cell_density
    )
