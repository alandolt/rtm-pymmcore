import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import pandas as pd
from .base import FeatureExtractor


"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SimpleFE(FeatureExtractor):
    def __init__(self, used_mask):
        self.used_mask = used_mask
        super().__init__()

    def extract_features(self, labels, image, df_tracked=None, metadata=None):
        table = skimage.measure.regionprops_table(
            labels[self.used_mask], properties=["label", "area"]
        )
        table = pd.DataFrame.from_dict(table)
        return table, None


class SpatialFE(FeatureExtractor):
    """Per-cell *spatial* features: centroid, area and nearest-neighbour distance.

    Unlike intensity-based extractors (e.g.
    :class:`~faro.feature_extraction.erk_ktr.FE_ErkKtr`), this one needs only a
    label image — it describes *where* cells are, not how bright they are.

    Its home is :class:`~faro.agents.fov_finder.FOVFinderAgent`, whose scan only
    *counts* cells.  Pair it with an
    :class:`~faro.agents.fov_finder.FOVCondition` on ``nn_dist`` to reject
    clumped fields there, e.g.::

        FOVCondition("nn_dist", "above", 15.0, min_fraction=0.7)

    i.e. "at least 70 % of cells have their nearest neighbour > 15 µm away".

    **You do not need this for** :class:`~faro.agents.grid_fov_finder.GridFOVFinderAgent`:
    that finder already computes cell **count and clumping geometrically** from
    the reconstructed centroid cloud (its ``min_cells`` / ``max_cells`` /
    ``clump_distance_um`` / ``max_clumped_fraction`` / ``min_nn_um`` /
    ``max_nn_um`` knobs — the same nearest-neighbour logic, in
    :mod:`faro.agents.fov_density`).  There, ``fov_conditions`` are reserved for
    per-cell **biology** the geometry can't see (e.g. ERK ``cnr`` via
    ``FE_ErkKtr``); expressing clumping as an ``nn_dist`` condition would just
    duplicate the built-in density band.

    Args:
        used_mask: Key into the ``labels`` dict selecting the label image.
        pixel_size_um: Camera pixel size (µm / px) used to report ``x``/``y``,
            ``nn_dist`` and ``area_um2`` in µm.  Leave at ``1.0`` to work in
            pixel units.

    Output columns (one row per cell): ``label``, ``area`` (px), ``area_um2``,
    ``x``, ``y`` (centroid, µm), ``nn_dist`` (µm; ``inf`` for a lone cell).
    """

    def __init__(self, used_mask, pixel_size_um: float = 1.0):
        self.used_mask = used_mask
        self.pixel_size_um = float(pixel_size_um)
        super().__init__()

    def extract_features(self, labels, image=None, df_tracked=None, metadata=None):
        from faro.agents.fov_density import nearest_neighbor_distances

        lab = labels[self.used_mask]
        table = skimage.measure.regionprops_table(
            lab, properties=["label", "area", "centroid"]
        )
        df = pd.DataFrame.from_dict(table)
        # regionprops centroid-0 = row (image y), centroid-1 = col (image x).
        df = df.rename(columns={"centroid-0": "y", "centroid-1": "x"})
        df["x"] = df["x"] * self.pixel_size_um
        df["y"] = df["y"] * self.pixel_size_um
        df["area_um2"] = df["area"].astype(float) * self.pixel_size_um**2
        xy = df[["x", "y"]].to_numpy(dtype=float)
        df["nn_dist"] = nearest_neighbor_distances(xy)
        return df, None
