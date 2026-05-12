"""Stimulate a small dot at a relative position on each cell."""

import numpy as np
from skimage.measure import regionprops
from skimage.morphology import disk, dilation

from .base import StimWithPipeline


class StimDotOnCell(StimWithPipeline):
    """Place a circular dot at a relative (y, x) position on each cell.

    Reads ``stim_y`` and ``stim_x`` from event metadata:

    * ``stim_y = 0`` → bottom of cell bounding box
    * ``stim_y = 1`` → top of cell bounding box
    * ``stim_x = 0`` → left edge
    * ``stim_x = 1`` → right edge

    Args:
        radius: Dot radius in pixels.
    """

    required_metadata: set[str] = {"stim_y", "stim_x"}

    def __init__(self, radius: int = 8):
        self.radius = radius

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images["labels"]
        h, w = labels.shape
        mask = np.zeros((h, w), dtype=np.uint8)

        stim_y_rel = metadata.get("stim_y", 0.5)
        stim_x_rel = metadata.get("stim_x", 0.5)
        selem = disk(self.radius)

        for prop in regionprops(labels):
            minr, minc, maxr, maxc = prop.bbox
            # Map relative position to pixel coordinates within bbox
            # stim_y=1 → top of cell (low row index in image coords)
            dot_r = int(maxr - stim_y_rel * (maxr - minr))
            dot_c = int(minc + stim_x_rel * (maxc - minc))
            dot_r = np.clip(dot_r, 0, h - 1)
            dot_c = np.clip(dot_c, 0, w - 1)

            local = np.zeros((h, w), dtype=np.uint8)
            local[dot_r, dot_c] = 1
            local = dilation(local, footprint=selem)
            mask = np.maximum(mask, local)

        return mask, None
