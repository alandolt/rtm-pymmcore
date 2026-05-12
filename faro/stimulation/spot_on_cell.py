"""Per-cell point-stimulation: place a small disk on each segmented cell.

Two variants:

* :class:`StimSpotOnCell` — discrete ``"top" | "middle" | "bottom"`` (matches
  Moritz EXP_24).
* :class:`StimSpotOnCellPolar` — continuous ``(angle_deg, radial_fraction)``
  (and optional ``spot_radius``) for Bayesian optimization.
"""

from __future__ import annotations

import math

import numpy as np
from skimage.draw import disk
from skimage.measure import regionprops

from .base import StimWithPipeline


def _paint_disk(out: np.ndarray, cy: float, cx: float, radius: int) -> None:
    rr, cc = disk((cy, cx), max(1, int(radius)), shape=out.shape)
    out[rr, cc] = True


class StimSpotOnCell(StimWithPipeline):
    """Place a disk on each cell at ``"top"``, ``"middle"``, or ``"bottom"``.

    Reproduces ``get_shape_percentage()`` from Moritz EXP_24 — same reference
    points and same default parameters (``spot_radius=5``,
    ``height_percentage=0.6``, ``offset=-5``):

    * ``"top"``    — topmost mask pixel (smallest y), pushed ``-offset`` px down.
    * ``"bottom"`` — bottommost mask pixel (largest y), pushed ``-offset`` px up.
    * ``"middle"`` — leftmost mask pixel at ``height_percentage`` of bbox height,
      pushed ``-offset`` px right.

    A negative ``offset`` (the Moritz default) moves the spot inward, so most
    of the disk lands inside the cell mask. ``clip_to_cell=True`` (default,
    extra to Moritz) intersects the spot with the source cell so light cannot
    leak onto neighbours; pass ``False`` to match Moritz exactly. Reads
    ``metadata["stim_location"]`` per event.
    """

    required_metadata = {"stim_location"}

    def __init__(
        self,
        spot_radius: int = 5,
        height_percentage: float = 0.6,
        offset: int = -5,
        clip_to_cell: bool = True,
        used_mask: str = "labels",
    ):
        self.spot_radius = int(spot_radius)
        self.height_percentage = float(height_percentage)
        self.offset = int(offset)
        self.clip_to_cell = bool(clip_to_cell)
        # Name of the segmentation entry to use. Pass e.g. ``"cells"`` when the
        # pipeline runs two segmentations (nucleus="labels" + whole-cell="cells")
        # so the spot lands on the cell shape, not the nucleus.
        self.used_mask = used_mask

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images[self.used_mask]
        location = (metadata or {})["stim_location"]
        light = np.zeros(labels.shape, dtype=bool)

        for prop in regionprops(labels):
            cell = labels == prop.label
            ys, xs = np.where(cell)
            if ys.size == 0:
                continue
            minr, _, maxr, _ = prop.bbox

            if location == "top":
                top_y = ys.min()
                y = top_y - self.offset             # offset=-5 -> shift +5 (down/into cell)
                x = xs[ys == top_y].mean()
            elif location == "bottom":
                bot_y = ys.max()
                y = bot_y + self.offset             # offset=-5 -> shift -5 (up/into cell)
                x = xs[ys == bot_y].mean()
            elif location == "middle":
                y = int(minr + self.height_percentage * (maxr - minr))
                row_xs = np.where(cell[y, :])[0]
                if row_xs.size == 0:
                    continue
                x = row_xs.min() - self.offset      # offset=-5 -> shift +5 (right/into cell)
            else:
                raise ValueError(
                    f"stim_location must be 'top'|'middle'|'bottom', got {location!r}"
                )

            spot = np.zeros_like(light)
            _paint_disk(spot, y, x, self.spot_radius)
            if self.clip_to_cell:
                spot &= cell
            light |= spot

        return light.astype("uint8"), None


class StimSpotOnCellPolar(StimWithPipeline):
    """Place a disk on each cell at a continuous ``(angle, radial_fraction)``.

    Reads from event metadata:

    * ``stim_angle_deg`` — direction of the spot from the centroid. With
      ``align_to_major_axis=True`` (default; best for elongated/anchor-shaped
      cells), 0° points along the cell's major axis (one anchor) and ±90° toward
      the sides. With ``align_to_major_axis=False`` it is image-aligned —
      0° = +y (down in image), 90° = +x (right).
    * ``stim_radial_fraction`` — distance along that ray, normalized so
      0.0 = centroid, 1.0 = on the cell's elliptical boundary.
    * ``stim_spot_radius`` (optional) — disk radius in px; falls back to the
      constructor default. Use this as a third Bayesian-opt axis if needed.

    Boundary distance uses the regionprops major/minor-axis ellipse — same
    approximation ``StimPercentageOfCell`` already relies on.
    """

    required_metadata = {"stim_angle_deg", "stim_radial_fraction"}

    def __init__(
        self,
        spot_radius: int = 5,
        align_to_major_axis: bool = True,
        used_mask: str = "labels",
    ):
        self.spot_radius = int(spot_radius)
        self.align_to_major_axis = bool(align_to_major_axis)
        self.used_mask = used_mask  # see StimSpotOnCell for rationale

    def get_stim_mask(self, label_images, metadata=None, img=None, tracks=None):
        labels = label_images[self.used_mask]
        meta = metadata or {}
        angle = math.radians(float(meta["stim_angle_deg"]))
        rf = float(meta["stim_radial_fraction"])
        radius = int(meta.get("stim_spot_radius", self.spot_radius))

        light = np.zeros(labels.shape, dtype=bool)
        for prop in regionprops(labels):
            a = max(prop.major_axis_length / 2, 1.0)
            b = max(prop.minor_axis_length / 2, 1.0)
            theta = prop.orientation  # major axis: (cos θ, sin θ) in (dy, dx)

            # ``local`` = angle in cell frame (used by the ellipse polar formula).
            # ``world`` = angle in image frame (used to project onto (dy, dx)).
            if self.align_to_major_axis:
                local, world = angle, angle + theta
            else:
                local, world = angle - theta, angle

            r_boundary = (a * b) / math.hypot(b * math.cos(local), a * math.sin(local))
            dy, dx = math.cos(world), math.sin(world)

            cy, cx = prop.centroid
            sy = cy + rf * r_boundary * dy
            sx = cx + rf * r_boundary * dx

            spot = np.zeros_like(light)
            _paint_disk(spot, sy, sx, radius)
            light |= spot & (labels == prop.label)

        return light.astype("uint8"), None
