import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import pandas as pd


"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class Segmentator:
    """
    Base class for all segmentators. Specific implementations should inherit
    from this class and override this method.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Parameters:
        image (np.ndarray): The image to segment.

        Returns:
        np.ndarray: The segmented image.
        """
        raise NotImplementedError("Subclasses should implement this!")


class SegmentatorBinary(Segmentator):
    """
    Binary segmentator.

    This class implements a simple binary segmentation. It segments an image
    by setting all non-zero pixels to 1 and all zero pixels to 0.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        binary_image = image > 0
        label_image = label(binary_image)
        return label_image


class DummySegmentator(Segmentator):
    """
    Dummy segmentator.

    This class implements a dummy segmentator that returns a label image where
    the mask is the whole input image.
    """

    def segment(self, image: np.ndarray) -> np.ndarray:
        return np.ones_like(image)
