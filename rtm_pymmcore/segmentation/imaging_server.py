import numpy as np
from .base_segmentation import Segmentator
import skimage
import imaging_server_kit

"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class SegmentatorImagingServerKit(Segmentator):

    def __init__(
        self, server: str, algorithm: str, model_param: dict = None, min_size: int = 0
    ):

        self.algorithm = algorithm
        self.model_param = model_param
        self.client = imaging_server_kit.Client(server)
        self.min_size = min_size

    def segment(self, img: np.ndarray) -> np.ndarray:
        """
        Run the an imagekit model on data and do post-processing (remove small cells)
        """
        params = {"image": img}
        if self.model_param is not None:
            params.update(self.model_param)

        labels = self.client.run_algorithm(self.algorithm, **params)[0][0]
        if self.min_size > 0:
            # remove cells below threshold
            labels = skimage.morphology.remove_small_objects(
                labels, min_size=self.min_size, connectivity=1
            )
        return labels
