import numpy as np
from skimage.measure import label

import skimage
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table
import pandas as pd
from .abstract_fe_optocheck import FeatureExtractorOptoCheck
from ..utils import labels_to_particles

"""
Segmentation module for image processing.

This module contains classes for segmenting images. The base class Segmentator
defines the interface for all segmentators. Specific implementations should
inherit from this class and override the segment method.
"""


class OptoCheckFE(FeatureExtractorOptoCheck):

    def extract_features(self, segmentation_results, image, df_tracked):
        tracked_label = labels_to_particles(
            segmentation_results[self.used_mask], df_tracked
        )
        tracked_label = np.expand_dims(tracked_label, 0)
        table = skimage.measure.regionprops_table(
            tracked_label, image, properties=["label", "mean_intensity"]
        )
        table = pd.DataFrame.from_dict(table)
        table = table.rename(
            {
                "mean_intensity": "optocheck_mean_intensity",
                "label": "particle",
            },
            axis=1,
        )
        table = pd.DataFrame.from_dict(table)
        df_tracked = df_tracked.merge(table, on=["particle"])

        return df_tracked
