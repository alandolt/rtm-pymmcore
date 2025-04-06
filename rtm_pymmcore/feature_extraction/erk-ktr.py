import numpy as np
import pandas as pd

from skimage.measure import label
from skimage.segmentation import expand_labels
from skimage.measure import regionprops_table

from rtm_pymmcore.feature_extraction.base_feature_extractor import FeatureExtractor


class FE_ErkKtr(FeatureExtractor):
    """
    Feature extractor for ERK-KTR biosensor.
    This class implements a feature extractor that extracts features from the
    ERK-KTR biosensor images. It creates a cytosolic ring around the nucleus and
    extracts mean intensities from both the nucleus and the ring.
    """

    def extract_ring(
        self, labels, margin=2, distance=4
    ):  # distance = 10 for 40x; 4px for 20x
        """Create the cytosolic rings for biosensor dependant on nuclear/cytosolic fluorescence intensity.
        Args:
            margin: nb pixels between nucleus and ring
            distance: nb pixels ring width (margin is subtracted)
        """
        labels_expanded_margin = expand_labels(labels, distance=margin)
        labels_expanded_rings = expand_labels(labels, distance=distance)
        labels_expanded_rings[labels_expanded_margin != 0] = 0
        return labels_expanded_rings.astype(int)

    def extract_features(self, labels, raw):
        """Create a table with features for every detected cell.
        Args:
            labels: frame with labeled nuclei
            raw: raw frame with dimensions [x,y,c]
            details: additional info from stardist (e.g. centroids)
        """
        raw = np.moveaxis(raw, 0, 2)  # CXY to XYC
        labels_ring = extract_ring(labels)  # create cytosolic rings
        # EXTRACT FEATURES
        table_nuc = regionprops_table(
            labels, raw, ["mean_intensity", "label", "centroid"]
        )  # extract features#"centroid"
        table_ring = regionprops_table(labels_ring, raw, ["mean_intensity", "label"])

        # CREATE TABLES
        table_nuc = pd.DataFrame.from_dict(table_nuc)
        table_ring = pd.DataFrame.from_dict(table_ring)
        table_nuc = table_nuc.rename(
            {
                "mean_intensity-0": "mean_intensity_C0_nuc",
                "mean_intensity-1": "mean_intensity_C1_nuc",
                "mean_intensity-2": "mean_intensity_C2_nuc",
            },
            axis="columns",
        )
        table_ring = table_ring.rename(
            {
                "mean_intensity-0": "mean_intensity_C0_ring",
                "mean_intensity-1": "mean_intensity_C1_ring",
                "mean_intensity-2": "mean_intensity_C2_ring",
            },
            axis="columns",
        )

        # CONCAT TABLES
        table = table_nuc.merge(table_ring, on=["label"])

        # CALCULATE the ERK ratio
        # table['ratio_ERK'] = table['mean_intensity_C1_ring']/table['mean_intensity_C1_nuc']

        # TODO add the points from stardist
        # table['x'] = details["points"][:,0]
        # table['y'] = details["points"][:,1]
        # need to match points by label, as sometimes nb cells can differ between label mask and detected nuclei.
        # this is very rare and hard to catch:
        # ValueError: Length of values (291) does not match length of index (290)
        # for the moment use centroids from label map region props.
        table = table.rename({"centroid-0": "x", "centroid-1": "y"}, axis="columns")

        # add empty particle column (this will be filled later if there are particles)
        table["particle"] = pd.Series(dtype="int")

        return table, labels_ring
