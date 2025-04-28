import numpy as np
import pandas as pd


class FeatureExtractorOptoCheck:
    """
    Base class for all segmentators. Specific implementations should inherit
    from this class and override this method.
    """

    def __init__(self, used_mask):
        self.used_mask = used_mask

    def extract_features(
        self, segmentation_results: dict, image: np.ndarray, df_tracked: pd.DataFrame
    ) -> pd.DataFrame:
        raise NotImplementedError("Subclasses should implement this!")
