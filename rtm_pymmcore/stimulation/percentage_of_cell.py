from .base_stimulation import Stim
import numpy as np
import skimage
import math


class StimPercentageOfCell(Stim):
    """
    Stimulate a percentage of the cell.

    This class implements a stimulation that stimulates a percentage of the cell.
    The percentage can be parametrized.
    """

    def above_line(self, i, j, x2, y2, x3, y3):
        v1 = (x2 - x3, y2 - y3)
        v2 = (x2 - i, y2 - j)
        xp = v1[0] * v2[1] - v1[1] * v2[0]
        return xp > 0

    def get_stim_mask(self, label_images, metadata: dict = None, img: np.array = None) -> np.ndarray:
        label_image = label_images["labels"]
        light_map = np.zeros_like(label_image, dtype=bool)
        props = skimage.measure.regionprops(label_image)
        if metadata is None:
            metadata = {}
        percentage_of_stim = metadata.get("stim_cell_percentage", 0.3)

        try:
            extent = 0.5 - percentage_of_stim
            # Koordinaten-Arrays einmal erstellen für bessere Performance
            y_coords, x_coords = np.indices(label_image.shape)
            
            for prop in props:
                label = prop.label
                single_label = label_image == label

                orientation = prop.orientation
                y0, x0 = prop.centroid

                # Find point where cutoff line and major axis intersect
                x2 = x0 - math.sin(orientation) * extent * prop.major_axis_length
                y2 = y0 - math.cos(orientation) * extent * prop.major_axis_length

                # find second point on line
                length = 0.5 * prop.minor_axis_length
                x3 = x2 + (length * math.cos(-orientation))
                y3 = y2 + (length * math.sin(-orientation))

                v1_x = x3 - x2  
                v1_y = y3 - y2
                v2_x = x3 - x_coords  
                v2_y = y3 - y_coords
                
                cross_product = v1_x * v2_y - v1_y * v2_x
                cutoff_mask = cross_product > 0

                frame_labeled_expanded = skimage.segmentation.expand_labels(
                    single_label, 5
                )
                stim_mask = np.logical_and(cutoff_mask, frame_labeled_expanded)

                light_map = np.logical_or(light_map, stim_mask)
                
            return light_map.astype("uint8"), None
        except Exception as e:
            print(e)
            return np.zeros_like(label_image), None