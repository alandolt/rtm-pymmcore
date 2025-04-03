# combines a segmentor, stimulator and tracker into a image processing pipeline.

import os

import numpy as np
import pandas as pd
import tifffile
from useq import MDAEvent

import rtm_pymmcore.segmentation.base_segmentator as base_segmentator
import rtm_pymmcore.stimulation.base_stim as base_stim
import rtm_pymmcore.tracking.base_tracker as base_tracker
from rtm_pymmcore.data_structures import Fov, ImgType
from rtm_pymmcore.utils import labels_to_particles


def store_img(img: np.array, metadata, path:str, folder: str):
    """Take the image and store it accordingly. Check the metadata for FOV index and timestamp."""
    img_type = metadata["img_type"]
    fname = metadata["fname"]
    tifffile.imwrite(
        os.path.join(path, folder, fname + ".tiff"),
        img,
        compression="zlib",
        compressionargs={"level": 5},
    )


# Create a new pipeline class that contains a segmentator and a stimulator
class ImageProcessingPipeline:
    def __init__(
        self,
        segmentator: base_segmentator.Segmentator,
        stimulator: base_stim.Stim,
        tracker: base_tracker.Tracker,
        storage_path: str,
        segmentation_channel: int = 0,
        
    ):
        self.segmentator = segmentator
        self.stimulator = stimulator
        self.tracker = tracker
        self.segmentation_channel = segmentation_channel
        self.storage_path = storage_path

    def run(self, img: np.ndarray, event: MDAEvent) -> dict:
        """
        Runs the image processing pipeline on the input image.

        Args:
            img (np.ndarray): The input image to process.
            event (MDAEvent): The MDAEvent used to capture the image, which also containins the metadata.

        Returns:
            dict: A dictionary containing the result of the pipeline.

        Pipeline Steps:
        1. Extract metadata from the event object.
        2. Segment the image using the segmentator.
        3. Extract features from the segmented image.
        4. Add frame-related information to the extracted features.
        5. Initialize (frame 0) or run the tracker.
        6. Remove duplicate tracks in the tracker.
        7. If stimulation is enabled, get the stimulated labels and mask.
        8. Store the intermediate tracks dataframe.
        9. Store the segmented images and labels.
        """

        # Rest of the code...

        metadata = event.metadata

        fov_obj: Fov = metadata["fov_object"]
        df_old = fov_obj.tracks  # get the previous table from the FOV-

        labels = self.segmentator.segment(img[self.segmentation_channel, :, :])
        if metadata["stim"] == True:
            stim_mask, labels_stim = self.stimulator.get_stim_mask(
                labels, metadata, img
            )
            fov_obj.stim_mask_queue.put_nowait(stim_mask)
            # TODO: Reenable, but make exception for stimwholeframe
            # mark in the df which cells have been stimulated
            # stim_index = np.where((df_tracked['frame']==metadata['timestep']) & (df_tracked['label'].isin(labels_stim)))[0]
            # df_tracked.loc[stim_index,'stim']=True

        # df_new, labels_rings = base_segmentator.extract_features(labels, img)
        df_new = pd.DataFrame([0])

        for key, value in metadata.items():
            if isinstance(value, (list, tuple)):
                df_new[key] = df_new.apply(lambda row: value, axis=1) 
            elif isinstance(value, dict):
                for subkey, subvalue in value.items():
                    df_new[subkey] = [subvalue] * len(df_new)
            else:
                df_new[key] = value

        NO_FEATURE_EXTRACTION = True
        if NO_FEATURE_EXTRACTION: 
            df_tracked = df_new
        else:
            df_tracked = self.tracker.track_cells(df_old, df_new, metadata)
        # store the tracks in the FOV queue
        fov_obj.tracks_queue.put(df_tracked)

        # after adding to queue, we have all the time to store the images and the tracks
        if metadata["stim"]:
            store_img(stim_mask, metadata, self.storage_path, "stim_mask")
        else:
            store_img(np.zeros_like(labels).astype(np.uint8), metadata, self.storage_path, "stim_mask")
            store_img(np.zeros_like(labels).astype(np.uint8), metadata, self.storage_path, "stim")

        if not df_tracked.empty:
            try:
                df_tracked = df_tracked.drop("fov_object", axis=1)
                df_tracked = df_tracked.drop("img_type", axis=1)
                # df_tracked = df_tracked.drop("channel", axis=1)
                df_tracked = df_tracked.drop("last_channel", axis=1)
            except KeyError:
                pass
        # df_datatypes = {
        #     "timestep": np.uint32,
        #     "particle": np.uint32,
        #     "label": np.uint32,
        #     "time": np.float32,
        #     "fov": np.uint16,
        #     "stim_exposure": np.float32,
        # }

        # try:
        #     df_tracked = df_tracked.astype(df_datatypes)
        # except ValueError as e:
        #     print(e)
        #     print("Error in converting datatypes. df_tracked:")
        #     print(df_tracked)

        df_tracked.to_parquet(
            os.path.join(self.storage_path, "tracks", f"{metadata['fname']}.parquet")
        )
        # particles = labels_to_particles(labels, df_tracked)
        # store_img(labels, metadata, self.storage_path, "labels")
        # # store_img(labels_rings, metadata, self.storage_path, "labels_rings")
        # store_img(particles, metadata, self.storage_path, "particles")

        # cleanup: delete the previous pickled tracks file
        if metadata["timestep"] > 0:
            fname_previous = f'{str(fov_obj.index).zfill(3)}_{str(metadata["timestep"]-1).zfill(5)}.parquet'
            os.remove(os.path.join(self.storage_path, "tracks", fname_previous))

        # TODO return something useful
        # TODO send tracks and stim_mask to the FOV queues

        return {"result": "STOP"}
