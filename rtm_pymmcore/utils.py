from typing import TypedDict
import numpy as np
import os
from skimage.util import map_array
import lzma
import pandas as pd


def create_folders(path, folders):
    """Create all folders if they don't already exist.

    Keyword arguments:
    path -- location of main folder
    folders -- list of all subfolders
    """

    for folder in folders:
        dir_name = os.path.join(path, folder)
        try:
            os.makedirs(dir_name)
            print("Directory", dir_name, "created ")
        except FileExistsError:
            print("Directory", dir_name, "already exists")


def labels_to_particles(labels, tracks):
    """Takes in a segmentation mask with labels and replaces them with track IDs that are consistent over time."""
    # For every frame
    # labels_stack = np.array(labels_stack)
    particles = np.zeros_like(labels)
    tracks_f = tracks[(tracks["timestep"] == tracks.timestep.max())]
    # particle_f = np.zeros((1024,1024))
    from_label = tracks_f["label"].values
    to_particle = tracks_f["particle"].values
    particles = map_array(labels, from_label, to_particle, out=particles)
    return particles
