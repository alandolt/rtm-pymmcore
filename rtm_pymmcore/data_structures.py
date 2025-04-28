import queue
from dataclasses import dataclass
import enum
import numpy as np
import pandas as pd


class Fov:
    def __init__(self, index: int):
        self.index = index
        self.stim_mask = None
        self.light_mask = None
        self.stim_mask_queue = queue.SimpleQueue()
        self.tracks_queue = queue.SimpleQueue()
        self.tracks = None
        self.linker = None


@dataclass
class Channel:
    name: str
    exposure: int
    group: str = None
    power: int = None
    device_name: str = None
    property_name: str = None


@dataclass
class StimChannel:
    name: str
    group: str
    power: int = None
    device_name: str = None
    power_property_name: str = None


@dataclass
class StimTreatment:
    stim_channel_name: str
    stim_channel_group: str
    stim_timestep: tuple
    stim_exposure: int
    stim_power: int = None
    stim_channel_device_name: str = None
    stim_channel_power_property_name: str = None
    stim_property = None
    stim_property_value = None


class ImgType(enum.Enum):
    IMG_RAW = enum.auto()
    IMG_STIM = enum.auto()
    IMG_OPTOCHECK = enum.auto()
