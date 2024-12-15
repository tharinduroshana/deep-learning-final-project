from enum import Enum

class Mode(Enum):
    SINGLE = 1
    DUAL = 2
    AUTUMN = 3
    WINTER = 4

class Action(Enum):
    FINE_TUNE = 1
    ENSEMBLE = 2

class ModelType(Enum):
    DENSENET121 = 1
    EFFICIENTNET_B0 = 2
    RESNET34 = 3
