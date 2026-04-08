from enum import Enum


class DriveState(Enum):
    GB_TRACK = "GB_TRACK"
    FTGONLY = "FTGONLY"
    TRAILING = "TRAILING"
    OVERTAKE = "OVERTAKE"
