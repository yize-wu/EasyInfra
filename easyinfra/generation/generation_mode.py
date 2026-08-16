from enum import Enum
class GenerationMode(Enum):
    SAMPLE = 0,
    ASSISTED_DECODING = 1,
    TREE_SAMPLE = 2,
    TREE_ASSISTED_DECODING = 3,
    