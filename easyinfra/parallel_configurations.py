from enum import Enum

class AttnParallelPolicy(Enum):
    REP = "replicate"
    TP = "tensor_parallel"

class ExpertParallelPolicy(Enum):
    REP = "replicate"
    TP = "tensor_parallel"
    EP = "expert_parallel"
    EASY = "easy"
