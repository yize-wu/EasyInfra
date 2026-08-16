from enum import Enum
class MoeStageName(Enum):
    ROUTING_DISPATCH = "routing_dispatch"    
    SHARED_EXPERT = "shared_expert"
    MOE = "moe"
    GATHER = "gather"

class AttentionStageName(Enum):
    ATTENTION = "attention"
