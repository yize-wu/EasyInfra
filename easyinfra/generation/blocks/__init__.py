from easyinfra import envs
from easyinfra.generation.parallel.parallel_configuration import MoeCommMode

# decide the impl of moeblock
if envs.MOE_COMM_MODE == MoeCommMode.ALL2ALL.value:
    from .moe.moe_all2all import MoeAll2AllBlock as MoeBlock
elif envs.MOE_COMM_MODE == MoeCommMode.GATHER_REDUCE_SCATTER.value:
    from .moe.moe_gather_reduce import MoeGatherReduceBlock as MoeBlock
elif envs.MOE_COMM_MODE == MoeCommMode.GATHER_All2All.value:
    from .moe.moe_gather_all2all import MoeGatherAll2AllBlock as MoeBlock
else:
    raise ValueError(f"Unsupported moe communication mode {envs.MOE_COMM_MOE}. Set MOE_COMM_MOE.")

from .attention import AttentionBlock
from .mlp import MlpBlock