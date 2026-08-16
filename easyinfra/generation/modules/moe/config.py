import easyinfra.envs as envs


def get_moe_impl():
    if envs.MOE_IMPL == "fused":
        from .fused_moe import FusedMoE
        return FusedMoE
    elif envs.MOE_IMPL in ("cutlass", "aten"):
        from .single_group_gemm_moe import SingleGroupGEMMMoE
        return SingleGroupGEMMMoE
    else:
        if envs.MOE_IMPL != "native":
            raise ValueError(f"Unrecognized MoE Implementation {envs.MOE_IMPL}")
        from .native_moe import NativeMoE
        return NativeMoE

def is_chunk_routing():
    return envs.CHUNK_ROUTING == "true"
