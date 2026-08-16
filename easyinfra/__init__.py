from . import envs
import torch
import os
from glob import glob
from . import envs
### import .so in the parent directory
if envs.MOE_IMPL in ("cutlass", "aten"):
    # lib_path = os.getenv("EASY_LIB", os.path.join(os.path.dirname(__file__), "../*.so"))
    HOME_PATH = os.getenv("HOME")
    lib_path_template = os.getenv("LIB_DIR_PATH", os.path.join(HOME_PATH, "lib/*.so"))
    libs = glob(lib_path_template)
    for file in libs:
        torch.ops.load_library(file)

    if envs.MOE_IMPL == "cutlass":
        if not hasattr(torch.ops.ops, "cutlass_grouped_gemm_single"):
            raise ValueError(f"cutlass MoE implementation is not installed. See install.sh.")
    if envs.MOE_IMPL == "aten":
        if not hasattr(torch.ops.ops, "aten_grouped_gemm_single"):
            raise ValueError(f"aten MoE implementation is not installed. See install.sh.")
            