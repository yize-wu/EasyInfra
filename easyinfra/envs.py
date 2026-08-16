import os
from typing import Callable, Any

DEFAULT_ATTENTION_BACKEND = "sdpa"
CONFIG_ROOT = os.getenv("CONFIG_ROOT", os.path.join(os.path.dirname(__file__), "../config"))

_environment_variables: dict[str, Callable[[], Any]] = {
    "ATTENTION_BACKEND": lambda: os.getenv("ATTENTION_BACKEND", DEFAULT_ATTENTION_BACKEND).lower(),
    "MOE_IMPL": lambda: os.getenv("MOE_IMPL", "fused").lower(),
    "CHUNK_ROUTING": lambda: os.getenv("CHUNK_ROUTING", "true").lower(),
    "NODE_CONFIG": lambda: os.getenv("NODE_CONFIG", os.path.join(CONFIG_ROOT, "node")),
    "EPLB_CONFIG": lambda: os.getenv("EPLB_CONFIG", os.path.join(CONFIG_ROOT, "moe")),
    "OUTER_ROOT_DIR": lambda: os.getenv("OUTER_ROOT_DIR", os.path.join(os.path.dirname(__file__), "../../")),
    "PRINT_COMM_TIME": lambda: os.getenv("PRINT_COMM_TIME", "0") != "0",
    "PRINT_MOE_COMPUTE_TIME": lambda: os.getenv("PRINT_MOE_COMPUTE_TIME", "0") != "0",
    "PRINT_EVENT_WAIT": lambda: os.getenv("PRINT_EVENT_WAIT", "0") != "0",
    "MOE_COMM_MODE": lambda: os.getenv("MOE_COMM_MODE", "all2all"),
    "CLASS_TIMING_MODE": lambda: os.getenv("CLASS_TIMING_MODE", "0"),
    "ENABLE_COMPUTE_COMM_OVERLAP": lambda: os.getenv("ENABLE_COMPUTE_COMM_OVERLAP", "y") == "y",
    "TOKEN_COUNT_DEBUG": lambda: os.getenv("TOKEN_COUNT_DEBUG", "n") == "y",
    "TRAJECTORY_DEBUG": lambda: os.getenv("TRAJECTORY_DEBUG", "n") == "y",
    "ENABLE_MOE_SCHEDULER": lambda: os.getenv("ENABLE_MOE_SCHEDULER", "") == "1", # false
    "ENABLE_MOE_PERMUTE_UNPERMUTE": lambda: os.getenv("ENABLE_MOE_PERMUTE_UNPERMUTE", "") == "1", # false
}

def __getattr__(name: str):
    if name in _environment_variables:
        return _environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")