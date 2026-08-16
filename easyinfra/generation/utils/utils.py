import torch
from collections import OrderedDict, UserDict
from typing import Tuple, Any, Optional
from dataclasses import dataclass

from ..cache_utils import TreeDynamicCache

class ModelOutput(OrderedDict):
    def __getitem__(self, k):
        if isinstance(k, str):
            inner_dict = dict(self.items())
            return inner_dict[k]
        else:
            return self.to_tuple()[k]
    def to_tuple(self) -> Tuple[Any]:
        """
        Convert self to a tuple containing all the attributes/keys that are not `None`.
        """
        return tuple(self[k] for k in self.keys())

@dataclass    
class GenerateDecoderOnlyTreeOutput(ModelOutput):
    tree_draft_tokens: torch.LongTensor = None
    retrieve_indices: torch.LongTensor = None
    scores: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[TreeDynamicCache] = None

def get_default_dtype():
    return torch.get_default_dtype()