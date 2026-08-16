import time
import warnings
from abc import ABC
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.nn import functional as F

# from transformers.pytorch_utils import isin_mps_friendly


class StoppingCriteria(ABC):
    def __call__(self, input_ids: torch.LongTensor, *args, **kwargs) -> torch.BoolTensor:
        raise NotImplementedError("StoppingCriteria needs to be subclassed")


class MaxLengthCriteria(StoppingCriteria):
    def __init__(self, max_length: int, max_position_embeddings: Optional[int] = None):
        self.max_length = max_length
        self.max_position_embeddings = max_position_embeddings

    def __call__(self, input_ids: torch.LongTensor, *args, **kwargs) -> torch.BoolTensor:
        cur_len = input_ids.shape[-1]
        is_done = cur_len >= self.max_length
        return torch.full((input_ids.shape[0],), is_done, device=input_ids.device, dtype=torch.bool)

class EosTokenCriteria(StoppingCriteria):
    def __init__(self, eos_token_id: Union[int, List[int], torch.Tensor]):
        if not isinstance(eos_token_id, torch.Tensor):
            if isinstance(eos_token_id, int):
                eos_token_id = [eos_token_id]
            eos_token_id = torch.tensor(eos_token_id)
        self.eos_token_id = eos_token_id

    def __call__(self, input_ids: torch.LongTensor, *args, **kwargs) -> torch.BoolTensor:
        self.eos_token_id = self.eos_token_id.to(input_ids.device)
        is_done = torch.isin(input_ids[:, -1], self.eos_token_id)
        # is_done = isin_mps_friendly(input_ids[:, -1], self.eos_token_id)
        return is_done


class ConfidenceCriteria(StoppingCriteria):
    def __init__(self, assistant_confidence_threshold):
        self.assistant_confidence_threshold = assistant_confidence_threshold

    def __call__(self, input_ids: torch.LongTensor, selected_token_probs: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        p = selected_token_probs[0].item()
        if p < self.assistant_confidence_threshold:
            return True
        return False


class StoppingCriteriaList(list):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.BoolTensor:
        is_done = torch.full((input_ids.shape[0],), False, device=input_ids.device, dtype=torch.bool)
        for criteria in self:
            is_done = is_done | criteria(input_ids, scores, **kwargs)
        return is_done

    @property
    def max_length(self) -> Optional[int]:
        for stopping_criterium in self:
            if isinstance(stopping_criterium, MaxLengthCriteria):
                return stopping_criterium.max_length
        return None


def validate_stopping_criteria(stopping_criteria: StoppingCriteriaList, max_length: int) -> StoppingCriteriaList:
    stopping_max_length = stopping_criteria.max_length
    new_stopping_criteria = deepcopy(stopping_criteria)
    if stopping_max_length is not None and stopping_max_length != max_length:
        warnings.warn("You set different `max_length` for stopping criteria and `max_length` parameter", UserWarning)
    elif stopping_max_length is None:
        new_stopping_criteria.append(MaxLengthCriteria(max_length=max_length))
    return new_stopping_criteria
