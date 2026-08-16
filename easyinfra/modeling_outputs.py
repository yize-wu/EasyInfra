from typing import Optional, Tuple
import torch

class CausalLMOutputWithPast:
    def __init__(self, loss, logits, past_key_values, hidden_states, attentions) -> None:
        self.loss: Optional[torch.FloatTensor] = loss
        self.logits: torch.FloatTensor = logits
        self.past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = past_key_values
        self.hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = hidden_states
        self.attentions: Optional[Tuple[torch.FloatTensor, ...]] = attentions
