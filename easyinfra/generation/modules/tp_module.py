from torch import nn

class TPModule(nn.Module):
    _is_tp_module = True
    _support_safetensor_slice = True
    