import torch
from torch.nn import SiLU
from torch.nn import ReLU
from torch.nn import ReLU6
from torch.nn import Tanh
from torch.nn import Sigmoid

ACT2FN = {
    "relu": ReLU(),
    "silu": SiLU(),
}