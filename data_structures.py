from typing import NamedTuple

import torch


class ModelOutput(NamedTuple):
    policy: torch.Tensor
    value: torch.Tensor
