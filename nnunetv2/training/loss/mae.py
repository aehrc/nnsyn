import torch
from torch import nn, Tensor
import numpy as np


class myMAE(nn.L1Loss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target)
    

class myMaskedMAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = myMAE()

    def forward(self, input: Tensor, target: Tensor, mask=None) -> Tensor:
        if mask is None:
            return self.mae(input, target)
        
        # Apply the mask to both input and target
        masked_input = input * mask
        masked_target = target * mask
        
        # Calculate the loss only on the masked regions
        return self.mae(masked_input, masked_target)
