import torch
from torch import nn, Tensor
import numpy as np


class myMSE(nn.MSELoss):
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return super().forward(input, target) 
    

class myMaskedMSE(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = myMSE()

    def forward(self, input: Tensor, target: Tensor, mask=None) -> Tensor:
        if mask is None:
            return self.mse(input, target)
        
        # Apply the mask to both input and target
        masked_input = input * mask
        masked_target = target * mask
        
        # Calculate the loss only on the masked regions
        return self.mse(masked_input, masked_target)
