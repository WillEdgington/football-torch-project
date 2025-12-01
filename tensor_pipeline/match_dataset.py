import numpy as np
import torch

from typing import Tuple
from torch.utils.data import Dataset

class MatchDataset(Dataset):
    def __init__(self, Xhome: torch.Tensor, Xaway: torch.Tensor, maskHome: torch.Tensor, 
                 maskAway: torch.Tensor, Y: torch.Tensor):
        self.Xhome = Xhome
        self.Xaway = Xaway
        self.maskHome = maskHome
        self.maskAway = maskAway
        self.Y = Y
    
    def __len__(self) -> int:
        return len(self.Y)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.Xhome[idx],
            self.Xaway[idx],
            self.maskHome[idx],
            self.maskAway[idx],
            self.Y[idx]
        )