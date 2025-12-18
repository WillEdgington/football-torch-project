import torch

from typing import Dict, List, Any

from .match_dataset import MatchDataset

class Transform:
    def __call__(self, 
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        raise NotImplementedError("__call__ method for Transform must be implemented")
    
    def connect(self, 
                ds: MatchDataset):
        raise NotImplementedError("connect method for Transform must be implemented")
    
class Compose(Transform):
    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
    
    def __call__(self,
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        for t in self.transforms:
            sample = t(sample)
        return sample
    
    def connect(self, 
                ds: MatchDataset):
        for t in self.transforms:
            t.connect(ds)