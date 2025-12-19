import torch
import random

from typing import Dict, List, Any
from torch.utils.data import Dataset

class Transform:
    def __call__(self, 
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        raise NotImplementedError("__call__ method for Transform must be implemented")
    
    def connect(self, 
                ds: Dataset):
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
                ds: Dataset):
        for t in self.transforms:
            t.connect(ds)

class RandomTokenUNK(Transform):
    def __init__(self,
                 prob: float=0.2,
                 intensity: float=0.1):
        assert 0 < prob <= 1, "prob must be in the range (0, 1]"
        assert 0 < intensity <= 1, "alpha must be in the range (0, 1]"
        self.prob = prob
        self.intensity = intensity
        self.featGroups = []

    def connect(self,
                ds: Dataset):
        
        grouped = {}

        for i, unkBuckets in zip(ds.tokenCols["index"], ds.tokenCols["unkBucketSize"]):
            assert unkBuckets > 0, "UNK augmentation requires UNK buckets"
            feature = ds.featureCols[i]
            base = feature.removeprefix("home_").removeprefix("away_")

            if base not in grouped:
                grouped[base] = {
                    "idx": [i],
                    "unkRange": (1, unkBuckets + 2) # assuming <pad> = 0, <unk_0> = 1
                }
            else:
                grouped[base]["idx"].append(i)
        
        self.featGroups = [
            {
                "idx": v["idx"],
                "unkRange": v["unkRange"]
            }
            for v in grouped.values()
        ]

    def __call__(self, 
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        if random.random() > self.prob:
            return sample
        
        xh = sample["home"].clone()
        xa = sample["away"].clone()
        mh = sample["mask_home"].bool()
        ma = sample["mask_away"].bool()

        for fg in self.featGroups:
            entities = []
            indexes = fg["idx"]
            for i in indexes:
                entities.append(xh[mh, i])
                entities.append(xa[ma, i])
            
            if len(entities) == 0:
                continue

            entities = torch.unique(torch.cat(entities))
            entities = entities[entities != 0]

            if len(entities) == 0:
                continue

            maskent = torch.rand(len(entities), device=entities.device) <= self.intensity
            unkRange = fg["unkRange"]

            for entity in entities[maskent].tolist():
                unkID = random.randrange(unkRange[0], unkRange[1])

                for i in indexes:
                    maskh = (xh[:, i] == entity)
                    xh[maskh, i] = unkID
                    maska = (xa[:, i] == entity)
                    xa[maska, i] = unkID
            
        sample["home"] = xh
        sample["away"] = xa
        return sample