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
        assert 0 < intensity <= 1, "intensity must be in the range (0, 1]"
        self.prob = prob
        self.intensity = intensity
        self.featGroups = []

    def connect(self,
                ds: Dataset):
        
        grouped = {}

        for i, unkBuckets, vocabSize in zip(ds.tokenCols["index"], ds.tokenCols["unkBucketSize"], ds.tokenCols["size"]):
            assert unkBuckets > 0, "UNK augmentation requires UNK buckets"
            feature = ds.featureCols[i]
            base = feature.removeprefix("home_").removeprefix("away_")

            if base not in grouped:
                grouped[base] = {
                    "idx": [i],
                    "unkRange": (1, unkBuckets + 2), # assuming <pad> = 0, <unk_0> = 1
                    "size": vocabSize
                }
            else:
                grouped[base]["idx"].append(i)
        
        self.featGroups = [
            {
                "idx": v["idx"],
                "unkRange": v["unkRange"],
                "size": v["size"]
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

        device = xh.device

        for fg in self.featGroups:
            indexes = fg["idx"]
            unklo, unkhi = fg["unkRange"]

            vals = []
            for i in indexes:
                if mh.any():
                    vals.append(xh[mh, i])
                if ma.any():
                    vals.append(xa[ma, i])

            if not vals:
                continue

            entities = torch.unique(torch.cat(vals))
            entities = entities[entities != 0] # remove <pad> entities

            if entities.numel() == 0:
                continue

            mutMask = torch.rand(
                entities.numel(), device=device
            ) <= self.intensity

            if not mutMask.any():
                continue

            mutEntities = entities[mutMask].long()

            unkIds = torch.randint(
                unklo, unkhi,
                (mutEntities.numel(),),
                device=device
            )

            lut = torch.arange(
                fg["size"], device=device
            )

            lut[mutEntities] = unkIds

            for i in indexes:
                xh[:, i] = lut[xh[:, i].long()]
                xa[:, i] = lut[xa[:, i].long()]

        sample["home"] = xh
        sample["away"] = xa
        return sample
    
class TemporalDropout(Transform):
    def __init__(self,
                 prob: float=0.2, 
                 minKeep: int = 1):
        assert 0 < prob <= 1, "prob must be in the range (0, 1]"
        assert minKeep > 0, "minKeep must be a positive integer"
        self.prob = prob
        self.maxDrop = 0
        self.minKeep = minKeep

    def connect(self,
                ds: Dataset):
        self.maxDrop = ds.seqLen - self.minKeep

    def __call__(self, 
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        if random.random() > self.prob:
            return sample
        
        for side in ("home", "away"):
            drop = int(random.random() ** 2 * self.maxDrop)
            sample[f"mask_{side}"][:drop] = 0
        
        return sample
    
class MissingValueAugment(Transform):
    def __init__(self,
                 prob: float=0.2,
                 intensity: float=0.2):
        assert 0 < prob <= 1, "prob must be in the range (0, 1]"
        assert 0 < intensity <= 1, "intensity must be in the range (0, 1]"
        self.prob = prob
        self.intensity = intensity

    def connect(self,
                ds: Dataset):
        return

    def __call__(self,
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        if random.random() > self.prob:
            return sample
        
        for side in ("home", "away"):
            missing = sample[f"missing_{side}"]
            corruption = torch.rand_like(missing.float()) < self.intensity
            missing |= corruption
        
        return sample
    
class ContinuousFeatureDropout(Transform):
    def __init__(self,
                 prob: float=0.2,
                 intensity: float=0.2):
        assert 0 < prob <= 1, "prob must be in the range (0, 1]"
        assert 0 < intensity <= 1, "intensity must be in the range (0, 1]"
        self.prob = prob
        self.intensity = intensity

    def connect(self,
                ds: Dataset):
        return

    def __call__(self,
                 sample: Dict[str, torch.Tensor|Dict[str, Any]]) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        if random.random() > self.prob:
            return sample
        
        for side in ("home", "away"):
            missing = sample[f"missing_{side}"]
            F = missing.shape[1]
            drop = torch.rand(F, device=missing.device) <= self.intensity
            missing[:, drop] = 1
        return sample