import torch
import json

from pathlib import Path
from typing import Dict, List, Any, Set

from .config import TENSORSDIR

class SampleStore:
    def __init__(self,
                 rootDir: str=TENSORSDIR,
                 shardSize: int=1024,
                 metadata: Dict[str, Any]|None=None,
                 device: str="cpu"):
        self.rootDir = Path(rootDir)
        self.shardSize = shardSize
        self.device = device

        self.shardDir = self.rootDir / "shards"
        self.shardDir.mkdir(parents=True, exist_ok=True)

        self.indexPath = self.rootDir / "index.json"
        self.statePath = self.rootDir / "store_state.json"

        self.metaPath = self.rootDir / "metadata.json"
        self.metadata = metadata

        self.indices: Dict[str, Dict[str, List[int]]] = {}
        self._shardCache: Dict[int, Dict[str, torch.Tensor]] = {}
        self._sampleKeys: Set[str]|None = None
        self.numSamples = 0

        self._activeShardId = None
        self._activeShardCount = 0
        self._activeShard: Dict[str, List[torch.Tensor]]|None = None

        if self.statePath.exists():
            self.load()
        assert self.metadata is not None, "metadata must not be None if no previous store for metadata found"

    def _shardId(self,
                 idx: int) -> int:
        return idx // self.shardSize
    
    def _shardPath(self,
                   shardId: int) -> Path:
        return self.shardDir / f"shard_{shardId:05d}.pt"
    
    def _resumeActiveShard(self):
        if self.numSamples == 0:
            return

        lastShardId = (self.numSamples - 1) // self.shardSize
        countInShard = self.numSamples % self.shardSize

        if countInShard == 0:
            return

        shardPath = self._shardPath(lastShardId)
        shard = torch.load(shardPath)

        if self._sampleKeys is not None:
            assert set(shard.keys()) == self._sampleKeys

        self._activeShardId = lastShardId
        self._activeShard = shard
        self._activeShardCount = countInShard

    def _startNewShard(self,
                       shardId: int,
                       sample: Dict[str, Any]):
        self._activeShardId = shardId
        self._activeShardCount = 0
        self._activeShard = {k: [] for k in sample.keys()}

        if self._sampleKeys is None:
            self._sampleKeys = set(sample.keys())
        else:
            assert set(sample.keys()) == self._sampleKeys

    def _flushActiveShard(self):
        if self._activeShard is None:
            return
        
        path = self._shardPath(shardId=self._activeShardId)
        torch.save(self._activeShard, path)

        self._activeShard = None
        self._activeShardId = None
        self._activeShardCount = 0

    def addToSplit(self,
                   split: str,
                   group: str,
                   indices: List[int]):
        self.indices.setdefault(split, {})
        self.indices[split].setdefault(group, [])
        self.indices[split][group].extend(indices)

    def finalise(self):
        self._flushActiveShard()
        self.save()

    def append(self, 
               sample: Dict[str, torch.Tensor]) -> int:
        idx = self.numSamples
        shardId = self._shardId(idx)
        
        if self._activeShardId is None:
            self._startNewShard(shardId, sample)

        if shardId != self._activeShardId:
            self._flushActiveShard()
            self._startNewShard(shardId, sample)
        
        for k, v in sample.items():
            self._activeShard[k].append(v.cpu())
        
        self._activeShardCount += 1
        self.numSamples += 1
        return idx
    
    def store(self,
              split: str,
              group: str,
              samples: Dict[str, torch.Tensor]|List[Dict[str, torch.Tensor]]):
        print(f"split: {split}, group: {group}, number of samples: {len(samples)}")
        if not isinstance(samples, list):
            samples = [samples]
        indices = [self.append(sample=sample) for sample in samples]
        self.addToSplit(split=split, group=group, indices=indices)

    def _loadShard(self,
                   shardId: int) -> Dict[str, Any]:
        return torch.load(self._shardPath(shardId=shardId), map_location=self.device)

    def get(self,
            idx: int):
        assert idx < self.numSamples and idx >= 0, f"index ({idx}) out of range for samples stored ({self.numSamples})"
        shardId = self._shardId(idx)
        offset = idx % self.shardSize

        if shardId not in self._shardCache:
            self._shardCache[shardId] = self._loadShard(shardId=shardId)
        
        shard = self._shardCache[shardId]

        return {
            key: val[offset]
            for key, val in shard.items()
        }
    
    def get_many(self, indices: List[int]) -> List[Dict[str, Any]]:
        return [self.get(i) for i in indices]

    def save(self):
        with open(self.indexPath, "w") as f:
            json.dump(self.indices, f, indent=2)

        if self.metadata is not None:
            with open(self.metaPath, "w") as f:
                json.dump(self.metadata, f, indent=2)

        with open(self.statePath, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    def load(self):
        if self.statePath.exists():
            with open(self.statePath) as f:
                state = json.load(f)
                self.load_state_dict(state)
        
        if self.metaPath.exists():
            with open(self.metaPath) as f:
                self.metadata = json.load(f)

        if self.indexPath.exists():
            with open(self.indexPath) as f:
                self.indices = json.load(f)

        self._resumeActiveShard()
    
    def state_dict(self):
        return {
            "shard_size": self.shardSize,
            "num_samples": self.numSamples,
            "sample_keys": list(self._sampleKeys),
        }

    def load_state_dict(self,
                        state: Dict[str, Any]):
        self.shardSize = state["shard_size"]
        self.numSamples = state["num_samples"]
        self._sampleKeys = set(state["sample_keys"])