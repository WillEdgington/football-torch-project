import torch

from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset
from pathlib import Path

from .transforms import Transform
from .sample_store import SampleStore

class MatchDataset(Dataset):
    def __init__(self,
                 store: SampleStore,
                 seqLen: int,
                 transform: Transform|None=None,
                 split: List[str]|str|None=None,
                 group: List[str]|str|None=None):
        self.store = store

        self._getMetaFromStore()
        assert seqLen <= self.maxSeqLen, f"seqLen ({seqLen}) must be less than or equal to maxSeqLen ({self.maxSeqLen})"
        self.seqLen = seqLen
        self._startT = self.maxSeqLen - self.seqLen

        self._getIndicesFromStore(split=split, group=group)

        self.transform = transform
        if transform is not None:
            self.transform.connect(ds=self)

    def __len__(self) -> int:
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.store.get(self.indices[idx])
        sample = {
            "home": sample["home"][self._startT:],
            "away": sample["away"][self._startT:],
            "mask_home": sample["mask_home"][self._startT:],
            "mask_away": sample["mask_away"][self._startT:],
            "missing_home": sample["missing_home"][self._startT:] if self.missingCols is not None else None,
            "missing_away": sample["missing_away"][self._startT:] if self.missingCols is not None else None,
            "y": sample["y"],
        }

        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def _getMetaFromStore(self):
        meta = self.store.metadata
        self.featureCols = meta["featureCols"]
        self.yCols = meta["yCols"]
        self.tokenCols = meta["tokenCols"]
        self.contCols = meta["contCols"]
        self.missingCols = meta.get("missingCols")
        self.maxSeqLen = meta["maxSeqLen"]

    def _getIndicesFromStore(self,
                             split: List[str]|str|None=None,
                             group: List[str]|str|None=None):
        indices = self.store.indices
        self.indices = []

        if split is None:
            split = list(indices.keys())
        self.split = split if isinstance(split, list) else [split]
        
        if group is None:
            group = list(indices[self.split[0]].keys())
        self.group = group if isinstance(group, list) else [group]

        for skey in self.split:
            assert skey in indices, f'"{skey}" is not a valid split in store'
            splitIndices = indices[skey]
            for gkey in self.group:
                assert gkey in splitIndices, f'"{gkey}" is not a valid group in store for split "{skey}"'
                self.indices.extend(splitIndices[gkey])