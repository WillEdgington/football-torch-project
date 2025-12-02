import numpy as np
import torch
import json

from typing import Dict, List
from torch.utils.data import Dataset
from pathlib import Path

from .config import UNKBUCKETDICT, TENSORSDIR

class MatchDataset(Dataset):
    def __init__(self, Xhome: torch.Tensor, Xaway: torch.Tensor, maskHome: torch.Tensor, 
                 maskAway: torch.Tensor, Y: torch.Tensor, featureCols: List[str], yCols: List[str],
                 unkBucketDict: Dict[str, int]=UNKBUCKETDICT):
        self.Xhome = Xhome
        self.Xaway = Xaway
        self.maskHome = maskHome
        self.maskAway = maskAway
        self.Y = Y

        self.yCols = yCols
        self.featureCols = featureCols

        self.tokenCols = [i for i, c in enumerate(featureCols) if c.endswith("_token")]
        self.contCols  = [i for i, c in enumerate(featureCols) if not c.endswith("_token")]

        self.unkBucketSizes = {
            i: unkBucketDict[featureCols[i].replace("_token", "").replace("home_", "").replace("away_", "")]
            for i in self.tokenCols
        }
    
    def __len__(self) -> int:
        return len(self.Y)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor|List[int]]:
        return {
            "home": self.Xhome[idx],
            "away": self.Xaway[idx],
            "mask_home": self.maskHome[idx],
            "mask_away": self.maskAway[idx],
            "y": self.Y[idx],
            "token_cols": self.tokenCols,
            "cont_cols": self.contCols,
        }
    
    def save(self, directory: str|None=None, parentDir: str=TENSORSDIR, fileDir: str="train"):
        path = Path(directory) if directory else Path(parentDir) / fileDir
        path.mkdir(parents=True, exist_ok=True)
        

        torch.save(self.Xhome, path / "Xhome.pt")
        torch.save(self.Xaway, path / "Xaway.pt")
        torch.save(self.maskHome, path / "maskHome.pt")
        torch.save(self.maskAway, path / "maskAway.pt")
        torch.save(self.Y, path / "Y.pt")

        meta = {
            "featureCols": self.featureCols,
            "yCols": self.yCols,
            "tokenCols": self.tokenCols,
            "contCols": self.contCols,
            "unkBucketSizes": self.unkBucketSizes
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @staticmethod
    def load(directory: str|None=None, parentDir: str=TENSORSDIR, fileDir: str="train") -> Dataset|None:
        path = Path(directory) if directory else Path(parentDir) / fileDir
        try:
            Xhome = torch.load(path / "Xhome.pt")
            Xaway = torch.load(path / "Xaway.pt")
            maskHome = torch.load(path / "maskHome.pt")
            maskAway = torch.load(path / "maskAway.pt")
            Y = torch.load(path / "Y.pt")

            with open(path / "metadata.json", "r") as f:
                meta = json.load(f)
            
            featureCols = meta["featureCols"]
            yCols = meta["yCols"]
            unkBucketSizes = {int(k): v for k, v in meta["unkBucketSizes"].items()}

            unkBucketDict = {
                featureCols[int(idx)].replace("_token", "").replace("home_", "").replace("away_", ""): size
                for idx, size in unkBucketSizes.items()
            }

            return MatchDataset(
                Xhome=Xhome, Xaway=Xaway,
                maskHome=maskHome, maskAway=maskAway,
                Y=Y,
                featureCols=featureCols,
                yCols=yCols,
                unkBucketDict=unkBucketDict,
            )
        except FileNotFoundError:
            return None