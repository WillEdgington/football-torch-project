import numpy as np
import torch
import json

from typing import Dict, List, Tuple, Any
from torch.utils.data import Dataset
from pathlib import Path

from .tokeniser import Tokeniser
from .transforms import Transform
from .config import TENSORSDIR, TOKENISERDIR

def constructTokenContCols(featureCols: List[str], 
                           tokeniserDir: str=TOKENISERDIR) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    tokenCols = {
        "index": [],
        "size": [],
        "unkBucketSize": []
    }
    contCols = {
        "index": [],
    }
    sizeCache = {}

    for i, col in enumerate(featureCols):
        if not col.endswith("_token"):
            contCols["index"].append(i)
            continue
        tokenCols["index"].append(i)
        
        base = col.removesuffix("_token")
        if base.startswith("home_") or base.startswith("away_"):
            base = base.removeprefix("home_").removeprefix("away_")
        fileName = f"{base}_tokeniser.json"
        
        if base in sizeCache:
            tokenCols["size"].append(sizeCache[base][0])
            tokenCols["unkBucketSize"].append(sizeCache[base][1])
            del sizeCache[base]
            continue

        with Tokeniser(train=False, fileName=fileName, fileDir=tokeniserDir) as tkn:
            sizeCache[base] = (len(tkn.idtos), tkn.unkBuckets)
            tokenCols["size"].append(sizeCache[base][0])
            tokenCols["unkBucketSize"].append(sizeCache[base][1])
        
    return tokenCols, contCols

class MatchDataset(Dataset):
    def __init__(self, 
                 Xhome: torch.Tensor, 
                 Xaway: torch.Tensor, 
                 maskHome: torch.Tensor, 
                 maskAway: torch.Tensor, 
                 Y: torch.Tensor, 
                 featureCols: List[str], 
                 yCols: List[str],
                 tokeniserDir: str=TOKENISERDIR,
                 transform: Transform | None=None):
        self.Xhome = Xhome
        self.Xaway = Xaway
        self.maskHome = maskHome
        self.maskAway = maskAway
        self.Y = Y

        self.yCols = yCols
        self.featureCols = featureCols
        self.tokenCols, self.contCols = constructTokenContCols(featureCols=featureCols, tokeniserDir=tokeniserDir)
        
        self.transform = transform
        if transform is not None:
            self.transform.connect(ds=self)
    
    def __len__(self) -> int:
        return len(self.Y)
    
    def __getitem__(self, 
                    idx: int) -> Dict[str, torch.Tensor|Dict[str, Any]]:
        sample = {
            "home": self.Xhome[idx],
            "away": self.Xaway[idx],
            "mask_home": self.maskHome[idx],
            "mask_away": self.maskAway[idx],
            "y": self.Y[idx],
        }

        if self.transform is not None:
            sample = self.transform(sample)

        return sample 
    
    def save(self, 
             directory: str|None=None, 
             parentDir: str=TENSORSDIR, 
             fileDir: str="train"):
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
        }

        with open(path / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
    
    @staticmethod
    def load(transform: Transform|None=None, 
             directory: str|None=None, 
             parentDir: str=TENSORSDIR, 
             fileDir: str="train") -> Dataset|None:
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

            return MatchDataset(
                Xhome=Xhome, Xaway=Xaway,
                maskHome=maskHome, maskAway=maskAway,
                Y=Y,
                featureCols=featureCols,
                yCols=yCols,
                transform=transform
            )
        except FileNotFoundError:
            return None