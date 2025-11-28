import json

from pathlib import Path
from typing import Dict, Any
from pandas import Series

from .config import NORMALISERDIR

class Normaliser:
    def __init__(self, eps: float=1e-8, train: bool=True, fileName: str|None=None, fileDir: str|None=NORMALISERDIR):
        self.eps = eps
        self.params = {}

        self.train = train
        self.fileName = fileName
        self.fileDir = fileDir

    def fit(self, series: Series, col: str|None=None):
        assert self.train, "normaliser must be in train mode to enable fitting abilities"
        assert str(series.dtype) in {"float64", "int64"}
        assert col or series.name, "col or series.name must be a string type"
        col = str(series.name) if col is None else col

        self.params[col] = {
            "mean": float(series.mean()),
            "std": max(self.eps, float(series.std())),
            "low": float(series.min()),
            "high": float(series.max())
        }
            
    def encodeSeries(self, series: Series, col: str|None=None, method: str="standard", fit: bool=True) -> Series:
        assert method in {"standard", "minmax"}, 'invalid method input, Choose between: "standard", "minmax"'
        if col is None:
            col = str(series.name)
        if self.train and fit:
            self.fit(series, col)

        p = self.params[col]
        return (series - p["mean"]) / p["std"] if method == "standard" else (series - p["low"]) / (p["high"] - p["low"])     
    
    def state_dict(self) -> Dict[str, Any]:
        return {
            "eps": self.eps,
            "params": self.params
        }
    
    def load_state_dict(self, state: Dict[str, Any]):
        self.eps = state["eps"]
        self.params = state["params"]

    def saveJson(self, fileName: str|None=None, fileDir: str|None=None):
        fileName = self.fileName if fileName is None else fileName
        fileDir = self.fileDir if fileDir is None else fileDir

        assert fileDir is not None, "Need a file directory. Set object.fileDir or pass directory name through method"
        assert fileName is not None, "Need a file Name. Set object.fileName or pass file name through method"
        assert fileName.endswith(".json"), f"file must be JSON. fileName: {fileName}"

        dirPath = Path(fileDir)
        dirPath.mkdir(parents=True, exist_ok=True)
        
        filePath = Path(fileDir) / fileName
        with open(filePath, "w") as f:
            json.dump(self.state_dict(), f, indent=2)

    def loadJson(self, fileName: str|None=None, fileDir: str|None=None):
        fileName = self.fileName if fileName is None else fileName
        fileDir = self.fileDir if fileDir is None else fileDir
        
        assert fileDir, "Need a file directory. Set object.fileDir or pass directory name through method"
        assert fileName, "Need a file Name. Set object.fileName or pass file name through method"
        assert fileName.endswith(".json"), f"file must be JSON. fileName: {fileName}"
        
        filePath = Path(fileDir) / fileName
        try:
            with open(filePath, "r") as f:
                state = json.load(f)
            self.load_state_dict(state=state)
        except FileNotFoundError:
            print(f"Could not load file from path: {filePath}")

    def freeze(self):
        self.train = False
    
    def __enter__(self):
        try:
            self.loadJson()
            return self
        except:
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saveJson()