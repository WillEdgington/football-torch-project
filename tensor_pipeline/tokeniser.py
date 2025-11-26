import json

from pathlib import Path
from pandas import Series
from typing import Any

from .config import TOKENISERDIR

class Tokeniser:
    def __init__(self, train: bool=True, unkBuckets: int=32, fileName: str|None=None, fileDir: str|None=TOKENISERDIR):
        self.unkBuckets = unkBuckets
        assert self.unkBuckets > 0, "number of <unk> buckets (unkBuckets) must be atleast 1."
        
        self.stoid = {"<pad>": 0}
        self.idtos = ["<pad>"]
        
        for i in range(unkBuckets):
            self.stoid[f"<unk_{i}>"] = 1 + i
            self.idtos.append(f"<unk_{i}>")

        self.train = train
        self.unkCount = 0
        self.fileName = fileName
        self.fileDir = fileDir
    
    def getId(self, token: str) -> int:
        if token in self.stoid:
            return self.stoid[token]
        if self.train:
            newID = len(self.idtos)
            self.stoid[token] = newID
            self.idtos.append(token)
            return newID
        self.unkCount += 1
        key = (hash(token) % self.unkBuckets)
        return self.stoid[f"<unk_{key}>"]
    
    def encodeSeries(self, series: Series) -> Series:
        return series.apply(self.getId).astype("int32")
    
    def state_dict(self):
        return {
            "unkBuckets": self.unkBuckets,
            "stoid": self.stoid,
            "idtos": self.idtos
        }
    
    def load_state_dict(self, state):
        self.unkBuckets = state["unkBuckets"]
        self.stoid = state["stoid"]
        self.idtos = state["idtos"]

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
            self.loadJson(fileName=self.fileName, fileDir=self.fileDir)
            return self
        except:
            return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.saveJson(fileName=self.fileName, fileDir=self.fileDir)
