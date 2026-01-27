import json
import time

from pathlib import Path
from typing import Dict, Any

from .config import EXPERIMENTDIR

class Trial:
    def __init__(self, path: Path):
        self.path = path
        self.definitionPath = path / "definition.json"
        self.statePath = path / "state.json"
        self.metricsPath = path / "metrics.pt"

        self.definition: Dict[str, Any]|None = None
        self.state: Dict[str, Any]|None = None

    def isComplete(self) -> bool:
        if self.state is None:
            self.loadState()
        return self.state["status"] == "completed"

    @classmethod
    def create(self,
               definition: Dict[str, Any],
               root: Path|str=EXPERIMENTDIR) -> "Trial":
        if not isinstance(root, Path):
            root = Path(root)
        createTime = time.time()
        trialID = f"trial_{int(createTime * 1e6)}"
        path = root / trialID
        path.mkdir(parents=True, exist_ok=False)

        with open(path / "definition.json", "w") as f:
            json.dump(definition, f, indent=2)
        
        state = {
            "status": "created",
            "current_epoch": 0,
            "max_epoch": definition.get("training", {}).get("epochs"),
            "created_at": createTime,
            "updated_at": createTime
        }

        with open(path / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        trial = self(path=path)
        trial.definition = definition
        trial.state = state        
        return trial

    @classmethod
    def load(self, 
             path: Path) -> "Trial":
        if not (path / "definition.json").exists():
            raise FileNotFoundError("Missing definition.json")
        
        trial = self(path=path)

        with open(trial.definitionPath) as f:
            trial.definition = json.load(f)
        
        trial.loadState()
        return trial
    
    def loadState(self):
        with open(self.statePath) as f:
            self.state = json.load(f)
    
    def saveState(self):
        if self.state is None:
            raise RuntimeError("Trial state not loaded")
        self.state["updated_at"] = time.time()
        with open(self.statePath, "w") as f:
            json.dump(self.state, f, indent=2)