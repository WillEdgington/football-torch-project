import json
import time

from pathlib import Path
from typing import Dict, Any

class Trial:
    def __init__(self,
                 path: Path):
        self.path = path
        self.modelPath = path / "models"
        self.modelPath.mkdir(parents=True, exist_ok=True)

        self.evalsPath = path / "evals"
        self.evalsPath.mkdir(parents=True, exist_ok=True)

        self.definitionPath = path / "definition.json"
        self.statePath = path / "state.json"
        self.metricsPath = path / "metrics.pt"

        self._definition: Dict[str, Any]|None = None
        self.state: Dict[str, Any]|None = None

    def isComplete(self) -> bool:
        return self.getState()["status"] == "completed"

    @classmethod
    def create(self,
               definition: Dict[str, Any],
               root: Path|str="saved_models") -> "Trial":
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
            "epochs_completed": 0,
            "max_epoch": definition.get("train", {}).get("epochs"),
            "created_at": createTime,
            "updated_at": createTime
        }

        with open(path / "state.json", "w") as f:
            json.dump(state, f, indent=2)

        trial = self(path=path)
        trial._definition = definition
        trial.state = state        
        return trial

    @classmethod
    def load(self, 
             path: Path) -> "Trial":
        if not (path / "definition.json").exists():
            raise FileNotFoundError("Missing definition.json")
        
        trial = self(path=path)

        trial.getDefinition()        
        trial.getState()
        return trial
    
    def getState(self) -> Dict[str, Any]:
        if self.state is None:
            if not self.statePath.exists():
                raise FileNotFoundError(f"Missing state in path: {self.statePath}")
            with open(self.statePath) as f:
                self.state = json.load(f)
        return self.state
    
    def getDefinition(self) -> Dict[str, Any]:
        if self._definition is None:
            if not self.definitionPath.exists():
                raise FileNotFoundError(f"Missing definition in path: {self.definitionPath}")
            with open(self.definitionPath) as f:
                self._definition = json.load(f)
        return self._definition

    def saveState(self):
        if self.state is None:
            raise RuntimeError("Trial state not loaded, load state before saving")
        self.state["updated_at"] = time.time()
        with open(self.statePath, "w") as f:
            json.dump(self.state, f, indent=2)