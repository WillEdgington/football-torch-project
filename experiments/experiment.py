import hashlib
import json
import torch
import numpy as np

from pathlib import Path

from typing import Dict, List, Any, Tuple, Set

from .trial_scheduler import TrialScheduler
from .config import TrainFn, ConstructorFn, EvaluatorFn
from .trial import Trial
from .trainer import Trainer
from .evaluator import Evaluator

def toJSONSafe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: toJSONSafe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [toJSONSafe(v) for v in obj]
    elif isinstance(obj, tuple):
        return [toJSONSafe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def hashDefinition(definition: Dict[str, Any]) -> str:
    blob = json.dumps(definition, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()

class Experiment:
    def __init__(self,
                 root: Path|str,
                 scheduler: TrialScheduler,
                 train: TrainFn,
                 constructer: ConstructorFn,
                 evaluator: Evaluator|None=None):
        if not isinstance(root, Path):
            root = Path(root)
            
        self.root = root
        self._trialsDir: Path = root / "trials"
        self._trialsDir.mkdir(parents=True, exist_ok=True)

        self.scheduler = scheduler
        self.train = train
        self.constructor = constructer

        self.evaluator = evaluator
        self.evalHash = hashDefinition(self.evaluator.evalDefinition)

        self._trialsJSON: Path = self.root / "trials.json"
        self.trials: List[Dict[str, Any]] = self._loadTrials()
        self._definitionHashes: Set[str] = self._loadDefinitionHashes()

    def _loadTrials(self) -> List[Dict[str, Any]]:
        if not self._trialsJSON.exists():
            return []
        
        with open(self._trialsJSON, "r") as f:
            trials = json.load(f)
        
        return trials

    def _saveTrials(self):
        json.dumps(self.trials)
        with open(self._trialsJSON, "w") as f:
            json.dump(self.trials, f, indent=2)

    def _loadDefinitionHashes(self):
        defHashes = set()
        for t in self.trials:
            defHashes.add(t["definition_hash"])
        return defHashes
    
    def _createNewDefinition(self) -> Tuple[Dict[str, Any], str]|Tuple[None, None]:
        while True:
            definition = self.scheduler.next()
            if definition is None:
                return None, None
            defHash = hashDefinition(definition=definition)
            if defHash not in self._definitionHashes:
                return definition, defHash

    def _createNextTrial(self) -> Tuple[Trial, int]|Tuple[None, None]:
        definition, defHash = self._createNewDefinition()
        if definition is None:
            return None, None
        
        trial = Trial.create(definition=definition, root=self._trialsDir)
        idx = len(self.trials)
        trialState = {
            "id": idx,
            "path": str(trial.path),
            "definition_hash": defHash,
            "evals": {}
        }
        self.trials.append(trialState)
        self._saveTrials()
        self._definitionHashes.add(defHash)
        return trial, idx

    def _prepareTrial(self) -> Tuple[Trial, int]|Tuple[None, None]:
        for i, t in enumerate(self.trials):
            trial = Trial.load(Path(t["path"]))
            if not trial.isComplete():
                return trial, i
        return self._createNextTrial()
    
    def evalTrial(self,
                  trial: Trial) -> Dict[str, Any]|None:
        if self.evaluator is None:
            return None
        evals = self.evaluator.run(trial)
        evals["eval_hash"] = self.evalHash
        return evals
        
    def eval(self, 
             overwrite: bool=False,
             save: bool=True):
        evalList = []
        for i, t in enumerate(self.trials):
            if not overwrite and t.get("evals"):
                evalList.append(t["evals"])
                continue
            trial = Trial.load(path=Path(t["path"]))
            if not trial.isComplete():
                continue
            trialEvals = self.evalTrial(trial=trial)
            self.trials[i]["evals"] = toJSONSafe(trialEvals)
            evalList.append(trialEvals)
            if save:
                self._saveTrials()
        
        return evalList

    def run(self):
        while True:
            trial, i = self._prepareTrial()
            if trial is None:
                break
            trainer = Trainer(trial=trial,
                              train=self.train,
                              constructor=self.constructor)
            trainer.run()
            self.trials[i]["evals"] = toJSONSafe(self.evalTrial(trial=trial))
            self._saveTrials()

        print(f"All trials complete.\nTotal trials: {len(self.trials)}")