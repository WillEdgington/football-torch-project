import hashlib
import json

from pathlib import Path

from typing import Dict, List, Any, Tuple

from .trial_scheduler import TrialScheduler
from .config import TrainFn, ConstructorFn, EvaluatorFn
from .trial import Trial
from .trainer import Trainer

def hash_definition(definition: Dict[str, Any]) -> str:
    blob = json.dumps(definition, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()

class Experiment:
    def __init__(self,
                 root: Path,
                 scheduler: TrialScheduler,
                 train: TrainFn,
                 constructer: ConstructorFn,
                 evaluator: EvaluatorFn|None=None):
        self.root = root
        self._trialsDir: Path = root / "trials"
        self._trialsDir.mkdir(parents=True, exist_ok=True)

        self.scheduler = scheduler
        self.train = train
        self.constructor = constructer
        self.evaluator = evaluator

        self._trialsJSON: Path = self.root / "trials.json"
        self.trials: List[Dict[str, Any]] = self._loadTrials()

    def _loadTrials(self) -> List[Dict[str, Any]]:
        if not self._trialsJSON.exists():
            return []
        
        with open(self._trialsJSON, "r") as f:
            trials = json.load(f)
        
        return trials

    def _saveTrials(self):
        with open(self._trialsJSON, "w") as f:
            json.dump(self.trials, f, indent=2)
    
    def _createNextTrial(self) -> Tuple[Trial, int]|Tuple[None, None]:
        definition = self.scheduler.next()
        if definition is None:
            return None, None
        
        trial = Trial.create(definition=definition, root=self._trialsDir)
        idx = len(self.trials)
        trialState = {
            "id": idx,
            "path": str(trial.path),
            "definition_hash": hash(trial.getDefinition()),
            "evals": {}
        }
        self.trials.append(trialState)
        self._saveTrials()
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
        evals = self.evaluator(trial)
        return evals
        
    def eval(self, 
             overwrite: bool=False,
             save: bool=True):
        evalList = []
        for i, t in enumerate(self.trials):
            if not overwrite and len(t["evals"].keys()) > 0:
                evalList.append(t["evals"])
                continue
            trial = Trial.load(path=Path(t["path"]))
            trialEvals = self.evalTrial(trial=trial)
            self.trials[i]["evals"] = trialEvals
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
            self.trials[i]["evals"] = self.evalTrial(trial=trial)
            self._saveTrials()

        print(f"All trials complete.\nTotal trials: {len(self.trials)}")