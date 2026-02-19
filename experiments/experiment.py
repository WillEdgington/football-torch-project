import json

from pathlib import Path

from typing import Dict, List, Any, Tuple, Set

from utils.format import hashDefinition
from .trial_scheduler import TrialScheduler
from .config import TrainFn, ConstructorFn
from .trial import Trial
from .trainer import Trainer
from .evaluator import Evaluator

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
        tmp = self._trialsJSON.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.trials, f, indent=2)

        tmp.replace(self._trialsJSON)

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
            "evals": []
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
        return evals
        
    def eval(self):
        if self.evaluator is None:
            print("Evaluator object must be assigned to Experiment.evaluator to use method Experiment.eval")
            return
        
        evalHash = self.evaluator.evalHash
        new = 0
        for i, t in enumerate(self.trials):
            if evalHash in t.get("evals", []):
                continue
            
            trial = Trial.load(path=Path(t["path"]))
            if not trial.isComplete():
                continue
            new += 1
            runHash = self.evaluator.run(trial)
            self.trials[i]["evals"].append(runHash)
            self._saveTrials()
        
        print(f"All evaluations complete.\nTotal trials: {len(self.trials)}\nNewly evaluated: {new}")

    def run(self):
        while True:
            trial, i = self._prepareTrial()
            if trial is None:
                break
            trainer = Trainer(trial=trial,
                              train=self.train,
                              constructor=self.constructor)
            trainer.run()
            if self.evaluator is not None:
                evalHash = self.evaluator.run(trial)
                if evalHash not in self.trials[i]["evals"]:
                    self.trials[i]["evals"].append(evalHash)
                    
            self._saveTrials()

        print(f"All trials complete.\nTotal trials: {len(self.trials)}")