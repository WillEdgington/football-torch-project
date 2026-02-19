import json
import torch

from pathlib import Path
from typing import Dict, Any

from utils.format import toJSONSafe, hashDefinition
from .trial import Trial
from .config import EvaluatorFn, ConstructorFn

def adaptDict(original: Dict[str, Any],
              new: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in new.items():
        if k in original and isinstance(original[k], dict) and isinstance(v, dict):
            original[k] = adaptDict(original[k], v)
        else:
            original[k] = v

    return original

class Evaluator:
    def __init__(self,
                 eval: EvaluatorFn,
                 constructor: ConstructorFn,
                 evalDefinition: Dict[str, Any]):
        self.constructor = constructor
        self.evalDefinition = evalDefinition
        self.eval = eval
        self.evalHash = hashDefinition(self.evalDefinition)

    def run(self,
            trial: Trial) -> Dict[str, Any]:
        loaded = self._loadTrial(trial)
        evals = {}
        for key, dataloader in loaded["dataloaders"].items():
            evals[f"{key}"] = self.eval(model=loaded["model"],
                                        dataloader=dataloader,
                                        device=self.device)
        trial._definition = None
        return self.save(trial, evals)
    
    def save(self,
             trial: Trial,
             results: Dict[str, Any]) -> str:
        evalPath = Path(trial.evalsPath) / f"eval_{self.evalHash}.json"

        if evalPath.exists():
            return self.evalHash
        
        store = {
            "eval_hash": self.evalHash,
            "eval_definition": self.evalDefinition,
            "results": toJSONSafe(results)
        }

        tmpPath = evalPath.with_suffix(".tmp")
        with open(tmpPath, "w") as f:
            json.dump(store, f, indent=2)
        tmpPath.replace(evalPath)
        
        return self.evalHash
    
    def _adaptDefinition(self,
                         trial: Trial) -> Trial:
        trial._definition = adaptDict(original=trial.getDefinition().copy(),
                                      new=self.evalDefinition)
        return trial
    
    def _loadTrial(self,
                   trial: Trial) -> Dict[str, Any]:
        trial = self._adaptDefinition(trial)
        loaded = self.constructor(trial)
        self.device = trial._definition.get("device", 
                                            "cuda" if torch.cuda.is_available else "cpu")
        return loaded