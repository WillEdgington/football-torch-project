import torch

from typing import Dict, Any

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
        self.prefix = self.evalDefinition.get("data", {}).get("prefix_label", "")

    def run(self,
            trial: Trial) -> Dict[str, Any]:
        loaded = self._loadTrial(trial)
        evals = {}
        for key, dataloader in loaded["dataloaders"].items():
            evals[f"{self.prefix}{key}"] = self.eval(model=loaded["model"],
                                                     dataloader=dataloader,
                                                     device=self.device)
        trial._definition = None
        return evals
    
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