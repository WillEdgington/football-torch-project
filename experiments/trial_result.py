import json
import torch

from typing import Dict, List, Any

from .trial import Trial

class TrialResult:
    def __init__(self,
                 trial: Trial):
        self._trial: Trial = trial
        self._metrics: Dict[str, List[float]] | None = None
        self._evals: Dict[str, Dict[str, Any]] | None = None

    @property
    def definition(self) -> Dict[str, Any]:
        return self._trial.getDefinition()
    
    @property
    def metrics(self) -> Dict[str, List[float]] | None:
        if self._metrics is None:
            if not self._trial.metricsPath.exists():
                return None
            self._metrics = torch.load(self._trial.metricsPath)
        return self._metrics
    
    @property
    def evals(self) -> Dict[str, Dict[str, Any]]:
        if self._evals is None:
            self._evals = {}
            for evalPath in self._trial.evalsPath.iterdir():
                if evalPath.suffix != ".json":
                    continue
                with open(evalPath) as f:
                    data = json.load(f)
                self._evals[data["eval_hash"]] = data
        return self._evals
    
    def evalHashes(self) -> List[str]:
        return list(self.evals.keys())
    
    def splits(self,
               evalHash: str) -> List[str]:
        return list(self.getEval(evalHash)["results"].keys())
    
    def getEval(self, 
                evalHash: str) -> Dict[str, Any]:
        if evalHash not in self.evals:
            availableHashes = "\n  ".join(self.evalHashes())
            raise ValueError(f"Eval hash not found: {evalHash}\nAvailable Eval hashes for trial:\n  {availableHashes}")
        return self.evals[evalHash]

    def getSplit(self, 
                 evalHash: str, 
                 split: str) -> Dict[str, Any]:
        results = self.getEval(evalHash)["results"]
        if split not in results:
            availableSplits = "\n  ".join(self.splits(evalHash))
            raise ValueError(f"Split not found: {split}\nAvailable splits:\n  {availableSplits}")
        return results[split]