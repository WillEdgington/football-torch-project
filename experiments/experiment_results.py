import json
import pandas as pd

from typing import Dict, Any, List, Sequence
from pathlib import Path

from .trial import Trial

def _flatten(dic: Dict[str, Any], 
             parentKey: str="", 
             sep: str=".") -> Dict[str, Any]:
    items = []
    for k, v in dic.items():
        newKey = f"{parentKey}{sep}{k}" if parentKey else k
        if isinstance(v, dict):
            items.extend(_flatten(v, parentKey=newKey, sep=sep).items())
            continue
        items.append((newKey, v))
    return dict(items)

def _dropKeys(flat: Dict[str, Any],
              exclude: Sequence[str]) -> Dict[str, Any]:
    if not exclude:
        return flat
    return {k: v for k, v in flat.items()
            if not any(k == ex or k.startswith(ex + ".") for ex in exclude)}

class ExperimentResults:
    def __init__(self,
                 root: Path|str):
        self.root: Path = root if isinstance(root, Path) else Path(root)
        self._trialsDir: Path = self.root / "trials"
        assert self._trialsDir.exists(), f"could not find any trials at: {self._trialsDir}"

        self._trialsJSON = self.root / "trials.json"
        assert self._trialsJSON.exists(), f"could not find trials.json file at: {self.root}"
        self.trials: List[Dict[str, Any]] = self._loadTrials()

    def _loadTrials(self) -> List[Dict[str, Any]]:
        with open(self._trialsJSON, "r") as f:
            return json.load(f)
        
    def _loadEval(self,
                  trial: Trial,
                  evalHash: str) -> Dict[str, Any] | None:
        evalPath = trial.evalsPath / f"eval_{evalHash}.json"
        if not evalPath.exists():
            return None
        with open(evalPath) as f:
            return json.load(f)

    def toDataFrame(self,
                    evalHash: str,
                    split: bool=False,
                    exclude: Sequence[str]= ("calibration", "confusion_matrix")) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        defRows: List[Dict[str, Any]] = []
        evalRows: List[Dict[str, Any]] = []

        for t in self.trials:
            if evalHash not in t.get("evals", []):
                continue

            trial = Trial.load(Path(t["path"]))
            trialID = t["id"]

            evalData = self._loadEval(trial, evalHash)
            if evalData is None:
                continue

            flatDef: Dict[str, Any] = _flatten(trial.getDefinition())
            flatDef["trial_id"] = trialID
            defRows.append(flatDef)

            flatEval: Dict[str, Any] = {"trial_id": trialID}
            for splitName, splitMetrics in evalData["results"].items():
                if not isinstance(splitMetrics, dict):
                    if splitName not in exclude:
                        flatEval[splitName] = splitMetrics
                    continue
                flat = _flatten(splitMetrics, parentKey=splitName)
                flat = _dropKeys(flat, [f"{splitName}.{ex}" for ex in exclude])
                flatEval.update(flat)
            evalRows.append(flatEval)
        
        defDf: pd.DataFrame = pd.DataFrame(defRows).set_index("trial_id")
        evalDf: pd.DataFrame = pd.DataFrame(evalRows).set_index("trial_id")
        
        if split:
            return defDf, evalDf
        
        return defDf.join(evalDf, how="inner")