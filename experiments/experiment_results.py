import json
import pandas as pd

from typing import Dict, Any, List, Sequence, Literal
from pathlib import Path

from .trial import Trial
from .trial_result import TrialResult

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

def normaliseMetric(df: pd.DataFrame,
                    col: str,
                    method: Literal["standard", "minmax"]="standard",
                    eps: float=1e-8) -> pd.Series:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    series = df[col].astype(float)
    if method == "standard":
        std = series.std()
        return (series - series.mean()) / max(eps, std)
    elif method == "minmax":
        low, high = series.min(), series.max()
        return (series - low) / (high - low)

def addCompositeScore(df: pd.DataFrame,
                      weights: Dict[str, float],
                      ascending: Dict[str, bool],
                      colName: str="composite_score",
                      normMethod: Literal["standard", "minmax"]|Dict[str, Literal["standard", "minmax"]]="standard") -> pd.DataFrame:
    if set(weights.keys()) != set(ascending.keys()):
        raise ValueError("weights and ascending must have the same keys")
    if isinstance(normMethod, dict):
        if set(weights.keys()) != set(normMethod.keys()):
            raise ValueError("if normMethod is dict then it must have the same keys as weights and ascending")

    df = df.copy()
    totalWeight = sum(weights.values())
    score = pd.Series(0.0, index=df.index)

    for col, weight in weights.items():
        score += normaliseMetric(df, 
                                 col, 
                                 method=normMethod if not isinstance(normMethod, dict) else normMethod) \
                                    * (weight / totalWeight) * (-1 if ascending[col] else 1)

    df[colName] = score
    return df

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
        
    def getTrial(self,
                 trial_id: int) -> TrialResult:
        matches = [t for t in self.trials if t["id"] == trial_id]
        if not matches:
            raise KeyError(f"Trial ID {trial_id} not found")
        return TrialResult(Trial.load(Path(matches[0]["path"])))

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
    
    def addNormalisedCol(self,
                         col: str,
                         df: pd.DataFrame|None=None,
                         evalHash: str|None=None,
                         method: Literal["standard", "minmax"]="standard",
                         eps: float=1e-8) -> pd.DataFrame:
        if df is None and evalHash is None:
            raise ValueError("Either df or evalHash must be provided.")
        
        df = self.toDataFrame(evalHash=evalHash) if df is None else df.copy()
        df[f"col_normalised"] = normaliseMetric(df=df,
                                                col=col,
                                                method=method,
                                                eps=eps)
    
    def addCompositeScore(self,
                          weights: Dict[str, float],
                          ascending: Dict[str, bool],
                          df: pd.DataFrame|None=None,
                          evalHash: str|None=None,
                          colName: str="composite_score",
                          normMethod: Literal["standard", "minmax"]|Dict[str, Literal["standard", "minmax"]]="standard") -> pd.DataFrame:
        if df is None and evalHash is None:
            raise ValueError("Either df or evalHash must be provided.")
        if df is None:
            df = self.toDataFrame(evalHash=evalHash)
        return addCompositeScore(df=df,
                                 weights=weights,
                                 ascending=ascending,
                                 colName=colName)