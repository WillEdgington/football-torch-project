import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Any, Set
from pandas import Series

from utils import prepareMatchDataFrame
from .tokeniser import Tokeniser
from .normaliser import Normaliser
from .match_dataset import MatchDataset
from .transforms import Transform
from .config import UNKBUCKETDICT, TOKENISERDIR, NORMALISERDIR, TENSORSDIR, PREMATCHDATACOLS

def tokenise(df: pd.DataFrame, train: bool=True, fileDir: str=TOKENISERDIR, 
             unkBucketDict: Dict[str, int]=UNKBUCKETDICT) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != "object" or col == "match_url":
            continue
        base = col
        if base.startswith("home_") or base.startswith("away_"):
            base = base.removeprefix("home_").removeprefix("away_")
        fileName = f"{base}_tokeniser.json"
        
        unkBuckets = unkBucketDict.get(base, 16)
        with Tokeniser(train=train, unkBuckets=unkBuckets, fileName=fileName, fileDir=fileDir) as tkn:
            df[f"{col}_token"] = tkn.encodeSeries(df[col])

    return df

def normalise(df: pd.DataFrame, train: bool=True, eps: float=1e-8,
              columns: List[str]=[], typeFilter: str|None="float64", method: str="standard",
              fileDir: str=NORMALISERDIR, fileName: str="numeric_normaliser.json", rememberMissing: bool=True) -> pd.DataFrame:
    assert method in {"standard", "minmax"}, 'invalid method input, Choose between: "standard", "minmax"'
    assert typeFilter is None or typeFilter in {"float64", "int64"}, 'typeFilter must be "float64", "int64", or None'
    typeFilters = {typeFilter} if typeFilter else {"float64", "int64"}

    if len(columns) == 0:
        for col in df.columns:
            if str(df[col].dtype) not in typeFilters:
                continue
            columns.append(col)

    with Normaliser(eps=eps, train=train, fileName=fileName, fileDir=fileDir) as nrm:
        if train:
            for col in columns:
                if col.startswith("away_"):
                    continue
                if col.startswith("home_"):
                    base = col.removeprefix("home_")
                    nrm.fit(pd.concat([
                        df[f"home_{base}"], 
                        df[f"away_{base}"]
                        ]), col=base)
                    continue
                nrm.fit(df[col])

        for col in columns:
            if col not in df.columns:
                continue
            base = col[5:] if col.startswith("home_") or col.startswith("away_") else col
            if rememberMissing:
                df[f"{col}_missing"] = df[col].isna().astype(np.int8)
            
            df[col] = df[col].fillna(value=nrm.params[base]["mean"])
            df[f"{col}_normalised"] = nrm.encodeSeries(df[col], col=base, method=method, fit=False)
    return df

def splitDataFrame(df: pd.DataFrame, trainSplit: float=0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert 0 < trainSplit < 1, "trainSplit must be in (0, 1)"
    splitIdx = round(len(df) * 0.8)
    return df[:splitIdx], df[splitIdx:]

def getTemporalSplits(df: pd.DataFrame, trainSplit: float=0.8, valSplit: float=0, sortingCol: str="date") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    assert 0 < trainSplit < 1, f"Training split (trainSplit={trainSplit}) must be in (0, 1)."
    assert 0 <= valSplit < 1, f"Validation split (valSplit={valSplit}) must be in [0, 1)."
    assert sortingCol in df.columns, f"Sorting column (sortingCol={sortingCol}) is not a column in DataFrame (df)."
    
    df = df.copy().sort_values(by=sortingCol)
    train, test = splitDataFrame(df, trainSplit=trainSplit)
    
    if valSplit == 0:
        return train, test, None
    train, val = splitDataFrame(train, trainSplit=1-valSplit)
    return train, test, val

def filterCols(df: pd.DataFrame, removeCols: List[str]=["match_id"]) -> pd.DataFrame:
    return df.copy().drop(columns=removeCols)

def getMatchIDDateTeamDF(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    home = df[["match_id", "date", "home_team"]].rename(
        columns={"home_team": "team"}
    )

    away = df[["match_id", "date", "away_team"]].rename(
        columns={"away_team": "team"}
    )
    longDf = pd.concat([home, away], axis=0, ignore_index=True)
    longDf.sort_values(["team", "date"], inplace=True)
    return longDf

def addDaysSinceLastGameToLongDf(longDf: pd.DataFrame, dropHelperCols: bool=True) -> pd.DataFrame:
    assert "team" in longDf.columns, 'longDf requires "team" column.'
    assert "date" in longDf.columns, 'longDf requires "date" column.'
    assert "match_id" in longDf.columns, 'longDf requires "match_id" column.'

    longDf = longDf.copy()
    longDf.sort_values(["team", "date"], inplace=True)
    longDf["prev_date"] = longDf.groupby("team")["date"].shift(1)
    longDf["days_since_last_game"] = (
        longDf["date"] - longDf["prev_date"]
    ).dt.days

    if dropHelperCols:
        longDf.drop(columns=["date", "prev_date"])
    return longDf

def mergeLongDfBackToMatchDf(df: pd.DataFrame, longDf: pd.DataFrame) -> pd.DataFrame:
    assert "team" in longDf.columns, 'longDf requires "team" column.'
    assert "match_id" in longDf.columns, 'longDf requires "match_id" column.'
    assert "days_since_last_game" in longDf.columns, 'longDf requires "days_since_last_game" column.'
    df = df.copy()
    
    home = longDf.rename(
        columns={"days_since_last_game": "home_days_since_last_game"}
    )[["match_id", "team", "home_days_since_last_game"]]
    away = longDf.rename(
        columns={"days_since_last_game": "away_days_since_last_game"}
    )[["match_id", "team", "away_days_since_last_game"]]

    df = df.merge(
        home, left_on=["match_id", "home_team"], right_on=["match_id", "team"], how="left"
    ).drop(columns=["team"])
    df = df.merge(
        away, left_on=["match_id", "away_team"], right_on=["match_id", "team"], how="left"
    ).drop(columns=["team"])
    return df
    
def addDaysSinceLastGame(df: pd.DataFrame) -> pd.DataFrame:
    assert "date" in df.columns, 'df requires "date" column.'
    assert "match_id" in df.columns, 'df requires "match_id" column.'
    df = df.copy()
    longDf = getMatchIDDateTeamDF(df=df)
    longDf = addDaysSinceLastGameToLongDf(longDf=longDf)
    df = mergeLongDfBackToMatchDf(df=df, longDf=longDf)
    return df

def matchDfToPerTeamDfs(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    assert "home_team" in df.columns and "away_team" in df.columns, 'df must have columns "home_team" and "away_team"'
    home = df.copy()
    home["team"] = home["home_team"]

    away = df.copy()
    away["team"] = away["away_team"]

    longDf = pd.concat([home, away], ignore_index=True)
    teams = {str(team): g.sort_values("date").reset_index(drop=True).drop(columns=["team"])
             for team, g in longDf.groupby("team")}
    return teams

def createY(df: pd.DataFrame, yCols: List[str]|str="result") -> Dict[str, np.ndarray]:
    df = df.copy()
    if isinstance(yCols, str):
        yCols = [yCols]
    if "result" in yCols:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
        df["result"] = df["goal_diff"].apply(lambda gd: 2 if gd > 0 else (0 if gd < 0 else 1))
    
    matchIds = df["match_id"].to_list()
    return {matchIds[i]: df[df["match_id"] == matchIds[i]][yCols].to_numpy(dtype=np.float32).reshape(-1) for i in range(len(matchIds))}

def buildTeamWindowsV0(teamDf: pd.DataFrame, 
                       featureCols: List[str], 
                       seqLen: int=20,
                       missingCols: List[str]|None=None) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    teamDf = teamDf.sort_values("date").reset_index(drop=True)
    matchIds = teamDf["match_id"].tolist()
    
    featN = len(featureCols)
    missN = len(missingCols) if missingCols else 0
    window = np.repeat(np.zeros(featN, dtype=np.float32).reshape(1, -1), seqLen, axis=0)
    mask = np.array([0]*(seqLen - 1) + [1], dtype=np.int32) # atleast one row must be unmasked
    missing = np.repeat(np.ones(missN, dtype=np.int8).reshape(1, -1), seqLen, axis=0) if missingCols else None
    
    windows = {matchIds[0]: (window.copy(), mask.copy(), missing.copy())}

    for i in range(len(matchIds) - 1):
        game = teamDf[teamDf["match_id"] == matchIds[i]]
        prevMatch = game[featureCols].to_numpy(dtype=np.float32)
        if mask[0] != 1 and i != 0:
            mask[0] = 1
            mask = np.roll(mask, shift=-1)

        window[0] = prevMatch.copy()
        window = np.roll(window, shift=-1, axis=0)

        if missing is not None:
            missing[0] = game[missingCols].to_numpy(dtype=np.int8)
            missing = np.roll(missing, shift=-1, axis=0)
        windows[matchIds[i + 1]] = (window.copy(), mask.copy(), missing.copy())

    return windows

def buildTeamWindows(teamDf: pd.DataFrame, 
                     featureCols: List[str], 
                     seqLen: int=20,
                     missingCols: List[str]|None=None,
                     preMatchData: Set[str]|None=PREMATCHDATACOLS) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray|None]]:
    if preMatchData is None:
        return buildTeamWindowsV0(teamDf=teamDf,
                                  featureCols=featureCols,
                                  seqLen=seqLen,
                                  missingCols=missingCols)
    
    teamDf = teamDf.sort_values("date").reset_index(drop=True)
    matchIds = teamDf["match_id"].tolist()
    
    windows = {}

    featN = len(featureCols)
    missN = len(missingCols) if missingCols else 0

    preFeatCols = np.array([int(feat in preMatchData) for feat in featureCols], 
                           dtype=np.float32)
    preMissCols = np.array([int(miss.removesuffix("_missing") + "_normalised" not in preMatchData) for miss in missingCols], 
                           dtype=np.int8)
    
    window = np.repeat(np.zeros(featN, dtype=np.float32).reshape(1, -1), seqLen, axis=0)
    mask = np.array([0]*seqLen, dtype=np.int32)
    missing = np.repeat(np.ones(missN, dtype=np.int8).reshape(1, -1), seqLen, axis=0) if missingCols else None

    for matchId in matchIds:
        game = teamDf[teamDf["match_id"] == matchId]
        assert len(game) == 1, f"Expected 1 row for match {matchId}, got {len(game)}"
        curMatch = game[featureCols].to_numpy(dtype=np.float32)
        
        if mask[0] != 1:
            mask[0] = 1
            mask = np.roll(mask, shift=-1)

        window[0] = curMatch.copy() * preFeatCols
        window = np.roll(window, shift=-1, axis=0)

        if missing is not None:
            missingInRow = game[missingCols].to_numpy(dtype=np.int8)
            missing[0] = missingInRow.copy() | preMissCols
            missing = np.roll(missing, shift=-1, axis=0)

        windows[matchId] = (window.copy(), mask.copy(), missing.copy())
        window[-1] = curMatch.copy()
        if missing is not None:
            missing[-1] = missingInRow.copy()

    return windows

def buildAllWindows(df: pd.DataFrame,
                    featureCols: list[str],
                    yCols: List[str]|str|None=None,
                    seqLen: int=20,
                    transform: Transform|None=None,
                    rememberMissing: bool=True,
                    preMatchData: Set[str]|None=PREMATCHDATACOLS) -> MatchDataset:
    """Set yCols to None to get columns with numeric values from featureCols as Y"""
    if isinstance(yCols, str):
        yCols = [yCols]
    if yCols is None:
        yCols = [col for col in featureCols if col in df.columns and str(df[col].dtype) in {"float64", "int64"} 
                 and not col.endswith("_days_since_last_game_normalised")
                 and not col.endswith("time_normalised")]
    
    missingCols = [col for col in df.columns if col.endswith("_missing")]

    teamDfs = matchDfToPerTeamDfs(df=df)
    Ydict = createY(df=df, yCols=yCols)
    teamWindows = {
        team: buildTeamWindows(teamDf=tdf, featureCols=featureCols, seqLen=seqLen, missingCols=missingCols, preMatchData=preMatchData) 
        for team, tdf in teamDfs.items()
    }
    
    Xhome = []
    Xaway = []
    Mhome = []
    Maway = []
    Yarr = []
    missingHome = []
    missingAway = []

    for matchId, homeTeam, awayTeam in zip(df["match_id"], df["home_team"], df["away_team"]):
        Xh, Mah, Mih = teamWindows[homeTeam][matchId]
        Xa, Maa, Mia = teamWindows[awayTeam][matchId]
        Y = Ydict[matchId]
        Xhome.append(Xh)
        Xaway.append(Xa)
        Mhome.append(Mah)
        Maway.append(Maa)
        if rememberMissing:
            missingHome.append(Mih)
            missingAway.append(Mia)
        Yarr.append(Y)
    
    Xhome = np.stack(Xhome)
    Xaway = np.stack(Xaway)
    Mhome = np.stack(Mhome)
    Maway = np.stack(Maway)
    if rememberMissing:
        missingHome = np.stack(missingHome)
        missingAway = np.stack(missingAway)
    Y  = np.stack(Yarr)

    return MatchDataset(
        Xhome=torch.from_numpy(Xhome),
        Xaway=torch.from_numpy(Xaway),
        maskHome=torch.from_numpy(Mhome),
        maskAway=torch.from_numpy(Maway),
        missingHome=torch.from_numpy(missingHome) if rememberMissing else None,
        missingAway=torch.from_numpy(missingAway) if rememberMissing else None,
        missingCols=missingCols,
        Y=torch.from_numpy(Y),
        featureCols=featureCols,
        yCols=yCols,
        transform=transform
    )

def createDataset(df: pd.DataFrame, featureCols: List[str]|None=None, type: str="train", tokeniserDir: str=TOKENISERDIR, 
                  unkBucketDict: Dict[str, int]=UNKBUCKETDICT, normMethod: str="standard",
                  normaliserDir: str=NORMALISERDIR, normaliserJSON: str="numeric_normaliser.json",
                  yCols: List[str]|str|None=None, seqLen: int=20, tensorDir: str=TENSORSDIR, 
                  transform: Transform|None=None, save: bool=True, rememberMissing: bool=True,
                  preMatchData: Set[str]|None=PREMATCHDATACOLS) -> MatchDataset:
    train = type == "train"
    df = tokenise(df=df, train=train, fileDir=tokeniserDir, unkBucketDict=unkBucketDict)
    df = addDaysSinceLastGame(df=df)
    df = normalise(df=df, train=train, method=normMethod, fileDir=normaliserDir, fileName=normaliserJSON, rememberMissing=rememberMissing)
    if featureCols is None:
        featureCols = [col for col in df.columns if col.endswith("_token") or col.endswith("_normalised")]
    
    ds = buildAllWindows(df=df, featureCols=featureCols, yCols=yCols, 
                         seqLen=seqLen, transform=transform, rememberMissing=rememberMissing, 
                         preMatchData=preMatchData)
    if save:
        ds.save(parentDir=tensorDir, fileDir=type)
    return ds

def tensorDatasetsFromMatchDf(df: pd.DataFrame|None=None, trainSplit: float=0.8, valSplit: float=0.2, save: bool=True,
                              featureCols: List[str]|None=None, yCols: List[str]|str|None="result", seqLen: int=20,
                              normMethod: str="standard", unkBucketDict: Dict[str, int]=UNKBUCKETDICT, 
                              normaliserDir: str=NORMALISERDIR, normaliserJSON: str="numeric_normaliser.json",
                              tokeniserDir: str=TOKENISERDIR, tensorDir: str=TENSORSDIR, 
                              trainTransform: Transform|None=None, rememberMissing: bool=True,
                              preMatchData: Set[str]|None=PREMATCHDATACOLS) -> Dict[str, MatchDataset]:
    if df is None:
        df = prepareMatchDataFrame()
    
    trainDf, testDf, valDf = getTemporalSplits(df=df, trainSplit=trainSplit, valSplit=valSplit)

    trainDs = createDataset(trainDf, featureCols=featureCols, type="train", tokeniserDir=tokeniserDir,
                            unkBucketDict=unkBucketDict, normMethod=normMethod,
                            normaliserDir=normaliserDir, normaliserJSON=normaliserJSON,
                            yCols=yCols, seqLen=seqLen, tensorDir=tensorDir, save=save, 
                            transform=trainTransform, rememberMissing=rememberMissing,
                            preMatchData=preMatchData)
    testDs = createDataset(testDf, featureCols=featureCols, type="test", tokeniserDir=tokeniserDir,
                            unkBucketDict=unkBucketDict, normMethod=normMethod,
                            normaliserDir=normaliserDir, normaliserJSON=normaliserJSON,
                            yCols=yCols, seqLen=seqLen, tensorDir=tensorDir, save=save,
                            rememberMissing=rememberMissing, preMatchData=preMatchData)
    valDs = createDataset(valDf, featureCols=featureCols, type="validation", tokeniserDir=tokeniserDir,
                          unkBucketDict=unkBucketDict, normMethod=normMethod,
                          normaliserDir=normaliserDir, normaliserJSON=normaliserJSON,
                          yCols=yCols, seqLen=seqLen, tensorDir=tensorDir, save=save,
                          rememberMissing=rememberMissing, preMatchData=preMatchData) if valDf is not None else None
    tensorDict = {
        "train": trainDs,
        "test": testDs,
    }
    if valDs is not None:
        tensorDict["validation"] = valDs

    return tensorDict

def createDataLoader(dataset: MatchDataset, batchSize: int=64, shuffle: bool=True, 
                     numWorkers: int=1, seed: int=42, pinMemory: bool=True) -> DataLoader:
    torch.manual_seed(seed=seed)
    dataloader = DataLoader(dataset=dataset, batch_size=batchSize, shuffle=shuffle, num_workers=numWorkers, pin_memory=pinMemory)
    return dataloader

def createDataLoaders(tensorDict: Dict[str, MatchDataset], batchSize: int=64, 
                      numWorkers: int=1, seed: int=42, pinMemory: bool=True) -> Dict[str, DataLoader]:
    loaderDict = {}
    for name, ds in tensorDict.items():
        shuffle = name == "train"
        loaderDict[name] = createDataLoader(dataset=ds, batchSize=batchSize, shuffle=shuffle,
                                            numWorkers=numWorkers, seed=seed, pinMemory=pinMemory)
    
    return loaderDict

def prepareData(df: pd.DataFrame|None=None, type: str|None=None, trainSplit: float=0.8, valSplit: float=0.2, save: bool=True,
                featureCols: List[str]|None=None, yCols: List[str]|str|None="result", seqLen: int=20,
                normMethod: str="standard", unkBucketDict: Dict[str, int]=UNKBUCKETDICT, 
                normaliserDir: str=NORMALISERDIR, normaliserJSON: str="numeric_normaliser.json",
                tokeniserDir: str=TOKENISERDIR, tensorDir: str=TENSORSDIR, batchSize: int=64,
                numWorkers: int=1, seed: int=42, pinMemory: bool=True, shuffle: bool|None=None,
                trainTransform: Transform|None=None, rememberMissing: bool=True, 
                preMatchData: Set[str]|None=PREMATCHDATACOLS) -> Dict[str, DataLoader]|DataLoader|None:
    """set type=None to get a dict object containing test, train, and validation (if available) DataLoader objects"""
    assert type is None or type in {"train", "test", "validation"}, 'type must be None or one of "train", "test", "validation"'
    tensorDict = {}
    types = [type] if type is not None else ["train", "test", "validation"]
    for split in types:
        transform = trainTransform if split == "train" else None
        tensorDict[split] = MatchDataset.load(transform=transform, parentDir=tensorDir, fileDir=split)

    if all(val is None for val in tensorDict.values()):
        tensorDict = tensorDatasetsFromMatchDf(df=df, trainSplit=trainSplit, valSplit=valSplit, save=save, featureCols=featureCols,
                                               yCols=yCols, seqLen=seqLen, normMethod=normMethod, unkBucketDict=unkBucketDict,
                                               normaliserDir=normaliserDir, normaliserJSON=normaliserJSON, tokeniserDir=tokeniserDir,
                                               tensorDir=tensorDir, trainTransform=trainTransform, rememberMissing=rememberMissing,
                                               preMatchData=preMatchData)
    if type is None:
        if tensorDict["validation"] is None:
            del tensorDict["validation"]
        return createDataLoaders(tensorDict=tensorDict, batchSize=batchSize, numWorkers=numWorkers, seed=seed, pinMemory=pinMemory)
    dataset = tensorDict[type]
    shuffle = shuffle if shuffle is not None else type == "train"
    return createDataLoader(dataset=dataset, batchSize=batchSize, shuffle=shuffle, 
                            numWorkers=numWorkers, seed=seed, pinMemory=pinMemory) if dataset is not None else None