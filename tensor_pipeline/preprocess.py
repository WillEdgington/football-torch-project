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
from .sample_store import SampleStore
from .config import UNKBUCKETDICT, TOKENISERDIR, NORMALISERDIR, TENSORSDIR, PREMATCHDATACOLS

def tokenise(df: pd.DataFrame, train: bool=True, fileDir: str=TOKENISERDIR, 
             unkBucketDict: Dict[str, int]=UNKBUCKETDICT) -> pd.DataFrame:
    df = df.copy()
    newCols = {}

    for col in df.columns:
        if df[col].dtype != "object" or col == "match_url":
            continue
        base = col
        if base.startswith("home_") or base.startswith("away_"):
            base = base.removeprefix("home_").removeprefix("away_")
        fileName = f"{base}_tokeniser.json"
        
        unkBuckets = unkBucketDict.get(base, 16)
        with Tokeniser(train=train, unkBuckets=unkBuckets, fileName=fileName, fileDir=fileDir) as tkn:
            newCols[f"{col}_token"] = tkn.encodeSeries(df[col])

    if newCols:
        df = pd.concat([df, pd.DataFrame(newCols, index=df.index)], axis=1)

    return df

def normalise(df: pd.DataFrame, train: bool=True, eps: float=1e-8,
              columns: List[str]=[], typeFilter: str|None="float64", method: str="standard",
              fileDir: str=NORMALISERDIR, fileName: str="numeric_normaliser.json", rememberMissing: bool=True) -> pd.DataFrame:
    assert method in {"standard", "minmax"}, 'invalid method input, Choose between: "standard", "minmax"'
    assert typeFilter is None or typeFilter in {"float64", "int64"}, 'typeFilter must be "float64", "int64", or None'
    typeFilters = {typeFilter} if typeFilter else {"float64", "int64"}
    df = df.copy()
    newCols = {}

    if len(columns) == 0:
        columns = [
            col for col in df.columns
            if str(df[col].dtype) in typeFilters
        ]

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
                newCols[f"{col}_missing"] = df[col].isna().astype(np.int8)
            
            filled = df[col].fillna(value=nrm.params[base]["mean"])
            newCols[f"{col}_normalised"] = nrm.encodeSeries(filled, col=base, method=method, fit=False)
    
    if newCols:
        df = pd.concat([df, pd.DataFrame(newCols, index=df.index)], axis=1)

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

def createY(df: pd.DataFrame,
            yCols: List[str]|str="result",
            groupCol: str="league") -> Dict[str, Tuple[np.ndarray, str]]:
    df = df.copy()
    if isinstance(yCols, str):
        yCols = [yCols]
    if "result" in yCols:
        df["goal_diff"] = df["home_goals"] - df["away_goals"]
        df["result"] = df["goal_diff"].apply(lambda gd: 2 if gd > 0 else (0 if gd < 0 else 1))
    
    matchIds = df["match_id"].to_list()
    return {
        matchIds[i]: (
            df[df["match_id"] == matchIds[i]][yCols].to_numpy(dtype=np.float32).reshape(-1),
            str(df.loc[df["match_id"] == matchIds[i], groupCol].iloc[0])
            )
        for i in range(len(matchIds))
        }

def buildTeamWindowsV0(teamDf: pd.DataFrame,
                       featureCols: List[str],
                       seqLen: int=20,
                       missingCols: List[str]|None=None) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray|None]]:
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

        windows[matchId] = (window.copy(), mask.copy(), missing.copy() if missing is not None else None)
        window[-1] = curMatch.copy()
        if missing is not None:
            missing[-1] = missingInRow.copy()

    return windows

def constructTokenContCols(featureCols: List[str], 
                           tokeniserDir: str=TOKENISERDIR) -> Tuple[Dict[str, List[int]], Dict[str, List[int]]]:
    tokenCols = {
        "index": [],
        "size": [],
        "unkBucketSize": []
    }
    contCols = {
        "index": [],
    }
    sizeCache = {}

    for i, col in enumerate(featureCols):
        if not col.endswith("_token"):
            contCols["index"].append(i)
            continue
        tokenCols["index"].append(i)
        
        base = col.removesuffix("_token")
        if base.startswith("home_") or base.startswith("away_"):
            base = base.removeprefix("home_").removeprefix("away_")
        fileName = f"{base}_tokeniser.json"
        
        if base in sizeCache:
            tokenCols["size"].append(sizeCache[base][0])
            tokenCols["unkBucketSize"].append(sizeCache[base][1])
            del sizeCache[base]
            continue

        with Tokeniser(train=False, fileName=fileName, fileDir=tokeniserDir) as tkn:
            sizeCache[base] = (len(tkn.idtos), tkn.unkBuckets)
            tokenCols["size"].append(sizeCache[base][0])
            tokenCols["unkBucketSize"].append(sizeCache[base][1])
        
    return tokenCols, contCols

def buildMetaData(featureCols: List[str],
                  yCols: List[str],
                  maxSeqLen: int=50,
                  tokenCols: Dict[str, List[int]]|None=None,
                  contCols: Dict[str, List[int]]|None=None,
                  missingCols: List[str]|None=None,
                  tokeniserDir: str=TOKENISERDIR) -> Dict[str, Any]:
    if tokenCols is None or contCols is None:
        tokenCols, contCols = constructTokenContCols(featureCols=featureCols,
                                                     tokeniserDir=tokeniserDir)
    meta = {
        "maxSeqLen": maxSeqLen,
        "featureCols": featureCols,
        "yCols": yCols,
        "tokenCols": tokenCols,
        "contCols": contCols,
    }
    if missingCols is not None:
        meta["missingCols"] = missingCols
    return meta

def buildSample(home: np.ndarray,
                away: np.ndarray,
                maskHome: np.ndarray,
                maskAway: np.ndarray,
                y: np.ndarray,
                missingHome: np.ndarray|None=None,
                missingAway: np.ndarray|None=None) -> Dict[str, torch.Tensor]:
    sample = {
        "home": torch.from_numpy(home),
        "away": torch.from_numpy(away),
        "mask_home": torch.from_numpy(maskHome),
        "mask_away": torch.from_numpy(maskAway),
        "y": torch.from_numpy(y), 
    }

    if missingHome is not None and missingAway is not None:
        sample["missing_home"] = torch.from_numpy(missingHome)
        sample["missing_away"] = torch.from_numpy(missingAway)
    
    return sample

def buildGroupedSamples(df: pd.DataFrame,
                        featureCols: List[str],
                        yCols: List[str]|str|None=None,
                        missingCols: List[str]|None=None,
                        maxSeqLen: int=50,
                        preMatchData: Set[str]|None=PREMATCHDATACOLS,
                        groupCol: str="league") -> Dict[str, List[Dict[str, torch.Tensor]]]:
    """Set yCols to None to get columns with numeric values from featureCols as Y"""
    if isinstance(yCols, str):
        yCols = [yCols]
    if yCols is None:
        yCols = [col for col in featureCols if col in df.columns and str(df[col].dtype) in {"float64", "int64"} 
                 and not col.endswith("_days_since_last_game_normalised")
                 and not col.endswith("time_normalised")]

    teamDfs = matchDfToPerTeamDfs(df=df)
    Ydict = createY(df=df, yCols=yCols, groupCol=groupCol)
    teamWindows = {
        team: buildTeamWindows(teamDf=tdf, featureCols=featureCols, seqLen=maxSeqLen, missingCols=missingCols, preMatchData=preMatchData) 
        for team, tdf in teamDfs.items()
    }

    groupedSamples = {}

    for matchId, homeTeam, awayTeam in zip(df["match_id"], df["home_team"], df["away_team"]):
        Xh, Mah, Mih = teamWindows[homeTeam][matchId]
        Xa, Maa, Mia = teamWindows[awayTeam][matchId]
        Y, group = Ydict[matchId]

        groupedSamples.setdefault(group, [])
        groupedSamples[group].append(buildSample(home=Xh,
                                                 away=Xa,
                                                 maskHome=Mah,
                                                 maskAway=Maa,
                                                 y=Y,
                                                 missingHome=Mih,
                                                 missingAway=Mia))

    return groupedSamples

def storeGroupedSamples(store: SampleStore,
                        groupedSamples: Dict[str, List[Dict[str, torch.Tensor]]],
                        split: str):
    for group, samples in groupedSamples.items():
        store.store(split=split, group=group, samples=samples)
    store.finalise()

def buildAndStoreSamplesFromDfSplit(df: pd.DataFrame,
                                    featureCols: List[str]|None=None,
                                    split: str="train",
                                    tokeniserDir: str=TOKENISERDIR,
                                    unkBucketDict: Dict[str, int]=UNKBUCKETDICT, 
                                    normMethod: str="standard",
                                    normaliserDir: str=NORMALISERDIR, 
                                    normaliserJSON: str="numeric_normaliser.json",
                                    yCols: List[str]|str|None=None, 
                                    maxSeqLen: int=50, 
                                    tensorDir: str=TENSORSDIR, 
                                    rememberMissing: bool=True,
                                    preMatchData: Set[str]|None=PREMATCHDATACOLS,
                                    shardSize: int=1024,
                                    groupCol: str="league") -> SampleStore:
    train = split == "train"
    df = tokenise(df=df, 
                  train=train, 
                  fileDir=tokeniserDir, 
                  unkBucketDict=unkBucketDict)
    df = addDaysSinceLastGame(df=df)
    df = normalise(df=df, 
                   train=train, 
                   method=normMethod, 
                   fileDir=normaliserDir, 
                   fileName=normaliserJSON, 
                   rememberMissing=rememberMissing)
    if featureCols is None:
        featureCols = [col for col in df.columns if col.endswith("_token") or col.endswith("_normalised")]
    missingCols = [col for col in df.columns if col.endswith("_missing")] if rememberMissing else None
    groupedSamples = buildGroupedSamples(df=df,
                                         featureCols=featureCols,
                                         yCols=yCols,
                                         maxSeqLen=maxSeqLen,
                                         missingCols=missingCols,
                                         preMatchData=preMatchData,
                                         groupCol=groupCol)
    meta = buildMetaData(featureCols=featureCols,
                         yCols=yCols,
                         maxSeqLen=maxSeqLen,
                         tokenCols=None,
                         contCols=None,
                         missingCols=missingCols,
                         tokeniserDir=tokeniserDir)
    store = SampleStore(rootDir=tensorDir,
                        shardSize=shardSize,
                        metadata=meta,
                        device="cpu")
    storeGroupedSamples(store=store,
                        groupedSamples=groupedSamples,
                        split=split)
    return store

def createTrainTestDatasets(store: SampleStore,
                            seqLen: int=20,
                            validation: bool=True,
                            trainTransform: Transform|None=None,
                            group: List[str]|str|None=None) -> Dict[str, MatchDataset]:
    "set group to None to get all groups"
    trainDs = MatchDataset(store=store,
                           seqLen=seqLen,
                           transform=trainTransform,
                           split="train",
                           group=group)
    testDs = MatchDataset(store=store,
                          seqLen=seqLen,
                          transform=None,
                          split="test",
                          group=group)
    datasets = {
        "train": trainDs,
        "test": testDs
    }
    if validation:
        datasets["validation"] = MatchDataset(store=store,
                                              seqLen=seqLen,
                                              transform=None,
                                              split="validation",
                                              group=group)
    return datasets

def tensorStoreFromMatchDF(df: pd.DataFrame|None=None,
                           trainSplit: float=0.8,
                           valSplit: float=0.2,
                           featureCols: List[str]|None=None,
                           yCols: List[str]|str|None="result",
                           maxSeqLen: int=50,
                           normMethod: str="standard",
                           unkBucketDict: Dict[str, int]=UNKBUCKETDICT,
                           normaliserDir: str=NORMALISERDIR,
                           normaliserJSON: str="numeric_normaliser.json",
                           tokeniserDir: str=TOKENISERDIR,
                           tensorDir: str=TENSORSDIR,
                           rememberMissing: bool=True,
                           preMatchData: Set[str]|None=PREMATCHDATACOLS,
                           shardSize: int=1024,
                           groupCol: str="league") -> SampleStore:
    if df is None:
        df = prepareMatchDataFrame()

    trainDf, testDf, valDf = getTemporalSplits(df=df, trainSplit=trainSplit, valSplit=valSplit)
    buildAndStoreSamplesFromDfSplit(df=trainDf,
                                    featureCols=featureCols,
                                    split="train",
                                    tokeniserDir=tokeniserDir,
                                    unkBucketDict=unkBucketDict,
                                    normMethod=normMethod,
                                    normaliserDir=normaliserDir,
                                    normaliserJSON=normaliserJSON,
                                    yCols=yCols,
                                    maxSeqLen=maxSeqLen,
                                    tensorDir=tensorDir,
                                    rememberMissing=rememberMissing,
                                    preMatchData=preMatchData,
                                    shardSize=shardSize,
                                    groupCol=groupCol)
    if valDf is not None:
        buildAndStoreSamplesFromDfSplit(df=valDf,
                                        featureCols=featureCols,
                                        split="validation",
                                        tokeniserDir=tokeniserDir,
                                        unkBucketDict=unkBucketDict,
                                        normMethod=normMethod,
                                        normaliserDir=normaliserDir,
                                        normaliserJSON=normaliserJSON,
                                        yCols=yCols,
                                        maxSeqLen=maxSeqLen,
                                        tensorDir=tensorDir,
                                        rememberMissing=rememberMissing,
                                        preMatchData=preMatchData,
                                        shardSize=shardSize,
                                        groupCol=groupCol)
    store = buildAndStoreSamplesFromDfSplit(df=testDf,
                                            featureCols=featureCols,
                                            split="test",
                                            tokeniserDir=tokeniserDir,
                                            unkBucketDict=unkBucketDict,
                                            normMethod=normMethod,
                                            normaliserDir=normaliserDir,
                                            normaliserJSON=normaliserJSON,
                                            yCols=yCols,
                                            maxSeqLen=maxSeqLen,
                                            tensorDir=tensorDir,
                                            rememberMissing=rememberMissing,
                                            preMatchData=preMatchData,
                                            shardSize=shardSize,
                                            groupCol=groupCol)
    return store

def checkForStore(tensorDir: str=TENSORSDIR) -> None|SampleStore:
    store = SampleStore(rootDir=tensorDir,
                        metadata={})
    if store.numSamples > 0:
        return store
    return None

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

def prepareData(df: pd.DataFrame|None=None,
                split: str|None=None,
                trainSplit: float=0.8,
                valSplit: float=0.2,
                featureCols: List[str]|None=None,
                yCols: List[str]|str|None="result",
                maxSeqLen: int=50,
                normMethod: str="standard",
                unkBucketDict: Dict[str, int]=UNKBUCKETDICT,
                normaliserDir: str=NORMALISERDIR,
                normaliserJSON: str="numeric_normaliser.json",
                tokeniserDir: str=TOKENISERDIR,
                tensorDir: str=TENSORSDIR,
                rememberMissing: bool=True,
                preMatchData: set[str]|None=PREMATCHDATACOLS,
                shardSize: int=1024,
                seqLen: int=20,
                trainTransform: Transform|None=None,
                groupCol: str="league",
                groups: List[str]|str|None=None,
                batchSize: int=64,
                numWorkers: int=1,
                seed: int=42,
                pinMemory: bool=True) -> Dict[str, DataLoader]:
    """set split=None to get a dict object containing test, train, and validation (if available) DataLoader objects"""
    assert split is None or split in {"train", "test", "validation"}, 'split must be None or one of "train", "test", "validation"'
    splits = [split] if split is not None else ["train", "test", "validation"]

    store = checkForStore(tensorDir=tensorDir)
    if store is None:
        store = tensorStoreFromMatchDF(df=df,
                                       trainSplit=trainSplit,
                                       valSplit=valSplit,
                                       featureCols=featureCols,
                                       yCols=yCols,
                                       maxSeqLen=maxSeqLen,
                                       normMethod=normMethod,
                                       unkBucketDict=unkBucketDict,
                                       normaliserDir=normaliserDir,
                                       normaliserJSON=normaliserJSON,
                                       tokeniserDir=tokeniserDir,
                                       tensorDir=tensorDir,
                                       rememberMissing=rememberMissing,
                                       preMatchData=preMatchData,
                                       shardSize=shardSize,
                                       groupCol=groupCol)

    datasets = createTrainTestDatasets(store=store,
                                       seqLen=seqLen,
                                       validation=valSplit > 0,
                                       trainTransform=trainTransform,
                                       group=groups)
    return createDataLoaders(tensorDict=datasets,
                             batchSize=batchSize,
                             numWorkers=numWorkers,
                             seed=seed,
                             pinMemory=pinMemory)

# def prepareData(df: pd.DataFrame|None=None, type: str|None=None, trainSplit: float=0.8, valSplit: float=0.2, save: bool=True,
#                 featureCols: List[str]|None=None, yCols: List[str]|str|None="result", seqLen: int=20,
#                 normMethod: str="standard", unkBucketDict: Dict[str, int]=UNKBUCKETDICT, 
#                 normaliserDir: str=NORMALISERDIR, normaliserJSON: str="numeric_normaliser.json",
#                 tokeniserDir: str=TOKENISERDIR, tensorDir: str=TENSORSDIR, batchSize: int=64,
#                 numWorkers: int=1, seed: int=42, pinMemory: bool=True, shuffle: bool|None=None,
#                 trainTransform: Transform|None=None, rememberMissing: bool=True, 
#                 preMatchData: Set[str]|None=PREMATCHDATACOLS) -> Dict[str, DataLoader]|DataLoader|None:
#     """set type=None to get a dict object containing test, train, and validation (if available) DataLoader objects"""
#     assert type is None or type in {"train", "test", "validation"}, 'type must be None or one of "train", "test", "validation"'
#     tensorDict = {}
#     types = [type] if type is not None else ["train", "test", "validation"]
#     for split in types:
#         transform = trainTransform if split == "train" else None
#         tensorDict[split] = MatchDataset.load(transform=transform, parentDir=tensorDir, fileDir=split)

#     if all(val is None for val in tensorDict.values()):
#         tensorDict = tensorDatasetsFromMatchDf(df=df, trainSplit=trainSplit, valSplit=valSplit, save=save, featureCols=featureCols,
#                                                yCols=yCols, seqLen=seqLen, normMethod=normMethod, unkBucketDict=unkBucketDict,
#                                                normaliserDir=normaliserDir, normaliserJSON=normaliserJSON, tokeniserDir=tokeniserDir,
#                                                tensorDir=tensorDir, trainTransform=trainTransform, rememberMissing=rememberMissing,
#                                                preMatchData=preMatchData)
#     if type is None:
#         if tensorDict["validation"] is None:
#             del tensorDict["validation"]
#         return createDataLoaders(tensorDict=tensorDict, batchSize=batchSize, numWorkers=numWorkers, seed=seed, pinMemory=pinMemory)
#     dataset = tensorDict[type]
#     shuffle = shuffle if shuffle is not None else type == "train"
#     return createDataLoader(dataset=dataset, batchSize=batchSize, shuffle=shuffle, 
#                             numWorkers=numWorkers, seed=seed, pinMemory=pinMemory) if dataset is not None else None