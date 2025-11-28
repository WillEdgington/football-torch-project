import pandas as pd

from typing import Tuple, List, Dict

from utils import prepareMatchDataFrame
from .tokeniser import Tokeniser
from .normaliser import Normaliser
from .config import UNKBUCKETDICT, TOKENISERDIR, NORMALISERDIR

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
              fileDir: str=NORMALISERDIR, fileName: str="numeric_normaliser.json") -> pd.DataFrame:
    assert method in {"standard", "minmax"}, 'invalid method input, Choose between: "standard", "minmax"'
    assert typeFilter is None or typeFilter in {"float64", "int64"}, 'typeFilter must be "float64", "int64", or None'
    typeFilters = {typeFilter} if typeFilter else {"float64", "int64"}

    if len(columns) == 0:
        for col in df.columns:
            if str(df[col].dtype) not in typeFilters:
                continue
            columns.append(col)

    with Normaliser(eps=eps, train=train, fileName=fileName, fileDir=fileDir) as nrm:
        for col in columns:
            if not col.startswith("home_"):
                continue
            base = col.removeprefix("home_")
            nrm.fit(pd.concat([
                df[f"home_{base}"], 
                df[f"away_{base}"]
                ]), col=base)

        for col in columns:
            if col not in df.columns:
                continue
            fit = True
            base = col
            if base.startswith("home_") or base.startswith("away_"):
                fit = False
                base = col.removeprefix("home_").removeprefix("away_")
            
            df[f"{col}_normalised"] = nrm.encodeSeries(df[col], col=base, method=method, fit=fit)
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

# df = prepareMatchDataFrame()
# print(f"Dataframe columns ({len(df.columns)}):\n{df.columns}")
# trainDf, testDf, valDf = getTemporalSplits(df, valSplit=0.2)

# print(f"(n-rows) train: {len(trainDf)}/{len(df)}, validation: {len(valDf)}/{len(df)}, test: {len(testDf)}/{len(df)}")
# trainDf = normalise(df=trainDf)
# print(trainDf.columns)
# trainDF = tokenise(df=trainDf, train=True)
# testDf = tokenise(df=testDf, train=False)
# print(testDf[testDf["league"] == "premier league"][testDf["home_manager_token"] <= 32][["date", "league", "home_team", "home_manager", "home_manager_token", "away_team"]])
# print(addDaysSinceLastGame(df=train)[["home_days_since_last_game", "away_days_since_last_game"]])
# print(matchDfToPerTeamDfs(df=df)["manchester united"].head(n=40)[["date", "home_team", "away_team", "home_goals", "away_goals"]])