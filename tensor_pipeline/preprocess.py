import pandas as pd

from typing import Tuple

from utils import prepareMatchDataFrame
from .tokeniser import Tokeniser

def tokenise(df: pd.DataFrame, train: bool=True, fileDir: str="saved_tokenisers") -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if df[col].dtype != "object" or col == "match_url":
            continue
        fileName = f"{col}_tokeniser.json"
        if col.startswith("home_") or col.startswith("away_"):
            fileName = fileName.removeprefix("home_").removeprefix("away_")
        
        with Tokeniser(train=train, fileName=fileName, fileDir=fileDir) as tkn:
            df[f"{col}_token"] = tkn.encodeSeries(df[col])

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

# df = prepareMatchDataFrame()
# print(f"Dataframe columns ({len(df.columns)}):\n{df.columns}")
# train, test, val = getTemporalSplits(df, valSplit=0.2)

# print(f"(n-rows) train: {len(train)}/{len(df)}, validation: {len(val)}/{len(df)}, test: {len(test)}/{len(df)}")