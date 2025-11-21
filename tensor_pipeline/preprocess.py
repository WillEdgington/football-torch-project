import pandas as pd

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