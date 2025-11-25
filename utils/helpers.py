import pandas as pd

from typing import List

from fbref_scraper import DatabaseReader, DBDIR, DBNAME, MATCHTABLE

def prepareMatchDataFrame(dbDir: str=DBDIR, dbName: str=DBNAME, 
                          dropNArows: List[str]|None=["home_goals", "away_goals"], 
                          dropCols: List[str]|None=["id", "match_url"]) -> pd.DataFrame:
    with DatabaseReader(dbDir=dbDir, dbName=dbName) as db:
        df = db.selectAll(tableName=MATCHTABLE, asDf=True)

    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            if col == "date" or df[col].dtype != "object":
                continue

            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                if col not in {"match_url", "season"}:
                    df[col] = df[col].str.replace("-", " ")
                    df[col] = df[col].str.replace("   ", " - ")
                continue
        
        df["date"] = pd.to_datetime(df["date"])
        
        if dropNArows:
            df.dropna(subset=dropNArows, inplace=True)
        if dropCols:
            df.drop(columns=dropCols, inplace=True)
        df.sort_values(by='date', inplace=True)
        return df
    return pd.DataFrame()