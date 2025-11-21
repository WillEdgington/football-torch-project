import pandas as pd

from fbref_scraper import DatabaseReader, DBDIR, DBNAME, MATCHTABLE

def prepareMatchDataFrame(dbDir: str=DBDIR, dbName: str=DBNAME) -> pd.DataFrame:
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
        df.sort_values(by='date')
        return df
    return pd.DataFrame()