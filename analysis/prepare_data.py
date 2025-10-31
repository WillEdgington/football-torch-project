import pandas as pd

from fbref_scraper import DatabaseReader, DBDIR, DBNAME, MATCHTABLE

def prepareMatchDataFrame(dbDir: str=DBDIR, dbName: str=DBNAME) -> pd.DataFrame:
    with DatabaseReader(dbDir=dbDir, dbName=dbName) as db:
        df = db.selectAll(tableName=MATCHTABLE, asDf=True)
    
    if isinstance(df, pd.DataFrame):
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by='date')
        return df
    return pd.DataFrame()