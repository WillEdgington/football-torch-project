import pandas as pd
import numpy as np
import unicodedata

from typing import Tuple, List, Dict, Any

from .config import DBRAWDIR, DBRAWNAME, DBDIR, DBNAME, METADATATABLE, MATCHTABLE
from .database_objects import DatabaseReader, DatabaseWriter

def removeAccents(text: str):
    if not isinstance(text, str):
        return text
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalised if not unicodedata.combining(c))

def loadRawData(dbDir: str=DBRAWDIR, dbName: str=DBRAWNAME, asDf: bool=True) -> Tuple[List[Any] | pd.DataFrame, List[Any] | pd.DataFrame]:
    with DatabaseReader(dbDir=dbDir, dbName=dbName) as db:
        meta = db.selectAll(tableName=METADATATABLE, orderby="id", asDf=asDf)
        stats = db.selectAll(tableName=MATCHTABLE, orderby="match_id", asDf=asDf)
    return meta, stats

def processMetaDataframe(metaDf: pd.DataFrame, dropCols: List[str]=[]) -> pd.DataFrame:
    if dropCols:
        metaDf = metaDf.drop(columns=[c for c in dropCols if c in metaDf.columns])

    for col in metaDf.columns:
        if metaDf[col].dtype != "object":
            continue
        
        if col == "time":
            timeSplit = metaDf["time"].astype(str).str.split(":", expand=True)
            timeSplit.columns = ["hours", "minutes"]

            timeSplit = timeSplit.apply(pd.to_numeric)

            metaDf["time"] = timeSplit["hours"] * 60 + timeSplit["minutes"]
            continue

        if col == "date":
            # metaDf["date"] = pd.to_datetime(metaDf["date"])
            continue

        metaDf[col] = (
            metaDf[col].astype(str)
            .str.replace(",", "", regex=False)
            .replace("None", "")
            .apply(removeAccents)
            .str.lower()
            .str.replace("-", " ")
        )
        try:
            metaDf[col] = pd.to_numeric(metaDf[col])
        except ValueError:
            continue

    return metaDf

def processStatsDataframe(statsDf: pd.DataFrame, dropCols: List[str]=[]) -> pd.DataFrame:
    if dropCols:
        statsDf = statsDf.drop(columns=[c for c in dropCols if c in statsDf.columns])
    
    statsDf["home_goals"] = pd.to_numeric(statsDf["home_goals"], errors="coerce")
    statsDf["away_goals"] = pd.to_numeric(statsDf["away_goals"], errors="coerce")

    for col in statsDf.columns:
        if statsDf[col].dtype != "object":
            continue
        
        statsDf[col] = (
            statsDf[col].astype(str)
            .str.replace(",", "", regex=False)
            .replace("None", "")
            .apply(removeAccents)
            .str.lower()
            .str.replace("-", " ")
        )
        try:
            statsDf[col] = pd.to_numeric(statsDf[col])
        except ValueError:
            continue
    
    return statsDf

def transformDataframes(metaDf: pd.DataFrame, statsDf: pd.DataFrame) -> pd.DataFrame:
    if metaDf["match_url"].duplicated().any() or statsDf["match_url"].duplicated().any():
        raise ValueError("Duplicate match_id detected.")
    
    dropFromMeta = list(statsDf.drop(columns=["match_url"]).columns) + ["score"]
    metaDf = processMetaDataframe(metaDf=metaDf, dropCols=dropFromMeta)
    statsDf = processStatsDataframe(statsDf=statsDf)
    
    df = metaDf.merge(statsDf, on="match_url", how="inner", validate="one_to_one").dropna(subset=["home_goals", "away_goals"])
    return df

def getSchemaFromDf(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    schema = {
        str(col): {"type": "REAL" if np.issubdtype(dtype, np.number) else "TEXT"}
        for col, dtype in df.dtypes.items()
    }
    return schema

def writeCleanDfToDB(df: pd.DataFrame, tableName: str=MATCHTABLE, dbDir: str=DBDIR, 
                     dbName: str=DBNAME, overwrite: bool=False, mute: bool=False):
    with DatabaseWriter(dbDir=dbDir, dbName=dbName) as db:
        schema = getSchemaFromDf(df)
        db.createTable(tableName=tableName, schema=schema, overwrite=overwrite)
        db.insertMany(tableName=tableName, records=df.to_dict(orient='records'))
        if not mute:
            print(f"Cleaned data saved to.\n    Path: {db.dbPath}\n    Table name: {tableName}\nclosing writer...")

def cleanFbrefData(rawDir: str=DBRAWDIR, rawName: str=DBRAWNAME, 
                   cleanDir: str=DBDIR, cleanName: str=DBNAME, 
                   mute: bool=False, overwrite: bool=False) -> pd.DataFrame:
    metaDf, statsDf = loadRawData(dbDir=rawDir, dbName=rawName, asDf=True)
    cleanDf = transformDataframes(metaDf=metaDf, statsDf=statsDf)
    writeCleanDfToDB(df=cleanDf, dbDir=cleanDir, dbName=cleanName, mute=mute, overwrite=overwrite)

    return cleanDf