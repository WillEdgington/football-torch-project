import pandas as pd
import unicodedata

from fbref_scraper import DatabaseReader, DBDIR, DBNAME, MATCHTABLE

def removeAccents(text: str):
    if not isinstance(text, str):
        return text
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalised if not unicodedata.combining(c))

def prepareMatchDataFrame(dbDir: str=DBDIR, dbName: str=DBNAME, normalizeNames: bool=True) -> pd.DataFrame:
    with DatabaseReader(dbDir=dbDir, dbName=dbName) as db:
        df = db.selectAll(tableName=MATCHTABLE, asDf=True)
    
    if isinstance(df, pd.DataFrame):
        for col in df.columns:
            if col == "date" or df[col].dtype != "object":
                continue

            try:
                df[col] = pd.to_numeric(df[col])
            except ValueError:
                df[col] = df[col].apply(removeAccents).str.lower()
                continue

        df["date"] = pd.to_datetime(df["date"])
        df.sort_values(by='date')
        return df
    return pd.DataFrame()

def prepareForAgainstDf(df: pd.DataFrame=prepareMatchDataFrame()) -> pd.DataFrame:
    homeCols = [col for col in df.columns if col.startswith("home_")]
    awayCols = [col for col in df.columns if col.startswith("away_")]

    homeRename, awayRename = {}, {}

    for col in homeCols:
        base = col.removeprefix("home_")
        if df[col].dtype == "object":
            homeRename[col] = base
            awayRename[col] = f"opposing_{base}"
            continue

        homeRename[col] = f"{base}_for"
        awayRename[col] = f"{base}_against"
    
    for col in awayCols:
        base = col.removeprefix("away_")
        if df[col].dtype == "object":
            awayRename[col] = base
            homeRename[col] = f"opposing_{base}"
            continue

        awayRename[col] = f"{base}_for"
        homeRename[col] = f"{base}_against"
    
    homeDf = df.rename(columns=homeRename)
    awayDf = df.rename(columns=awayRename)

    return pd.concat([homeDf, awayDf], ignore_index=True).sort_values(by="date")

def focalForAgainstDf(*calcDifFor: str, df: pd.DataFrame, colKey: str, name: str) -> pd.DataFrame:
    df = df.copy()
    if "home_team" in df.columns:
        df = prepareForAgainstDf(df)

    df = df.loc[df[colKey].apply(removeAccents).str.lower() == removeAccents(name).lower()]

    for col in calcDifFor:
        if f"{col}_for" not in df.columns or f"{col}_against" not in df.columns or f"{col}_diff" in df.columns:
            continue
        df.dropna(subset=[f"{col}_for", f"{col}_against"], inplace=True)
        df[f"{col}_diff"] = df[f"{col}_for"] - df[f"{col}_against"]
    
    return df

def getFocalMeans(df:pd.DataFrame, nameKey: str, valKey: str) -> pd.DataFrame:
    calcDifFor = ""
    if valKey.endswith("_diff"):
        calcDifFor = valKey.removesuffix("_diff")
    means = []
    for name in df[nameKey].unique():
        focalDf = focalForAgainstDf(calcDifFor, df=df, colKey=nameKey, name=name)
        if len(focalDf[nameKey]) < 20:
            continue
        means.append({"manager": name,
                      f"{valKey}_mean": focalDf[valKey].mean()})
    
    return pd.DataFrame(means).dropna(subset=f"{valKey}_mean")

def addRollingAvg(df: pd.DataFrame, nameCol: str, valueCol: str, window: int=20, 
                  minPeriods: int=1, calcForDays: bool=False) -> pd.DataFrame:
    assert window >= 1, "window must be greater than zero."
    df = df.sort_values([nameCol, "date"]).copy()

    if calcForDays:
        window = pd.Timedelta(days=window)

    df[f"{valueCol}_roll_mean_{nameCol}"] = (
        df.groupby(nameCol)[valueCol]
        .rolling(window=window, min_periods=minPeriods)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df.dropna(subset=[f"{valueCol}_roll_mean_{nameCol}"], inplace=True)
    return df

def addRollingLeagueDeviation(df: pd.DataFrame, nameCol: str, valueCol: str, window: int=20, minPeriods: int=1, calcForDays: bool=False) -> pd.DataFrame:
    df = df.copy()
    df = addRollingAvg(df=df, nameCol="league", valueCol=valueCol, window=window * 20, minPeriods=minPeriods * 20, calcForDays=calcForDays)
    df[f"{valueCol}_dev"] = (
        df[f"{valueCol}"] / df[f"{valueCol}_roll_mean_league"] - 1
    )
    df = addRollingAvg(df=df, nameCol=nameCol, valueCol=f"{valueCol}_dev", window=window, minPeriods=minPeriods, calcForDays=calcForDays)

    return df

def addRollingLeagueDevsAndDiff(df: pd.DataFrame, nameCol: str, valueCol: str, window: int=20, minPeriods:int=1, calcForDays: bool=False) -> pd.DataFrame:
    df = df.copy()
    df = addRollingLeagueDeviation(df=df, nameCol=nameCol, valueCol=f"{valueCol}_for", window=window, minPeriods=minPeriods, calcForDays=calcForDays)
    df = addRollingLeagueDeviation(df=df, nameCol=nameCol, valueCol=f"{valueCol}_against", window=window, minPeriods=minPeriods, calcForDays=calcForDays)
    df[f"{valueCol}_diff_dev_roll_mean_{nameCol}"] = (
        df[f"{valueCol}_for_dev_roll_mean_{nameCol}"] - df[f"{valueCol}_against_dev_roll_mean_{nameCol}"]
    )
    return df

def getMostRecentRows(df: pd.DataFrame, nameCol: str, daysAgo: int|None=None) -> pd.DataFrame:
    df = df.copy()
    df = (
        df.sort_values(by=[nameCol, "date"], ascending=[True, False])
        .groupby(nameCol, as_index=False)
        .first()
    )
    if daysAgo is not None:
        cutoffdate = pd.Timestamp.now() - pd.Timedelta(days=daysAgo)
        return df[df["date"] >= cutoffdate]
    return df