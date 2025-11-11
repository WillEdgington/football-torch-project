import pandas as pd
import numpy as np
import unicodedata

from typing import Dict, Any, List
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

from fbref_scraper import DatabaseReader, DBDIR, DBNAME, MATCHTABLE

def removeAccents(text: str):
    if not isinstance(text, str):
        return text
    normalised = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalised if not unicodedata.combining(c))

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
                df[col] = df[col].apply(removeAccents).str.lower()
                df[col] = df[col].str.replace("-", " ")
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
                  method: str="simple") -> pd.DataFrame:
    assert window >= 1, "window must be greater than zero."
    df = df.sort_values([nameCol, "date"]).copy()

    grouped = df.groupby(nameCol)[valueCol]
    newCol = f"{valueCol}_roll_mean_{nameCol}"

    if method == "ema":
        df[newCol] = grouped.transform(lambda x: x.ewm(span=window, adjust=False).mean())
    elif method == "gaussian":
        df[newCol] = grouped.transform(lambda x: x.rolling(window=window, win_type=method).mean(std=window/4))
    else:
        df[newCol] = grouped.transform(lambda x: x.rolling(window=window).mean())

    return df

def addRollingLeagueDeviation(df: pd.DataFrame, nameCol: str, valueCol: str, window: int=20, 
                              method: str="simple") -> pd.DataFrame:
    df = df.copy()
    df = addRollingAvg(df=df, nameCol="league", valueCol=valueCol, window=2 * 10, method="simple") # last 10 matches (1 match week)
    df[f"{valueCol}_dev"] = (
        df[f"{valueCol}"] / df[f"{valueCol}_roll_mean_league"] - 1
    )
    df = addRollingAvg(df=df, nameCol=nameCol, valueCol=f"{valueCol}_dev", window=window, 
                       method=method)
    return df

def addRollingLeagueDevsAndDiff(df: pd.DataFrame, nameCol: str, valueCol: str, window: int=20, 
                                method: str="simple", minGames:int=1) -> pd.DataFrame:
    df = df.copy()
    df = addRollingLeagueDeviation(df=df, nameCol=nameCol, valueCol=f"{valueCol}_for", window=window, method=method)
    df = addRollingLeagueDeviation(df=df, nameCol=nameCol, valueCol=f"{valueCol}_against", window=window, method=method)
    df[f"{valueCol}_diff_dev_roll_mean_{nameCol}"] = (
        df[f"{valueCol}_for_dev_roll_mean_{nameCol}"] - df[f"{valueCol}_against_dev_roll_mean_{nameCol}"]
    )
    return df.groupby(by=nameCol).filter(func=lambda g: len(g) >= minGames)

def cutoffByDate(df: pd.DataFrame, daysAgo: int=0, weeksAgo: int=0) -> pd.DataFrame:
    assert daysAgo >= 0 and weeksAgo >= 0, "daysAgo must be positive"
    df = df.copy()
    cutoffdate = pd.Timestamp.now() - pd.Timedelta(days=daysAgo, weeks=weeksAgo)
    return df[df["date"] >= cutoffdate]

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

def getMatchStatsKeys(df: pd.DataFrame) -> List[str]:
    stats = []
    for col in df.columns:
        if df[col].dtype == "object" or not col.startswith("home_"):
            continue
        stats.append(col.removeprefix("home_"))
    
    return stats

def getMeans(series: pd.Series, window: int|None=None, method: str="simple") -> pd.Series:
    if window is None:
        window = len(series)
    
    match method:
        case "ema":
            return series.ewm(span=window, adjust=False).mean()
        case "gaussian":
            return series.rolling(window=window, win_type=method).mean(std=window/4)
        case _:
            return series.rolling(window=window).mean()

def winProbGivenDominance(df: pd.DataFrame, stat: str, window: int|None=None, method: str="simple") -> Dict[str, Any]:
    df = df.copy()
    df.sort_values("date", inplace=True)
    statDiff = df[f"home_{stat}"] - df[f"away_{stat}"]

    homeWinsWhenDom = (df["home_goals"] > df["away_goals"]) & (statDiff > 0)
    awayWinsWhenDom = (df["away_goals"] > df["home_goals"]) & (statDiff < 0)

    homeDoms = statDiff > 0
    awayDoms = statDiff < 0

    homeProbSeries = getMeans(series=homeWinsWhenDom.astype(int), window=window, method=method)
    awayProbSeries = getMeans(series=awayWinsWhenDom.astype(int), window=window, method=method)

    homeDomSeries = getMeans(series=homeDoms.astype(int), window=window, method=method)
    awayDomSeries = getMeans(series=awayDoms.astype(int), window=window, method=method)

    homeProb = (homeProbSeries / homeDomSeries).iloc[-1]
    awayProb = (awayProbSeries / awayDomSeries).iloc[-1]

    return {"stat": stat, "win_prob": (homeProb + awayProb) / 2, "home_win_prob": homeProb, "away_win_prob": awayProb}

def getWinProbs(df: pd.DataFrame, getTopN: int=10, filterCol: str|None=None, filter: str="", window: int|None=None, method: str="simple") -> pd.DataFrame:
    stats = getMatchStatsKeys(df=df)
    if filterCol and f"home_{filterCol}" in df.columns:
        homeDf = df[df[f"home_{filterCol}"] == filter]
        awayDf = df[df[f"away_{filterCol}"] == filter]
        homeResults = [{"stat": stat, "win_prob": winProbGivenDominance(homeDf, stat, window=window, method=method)["home_win_prob"]} for stat in stats if stat != "goals"]
        awayResults = [{"stat": stat, "win_prob": winProbGivenDominance(awayDf, stat, window=window, method=method)["away_win_prob"]} for stat in stats if stat != "goals"]
        results = [
            {"stat": homeResults[i]["stat"], 
             "win_prob": (homeResults[i]["win_prob"] + awayResults[i]["win_prob"]) / 2,
             "home_win_prob": homeResults[i]["win_prob"],
             "away_win_prob": awayResults[i]["win_prob"] 
            } for i in range(len(homeResults))
            ]
    else:
        df = df[df[filterCol] == filter] if filterCol is not None else df
        results = [winProbGivenDominance(df, stat, window=window) for stat in stats if stat != "goals"]
    resultDf = pd.DataFrame(results)
    resultDf = resultDf.sort_values(by="win_prob", ascending=False).head(n=getTopN)
    return resultDf

def prepareLinearRegression(x, y) -> Dict[str, Any]:
    model = LinearRegression().fit(x, y)
    yPred = model.predict(x)
    
    return {
        "r2": r2_score(y, yPred),
        "mse": mean_squared_error(y, yPred),
        "intercept": model.intercept_,
        "coefficients": model.coef_,
        "y_pred": yPred,
        "residuals": y - yPred
    }

def getRegressionStats(df: pd.DataFrame, stats: List[str]|None=None, yKey: str="goals_diff", trackSufX: List[str]=["diff", "for", "against"],
                       filterCol: str|None=None, filter: str="") -> pd.DataFrame:
    for suf in trackSufX:
        assert suf in {"for", "against", "diff"}, f'invalid suffix to track for x in trackSufX ({suf}). "diff", "for", "against" are the only valid entries.'
    
    df = df.copy()    
    if filterCol:
        df = df[df[filterCol] == filter]

    if yKey not in df.columns and yKey.endswith("_diff"):
        baseKey = yKey.removesuffix("_diff")
        df[yKey] = df[f"{baseKey}_for"] - df[f"{baseKey}_against"] 

    assert yKey in df.columns, f"key given for y column ({yKey}) could not be found in the given dataframe."
    
    if stats is None:
        stats = [col.removesuffix("_for") for col in df.columns if col.endswith("_for")]

    stats = [stat for stat in stats if f"{stat}_for" in df.columns and f"{stat}_against" in df.columns]
    results = []

    for stat in stats:
        filteredDf = df.dropna(subset=[f"{stat}_for", f"{stat}_against"])
        y = filteredDf[yKey].values.reshape(-1, 1)
        for suf in trackSufX:
            if f"{stat}_{suf}" == yKey:
                continue
            if suf == "diff":
                x = (filteredDf[f"{stat}_for"] - filteredDf[f"{stat}_against"]).values.reshape(-1, 1)
            else:
                x = (filteredDf[f"{stat}_{suf}"]).values.reshape(-1, 1)
            lrdict = prepareLinearRegression(x=x, y=y)
            lrdict["stat"] = f"{stat}_{suf}"
            results.append(lrdict)
            
    return pd.DataFrame(results).sort_values(by="r2", ascending=False)