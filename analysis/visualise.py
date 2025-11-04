import matplotlib.pyplot as plt
import numpy as np

from .prepare_data import prepareForAgainstDf, addRollingLeagueDevsAndDiff, getMostRecentRows

def plotScatterForAgainst(nameCol: str, valueCol: str, window: int=10, daysAgo: int|None=None,
                          minPeriods: int=1, getTopN: int=20, title: str="", calcForDays: bool=False):
    df = prepareForAgainstDf()
    df = addRollingLeagueDevsAndDiff(df=df, nameCol=nameCol, valueCol=valueCol,
                                     window=window, minPeriods=minPeriods, calcForDays=calcForDays)
    df = getMostRecentRows(df=df, nameCol=nameCol, daysAgo=daysAgo)
    df = df.sort_values(by=f"{valueCol}_diff_dev_roll_mean_{nameCol}", ascending=False)
    df = df.head(n=getTopN).reset_index(drop=True)

    plt.scatter(df[f"{valueCol}_for_dev_roll_mean_{nameCol}"], -df[f"{valueCol}_against_dev_roll_mean_{nameCol}"])
    for i, label in enumerate(df[nameCol]):
        plt.annotate(label,
                    (df[f"{valueCol}_for_dev_roll_mean_{nameCol}"][i], -df[f"{valueCol}_against_dev_roll_mean_{nameCol}"][i]),
                    )
    plt.xlabel(f"{valueCol} for")
    plt.ylabel(f"{valueCol} against")
    plt.title(title)
    plt.show()

def plotBarForAgainst(nameCol: str, valueCol: str, window: int=20, daysAgo: int|None=None,
                          minPeriods: int=1, getTopN: int=10, title: str="", calcForDays: bool=False):
    df = prepareForAgainstDf()
    df = addRollingLeagueDevsAndDiff(df=df, nameCol=nameCol, valueCol=valueCol,
                                     window=window, minPeriods=minPeriods, calcForDays=calcForDays)
    df = getMostRecentRows(df=df, nameCol=nameCol, daysAgo=daysAgo)
    df = df.sort_values(by=f"{valueCol}_diff_dev_roll_mean_{nameCol}", ascending=False)
    df = df.head(n=getTopN).reset_index(drop=True)

    x = np.arange(len(df))
    width = 0.25

    forMeans = df[f"{valueCol}_for_dev_roll_mean_{nameCol}"]
    againstMeans = -df[f"{valueCol}_against_dev_roll_mean_{nameCol}"]
    diffMeans = df[f"{valueCol}_diff_dev_roll_mean_{nameCol}"]
    valLabel = valueCol.replace("_", " ").capitalize()

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, forMeans, width, label=f"{valLabel} For", color="tab:green")
    plt.bar(x, againstMeans, width, label=f"{valLabel} Against", color="tab:red")
    plt.bar(x + width, diffMeans, width, label=f"{valLabel} Diff", color="tab:blue")

    plt.xticks(x, list(df[nameCol]), rotation=45, ha="right")
    plt.xlabel(nameCol.capitalize())
    plt.ylabel("Rolling League Deviation")
    plt.title(title or f"{valLabel} Rolling League Deviations (Top {getTopN})")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()