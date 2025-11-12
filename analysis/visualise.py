import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .prepare_data import (
                        prepareForAgainstDf, addRollingLeagueDevsAndDiff, getMostRecentRows, cutoffByDate, prepareMatchDataFrame,
                        getWinProbs, getLinearRegressionStats, getXYAndLinearRegression, getLogisticRegressionStats
                        )

def plotScatterForAgainst(nameCol: str, valueCol: str, window: int=10, daysAgo: int|None=None,
                          minGames: int=1, getTopN: int=20, title: str=""):
    df = prepareForAgainstDf()
    df = addRollingLeagueDevsAndDiff(df=df, nameCol=nameCol, valueCol=valueCol,
                                     window=window, minGames=minGames)
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

def plotBarForAgainst(nameCol: str, valueCol: str, window: int=20, daysSinceFirst: int|None=None, 
                      daysSinceLast: int|None=None, method: str="simple", minGames: int=1, getTopN: int=10, title: str=""):
    df = prepareForAgainstDf()
    if daysSinceFirst:
        df = cutoffByDate(df=df, daysAgo=daysSinceFirst)

    df = addRollingLeagueDevsAndDiff(df=df, nameCol=nameCol, valueCol=valueCol, window=window,
                                     minGames=minGames, method=method)
    df = getMostRecentRows(df=df, nameCol=nameCol, daysAgo=daysSinceLast)
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
    plt.title(title or f"{nameCol} {valLabel} Rolling League Deviations (Top {getTopN})")
    plt.legend()
    plt.tight_layout()
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.show()

def plotTimeForAgainst(nameCol: str, valueCol: str, window: int=20, daysSinceFirst: int|None=None, 
                      daysSinceLast: int|None=None, method: str="simple", minGames: int=1, getTopN: int=10, title: str=""):
    df = prepareForAgainstDf()
    if daysSinceFirst:
        df = cutoffByDate(df=df, daysAgo=daysSinceFirst)

    df = addRollingLeagueDevsAndDiff(df=df, nameCol=nameCol, valueCol=valueCol, window=window,
                                     minGames=minGames, method=method)
    recentDf = getMostRecentRows(df=df, nameCol=nameCol, daysAgo=daysSinceLast)
    recentDf = recentDf.sort_values(by=f"{valueCol}_diff_dev_roll_mean_{nameCol}", ascending=False)
    top = recentDf.head(n=getTopN).reset_index(drop=True)[nameCol].unique()

    for name in top:
        nameDf = df[df[nameCol] == name]
        plt.plot(nameDf["date"], nameDf[f"{valueCol}_diff_dev_roll_mean_{nameCol}"], label=name)
    
    plt.ylabel("Rolling League Deviation")
    plt.ylabel(f"{valueCol} Diff Rolling league Dev")
    plt.xlabel("Date")
    plt.title(f"{nameCol} {valueCol} Rolling Deviation Over Time (Top {getTopN})")
    plt.legend()
    plt.show()

def plotBarWinningStats(getTopN: int=10, showHomeAway: bool=False, filterCol: str|None=None, filter: str="", window: int|None=200, method: str="ema"):
    df = prepareMatchDataFrame()
    resultDf = getWinProbs(df=df, getTopN=getTopN, filterCol=filterCol, filter=filter, window=window, method=method)

    x = np.arange(len(resultDf))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    if showHomeAway:
        ax.bar(x - width/2, resultDf["home_win_prob"], width=width, label="Home")
        ax.bar(x + width/2, resultDf["away_win_prob"], width=width, label="Away")
        ax.legend()
    else:
        ax.bar(x, resultDf["win_prob"], width=width)

    filterLabel = "" if filterCol is None else f" for {filterCol} - {filter}"
    
    ax.set_xticks(x)
    ax.set_xticklabels(resultDf["stat"].str.replace("_", " "), rotation=45, ha="right")
    ax.set_ylabel("P(Win | Dominated Stat)")
    ax.set_title(f"How Match Stats Translate to Wins{filterLabel}")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def plotBarR2Stats(getTopN: int=15, daysSinceFirst: int|None=None, filterCol: str|None=None, filter: str="", relevanceThresh: float=0.1):
    df = prepareForAgainstDf()

    if daysSinceFirst:
        df = cutoffByDate(df=df, daysAgo=daysSinceFirst)
    
    regdf = getLinearRegressionStats(df=df, filterCol=filterCol, filter=filter).head(n=getTopN)
    regdf["sign"] = np.sign(regdf["coefficients"].apply(lambda x: x[0]))
    regdf["colour"] = regdf["sign"].map({1: "tab:blue", -1: "tab:red", 0:"gray"})

    filterLabel = "" if filterCol is None else f" for {filterCol} - {filter}"

    plt.figure(figsize=(10, 6))
    plt.bar(regdf["stat"].str.replace("_", " "), regdf["r2"], color=regdf["colour"])
    
    plt.axhline(y=relevanceThresh, color="black", linestyle="--", linewidth=1, alpha=0.5)
    plt.text(len(regdf) - 2.5, relevanceThresh + 0.005, f"R2 threshold = {relevanceThresh:.2f}", color="black", fontsize=9, va="bottom")
    
    plt.ylabel("R2 value")
    plt.title(f"Correlation of Match Stats with Goal Difference{filterLabel}")
    plt.xticks(rotation=45, ha="right")

    handles = [
        plt.Line2D([0], [0], color="tab:blue", lw=6, label="Positive correlation"),
        plt.Line2D([0], [0], color="tab:red", lw=6, label="Negative correlation"),
    ]
    plt.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()

def plotXYWithLinearRegression(xCol: str, yCol: str, daysSinceFirst: int|None=None, filterCol: str|None=None, filter: str=""):
    df = prepareForAgainstDf()

    if daysSinceFirst:
        df = cutoffByDate(df=df, daysAgo=daysSinceFirst)
    
    x, y, lrdict = getXYAndLinearRegression(df=df, xKey=xCol, yKey=yCol, filterCol=filterCol, filter=filter)
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=100, color="gray", alpha=0.6)

    plt.plot(x, lrdict["y_pred"], color="blue", linewidth=2, label=f"f(X)={lrdict['coefficients'][0][0]:.1f}X + {lrdict['intercept'][0]:.1f}")
    
    xLab, yLab = xCol.replace("_", " ").title(), yCol.replace("_", " ").title()
    filterLabel = "" if filterCol is None else f" for {filterCol} - {filter}"
    plt.xlabel(xLab)
    plt.ylabel(yLab)
    plt.title(f"{xLab} vs {yLab}{filterLabel}")
    

    text = f"R2 = {lrdict['r2']:.3f}\nMSE = {lrdict['mse']:.3f}\n"
    plt.text(0.05, 0.95, text, transform=plt.gca().transAxes,
            fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    plt.legend()
    plt.tight_layout()
    plt.show()