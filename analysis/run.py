from .visualise import plotBarForAgainst, plotTimeForAgainst, plotBarWinningStats, plotBarR2Stats

if __name__=="__main__":
    method = "ema"
    nameCol = "manager"
    valueCol = "xg"
    # plotBarForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=365*2,
    #                   daysSinceLast=30, method=method, minGames=20, getTopN=10)
    # plotTimeForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=365*2,
    #                    daysSinceLast=30, method=method, minGames=20, getTopN=5)
    filterCol = "team"
    filter = "brentford"

    plotBarWinningStats(showHomeAway=True, filterCol=filterCol, filter=filter, window=40, method="ema")
    plotBarR2Stats(getTopN=30, daysSinceFirst=365, filterCol=filterCol, filter=filter)