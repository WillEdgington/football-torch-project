from .visualise import plotBarForAgainst, plotTimeForAgainst, plotBarWinningStats, plotBarR2Stats, plotXYWithRegression

if __name__=="__main__":
    method = "ema"
    nameCol = "manager"
    valueCol = "xg"
    daysSinceFirst = 365*2
    # plotBarForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=daysSinceFirst,
    #                   daysSinceLast=30, method=method, minGames=20, getTopN=10)
    # plotTimeForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=daysSinceFirst,
    #                    daysSinceLast=30, method=method, minGames=20, getTopN=5)
    filterCol = "team"
    filter = "manchester united"

    plotBarWinningStats(showHomeAway=True, filterCol=filterCol, filter=filter, window=40, method="ema")
    plotBarR2Stats(getTopN=30, daysSinceFirst=daysSinceFirst, filterCol=filterCol, filter=filter)
    plotXYWithRegression(xCol="xg_for", yCol="goals_diff", daysSinceFirst=daysSinceFirst, filterCol=filterCol, filter=filter)