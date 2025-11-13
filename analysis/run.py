from .visualise import plotBarForAgainst, plotTimeForAgainst, plotBarWinningStats, plotBarR2Stats, plotXYWithLinearRegression, plotOddsRatioStats

if __name__=="__main__":
    method = "ema"
    nameCol = "manager"
    valueCol = "xg"
    daysSinceFirst = 365*2
    plotBarForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=daysSinceFirst,
                      daysSinceLast=30, method=method, minGames=20, getTopN=10)
    plotTimeForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=daysSinceFirst,
                       daysSinceLast=30, method=method, minGames=20, getTopN=5)
    filterCol = "team"
    filter = "liverpool"
    yKey = "win"

    plotBarWinningStats(showHomeAway=True, filterCol=filterCol, filter=filter, window=40, method="ema")
    plotBarR2Stats(getTopN=30, daysSinceFirst=daysSinceFirst, filterCol=filterCol, filter=filter)
    plotXYWithLinearRegression(xCol="xg_for", yCol="goals_diff", daysSinceFirst=daysSinceFirst, filterCol=filterCol, filter=filter)
    plotOddsRatioStats(getMostSigN=20, daysSinceFirst=daysSinceFirst, filterCol=filterCol, filter=filter, yKey="win", pSigLevel=0.05)