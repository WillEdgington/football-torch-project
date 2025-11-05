from .visualise import plotBarForAgainst, plotTimeForAgainst

if __name__=="__main__":
    method = "ema"
    nameCol = "manager"
    valueCol = "xg"
    plotBarForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=365*2,
                      daysSinceLast=30, method=method, minGames=20)
    plotTimeForAgainst(nameCol=nameCol, valueCol=valueCol, window=40, daysSinceFirst=365*2,
                       daysSinceLast=30, method=method, minGames=20, getTopN=5)