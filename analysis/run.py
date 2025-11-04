from .visualise import plotBarForAgainst

if __name__=="__main__":
    nameCol = "manager"
    valueCol = "xg"
    plotBarForAgainst(nameCol=nameCol, valueCol=valueCol, daysAgo=30, minPeriods=20)