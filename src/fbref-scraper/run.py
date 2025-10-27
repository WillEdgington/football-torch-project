from scraper import scrapeLeaguesSeasonsFixturesData
from fetcher import FBRefFetcher
from config import DBNAME, DBDIR

if __name__=="__main__":
    fetcher = FBRefFetcher()
    scrapeLeaguesSeasonsFixturesData(fetcher=fetcher, fileDir=DBDIR, fileName=DBNAME, useDb=True)