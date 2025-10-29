from scraper import scrapeLeaguesSeasonsFixturesData, scrapeMatchData
from fetcher import FBRefFetcher

if __name__=="__main__":
    fetcher = FBRefFetcher()
    # scrapeLeaguesSeasonsFixturesData(fetcher=fetcher, useDb=True)
    scrapeMatchData(fetcher=fetcher)