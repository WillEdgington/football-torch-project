from .cleaner import cleanFbrefData
from .fetcher import FBRefFetcher
from .scraper import scrapeLeaguesSeasonsFixturesData, scrapeMatchData

if __name__ == "__main__":
    fetcher = FBRefFetcher()
    scrapeLeaguesSeasonsFixturesData(fetcher=fetcher, useDb=True)
    scrapeMatchData(fetcher=fetcher)
    cleanFbrefData(overwrite=True)
