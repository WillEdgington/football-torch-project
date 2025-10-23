import os
import pandas as pd
from typing import List, Dict
from pathlib import Path

from config import LEAGUES, BASEURL
from fetcher import FBRefFetcher
from league_discovery import discoverSeasonURLs
from parser import parseSchedulePage

def scrapeSeasonsFixturesData(league: str, fetcher: FBRefFetcher | None=None, limit: int=9, mute: bool=False) -> List[Dict[str, str]]:
    assert limit >= 1, "You must scrape atleast one season"
    if fetcher == None:
        fetcher = FBRefFetcher()

    if not mute:
        print(f"Scraping {league}...")
    seasons = discoverSeasonURLs(league)

    allMatches = []

    for season, seasonURL in seasons[:limit][::-1]:
        if not mute:
            print(f"    Season: {season}")
        scheduleURL = seasonURL.replace(f"/{season}/", f"/{season}/schedule/") \
                                   .replace("-Stats", "-Scores-and-Fixtures")
        html = fetcher.fetch(scheduleURL)
        matches = parseSchedulePage(html)
        for match in matches:
            match["league"] = league
            match["season"] = season
        allMatches += matches

    return allMatches

def scrapeLeaguesSeasonsFixturesData(fetcher: FBRefFetcher | None=None, fileDir: str="data/processed/", 
                                     fileName: str="match_metadata.csv", save: bool=True, mute: bool=False) -> pd.DataFrame:
    filePath = Path(fileDir) / fileName
    if fetcher == None:
        fetcher = FBRefFetcher()

    allMatches = []

    for league in LEAGUES.keys():
        allMatches += scrapeSeasonsFixturesData(league)
    
    df = pd.DataFrame(allMatches)
    if save:
        filePath.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filePath, index=False)
        print(f"\nSAVED {len(df)} MATCHES TO: {filePath}")
    
    return df