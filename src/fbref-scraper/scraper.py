import os
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path

from config import LEAGUES, MATCHMETADATASCHEMA, DBRAWDIR, DBRAWNAME, METADATATABLE, MATCHTABLE
from fetcher import FBRefFetcher
from league_discovery import discoverSeasonURLs
from parser import parseSchedulePage, parseMatchPage
from database_objects import DatabaseWriter

def scrapeSeasonsFixturesData(league: str, fetcher: FBRefFetcher | None=None, limit: int=9, 
                              cachehtml: bool=True, mute: bool=False) -> List[Dict[str, str]]:
    assert limit >= 1, "You must scrape atleast one season"
    if fetcher == None:
        fetcher = FBRefFetcher()

    if not mute:
        print(f"Scraping {league}...")
    seasons = discoverSeasonURLs(league, fetcher=fetcher, cachehtml=cachehtml)

    allMatches = []

    for season, seasonURL in seasons[:limit][::-1]:
        if not mute:
            print(f"    Season: {season}")
        scheduleURL = seasonURL.replace(f"/{season}/", f"/{season}/schedule/") \
                                   .replace("-Stats", "-Scores-and-Fixtures")
        html = fetcher.fetch(scheduleURL, cache=cachehtml)
        matches = parseSchedulePage(html)
        for match in matches:
            match["league"] = league
            match["season"] = season
        allMatches += matches

    return allMatches

def scrapeLeaguesSeasonsFixturesData(fetcher: FBRefFetcher | None=None, fileDir: str=DBRAWDIR, 
                                     fileName: str=DBRAWNAME, save: bool=True, cachehtml: bool=True,
                                     useDb: bool=True, mute: bool=False) -> pd.DataFrame:
    filePath = Path(fileDir) / fileName
    if fetcher is None:
        fetcher = FBRefFetcher()

    allMatches = []

    for league in LEAGUES.keys():
        allMatches += scrapeSeasonsFixturesData(league, cachehtml=cachehtml, mute=mute)
    
    df = pd.DataFrame(allMatches)
    if save:
        if useDb:
            db = DatabaseWriter(dbDir=fileDir, dbName=fileName)
            db.createTable(tableName=METADATATABLE, schema=MATCHMETADATASCHEMA)
            db.insertMany(tableName=METADATATABLE, records=allMatches)
            db.close()
            if not mute:
                print(f"\nSAVED {len(df)} MATCHES TO: {db.dbPath}")
        else:    
            filePath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filePath, index=False)
            if not mute:
                print(f"\nSAVED {len(df)} MATCHES TO: {filePath}")
    return df

# Match data scraper

def createMatchSchema(scrapedMatches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    schema = {}
    for key in scrapedMatches[0].keys():
        schema[key] = {}
        if key in MATCHMETADATASCHEMA:
            schema[key]["type"] = MATCHMETADATASCHEMA[key]["type"]
            continue
        if key == "match_id":
            schema[key]["type"] = "INTEGER"
        else:
            schema[key]["type"] = "TEXT"

    return schema

def scrapeMatchData(fetcher: FBRefFetcher | None=None, dbDir: str=DBRAWDIR, 
                    dbName: str=DBRAWNAME, metadataName: str=METADATATABLE, savepoint: int=10, 
                    cachehtml: bool=False, mute: bool=False):
    assert savepoint > 0, "savepoint must be greater than 0."
    tableName = MATCHTABLE
    if fetcher is None:
        fetcher = FBRefFetcher()
    
    with DatabaseWriter(dbDir=dbDir, dbName=dbName) as db:
        metadata = db.selectCols('id', 'match_url', tableName=metadataName, orderby="id")
        totalMatches = len(metadata)
        if not mute:
            print(f"Found {totalMatches} matches in metadata table.")

        scrapedMatches = []
        scrapeCount = 0

        for matchTuple in metadata:
            scrapeCount += 1
            matchId, matchUrl = matchTuple[:2]
            if (matchUrl is None) or (db.rowExists(tableName=tableName, colValue=matchUrl, col="match_url")):
                continue
            
            if not mute:
                print(f"[{scrapeCount}/{totalMatches}] Scraping data from: {matchUrl}")
            
            html = fetcher.fetch(matchUrl, cache=cachehtml)
            matchdata = parseMatchPage(html, matchDict={})
            matchdata["match_url"] = matchUrl
            matchdata["match_id"] = matchId

            scrapedMatches.append(matchdata.copy())

            if scrapeCount % savepoint == 0:
                if scrapeCount == savepoint:
                    if not mute:
                        print(f"Creating table: {tableName} in DB...")
                    db.createTable(tableName=tableName, schema=createMatchSchema(scrapedMatches=scrapedMatches), overwrite=True)
                
                if not mute:
                    print(f"Saving {len(scrapedMatches)} matches to DB...")

                db.insertMany(tableName=tableName, records=scrapedMatches)
                scrapedMatches.clear()

        if scrapedMatches:
            if not mute:
                print(f"Saving final {len(scrapedMatches)} matches to DB...")
            db.insertMany(tableName=tableName, records=scrapedMatches)

    if not mute:
        print("Scraping complete.")