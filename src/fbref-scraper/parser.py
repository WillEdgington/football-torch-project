from bs4 import BeautifulSoup
from typing import Dict, List

from config import BASEURL, MATCHMETADATASCHEMA

def parseSchedulePage(html: str) -> List[Dict[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    matches = []
    table = soup.select_one("table.stats_table")
    if not table:
        return []
    
    for row in table.select("tbody tr"):
        if "class" in row.attrs and ("thead" in row["class"] or "spacer" in row["class"]):
            continue
        
        matchDict = {}
        for name, schemadict in MATCHMETADATASCHEMA.items():
            if name == "match_url":
                score = row.select_one("td[data-stat='score']")
                matchLink = score.select_one("a")["href"] if score and score.select_one("a") else None
                matchDict[name] = f"{BASEURL}{matchLink}" if matchLink else None
                continue
            stat = row.select_one(f"td[data-stat='{schemadict['data-stat']}']")
            matchDict[name] = stat.text.strip() if stat else None

        if not (matchDict["match_url"] and matchDict["date"] and matchDict["home_team"] and matchDict["away_team"]):
            continue

        matches.append(matchDict)
    
    return matches

def parseMatchPage(html: str):
    soup = BeautifulSoup(html, "html.parser")
    matchDict = {}

    # parse individual sections of page:
    # scorebox (goals, W-D-L form, xG, Manager, Captain, data/time, week, attendance, venue, officials)
    # field_wrap

