from bs4 import BeautifulSoup
from typing import Dict, List

from config import BASEURL

def parseSchedulePage(html: str) -> List[Dict[str, str | None]]:
    soup = BeautifulSoup(html, "html.parser")
    matches = []
    table = soup.select_one("table.stats_table")
    if not table:
        return []
    
    for row in table.select("tbody tr"):
        if "class" in row.attrs and ("thead" in row["class"] or "spacer" in row["class"]):
            continue
        
        # possible data-stat: 'gameweek', 'dayofweek', 'date', 'start_time', 'home_team', 'home_xg' (2017-2018 upwards for top leagues), 'score',
        # 'away_xg' (same availability as home_xg), 'away_team', 'attendance', 'venue', 'referee', 'match_report' (anchor tag), 'notes'
        date = row.select_one("td[data-stat='date']")
        home = row.select_one("td[data-stat='home_team']")
        score = row.select_one("td[data-stat='score']")
        away = row.select_one("td[data-stat='away_team']")
        matchLink = score.select_one("a")["href"] if score and score.select_one("a") else None

        if not (date and home and away):
            continue

        matches.append({
            "date": date.text.strip(),
            "date": date.text.strip(),
            "home_team": home.text.strip(),
            "away_team": away.text.strip(),
            "score": score.text.strip() if score else None,
            "match_url": f"https://fbref.com{matchLink}" if matchLink else None
        })
    
    return matches