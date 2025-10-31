from bs4 import BeautifulSoup
from typing import List, Tuple

from config import BASEURL, LEAGUES
from fetcher import FBRefFetcher

def getLeagueHistoryURL(league: str) -> str:
    return f"{BASEURL}/en/comps/{LEAGUES[league]}/history/{league}-Seasons"

def discoverSeasonURLs(league: str, fetcher: FBRefFetcher | None=None, cachehtml: bool=True) -> List[Tuple[str, str]]:
    if fetcher is None:
        fetcher = FBRefFetcher()

    url = getLeagueHistoryURL(league)
    html = fetcher.fetch(url, cache=cachehtml)
    soup = BeautifulSoup(html, "html.parser")

    seasons = []
    table = soup.select_one("table#seasons")
    if not table:
        return []

    for row in table.select("tbody tr"):
        seasonTag = row.select_one("th a")
        if seasonTag and "href" in seasonTag.attrs:
            seasonName = seasonTag.text.strip()
            seasonURL = BASEURL + seasonTag["href"]
            seasons.append((seasonName, seasonURL))
    return seasons