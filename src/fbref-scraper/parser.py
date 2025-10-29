import re

from bs4 import BeautifulSoup
from typing import Dict, List, Tuple

from config import BASEURL, MATCHMETADATASCHEMA
from fetcher import FBRefFetcher

def cleanText(text: str | None) -> str | None:
    if not text:
        return None
    return text.replace("\xa0", " ").strip()

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


# MATCH PAGE PARSERS

def parseMatchPage(html: str, matchDict: Dict[str, str | None]={}) -> Dict[str, str | None]:
    soup = BeautifulSoup(html, "html.parser")

    # parse individual sections of page:
    # class="scorebox" (goals, W-D-L form, xG, Manager, Captain, data/time, week, attendance, venue, officials)
    # id="field_wrap" (formation, line-up)
    # id="events_wrap" (match summary)
    # id="team_stats" (possesion, passing, shots, saves, cards)
    # id="team_stats_extra" (fouls, corners, crosses, touches, tackles, interceptions, aerials wond, clearances, offsides, goal kicks, throw ins, long balls)
    # id="all_player_stats_{team_hash}" (player stats of team)
    # id="all_keeper_stats_{team_hash}" (keeper stats of team)
    # id="all_shots" (shots)

    matchDict = parseScorebox(soup=soup, matchDict=matchDict)
    matchDict = parseTeamStats(soup=soup, matchDict=matchDict)
    matchDict = parseTeamStatsExtra(soup=soup, matchDict=matchDict)

    return matchDict

def parseScorebox(soup: BeautifulSoup, matchDict: Dict[str, str | None]={}) -> Dict[str, str | None]:
    scorebox = soup.select_one("div.scorebox")
    if not scorebox:
        return matchDict
    
    divs = scorebox.find_all("div", recursive=False)
    if len(divs) < 2:
        return matchDict
    homeDiv, awayDiv = divs[:2]

    def extractTeamInfo(teamDiv, prefix):
        teamNameTag = teamDiv.select_one("strong a")
        matchDict[f"{prefix}_team"] = cleanText(teamNameTag.text) if teamNameTag else None

        goalsTag = teamDiv.select_one(".scores .score")
        xgTag = teamDiv.select_one(".scores .score_xg")

        matchDict[f"{prefix}_goals"] = cleanText(goalsTag.text) if goalsTag else None
        matchDict[f"{prefix}_xg"] = cleanText(xgTag.text) if xgTag else None
        
        datapoints = teamDiv.select("div.datapoint")
        for dp in datapoints:
            labelTag = dp.select_one("strong")
            if not labelTag:
                continue
            label = labelTag.text.strip().lower()
            if "manager" in label:
                matchDict[f"{prefix}_manager"] = cleanText(dp.get_text(separator="", strip=True).replace("Manager:", ""))
            if "captain" in label:
                capTag = dp.select_one("a")
                matchDict[f"{prefix}_captain"] = cleanText(capTag.text.strip() if capTag else dp.get_text().replace("Captain:", ""))

    extractTeamInfo(teamDiv=homeDiv, prefix="home")
    extractTeamInfo(teamDiv=awayDiv, prefix="away")
    # extracting ".scorebox_meta" could be a future development; however, most of that data we already get from the fixture table

    return matchDict

def countCards(div: BeautifulSoup) -> Tuple[str | None, str | None]:
    if not div:
        return "0", "0"
    yellow = len(div.select(".yellow_card"))
    red = len(div.select(".red_card")) + len(div.select(".yellow_red_card"))
    return str(yellow), str(red)

def parseTeamStats(soup: BeautifulSoup, matchDict: Dict[str, str | None]={}) -> Dict[str, str | None]:
    statDiv = soup.select_one("#team_stats")
    statTable = statDiv.select_one("table") if statDiv else None
    statTr = statTable.find_all("tr", recursive=False)[1:] if statTable else []

    statLines = []
    for i in range(1, len(statTr), 2):
        statTd = statTr[i].find_all("td", recursive=False)
        statLines.append([statTd[j].select_one("div") for j in range(len(statTd))])

    if len(statLines) < 5:
        return matchDict
    
    statKeys = ["possession", ["passes", "pass_attempts"], ["sots", "shots"], "saves", ["yellow_cards", "red_cards"]]

    for i, key in enumerate(statKeys):
        for j, prefix in enumerate(["home", "away"]):
            if i == 0: # possession
                possession = cleanText(statLines[i][j].get_text(strip=True)) if statLines[i][j] else None
                possession = possession.replace("%", "") if possession else None
                matchDict[f"{prefix}_{key}"] = possession
                continue
            elif i == 4: # cards
                cards = countCards(statLines[i][j])
                matchDict[f"{prefix}_{key[0]}"] = cards[0]
                matchDict[f"{prefix}_{key[1]}"] = cards[1]
                continue

            text = cleanText(statLines[i][j].get_text(" ", strip=True)) if statLines[i][j] else None
            stats = re.findall(r"(\d+)", text if text else "")
            if prefix == "away": # percent is central (home->int% int%<-away)
                stats = stats[1:] + stats[:1]

            if isinstance(key, list): # not saves
                matchDict[f"{prefix}_{key[0]}"] = stats[0] if len(stats) >= 2 else None
                matchDict[f"{prefix}_{key[1]}"] = stats[1] if len(stats) >= 2 else None
                continue
            matchDict[f"{prefix}_{key}"] = stats[0] if len(stats) >= 1 else None
    
    return matchDict
    
def normalizeLabel(label: str) -> str:
    label = label.strip().lower().replace(" ", "_")
    label = re.sub(r"[^a-z0-9_]+", "", label)
    return label.strip("_")

def parseTeamStatsExtra(soup: BeautifulSoup, matchDict: Dict[str, str | None]={}) -> Dict[str, str | None]:
    statDiv = soup.select_one("#team_stats_extra")
    if not statDiv:
        return matchDict
    
    for groupDiv in statDiv.find_all("div", recursive=False):
        divs = groupDiv.find_all("div", recursive=False)[3:]
        for i in range(0, len(divs), 3):
            if i + 2 >= len(divs):
                continue

            homeVal = cleanText(divs[i].get_text(strip=True))
            label = cleanText(divs[i+1].get_text(strip=True))
            awayVal = cleanText(divs[i+2].get_text(strip=True))

            if not label:
                continue

            label = normalizeLabel(label)
            matchDict[f"home_{label}"] = homeVal
            matchDict[f"away_{label}"] = awayVal

    return matchDict

# fetcher = FBRefFetcher()
# html = fetcher.fetch(url="https://fbref.com/en/matches/dbcf5536/Swansea-City-West-Bromwich-Albion-December-9-2017-Premier-League")
# print(parseMatchPage(html))
# html2 = fetcher.fetch(url="https://fbref.com/en/matches/a562da13/Manchester-United-Chelsea-September-20-2025-Premier-League")
# print(parseMatchPage(html2))