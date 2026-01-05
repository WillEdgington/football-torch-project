BASEURL = "https://fbref.com"
CACHEDIR = "data/raw/fbref-html"
RATELIMITSECONDS = 6

LEAGUES = {
    "Premier-League": "9", # england
    "Championship": "10",
    "La-Liga": "12", # spain
    "Segunda-Division": "17",
    "Serie-A": "11", # italy
    "Serie-B": "18",
    "Bundesliga": "20", # germany
    "2-Bundesliga": "33",
    "Ligue-1": "13", # france
    "Ligue-2": "60",
    "Belgian-Pro-League": "37", # belgium
    "Challenger-Pro-League": "69",
    "Super-Lig": "26", # turkey (no data for 2nd div)
    "Primeira-Liga": "32", # portugal (no data for 2nd div)
}

MATCHMETADATASCHEMA = {
    "match_week": {"data-stat": "gameweek", "type": "TEXT"},
    "day": {"data-stat": "dayofweek", "type": "TEXT"},
    "date": {"data-stat": "date", "type": "TEXT"},
    "time": {"data-stat": "start_time", "type": "TEXT"},
    "home_team": {"data-stat": "home_team", "type": "TEXT"},
    "home_xg": {"data-stat": "home_xg", "type": "TEXT"},
    "score": {"data-stat": "score", "type": "TEXT"},
    "away_xg": {"data-stat": "away_xg", "type": "TEXT"},
    "away_team": {"data-stat": "away_team", "type": "TEXT"},
    "attendance": {"data-stat": "attendance", "type": "TEXT"},
    "venue": {"data-stat": "venue", "type": "TEXT"},
    "referee": {"data-stat": "referee", "type": "TEXT"},
    "match_url": {"data-stat": None, "type": "TEXT UNIQUE"}
}

METADATATABLE = "match_metadata"
MATCHTABLE = "match_data"

DBRAWDIR = "data/raw"
DBRAWNAME = "fbref_data_raw.db"

DBDIR = "data/processed"
DBNAME = "fbref_data.db"