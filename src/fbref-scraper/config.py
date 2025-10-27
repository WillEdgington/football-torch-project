BASEURL = "https://fbref.com"
CACHEDIR = "data/raw/fbref-html"
RATELIMITSECONDS = 6

LEAGUES = {
    "Premier-League": "9",
    "La-Liga": "12",
    "Serie-A": "11",
    "Bundesliga": "20",
    "Ligue-1": "13"
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

DBDIR = "data/processed"
DBNAME = "fbref_data.db"