BASEURL = "https://fbref.com"
CACHEDIR = "data/raw/fbref-html"
RATELIMITSECONDS = 6

LEAGUES = {
    "Premier-League": "9", # england
    "Championship": "10", # england 2nd div
    "La-Liga": "12", # spain
    "Segunda-Division": "17", # spain 2nd div
    "Serie-A": "11", # italy
    "Serie-B": "18", # italy 2nd div
    "Bundesliga": "20", # germany
    "2-Bundesliga": "33", # germany 2nd div
    "Ligue-1": "13", # france
    "Ligue-2": "60", # france 2nd div
    "Belgian-Pro-League": "37", # belgium
    "Challenger-Pro-League": "69", # belgium 2nd div
    "Super-Lig": "26", # turkey
    "Primeira-Liga": "32", # portugal
    "Austrian-Bundesliga": "56", # austria
    "Danish-Superliga": "50", # denmark
    "Eredivisie": "23", # nederlands
    "Super-League-Greece": "27", # greece
    "Czech-First-League": "66", # czech
    "Scottish-Premiership": "40", # scotland
    "Ukrainian-Premier-League": "39", # ukraine
    "Serbian-SuperLiga": "54", # serbia
    "Hrvatska-NL": "63", # croatia
    "Swiss-Super-League": "57", # switzerland
    "A-League-Men": "65", # australia
    "Bulgarian-First-League": "67", # bulgaria
    "NB-I": "46", # hungary
    "Liga-MX": "31", # mexico
    "Saudi-Pro-League": "70", # saudi arabia
    "South-African-Premiership": "52", # south africa
    "Champions-League": "8", # uefa europe
    "Europa-League": "19", # uefa europe 2nd div
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