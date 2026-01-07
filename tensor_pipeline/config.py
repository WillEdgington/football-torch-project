UNKBUCKETDICT = {
    "captain": 16,
    "day": 1,
    "league": 4,
    "manager": 32,
    "referee": 32,
    "season": 8,
    "team": 32,
    "venue": 32
}

PREMATCHDATACOLS = {
    "day_token",
    "venue_token",
    "league_token",
    "season_token",
    "home_team_token",
    "away_team_token",
    "match_week_normalised",
    "time_normalised",
    "home_days_since_last_game_normalised",
    "away_days_since_last_game_normalised"
}

TOKENISERDIR = "tensor_pipeline/saved_tokenisers"
NORMALISERDIR = "tensor_pipeline/saved_normalisers"

TENSORSDIR = "data/tensors"
TRAINDATADIR = TENSORSDIR + "/train"
VALDATADIR = TENSORSDIR + "/val"
TESTDATADIR = TENSORSDIR + "/test"