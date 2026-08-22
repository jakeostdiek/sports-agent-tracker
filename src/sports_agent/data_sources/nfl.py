"""
NFL data source through nflreadpy (https://nflreadpy.nflverse.com/)

Normalizes nflreadpy to a common schema for downstream nodes

nflreadpy only supports filtering by season so functions here will load and filter down
"""

import pandas as pd
import nflreadpy as nfl

games_columns = ["game_id", "home_team", "away_team", "home_score", "away_score"]
stats_columns = ["player_id", "player_name", "team", "stat_category", "stat_value"]

stats_id_columns = [
    "player_id",
    "player_name",
    "player_display_name",
    "position",
    "position_group",
    "headshot_url",
    "season",
    "week",
    "season_type",
    "game_id",
    "team",
    "opponent_team",
]

def fetch_games(season: int, week: int) -> pd.DataFrame:
    schedule_p1 = nfl.load_schedules(season)
    schedule_p1 = schedule_p1.filter(schedule_p1["week"] == week)
    schedule_df = schedule_p1.to_pandas()

    if schedule_df.empty:
        return pd.DataFrame(columns=games_columns)

    out = schedule_df[["game_id", "home_team", "away_team", "home_score", "away_score"]]
    return out.reset_index(drop=True)

def fetch_player_stats(season: int, week: int) -> pd.DataFrame:
    stats_p1 = nfl.load_player_stats(season, summary_level="week")
    stats_p1 = stats_p1.filter(stats_p1["week"] == week)
    stats_df = stats_p1.to_pandas()

    if stats_df.empty:
        return pd.DataFrame(columns=stats_columns)

    stat_columns = [
        c
        for c in stats_df.columns
        if c not in stats_id_columns and pd.api.types.is_numeric_dtype(stats_df[c])
    ]

    melted = stats_df.melt(
        id_vars=["player_id", "player_name", "team"],
        value_vars=stat_columns,
        var_name="stat_category",
        value_name="stat_value",
    )

    # Drop zero/NaN stats like a QB's receptions
    melted = melted[melted["stat_value"].notna() & (melted["stat_value"] != 0)]

    return melted[stats_columns].reset_index(drop=True)