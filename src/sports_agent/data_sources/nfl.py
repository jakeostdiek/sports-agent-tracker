"""
NFL data source, backed by nflreadpy (https://nflreadpy.nflverse.com/),
which pulls from the nflverse-data project. Free, no API key required.

Normalizes nflreadpy's output to the common schema every downstream node
expects:

    fetch_games ->        game_id, home_team, away_team, home_score, away_score
    fetch_player_stats -> player_id, player_name, team, stat_category, stat_value

nflreadpy returns Polars DataFrames and only supports filtering by season
(not week) at the load level, so both functions here load the full season
and filter down to the requested week themselves.
"""

import pandas as pd
import nflreadpy as nfl

GAMES_COLUMNS = ["game_id", "home_team", "away_team", "home_score", "away_score"]
STATS_COLUMNS = ["player_id", "player_name", "team", "stat_category", "stat_value"]

# Columns in load_player_stats() that identify the player/game rather than
# holding a stat value -- everything else is treated as a candidate stat.
_STATS_ID_COLUMNS = {
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
}


def fetch_games(season: int, week: int) -> pd.DataFrame:
    """Load the season's schedule via load_schedules() and filter to `week`."""
    schedule_pl = nfl.load_schedules(season)
    schedule_pl = schedule_pl.filter(schedule_pl["week"] == week)
    schedule_df = schedule_pl.to_pandas()

    if schedule_df.empty:
        return pd.DataFrame(columns=GAMES_COLUMNS)

    out = schedule_df[["game_id", "home_team", "away_team", "home_score", "away_score"]]
    return out.reset_index(drop=True)


def fetch_full_schedule(season: int) -> pd.DataFrame:
    """
    Load the ENTIRE season's schedule (all weeks), unlike fetch_games which
    is scoped to one week. Used for scanning across a season to find which
    games have finished (home_score/away_score non-null) regardless of
    which week they're in -- see scripts/run_scheduled.py.

    Columns: game_id, week, home_team, away_team, home_score, away_score.
    A game has finished if and only if both score columns are non-null.
    """
    schedule_pl = nfl.load_schedules(season)
    schedule_df = schedule_pl.to_pandas()

    if schedule_df.empty:
        return pd.DataFrame(columns=["game_id", "week", "home_team", "away_team", "home_score", "away_score"])

    out = schedule_df[["game_id", "week", "home_team", "away_team", "home_score", "away_score"]]
    return out.reset_index(drop=True)


def fetch_player_stats(season: int, week: int) -> pd.DataFrame:
    """
    Load weekly player stats via load_player_stats() and melt the wide
    per-stat columns (passing_yards, rushing_tds, receptions, ...) into
    the common long-format schema: one row per (player, stat_category).

    Only numeric stat columns are melted; zero-value stats are dropped
    (e.g. a WR's passing_yards=0 row) to keep the output focused on stats
    that actually apply to each player.
    """
    stats_pl = nfl.load_player_stats(season, summary_level="week")
    stats_pl = stats_pl.filter(stats_pl["week"] == week)
    stats_df = stats_pl.to_pandas()

    if stats_df.empty:
        return pd.DataFrame(columns=STATS_COLUMNS)

    stat_columns = [
        c
        for c in stats_df.columns
        if c not in _STATS_ID_COLUMNS and pd.api.types.is_numeric_dtype(stats_df[c])
    ]

    melted = stats_df.melt(
        id_vars=["player_id", "player_name", "team"],
        value_vars=stat_columns,
        var_name="stat_category",
        value_name="stat_value",
    )

    # Drop zero/NaN stats -- e.g. a QB's "receptions" row -- so downstream
    # modeling only sees stats that were actually meaningfully recorded.
    melted = melted[melted["stat_value"].notna() & (melted["stat_value"] != 0)]

    return melted[STATS_COLUMNS].reset_index(drop=True)