"""
Registry of per-sport data source modules.

Every module in this package must expose two functions with this exact
signature so main.py can call them generically:

    fetch_games(season: int, week: int) -> pd.DataFrame
    fetch_player_stats(season: int, week: int) -> pd.DataFrame

Both should return dataframes normalized to the common schema (see each
module's docstring) regardless of what shape the underlying API returns --
that normalization is the whole point of this layer.
"""

#from . import cfb
from . import nfl
#from . import nba

SOURCES = {
    #"cfb": cfb,
    "nfl": nfl
    #"nba": nba,
}


def get_source(sport: str):
    try:
        return SOURCES[sport]
    except KeyError:
        raise ValueError(
            f"No data source registered for sport={sport!r}. "
            f"Available: {list(SOURCES.keys())}"
        )