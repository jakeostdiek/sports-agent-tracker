"""
Registry of per-sport "expected performance" models.

Every module in this package must expose:

    compute_expected(stats_df: pd.DataFrame, season: int, week: int) -> pd.DataFrame

which takes the normalized stats dataframe (common schema from data_sources)
and returns it with expected_value, delta, and z_score columns added.
"""

#from . import cfb
from . import nfl
#from . import nba

MODELS = {
    #"cfb": cfb,
    "nfl": nfl
    #"nba": nba,
}


def get_model(sport: str):
    try:
        return MODELS[sport]
    except KeyError:
        raise ValueError(
            f"No model registered for sport={sport!r}. Available: {list(MODELS.keys())}"
        )