"""
NFL "expected performance" model

Build a trailing baseline from the 'lookback_weeks'. Pripr weeks in the same season
Then compare the current week's actual stats against it.

Rate/efficiency stats produce misleading z-scores for low-usage players.
To fix that, stats are group and gated so they only count if a volume threshold is met.
"""

import pandas as pd
from ..data_sources import nfl as nfl_source

default_lookback_weeks = 4
# minimum historical weeks required before trusting a std-based z-score

min_history_weeks = 3

max_abs_z_score = 6.0
# hard cap on |z_score|. Even with volume gating some data will have too low of variance

# volume gating thresholds
min_attempts = 10
min_carries = 3
min_targets = 3

volume_stats = {"attempts", "carries", "targets"}

_extra_family_members = {
    "passing": {"completions", "attempts", "sacks_suffered", "sack_yards_lost",
                "sack_fumbles", "sack_fumbles_lost", "pacr"},
    "rushing": {"carries"},
    "receiving": {"receptions", "targets", "racr", "target_share",
                  "air_yards_share", "wopr"}
}

_family_volume = {
    "passing": ("attempts", min_attempts),
    "rushing": ("carries", min_carries),
    "receiving": ("targets", min_targets),
}


def _classify(stat_category: str):
    """Return (family, volume_stat, min_volume) for a gated stat_category, else None"""
    for family, members in _extra_family_members.items():
        if stat_category.startswith(f"{family}_") or stat_category in members:
            volume_stat, min_volume = _family_volume[family]
            return family, volume_stat, min_volume
    return None


def _fetch_history(season: int, week: int, lookback_weeks: int) -> pd.DataFrame:
    """Pull Player stats for up to 'lookback_weeks' prior weeks, tagged with 'week'."""
    start_week = max(1, week - lookback_weeks)
    frames = []
    for w in range(start_week, week):
        wk_df = nfl_source.fetch_player_stats(season, w)
        if not wk_df.empty:
            frames.append(wk_df.assign(week=w))

    if not frames:
        return pd.DataFrame(columns=["player_id", "team", "stat_category", "stat_value", "week"])

    return pd.concat(frames, ignore_index=True)


def _volume_lookup(df: pd.DataFrame, group_cols: list) -> dict:
    """
    Build {(*, volume_stat): value} lookup for the three volume stats,
    keyed by group_cols + stat_category. group_cols is e.g. ["player_id"]
    for a single week, or ["player_id", "week"] across multiple weeks.
    """
    vol_rows = df[df["stat_category"].isin(volume_stats)]
    return {
        tuple(row[c] for c in group_cols) + (row["stat_category"],): row["stat_value"]
        for row in vol_rows.to_dict("records")
    }


def _meets_volume(row, lookup: dict, group_cols: list) -> bool:
    """True if row's stat_category is ungated, or its family's volume stat met threshold"""
    classification = _classify(row["stat_category"])
    if classification is None:
        return True
    _, volume_stat, min_volume = classification
    key = tuple(row[c] for c in group_cols) + (volume_stat,)
    return lookup.get(key, 0) >= min_volume


def compute_expected(
        stats_df: pd.DataFrame,
        season: int,
        week: int,
        lookback_weeks: int = default_lookback_weeks,
) -> pd.DataFrame:
    """
    Args:
        stats_df: current week's actual stats (from fetch_player_stats),
            columns: player_id, player_name, team, stat_category, stat_value.
        season: season the current week belongs to.
        week: the week being evaluated. History is pulled from weeks before
            this one, within the same season.
        lookback_weeks: how many prior weeks to build the baseline from.

    Returns:
        stats_df with three columns added: expected_value, delta, z_score.
        A row gets NaN in all three if: there's no history for that
        (player, stat_category); the stat's family didn't meet its volume
        threshold in the current week; or too few historical weeks met the
        volume threshold to trust a std estimate.
    """
    history_df = _fetch_history(season, week, lookback_weeks)

    if history_df.empty:
        return stats_df.assign(expected_value=pd.NA, delta=pd.NA, z_score=pd.NA)

    # Filter historical rows to only those whose family met the volume
    # threshold (in that same week) so a player's low usage early season
    # weeks don't drag down or inflate the baseline for their efficiency stats
    hist_volume = _volume_lookup(history_df, ["player_id", "week"])
    hist_mask = history_df.apply(
        lambda r: _meets_volume(r, hist_volume, ["player_id", "week"]), axis=1
    )
    filtered_history = history_df[hist_mask]

    baseline = (
        filtered_history.groupby(["player_id", "stat_category"])["stat_value"]
        .agg(expected_value="mean", _std="std", _n="count")
        .reset_index()
    )

    merged = stats_df.merge(baseline, on=["player_id", "stat_category"], how="left")
    merged["delta"] = merged["stat_value"] - merged["expected_value"]

    # current week volume gating: does this row's family meet the threshold (this week)
    cur_volume = _volume_lookup(stats_df, ["player_id"])
    volume_ok = merged.apply(
        lambda r: _meets_volume(r, cur_volume, ["player_id"]), axis=1
    )

    enough_history = merged["_n"].fillna(0) >= min_history_weeks
    nonzero_std = merged["_std"].fillna(0) > 0
    valid = enough_history & nonzero_std & volume_ok

    merged["z_score"] = pd.NA
    merged.loc[valid, "z_score"] = merged.loc[valid, "delta"] / merged.loc[valid, "_std"]
    merged.loc[valid, "z_score"] = merged.loc[valid, "z_score"].clip(
        -max_abs_z_score, max_abs_z_score
    )

    # for rows that failed only the volume gate, also null out expected_value/delta
    # a baseline built on too few qualifying weeks isn't meaningful
    # to show even without a z_score attached to it.
    merged.loc[~volume_ok, ["expected_value", "delta"]] = pd.NA

    return merged.drop(columns=["_std", "_n"])