"""
Scheduled entry point for autonomous NFL report generation.

Meant to be invoked periodically (e.g. by GitHub Actions on a cron
schedule). Each run:
  1. Determines the current NFL season from today's date.
  2. Scans the FULL season schedule for games that have finished
     (home_score/away_score populated).
  3. Skips any game_id already recorded in data/processed_games.json.
  4. Generates a report for each newly-finished game (grouped by week,
     since run_pipeline still ingests/models at the week level).
  5. Writes each report to reports/<season>/week_<week>/<game_id>.md
  6. Updates data/processed_games.json with the newly-processed game_ids.

Exits with code 0 even if there's nothing new to do -- "no new finished
games" is the normal, expected outcome most times this runs, not a failure.
"""

import json
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Optional

from sports_agent.data_sources import nfl as nfl_source
from sports_agent.main import run_pipeline

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_GAMES_PATH = REPO_ROOT / "data" / "processed_games.json"
REPORTS_DIR = REPO_ROOT / "reports"


def get_current_season(today: date) -> Optional[int]:
    """
    NFL seasons are labeled by the year they START in (e.g. games played
    Jan 2027 are still part of the "2026 season"). Returns None during the
    offseason (roughly March-July) when there's nothing to check.
    """
    if today.month >= 8:
        return today.year
    if today.month <= 2:
        return today.year - 1
    return None


def load_processed_games() -> set:
    if not PROCESSED_GAMES_PATH.exists():
        return set()
    with open(PROCESSED_GAMES_PATH) as f:
        return set(json.load(f))


def save_processed_games(processed: set) -> None:
    PROCESSED_GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(PROCESSED_GAMES_PATH, "w") as f:
        json.dump(sorted(processed), f, indent=2)


def write_report(season: int, week: int, game_report: dict) -> None:
    out_dir = REPORTS_DIR / str(season) / f"week_{week}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{game_report['game_id']}.md"
    header = f"# {game_report['away_team']} @ {game_report['home_team']}\n\n"
    out_path.write_text(header + game_report["report"], encoding="utf-8")


def main() -> int:
    today = date.today()
    season = get_current_season(today)
    PROCESSED_GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if season is None:
        print(f"{today}: NFL offseason, nothing to check.")
        return 0

    print(f"Checking season {season} for newly finished games...")

    schedule_df = nfl_source.fetch_full_schedule(season)
    finished = schedule_df[
        schedule_df["home_score"].notna() & schedule_df["away_score"].notna()
    ]

    processed = load_processed_games()
    new_finished = finished[~finished["game_id"].astype(str).isin(processed)]

    if new_finished.empty:
        print("No new finished games since last run.")
        return 0

    print(f"Found {len(new_finished)} newly finished game(s):")
    for _, g in new_finished.iterrows():
        print(f"  - {g['game_id']} ({g['away_team']} @ {g['home_team']}, week {g['week']})")

    # run_pipeline ingests/models at the week level, so group by week and
    # make one call per week, scoped to just that week's newly finished games.
    by_week = defaultdict(list)
    for _, g in new_finished.iterrows():
        by_week[int(g["week"])].append(str(g["game_id"]))

    newly_processed = set()
    had_errors = False

    for week, game_ids in sorted(by_week.items()):
        print(f"\nGenerating reports for week {week}, games: {game_ids}")
        result = run_pipeline(sport="nfl", season=season, week=week, target_game_ids=game_ids)

        if result["errors"]:
            had_errors = True
            print(f"  Errors during week {week}:")
            for err in result["errors"]:
                print(f"    - {err}")

        for game_report in result["game_reports"]:
            write_report(season, week, game_report)
            newly_processed.add(game_report["game_id"])
            print(f"  Wrote report for {game_report['game_id']}")

    processed |= newly_processed
    save_processed_games(processed)
    print(f"\nDone. {len(newly_processed)} report(s) written.")

    # Errors from individual games are logged but don't fail the whole run --
    # partial success (some reports written) is still useful. A completely
    # empty newly_processed set despite finding new_finished games points
    # to a systemic failure worth surfacing as a failed CI run.
    if had_errors and not newly_processed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
