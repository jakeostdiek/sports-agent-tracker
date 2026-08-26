import os
import operator
import pandas as pd
from typing import TypedDict, List, Optional, Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from .data_sources import get_source
from .modeling import get_model

load_dotenv()

# ---------------------------------------------------------------------------
# 1. STATE
# ---------------------------------------------------------------------------
# `sport` drives which data_sources/modeling module each node dispatches to.
# Everything else stays generic -- adding a new sport means writing
# data_sources/<sport>.py + modeling/<sport>.py, not touching this file.

# Outlier z_score thresholds shared by every game's report.
OVERPERFORMER_Z = 1.5
UNDERPERFORMER_Z = -1.5


class GameReport(TypedDict):
    game_id: str
    home_team: str
    away_team: str
    report: str


class PipelineState(TypedDict):
    sport: str          # e.g. "cfb", "nba" -- must match a registered module
    season: int
    week: int
    # If set, generate_game_reports only produces reports for games whose
    # game_id is in this list (used by scripts/run_scheduled.py to avoid
    # re-generating reports for games already processed in a prior run).
    # None (the default) means "every game in raw_games_df", as before.
    target_game_ids: Optional[List[str]]

    raw_games_df: Optional[pd.DataFrame]
    raw_stats_df: Optional[pd.DataFrame]

    expected_df: Optional[pd.DataFrame]

    game_reports: Annotated[List[GameReport], operator.add]
    errors: Annotated[List[str], operator.add]


# ---------------------------------------------------------------------------
# 2. NODES
# ---------------------------------------------------------------------------
# These stay thin: look up the right module for state["sport"], call it,
# store the result. All the sport-specific logic lives in data_sources/ and
# modeling/, not here.

def ingest_games(state: PipelineState) -> dict:
    try:
        source = get_source(state["sport"])
        games_df = source.fetch_games(state["season"], state["week"])
        return {"raw_games_df": games_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"ingest_games failed: {e}"]}


def ingest_player_stats(state: PipelineState) -> dict:
    try:
        source = get_source(state["sport"])
        stats_df = source.fetch_player_stats(state["season"], state["week"])
        return {"raw_stats_df": stats_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"ingest_player_stats failed: {e}"]}


def model_expected_performance(state: PipelineState) -> dict:
    try:
        model = get_model(state["sport"])
        expected_df = model.compute_expected(
            state["raw_stats_df"], state["season"], state["week"]
        )
        return {"expected_df": expected_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"model_expected_performance failed: {e}"]}


def _find_outliers(game_expected_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sport-agnostic: works off the standardized z_score column. Scoped to
    whatever subset of expected_df is passed in (e.g. just one game's players)."""
    over_df = game_expected_df[game_expected_df["z_score"] >= OVERPERFORMER_Z].sort_values(
        "z_score", ascending=False
    )
    under_df = game_expected_df[game_expected_df["z_score"] <= UNDERPERFORMER_Z].sort_values(
        "z_score"
    )
    return over_df, under_df


def _write_game_report(
    llm, sport: str, season: int, week: int, home_team: str, away_team: str,
    over_df: pd.DataFrame, under_df: pd.DataFrame,
) -> str:
    prompt = f"""
You are a sports analyst. Write a concise report for a single {sport.upper()} game:
{away_team} @ {home_team}, Season {season}, Week {week}.

Overperformers (did better than expected):
{over_df.to_string(index=False) if not over_df.empty else 'None found.'}

Underperformers (did worse than expected):
{under_df.to_string(index=False) if not under_df.empty else 'None found.'}

For each notable player, note what was expected vs. what happened, and a plausible reason if apparent from the data (matchup, injury, game script, etc).
"""
    response = llm.invoke(prompt)
    return response.content


def generate_game_reports(state: PipelineState) -> dict:
    """
    Loops over every game in raw_games_df, filters expected_df down to just
    that game's two teams, and generates one report per game -- rather than
    one league-wide report for the whole week.

    Runs sequentially (one LLM call per game). For a normal NFL week
    (~14-16 games) this is fine; if this ever needs to scale to something
    with many more games per period, this is the node to convert into a
    parallel fan-out (e.g. LangGraph's Send API) instead of a plain loop.
    """
    try:
        games_df = state["raw_games_df"]
        expected_df = state["expected_df"]

        if games_df is None or games_df.empty:
            return {"errors": state["errors"] + ["generate_game_reports: no games to report on"]}

        target_game_ids = state.get("target_game_ids")
        if target_game_ids is not None:
            games_df = games_df[games_df["game_id"].astype(str).isin(target_game_ids)]
            if games_df.empty:
                return {"errors": state["errors"] + ["generate_game_reports: none of target_game_ids matched raw_games_df"]}

        llm = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")

        reports: List[GameReport] = []
        node_errors: List[str] = []

        for _, game in games_df.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            game_id = str(game["game_id"])

            try:
                game_expected_df = expected_df[
                    expected_df["team"].isin([home_team, away_team])
                ]
                over_df, under_df = _find_outliers(game_expected_df)
                report_text = _write_game_report(
                    llm, state["sport"], state["season"], state["week"],
                    home_team, away_team, over_df, under_df,
                )
                reports.append({
                    "game_id": game_id,
                    "home_team": home_team,
                    "away_team": away_team,
                    "report": report_text,
                })
            except Exception as e:
                node_errors.append(f"generate_game_reports failed for game {game_id}: {e}")

        return {"game_reports": reports, "errors": state["errors"] + node_errors}
    except Exception as e:
        return {"errors": state["errors"] + [f"generate_game_reports failed: {e}"]}


# ---------------------------------------------------------------------------
# 3. GRAPH WIRING
# ---------------------------------------------------------------------------
graph = StateGraph(PipelineState)

graph.add_node("ingest_games", ingest_games)
graph.add_node("ingest_player_stats", ingest_player_stats)
graph.add_node("model_expected_performance", model_expected_performance)
graph.add_node("generate_game_reports", generate_game_reports)

graph.add_edge(START, "ingest_games")
graph.add_edge(START, "ingest_player_stats")
graph.add_edge("ingest_games", "model_expected_performance")
graph.add_edge("ingest_player_stats", "model_expected_performance")
graph.add_edge("model_expected_performance", "generate_game_reports")
graph.add_edge("generate_game_reports", END)

app = graph.compile()


# ---------------------------------------------------------------------------
# 4. RUN
# ---------------------------------------------------------------------------
def run_pipeline(
    sport: str, season: int, week: int, target_game_ids: Optional[List[str]] = None
) -> PipelineState:
    initial_state: PipelineState = {
        "sport": sport,
        "season": season,
        "week": week,
        "target_game_ids": target_game_ids,
        "raw_games_df": None,
        "raw_stats_df": None,
        "expected_df": None,
        "game_reports": [],
        "errors": [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    # cfb/nba data sources are stubbed -- fill those in later when you
    # circle back to add more sports.
    result = run_pipeline(sport="nfl", season=2025, week=1)

    if result["errors"]:
        print("Errors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")

    for game_report in result["game_reports"]:
        print(f"\n=== {game_report['away_team']} @ {game_report['home_team']} ===")
        print(game_report["report"])