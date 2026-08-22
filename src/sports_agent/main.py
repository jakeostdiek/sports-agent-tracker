import os

import operator
import pandas as pd
from typing import TypedDict, List, Optional, Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Uncomment later
from .data_sources import get_source
from .modeling import get_model

load_dotenv()


# start the STATE

# everything each node reads and writes lives here
class PipelineState(TypedDict):
    sport: str
    season: int
    week: int

    # optional to be pd.DataFrame | None
    raw_games_df: Optional[pd.DataFrame] # schedule/results
    raw_stats_df: Optional[pd.DataFrame] # player box score

    expected_df: Optional[pd.DataFrame]

    overperformers_df: Optional[pd.DataFrame]
    underperformers_df: Optional[pd.DataFrame]

    report: Optional[str]
    errors: Annotated[List[str], operator.add]


"""
Nodes
All the sport-specific logic will stat in data_sources/ and modeling/
"""


def ingest_games(state: PipelineState) -> dict:
    try:
        source = get_source(state['sport'])
        games_df = source.fetch_games(state["season"], state["week"])
        return {"raw_games_df": games_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"ingest_games failed: {e}"]}


def ingest_player_stats(state: PipelineState) -> dict:
    try:
        source = get_source(state['sport'])
        stats_df = source.fetch_player_stats(state["season"], state["week"])
        return {"raw_stats_df": stats_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"ingest_player_stats failed: {e}"]}


def model_expected_performances(state: PipelineState) -> dict:
    try:
        model = get_model(state["sport"])
        expected_df = model.compute_expected(
            state["raw_stats_df"], state["season"], state["week"]
        )
        return {"expected_df": expected_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"model_expected_performances failed: {e}"]}


def find_outliers(state: PipelineState) -> dict:
    """Sport-agnostic: works off the standardized z_score column, so this node doesn't need a per-sport variant."""
    try:
        df = state["expected_df"]
        over_df = df[df["z_score"] >= 1.5].sort_values("z_score", ascending=False)
        under_df = df[df["z_score"] <= -1.5].sort_values("z_score")
        return {"overperformers_df": over_df, "underperformers_df": under_df}
    except Exception as e:
        return {"errors": state["errors"] + [f"find_outliers failed: {e}"]}


def generate_report(state: PipelineState) -> dict:
    try:
        llm = init_chat_model("claude-sonnet-4-6", model_provider="anthropic")


        prompt = f"""
You are a sports analyst. Write a concise report for {state['sport'].upper()},
Season {state['season']}, Week {state['week']},

Overperformers (did better than expected):
{state['overperformers_df'].to_string(index=False) if not state['overperformers_df'].empty else "None found."}

Underperformers (did worse than expected):
{state['underperformers_df'].to_string(index=False) if not state['underperformers_df'].empty else "None found."}

For each notable player, note what was expected vs. what happened, and a plausible reason if apparent from 
the data (matchup, injury, game script, etc).
"""
        response = llm.invoke(prompt)
        return {"report": response.content}
    except Exception as e:
        return {"errors": state["errors"] + [f"generate_report failed: {e}"]}



"""
Graph Wiring
"""


graph = StateGraph(PipelineState)

graph.add_node("ingest_games", ingest_games)
graph.add_node("ingest_player_stats", ingest_player_stats)
graph.add_node("model_expected_performances", model_expected_performances)
graph.add_node("find_outliers", find_outliers)
graph.add_node("generate_report", generate_report)

graph.add_edge(START, "ingest_games")
graph.add_edge(START, "ingest_player_stats")
graph.add_edge("ingest_games", "model_expected_performances")
graph.add_edge("ingest_player_stats", "model_expected_performances")
graph.add_edge("model_expected_performances", "find_outliers")
graph.add_edge("find_outliers", "generate_report")
graph.add_edge("generate_report", END)

app = graph.compile()



"""
Run
"""



def run_pipeline(sport: str, season: int, week: int) -> PipelineState:
    initial_state: PipelineState = {
        "sport": sport,
        "season": season,
        "week": week,
        "raw_games_df": None,
        "raw_stats_df": None,
        "expected_df": None,
        "overperformers_df": None,
        "underperformers_df": None,
        "report": None,
        "errors": [],
    }
    return app.invoke(initial_state)


if __name__ == "__main__":
    # cfb/nba data sources not filled in yet
    result = run_pipeline(sport="nfl", season=2025, week=1)

    if result["errors"]:
        print("Errors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")

    print(result["report"])