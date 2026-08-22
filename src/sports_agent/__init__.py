from .main import run_pipeline


def main() -> None:
    result = run_pipeline(sport="nfl", season=2025, week=1)

    if result["errors"]:
        print("Errors encountered:")
        for err in result["errors"]:
            print(f"  - {err}")

    print(result["report"])