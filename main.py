"""
Bosman AI Coach — Phase 1 CLI.

Run without arguments to pick from the bundled simulated scenarios:
    python main.py

Run with --list to just see the scenario names:
    python main.py --list

Run with --scenario "<name>" to skip the picker:
    python main.py --scenario "Losing late, opponent attacking down the left"

This script proves out the core reasoning loop (match state -> prompt ->
Ollama -> validated advice) before any GUI or FIFA integration is built.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import yaml

from core.schemas import MatchState, AppConfig
from core.ollama_client import OllamaClient, OllamaError
from core.reasoning_engine import get_advice

ROOT = Path(__file__).parent
SCENARIOS_PATH = ROOT / "data" / "simulated_scenarios.json"
CONFIG_PATH = ROOT / "config.yaml"


def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig.model_validate(raw)
    return AppConfig()


def load_scenarios() -> list[dict]:
    with open(SCENARIOS_PATH) as f:
        return json.load(f)["scenarios"]


def print_advice(advice) -> None:
    print("\n" + "=" * 60)
    print("BOSMAN AI COACH — RECOMMENDATION")
    print("=" * 60)
    print(f"\nTOP: {advice.top_suggestion}\n")
    if advice.formation_change:
        print(f"  Formation change   -> {advice.formation_change.value}")
    if advice.style_change:
        print(f"  Playing style      -> {advice.style_change.value}")
    if advice.secondary_considerations:
        print("  Also consider:")
        for consideration in advice.secondary_considerations:
            print(f"    - {consideration}")
    print("=" * 60 + "\n")


def pick_scenario_interactively(scenarios: list[dict]) -> dict:
    print("Available simulated scenarios:\n")
    for i, s in enumerate(scenarios, 1):
        print(f"  {i}. {s['name']}")
    choice = input(f"\nPick a scenario [1-{len(scenarios)}]: ").strip()
    try:
        idx = int(choice) - 1
        return scenarios[idx]
    except (ValueError, IndexError):
        print("Invalid choice.")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bosman AI Coach — Phase 1 CLI")
    parser.add_argument("--list", action="store_true", help="List scenario names and exit")
    parser.add_argument("--scenario", type=str, help="Scenario name to run directly")
    args = parser.parse_args()

    scenarios = load_scenarios()

    if args.list:
        for s in scenarios:
            print(s["name"])
        return

    if args.scenario:
        matches = [s for s in scenarios if s["name"] == args.scenario]
        if not matches:
            print(f"No scenario named '{args.scenario}'. Use --list to see options.")
            sys.exit(1)
        scenario = matches[0]
    else:
        scenario = pick_scenario_interactively(scenarios)

    state = MatchState.model_validate(scenario["state"])
    config = load_config()
    client = OllamaClient(config)

    print(f"\nScenario: {scenario['name']}")
    print(f"Asking {config.model} at {config.ollama_host} for advice...")

    if not client.health_check():
        print(
            f"\nCouldn't reach Ollama at {config.ollama_host}.\n"
            "Make sure it's running (`ollama serve`) and the model is pulled "
            f"(`ollama pull {config.model}`)."
        )
        sys.exit(1)

    try:
        advice = get_advice(state, client)
    except OllamaError as e:
        print(f"\nError getting advice: {e}")
        sys.exit(1)

    print_advice(advice)


if __name__ == "__main__":
    main()
