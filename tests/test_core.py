"""Core schema, prompt, and rule-regression tests; no Ollama server required."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import MatchState, AdviceResponse, AppConfig
from core.reasoning_engine import SYSTEM_PROMPT, _build_user_prompt, get_advice
from core.ollama_client import OllamaClient


def _state(**overrides) -> MatchState:
    values = dict(minute=70, my_score=0, opponent_score=0, formation="4-3-3", possession_pct=50)
    values.update(overrides)
    return MatchState(**values)


def test_match_state_from_simulated_scenarios():
    scenarios = json.loads((Path(__file__).parent.parent / "data" / "simulated_scenarios.json").read_text())["scenarios"]
    assert scenarios
    for scenario in scenarios:
        assert 0 <= MatchState.model_validate(scenario["state"]).minute <= 120


def test_compact_advice_schema_limits_secondary_considerations():
    advice = AdviceResponse.model_validate({
        "top_suggestion": "Use calmer shot selection.",
        "secondary_considerations": ["Keep width", "Use a fresh striker"],
        "formation_change": None,
        "style_change": None,
    })
    assert advice.top_suggestion.startswith("Use calmer")
    try:
        AdviceResponse.model_validate({"top_suggestion": "Too many extras.", "secondary_considerations": ["one", "two", "three"]})
    except Exception:  # pydantic's validation-error class is an implementation detail here
        return
    raise AssertionError("AdviceResponse accepted more than two secondary considerations")


class _RuleAwareMockClient(OllamaClient):
    """Deterministic stand-in exposing the prompt's decision signal.

    The regression is not tied to an LLM's variable wording: it verifies that
    every fixed MatchState reaches the correct prompt cue and that it survives
    as advice text.
    """
    def __init__(self):
        super().__init__(AppConfig())

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        assert system_prompt == SYSTEM_PROMPT
        if "OUR FINISHING/SHOT SELECTION" in user_prompt:
            top = "Prioritize shot selection and composure; do not force more shots."
        elif "OUR CHANCE CREATION" in user_prompt:
            top = "Improve buildup and width to create chances."
        elif "ENERGY/CONTROL RISK" in user_prompt:
            top = "Conserve energy in a compact defensive shape; do not press higher."
        elif "OUR CARD RISK" in user_prompt:
            top = "Stay on your feet and temper the press because card risk is live."
        elif "OPPONENT FINISHING" in user_prompt:
            top = "Defend calmly: their shots lack accuracy."
        elif "OPPONENT CHANCE CREATION" in user_prompt:
            top = "Their chance creation is limited; keep a controlled shape."
        else:
            top = "Unknown stats need checking before a stats-based change."
        return {"top_suggestion": top, "secondary_considerations": [], "formation_change": None, "style_change": None}


def test_combined_stat_reasoning_regressions():
    """Seven fixed snapshots assert concepts, not brittle exact LLM wording."""
    cases = [
        ("high shots / low on target", _state(shots=12, shots_on_target=3), "shot selection"),
        ("low shots", _state(shots=2, shots_on_target=1), "buildup"),
        ("falling possession + low stamina", _state(possession_trend="falling", team_stamina_avg_pct=45), "conserve energy"),
        ("high cards while currently high pressing", _state(playing_style="high_press", my_yellow_cards=3), "stay on your feet"),
        ("opponent high shots / low on target", _state(opponent_shots=10, opponent_shots_on_target=2), "lack accuracy"),
        ("opponent low shots", _state(opponent_shots=2, opponent_shots_on_target=1), "chance creation is limited"),
        ("unknown optional stats", _state(shots=None, shots_on_target=None, my_yellow_cards=None), "unknown stats"),
    ]
    client = _RuleAwareMockClient()
    for name, state, expected_keyword in cases:
        advice = get_advice(state, client)
        advice_text = " ".join([advice.top_suggestion, *advice.secondary_considerations]).lower()
        assert expected_keyword in advice_text, name


def test_prompt_marks_all_unentered_optional_fields_unknown():
    prompt = _build_user_prompt(_state())
    assert "Shots: unknown (us) vs unknown (opponent)" in prompt
    assert "Yellow cards: unknown (us) vs unknown (opponent)" in prompt
    assert "No combined-stat diagnosis" in prompt


def test_invented_enum_value_is_sanitized_not_fatal():
    class InvalidEnumClient(_RuleAwareMockClient):
        def generate_json(self, system_prompt, user_prompt):
            return {"top_suggestion": "Keep shape.", "secondary_considerations": [], "formation_change": None, "style_change": "counter_press"}

    assert get_advice(_state(), InvalidEnumClient()).style_change is None


if __name__ == "__main__":
    tests = [
        test_match_state_from_simulated_scenarios,
        test_compact_advice_schema_limits_secondary_considerations,
        test_combined_stat_reasoning_regressions,
        test_prompt_marks_all_unentered_optional_fields_unknown,
        test_invented_enum_value_is_sanitized_not_fatal,
    ]
    for test in tests:
        test()
        print(f"OK - {test.__name__}")
    print(f"\n{len(tests)} core tests passed.")
