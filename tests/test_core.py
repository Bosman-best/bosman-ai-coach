"""
Sanity tests that don't require a running Ollama server.
Run with: python -m pytest tests/ -v   (or just: python tests/test_core.py)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.schemas import MatchState, AdviceResponse, AppConfig
from core.reasoning_engine import _build_user_prompt, get_advice
from core.ollama_client import OllamaClient


def test_match_state_from_simulated_scenarios():
    scenarios_path = Path(__file__).parent.parent / "data" / "simulated_scenarios.json"
    scenarios = json.loads(scenarios_path.read_text())["scenarios"]
    assert len(scenarios) > 0
    for s in scenarios:
        state = MatchState.model_validate(s["state"])
        assert 0 <= state.minute <= 120
    print(f"OK - validated {len(scenarios)} simulated scenarios")


def test_prompt_building():
    state = MatchState(
        minute=70, my_score=1, opponent_score=2, formation="4-3-3",
        possession_pct=40, opponent_threat_side="left_wing", striker_stamina_pct=35,
    )
    prompt = _build_user_prompt(state)
    assert "losing 1-2" in prompt
    assert "left_wing" in prompt
    assert "Minute: 70" in prompt
    print("OK - prompt building includes score, minute, and threat side")


def test_advice_schema_validation():
    raw = {
        "summary": "Switch to 4-2-3-1 and add defensive solidity.",
        "formation_change": "4-2-3-1",
        "style_change": "counter_attack",
        "substitution_suggestions": ["Bring on a fresh striker for the tiring one"],
        "tactical_instructions": ["Reduce defensive width", "Press higher on turnovers"],
        "reasoning": "You're chasing the game and the striker is gassed.",
    }
    advice = AdviceResponse.model_validate(raw)
    assert advice.formation_change.value == "4-2-3-1"
    print("OK - AdviceResponse validates a well-formed model response")


class _MockOllamaClient(OllamaClient):
    """Stands in for OllamaClient without hitting the network."""
    def __init__(self, canned_response: dict):
        super().__init__(AppConfig())
        self._canned = canned_response

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        return self._canned


def test_full_reasoning_loop_with_mocked_model():
    state = MatchState(
        minute=82, my_score=1, opponent_score=0, formation="4-2-3-1",
        possession_pct=55,
    )
    canned = {
        "summary": "Shore up the midfield to protect the lead.",
        "formation_change": None,
        "style_change": "park_the_bus",
        "substitution_suggestions": ["Bring on a defensive midfielder for legs"],
        "tactical_instructions": ["Drop deeper", "Kill the tempo"],
        "reasoning": "You're a goal up late; prioritize not conceding over chasing a second.",
    }
    client = _MockOllamaClient(canned)
    advice = get_advice(state, client)
    assert advice.style_change.value == "park_the_bus"
    assert advice.formation_change is None
    print("OK - full get_advice() loop works end-to-end with a mocked model response")


def test_invented_enum_value_is_sanitized_not_fatal():
    """Regression test for a real failure: qwen2.5:3b once returned
    style_change='counter_press', which isn't in our enum. That should be
    dropped to null, not blow up the whole advice response."""
    state = MatchState(
        minute=82, my_score=1, opponent_score=0, formation="4-2-3-1",
        possession_pct=55,
    )
    canned = {
        "summary": "Maintain possession and counter-press to prevent a late surge.",
        "formation_change": None,
        "style_change": "counter_press",  # invalid - not in PlayingStyle enum
        "substitution_suggestions": ["Bring in a fresh defender"],
        "tactical_instructions": ["Press aggressively from your own half"],
        "reasoning": "Protect the lead while stopping the opponent's fresh legs.",
    }
    client = _MockOllamaClient(canned)
    advice = get_advice(state, client)
    assert advice.style_change is None
    assert advice.summary == canned["summary"]
    print("OK - invented enum value ('counter_press') is dropped instead of crashing")


if __name__ == "__main__":
    test_match_state_from_simulated_scenarios()
    test_prompt_building()
    test_advice_schema_validation()
    test_full_reasoning_loop_with_mocked_model()
    test_invented_enum_value_is_sanitized_not_fatal()
    print("\nAll tests passed.")
