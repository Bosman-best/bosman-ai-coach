"""Build and validate read-only live-match coaching advice."""

from __future__ import annotations

from pydantic import ValidationError

from core.schemas import MatchState, AdviceResponse, Formation, PlayingStyle
from core.ollama_client import OllamaClient, OllamaError


SYSTEM_PROMPT = """You are Bosman, an expert football (soccer) tactical coach
assisting a player during a live match. The supplied MatchState is read-only:
you advise the player, but never claim to control the game, read new game data,
or make changes yourself.

Give precise, actionable advice based only on stated information. A value of
"unknown" means it was not entered: it is not zero, neutral, or evidence for a
confident conclusion. Do not infer a statistic from an unknown value. If a
recommendation depends on an unknown field, either avoid it or explicitly say
that it needs checking.

First select the dominant priority: numerical disadvantage overrides all other
states; otherwise protect a late lead, chase a late deficit, break down a low
block, or manage an even contest. Then apply the combined-stat rules below;
they override generic advice when applicable:

- HIGH SHOTS + LOW SHOTS ON TARGET (for either side): this is chance creation
  with poor finishing/composure or shot selection. For us, prioritize a calmer
  final pass, better shot selection, or a finishing/personnel change. Do NOT
  prescribe more attacking width, tempo, or shot volume merely to create more
  chances. For the opponent, defend the box and expect wasteful attempts rather
  than assuming their attack is ineffective.
- LOW SHOTS OVERALL: this is a chance-creation problem. For us, prioritize
  buildup, width, movement, overloads, or tempo—not finishing personnel as the
  primary fix. Apply this only when both shot count and the relevant context are
  known; do not call unknown shots low.
- FALLING POSSESSION + LOW TEAM STAMINA: treat this as an energy/control risk.
  Prioritize conserving energy, a compact defensive shape, and fresh legs;
  explicitly avoid telling the player to press higher.
- HIGH YELLOW-CARD COUNT: card risk is live. Temper or avoid aggressive
  pressing, hard tackling, and physical challenges for that team; favor staying
  on feet and controlled pressure. This constraint applies even if chasing.

Use the explicit "Decision signals" in the match situation as a cross-check.
They are derived only from entered fields and must not be contradicted.

Keep the response glanceable during live play: one clearly prioritized action
and zero, one, or two brief secondary considerations. Do not produce a report,
a flat list of unrelated instructions, or a long explanation.

Respond with one JSON object and nothing else:
{
  "top_suggestion": "one short, prioritized action",
  "secondary_considerations": ["short supporting action", "optional second supporting action"],
  "formation_change": "4-3-3" | "4-4-2" | "3-5-2" | "4-2-3-1" | "5-3-2" | "3-4-2-1" | null,
  "style_change": "balanced" | "possession" | "counter_attack" | "long_ball" | "high_press" | "park_the_bus" | null
}

secondary_considerations has at most two items. Set formation_change or
style_change only for an actual change from the stated current setting;
otherwise use null. Never invent enum values."""


def _value_or_unknown(value: object) -> str:
    """Render optional MatchState values without allowing None to masquerade as 0."""
    return "unknown" if value is None else str(value)


def _decision_signals(state: MatchState) -> list[str]:
    """Derive only safe, combined-stat prompt cues from entered values.

    Thresholds are deliberately conservative so a small sample is not treated
    as a tactical diagnosis. Unknown inputs never produce a signal.
    """
    signals: list[str] = []

    if state.shots is not None and state.shots_on_target is not None and state.shots >= 8 and state.shots_on_target / state.shots <= 0.35:
        signals.append(
            "OUR FINISHING/SHOT SELECTION: high shot volume but low accuracy; "
            "target composure, personnel, or shot selection—not more chance creation."
        )
    elif state.shots is not None and state.shots <= 3:
        signals.append("OUR CHANCE CREATION: low shot volume; target buildup, width, movement, or tempo.")

    if state.opponent_shots is not None and state.opponent_shots_on_target is not None and state.opponent_shots >= 8 and state.opponent_shots_on_target / state.opponent_shots <= 0.35:
        signals.append("OPPONENT FINISHING: high shots but low accuracy; defend calmly and avoid overreacting.")
    elif state.opponent_shots is not None and state.opponent_shots <= 3:
        signals.append("OPPONENT CHANCE CREATION: low shot volume; their threat is currently limited.")

    if state.possession_trend is not None and state.team_stamina_avg_pct is not None:
        if state.possession_trend.value == "falling" and state.team_stamina_avg_pct <= 55:
            signals.append(
                "ENERGY/CONTROL RISK: possession is falling and team stamina is low; "
                "conserve energy, keep a compact shape, and do not press higher."
            )

    if state.my_yellow_cards is not None and state.my_yellow_cards >= 2:
        signals.append("OUR CARD RISK: high yellow-card count; temper pressing and hard tackling, stay on feet.")
    if state.opponent_yellow_cards is not None and state.opponent_yellow_cards >= 2:
        signals.append("OPPONENT CARD RISK: high yellow-card count; draw pressure rather than invite hard challenges.")

    return signals or ["No combined-stat diagnosis is available from the entered data; do not guess from unknown fields."]


def _build_user_prompt(state: MatchState) -> str:
    diff = state.score_diff()
    situation = "drawing" if diff == 0 else ("winning" if diff > 0 else "losing")
    lines = [
        f"Minute: {state.minute}",
        f"Score: {situation} {state.my_score}-{state.opponent_score}",
        f"Current formation: {state.formation.value}",
        f"Current playing style: {state.playing_style.value}",
        f"Possession: {state.possession_pct}%",
        f"Match half/state: {_value_or_unknown(state.match_half.value if state.match_half else None)}",
        f"Possession trend: {_value_or_unknown(state.possession_trend.value if state.possession_trend else None)}",
        f"Opponent threat side: {_value_or_unknown(state.opponent_threat_side)}",
        f"Team average stamina: {_value_or_unknown(state.team_stamina_avg_pct)}%",
        f"Striker stamina: {_value_or_unknown(state.striker_stamina_pct)}%",
        f"Key player stamina: {_value_or_unknown(state.key_player_stamina_pct)}%",
        f"Shots: {_value_or_unknown(state.shots)} (us) vs {_value_or_unknown(state.opponent_shots)} (opponent)",
        f"Shots on target: {_value_or_unknown(state.shots_on_target)} (us) vs {_value_or_unknown(state.opponent_shots_on_target)} (opponent)",
        f"Corners: {_value_or_unknown(state.corners)} (us) vs {_value_or_unknown(state.opponent_corners)} (opponent)",
        f"Pass accuracy: {_value_or_unknown(state.pass_accuracy_pct)}% (us) vs {_value_or_unknown(state.opponent_pass_accuracy_pct)}% (opponent)",
        f"Fouls committed: {_value_or_unknown(state.fouls_committed)} (us) vs {_value_or_unknown(state.opponent_fouls_committed)} (opponent)",
        f"Menu formation read: {_value_or_unknown(state.menu_formation.value if state.menu_formation else None)}",
        f"Yellow cards: {_value_or_unknown(state.my_yellow_cards)} (us) vs {_value_or_unknown(state.opponent_yellow_cards)} (opponent)",
        f"Our red card: {'yes' if state.red_card else 'no'}",
        f"Opponent red card: {'yes' if state.opponent_red_card else 'no'}",
    ]
    if state.notes:
        lines.append(f"Additional context: {state.notes}")
    lines.extend(["Decision signals:", *[f"- {signal}" for signal in _decision_signals(state)]])
    return "Match situation:\n" + "\n".join(lines) + "\n\nGive live-play tactical advice."


def _sanitize_enum_fields(raw: dict) -> dict:
    sanitized = dict(raw)
    formation_values = {f.value for f in Formation}
    style_values = {s.value for s in PlayingStyle}
    if sanitized.get("formation_change") not in formation_values | {None}:
        sanitized["formation_change"] = None
    if sanitized.get("style_change") not in style_values | {None}:
        sanitized["style_change"] = None
    return sanitized


def get_advice(state: MatchState, client: OllamaClient) -> AdviceResponse:
    """Return validated advice; the client receives only the supplied snapshot."""
    raw = _sanitize_enum_fields(client.generate_json(SYSTEM_PROMPT, _build_user_prompt(state)))
    try:
        return AdviceResponse.model_validate(raw)
    except ValidationError as e:
        raise OllamaError(f"Model returned JSON that doesn't match the expected advice shape:\n{raw}\n\n{e}") from e
