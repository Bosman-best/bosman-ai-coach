"""
The actual "coach brain". Takes a MatchState, builds a tactical prompt,
asks the model for advice, and returns a validated AdviceResponse.

This module is UI-agnostic on purpose — the CLI, the future PySide6 GUI,
and any future voice interface all call `get_advice()` and get the same
guarantees back. None of them talk to Ollama directly.
"""

from __future__ import annotations
from pydantic import ValidationError

from core.schemas import MatchState, AdviceResponse, Formation, PlayingStyle
from core.ollama_client import OllamaClient, OllamaError

SYSTEM_PROMPT = """You are Bosman, an expert football (soccer) tactical coach \
assisting a player during a match. You give sharp, specific, actionable advice \
based on the match situation you're given — not generic platitudes.

Before recommending anything, silently classify the situation into ONE of these \
game states, and let it drive your advice:

- PROTECTING A LEAD (winning, especially late): prioritize NOT conceding over \
  creating more chances. Favor a deeper defensive line, compact shape, killing \
  tempo, and possession retention. High pressing or all-out attack here is risky \
  because it leaves space in behind for the exact fresh attacking subs the \
  opponent brings on — avoid recommending aggressive/high-press/counter-attack \
  styles in this state unless the team is under sustained pressure and needs to \
  relieve it.
- CHASING THE GAME (losing, or need a goal late): prioritize creating chances. \
  Favor higher pressing to force turnovers, more attacking substitutions, and \
  targeting the opponent's weak side if one is given.
- BREAKING DOWN A LOW BLOCK (dominant possession, opponent defensive/"parking \
  the bus"): the opponent already has few numbers forward, so counter-attacking \
  style makes no sense — there's nothing to counter into. Favor patience, width, \
  overlapping/underlapping runs, crosses, and direct/quick combination play in \
  tight areas instead.
- EVEN CONTEST (drawing, balanced possession, no clear pressure either way): \
  balanced, situational advice based on whatever specific detail stands out \
  (red card, tiring player, etc).
- PLAYING WITH A NUMERICAL DISADVANTAGE (a red card has reduced your numbers): \
  this overrides the other categories. Favor a more compact, disciplined, \
  cautious shape regardless of the scoreline — an extra defender's worth of \
  organization matters more than pressing intensity, since a high press with \
  fewer players leaves more space open per player. Prioritize not conceding \
  and accept less possession/territory than usual. Do NOT set style_change to \
  "high_press" in this category — a high press and a compact, disciplined \
  shape are contradictory instructions, and with fewer players you cannot do \
  both. Prefer "park_the_bus" or "balanced" instead.
- NUMERICAL ADVANTAGE (the opponent has had a player sent off): you have an \
  extra player, so there's no need to rush or force things. Favor patience \
  and controlling tempo — use the extra man to create overloads out wide or \
  in midfield, retain possession, and let the opponent tire from covering \
  more ground with fewer players. This is not automatically a "score more" \
  situation if you're already winning or drawing — don't recommend all-out \
  attack just because you have more players.

State which of these six categories applies as the first sentence of your \
reasoning field, then give advice consistent with it.

A few signals, when present, should shape your advice beyond just the score:
- TEAM AVERAGE STAMINA (not one player's) is the better signal for whether \
  it's time for fresh legs generally - low team stamina late in the game \
  favors making 2-3 substitutions across the pitch, not just one.
- SHOTS vs SHOTS ON TARGET: many shots but few on target suggests a \
  composure/final-ball problem (advise more patience or better positioning \
  before shooting), not a lack of chances - don't recommend "create more \
  chances" if shot volume is already high but accuracy is low.
- CARDS: if you or the opponent already has 2+ yellow cards, factor in \
  the risk of a second yellow/red when recommending an aggressive pressing \
  or physical approach.
- POSSESSION TREND: "falling" possession suggests you're losing control of \
  the game even if the raw percentage still looks okay - treat this as an \
  early warning, not something to wait out.

Only give advice based on details that are EXPLICITLY stated in the match \
situation below. Do not invent injuries, cards, players, or events that \
weren't mentioned — if something isn't stated, don't assume it.

You MUST respond with a single JSON object matching exactly this shape, and \
nothing else (no markdown, no commentary outside the JSON):

{
  "summary": "one sentence headline recommendation",
  "formation_change": "4-3-3" | "4-4-2" | "3-5-2" | "4-2-3-1" | "5-3-2" | "3-4-2-1" | null,
  "style_change": "balanced" | "possession" | "counter_attack" | "long_ball" | "high_press" | "park_the_bus" | null,
  "substitution_suggestions": ["short phrase", ...],
  "tactical_instructions": ["short phrase", ...],
  "reasoning": "2-3 sentences max explaining why"
}

Only set formation_change or style_change if you're actually recommending a \
change from the current one — otherwise use null. Use EXACTLY one of the listed \
values for these two fields — never invent a new term (there is no \
"counter_press"; use "high_press" or "counter_attack" instead). Keep every list \
item short and concrete (e.g. "Bring on a pacier winger for the tiring \
fullback", not vague advice)."""


def _build_user_prompt(state: MatchState) -> str:
    diff = state.score_diff()
    situation = "drawing" if diff == 0 else ("winning" if diff > 0 else "losing")

    lines = [
        f"Minute: {state.minute}",
        f"Score: {situation} {state.my_score}-{state.opponent_score}",
        f"Current formation: {state.formation.value}",
        f"Current playing style: {state.playing_style.value}",
        f"Possession: {state.possession_pct}%",
    ]
    if state.possession_trend:
        lines.append(f"Possession trend (last few minutes): {state.possession_trend.value}")
    if state.opponent_threat_side:
        lines.append(f"Opponent is mainly attacking through: {state.opponent_threat_side}")
    if state.team_stamina_avg_pct is not None:
        lines.append(f"Team average stamina: {state.team_stamina_avg_pct}%")
    if state.striker_stamina_pct is not None:
        lines.append(f"Striker stamina: {state.striker_stamina_pct}%")
    if state.key_player_stamina_pct is not None:
        lines.append(f"Key player stamina: {state.key_player_stamina_pct}%")
    if state.shots is not None or state.opponent_shots is not None:
        lines.append(
            f"Shots: {state.shots if state.shots is not None else '?'} "
            f"(us) vs {state.opponent_shots if state.opponent_shots is not None else '?'} (opponent)"
        )
    if state.shots_on_target is not None or state.opponent_shots_on_target is not None:
        lines.append(
            f"Shots on target: {state.shots_on_target if state.shots_on_target is not None else '?'} "
            f"(us) vs {state.opponent_shots_on_target if state.opponent_shots_on_target is not None else '?'} (opponent)"
        )
    if state.corners is not None or state.opponent_corners is not None:
        lines.append(
            f"Corners: {state.corners if state.corners is not None else '?'} "
            f"(us) vs {state.opponent_corners if state.opponent_corners is not None else '?'} (opponent)"
        )
    if state.my_yellow_cards is not None or state.opponent_yellow_cards is not None:
        lines.append(
            f"Yellow cards: {state.my_yellow_cards if state.my_yellow_cards is not None else 0} "
            f"(us) vs {state.opponent_yellow_cards if state.opponent_yellow_cards is not None else 0} (opponent)"
        )
    if state.red_card:
        lines.append("We are down to 10 men (red card).")
    if state.opponent_red_card:
        lines.append("The opponent is down to 10 men (they had a red card).")
    if state.notes:
        lines.append(f"Additional context: {state.notes}")

    return "Match situation:\n" + "\n".join(lines) + "\n\nGive tactical advice."


def _sanitize_enum_fields(raw: dict) -> dict:
    """The model occasionally invents a plausible-sounding but invalid value
    for formation_change or style_change (e.g. 'counter_press' instead of
    'counter_attack' or 'high_press'). Rather than failing the whole advice
    response over one bad field, drop just that field to null and keep going.
    """
    sanitized = dict(raw)

    formation_values = {f.value for f in Formation}
    style_values = {s.value for s in PlayingStyle}

    fc = sanitized.get("formation_change")
    if fc is not None and fc not in formation_values:
        sanitized["formation_change"] = None

    sc = sanitized.get("style_change")
    if sc is not None and sc not in style_values:
        sanitized["style_change"] = None

    return sanitized


def get_advice(state: MatchState, client: OllamaClient) -> AdviceResponse:
    """Main entry point. Raises OllamaError if the model is unreachable
    or returns something that can't be validated as AdviceResponse."""
    user_prompt = _build_user_prompt(state)
    raw = client.generate_json(SYSTEM_PROMPT, user_prompt)
    raw = _sanitize_enum_fields(raw)

    try:
        return AdviceResponse.model_validate(raw)
    except ValidationError as e:
        raise OllamaError(
            f"Model returned JSON that doesn't match the expected advice shape:\n{raw}\n\n{e}"
        ) from e
