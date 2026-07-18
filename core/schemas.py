"""
Data contracts for the Bosman AI Coach.

Every module in the system (data input, GUI, future vision/voice modules)
reads or writes ONLY these shapes. Nothing else should define its own
ad-hoc dict for match data or advice — that's what keeps swapping the
input source (manual -> simulated -> OCR) from requiring a rewrite.
"""

from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class Formation(str, Enum):
    F_442 = "4-4-2"
    F_433 = "4-3-3"
    F_352 = "3-5-2"
    F_4231 = "4-2-3-1"
    F_532 = "5-3-2"
    F_3421 = "3-4-2-1"


class PlayingStyle(str, Enum):
    BALANCED = "balanced"
    POSSESSION = "possession"
    COUNTER_ATTACK = "counter_attack"
    LONG_BALL = "long_ball"
    HIGH_PRESS = "high_press"
    PARK_THE_BUS = "park_the_bus"


class PossessionTrend(str, Enum):
    RISING = "rising"        # gaining more of the ball recently
    FALLING = "falling"      # losing more of the ball recently
    STABLE = "stable"


class MatchState(BaseModel):
    """A snapshot of the match at one point in time. This is the ONLY
    thing the reasoning engine ever sees — it doesn't care where it came
    from (manual entry, a simulator, or vision reading FIFA's HUD/stats
    screens). This intentionally does NOT include full player positions —
    that isn't reliably readable from a game screen, so we don't pretend
    otherwise. It focuses on team-level signals that ARE genuinely
    extractable: score, clock, stats-screen numbers, and stamina averaged
    across the squad rather than a single player."""

    minute: int = Field(ge=0, le=120)
    my_score: int = Field(ge=0)
    opponent_score: int = Field(ge=0)
    formation: Formation
    playing_style: PlayingStyle = PlayingStyle.BALANCED
    possession_pct: int = Field(ge=0, le=100)
    possession_trend: Optional[PossessionTrend] = Field(
        default=None, description="How possession has shifted over the last few minutes"
    )
    opponent_threat_side: Optional[str] = Field(
        default=None, description="e.g. 'left_wing', 'right_wing', 'through_middle'"
    )

    # Team-wide fatigue, not a single player - far more representative of
    # whether it's time to make changes than one striker's stamina bar.
    team_stamina_avg_pct: Optional[int] = Field(default=None, ge=0, le=100)
    # Kept for cases where one specific player's fatigue matters on top of
    # the team average (e.g. your one key creative player is gassed even
    # though the team overall is fine).
    striker_stamina_pct: Optional[int] = Field(default=None, ge=0, le=100)
    key_player_stamina_pct: Optional[int] = Field(default=None, ge=0, le=100)

    # Match-stats-screen numbers - genuine OCR targets, not tracking.
    shots: Optional[int] = Field(default=None, ge=0)
    shots_on_target: Optional[int] = Field(default=None, ge=0)
    opponent_shots: Optional[int] = Field(default=None, ge=0)
    opponent_shots_on_target: Optional[int] = Field(default=None, ge=0)
    corners: Optional[int] = Field(default=None, ge=0)
    opponent_corners: Optional[int] = Field(default=None, ge=0)
    my_yellow_cards: Optional[int] = Field(default=None, ge=0)
    opponent_yellow_cards: Optional[int] = Field(default=None, ge=0)

    red_card: bool = False
    opponent_red_card: bool = False
    notes: Optional[str] = Field(
        default=None, description="Free-text extra context, e.g. 'keeper on a yellow'"
    )

    def score_diff(self) -> int:
        return self.my_score - self.opponent_score


class AdviceResponse(BaseModel):
    """Structured advice returned by the reasoning engine. Requesting
    JSON output in this exact shape (rather than parsing free text) is
    what makes the GUI layer trivial and deterministic."""

    summary: str = Field(description="One sentence, the headline recommendation")
    formation_change: Optional[Formation] = None
    style_change: Optional[PlayingStyle] = None
    substitution_suggestions: list[str] = Field(default_factory=list)
    tactical_instructions: list[str] = Field(default_factory=list)
    reasoning: str = Field(description="Short explanation of WHY, 2-3 sentences max")


class AppConfig(BaseModel):
    ollama_host: str = "http://localhost:11434"
    model: str = "qwen2.5:3b"
    request_timeout_s: int = 30

    # Phase 5 (voice) - path to a downloaded vosk model directory. None
    # (the default) falls back to voice/model in the GUI. Only needed if
    # you want dictation; TTS (voice.tts) has no model-path dependency.
    voice_stt_model_path: Optional[str] = None
