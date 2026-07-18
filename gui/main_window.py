"""
Bosman AI Coach — Phase 2 GUI.

This is a thin wrapper: it builds a MatchState from form fields, calls
core.reasoning_engine.get_advice() on a background thread (so the window
doesn't freeze while Ollama thinks), and renders the returned
AdviceResponse. No tactical logic lives here — all of that stays in core/.

Run with: python gui/main_window.py   (or the run_gui.py launcher at the
project root, once you're not calling this file directly from elsewhere).
"""

from __future__ import annotations
import json
import sys
from pathlib import Path

import yaml
from pydantic import ValidationError
from PySide6.QtCore import Qt, QThread, Signal, QObject
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QSpinBox, QComboBox, QCheckBox, QLineEdit, QPushButton,
    QLabel, QTextEdit, QSlider, QMessageBox, QScrollArea, QSizePolicy, QFrame,
    QGraphicsDropShadowEffect,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.schemas import MatchState, AdviceResponse, Formation, PlayingStyle, PossessionTrend, MatchHalf, AppConfig
from core.ollama_client import OllamaClient, OllamaError
from core.reasoning_engine import get_advice
from voice.tts import TTSEngine, advice_to_speech_text
from voice.stt import VoiceInputEngine

CONFIG_PATH = ROOT / "config.yaml"
SCENARIOS_PATH = ROOT / "data" / "simulated_scenarios.json"
DEFAULT_VOSK_MODEL_PATH = ROOT / "voice" / "model"
BACKGROUND_PATH = Path(__file__).parent / "assets" / "coach_bg.jpg"

THREAT_SIDE_OPTIONS = ["(none)", "left_wing", "right_wing", "through_middle"]


class CoverBackgroundWidget(QWidget):
    """Paint ``coach_bg.jpg`` behind child widgets using a cover crop.

    A QPixmap is scaled with KeepAspectRatioByExpanding, then centered and
    clipped by the widget, so resize/maximize never stretches the photo. The
    dark overlay is painted here rather than relying on stylesheet alpha, which
    keeps every form panel independently opaque and legible.
    """

    def __init__(self, image_path: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self._background = QPixmap(str(image_path)) if image_path.exists() else QPixmap()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        painter = QPainter(self)
        target = self.rect()
        if self._background.isNull():
            painter.fillRect(target, QColor("#071426"))
        else:
            scaled = self._background.scaled(
                target.size(), Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation
            )
            x = (target.width() - scaled.width()) // 2
            y = (target.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        # Slightly stronger than the first pass so translucent panels remain
        # legible over floodlights/crowd detail without hiding the photo.
        painter.fillRect(target, QColor(0, 0, 0, 153))  # approximately rgba(0,0,0,0.60)


def load_config() -> AppConfig:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            raw = yaml.safe_load(f) or {}
        return AppConfig.model_validate(raw)
    return AppConfig()


def load_scenarios() -> list[dict]:
    if SCENARIOS_PATH.exists():
        with open(SCENARIOS_PATH) as f:
            return json.load(f)["scenarios"]
    return []


class AdviceWorker(QObject):
    """Runs get_advice() off the GUI thread so the window stays responsive
    while waiting on Ollama (which can take anywhere from ~1s to over a
    minute on this hardware)."""

    finished = Signal(object)   # AdviceResponse
    failed = Signal(str)        # error message

    def __init__(self, state: MatchState, client: OllamaClient):
        super().__init__()
        self.state = state
        self.client = client

    def run(self) -> None:
        try:
            advice = get_advice(self.state, self.client)
            self.finished.emit(advice)
        except OllamaError as e:
            self.failed.emit(str(e))
        except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI
            self.failed.emit(f"Unexpected error: {e}")


class CallableWorker(QObject):
    """Runs an arbitrary zero-argument callable off the GUI thread. Used
    for voice.TTSEngine.speak() (blocks for the duration of the speech)
    and voice.VoiceInputEngine.stop_recording() (blocks briefly while
    vosk finalizes the transcript) - same reasoning as AdviceWorker
    above: anything that can take more than a beat must not run on the
    thread that's also repainting the window.
    """

    finished = Signal(object)   # whatever the callable returned
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self) -> None:
        try:
            result = self.fn()
            self.finished.emit(result)
        except Exception as e:  # noqa: BLE001 - surface anything unexpected to the UI
            self.failed.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bosman AI Coach")
        self.resize(900, 760)
        self.setMinimumSize(640, 520)

        self.config = load_config()
        self.client = OllamaClient(self.config)
        self.scenarios = load_scenarios()

        self._thread: QThread | None = None
        self._worker: AdviceWorker | None = None

        # Background-thread bookkeeping shared by the speak/dictate
        # actions - kept separate from the advice thread above so
        # reading advice aloud and asking for new advice can't collide.
        self._voice_thread: QThread | None = None
        self._voice_worker: CallableWorker | None = None
        self._last_advice: AdviceResponse | None = None

        stt_model_path = str(
            getattr(self.config, "voice_stt_model_path", None) or DEFAULT_VOSK_MODEL_PATH
        )
        self.tts_engine = TTSEngine()
        self.stt_engine = VoiceInputEngine(model_path=stt_model_path)

        self._build_ui()
        self._apply_voice_availability()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        """Build scrollable panels over a cover-scaled background.

        The content uses Qt layouts exclusively: QVBoxLayout for the section
        stack, QFormLayout for form rows, and QHBoxLayout for paired controls.
        QScrollArea owns overflow at every window size, so controls are never
        clipped or manually positioned.
        """
        central = CoverBackgroundWidget(BACKGROUND_PATH)
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")

        content = QWidget()
        content.setObjectName("scrollContent")
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 24, 24, 28)
        layout.setSpacing(16)

        title = QLabel("BOSMAN AI COACH")
        title.setObjectName("appTitle")
        title.setAlignment(Qt.AlignCenter)
        subtitle = QLabel("Live tactical guidance — read-only match analysis")
        subtitle.setObjectName("appSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._build_scenario_bar())
        layout.addWidget(self._build_match_form())
        layout.addWidget(self._build_stats_form())

        self.ask_button = QPushButton("ASK COACH")
        self.ask_button.setObjectName("askButton")
        self.ask_button.setMinimumHeight(44)
        self.ask_button.clicked.connect(self._on_ask_coach)
        layout.addWidget(self.ask_button)

        self.status_label = QLabel("")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)
        layout.addWidget(self._build_output_panel())
        layout.addStretch(1)

        self.scroll_area.setWidget(content)
        outer.addWidget(self.scroll_area)
        self._apply_theme()
        self._apply_text_and_control_shadows(content)

    def _apply_text_and_control_shadows(self, content: QWidget) -> None:
        """Give text-bearing widgets a consistent, subtle floating shadow.

        Qt stylesheets do not implement CSS ``text-shadow``. Applying a
        QGraphicsDropShadowEffect to each leaf control shadows its rendered
        text as well as its outline, so labels, values, buttons and dropdowns
        remain readable over the moving photo background.
        """
        shadowed_types = (
            QGroupBox, QLabel, QLineEdit, QComboBox, QSpinBox, QPushButton,
            QCheckBox, QTextEdit,
        )
        for widget in [content, *content.findChildren(QWidget)]:
            if not isinstance(widget, shadowed_types):
                continue
            shadow = QGraphicsDropShadowEffect(widget)
            shadow.setColor(QColor(0, 0, 0, 205))  # rgba(0, 0, 0, 0.80)
            shadow.setOffset(1.5, 2.0)
            shadow.setBlurRadius(4.0)
            widget.setGraphicsEffect(shadow)

    def _apply_theme(self) -> None:
        """Use translucent panel grouping while preserving high-contrast controls."""
        self.setStyleSheet("""
            QWidget { color: #edf4ff; font-family: Segoe UI, Arial, sans-serif; }
            QWidget#scrollContent { background: transparent; }
            QLabel#appTitle { font-size: 28px; font-weight: 800; letter-spacing: 2px; color: #ffffff; padding-top: 4px; }
            QLabel#appSubtitle { font-size: 13px; font-weight: 500; color: #c8d8eb; padding-bottom: 4px; }
            QGroupBox { background-color: rgba(2, 14, 35, 46); border: 1px solid rgba(141, 212, 255, 155); border-radius: 8px; margin-top: 14px; padding: 16px 14px 14px 14px; font-size: 16px; font-weight: 750; color: #f4f8ff; }
            QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 7px; color: #8dd4ff; background-color: transparent; }
            QLabel { font-size: 13px; font-weight: 600; color: #f4f8ff; }
            QLineEdit, QComboBox, QSpinBox, QTextEdit { background-color: rgba(242, 246, 251, 232); color: #10213b; border: 1px solid #8db9df; border-radius: 4px; min-height: 26px; padding: 3px 7px; font-size: 13px; font-weight: 500; }
            QComboBox::drop-down { border: 0; width: 22px; }
            QSlider::groove:horizontal { height: 7px; border-radius: 3px; background: #476582; }
            QSlider::handle:horizontal { width: 16px; margin: -5px 0; border-radius: 8px; background: #8dd4ff; }
            QPushButton { background-color: #244d78; border: 1px solid #6397c5; border-radius: 5px; padding: 7px 12px; font-size: 13px; font-weight: 700; color: #ffffff; }
            QPushButton:hover { background-color: #32699f; }
            QPushButton#askButton { background-color: #11a36d; border-color: #6de4b8; font-size: 16px; font-weight: 800; letter-spacing: 1px; }
            QPushButton#askButton:hover { background-color: #16ba7c; }
            QTextEdit { min-height: 120px; }
            QLabel#statusLabel { color: #d0e2f4; font-size: 12px; font-weight: 600; }
            QCheckBox { font-size: 13px; font-weight: 600; spacing: 7px; }
        """)

    def _build_scenario_bar(self) -> QWidget:
        box = QGroupBox("Load a simulated scenario (optional)")
        layout = QHBoxLayout(box)

        self.scenario_combo = QComboBox()
        self.scenario_combo.addItem("(manual entry)")
        for s in self.scenarios:
            self.scenario_combo.addItem(s["name"])
        self.scenario_combo.currentIndexChanged.connect(self._on_scenario_selected)
        layout.addWidget(self.scenario_combo)
        return box

    def _build_match_form(self) -> QWidget:
        box = QGroupBox("Match situation")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)

        self.minute_spin = QSpinBox()
        self.minute_spin.setRange(0, 120)
        self.minute_spin.setValue(45)
        form.addRow("Minute:", self.minute_spin)

        self.match_half_combo = QComboBox()
        self.match_half_combo.addItems(["(not tracked)"] + [half.value for half in MatchHalf])
        form.addRow("Match half/state:", self.match_half_combo)

        score_row = QHBoxLayout()
        self.my_score_spin = QSpinBox()
        self.my_score_spin.setRange(0, 20)
        self.opp_score_spin = QSpinBox()
        self.opp_score_spin.setRange(0, 20)
        score_row.addWidget(QLabel("Us:"))
        score_row.addWidget(self.my_score_spin)
        score_row.addWidget(QLabel("Opponent:"))
        score_row.addWidget(self.opp_score_spin)
        score_widget = QWidget()
        score_widget.setLayout(score_row)
        form.addRow("Score:", score_widget)

        self.formation_combo = QComboBox()
        self.formation_combo.addItems([f.value for f in Formation])
        form.addRow("Formation:", self.formation_combo)

        self.style_combo = QComboBox()
        self.style_combo.addItems([s.value for s in PlayingStyle])
        form.addRow("Playing style:", self.style_combo)

        self.possession_slider = QSlider(Qt.Horizontal)
        self.possession_slider.setRange(0, 100)
        self.possession_slider.setValue(50)
        self.possession_label = QLabel("50%")
        self.possession_slider.valueChanged.connect(
            lambda v: self.possession_label.setText(f"{v}%")
        )
        poss_row = QHBoxLayout()
        poss_row.addWidget(self.possession_slider)
        poss_row.addWidget(self.possession_label)
        poss_widget = QWidget()
        poss_widget.setLayout(poss_row)
        form.addRow("Possession:", poss_widget)

        self.possession_trend_combo = QComboBox()
        self.possession_trend_combo.addItems(["(not tracked)"] + [t.value for t in PossessionTrend])
        form.addRow("Possession trend:", self.possession_trend_combo)

        self.threat_side_combo = QComboBox()
        self.threat_side_combo.addItems(THREAT_SIDE_OPTIONS)
        form.addRow("Opponent's main threat:", self.threat_side_combo)

        self.team_stamina_spin = QSpinBox()
        self.team_stamina_spin.setRange(0, 100)
        self.team_stamina_spin.setValue(100)
        self.team_stamina_spin.setSpecialValueText("(not tracked)")
        form.addRow("Team average stamina:", self.team_stamina_spin)

        self.striker_stamina_spin = QSpinBox()
        self.striker_stamina_spin.setRange(0, 100)
        self.striker_stamina_spin.setValue(100)
        self.striker_stamina_spin.setSpecialValueText("(not tracked)")
        form.addRow("Striker stamina:", self.striker_stamina_spin)

        self.key_player_stamina_spin = QSpinBox()
        self.key_player_stamina_spin.setRange(0, 100)
        self.key_player_stamina_spin.setValue(100)
        self.key_player_stamina_spin.setSpecialValueText("(not tracked)")
        form.addRow("Key player stamina:", self.key_player_stamina_spin)

        self.red_card_check = QCheckBox("We are down to 10 men")
        form.addRow("", self.red_card_check)

        self.opponent_red_card_check = QCheckBox("Opponent is down to 10 men")
        form.addRow("", self.opponent_red_card_check)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Optional extra context, e.g. 'keeper on a yellow'")
        self.dictate_button = QPushButton("\U0001F3A4 Dictate")
        self.dictate_button.setCheckable(True)
        self.dictate_button.clicked.connect(self._on_dictate_toggled)
        notes_row = QHBoxLayout()
        notes_row.addWidget(self.notes_edit, 1)
        notes_row.addWidget(self.dictate_button)
        notes_widget = QWidget()
        notes_widget.setLayout(notes_row)
        form.addRow("Notes:", notes_widget)

        return box

    def _build_stats_form(self) -> QWidget:
        box = QGroupBox("Match stats (optional — pause/stats or tactics menus)")
        form = QFormLayout(box)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)

        def _stat_spinbox(maximum: int = 50) -> QSpinBox:
            s = QSpinBox()
            s.setRange(0, maximum)
            s.setValue(0)
            s.setSpecialValueText("(not tracked)")
            return s

        def _paired_row(label_left: str, label_right: str, maximum: int = 50) -> tuple[QSpinBox, QSpinBox, QWidget]:
            left_spin = _stat_spinbox(maximum)
            right_spin = _stat_spinbox(maximum)
            row = QHBoxLayout()
            row.addWidget(QLabel(label_left))
            row.addWidget(left_spin)
            row.addWidget(QLabel(label_right))
            row.addWidget(right_spin)
            widget = QWidget()
            widget.setLayout(row)
            return left_spin, right_spin, widget

        self.shots_spin, self.opp_shots_spin, shots_widget = _paired_row("Us:", "Opponent:")
        form.addRow("Shots:", shots_widget)

        self.shots_on_target_spin, self.opp_shots_on_target_spin, sot_widget = _paired_row("Us:", "Opponent:")
        form.addRow("Shots on target:", sot_widget)

        self.corners_spin, self.opp_corners_spin, corners_widget = _paired_row("Us:", "Opponent:")
        form.addRow("Corners:", corners_widget)

        self.pass_accuracy_spin, self.opp_pass_accuracy_spin, pass_accuracy_widget = _paired_row("Us:", "Opponent:", maximum=100)
        form.addRow("Pass accuracy %:", pass_accuracy_widget)

        self.fouls_spin, self.opp_fouls_spin, fouls_widget = _paired_row("Us:", "Opponent:")
        form.addRow("Fouls committed:", fouls_widget)

        self.menu_formation_combo = QComboBox()
        self.menu_formation_combo.addItems(["(not tracked)"] + [f.value for f in Formation])
        form.addRow("Tactics-menu formation:", self.menu_formation_combo)

        self.my_yellows_spin, self.opp_yellows_spin, yellows_widget = _paired_row("Us:", "Opponent:")
        form.addRow("Yellow cards:", yellows_widget)

        return box

    def _build_output_panel(self) -> QWidget:
        box = QGroupBox("Recommendation")
        layout = QVBoxLayout(box)

        header_row = QHBoxLayout()
        header_row.addStretch()
        self.read_aloud_check = QCheckBox("\U0001F50A Read aloud when ready")
        header_row.addWidget(self.read_aloud_check)
        self.speak_now_button = QPushButton("\U0001F50A Speak")
        self.speak_now_button.clicked.connect(self._on_speak_now)
        self.speak_now_button.setEnabled(False)  # enabled once advice exists
        header_row.addWidget(self.speak_now_button)
        header_widget = QWidget()
        header_widget.setLayout(header_row)
        layout.addWidget(header_widget)

        self.output_view = QTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setPlaceholderText(
            "Fill in the match situation above (or load a scenario) and click ASK COACH."
        )
        layout.addWidget(self.output_view)
        return box

    def _apply_voice_availability(self) -> None:
        """Voice is genuinely optional - if pyttsx3/vosk/a mic aren't
        set up, disable the relevant controls rather than let them throw
        a confusing error the first time they're clicked, same approach
        as the "Ollama unreachable" check already used for advice."""
        if not self.tts_engine.is_available():
            self.read_aloud_check.setEnabled(False)
            self.speak_now_button.setEnabled(False)
            reason = self.tts_engine.status.reason or "Text-to-speech unavailable."
            self.read_aloud_check.setToolTip(reason)
            self.speak_now_button.setToolTip(reason)

        if not self.stt_engine.is_available():
            self.dictate_button.setEnabled(False)
            reason = self.stt_engine.status.reason or "Speech-to-text unavailable."
            self.dictate_button.setToolTip(reason)

    # ------------------------------------------------------------------
    # Scenario loading
    # ------------------------------------------------------------------
    def _on_scenario_selected(self, index: int) -> None:
        if index <= 0:
            return
        scenario = self.scenarios[index - 1]
        try:
            state = MatchState.model_validate(scenario["state"])
            self._populate_form_from_state(state)
        except (ValidationError, KeyError, ValueError) as e:
            QMessageBox.warning(
                self, "Couldn't load scenario",
                f"'{scenario.get('name', '?')}' couldn't be loaded: {e}\n\n"
                "The form below hasn't been changed - manual entry still works."
            )
            # Reset the picker so it doesn't look like this scenario is
            # active when it actually failed to apply.
            self.scenario_combo.blockSignals(True)
            self.scenario_combo.setCurrentIndex(0)
            self.scenario_combo.blockSignals(False)

    def _populate_form_from_state(self, state: MatchState) -> None:
        self.minute_spin.setValue(state.minute)
        self.match_half_combo.setCurrentText(state.match_half.value if state.match_half else "(not tracked)")
        self.my_score_spin.setValue(state.my_score)
        self.opp_score_spin.setValue(state.opponent_score)
        self.formation_combo.setCurrentText(state.formation.value)
        self.style_combo.setCurrentText(state.playing_style.value)
        self.possession_slider.setValue(state.possession_pct)
        self.possession_trend_combo.setCurrentText(
            state.possession_trend.value if state.possession_trend else "(not tracked)"
        )
        self.threat_side_combo.setCurrentText(state.opponent_threat_side or "(none)")
        self.team_stamina_spin.setValue(
            state.team_stamina_avg_pct if state.team_stamina_avg_pct is not None else 100
        )
        self.striker_stamina_spin.setValue(
            state.striker_stamina_pct if state.striker_stamina_pct is not None else 100
        )
        self.key_player_stamina_spin.setValue(
            state.key_player_stamina_pct if state.key_player_stamina_pct is not None else 100
        )
        self.shots_spin.setValue(state.shots or 0)
        self.opp_shots_spin.setValue(state.opponent_shots or 0)
        self.shots_on_target_spin.setValue(state.shots_on_target or 0)
        self.opp_shots_on_target_spin.setValue(state.opponent_shots_on_target or 0)
        self.corners_spin.setValue(state.corners or 0)
        self.opp_corners_spin.setValue(state.opponent_corners or 0)
        self.pass_accuracy_spin.setValue(state.pass_accuracy_pct or 0)
        self.opp_pass_accuracy_spin.setValue(state.opponent_pass_accuracy_pct or 0)
        self.fouls_spin.setValue(state.fouls_committed or 0)
        self.opp_fouls_spin.setValue(state.opponent_fouls_committed or 0)
        self.menu_formation_combo.setCurrentText(state.menu_formation.value if state.menu_formation else "(not tracked)")
        self.my_yellows_spin.setValue(state.my_yellow_cards or 0)
        self.opp_yellows_spin.setValue(state.opponent_yellow_cards or 0)
        self.red_card_check.setChecked(state.red_card)
        self.opponent_red_card_check.setChecked(state.opponent_red_card)
        self.notes_edit.setText(state.notes or "")

    # ------------------------------------------------------------------
    # Form -> MatchState
    # ------------------------------------------------------------------
    def _build_match_state(self) -> MatchState:
        threat_side = self.threat_side_combo.currentText()
        trend = self.possession_trend_combo.currentText()
        match_half = self.match_half_combo.currentText()
        menu_formation = self.menu_formation_combo.currentText()

        def _optional_stat(spin: QSpinBox):
            return spin.value() if spin.value() > 0 else None

        return MatchState(
            minute=self.minute_spin.value(),
            my_score=self.my_score_spin.value(),
            opponent_score=self.opp_score_spin.value(),
            formation=Formation(self.formation_combo.currentText()),
            playing_style=PlayingStyle(self.style_combo.currentText()),
            possession_pct=self.possession_slider.value(),
            match_half=None if match_half == "(not tracked)" else MatchHalf(match_half),
            possession_trend=None if trend == "(not tracked)" else PossessionTrend(trend),
            opponent_threat_side=None if threat_side == "(none)" else threat_side,
            team_stamina_avg_pct=(
                None if self.team_stamina_spin.value() == 100
                else self.team_stamina_spin.value()
            ),
            striker_stamina_pct=(
                None if self.striker_stamina_spin.value() == 100
                else self.striker_stamina_spin.value()
            ),
            key_player_stamina_pct=(
                None if self.key_player_stamina_spin.value() == 100
                else self.key_player_stamina_spin.value()
            ),
            shots=_optional_stat(self.shots_spin),
            opponent_shots=_optional_stat(self.opp_shots_spin),
            shots_on_target=_optional_stat(self.shots_on_target_spin),
            opponent_shots_on_target=_optional_stat(self.opp_shots_on_target_spin),
            corners=_optional_stat(self.corners_spin),
            opponent_corners=_optional_stat(self.opp_corners_spin),
            pass_accuracy_pct=_optional_stat(self.pass_accuracy_spin),
            opponent_pass_accuracy_pct=_optional_stat(self.opp_pass_accuracy_spin),
            fouls_committed=_optional_stat(self.fouls_spin),
            opponent_fouls_committed=_optional_stat(self.opp_fouls_spin),
            menu_formation=None if menu_formation == "(not tracked)" else Formation(menu_formation),
            my_yellow_cards=_optional_stat(self.my_yellows_spin),
            opponent_yellow_cards=_optional_stat(self.opp_yellows_spin),
            red_card=self.red_card_check.isChecked(),
            opponent_red_card=self.opponent_red_card_check.isChecked(),
            notes=self.notes_edit.text() or None,
        )

    # ------------------------------------------------------------------
    # Ask coach
    # ------------------------------------------------------------------
    def _on_ask_coach(self) -> None:
        if not self.client.health_check():
            QMessageBox.warning(
                self, "Ollama unreachable",
                f"Could not reach Ollama at {self.config.ollama_host}.\n"
                f"Make sure `ollama serve` is running and {self.config.model} is pulled."
            )
            return

        try:
            state = self._build_match_state()
        except Exception as e:  # noqa: BLE001 - show validation errors to the user
            QMessageBox.warning(self, "Invalid match situation", str(e))
            return

        self.ask_button.setEnabled(False)
        self.ask_button.setText("THINKING...")
        self.status_label.setText(f"Asking {self.config.model} for advice — this can take a while on CPU...")
        self.output_view.clear()

        self._thread = QThread()
        self._worker = AdviceWorker(state, self.client)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.finished.connect(self._on_advice_ready)
        self._worker.failed.connect(self._on_advice_failed)
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._reset_ask_button)

        self._thread.start()

    def _reset_ask_button(self) -> None:
        self.ask_button.setEnabled(True)
        self.ask_button.setText("ASK COACH")

    def _on_advice_ready(self, advice: AdviceResponse) -> None:
        self.status_label.setText("Done.")
        self.output_view.setHtml(self._format_advice_html(advice))
        self._last_advice = advice
        if self.tts_engine.is_available():
            self.speak_now_button.setEnabled(True)
            if self.read_aloud_check.isChecked():
                self._speak(advice)

    def _on_advice_failed(self, message: str) -> None:
        self.status_label.setText("Error.")
        QMessageBox.critical(self, "Error getting advice", message)

    # ------------------------------------------------------------------
    # Voice: speak the advice aloud
    # ------------------------------------------------------------------
    def _on_speak_now(self) -> None:
        if getattr(self, "_last_advice", None) is not None:
            self._speak(self._last_advice)

    def _speak(self, advice: AdviceResponse) -> None:
        if self._voice_thread is not None:
            return  # already speaking or dictating - don't overlap
        self.speak_now_button.setEnabled(False)
        self.status_label.setText("Speaking...")
        self._run_voice_task(
            fn=lambda: self.tts_engine.speak_advice(advice),
            on_done=lambda _result: self._on_speak_finished(),
            on_error=lambda msg: self._on_speak_failed(msg),
        )

    def _on_speak_finished(self) -> None:
        self.status_label.setText("Done.")
        self.speak_now_button.setEnabled(True)

    def _on_speak_failed(self, message: str) -> None:
        self.status_label.setText("Error.")
        self.speak_now_button.setEnabled(True)
        QMessageBox.warning(self, "Couldn't speak advice", message)

    # ------------------------------------------------------------------
    # Voice: dictate the notes field
    # ------------------------------------------------------------------
    def _on_dictate_toggled(self, checked: bool) -> None:
        if checked:
            try:
                self.stt_engine.start_recording()
            except RuntimeError as e:
                self.dictate_button.setChecked(False)
                QMessageBox.warning(self, "Couldn't start dictation", str(e))
                return
            self.dictate_button.setText("\u23F9 Stop && transcribe")
            self.status_label.setText("Listening... click again when you're done talking.")
        else:
            self.dictate_button.setEnabled(False)
            self.dictate_button.setText("Transcribing...")
            self.status_label.setText("Transcribing...")
            self._run_voice_task(
                fn=self.stt_engine.stop_recording,
                on_done=self._on_dictation_transcribed,
                on_error=self._on_dictation_failed,
            )

    def _on_dictation_transcribed(self, text: str) -> None:
        self.dictate_button.setEnabled(True)
        self.dictate_button.setText("\U0001F3A4 Dictate")
        self.status_label.setText("Done." if text else "Heard nothing - try again.")
        if text:
            existing = self.notes_edit.text().strip()
            self.notes_edit.setText(f"{existing} {text}".strip() if existing else text)

    def _on_dictation_failed(self, message: str) -> None:
        self.dictate_button.setEnabled(True)
        self.dictate_button.setChecked(False)
        self.dictate_button.setText("\U0001F3A4 Dictate")
        self.status_label.setText("Error.")
        QMessageBox.warning(self, "Dictation error", message)

    # ------------------------------------------------------------------
    # Shared background-thread runner for voice actions
    # ------------------------------------------------------------------
    def _run_voice_task(self, fn, on_done, on_error) -> None:
        self._voice_thread = QThread()
        self._voice_worker = CallableWorker(fn)
        self._voice_worker.moveToThread(self._voice_thread)

        self._voice_thread.started.connect(self._voice_worker.run)
        self._voice_worker.finished.connect(on_done)
        self._voice_worker.failed.connect(on_error)
        self._voice_worker.finished.connect(self._voice_thread.quit)
        self._voice_worker.failed.connect(self._voice_thread.quit)
        self._voice_thread.finished.connect(self._clear_voice_thread)

        self._voice_thread.start()

    def _clear_voice_thread(self) -> None:
        self._voice_thread = None
        self._voice_worker = None

    @staticmethod
    def _format_advice_html(advice: AdviceResponse) -> str:
        parts = [f"<h3>Top suggestion: {advice.top_suggestion}</h3>"]
        if advice.formation_change:
            parts.append(f"<b>Formation change:</b> {advice.formation_change.value}<br>")
        if advice.style_change:
            parts.append(f"<b>Playing style:</b> {advice.style_change.value}<br>")
        if advice.secondary_considerations:
            parts.append("<b>Also consider:</b><ul>")
            parts += [f"<li>{item}</li>" for item in advice.secondary_considerations]
            parts.append("</ul>")
        return "".join(parts)


def main() -> None:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
