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
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QSpinBox, QComboBox, QCheckBox, QLineEdit, QPushButton,
    QLabel, QTextEdit, QSlider, QMessageBox,
)

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.schemas import MatchState, AdviceResponse, Formation, PlayingStyle, PossessionTrend, AppConfig
from core.ollama_client import OllamaClient, OllamaError
from core.reasoning_engine import get_advice
from voice.tts import TTSEngine, advice_to_speech_text
from voice.stt import VoiceInputEngine

CONFIG_PATH = ROOT / "config.yaml"
SCENARIOS_PATH = ROOT / "data" / "simulated_scenarios.json"
DEFAULT_VOSK_MODEL_PATH = ROOT / "voice" / "model"
BANNER_PATH = Path(__file__).parent / "assets" / "header_banner.png"

THREAT_SIDE_OPTIONS = ["(none)", "left_wing", "right_wing", "through_middle"]


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
        self.resize(760, 700)

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
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)

        root_layout.addWidget(self._build_header_banner())
        root_layout.addWidget(self._build_scenario_bar())
        root_layout.addWidget(self._build_match_form())
        root_layout.addWidget(self._build_stats_form())

        self.ask_button = QPushButton("ASK COACH")
        self.ask_button.setMinimumHeight(40)
        self.ask_button.clicked.connect(self._on_ask_coach)
        root_layout.addWidget(self.ask_button)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: gray;")
        root_layout.addWidget(self.status_label)

        root_layout.addWidget(self._build_output_panel())

    def _build_header_banner(self) -> QWidget:
        """Illustrated header banner - original artwork generated by
        gui/assets/generate_banner.py, no real people or club branding
        (see that file's docstring for why). Falls back to a plain title
        label if the PNG is ever missing, so a bad/absent asset can't
        break the app from launching."""
        label = QLabel()
        pixmap = QPixmap(str(BANNER_PATH)) if BANNER_PATH.exists() else QPixmap()
        if not pixmap.isNull():
            label.setPixmap(pixmap)
            label.setScaledContents(True)
            label.setFixedHeight(160)
        else:
            label.setText("BOSMAN AI COACH")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(
                "background-color: #0d1b36; color: white; font-size: 28px; "
                "font-weight: bold; padding: 24px;"
            )
        return label

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

        self.minute_spin = QSpinBox()
        self.minute_spin.setRange(0, 120)
        self.minute_spin.setValue(45)
        form.addRow("Minute:", self.minute_spin)

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
        box = QGroupBox("Match stats (optional - from the pause/stats screen)")
        form = QFormLayout(box)

        def _stat_spinbox() -> QSpinBox:
            s = QSpinBox()
            s.setRange(0, 50)
            s.setValue(0)
            s.setSpecialValueText("(not tracked)")
            return s

        def _paired_row(label_left: str, label_right: str) -> tuple[QSpinBox, QSpinBox, QWidget]:
            left_spin = _stat_spinbox()
            right_spin = _stat_spinbox()
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

        def _optional_stat(spin: QSpinBox):
            return spin.value() if spin.value() > 0 else None

        return MatchState(
            minute=self.minute_spin.value(),
            my_score=self.my_score_spin.value(),
            opponent_score=self.opp_score_spin.value(),
            formation=Formation(self.formation_combo.currentText()),
            playing_style=PlayingStyle(self.style_combo.currentText()),
            possession_pct=self.possession_slider.value(),
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
