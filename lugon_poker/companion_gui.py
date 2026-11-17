"""Lugon Poker Lab — Live Screen Capture & Strategic Assistant Window.

Supports manual hand input and real-time Windows table capture with one fixed
local model (qwen3-vl:2b-instruct) for visual reading and strategic explanations.
"""

from __future__ import annotations

import argparse
import ctypes
from dataclasses import replace
import json
import os
from pathlib import Path
import re
import threading
import time
import tkinter as tk
from typing import Any, Callable, Mapping, Sequence

from . import __version__
from .agent_bridge import (
    DEFAULT_BRIDGE_DIR,
    AgentBridge,
    build_decision_context_from_inputs,
    discover_local_models,
)
from .assistant import (
    LegalAction,
    normalize_game_type,
    normalize_play_style,
    validate_and_clamp_action,
)
from .three_max_advisor import (
    NODE_LABELS,
    SOLVER_NODES,
    infer_live_three_max_request,
    predict_hand_advice_from_model,
    solve_hand_advice,
)
from .live_vision import (
    DEFAULT_TABLE_PRESET,
    analyze_live_image,
    warmup_vision_model,
)
from .persistence import ProfileStore
from .screen_capture import (
    DEFAULT_CAPTURES_DIR,
    CaptureTarget,
    capture_screen_region,
    capture_target,
    compute_frame_signature,
    enable_dpi_awareness,
    frame_difference,
    get_primary_screen_bounds,
    get_work_area_bounds,
    get_window_hwnd_from_tk,
    get_window_rect,
    image_to_ppm_bytes,
    is_window_maximized,
    is_key_pressed,
    list_windows,
    maximize_window,
    mask_screen_rect_on_image,
    resize_rgb_image,
    save_diagnostic_capture,
    set_window_bounds,
)
from .vision import RGBImage, encode_png_bytes

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "data" / "settings.json"

# Dropdown label -> canonical game type understood by assistant.py / live_vision.py / mercury_vision.py.
GAME_TYPE_LABELS: dict[str, str] = {
    "Spin & Go": "spin_and_go",
    "Tournament (MTT)": "tournament",
}


def game_type_label(game_type: str | None) -> str:
    """Return the dropdown label for a (possibly aliased) game type."""
    key = normalize_game_type(game_type)
    return next(label for label, value in GAME_TYPE_LABELS.items() if value == key)


PLAY_STYLE_LABELS: dict[str, str] = {
    "Balanced": "balanced",
    "Passive": "passive",
    "Aggressive": "aggressive",
    "Creative": "creative",
}


def play_style_label(play_style: str | None) -> str:
    """Return the dropdown label for a (possibly aliased) play style."""
    key = normalize_play_style(play_style)
    return next(label for label, value in PLAY_STYLE_LABELS.items() if value == key)


def load_settings(path: Path) -> dict[str, Any]:
    """Read persisted UI settings; a missing or corrupt file yields defaults instead of an error."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def save_settings(path: Path, settings: dict[str, Any]) -> None:
    """Persist UI settings; failures (read-only disk, etc.) never interrupt play."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    except OSError:
        pass


def _set_windows_app_user_model_id() -> None:
    """Give the Python-hosted window its own Windows taskbar identity and icon."""
    if os.name != "nt":
        return
    try:
        set_app_id = ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID
        set_app_id.argtypes = [ctypes.c_wchar_p]
        set_app_id.restype = ctypes.c_long
        set_app_id("LugonPokerLab.Companion")
    except (AttributeError, OSError):
        pass


_BLINDS_TITLE_RE = re.compile(r"blinds\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def _pick_installed_strategy_model(installed_models: Sequence[str]) -> str | None:
    """Pick the best installed local text/strategy model, excluding pure vision-language models."""
    vision_markers = ("-vl", "vl:", "llava", "bakllava", "moondream", "minicpm-v")
    candidates = [
        name.strip()
        for name in installed_models
        if name and not any(marker in name.lower() for marker in vision_markers)
    ]
    if not candidates:
        return None
    preferences = ("gpt-oss:20b", "gpt-oss", "qwen2.5", "qwen3", "llama3.1", "llama3", "mistral", "gemma")
    for pref in preferences:
        for name in candidates:
            if pref in name.lower():
                return name
    return candidates[0]


def _pick_strategy_model(installed_models: Sequence[str]) -> str:
    """Pick the preferred strategy model name, defaulting to 'gpt-oss:20b' when none are present."""
    return _pick_installed_strategy_model(installed_models) or "gpt-oss:20b"


def _infer_big_blind(
    *,
    report_big_blind: float | None,
    capture_info: Mapping[str, Any] | None,
    pot: float,
    hero_stack: float,
    game_type: str,
) -> float:
    """Infer the effective big blind without mixing BB-display tables and chip-display tables."""
    if isinstance(report_big_blind, (int, float)) and report_big_blind > 0:
        return float(report_big_blind)
    title = str((capture_info or {}).get("title") or "")
    match = _BLINDS_TITLE_RE.search(title)
    if match:
        title_bb = float(match.group(2))
        if title_bb > 0:
            # If displayed stack and pot are tiny relative to title BB (e.g. 2.7 BB vs Blinds 30/60),
            # the client is displaying values directly in Big Blinds.
            if hero_stack < title_bb * 0.5 and pot < title_bb:
                return 1.0
            return title_bb
    if game_type == "spin_and_go" and hero_stack <= 35.0:
        return 1.0
    return 2.0


def evaluate_vision_report(report: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Classify a calibrated or Vision-LLM report into ('ready', 'waiting', or 'failed') without guessing."""
    if not report.get("table_count"):
        reasons = list(report.get("ambiguities") or ['Configured single-table layout was not recognized.'])
        return "failed", {"reasons": reasons}

    if report.get("hero_to_act") is False:
        return "waiting", {"hero_to_act": False}

    ambiguities = list(report.get("ambiguities") or [])
    if ambiguities:
        return "failed", {"reasons": ambiguities}

    if report.get("hero_to_act") is not True:
        return "failed", {"reasons": ["Could not confirm whether it is the Hero's turn."]}

    if report.get("hand_state") is None:
        return "failed", {"reasons": ['Required table fields could not be read.']}

    hs = report["hand_state"]
    players = report.get("players") or []
    hero_obj = next((p for p in players if p.get("hero")), None)
    if hero_obj is None or hero_obj.get("stack") is None:
        return "failed", {"reasons": ["Hero's remaining stack is unreadable; I will not assume 100 chips."]}
    hero_stack = float(hero_obj["stack"])
    hero_pos = str(hero_obj.get("position") or "UNKNOWN") if hero_obj else "UNKNOWN"

    players_by_nick = {
        str(p.get("nickname") or "").strip().lower(): p
        for p in players
        if not p.get("hero") and p.get("nickname")
    }
    opponents: list[dict[str, Any]] = []
    for opp in hs.get("opponents", []):
        nick = str(opp["nickname"])
        p_info = players_by_nick.get(nick.strip().lower(), {})
        entry: dict[str, Any] = {
            "nickname": nick,
            "range": str(opp.get("range") or "*"),
        }
        if p_info.get("position"):
            entry["position"] = str(p_info["position"])
        if isinstance(p_info.get("stack"), (int, float)):
            entry["stack"] = float(p_info["stack"])
        if isinstance(p_info.get("bet"), (int, float)):
            entry["bet"] = float(p_info["bet"])
        opponents.append(entry)

    blinds_raw = report.get("blinds") if isinstance(report.get("blinds"), dict) else {}
    big_blind = blinds_raw.get("big")
    return "ready", {
        "hero_cards": list(hs["hero"]["cards"]),
        "board": list(hs.get("board") or []),
        "pot": float(hs["pot"]),
        "to_call": float(hs.get("to_call", 0.0)),
        "hero_stack": hero_stack,
        "hero_position": hero_pos,
        "big_blind": float(big_blind) if isinstance(big_blind, (int, float)) and big_blind > 0 else None,
        "opponents": opponents,
    }


class AssistantWindow:
    """Compact assistant window for live table capture and manual hand analysis."""

    _VISION_PROVIDER = "ollama"
    _VISION_MODEL = "qwen3-vl:2b-instruct"
    _LIVE_VISION_TIMEOUT_SECONDS = 15.0
    _LIVE_INTERVAL_MS = 2000
    _PRESET_LABELS = {
        "PS Mercury 4-Color": DEFAULT_TABLE_PRESET,
        "888 BLAST": "888_blast",
        "Generic 4-Color": "generic_4color",
    }

    def __init__(
        self,
        root: tk.Tk,
        bridge_dir: Path = DEFAULT_BRIDGE_DIR,
        db_path: Path | None = None,
        captures_dir: Path = DEFAULT_CAPTURES_DIR,
        enable_background_llm: bool = True,
        settings_path: Path | None = None,
    ) -> None:
        self.root = root
        self.root.title(f"Lugon Poker Lab v{__version__}")
        self.bridge = AgentBridge(bridge_dir)
        self.captures_dir = Path(captures_dir)
        # Tests and alternative bridge dirs get an isolated settings file so they never touch the user's real one.
        self.settings_path = Path(settings_path) if settings_path else (
            DEFAULT_SETTINGS_PATH if Path(bridge_dir) == DEFAULT_BRIDGE_DIR else Path(bridge_dir) / "settings.json"
        )
        self._settings = load_settings(self.settings_path)
        self._game_type = normalize_game_type(self._settings.get("game_type"))
        self._play_style = normalize_play_style(self._settings.get("play_style"))
        self.store = ProfileStore(db_path if db_path else ":memory:")
        self._store_closed = False
        self._active_request_id: str | None = None
        self._last_rendered_response_id: str | None = None
        self._analysis_seq = 0
        self._hand_counter = 0
        self._closing = False
        self._enable_background_llm = enable_background_llm
        self._poll_after_id: str | None = None
        self._poll_tick_count = 0
        self._f8_was_down = False
        self._layout_after_id: str | None = None
        self._tiled_table_state: tuple[int, tuple[int, int, int, int], bool] | None = None

        # On-demand & live screen capture state
        self._capture_target = CaptureTarget(kind="fullscreen", title="Full Screen")
        self._capture_seq = 0
        self._capture_in_flight = False
        self._diagnostic_capture_in_flight = False
        self._pending_force_capture = False
        self._live_enabled = False
        self._live_after_id: str | None = None
        self._last_frame_sig: bytes | None = None
        self._last_analyzed_state_key: tuple[Any, ...] | None = None
        self._last_auto_saved_state_key: tuple[Any, ...] | None = None
        self._thumb_photo: tk.PhotoImage | None = None

        try:
            area_x, area_y, area_w, area_h = get_work_area_bounds()
        except RuntimeError:
            area_x, area_y = 0, 0
            area_w, area_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        win_w = min(560, max(380, round(area_w * 0.28)))
        # Tk's geometry height excludes the title bar and window borders.
        win_h = max(480, area_h - 36)
        win_x = area_x + area_w - win_w
        self.root.geometry(f"{win_w}x{win_h}+{win_x}+{area_y}")
        self.root.minsize(360, min(600, win_h))
        self.root.resizable(True, True)
        self.root.configure(bg="#0f172a")
        self.root.attributes("-topmost", True)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        assets_dir = Path(__file__).resolve().parents[1] / "assets"
        ico_file = assets_dir / "lugon_poker_lab.ico"
        png64_file = assets_dir / "lugon_poker_lab_64.png"
        self._logo_img: tk.PhotoImage | None = None
        if ico_file.exists():
            try:
                self.root.iconbitmap(str(ico_file))
            except Exception:
                pass
        if png64_file.exists():
            try:
                self._logo_img = tk.PhotoImage(file=str(png64_file))
                self.root.iconphoto(True, self._logo_img)
            except Exception:
                self._logo_img = None

        self._build_ui()
        self.root.bind("<F8>", self._on_local_f8)
        if self._enable_background_llm:
            threading.Thread(
                target=warmup_vision_model,
                args=(self._VISION_PROVIDER, self._VISION_MODEL),
                daemon=True,
            ).start()
        self._poll_after_id = self.root.after(150, self._poll_agent_response)
        self._layout_after_id = self.root.after(600, self._auto_arrange_poker_layout)

    def _on_local_f8(self, _event: Any = None) -> None:
        self._f8_was_down = True
        self.capture_once(force=True)

    def _schedule_ui(self, callback: Callable[[], Any]) -> None:
        if self._closing:
            return
        try:
            self.root.after(0, callback)
        except (RuntimeError, tk.TclError):
            pass

    def _on_close(self) -> None:
        self._closing = True
        self._stop_live_capture()
        if self._layout_after_id is not None:
            try:
                self.root.after_cancel(self._layout_after_id)
            except tk.TclError:
                pass
            self._layout_after_id = None
        self._restore_tiled_table()
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        if not self._store_closed:
            self.store.close()
            self._store_closed = True
        self._thumb_photo = None
        self._logo_img = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass

    def _build_ui(self) -> None:
        outer = tk.Frame(self.root, bg="#0f172a", padx=10, pady=8)
        outer.pack(fill=tk.BOTH, expand=True)

        # Header: Logo + Title
        top_row = tk.Frame(outer, bg="#0f172a")
        top_row.pack(fill=tk.X, pady=(0, 8))

        if self._logo_img is not None:
            tk.Label(top_row, image=self._logo_img, bg="#0f172a", bd=0).pack(side=tk.LEFT, padx=(0, 8))

        title_col = tk.Frame(top_row, bg="#0f172a")
        title_col.pack(side=tk.LEFT, fill=tk.X, expand=True)
        title_line = tk.Frame(title_col, bg="#0f172a")
        title_line.pack(fill=tk.X)
        tk.Label(
            title_line,
            text="LUGON POKER LAB",
            fg="#facc15",
            bg="#0f172a",
            font=("Segoe UI", 11, "bold"),
            anchor=tk.W,
        ).pack(side=tk.LEFT)
        # Visible build number: captures are archived under data/captures/v<version>/ with the same tag.
        tk.Label(
            title_line,
            text=f"v{__version__}",
            fg="#38bdf8",
            bg="#0f172a",
            font=("Consolas", 9, "bold"),
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(8, 0), pady=(2, 0))

        status_line = tk.Frame(title_col, bg="#0f172a")
        status_line.pack(fill=tk.X)
        self.status_dot = tk.Label(
            status_line,
            text="●",
            fg="#22c55e",
            bg="#0f172a",
            font=("Segoe UI", 10, "bold"),
        )
        self.status_dot.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready to analyze")
        tk.Label(
            status_line,
            textvariable=self.status_var,
            fg="#94a3b8",
            bg="#0f172a",
            font=("Segoe UI", 9),
        ).pack(side=tk.LEFT, padx=(4, 0))

        # Real-Time Screen Capture Card
        capture_card = tk.Frame(
            outer,
            bg="#1e293b",
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        capture_card.pack(fill=tk.X, pady=(0, 8))

        capture_top = tk.Frame(capture_card, bg="#1e293b")
        capture_top.pack(fill=tk.X)

        preview_container = tk.Frame(
            capture_top,
            bg="#0f172a",
            width=156,
            height=88,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        preview_container.pack(side=tk.LEFT, padx=(0, 10))
        preview_container.pack_propagate(False)
        self.preview_label = tk.Label(
            preview_container,
            text="No capture",
            fg="#64748b",
            bg="#0f172a",
            font=("Segoe UI", 10),
        )
        self.preview_label.pack(fill=tk.BOTH, expand=True)

        cap_right = tk.Frame(capture_top, bg="#1e293b")
        cap_right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.capture_info_var = tk.StringVar(value="")
        cap_btns = tk.Frame(cap_right, bg="#1e293b")
        self.capture_buttons_frame = cap_btns
        cap_btns.pack(fill=tk.X, pady=(0, 2))

        def _cap_btn(parent: tk.Frame, text: str, cmd: Callable[[], Any], bg: str = "#334155", bold: bool = False) -> tk.Button:
            return tk.Button(
                parent,
                text=text,
                command=cmd,
                bg=bg,
                fg="#ffffff",
                activebackground="#475569",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                padx=10,
                pady=6,
                font=("Segoe UI", 9, "bold" if bold else "normal"),
                cursor="hand2",
            )

        self.live_btn = _cap_btn(cap_btns, "Auto 2s: OFF", self._toggle_live_capture, bg="#334155", bold=True)
        self.live_btn.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.diagnostic_capture_btn = _cap_btn(
            cap_btns,
            "Capture",
            self.capture_diagnostic_frame,
            bg="#7c3aed",
            bold=True,
        )
        self.diagnostic_capture_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(6, 0))
        self.capture_options_btn = _cap_btn(
            cap_btns,
            'Options ▸',
            self._toggle_capture_options,
            bg="#334155",
        )
        self.capture_options_btn.pack(side=tk.LEFT, padx=(6, 0))
        self.capture_options_panel = tk.Frame(capture_card, bg="#0f172a", padx=10, pady=8)
        self.capture_options_panel.pack(fill=tk.X, pady=(10, 0))
        tk.Label(
            self.capture_options_panel,
            text='ANALYSIS OPTIONS',
            fg="#facc15",
            bg="#0f172a",
            font=("Segoe UI", 10, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(0, 4))

        def _style_opt_menu(opt: tk.OptionMenu) -> None:
            opt.configure(
                bg="#1e293b",
                fg="#f8fafc",
                activebackground="#334155",
                activeforeground="#ffffff",
                relief=tk.FLAT,
                highlightthickness=1,
                highlightbackground="#475569",
                font=("Segoe UI", 10),
                padx=7,
                pady=3,
                anchor=tk.W,
            )
            opt["menu"].configure(bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 10))

        def _add_option_row(
            label_text: str,
            variable: tk.StringVar,
            choices: tuple[str, ...] | list[str],
            explanation: str,
            command: Callable[[str], Any] | None = None,
        ) -> tk.Frame:
            group = tk.Frame(self.capture_options_panel, bg="#0f172a")
            group.pack(fill=tk.X, pady=(2, 4))
            row = tk.Frame(group, bg="#0f172a")
            row.pack(fill=tk.X)
            tk.Label(
                row,
                text=label_text,
                fg="#cbd5e1",
                bg="#0f172a",
                font=("Segoe UI", 10, "bold"),
                width=14,
                anchor=tk.W,
            ).pack(side=tk.LEFT, padx=(0, 6))
            menu = tk.OptionMenu(row, variable, *choices, command=command)
            _style_opt_menu(menu)
            menu.pack(side=tk.LEFT, fill=tk.X, expand=True)
            tk.Label(
                group,
                text=explanation,
                fg="#94a3b8",
                bg="#0f172a",
                font=("Segoe UI", 10),
                anchor=tk.W,
                justify=tk.LEFT,
                wraplength=370,
            ).pack(fill=tk.X, padx=(116, 0), pady=(2, 0))
            return row

        self.game_type_var = tk.StringVar(value=game_type_label(self._game_type))
        _add_option_row(
            "Game",
            self.game_type_var,
            tuple(GAME_TYPE_LABELS.keys()),
            "Spin & Go is a fast three-player format. Tournaments (MTT) include payout pressure.",
            command=lambda _label: self._on_game_type_changed(),
        )

        self.play_style_var = tk.StringVar(value=play_style_label(self._play_style))
        _add_option_row(
            "Style",
            self.play_style_var,
            tuple(PLAY_STYLE_LABELS.keys()),
            "Changes the agent's style: Balanced, Passive (lower risk and smaller sizing), Aggressive (more pressure), or Creative (advanced lines).",
            command=lambda _label: self._on_play_style_changed(),
        )

        reader_group = tk.Frame(self.capture_options_panel, bg="#0f172a")
        reader_group.pack(fill=tk.X, pady=(2, 6))
        reader_row = tk.Frame(reader_group, bg="#0f172a")
        reader_row.pack(fill=tk.X)
        tk.Label(
            reader_row,
            text='Vision model',
            fg="#cbd5e1",
            bg="#0f172a",
            font=("Segoe UI", 10, "bold"),
            width=14,
            anchor=tk.W,
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Label(
            reader_row,
            text="Qwen3-VL 2B Instruct",
            fg="#f8fafc",
            bg="#1e293b",
            font=("Segoe UI", 10, "bold"),
            padx=8,
            pady=4,
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        tk.Label(
            reader_group,
            text='Qwen reads the table. With Auto off, it also explains decisions using the ICM solver result.',
            fg="#94a3b8",
            bg="#0f172a",
            font=("Segoe UI", 10),
            anchor=tk.W,
            justify=tk.LEFT,
            wraplength=370,
        ).pack(fill=tk.X, padx=(116, 0), pady=(2, 0))

        self.preset_var = tk.StringVar(value="888 BLAST")
        _add_option_row(
            "Table layout",
            self.preset_var,
            tuple(self._PRESET_LABELS.keys()),
            "Selects the visual rules and card suits. 888 BLAST is selected automatically when the table opens.",
        )

        self.global_hotkey_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.capture_options_panel,
            text="Global F8 hotkey | analyzes even when another window is active",
            variable=self.global_hotkey_var,
            bg="#0f172a",
            fg="#e2e8f0",
            selectcolor="#1e293b",
            activebackground="#0f172a",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
            bd=0,
            highlightthickness=0,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(3, 0))

        self.save_captures_var = tk.BooleanVar(value=True)
        tk.Checkbutton(
            self.capture_options_panel,
            text="Save frames automatically only on my turn",
            variable=self.save_captures_var,
            bg="#0f172a",
            fg="#e2e8f0",
            selectcolor="#1e293b",
            activebackground="#0f172a",
            activeforeground="#ffffff",
            font=("Segoe UI", 10),
            bd=0,
            highlightthickness=0,
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(1, 0))

        self._capture_options_expanded = False
        self.capture_options_panel.pack_forget()

        # Compact Manual Hand Input Bar
        input_card = tk.Frame(
            outer,
            bg="#1e293b",
            padx=10,
            pady=8,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        input_card.pack(fill=tk.X, pady=(0, 10))

        self.hero_var = tk.StringVar(value="")
        self.board_var = tk.StringVar(value="")
        self.pos_var = tk.StringVar(value="BTN")
        self.pot_var = tk.StringVar(value="")
        self.to_call_var = tk.StringVar(value="")
        self.stack_var = tk.StringVar(value="")
        self.villain_var = tk.StringVar(value='VILLAIN')
        self.range_var = tk.StringVar(value="*")

        def _entry(parent: tk.Frame, var: tk.StringVar, width: int) -> tk.Entry:
            widget = tk.Entry(
                parent,
                textvariable=var,
                width=width,
                bg="#0f172a",
                fg="#f8fafc",
                insertbackground="#f8fafc",
                relief=tk.FLAT,
                font=("Consolas", 9),
                highlightthickness=1,
                highlightbackground="#334155",
            )
            widget.bind("<Return>", lambda _e: self.analyze_manual_inputs())
            return widget

        def _lbl(parent: tk.Frame, text: str) -> tk.Label:
            return tk.Label(parent, text=text, fg="#94a3b8", bg="#1e293b", font=("Segoe UI", 8))

        r1 = tk.Frame(input_card, bg="#1e293b")
        r1.pack(fill=tk.X, pady=2)
        _lbl(r1, "Hero:").pack(side=tk.LEFT)
        _entry(r1, self.hero_var, 8).pack(side=tk.LEFT, padx=(3, 8))
        _lbl(r1, "Board:").pack(side=tk.LEFT)
        _entry(r1, self.board_var, 16).pack(side=tk.LEFT, padx=(3, 8), fill=tk.X, expand=True)
        _lbl(r1, "Pos:").pack(side=tk.LEFT)
        _entry(r1, self.pos_var, 5).pack(side=tk.LEFT, padx=(3, 0))

        r2 = tk.Frame(input_card, bg="#1e293b")
        r2.pack(fill=tk.X, pady=2)
        _lbl(r2, 'Pot:').pack(side=tk.LEFT)
        _entry(r2, self.pot_var, 7).pack(side=tk.LEFT, padx=(3, 8))
        _lbl(r2, 'To call:').pack(side=tk.LEFT)
        _entry(r2, self.to_call_var, 7).pack(side=tk.LEFT, padx=(3, 8))
        _lbl(r2, "Stack:").pack(side=tk.LEFT)
        _entry(r2, self.stack_var, 7).pack(side=tk.LEFT, padx=(3, 0))

        manual_footer = tk.Frame(input_card, bg="#1e293b")
        manual_footer.pack(fill=tk.X, pady=(2, 0))
        self.opponent_options_btn = tk.Button(
            manual_footer,
            text='Opponent and range ▸',
            command=self._toggle_opponent_options,
            bg="#1e293b",
            fg="#94a3b8",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=0,
            pady=2,
            font=("Segoe UI", 8),
            cursor="hand2",
        )
        self.opponent_options_btn.pack(side=tk.LEFT)
        self._opponent_options_expanded = False
        self.solver_options_btn = tk.Button(
            manual_footer,
            text='ICM solver ▸',
            command=self._toggle_solver_options,
            bg="#1e293b",
            fg="#93c5fd",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=6,
            pady=2,
            font=("Segoe UI", 8),
            cursor="hand2",
        )
        self.solver_options_btn.pack(side=tk.LEFT, padx=(8, 0))
        self._solver_options_expanded = False
        tk.Button(
            manual_footer,
            text='Analyze',
            command=self.analyze_manual_inputs,
            bg="#2563eb",
            fg="#ffffff",
            activebackground="#1d4ed8",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            padx=10,
            pady=2,
            font=("Segoe UI", 8, "bold"),
            cursor="hand2",
        ).pack(side=tk.RIGHT)
        r3 = tk.Frame(input_card, bg="#1e293b")
        self.opponent_options_panel = r3
        _lbl(r3, 'Villain:').pack(side=tk.LEFT)
        _entry(r3, self.villain_var, 9).pack(side=tk.LEFT, padx=(3, 8))
        _lbl(r3, "Range:").pack(side=tk.LEFT)
        _entry(r3, self.range_var, 16).pack(side=tk.LEFT, padx=(3, 8), fill=tk.X, expand=True)

        self.solver_options_panel = tk.Frame(input_card, bg="#172554", padx=7, pady=6)
        solver_heading = tk.Frame(self.solver_options_panel, bg="#172554")
        solver_heading.pack(fill=tk.X)
        tk.Label(
            solver_heading,
            text='Starting stacks in BB · preflop push/fold · ICM 50/30/20 · blinds 0.5/1 BB · no ante',
            fg="#bfdbfe",
            bg="#172554",
            font=("Segoe UI", 8, "bold"),
            anchor=tk.W,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.solver_enabled_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            solver_heading,
            text='Use in this analysis',
            variable=self.solver_enabled_var,
            fg="#f8fafc",
            bg="#172554",
            activebackground="#172554",
            activeforeground="#ffffff",
            selectcolor="#0f172a",
            font=("Segoe UI", 8),
        ).pack(side=tk.RIGHT)
        solver_stack_row = tk.Frame(self.solver_options_panel, bg="#172554")
        solver_stack_row.pack(fill=tk.X, pady=(4, 2))
        self.solver_stack_vars = {seat: tk.StringVar(value="") for seat in ("BTN", "SB", "BB")}
        for seat in ("BTN", "SB", "BB"):
            tk.Label(solver_stack_row, text=f"{seat}:", fg="#94a3b8", bg="#172554", font=("Segoe UI", 8)).pack(side=tk.LEFT)
            solver_entry = _entry(solver_stack_row, self.solver_stack_vars[seat], 6)
            solver_entry.configure(bg="#0f172a")
            solver_entry.pack(side=tk.LEFT, padx=(2, 8))

        solver_node_row = tk.Frame(self.solver_options_panel, bg="#172554")
        solver_node_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(solver_node_row, text='Spot:', fg="#94a3b8", bg="#172554", font=("Segoe UI", 8)).pack(side=tk.LEFT)
        self.solver_node_by_label = {label: key for key, label in NODE_LABELS.items()}
        self.solver_node_var = tk.StringVar(value=NODE_LABELS["btn_push"])
        solver_node_menu = tk.OptionMenu(solver_node_row, self.solver_node_var, *self.solver_node_by_label.keys())
        solver_node_menu.configure(
            bg="#0f172a",
            fg="#e2e8f0",
            activebackground="#334155",
            activeforeground="#ffffff",
            relief=tk.FLAT,
            highlightthickness=1,
            highlightbackground="#334155",
            font=("Segoe UI", 8),
            padx=3,
            pady=1,
        )
        solver_node_menu["menu"].configure(bg="#1e293b", fg="#f8fafc", font=("Segoe UI", 8))
        solver_node_menu.pack(side=tk.LEFT, padx=(4, 8), fill=tk.X, expand=True)
        tk.Label(
            solver_node_row,
            text='With the solver active, Pot/To call/Stack are shown in BB.',
            fg="#64748b",
            bg="#172554",
            font=("Segoe UI", 7),
        ).pack(side=tk.LEFT)
        # Computed State & Math Card
        state_card = tk.Frame(
            outer,
            bg="#1e293b",
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        state_card.pack(fill=tk.X, pady=(0, 10))

        self.cards_var = tk.StringVar(value='Cards: —   |   Board: —')
        tk.Label(
            state_card,
            textvariable=self.cards_var,
            fg="#f8fafc",
            bg="#1e293b",
            font=("Consolas", 11, "bold"),
            anchor=tk.W,
        ).pack(fill=tk.X)

        self.metrics_var = tk.StringVar(value='Pot: —   |   To call: —   |   Equity: —   |   Pot Odds: —')
        tk.Label(
            state_card,
            textvariable=self.metrics_var,
            fg="#cbd5e1",
            bg="#1e293b",
            font=("Consolas", 9),
            anchor=tk.W,
        ).pack(fill=tk.X, pady=(4, 0))

        self.hud_var = tk.StringVar(value='Player stats: —')
        tk.Label(
            state_card,
            textvariable=self.hud_var,
            fg="#94a3b8",
            bg="#1e293b",
            font=("Consolas", 8),
            anchor=tk.W,
            justify=tk.LEFT,
        ).pack(fill=tk.X, pady=(4, 0))

        # Main Action Suggestion Banner
        self.action_banner = tk.Frame(
            outer,
            bg="#1e3a8a",
            padx=14,
            pady=12,
            highlightthickness=2,
            highlightbackground="#3b82f6",
        )
        self.action_banner.pack(fill=tk.X, pady=(0, 10))

        self.banner_title_var = tk.StringVar(value="")
        self.banner_title_label = tk.Label(
            self.action_banner,
            textvariable=self.banner_title_var,
            fg="#93c5fd",
            bg="#1e3a8a",
            font=("Segoe UI", 8, "bold"),
        )

        self.action_var = tk.StringVar(value='WAITING FOR CAPTURE')
        self.action_label = tk.Label(
            self.action_banner,
            textvariable=self.action_var,
            fg="#ffffff",
            bg="#1e3a8a",
            font=("Segoe UI", 20, "bold"),
            anchor=tk.W,
        )
        self.action_label.pack(fill=tk.X, pady=(2, 0))

        # Reasoning Box
        reason_frame = tk.Frame(
            outer,
            bg="#1e293b",
            padx=12,
            pady=10,
            highlightthickness=1,
            highlightbackground="#334155",
        )
        reason_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))

        self.reason_text = tk.Text(
            reason_frame,
            wrap=tk.WORD,
            bg="#1e293b",
            fg="#e2e8f0",
            font=("Segoe UI", 10),
            bd=0,
            highlightthickness=0,
            height=9,
        )
        self.reason_text.pack(fill=tk.BOTH, expand=True)

    def _toggle_capture_options(self) -> None:
        self._capture_options_expanded = not self._capture_options_expanded
        if self._capture_options_expanded:
            self.capture_options_panel.pack(fill=tk.X, pady=(10, 0))
            self.capture_options_btn.configure(text='Options ▾')
        else:
            self.capture_options_panel.pack_forget()
            self.capture_options_btn.configure(text='Options ▸')

    def _toggle_opponent_options(self) -> None:
        self._opponent_options_expanded = not self._opponent_options_expanded
        if self._opponent_options_expanded:
            self.opponent_options_panel.pack(fill=tk.X, pady=(0, 2))
            self.opponent_options_btn.configure(text='Opponent and range ▾')
        else:
            self.opponent_options_panel.pack_forget()
            self.opponent_options_btn.configure(text='Opponent and range ▸')

    def _toggle_solver_options(self) -> None:
        self._solver_options_expanded = not self._solver_options_expanded
        if self._solver_options_expanded:
            self.solver_options_panel.pack(fill=tk.X, pady=(3, 0))
            self.solver_options_btn.configure(text='ICM solver ▾')
        else:
            self.solver_options_panel.pack_forget()
            self.solver_options_btn.configure(text='ICM solver ▸')

    def _current_game_type(self) -> str:
        """Canonical game type selected in the dropdown ('spin_and_go' or 'tournament')."""
        return GAME_TYPE_LABELS.get(self.game_type_var.get(), self._game_type)

    def _current_play_style(self) -> str:
        """Canonical play style selected in the dropdown ('balanced', 'passive', 'aggressive', 'creative')."""
        return PLAY_STYLE_LABELS.get(self.play_style_var.get(), self._play_style)

    def _on_game_type_changed(self) -> None:
        self._game_type = self._current_game_type()
        self._settings["game_type"] = self._game_type
        save_settings(self.settings_path, self._settings)
        # The same table state must be re-advised under the new format instead of being skipped as a duplicate.
        self._last_analyzed_state_key = None
        self.capture_info_var.set("")

    def _on_play_style_changed(self) -> None:
        self._play_style = self._current_play_style()
        self._settings["play_style"] = self._play_style
        save_settings(self.settings_path, self._settings)
        self._last_analyzed_state_key = None
        self.capture_info_var.set("")

    def _set_reasoning(self, text: str) -> None:
        self.reason_text.configure(state=tk.NORMAL)
        self.reason_text.delete("1.0", tk.END)
        self.reason_text.insert(tk.END, text)
        self.reason_text.configure(state=tk.DISABLED)

    def _style_banner(self, bg: str, border: str, sub_fg: str) -> None:
        self.action_banner.configure(bg=bg, highlightbackground=border)
        self.banner_title_label.configure(bg=bg, fg=sub_fg)
        self.action_label.configure(bg=bg, fg="#ffffff")

    def _set_banner_title(self, title: str) -> None:
        self.banner_title_var.set(title)
        if title:
            if not self.banner_title_label.winfo_manager():
                self.banner_title_label.pack(anchor=tk.W, before=self.action_label)
        else:
            self.banner_title_label.pack_forget()

    def _apply_action_banner_style(self, action: str) -> None:
        act = action.lower().strip()
        if act in ("raise", "bet"):
            self._style_banner(bg="#065f46", border="#10b981", sub_fg="#6ee7b7")
        elif act in ("call", "check"):
            self._style_banner(bg="#1e3a8a", border="#3b82f6", sub_fg="#93c5fd")
        else:
            self._style_banner(bg="#7f1d1d", border="#ef4444", sub_fg="#fca5a5")

    def _update_preview_thumbnail(self, ppm_bytes: bytes) -> None:
        try:
            self._thumb_photo = tk.PhotoImage(data=ppm_bytes)
            self.preview_label.configure(image=self._thumb_photo, text="")
        except tk.TclError:
            pass

    def _set_capture_target(self, target: CaptureTarget) -> None:
        self._capture_target = target
        self._last_frame_sig = None
        self._last_analyzed_state_key = None

    @staticmethod
    def _is_888_table_title(title: str) -> bool:
        lowered = title.casefold()
        return "blast" in lowered or ("888" in lowered and "poker" in lowered)

    def _auto_arrange_poker_layout(self) -> None:
        """Find an open 888 BLAST table and tile it beside the assistant."""
        self._layout_after_id = None
        if self._closing:
            return
        try:
            self.root.update_idletasks()
            self_hwnd = get_window_hwnd_from_tk(self.root.winfo_id())
            exclude = {self_hwnd} if self_hwnd else set()
            windows = list_windows(exclude_hwnds=exclude)
        except RuntimeError:
            # Native window enumeration is only available on Windows.
            return
        except Exception:
            windows = []

        candidates = [window for window in windows if self._is_888_table_title(window.title) and not window.minimized]
        if candidates:
            # Prefer the actual BLAST table title when several 888 windows are open.
            target_window = next((window for window in candidates if "blast" in window.title.casefold()), candidates[0])
            target = CaptureTarget(kind="window", hwnd=target_window.hwnd, title=target_window.title)
            arranged = self._apply_poker_table_layout(target)
            self._set_capture_target(target)
            if hasattr(self, "preset_var"):
                self.preset_var.set("888 BLAST")
            if arranged:
                self.capture_info_var.set('Table ready')
                return
            self.capture_info_var.set("")

        # The poker client may be starting after the assistant. Keep looking until it appears.
        if not self._closing:
            self._layout_after_id = self.root.after(1500, self._auto_arrange_poker_layout)

    def _apply_poker_table_layout(self, target: CaptureTarget) -> bool:
        if target.kind != "window" or not target.hwnd:
            return False
        try:
            try:
                area_x, area_y, area_w, area_h = get_work_area_bounds()
            except RuntimeError:
                area_x, area_y = 0, 0
                area_w, area_h = self.root.winfo_screenwidth(), self.root.winfo_screenheight()

            table_window = next((window for window in list_windows() if window.hwnd == target.hwnd), None)
            if table_window is None:
                return False
            if self._tiled_table_state is None or self._tiled_table_state[0] != target.hwnd:
                self._restore_tiled_table()
                original_rect = get_window_rect(target.hwnd) or table_window.rect
                self._tiled_table_state = (
                    target.hwnd,
                    original_rect,
                    is_window_maximized(target.hwnd),
                )

            assistant_w = min(560, max(360, round(area_w * 0.28)))
            assistant_w = min(assistant_w, max(360, area_w - 500))
            table_w = area_w - assistant_w
            if table_w < 400:
                return False
            set_window_bounds(target.hwnd, (area_x, area_y, table_w, area_h))
            self.root.geometry(
                f"{assistant_w}x{max(480, area_h - 36)}+{area_x + table_w}+{area_y}"
            )
            self.root.update_idletasks()
            self.root.lift()
            return True
        except (RuntimeError, OSError, tk.TclError):
            return False

    def _restore_tiled_table(self) -> None:
        state = self._tiled_table_state
        self._tiled_table_state = None
        if state is None:
            return
        hwnd, original_rect, was_maximized = state
        try:
            if was_maximized:
                maximize_window(hwnd)
            else:
                set_window_bounds(hwnd, original_rect)
        except (RuntimeError, OSError):
            pass

    def _toggle_live_capture(self) -> None:
        if self._live_enabled:
            self._stop_live_capture()
        else:
            self._start_live_capture()

    def _start_live_capture(self) -> None:
        self._live_enabled = True
        self.live_btn.configure(text="Auto 2s: ON", bg="#16a34a")
        self.capture_once(force=False, is_live_tick=True)

    def _stop_live_capture(self) -> None:
        self._live_enabled = False
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except tk.TclError:
                pass
            self._live_after_id = None
        if not self._closing:
            self.live_btn.configure(text="Auto 2s: OFF", bg="#334155")

    def _schedule_next_live_tick(self) -> None:
        if not self._live_enabled or self._closing:
            return
        if self._live_after_id is not None:
            try:
                self.root.after_cancel(self._live_after_id)
            except tk.TclError:
                pass
        self._live_after_id = self.root.after(self._LIVE_INTERVAL_MS, self._on_live_tick)

    def _on_live_tick(self) -> None:
        self._live_after_id = None
        if not self._live_enabled or self._closing:
            return
        self.capture_once(force=False, is_live_tick=True)

    def _on_frame_grabbed(self, seq: int, thumb_ppm: bytes, cap_meta: dict[str, Any]) -> None:
        if self._closing or seq != self._capture_seq:
            return
        self._update_preview_thumbnail(thumb_ppm)
        self.capture_info_var.set(
            f"Frame {cap_meta['width']}×{cap_meta['height']} ({cap_meta['capture_ms']:.0f}ms) · analyzing..."
        )

    def capture_diagnostic_frame(self) -> None:
        """Save the current desktop frame to this build's archive without stopping Auto or calling the model."""
        if self._closing or self._diagnostic_capture_in_flight:
            return
        self._diagnostic_capture_in_flight = True
        target = self._capture_target
        game_type = self._current_game_type()
        preset = self._PRESET_LABELS.get(self.preset_var.get(), DEFAULT_TABLE_PRESET)
        companion_rect: tuple[int, int, int, int] | None = None
        try:
            self.root.update_idletasks()
            hwnd = get_window_hwnd_from_tk(self.root.winfo_id())
            companion_rect = get_window_rect(hwnd)
        except Exception:
            pass

        def _worker() -> None:
            try:
                _table_image, target_meta = capture_target(target)
                desktop_image = capture_screen_region(*get_primary_screen_bounds())
                metadata = {
                    "app_version": f"v{__version__}",
                    "game_type": game_type,
                    "captured_at": target_meta.get("captured_at"),
                    "target": target_meta,
                    "companion_rect": list(companion_rect) if companion_rect is not None else None,
                    "mode": "capture_only",
                    "preset": preset,
                    "evaluate_status": "not_analyzed",
                    "vision_source": None,
                    "vision_latency_ms": None,
                    "hero_cards": [],
                    "board": [],
                    "pot": None,
                    "pot_source": None,
                    "pot_label_text": None,
                    "to_call": None,
                    "to_call_source": None,
                    "action_buttons": [],
                    "hero_to_act": None,
                    "players": [],
                    "ambiguities": [],
                    "user_triggered": True,
                }
                save_diagnostic_capture(
                    fullscreen_image=desktop_image,
                    metadata=metadata,
                    captures_dir=self.captures_dir,
                )
            except Exception:
                pass
            finally:
                self._diagnostic_capture_in_flight = False

        threading.Thread(target=_worker, daemon=True).start()

    def capture_once(self, force: bool = True, is_live_tick: bool = False) -> None:
        """Capture one frame from the selected screen target and run hybrid table analysis."""
        if self._closing:
            return
        if self._capture_in_flight:
            if force:
                self._pending_force_capture = True
            return

        self._capture_in_flight = True
        self._capture_seq += 1
        seq = self._capture_seq
        target = self._capture_target
        # 888 is not covered by the calibrated Mercury reader; use the fixed local VLM directly.
        mode = "vlm"
        preset = self._PRESET_LABELS.get(self.preset_var.get(), DEFAULT_TABLE_PRESET)
        game_type = self._current_game_type()
        save_diagnostic = bool(self.save_captures_var.get())
        v_provider, v_model = self._VISION_PROVIDER, self._VISION_MODEL
        prev_sig = self._last_frame_sig
        self_rect: tuple[int, int, int, int] | None = None
        try:
            self.root.update_idletasks()
            self_hwnd = get_window_hwnd_from_tk(self.root.winfo_id())
            self_rect = get_window_rect(self_hwnd)
        except Exception:
            self_rect = None

        if force:
            self.status_dot.configure(fg="#facc15")
            self.status_var.set('Capturing frame...')
            self.capture_info_var.set('Capturing...')

        def _worker() -> None:
            try:
                image, cap_meta = capture_target(target)
                if self_rect is not None and target.kind in ("fullscreen", "region"):
                    rect_list = cap_meta.get("rect")
                    if isinstance(rect_list, list) and len(rect_list) == 4:
                        mask_screen_rect_on_image(
                            image,
                            (int(rect_list[0]), int(rect_list[1]), int(rect_list[2]), int(rect_list[3])),
                            self_rect,
                        )
                thumb_ppm = image_to_ppm_bytes(resize_rgb_image(image, 150, 84))
                sig = compute_frame_signature(image)
                diff = frame_difference(sig, prev_sig)
                if is_live_tick and not force and prev_sig is not None and diff < 0.003:
                    self._schedule_ui(
                        lambda: self._on_capture_finished(
                            seq=seq,
                            status="unchanged",
                            thumb_ppm=thumb_ppm,
                            cap_meta=cap_meta,
                            frame_sig=sig,
                            report=None,
                            snapshot_path=None,
                            error_msg=None,
                            force=force,
                        )
                    )
                    return

                self._schedule_ui(lambda: self._on_frame_grabbed(seq, thumb_ppm, cap_meta))

                report = analyze_live_image(
                    image=image,
                    mode=mode,
                    vlm_provider=v_provider,
                    vlm_model=v_model,
                    timeout_seconds=self._LIVE_VISION_TIMEOUT_SECONDS,
                    preset=preset,
                    game_type=game_type,
                )

                snapshot_path = self.bridge.bridge_dir / "latest_capture.png"
                try:
                    snapshot_path.write_bytes(encode_png_bytes(image, compress_level=1))
                except OSError:
                    snapshot_path = None

                if not report.get("hero_to_act"):
                    self._last_auto_saved_state_key = None
                elif save_diagnostic:
                    hero_player = next(
                        (player for player in (report.get("players") or []) if player.get("hero")),
                        None,
                    )

                    def _rounded_or_none(value: Any) -> float | None:
                        try:
                            return round(float(value), 2) if value is not None else None
                        except (TypeError, ValueError):
                            return None

                    state_key = (
                        tuple(str(card) for card in (report.get("hero_cards") or [])),
                        tuple(str(card) for card in (report.get("board") or [])),
                        _rounded_or_none(report.get("pot")),
                        _rounded_or_none(report.get("to_call")),
                        _rounded_or_none(hero_player.get("stack") if hero_player else None),
                        tuple(str(button) for button in (report.get("action_buttons") or [])),
                    )
                    if state_key != self._last_auto_saved_state_key:
                        eval_kind, _ = evaluate_vision_report(report)
                        diag_meta = {
                            "app_version": f"v{__version__}",
                            "game_type": game_type,
                            "captured_at": cap_meta.get("captured_at"),
                            "target": {"kind": "archived_image", "title": "Analyzed table frame"},
                            "source_target": cap_meta,
                            "companion_rect": None,
                            "automatic_capture": True,
                            "capture_reason": "Hero's turn confirmed by the action buttons",
                            "mode": mode,
                            "preset": preset,
                            "evaluate_status": eval_kind,
                            "vision_source": report.get("vision_source"),
                            "vision_latency_ms": report.get("vision_latency_ms"),
                            "hero_cards": report.get("hero_cards"),
                            "hero_stack_text": report.get("hero_stack_text"),
                            "board": report.get("board"),
                            "pot": report.get("pot"),
                            "pot_source": report.get("pot_source"),
                            "pot_label_text": report.get("pot_label_text"),
                            "to_call": report.get("to_call"),
                            "to_call_source": report.get("to_call_source"),
                            "action_buttons": report.get("action_buttons"),
                            "hero_to_act": True,
                            "players": report.get("players"),
                            "ambiguities": report.get("ambiguities"),
                            "mercury_vision": report.get("mercury_vision"),
                            "raw_vlm_response": report.get("raw_vlm_response"),
                        }
                        try:
                            save_diagnostic_capture(
                                fullscreen_image=image,
                                metadata=diag_meta,
                                captures_dir=self.captures_dir,
                            )
                            self._last_auto_saved_state_key = state_key
                        except OSError:
                            pass

                self._schedule_ui(
                    lambda: self._on_capture_finished(
                        seq=seq,
                        status="parsed",
                        thumb_ppm=None,
                        cap_meta=cap_meta,
                        frame_sig=sig,
                        report=report,
                        snapshot_path=snapshot_path,
                        error_msg=None,
                        force=force,
                    )
                )

            except Exception as exc:
                self._schedule_ui(
                    lambda err=str(exc): self._on_capture_finished(
                        seq=seq,
                        status="error",
                        thumb_ppm=None,
                        cap_meta=None,
                        frame_sig=None,
                        report=None,
                        snapshot_path=None,
                        error_msg=err,
                        force=force,
                    )
                )

        threading.Thread(target=_worker, daemon=True).start()

    def _on_capture_finished(
        self,
        seq: int,
        status: str,
        thumb_ppm: bytes | None,
        cap_meta: dict[str, Any] | None,
        frame_sig: bytes | None,
        report: dict[str, Any] | None,
        snapshot_path: Path | None,
        error_msg: str | None,
        force: bool,
    ) -> None:
        self._capture_in_flight = False
        if self._closing:
            return
        if thumb_ppm is not None:
            self._update_preview_thumbnail(thumb_ppm)

        if seq == self._capture_seq:
            if status == "error":
                self._last_analyzed_state_key = None
                self.capture_info_var.set('Capture error')
                self._render_read_failure(self._capture_target.describe(), [error_msg or "Unknown error"])
            elif status == "unchanged" and cap_meta is not None:
                self.capture_info_var.set(
                    f"{cap_meta['width']}×{cap_meta['height']} ({cap_meta['capture_ms']:.0f}ms) · unchanged"
                )
            elif status == "parsed" and cap_meta is not None and report is not None:
                self._last_frame_sig = frame_sig
                v_src = report.get("vision_source", "vision")
                v_ms = float(report.get("vision_latency_ms", 0))
                t_label = f"{v_ms / 1000.0:.2f}s" if v_ms >= 1000 else f"{v_ms:.0f}ms"
                self.capture_info_var.set(
                    f"{cap_meta['width']}×{cap_meta['height']} · {v_src} ({t_label})"
                )
                capture_info = {
                    **cap_meta,
                    "vision_source": v_src,
                    "vision_latency_ms": v_ms,
                    "snapshot_path": str(snapshot_path) if snapshot_path else None,
                    "game_type": report.get("game_type"),
                    "app_version": f"v{__version__}",
                }
                source_label = f"Live · {v_src}"
                self._apply_vision_report(
                    report=report,
                    source_label=source_label,
                    hand_prefix="LIVE",
                    capture_info=capture_info,
                    deduplicate_ready=not force,
                )

        if self._pending_force_capture:
            self._pending_force_capture = False
            self.capture_once(force=True)
        elif self._live_enabled:
            self._schedule_next_live_tick()

    def _apply_vision_report(
        self,
        report: dict[str, Any],
        source_label: str,
        hand_prefix: str,
        capture_info: dict[str, Any] | None = None,
        deduplicate_ready: bool = False,
    ) -> None:
        kind, payload = evaluate_vision_report(report)
        if kind == "failed":
            self._last_analyzed_state_key = None
            self._render_read_failure(source_label, payload["reasons"])
            return
        if kind == "waiting":
            self._last_analyzed_state_key = None
            self._analysis_seq += 1
            self._render_waiting_turn()
            return

        state_key = (
            tuple(payload["hero_cards"]),
            tuple(payload["board"]),
            round(float(payload["pot"]), 2),
            round(float(payload["to_call"]), 2),
            tuple((opp["nickname"], opp.get("range", "*")) for opp in payload["opponents"]),
        )
        if deduplicate_ready and state_key == self._last_analyzed_state_key:
            self.capture_info_var.set(f"{self.capture_info_var.get()} · decision held")
            return
        self._last_analyzed_state_key = state_key

        self.hero_var.set(" ".join(payload["hero_cards"]))
        self.board_var.set(" ".join(payload["board"]))
        self.pot_var.set(f"{payload['pot']:g}")
        self.to_call_var.set(f"{payload['to_call']:g}")
        self.stack_var.set(f"{payload['hero_stack']:g}")
        self.pos_var.set(payload["hero_position"])
        if payload["opponents"]:
            self.villain_var.set(payload["opponents"][0]["nickname"])
            self.range_var.set("*")

        self._hand_counter += 1
        self._run_strategic_analysis(
            hero_cards=payload["hero_cards"],
            board=payload["board"],
            pot=payload["pot"],
            to_call=payload["to_call"],
            hero_stack=payload["hero_stack"],
            hero_position=payload["hero_position"],
            opponents=payload["opponents"],
            hand_id=f"{hand_prefix}_{self._hand_counter:04d}",
            source_label=source_label,
            capture_info=capture_info,
            report_big_blind=payload.get("big_blind"),
        )

    def _render_read_failure(self, source_label: str, details: list[str]) -> None:
        self._active_request_id = None
        self.status_dot.configure(fg="#ef4444")
        self.status_var.set(f"Read failed ({source_label[:28]})")
        self.cards_var.set('Cards: —   |   Board: —')
        self.metrics_var.set('Pot: —   |   To call: —   |   Equity: —   |   Pot Odds: —')
        self.hud_var.set('Player stats: —')
        self._set_banner_title('RECOGNITION / VALIDATION ERROR')
        self.action_var.set('READ FAILED')
        self._style_banner(bg="#7f1d1d", border="#ef4444", sub_fg="#fca5a5")
        self._set_reasoning('Could not validate the hand state:\n' + "\n".join(f"• {d}" for d in details))

    def _render_waiting_turn(self) -> None:
        self._active_request_id = None
        self._active_context = None
        self._active_solver_recommendation = None
        self.status_dot.configure(fg="#38bdf8")
        self.status_var.set("Waiting for your turn")
        self._set_banner_title("")
        self.action_var.set("Waiting for your turn")
        self._style_banner(bg="#1e293b", border="#475569", sub_fg="#94a3b8")

    def analyze_manual_inputs(self) -> None:
        required = {
            "Hero cards": self.hero_var.get().strip(),
            "Pot": self.pot_var.get().strip(),
            "To call": self.to_call_var.get().strip(),
            "Stack": self.stack_var.get().strip(),
        }
        missing = [label for label, value in required.items() if not value]
        if missing:
            self._render_read_failure('Manual input', [f"Fill in the required fields: {', '.join(missing)}."])
            return
        try:
            hero_cards = self.hero_var.get().split()
            if len(hero_cards) != 2:
                self._render_read_failure('Manual input', ['Enter exactly two Hero cards.'])
                return
            board = self.board_var.get().split()
            pot = float(self.pot_var.get())
            to_call = float(self.to_call_var.get())
            hero_stack = float(self.stack_var.get())
            hero_pos = self.pos_var.get().strip().upper() or "BTN"
            villain = self.villain_var.get().strip() or "VILLAIN"
            v_range = self.range_var.get().strip() or "*"
            opponents = [{"nickname": villain, "range": v_range}]
            solver_inputs = {seat: var.get().strip() for seat, var in self.solver_stack_vars.items()}
            solver_request: dict[str, Any] | None = None
            if self.solver_enabled_var.get():
                if not all(solver_inputs.values()):
                    self._render_read_failure('ICM solver', ['Enter the BTN, SB, and BB stacks in big blinds.'])
                    return
                if board:
                    self._render_read_failure('ICM solver', ['The 3-max solver only supports preflop push/fold decisions.'])
                    return
                if self._current_game_type() != "spin_and_go":
                    self._render_read_failure('ICM solver', ['The solver is configured for 3-max Spin & Go.'])
                    return
                stacks_bb = tuple(float(solver_inputs[seat]) for seat in ("BTN", "SB", "BB"))
                node = self.solver_node_by_label.get(self.solver_node_var.get())
                if node not in SOLVER_NODES:
                    self._render_read_failure('ICM solver', ['Choose a valid 3-max spot.'])
                    return
                definition = SOLVER_NODES[node]
                hero_pos = definition["position"]
                self.pos_var.set(hero_pos)
                if definition["action"] == "push" and to_call > 1e-6:
                    self._render_read_failure('ICM solver', ["For this shove spot, set 'To call' to 0 BB."])
                    return
                if definition["action"] == "call" and to_call <= 1e-6:
                    self._render_read_failure('ICM solver', ["For this call spot, 'To call' must be greater than 0 BB."])
                    return
                opponents = [
                    {"nickname": seat, "range": "*"}
                    for seat in ("BTN", "SB", "BB")
                    if seat != hero_pos
                ]
                solver_request = {
                    "node": node,
                    "stacks_bb": stacks_bb,
                    "payouts": (0.5, 0.3, 0.2),
                    "small_blind_bb": 0.5,
                    "big_blind_bb": 1.0,
                    "ante_bb": 0.0,
                }
        except ValueError as exc:
            self._render_read_failure('Manual input', [f"Invalid numeric value: {exc}"])
            return

        self._hand_counter += 1
        self._run_strategic_analysis(
            hero_cards=hero_cards,
            board=board,
            pot=pot,
            to_call=to_call,
            hero_stack=hero_stack,
            hero_position=hero_pos,
            opponents=opponents,
            hand_id=f"STUDY_{self._hand_counter:04d}",
            source_label='Manual Analysis',
            solver_request=solver_request,
        )

    def _run_strategic_analysis(
        self,
        hero_cards: list[str],
        board: list[str],
        pot: float,
        to_call: float,
        hero_stack: float,
        hero_position: str,
        opponents: list[dict[str, Any]],
        hand_id: str,
        source_label: str,
        capture_info: dict[str, Any] | None = None,
        solver_request: dict[str, Any] | None = None,
        report_big_blind: float | None = None,
    ) -> None:
        game_type = self._current_game_type()
        effective_bb = (
            1.0
            if solver_request
            else _infer_big_blind(
                report_big_blind=report_big_blind,
                capture_info=capture_info,
                pot=pot,
                hero_stack=hero_stack,
                game_type=game_type,
            )
        )
        try:
            ctx = build_decision_context_from_inputs(
                hero_cards=hero_cards,
                board=board,
                pot=pot,
                to_call=to_call,
                hero_stack=hero_stack,
                hero_position=hero_position,
                opponents=opponents,
                big_blind=effective_bb,
                equity_iterations=1_500,
                profile_store=self.store,
                hand_id=hand_id,
                game_type=game_type,
                play_style=self._current_play_style(),
            )
        except Exception as exc:
            self._render_read_failure(source_label, [f"Invalid hand state: {exc}"])
            return

        if solver_request is None:
            live_req = infer_live_three_max_request(
                game_type=game_type,
                board=board,
                hero_position=hero_position,
                hero_stack=hero_stack,
                to_call=to_call,
                big_blind=effective_bb,
                opponents=opponents,
            )
            if live_req is not None:
                predicted = predict_hand_advice_from_model(hero_cards=hero_cards, **live_req)
                if predicted is not None:
                    if SOLVER_NODES[live_req["node"]]["action"] == "push" and not any(
                        item.action == "fold" for item in ctx.legal_actions
                    ):
                        ctx = replace(ctx, legal_actions=(*ctx.legal_actions, LegalAction("fold", 0.0, 0.0)))
                    ctx = replace(ctx, three_max_solver=predicted)

        if solver_request and SOLVER_NODES[solver_request["node"]]["action"] == "push":
            if not any(item.action == "fold" for item in ctx.legal_actions):
                ctx = replace(ctx, legal_actions=(*ctx.legal_actions, LegalAction("fold", 0.0, 0.0)))

        self._analysis_seq += 1
        seq = self._analysis_seq
        self._active_request_id = None
        self._active_solver_recommendation = None
        if solver_request:
            self.status_dot.configure(fg="#f59e0b")
            self.status_var.set('Calculating the 3-max ICM solver...')
            self._set_banner_title("3-MAX ICM SOLVER")
            self.action_var.set('CALCULATING...')
            self._style_banner(bg="#713f12", border="#f59e0b", sub_fg="#fde68a")
            self._set_reasoning('Calculating push/fold strategy from the specified stacks and spot.')

            def _solver_worker(current_seq: int, base_context: Any, request: dict[str, Any]) -> None:
                try:
                    advice = solve_hand_advice(hero_cards=hero_cards, **request)
                    enriched_context = replace(base_context, three_max_solver=advice)
                except Exception as exc:
                    self._schedule_ui(
                        lambda s=current_seq, err=str(exc): self._render_solver_failure(s, err)
                    )
                    return
                self._schedule_ui(
                    lambda s=current_seq, c=enriched_context: self._dispatch_strategic_context(
                        s, c, source_label, capture_info
                    )
                )

            threading.Thread(
                target=_solver_worker,
                args=(seq, ctx, solver_request),
                daemon=True,
            ).start()
            return

        self._dispatch_strategic_context(seq, ctx, source_label, capture_info)

    def _render_solver_failure(self, seq: int, error: str) -> None:
        if self._closing or seq != self._analysis_seq:
            return
        self._render_read_failure('ICM solver', [error])

    def _dispatch_strategic_context(
        self,
        seq: int,
        ctx: Any,
        source_label: str,
        capture_info: dict[str, Any] | None,
    ) -> None:
        if self._closing or seq != self._analysis_seq:
            return
        req = self.bridge.publish_request(
            ctx,
            agent_target="antigravity",
            capture_info=capture_info,
        )
        self._active_request_id = req.request_id
        self._active_context = ctx

        base = req.baseline_recommendation
        self._active_solver_recommendation = base if ctx.three_max_solver else None
        engine_tag = "3-Max ML model" if base.get("provider") == "three_max_unequal_model" else "Local engine"
        self._render_decision(
            ctx=ctx,
            action=str(base["recommended_action"]),
            amount=float(base["recommended_amount"]),
            reasoning=str(base["reasoning"]),
            agent_label=f"{source_label} · {engine_tag}",
        )

        if not self._enable_background_llm:
            return
        if self._live_enabled:
            return

        def _llm_worker(current_seq: int, context: Any, req_id: str, src: str) -> None:
            if self._closing:
                return
            discovered = discover_local_models(timeout=0.4)
            ollama_pick = _pick_installed_strategy_model(discovered.get("ollama_models") or ())
            openai_pick = _pick_installed_strategy_model(discovered.get("openai_local_models") or ())
            if ollama_pick is not None:
                provider, model = "ollama", ollama_pick
            elif openai_pick is not None:
                provider, model = "openai_compatible", openai_pick
            else:
                # No dedicated strategy LLM available locally; avoid querying the 2B vision model for poker strategy.
                return
            rec = self.bridge.query_local_gpt_oss(
                context,
                provider=provider,
                model=model,
                timeout_seconds=20.0,
            )
            label = f"{src} · {provider}:{model}" + (" [fallback]" if rec.fallback_used else "")
            self._schedule_ui(
                lambda s=current_seq, c=context, r_id=req_id, a=rec.recommended_action, amt=rec.recommended_amount, rsn=rec.reasoning, lbl=label: self._on_llm_finished(
                    s, c, r_id, a, amt, rsn, lbl
                )
            )

        threading.Thread(
            target=_llm_worker,
            args=(seq, ctx, req.request_id, source_label),
            daemon=True,
        ).start()

    def _on_llm_finished(
        self,
        seq: int,
        ctx: Any,
        req_id: str,
        action: str,
        amount: float,
        reasoning: str,
        agent_label: str,
    ) -> None:
        if self._closing or seq != self._analysis_seq or req_id != self._active_request_id:
            return
        self._render_decision(
            ctx=ctx,
            action=action,
            amount=amount,
            reasoning=reasoning,
            agent_label=agent_label,
        )

    def _render_decision(
        self,
        ctx: Any,
        action: str,
        amount: float,
        reasoning: str,
        agent_label: str,
    ) -> None:
        action, amount, exact_legal = validate_and_clamp_action(action, amount, ctx.legal_actions)
        if not exact_legal:
            reasoning = (
                reasoning.rstrip()
                + '\n\nThe recommendation was adjusted to the legal actions and remaining stack for this hand.'
            )
        self.status_dot.configure(fg="#22c55e")
        self.status_var.set(agent_label)
        board_str = " ".join(ctx.board) if ctx.board else 'Preflop'
        self.cards_var.set(f"Hero: {' '.join(ctx.hero_cards)}   |   Board: {board_str}")
        po = ctx.math_strategy.get("pot_odds")
        po_str = f"{po:.0%}" if po is not None else "0%"
        eq_share = float(ctx.equity["share"])
        spr_str = f"{ctx.spr:.1f}" if ctx.spr is not None else "—"
        self.metrics_var.set(
            f"Pot: {ctx.pot:.1f}  |  To call: {ctx.to_call:.1f}  |  Equity: {eq_share:.0%}  |  Pot Odds: {po_str}  |  SPR: {spr_str}"
        )

        hud_parts: list[str] = []
        for opp_name, prof in ctx.opponent_profiles.items():
            if prof and prof.get("hands_observed", 0) > 0:
                vpip = prof["vpip"]
                pfr = prof["pfr"]
                v_str = f"{vpip['empirical_rate']:.0%}" if vpip.get("empirical_rate") is not None else "n/d"
                p_str = f"{pfr['empirical_rate']:.0%}" if pfr.get("empirical_rate") is not None else "n/d"
                hud_parts.append(
                    f"{opp_name} (n={prof['hands_observed']}): VPIP {v_str}, PFR {p_str} [{vpip['sample_adequacy']}]"
                )
            else:
                hud_parts.append(f"{opp_name}: no sample")
        self.hud_var.set("HUD: " + (" | ".join(hud_parts) if hud_parts else "—"))

        self._set_banner_title("")
        amt_suffix = f"  ({amount:.1f})" if amount > 0 else ""
        action_label = action.upper()
        if action in ("bet", "raise") and amount >= ctx.hero_stack - 0.01:
            action_label = "ALL-IN · SHOVE"
        elif action == "call" and amount >= ctx.hero_stack - 0.01:
            action_label = "CALL · ALL-IN"
        if ctx.three_max_solver:
            advice = ctx.three_max_solver
            action_label = (
                "ALL-IN · SHOVE"
                if advice.get("solver_action") == "push" and action != "fold"
                else action_label
            )
            frequency = float(advice.get("fold_frequency", 0.0)) if action == "fold" else float(advice.get("action_frequency", 0.0))
            action_label += f" · {frequency:.0%}"
        self.action_var.set(f"{action_label}{amt_suffix}")
        self._apply_action_banner_style(action)
        self._set_reasoning(reasoning)

    def _poll_agent_response(self) -> None:
        if self._closing:
            return
        if self._enable_background_llm and self.global_hotkey_var.get():
            f8_down = is_key_pressed(0x77)
            if f8_down and not self._f8_was_down:
                self._f8_was_down = True
                self.capture_once(force=True)
            elif not f8_down:
                self._f8_was_down = False

        self._poll_tick_count += 1
        if self._poll_tick_count % 3 == 0 and self._active_request_id is not None:
            resp = self.bridge.read_latest_response(expected_request_id=self._active_request_id)
            if resp:
                resp_key = f"{resp.get('request_id')}_{resp.get('responded_at')}"
                if resp_key != self._last_rendered_response_id:
                    self._last_rendered_response_id = resp_key
                    act = str(resp.get("recommended_action", "check"))
                    amt = float(resp.get("recommended_amount", 0.0))
                    agent_name = str(resp.get("agent_name", "Antigravity"))
                    reasoning = str(resp.get("reasoning", ""))
                    solver_rec = getattr(self, "_active_solver_recommendation", None)
                    if solver_rec and getattr(self, "_active_context", None) is not None:
                        act = str(solver_rec["recommended_action"])
                        amt = float(solver_rec["recommended_amount"])
                        llm_note = reasoning.strip()
                        reasoning = str(solver_rec["reasoning"])
                        if llm_note:
                            reasoning += f"\n\nAgent explanation: {llm_note}"
                    active_context = getattr(self, "_active_context", None)
                    if active_context is not None:
                        self._render_decision(
                            ctx=active_context,
                            action=act,
                            amount=amt,
                            reasoning=reasoning,
                            agent_label=f"Live response ({agent_name})",
                        )
        self._poll_after_id = self.root.after(150, self._poll_agent_response)


def main(argv: list[str] | None = None) -> int:
    _set_windows_app_user_model_id()
    enable_dpi_awareness()
    default_db = Path(__file__).resolve().parents[1] / "data" / "profiles.sqlite"
    parser = argparse.ArgumentParser(
        description="Lugon Poker Lab — Real-time poker analysis with screen capture and manual input."
    )
    parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--db", type=Path, default=default_db)
    parser.add_argument("--captures-dir", type=Path, default=DEFAULT_CAPTURES_DIR)
    args = parser.parse_args(argv)

    root = tk.Tk()
    app = AssistantWindow(
        root=root,
        bridge_dir=args.bridge_dir,
        db_path=args.db,
        captures_dir=args.captures_dir,
    )
    try:
        root.mainloop()
    finally:
        app.store.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
