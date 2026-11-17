"""Game-format selector (Spin & Go vs tournament), release versioning and capture replay."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from lugon_poker import __version__
from lugon_poker.assistant import (
    DecisionContext,
    GameAssistant,
    LegalAction,
    build_assistant_prompt,
    normalize_game_type,
)
from lugon_poker.agent_bridge import build_decision_context_from_inputs
from lugon_poker.companion_gui import (
    GAME_TYPE_LABELS,
    game_type_label,
    load_settings,
    save_settings,
)
from lugon_poker.live_vision import analyze_live_image, build_vision_prompt
from lugon_poker.screen_capture import format_capture_version_tag, save_diagnostic_capture
from lugon_poker.synthetic import render_scene
from lugon_poker.vision import load_scene

SCENE_PATH = Path(__file__).parents[1] / "examples" / "vision" / "flop_table.json"


def _marginal_call_context(game_type: str) -> DecisionContext:
    """Call is +EV in chips (35% equity vs 33.3% pot odds) but commits half the stack."""
    pot_odds = 10.0 / 30.0
    equity = 0.35
    return DecisionContext(
        hand_id="GT_1",
        street="turn",
        hero_nickname="HERO",
        hero_position="BTN",
        hero_cards=("Qs", "Jh"),
        board=("Ks", "7d", "2c", "3h"),
        hero_hand_description="high card",
        pot=20.0,
        to_call=10.0,
        hero_stack=20.0,
        effective_stack=20.0,
        spr=1.0,
        legal_actions=(
            LegalAction("fold", 0.0, 0.0),
            LegalAction("call", 10.0, 10.0),
            LegalAction("raise", 20.0, 20.0),
        ),
        equity={"share": equity, "win_rate": 0.33, "tie_rate": 0.04, "loss_rate": 0.63, "samples": 1000, "method": "monte_carlo"},
        math_strategy={
            "decision": "call",
            "equity": equity,
            "pot_odds": pot_odds,
            "call_ev": equity * 30.0 - 10.0,
            "fold_ev": 0.0,
            "explanation": "x",
        },
        recent_actions=(),
        opponent_profiles={},
        game_type=game_type,
    )


class GameTypeStrategyTests(unittest.TestCase):
    def test_normalize_game_type_accepts_aliases_and_defaults_to_spin_and_go(self):
        for alias in ("tournament", "Torneio", "TORNEIOS", "mtt", " MTT "):
            self.assertEqual(normalize_game_type(alias), "tournament")
        for value in ("spin_and_go", "", None, "cash", "qualquer coisa"):
            self.assertEqual(normalize_game_type(value), "spin_and_go")

    def test_dropdown_labels_round_trip_to_canonical_game_types(self):
        self.assertEqual(set(GAME_TYPE_LABELS.values()), {"spin_and_go", "tournament"})
        for label, key in GAME_TYPE_LABELS.items():
            self.assertEqual(game_type_label(key), label)
        self.assertEqual(game_type_label("mtt"), "Torneio (MTT)")

    def test_prompt_and_context_carry_the_selected_format(self):
        spin = build_assistant_prompt(_marginal_call_context("spin_and_go"))
        mtt = build_assistant_prompt(_marginal_call_context("tournament"))
        self.assertIn("Spin & Go", spin["system"])
        self.assertIn("ICM", mtt["system"])
        self.assertNotEqual(spin["system"], mtt["system"])
        self.assertEqual(json.loads(spin["user"])["game_type"], "spin_and_go")
        self.assertEqual(json.loads(mtt["user"])["game_type"], "tournament")

    def test_tournament_demands_icm_margin_on_a_marginal_call_that_spin_and_go_takes(self):
        spin = GameAssistant(provider="heuristic").recommend(_marginal_call_context("spin_and_go"))
        mtt = GameAssistant(provider="heuristic").recommend(_marginal_call_context("tournament"))
        self.assertEqual(spin.recommended_action, "call")
        self.assertEqual(mtt.recommended_action, "fold")
        self.assertIn("ICM", mtt.reasoning)
        self.assertTrue(any("Torneio" in item for item in mtt.assumptions))
        self.assertTrue(any("Spin & Go" in item for item in spin.assumptions))

    def test_clear_equity_edge_is_still_called_in_both_formats(self):
        # 60% equity vs 33% pot odds must survive the tournament ICM margin.
        for game in ("spin_and_go", "tournament"):
            ctx = replace(_marginal_call_context(game), equity={**_marginal_call_context(game).equity, "share": 0.60})
            ctx = replace(ctx, math_strategy={**ctx.math_strategy, "call_ev": 0.60 * 30.0 - 10.0})
            rec = GameAssistant(provider="heuristic").recommend(ctx)
            self.assertIn(rec.recommended_action, ("call", "raise"), game)

    def test_build_decision_context_defaults_to_spin_and_go_and_honours_tournament(self):
        base = dict(hero_cards=["As", "Kd"], board=[], pot=3.0, to_call=1.0, hero_stack=20.0, equity_iterations=200)
        self.assertEqual(build_decision_context_from_inputs(**base).game_type, "spin_and_go")
        self.assertEqual(build_decision_context_from_inputs(**base, game_type="tournament").game_type, "tournament")


class VisionGameTypeTests(unittest.TestCase):
    def test_vision_prompt_ignores_prize_banner_and_adapts_opponent_seats(self):
        spin = build_vision_prompt(game_type="spin_and_go")
        mtt = build_vision_prompt(game_type="tournament")
        for prompt in (spin, mtt):
            self.assertIn("IGNORE the prize/multiplier banner", prompt)
            self.assertIn("NOT cards", prompt)
        self.assertIn("top-left / top-right", spin)
        self.assertNotIn("top-left / top-right", mtt)
        self.assertIn("EVERY other seated player", mtt)

    def test_report_is_stamped_with_normalized_game_type_and_app_version(self):
        scene = load_scene(SCENE_PATH)
        report = analyze_live_image(render_scene(scene, 960, 540), mode="calibrated", game_type="Torneio")
        self.assertEqual(report["game_type"], "tournament")
        self.assertEqual(report["app_version"], __version__)
        default = analyze_live_image(render_scene(scene, 960, 540), mode="calibrated")
        self.assertEqual(default["game_type"], "spin_and_go")


class VersionedCaptureTests(unittest.TestCase):
    def test_version_tag_is_normalized_and_sanitized(self):
        self.assertEqual(format_capture_version_tag("0.3.0"), "v0.3.0")
        self.assertEqual(format_capture_version_tag("v1.2.3"), "v1.2.3")
        self.assertEqual(format_capture_version_tag("1.0/..\\evil"), "v1.0_.._evil")
        self.assertEqual(format_capture_version_tag(""), "")
        self.assertEqual(format_capture_version_tag(), f"v{__version__}")

    def test_captures_are_archived_per_version_and_meta_records_the_build(self):
        from lugon_poker.vision import RGBImage

        img = RGBImage(8, 6, bytearray((18, 100, 75) * 48))
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            png_a, json_a = save_diagnostic_capture(img, {"n": 1}, captures_dir=root, version="0.2.0")
            png_b, json_b = save_diagnostic_capture(img, {"n": 2}, captures_dir=root)
            self.assertEqual(png_a.parent.name, "v0.2.0")
            self.assertEqual(png_b.parent.name, f"v{__version__}")
            self.assertEqual(json.loads(json_a.read_text(encoding="utf-8"))["app_version"], "v0.2.0")
            self.assertEqual(json.loads(json_b.read_text(encoding="utf-8"))["app_version"], f"v{__version__}")
            self.assertEqual(list(root.glob("cap_*")), [])  # nothing is dropped loose in the root folder

    def test_pruning_only_touches_the_current_version_folder(self):
        from lugon_poker.vision import RGBImage

        img = RGBImage(4, 4, bytearray((10, 10, 10) * 16))
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            for _ in range(3):
                save_diagnostic_capture(img, {}, captures_dir=root, version="0.1.0")
            for _ in range(5):
                save_diagnostic_capture(img, {}, captures_dir=root, version="0.2.0", max_saved=2)
            self.assertEqual(len(list((root / "v0.1.0").glob("cap_*_full.png"))), 3)
            self.assertEqual(len(list((root / "v0.2.0").glob("cap_*_full.png"))), 2)


class SettingsPersistenceTests(unittest.TestCase):
    def test_settings_round_trip_and_corruption_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "nested" / "settings.json"
            self.assertEqual(load_settings(path), {})
            save_settings(path, {"game_type": "tournament"})
            self.assertEqual(load_settings(path), {"game_type": "tournament"})
            path.write_text("{not json", encoding="utf-8")
            self.assertEqual(load_settings(path), {})
            path.write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(load_settings(path), {})

    def test_assistant_window_dropdown_persists_and_drives_the_analysis(self):
        import tkinter as tk

        from lugon_poker.companion_gui import AssistantWindow

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Display not available: {exc}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            bridge_dir = Path(tmp_dir) / "bridge"
            settings_path = Path(tmp_dir) / "settings.json"
            app = None
            app2 = None
            root2 = None
            try:
                root.withdraw()
                app = AssistantWindow(
                    root=root,
                    bridge_dir=bridge_dir,
                    captures_dir=Path(tmp_dir) / "captures",
                    enable_background_llm=False,
                    settings_path=settings_path,
                )
                self.assertEqual(app.game_type_var.get(), "Spin & Go")
                self.assertEqual(app._current_game_type(), "spin_and_go")
                self.assertIn(f"v{__version__}", root.title())
                self.assertEqual(app.hero_var.get(), "")
                self.assertEqual(app.pot_var.get(), "")
                self.assertFalse(app.capture_options_panel.winfo_manager())
                self.assertFalse(app.opponent_options_panel.winfo_manager())
                app.capture_options_btn.invoke()
                self.assertEqual(app.capture_options_panel.winfo_manager(), "pack")
                app.capture_options_btn.invoke()
                app.opponent_options_btn.invoke()
                self.assertEqual(app.opponent_options_panel.winfo_manager(), "pack")
                app.opponent_options_btn.invoke()
                app.analyze_manual_inputs()
                self.assertIn("Preenche os campos obrigatórios", app.reason_text.get("1.0", tk.END))

                # Click the real menu entry (index 1 = Torneio) so the widget wiring is covered, not just the handler.
                app.game_menu["menu"].invoke(1)
                self.assertEqual(app._current_game_type(), "tournament")
                app.hero_var.set("As Ah")
                app.board_var.set("Ks 7d 2c")
                app.pot_var.set("25")
                app.to_call_var.set("4")
                app.stack_var.set("96")
                app.analyze_manual_inputs()
                pending = app.bridge.read_pending_request()
                self.assertEqual(pending["context"]["game_type"], "tournament")
                self.assertEqual(app.banner_title_var.get(), "")
                self.assertEqual(json.loads(settings_path.read_text(encoding="utf-8"))["game_type"], "tournament")

                # A fresh launch remembers the last chosen format.
                root2 = tk.Tk()
                root2.withdraw()
                app2 = AssistantWindow(
                    root=root2,
                    bridge_dir=bridge_dir,
                    captures_dir=Path(tmp_dir) / "captures",
                    enable_background_llm=False,
                    settings_path=settings_path,
                )
                self.assertEqual(app2.game_type_var.get(), "Torneio (MTT)")
            finally:
                for window in (app, app2):
                    if window is not None:
                        window._on_close()
                for r in (root, root2):
                    try:
                        if r is not None:
                            r.destroy()
                    except Exception:
                        pass


if __name__ == "__main__":
    unittest.main()
