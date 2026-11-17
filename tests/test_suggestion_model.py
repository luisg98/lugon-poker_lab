"""Unit tests for Phase 1 suggestion upgrades: range inference, post-flop context, and 3-max live integration."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import time
import tkinter as tk
import unittest

from lugon_poker.agent_bridge import build_decision_context_from_inputs
from lugon_poker.assistant import (
    DecisionContext,
    GameAssistant,
    LegalAction,
    _analyze_board_and_hand,
    build_assistant_prompt,
    normalize_play_style,
)
from lugon_poker.companion_gui import (
    PLAY_STYLE_LABELS,
    AssistantWindow,
    _infer_big_blind,
    _pick_installed_strategy_model,
    evaluate_vision_report,
    play_style_label,
)
from lugon_poker.range_estimator import RANGE_TIERS, estimate_opponent_range
from lugon_poker.ranges import parse_range
from lugon_poker.three_max_advisor import (
    infer_live_three_max_request,
    predict_hand_advice_from_model,
    solve_hand_advice,
)


class RangeEstimatorTests(unittest.TestCase):
    def test_all_range_tiers_parse_cleanly(self):
        counts = [len(parse_range(text)) for _, _, text in RANGE_TIERS]
        self.assertEqual(len(counts), 6)
        self.assertEqual(counts, sorted(counts))
        self.assertTrue(all(c > 0 for c in counts))

    def test_explicit_range_is_preserved(self):
        est = estimate_opponent_range(
            street="preflop",
            pot=7.6,
            to_call=2.7,
            hero_stack=2.7,
            explicit_range="QQ+,AKs",
        )
        self.assertTrue(est.is_explicit)
        self.assertEqual(est.range_text, "QQ+,AKs")

    def test_live_0004_multiway_allin_folds_k3o(self):
        ctx = build_decision_context_from_inputs(
            hero_cards=["Kh", "3c"],
            board=[],
            pot=7.6,
            to_call=2.7,
            hero_stack=2.7,
            hero_position="BTN",
            big_blind=1.0,
            opponents=[{"nickname": "lugon98"}, {"nickname": "fdkjdkjfd"}],
            equity_iterations=1500,
            seed=7,
            game_type="spin_and_go",
        )
        self.assertIsNotNone(ctx.assumed_ranges)
        self.assertLess(ctx.equity["share"], ctx.math_strategy["pot_odds"])
        rec = GameAssistant(provider="heuristic").recommend(ctx)
        self.assertEqual(rec.recommended_action, "fold")

    def test_profile_vpip_shifts_inferred_range_only_when_adequate_sample(self):
        loose_prof = {"vpip": {"opportunities": 30, "empirical_rate": 0.60}}
        tight_prof = {"vpip": {"opportunities": 30, "empirical_rate": 0.12}}
        tiny_prof = {"vpip": {"opportunities": 5, "empirical_rate": 0.90}}

        base = estimate_opponent_range(
            street="preflop",
            pot=3.5,
            to_call=2.0,
            hero_stack=15.0,
            big_blind=1.0,
            opp_position="BTN",
        )
        loose = estimate_opponent_range(
            street="preflop",
            pot=3.5,
            to_call=2.0,
            hero_stack=15.0,
            big_blind=1.0,
            opp_position="BTN",
            opponent_profile=loose_prof,
        )
        tight = estimate_opponent_range(
            street="preflop",
            pot=3.5,
            to_call=2.0,
            hero_stack=15.0,
            big_blind=1.0,
            opp_position="BTN",
            opponent_profile=tight_prof,
        )
        ignored = estimate_opponent_range(
            street="preflop",
            pot=3.5,
            to_call=2.0,
            hero_stack=15.0,
            big_blind=1.0,
            opp_position="BTN",
            opponent_profile=tiny_prof,
        )
        self.assertGreater(len(parse_range(loose.range_text)), len(parse_range(base.range_text)))
        self.assertLess(len(parse_range(tight.range_text)), len(parse_range(base.range_text)))
        self.assertEqual(ignored.range_text, base.range_text)


class PostflopContextualTests(unittest.TestCase):
    def test_board_pair_is_not_misclassified_as_strong_pair(self):
        info = _analyze_board_and_hand(("7h", "6c"), ("Ks", "Kd", "2c"), "BTN")
        self.assertEqual(info["hand_tier"], "board_pair_only")
        self.assertEqual(info["board_texture"], "paired")

    def test_dry_ip_bet_uses_smaller_cbet_sizing_than_wet_board(self):
        dry_ctx = build_decision_context_from_inputs(
            hero_cards=["As", "Kd"],
            board=["Kh", "7c", "2d"],
            pot=10.0,
            to_call=0.0,
            hero_stack=40.0,
            hero_position="BTN",
            equity_iterations=600,
        )
        wet_ctx = build_decision_context_from_inputs(
            hero_cards=["As", "Kd"],
            board=["Kh", "Qh", "Jh"],
            pot=10.0,
            to_call=0.0,
            hero_stack=40.0,
            hero_position="BTN",
            equity_iterations=600,
        )
        assistant = GameAssistant(provider="heuristic")
        dry_rec = assistant.recommend(dry_ctx)
        wet_rec = assistant.recommend(wet_ctx)
        self.assertEqual(dry_rec.recommended_action, "bet")
        self.assertEqual(wet_rec.recommended_action, "bet")
        self.assertLess(dry_rec.recommended_amount, wet_rec.recommended_amount)

    def test_ip_flush_draw_triggers_semi_bluff_bet(self):
        ctx = DecisionContext(
            hand_id="FD_1",
            street="flop",
            hero_nickname="HERO",
            hero_position="BTN",
            hero_cards=("As", "Ts"),
            board=("Ks", "7s", "2c"),
            hero_hand_description="high card",
            pot=10.0,
            to_call=0.0,
            hero_stack=30.0,
            effective_stack=30.0,
            spr=3.0,
            legal_actions=(LegalAction("check", 0.0, 0.0), LegalAction("bet", 2.0, 30.0)),
            equity={"share": 0.44, "win_rate": 0.43, "tie_rate": 0.02, "loss_rate": 0.55, "samples": 1000, "method": "monte_carlo"},
            math_strategy={"decision": "check", "equity": 0.44, "pot_odds": None, "call_ev": None, "fold_ev": 0.0, "explanation": "x"},
            recent_actions=(),
            opponent_profiles={"VILLAIN": {}},
            game_type="spin_and_go",
        )
        rec = GameAssistant(provider="heuristic").recommend(ctx)
        self.assertEqual(rec.recommended_action, "bet")
        self.assertIn("semi-bluff", rec.reasoning)


class LiveThreeMaxIntegrationTests(unittest.TestCase):
    def test_trained_model_advice_runs_in_milliseconds_in_domain(self):
        t0 = time.perf_counter()
        advice = solve_hand_advice(
            hero_cards=["As", "Kd"],
            node="btn_push",
            stacks_bb=(8.0, 10.0, 12.0),
            prefer_model=True,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        self.assertEqual(advice["engine"], "lugon_three_max_unequal_model")
        self.assertEqual(advice["recommended_action"], "push")
        self.assertLess(elapsed_ms, 250.0)

    def test_out_of_domain_stacks_do_not_use_model_prediction(self):
        advice = predict_hand_advice_from_model(
            hero_cards=["As", "Kd"],
            node="btn_push",
            stacks_bb=(2.7, 5.0, 4.0),
        )
        self.assertIsNone(advice)

    def test_infer_live_three_max_request_detects_nodes_and_guards_non_pushfold(self):
        req_btn = infer_live_three_max_request(
            game_type="spin_and_go",
            board=[],
            hero_position="BTN",
            hero_stack=9.0,
            to_call=0.0,
            big_blind=1.0,
            opponents=[
                {"nickname": "P1", "position": "SB", "stack": 10.0, "bet": 0.5},
                {"nickname": "P2", "position": "BB", "stack": 11.0, "bet": 1.0},
            ],
        )
        self.assertIsNotNone(req_btn)
        self.assertEqual(req_btn["node"], "btn_push")
        self.assertEqual(req_btn["stacks_bb"], (9.0, 10.0, 11.0))

        req_bb = infer_live_three_max_request(
            game_type="spin_and_go",
            board=[],
            hero_position="BB",
            hero_stack=10.0,
            to_call=9.0,
            big_blind=1.0,
            opponents=[
                {"nickname": "P1", "position": "BTN", "stack": 10.0, "bet": 10.0},
                {"nickname": "P2", "position": "SB", "stack": 8.0, "bet": 0.5},
            ],
        )
        self.assertIsNotNone(req_bb)
        self.assertEqual(req_bb["node"], "bb_call_vs_btn_shove")

        # Deep stack > 15 BB open should not force push/fold
        self.assertIsNone(
            infer_live_three_max_request(
                game_type="spin_and_go",
                board=[],
                hero_position="BTN",
                hero_stack=22.0,
                to_call=0.0,
                big_blind=1.0,
                opponents=[
                    {"nickname": "P1", "position": "SB", "stack": 20.0, "bet": 0.5},
                    {"nickname": "P2", "position": "BB", "stack": 20.0, "bet": 1.0},
                ],
            )
        )

    def test_vision_models_excluded_from_strategy_picker(self):
        self.assertIsNone(_pick_installed_strategy_model(["qwen3-vl:2b-instruct", "llava:7b"]))
        self.assertEqual(_pick_installed_strategy_model(["qwen3-vl:2b-instruct", "qwen2.5:7b"]), "qwen2.5:7b")

    def test_infer_big_blind_distinguishes_bb_display_vs_chip_display(self):
        self.assertEqual(
            _infer_big_blind(
                report_big_blind=None,
                capture_info={"title": "BLAST Sit & Go Blinds: 30/60"},
                pot=7.6,
                hero_stack=2.7,
                game_type="spin_and_go",
            ),
            1.0,
        )
        self.assertEqual(
            _infer_big_blind(
                report_big_blind=None,
                capture_info={"title": "BLAST Sit & Go Blinds: 30/60"},
                pot=180.0,
                hero_stack=600.0,
                game_type="spin_and_go",
            ),
            60.0,
        )

    def test_evaluate_vision_report_preserves_opponent_positions_and_stacks(self):
        report = {
            "table_count": 1,
            "ambiguities": [],
            "hero_to_act": True,
            "blinds": {"small": 0.5, "big": 1.0},
            "players": [
                {"nickname": "HERO", "position": "BTN", "stack": 10.0, "bet": 0.0, "hero": True},
                {"nickname": "OPP_SB", "position": "SB", "stack": 9.5, "bet": 0.5, "hero": False},
                {"nickname": "OPP_BB", "position": "BB", "stack": 11.0, "bet": 1.0, "hero": False},
            ],
            "hand_state": {
                "hero": {"cards": ["As", "Kd"]},
                "board": [],
                "pot": 1.5,
                "to_call": 0.0,
                "opponents": [{"nickname": "OPP_SB"}, {"nickname": "OPP_BB"}],
            },
        }
        kind, payload = evaluate_vision_report(report)
        self.assertEqual(kind, "ready")
        self.assertEqual(payload["big_blind"], 1.0)
        self.assertEqual(payload["opponents"][0]["position"], "SB")
        self.assertEqual(payload["opponents"][0]["stack"], 9.5)
        self.assertEqual(payload["opponents"][1]["position"], "BB")
        self.assertEqual(payload["opponents"][1]["bet"], 1.0)


class PlayStylePromptTests(unittest.TestCase):
    def test_normalize_and_labels_round_trip(self):
        self.assertEqual(normalize_play_style(None), "balanced")
        self.assertEqual(normalize_play_style("Equilibrado"), "balanced")
        self.assertEqual(normalize_play_style("Passivo"), "passive")
        self.assertEqual(normalize_play_style("Agressivo"), "aggressive")
        self.assertEqual(normalize_play_style("Criativo"), "creative")
        self.assertEqual(set(PLAY_STYLE_LABELS.values()), {"balanced", "passive", "aggressive", "creative"})
        for label, key in PLAY_STYLE_LABELS.items():
            self.assertEqual(play_style_label(key), label)

    def test_play_style_changes_agent_prompt_only_not_heuristic_math(self):
        contexts = {
            style: build_decision_context_from_inputs(
                hero_cards=["As", "Kd"],
                board=["Kh", "7c", "2d"],
                pot=10.0,
                to_call=0.0,
                hero_stack=40.0,
                equity_iterations=400,
                seed=7,
                play_style=style,
            )
            for style in ("balanced", "passive", "aggressive", "creative")
        }
        prompts = {style: build_assistant_prompt(ctx) for style, ctx in contexts.items()}
        self.assertIn("Equilibrado", prompts["balanced"]["system"])
        self.assertIn("sizings mais curtos", prompts["passive"]["system"])
        self.assertIn("vender caro cada carta", prompts["aggressive"]["system"])
        self.assertIn("fundamento teórico", prompts["creative"]["system"])
        for style, payload in prompts.items():
            self.assertEqual(json.loads(payload["user"])["play_style"], style)

        # Prompt-only change: deterministic heuristic output and math stay identical across styles.
        assistant = GameAssistant(provider="heuristic")
        recs = [assistant.recommend(ctx) for ctx in contexts.values()]
        self.assertEqual(len({(r.recommended_action, r.recommended_amount) for r in recs}), 1)

    def test_assistant_window_persists_play_style_option(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            settings_file = tmp / "settings.json"
            root = tk.Tk()
            root.withdraw()
            try:
                app = AssistantWindow(
                    root=root,
                    bridge_dir=tmp / "bridge",
                    captures_dir=tmp / "caps",
                    enable_background_llm=False,
                    settings_path=settings_file,
                )
                self.assertEqual(app.play_style_var.get(), "Equilibrado")
                app.play_style_var.set("Criativo")
                app._on_play_style_changed()
                saved = json.loads(settings_file.read_text(encoding="utf-8"))
                self.assertEqual(saved["play_style"], "creative")
            finally:
                app._on_close()


if __name__ == "__main__":
    unittest.main()
