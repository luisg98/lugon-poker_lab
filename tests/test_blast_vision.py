"""Unit tests for deterministic 888 BLAST visual extraction and numeric-collage VLM fusion."""

from __future__ import annotations

import json
from pathlib import Path
import unittest

from lugon_poker.blast_vision import (
    _BLAST_TEMPLATES,
    analyze_blast_table,
    build_blast_collage_prompt,
    build_blast_numeric_collage,
)
from lugon_poker.live_vision import _parse_888_pot_label, extract_table_with_vlm
from lugon_poker.vision import load_png

CAPTURES_ROOT = Path(__file__).parents[1] / "data" / "captures"


def _find_capture(version: str, prefix: str) -> Path | None:
    folder = CAPTURES_ROOT / version
    if not folder.exists():
        return None
    matches = sorted(folder.glob(f"{prefix}*.png"))
    return matches[0] if matches else None


class BlastVisionTests(unittest.TestCase):
    def test_all_thirteen_rank_templates_are_defined(self):
        self.assertEqual(set(_BLAST_TEMPLATES.keys()), set("23456789TJQKA"))
        for tpl in _BLAST_TEMPLATES.values():
            self.assertEqual(len(tpl), 12 * 16)

    def test_pot_label_parser_handles_clipped_bb_suffix_and_rejects_non_pot(self):
        self.assertEqual(_parse_888_pot_label("Pote total: 2,2 BB"), 2.2)
        self.assertEqual(_parse_888_pot_label("Pote total: 2,2 E"), 2.2)
        self.assertEqual(_parse_888_pot_label("Pote total: 12,8 B"), 12.8)
        self.assertIsNone(_parse_888_pot_label("7,5 BB"))
        self.assertIsNone(_parse_888_pot_label("Pote: 3,3 BB"))

    def test_lobby_window_is_rejected_by_blast_detector(self):
        lobby = _find_capture("v1.0.6", "cap_20261117_172031")
        if lobby is None:
            self.skipTest("archived lobby capture not present")
        self.assertIsNone(analyze_blast_table(load_png(lobby)))

    def test_real_v108_v109_v110_failure_frames_are_read_deterministically(self):
        cases = [
            # (version, prefix, expected_hero, expected_board, expected_pos, expected_tl_active, expected_tr_active)
            ("v1.0.8", "cap_20261117_183602", ["6c", "5d"], [], "BTN", False, True),
            ("v1.0.8", "cap_20261117_184231", ["Js", "6c"], ["9c", "4s", "2h", "Jh"], "SB", True, True),
            ("v1.0.8", "cap_20261117_184249", ["Js", "6c"], ["9c", "4s", "2h", "Jh"], "SB", False, True),
            ("v1.0.9", "cap_20261117_185418", ["Ah", "3s"], ["3h", "As", "Kc"], "BB", True, True),
            ("v1.0.9", "cap_20261117_185548", ["Jc", "4c"], ["Kh", "7d", "5s"], "BB", False, True),
            ("v1.0.10", "cap_20261117_185733", ["Jc", "8h"], [], "BTN", False, True),
        ]
        checked = 0
        for ver, prefix, exp_hero, exp_board, exp_pos, exp_tl_act, exp_tr_act in cases:
            path = _find_capture(ver, prefix)
            if path is None:
                continue
            checked += 1
            blast = analyze_blast_table(load_png(path))
            self.assertIsNotNone(blast, prefix)
            assert blast is not None
            self.assertTrue(blast["hero_complete"], prefix)
            self.assertTrue(blast["board_complete"], prefix)
            self.assertTrue(blast["hero_to_act_visual"], prefix)
            self.assertEqual(blast["hero_cards"], exp_hero, prefix)
            self.assertEqual(blast["board"], exp_board, prefix)
            self.assertEqual(blast["hero_position"], exp_pos, prefix)
            self.assertEqual(blast["seat_left_active"], exp_tl_act, prefix)
            self.assertEqual(blast["seat_right_active"], exp_tr_act, prefix)
        if checked == 0:
            self.skipTest("no archived 888 BLAST captures found")

    def test_collage_fusion_merges_pixel_cards_positions_and_folded_flag(self):
        path = _find_capture("v1.0.8", "cap_20261117_184249")
        if path is None:
            self.skipTest("archived 184249 capture not present")
        img = load_png(path)
        blast = analyze_blast_table(img)
        self.assertIsNotNone(blast)
        assert blast is not None
        collage = build_blast_numeric_collage(img, blast)
        self.assertGreaterEqual(collage.width, 720)
        self.assertIn("TOP BAND", build_blast_collage_prompt(blast))

        captured_prompts: list[str] = []

        def fake_transport(url, headers, payload, timeout):
            captured_prompts.append(payload["messages"][0]["content"])
            reply = {
                "pot_label_text": "Pote total: 12,8 BB",
                "hero_nickname": "lugon98",
                "hero_stack_text": "2,2 BB",
                "seat_left": {"nickname": "Mecaccilamar", "stack_text": "7,5 BB"},
                "seat_right": {"nickname": "dester18", "stack_text": "0 BB"},
                "action_buttons": ["DESISTIR", "ALL-IN 4,2 BB"],
            }
            return {"message": {"content": json.dumps(reply)}}

        rep = extract_table_with_vlm(
            img,
            provider="ollama",
            model="qwen3-vl:2b-instruct",
            preset="888_blast",
            game_type="spin_and_go",
            transport=fake_transport,
        )
        self.assertEqual(len(captured_prompts), 1)
        self.assertIn("888 BLAST poker collage", captured_prompts[0])
        self.assertEqual(rep["hero_cards"], ["Js", "6c"])
        self.assertEqual(rep["board"], ["9c", "4s", "2h", "Jh"])
        self.assertEqual(rep["pot"], 12.8)
        self.assertEqual(rep["to_call"], 2.2)
        self.assertIsNotNone(rep["hand_state"])
        hero_p = next(p for p in rep["players"] if p["hero"])
        self.assertEqual(hero_p["position"], "SB")
        self.assertEqual(hero_p["stack"], 2.2)
        opps_by_name = {p["nickname"]: p for p in rep["players"] if not p["hero"]}
        self.assertNotIn("lugon98", opps_by_name)
        self.assertTrue(opps_by_name["Mecaccilamar"]["folded"])
        self.assertEqual(opps_by_name["Mecaccilamar"]["position"], "BB")
        self.assertEqual(opps_by_name["Mecaccilamar"]["stack"], 7.5)
        self.assertFalse(opps_by_name["dester18"]["folded"])
        self.assertEqual(opps_by_name["dester18"]["position"], "BTN")
        self.assertEqual(opps_by_name["dester18"]["stack"], 0.0)
        # Only active (non-folded) opponents enter hand_state!
        self.assertEqual([o["nickname"] for o in rep["hand_state"]["opponents"]], ["dester18"])


if __name__ == "__main__":
    unittest.main()
