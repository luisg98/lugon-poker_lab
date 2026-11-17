import json
from pathlib import Path
import tempfile
import unittest

from lugon_poker.companion_gui import _pick_fast_strategy_model, evaluate_vision_report
from lugon_poker.live_vision import (
    DEFAULT_TABLE_PRESET,
    DEFAULT_VLM_TIMEOUT_SECONDS,
    _is_thinking_vlm,
    _parse_optional_nonneg_float,
    analyze_live_image,
    build_vision_prompt,
    classify_mercury_suit_rgb,
    discover_vision_models,
    extract_table_with_vlm,
    normalize_card_token,
    pick_vision_model,
)
from lugon_poker.mercury_vision import (
    _MERCURY_RANK_TEMPLATES,
    TEMPLATE_HEIGHT,
    TEMPLATE_WIDTH,
    analyze_mercury_table,
    detect_mercury_table_rect,
)
from lugon_poker.synthetic import render_scene
from lugon_poker.vision import RGBImage, load_scene


SCENE_PATH = Path(__file__).parents[1] / "examples" / "vision" / "flop_table.json"


class LiveVisionTests(unittest.TestCase):
    def test_normalize_card_token_handles_tens_and_unicode_suits(self):
        self.assertEqual(normalize_card_token("10s"), "Ts")
        self.assertEqual(normalize_card_token("10♠"), "Ts")
        self.assertEqual(normalize_card_token("a♥"), "Ah")
        self.assertEqual(normalize_card_token("Q♦"), "Qd")
        self.assertEqual(normalize_card_token("j♣"), "Jc")
        with self.assertRaises(ValueError):
            normalize_card_token("K5")

    def test_discover_vision_models_filters_and_prioritizes_qwen3_vl(self):
        def fake_get(url: str, timeout: float):
            if "11434" in url:
                return {
                    "models": [
                        {"name": "gpt-oss:20b"},
                        {"name": "llama3.1:8b"},
                        {"name": "llava:7b"},
                        {"name": "qwen3-vl:4b"},
                    ]
                }
            return {"data": []}

        found = discover_vision_models(http_get=fake_get)
        self.assertTrue(found["ollama_online"])
        self.assertEqual(found["ollama_vision_models"], ["llava:7b", "qwen3-vl:4b"])
        self.assertEqual(found["recommended_provider"], "ollama")
        self.assertEqual(found["recommended_model"], "qwen3-vl:4b")

        # Prefer the smaller instruct model over larger or thinking variants.
        self.assertEqual(
            pick_vision_model(["qwen3-vl:4b", "qwen3-vl:2b-instruct", "qwen3-vl:4b-instruct"]),
            "qwen3-vl:2b-instruct",
        )
        self.assertEqual(
            pick_vision_model(["qwen3-vl:4b", "qwen2.5vl:3b"]),
            "qwen2.5vl:3b",
        )
        self.assertEqual(DEFAULT_VLM_TIMEOUT_SECONDS, 10.0)
        self.assertFalse(_is_thinking_vlm("qwen3-vl:4b-instruct"))
        self.assertTrue(_is_thinking_vlm("qwen3-vl:4b"))

    def test_timeout_failure_returns_clear_message(self):
        blank = RGBImage(640, 360, bytearray((20, 20, 20) * 640 * 360))

        def fake_timeout(url, headers, payload, timeout):
            self.assertEqual(timeout, 10.0)
            self.assertEqual(payload["options"]["num_ctx"], 3072)
            self.assertEqual(payload["options"]["num_predict"], 320)
            raise TimeoutError("timed out")

        report = extract_table_with_vlm(
            blank,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            transport=fake_timeout,
        )
        self.assertIsNone(report["hand_state"])
        self.assertTrue(any("10s" in amb for amb in report["ambiguities"]))

    def test_european_decimal_and_fast_advisor_picker(self):
        ambiguities: list[str] = []
        self.assertEqual(_parse_optional_nonneg_float("12,2 BB", "hero_stack", ambiguities), 12.2)
        self.assertEqual(_parse_optional_nonneg_float("0,5", "to_call", ambiguities), 0.5)
        self.assertEqual(_parse_optional_nonneg_float("1.500 $", "hero_stack", ambiguities), 1500.0)
        self.assertEqual(_parse_optional_nonneg_float("1.250,50 €", "pot", ambiguities), 1250.5)
        self.assertEqual(_parse_optional_nonneg_float("$1,250", "pot", ambiguities), 1250.0)
        self.assertEqual(ambiguities, [])

        self.assertEqual(classify_mercury_suit_rgb(139, 44, 44), "h")
        self.assertEqual(classify_mercury_suit_rgb(40, 85, 124), "d")
        self.assertEqual(classify_mercury_suit_rgb(45, 74, 54), "c")
        self.assertEqual(classify_mercury_suit_rgb(43, 45, 48), "s")
        self.assertEqual(classify_mercury_suit_rgb(135, 60, 60), "h")
        self.assertEqual(classify_mercury_suit_rgb(69, 121, 137), "d")
        self.assertEqual(classify_mercury_suit_rgb(98, 146, 67), "c")
        self.assertEqual(classify_mercury_suit_rgb(101, 101, 101), "s")
        self.assertIsNone(classify_mercury_suit_rgb(245, 245, 245))

        mercury_prompt = build_vision_prompt(DEFAULT_TABLE_PRESET)
        self.assertIn("red card=h, blue card=d, green card=c", mercury_prompt)
        self.assertIn("Desisto=Fold", mercury_prompt)
        self.assertNotIn("Convert 1.500 -> 1500", mercury_prompt)
        self.assertIn("NEVER multiply BB decimals by 1000", mercury_prompt)

        self.assertEqual(
            _pick_fast_strategy_model(["gpt-oss:20b", "qwen2.5:7b-instruct", "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M"]),
            "hf.co/LiquidAI/LFM2.5-2.6B-GGUF:Q4_K_M",
        )

    def test_extract_table_with_vlm_ready_waiting_and_never_guesses_on_errors(self):
        blank = RGBImage(640, 360, bytearray((20, 20, 20) * 640 * 360))

        # 1. Valid Hero turn
        def fake_ready(url, headers, payload, timeout):
            self.assertEqual(payload["model"], "qwen3-vl:4b")
            self.assertEqual(len(payload["messages"][0]["images"]), 1)
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "table_visible": True,
                            "hero_cards": ["A♠", "10♥"],
                            "board": ["Ks", "7d", "2c"],
                            "pot": 25,
                            "to_call": 4,
                            "hero_stack": 96,
                            "hero_position": "BTN",
                            "hero_to_act": True,
                            "active_player": "HERO",
                            "opponents": [
                                {"nickname": "Vitor", "stack": 100, "bet": 4, "position": "CO", "folded": False}
                            ],
                            "ambiguities": [],
                        }
                    )
                }
            }

        report = extract_table_with_vlm(blank, provider="ollama", model="qwen3-vl:4b", transport=fake_ready)
        kind, payload = evaluate_vision_report(report)
        self.assertEqual(kind, "ready")
        self.assertEqual(payload["hero_cards"], ["As", "Th"])
        self.assertEqual(payload["board"], ["Ks", "7d", "2c"])
        self.assertEqual(payload["pot"], 25.0)
        self.assertEqual(payload["to_call"], 4.0)

        # 2. Waiting on opponent turn (to_call not shown yet -> waiting, not failed)
        def fake_waiting(url, headers, payload, timeout):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "table_visible": True,
                            "hero_cards": ["As", "Ah"],
                            "board": [],
                            "pot": 3.5,
                            "to_call": None,
                            "hero_stack": 98.5,
                            "hero_position": "SB",
                            "hero_to_act": False,
                            "active_player": "VILLAIN",
                            "opponents": ["VILLAIN"],
                            "ambiguities": [],
                        }
                    )
                }
            }

        waiting_report = extract_table_with_vlm(blank, provider="ollama", model="qwen3-vl:4b", transport=fake_waiting)
        kind, payload = evaluate_vision_report(waiting_report)
        self.assertEqual(kind, "waiting")
        self.assertEqual(payload["active_player"], "VILLAIN")

        # 3. Duplicate / invalid cards -> LEITURA FALHADA
        def fake_dup(url, headers, payload, timeout):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "table_visible": True,
                            "hero_cards": ["As", "Ah"],
                            "board": ["As", "7d", "2c"],
                            "pot": 20,
                            "to_call": 0,
                            "hero_to_act": True,
                            "opponents": [{"nickname": "VILLAIN"}],
                            "ambiguities": [],
                        }
                    )
                }
            }

        dup_report = extract_table_with_vlm(blank, provider="ollama", model="qwen3-vl:4b", transport=fake_dup)
        kind, payload = evaluate_vision_report(dup_report)
        self.assertEqual(kind, "failed")
        self.assertIsNone(dup_report["hand_state"])

        # 4. Unreachable endpoint -> fails cleanly
        def fake_err(url, headers, payload, timeout):
            raise OSError("connection refused")

        err_report = extract_table_with_vlm(blank, provider="ollama", model="qwen3-vl:4b", transport=fake_err)
        kind, _ = evaluate_vision_report(err_report)
        self.assertEqual(kind, "failed")

    def test_hybrid_mode_uses_calibrated_fast_path_and_falls_back_to_vlm(self):
        scene = load_scene(SCENE_PATH)
        synth_img = render_scene(scene, 960, 540)
        called = {"count": 0}

        def should_not_be_called(*args, **kwargs):
            called["count"] += 1
            raise AssertionError("VLM should not be called on valid calibrated table")

        cal_report = analyze_live_image(synth_img, mode="hybrid", transport=should_not_be_called)
        self.assertEqual(called["count"], 0)
        self.assertEqual(cal_report["vision_source"], "calibrated")
        self.assertIsNotNone(cal_report["hand_state"])

        # Non-calibrated image in hybrid mode delegates to VLM
        real_table_mock = RGBImage(800, 450, bytearray((30, 50, 70) * 800 * 450))

        def fake_vlm(url, headers, payload, timeout):
            called["count"] += 1
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "table_visible": True,
                                    "hero_cards": ["Qd", "Qc"],
                                    "board": ["Js", "8h", "3d"],
                                    "pot": 18,
                                    "to_call": 6,
                                    "hero_stack": 90,
                                    "hero_position": "CO",
                                    "hero_to_act": True,
                                    "active_player": "HERO",
                                    "opponents": [{"nickname": "REG1"}],
                                    "ambiguities": [],
                                }
                            )
                        }
                    }
                ]
            }

        vlm_report = analyze_live_image(
            real_table_mock,
            mode="hybrid",
            vlm_provider="openai_compatible",
            vlm_model="gpt-4o-mini",
            transport=fake_vlm,
        )
        self.assertEqual(called["count"], 1)
        self.assertEqual(vlm_report["vision_source"], "openai_compatible:gpt-4o-mini")
        self.assertEqual(vlm_report["hand_state"]["hero"]["cards"], ["Qd", "Qc"])

        # Calibrated-only mode never invokes VLM
        cal_only = analyze_live_image(real_table_mock, mode="calibrated", transport=should_not_be_called)
        self.assertEqual(called["count"], 1)
        self.assertEqual(cal_only["table_count"], 0)

    def test_assistant_window_live_capture_and_bridge_round_trip(self):
        import tkinter as tk
        from lugon_poker.companion_gui import AssistantWindow
        from lugon_poker.screen_capture import CaptureTarget, get_window_hwnd_from_tk
        from lugon_poker.vision import analyze_image

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Display not available: {exc}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            bridge_dir = Path(tmp_dir) / "bridge"
            captures_dir = Path(tmp_dir) / "captures"
            app = None
            try:
                root.withdraw()
                app = AssistantWindow(
                    root=root,
                    bridge_dir=bridge_dir,
                    captures_dir=captures_dir,
                    enable_background_llm=False,
                )
                scene = load_scene(SCENE_PATH)
                report = analyze_image(render_scene(scene, 960, 540))
                report["vision_source"] = "calibrated"
                app._apply_vision_report(
                    report=report,
                    source_label="Live · calibrated",
                    hand_prefix="LIVE",
                    capture_info={"kind": "region", "rect": [0, 0, 960, 540]},
                )
                pending = app.bridge.read_pending_request()
                self.assertIsNotNone(pending)
                self.assertEqual(pending["capture_info"], {"kind": "region", "rect": [0, 0, 960, 540]})
                self.assertEqual(app.hero_var.get(), "As Ah")
                self.assertEqual(app.board_var.get(), "Ks 7d 2c")
                self.assertIsInstance(get_window_hwnd_from_tk(root.winfo_id()), int)
            finally:
                if app is not None:
                    app._on_close()
                    del app
                else:
                    try:
                        root.destroy()
                    except Exception:
                        pass

    def test_end_to_end_live_window_capture_reads_displayed_table(self):
        import sys
        if sys.platform != "win32":
            self.skipTest("Windows-only live capture test")
        import tkinter as tk
        from lugon_poker.screen_capture import capture_target, CaptureTarget, enable_dpi_awareness, get_window_hwnd_from_tk, image_to_ppm_bytes

        enable_dpi_awareness()
        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Display not available: {exc}")

        result = {}
        try:
            scene = load_scene(SCENE_PATH)
            table_img = render_scene(scene, 960, 540)
            root.overrideredirect(True)
            root.geometry("960x540+20+20")
            photo = tk.PhotoImage(data=image_to_ppm_bytes(table_img))
            lbl = tk.Label(root, image=photo, bd=0, highlightthickness=0)
            lbl.pack()

            def _grab_and_parse():
                hwnd = get_window_hwnd_from_tk(root.winfo_id())
                captured_img, meta = capture_target(CaptureTarget(kind="window", hwnd=hwnd, title="LiveTable"))
                report = analyze_live_image(captured_img, mode="calibrated")
                result["meta"] = meta
                result["report"] = report
                root.destroy()

            root.after(180, _grab_and_parse)
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

        self.assertIn("report", result)
        report = result["report"]
        kind, payload = evaluate_vision_report(report)
        self.assertEqual(kind, "ready")
        self.assertEqual(payload["hero_cards"], ["As", "Ah"])
        self.assertEqual(payload["board"], ["Ks", "7d", "2c"])
        self.assertEqual(payload["pot"], 25.0)
        self.assertEqual(payload["to_call"], 4.0)

    def _fill_rect(self, img: RGBImage, x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
        row_bytes = bytes(rgb) * max(0, x1 - x0)
        stride = img.width * 3
        for y in range(max(0, y0), min(img.height, y1)):
            s = y * stride + max(0, x0) * 3
            img.pixels[s : s + len(row_bytes)] = row_bytes

    def _draw_mercury_card(
        self,
        img: RGBImage,
        x0: int,
        y0: int,
        width: int,
        height: int,
        card: str,
        glyph: str | None = None,
    ) -> None:
        """Draw a Mercury card; ``glyph`` overrides the rank bitmap ('|'-separated rows) and ``rank='?'`` leaves it blank."""
        suit_colors = {
            "h": (146, 65, 65),
            "d": (72, 127, 143),
            "c": (105, 157, 71),
            "s": (101, 101, 101),
        }
        rank, suit = card[0], card[1]
        self._fill_rect(img, x0, y0, x0 + width, y0 + height, suit_colors[suit])
        if rank == "?":
            return
        tpl = (
            tuple(1 if ch == "#" else 0 for ch in glyph.replace("|", ""))
            if glyph is not None
            else _MERCURY_RANK_TEMPLATES[rank]
        )
        gx0, gy0 = x0 + 8, y0 + 10
        for dy in range(TEMPLATE_HEIGHT):
            for dx in range(TEMPLATE_WIDTH):
                if tpl[dy * TEMPLATE_WIDTH + dx]:
                    self._fill_rect(img, gx0 + dx * 2, gy0 + dy * 2, gx0 + dx * 2 + 2, gy0 + dy * 2 + 2, (250, 250, 250))

    def _render_mock_mercury_screen(
        self,
        *,
        hero_cards: tuple[str, str],
        board: tuple[str, ...],
        has_fold: bool,
        has_action_btns: bool,
        seat_left_active: bool = True,
        seat_right_active: bool = True,
        dealer: str = "right",
        window_rect: tuple[int, int, int, int] = (4, 31, 1411, 1000),
        titlebar_rgb: tuple[int, int, int] = (243, 243, 243),
        glyph_overrides: dict[str, str] | None = None,
    ) -> RGBImage:
        # 1920x1080 desktop frame with PokerStars Mercury client at window_rect.
        img = RGBImage(1920, 1080, bytearray((15, 23, 42) * 1920 * 1080))
        tx, ty, tw, th = window_rect
        tb_top = max(2, ty - 29)
        self._fill_rect(img, tx, tb_top, tx + tw, ty, titlebar_rgb)  # OS titlebar
        self._fill_rect(img, tx, ty, tx + tw, ty + th, (52, 74, 102))  # Mercury steel-blue bg
        self._fill_rect(img, tx + round(0.18 * tw), ty + round(0.28 * th), tx + round(0.82 * tw), ty + round(0.60 * th), (24, 24, 24))  # Dark felt
        border_y = ty + th + 1
        border_color = titlebar_rgb if titlebar_rgb != (243, 243, 243) else (69, 69, 68)
        self._fill_rect(img, tx, border_y, tx + tw, border_y + 1, border_color)  # Bottom border
        self._fill_rect(img, tx, 1033, 1920, 1080, (36, 35, 34))  # Taskbar

        overrides = glyph_overrides or {}
        hero_u = (0.4382, 0.5025)
        for u, card in zip(hero_u, hero_cards):
            self._draw_mercury_card(img, tx + round(u * tw), ty + round(0.606 * th), 88, 60, card, overrides.get(card))

        board_u = (0.3350, 0.4021, 0.4700, 0.5378, 0.6049)
        for u, card in zip(board_u, board):
            self._draw_mercury_card(img, tx + round(u * tw), ty + round(0.315 * th), 88, 120, card, overrides.get(card))

        if has_action_btns:
            if has_fold:
                self._fill_rect(img, tx + round(0.510 * tw), ty + round(0.920 * th), tx + round(0.640 * tw), ty + round(0.965 * th), (175, 42, 42))
            self._fill_rect(img, tx + round(0.675 * tw), ty + round(0.920 * th), tx + round(0.810 * tw), ty + round(0.965 * th), (175, 42, 42))
            self._fill_rect(img, tx + round(0.840 * tw), ty + round(0.920 * th), tx + round(0.955 * tw), ty + round(0.965 * th), (175, 42, 42))

        if seat_left_active:
            self._fill_rect(img, tx + round(0.105 * tw), ty + round(0.128 * th), tx + round(0.165 * tw), ty + round(0.160 * th), (150, 50, 50))
            self._fill_rect(img, tx + round(0.065 * tw), ty + round(0.180 * th), tx + round(0.130 * tw), ty + round(0.200 * th), (220, 220, 220))
        if seat_right_active:
            self._fill_rect(img, tx + round(0.775 * tw), ty + round(0.128 * th), tx + round(0.845 * tw), ty + round(0.160 * th), (150, 50, 50))
            self._fill_rect(img, tx + round(0.820 * tw), ty + round(0.180 * th), tx + round(0.890 * tw), ty + round(0.200 * th), (220, 220, 220))

        if dealer == "left":
            self._fill_rect(img, tx + round(0.195 * tw), ty + round(0.310 * th), tx + round(0.215 * tw), ty + round(0.340 * th), (190, 45, 45))
        elif dealer == "right":
            self._fill_rect(img, tx + round(0.785 * tw), ty + round(0.310 * th), tx + round(0.805 * tw), ty + round(0.340 * th), (190, 45, 45))
        elif dealer == "hero":
            self._fill_rect(img, tx + round(0.375 * tw), ty + round(0.540 * th), tx + round(0.388 * tw), ty + round(0.560 * th), (190, 45, 45))
        return img

    def test_mercury_crop_and_fusion_fixes_real_capture_failures(self):
        # 1. Preflop check-available with VLM hallucinating board=hero_cards, to_call=1, and leaked Luisfg98
        screen = self._render_mock_mercury_screen(
            hero_cards=("5h", "2h"),
            board=(),
            has_fold=False,
            has_action_btns=True,
            seat_left_active=False,
            seat_right_active=True,
            dealer="right",
        )
        self.assertEqual(detect_mercury_table_rect(screen), (4, 31, 1411, 1000))
        merc = analyze_mercury_table(screen)
        self.assertIsNotNone(merc)
        self.assertEqual(merc["hero_cards"], ["5h", "2h"])
        self.assertEqual(merc["board"], [])
        self.assertTrue(merc["hero_to_act"])
        self.assertTrue(merc["check_available"])
        self.assertEqual(merc["hero_position"], "BB")

        def fake_noisy_vlm(url, headers, payload, timeout):
            self.assertEqual(payload["options"]["num_predict"], 140)
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "table_visible": True,
                            "hero_cards": [],
                            "board": ["5h", "2h"],
                            "pot": "2 BB",
                            "to_call": 1,
                            "hero_stack": "7,17 BB",
                            "hero_position": "BTN",
                            "hero_to_act": True,
                            "opponents": ["Luisfg98", "Passo"],
                            "ambiguities": ["hero_cards unreadable"],
                        }
                    )
                }
            }

        fixed = extract_table_with_vlm(
            screen,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            preset=DEFAULT_TABLE_PRESET,
            transport=fake_noisy_vlm,
        )
        kind, payload = evaluate_vision_report(fixed)
        self.assertEqual(kind, "ready")
        self.assertEqual(payload["hero_cards"], ["5h", "2h"])
        self.assertEqual(payload["board"], [])
        self.assertEqual(payload["to_call"], 0.0)
        self.assertEqual(payload["hero_position"], "BB")
        self.assertEqual([o["nickname"] for o in payload["opponents"]], ["Oponente Dir"])

        # 2. Flop/turn suit correction (6c -> 6h) & opponent button filtering
        turn_screen = self._render_mock_mercury_screen(
            hero_cards=("5h", "2h"),
            board=("7h", "5s", "6h", "4h"),
            has_fold=False,
            has_action_btns=True,
            seat_left_active=False,
            seat_right_active=True,
            dealer="right",
        )

        def fake_suit_glitch_vlm(url, headers, payload, timeout):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "table_visible": True,
                            "hero_cards": ["5h", "2h"],
                            "board": ["7h", "5s", "6c", "4h"],
                            "pot": 2.0,
                            "to_call": 1.0,
                            "hero_stack": 7.17,
                            "hero_to_act": True,
                            "opponents": ["Kubo1975", "Pago"],
                            "ambiguities": [],
                        }
                    )
                }
            }

        fixed_turn = extract_table_with_vlm(
            turn_screen,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            preset=DEFAULT_TABLE_PRESET,
            transport=fake_suit_glitch_vlm,
        )
        kind2, payload2 = evaluate_vision_report(fixed_turn)
        self.assertEqual(kind2, "ready")
        self.assertEqual(payload2["board"], ["7h", "5s", "6h", "4h"])
        self.assertEqual(payload2["to_call"], 0.0)
        self.assertEqual([o["nickname"] for o in payload2["opponents"]], ["Kubo1975"])

        # 3. Waiting frame (checkboxes / no red buttons) overrides VLM hero_to_act=True -> waiting
        wait_screen = self._render_mock_mercury_screen(
            hero_cards=("9s", "5h"),
            board=("9d", "Qh", "6c", "5s", "Jc"),
            has_fold=False,
            has_action_btns=False,
            seat_left_active=True,
            seat_right_active=True,
            dealer="left",
        )

        def fake_wait_vlm(url, headers, payload, timeout):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "table_visible": True,
                            "hero_cards": ["9s", "5h"],
                            "board": ["9d", "Qh", "6c", "5s", "Jc"],
                            "pot": 9,
                            "to_call": 1,
                            "hero_stack": 34,
                            "hero_to_act": True,
                            "opponents": ["avrouner", "Kubo1975"],
                            "ambiguities": [],
                        }
                    )
                }
            }

        fixed_wait = extract_table_with_vlm(
            wait_screen,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            preset=DEFAULT_TABLE_PRESET,
            transport=fake_wait_vlm,
        )
        kind3, _ = evaluate_vision_report(fixed_wait)
        self.assertEqual(kind3, "waiting")

        # 4. Focused Windows accent titlebar (#0078D4) + shifted/smaller window (8, 58, 1348, 938) + 8h/8s disambiguation
        shifted = self._render_mock_mercury_screen(
            hero_cards=("Kc", "3h"),
            board=("Ah", "5d", "Th", "8h", "8s"),
            has_fold=True,
            has_action_btns=True,
            seat_left_active=True,
            seat_right_active=True,
            dealer="hero",
            window_rect=(8, 58, 1348, 938),
            titlebar_rgb=(0, 120, 212),
        )
        self.assertEqual(detect_mercury_table_rect(shifted), (8, 58, 1348, 938))
        merc_shifted = analyze_mercury_table(shifted)
        self.assertIsNotNone(merc_shifted)
        self.assertEqual(merc_shifted["hero_cards"], ["Kc", "3h"])
        self.assertEqual(merc_shifted["board"], ["Ah", "5d", "Th", "8h", "8s"])
        self.assertEqual(merc_shifted["hero_position"], "BTN")
        self.assertFalse(merc_shifted["check_available"])

    # '9' bitmap measured on a real 1323x917 PokerStars client (v0.2.0 captures 194853/194921).
    _REAL_NINE_GLYPH = (
        "..########..|.###########|############|###......###|###.......##|###.......##|###......###|"
        "############|.###########|..######.###|.........###|###......###|###......###|###......###|"
        "############|.##########."
    )

    def _fake_vlm(self, **fields):
        base = {
            "table_visible": True,
            "hero_cards": [],
            "board": [],
            "pot": 1.5,
            "to_call": 0.5,
            "hero_stack": 45.2,
            "hero_to_act": True,
            "opponents": ["LaPitchoune52"],
            "ambiguities": [],
        }
        base.update(fields)

        def transport(url, headers, payload, timeout):
            self.last_vlm_prompt = payload["messages"][0]["content"]
            return {"message": {"content": json.dumps(base)}}

        return transport

    def test_real_world_nine_glyph_is_read_instead_of_rejected(self):
        # Before 0.3.0 every 9 scored 0.92 vs 0.90 for the '8' (margin 0.03 < 0.04) and the card was dropped.
        screen = self._render_mock_mercury_screen(
            hero_cards=("Ac", "9h"),
            board=("9d", "Qh", "6c"),
            has_fold=False,
            has_action_btns=True,
            glyph_overrides={"9h": self._REAL_NINE_GLYPH, "9d": self._REAL_NINE_GLYPH},
        )
        merc = analyze_mercury_table(screen)
        self.assertTrue(merc["hero_complete"])
        self.assertTrue(merc["board_complete"])
        self.assertEqual(merc["hero_cards"], ["Ac", "9h"])
        self.assertEqual(merc["board"], ["9d", "Qh", "6c"])

    def test_prize_banner_is_never_read_as_board_cards(self):
        # Real v0.2.0 failure: the "€5 ♠♠♠♠" banner became board 5s5s5s5s5s and every preflop hand failed validation.
        screen = self._render_mock_mercury_screen(
            hero_cards=("Kd", "6d"),
            board=(),
            has_fold=True,
            has_action_btns=True,
            seat_left_active=True,
            seat_right_active=False,
            dealer="hero",
        )
        report = extract_table_with_vlm(
            screen,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            preset=DEFAULT_TABLE_PRESET,
            transport=self._fake_vlm(board=["5s"] * 5, hero_cards=["Kd", "6c"]),
        )
        kind, payload = evaluate_vision_report(report)
        self.assertEqual(kind, "ready")
        self.assertEqual(payload["board"], [])
        self.assertEqual(payload["hero_cards"], ["Kd", "6d"])  # suit read from the card colour, not from the VLM

        # The VLM crop must start below the banner (5%-12.5% of the table height).
        from lugon_poker.mercury_vision import prepare_mercury_vlm_crop

        rect = detect_mercury_table_rect(screen)
        crop = prepare_mercury_vlm_crop(screen, rect)
        self.assertLessEqual(crop.height, rect[3] - round(0.125 * rect[3]))

    def test_pixel_board_slot_count_beats_phantom_vlm_cards(self):
        # Three card bodies on the felt (third rank unreadable) while the VLM claims five: never trust the extras.
        screen = self._render_mock_mercury_screen(
            hero_cards=("Kd", "6d"),
            board=("9d", "Qh", "?c"),
            has_fold=True,
            has_action_btns=True,
        )
        report = extract_table_with_vlm(
            screen,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            preset=DEFAULT_TABLE_PRESET,
            transport=self._fake_vlm(board=["5s"] * 5),
        )
        self.assertEqual(report["board"], ["9d", "Qh"])
        self.assertTrue(any("board[2]" in item for item in report["ambiguities"]))
        self.assertEqual(evaluate_vision_report(report)[0], "failed")

    def test_tournament_mode_skips_spin_and_go_seat_layout_and_uses_the_open_prompt(self):
        screen = self._render_mock_mercury_screen(
            hero_cards=("Kd", "6d"),
            board=("9d", "Qh", "6c"),
            has_fold=True,
            has_action_btns=True,
            seat_left_active=True,
            seat_right_active=True,
        )
        spin = analyze_mercury_table(screen, game_type="spin_and_go")
        mtt = analyze_mercury_table(screen, game_type="tournament")
        self.assertEqual(spin["active_opponents_count"], 2)
        self.assertIsNone(mtt["active_opponents_count"])
        self.assertIsNone(mtt["hero_position"])
        self.assertEqual(mtt["hero_cards"], spin["hero_cards"])
        self.assertEqual(mtt["board"], spin["board"])

        report = extract_table_with_vlm(
            screen,
            provider="ollama",
            model="qwen3-vl:4b-instruct",
            preset=DEFAULT_TABLE_PRESET,
            game_type="tournament",
            transport=self._fake_vlm(board=["9d", "Qh", "6c"], opponents=["avrouner", "Kubo1975", "tiger77"]),
        )
        self.assertIn("EVERY other seated player", self.last_vlm_prompt)
        self.assertEqual([p["nickname"] for p in report["players"] if not p["hero"]], ["avrouner", "Kubo1975", "tiger77"])

    def test_replay_rereads_archived_captures_with_the_current_code(self):
        from lugon_poker.replay import format_replay, replay_captures
        from lugon_poker.screen_capture import save_diagnostic_capture

        screen = self._render_mock_mercury_screen(
            hero_cards=("Kd", "6d"),
            board=(),
            has_fold=True,
            has_action_btns=True,
            seat_left_active=True,
            seat_right_active=False,
            dealer="hero",
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_diagnostic_capture(
                screen,
                {
                    "evaluate_status": "failed",
                    "hero_cards": ["Kd", "6c"],
                    "board": ["5s"] * 5,
                    "ambiguities": ["cards: cartas duplicadas detetadas entre hero_cards e board"],
                },
                captures_dir=Path(tmp_dir),
                version="0.2.0",
            )
            records = replay_captures(Path(tmp_dir) / "v0.2.0")
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["old_version"], "v0.2.0")
        self.assertEqual(rec["old"]["status"], "failed")
        self.assertEqual(rec["old"]["board"], ["5s"] * 5)
        self.assertEqual(rec["new"]["status"], "complete")
        self.assertEqual(rec["new"]["hero"], ["Kd", "6d"])
        self.assertEqual(rec["new"]["board"], [])
        text = format_replay(records)
        self.assertIn("!=", text)
        self.assertIn("1 capturas reanalisadas, 1 com leitura diferente", text)


if __name__ == "__main__":
    unittest.main()
