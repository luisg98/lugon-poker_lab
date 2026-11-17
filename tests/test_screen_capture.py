import tempfile
import unittest
from pathlib import Path

from lugon_poker.screen_capture import (
    CaptureTarget,
    _bgra_to_rgb_image,
    capture_window,
    compute_frame_signature,
    crop_rgb_image,
    format_capture_version_tag,
    frame_difference,
    get_window_hwnd_from_tk,
    image_to_ppm_bytes,
    mask_screen_rect_on_image,
    resize_rgb_image,
    save_diagnostic_capture,
)
from lugon_poker.synthetic import _filled_rect, render_scene
from lugon_poker.vision import RGBImage, encode_png_bytes, load_png, load_scene


SCENE_PATH = Path(__file__).parents[1] / "examples" / "vision" / "flop_table.json"


class ScreenCaptureTests(unittest.TestCase):
    def test_capture_target_validation_and_description(self):
        fs = CaptureTarget(kind="fullscreen")
        self.assertIn("Ecrã", fs.describe())

        reg = CaptureTarget(kind="region", region=(100, 200, 960, 540))
        self.assertIn("960×540", reg.describe())

        win = CaptureTarget(kind="window", hwnd=12345, title="PokerNow - Mesa 1")
        self.assertIn("PokerNow", win.describe())

        with self.assertRaises(ValueError):
            CaptureTarget(kind="region", region=(0, 0, 0, 100))
        with self.assertRaises(ValueError):
            CaptureTarget(kind="window", hwnd=None)
        with self.assertRaises(ValueError):
            CaptureTarget(kind="invalid")

    def test_bgra_to_rgb_crop_resize_and_ppm(self):
        # 2x2 BGRA pixels: red, green, blue, white
        bgra = bytearray([
            0, 0, 255, 255,
            0, 255, 0, 255,
            255, 0, 0, 255,
            255, 255, 255, 255,
        ])
        img = _bgra_to_rgb_image(bgra, 2, 2)
        self.assertEqual(img.get(0, 0), (255, 0, 0))
        self.assertEqual(img.get(1, 0), (0, 255, 0))
        self.assertEqual(img.get(0, 1), (0, 0, 255))
        self.assertEqual(img.get(1, 1), (255, 255, 255))

        cropped = crop_rgb_image(img, 1, 0, 1, 2)
        self.assertEqual((cropped.width, cropped.height), (1, 2))
        self.assertEqual(cropped.get(0, 0), (0, 255, 0))
        self.assertEqual(cropped.get(0, 1), (255, 255, 255))

        ppm = image_to_ppm_bytes(cropped)
        self.assertTrue(ppm.startswith(b"P6\n1 2\n255\n"))
        self.assertEqual(len(ppm), len(b"P6\n1 2\n255\n") + 6)

        big = RGBImage(1600, 900, bytearray((12, 34, 56) * 1600 * 900))
        resized = resize_rgb_image(big, 800, 600)
        self.assertEqual((resized.width, resized.height), (800, 450))
        self.assertEqual(resized.get(400, 225), (12, 34, 56))
        self.assertIs(resize_rgb_image(resized, 800, 600), resized)

    def test_encode_png_bytes_round_trip(self):
        scene = load_scene(SCENE_PATH)
        image = render_scene(scene, 640, 360)
        raw = encode_png_bytes(image)
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir) / "encoded.png"
            p.write_bytes(raw)
            loaded = load_png(p)
        self.assertEqual((loaded.width, loaded.height), (640, 360))
        self.assertEqual(loaded.pixels, image.pixels)

    def test_frame_signature_detects_table_changes_and_ignores_identical_frames(self):
        scene = load_scene(SCENE_PATH)
        img1 = render_scene(scene, 960, 540)
        img2 = render_scene(scene, 960, 540)
        sig1 = compute_frame_signature(img1)
        sig2 = compute_frame_signature(img2)
        self.assertEqual(frame_difference(sig1, sig2), 0.0)
        self.assertEqual(frame_difference(sig1, None), 1.0)

        # Turn card added -> clear localized difference above threshold
        turn_scene = dict(scene)
        turn_scene["board"] = [*scene["board"], "Jc"]
        img_turn = render_scene(turn_scene, 960, 540)
        sig_turn = compute_frame_signature(img_turn)
        diff = frame_difference(sig1, sig_turn)
        self.assertGreater(diff, 0.003)

        # Small single-pixel region change
        img3 = render_scene(scene, 960, 540)
        _filled_rect(img3, 430, 350, 36, 50, (18, 100, 75))
        self.assertGreater(frame_difference(sig1, compute_frame_signature(img3)), 0.003)

    def test_mask_screen_rect_on_image_blanks_overlapping_area(self):
        img = RGBImage(10, 10, bytearray((200, 200, 200) * 100))
        mask_screen_rect_on_image(
            img,
            capture_rect=(100, 100, 10, 10),
            mask_rect=(106, 104, 10, 10),
            fill=(15, 23, 42),
        )
        self.assertEqual(img.get(0, 0), (200, 200, 200))
        self.assertEqual(img.get(5, 3), (200, 200, 200))
        self.assertEqual(img.get(6, 4), (15, 23, 42))
        self.assertEqual(img.get(9, 9), (15, 23, 42))

    def test_save_diagnostic_capture_writes_png_and_json_and_prunes_oldest(self):
        import json
        import time

        img = RGBImage(8, 6, bytearray((18, 100, 75) * 48))
        with tempfile.TemporaryDirectory() as tmp_dir:
            cap_dir = Path(tmp_dir) / "captures"
            paths = []
            for idx in range(3):
                png_p, json_p = save_diagnostic_capture(
                    fullscreen_image=img,
                    metadata={"seq": idx, "status": "ready"},
                    captures_dir=cap_dir,
                    max_saved=2,
                )
                paths.append((png_p, json_p))
                time.sleep(0.01)

            # Captures are archived under data/captures/v<version>/ so they trace back to the build that took them.
            version_dir = cap_dir / format_capture_version_tag()
            remaining_pngs = sorted(version_dir.glob("cap_*_full.png"))
            remaining_jsons = sorted(version_dir.glob("cap_*_meta.json"))
            self.assertEqual(len(remaining_pngs), 2)
            self.assertEqual(len(remaining_jsons), 2)
            loaded = load_png(remaining_pngs[-1])
            self.assertEqual((loaded.width, loaded.height), (8, 6))
            meta = json.loads(remaining_jsons[-1].read_text(encoding="utf-8"))
            self.assertEqual(meta["seq"], 2)
            self.assertEqual(meta["app_version"], format_capture_version_tag())

    def test_capture_tk_window_on_windows(self):
        import sys
        if sys.platform != "win32":
            self.skipTest("Windows-only PrintWindow test")
        import tkinter as tk

        try:
            root = tk.Tk()
        except tk.TclError as exc:
            self.skipTest(f"Display not available: {exc}")

        captured = {}
        try:
            root.title("LugonCaptureProbe")
            root.geometry("320x200+50+50")
            root.configure(bg="#12644b")
            tk.Label(root, text="TESTE", bg="#12644b", fg="#ffffff").pack(expand=True)

            def _do_capture():
                hwnd = get_window_hwnd_from_tk(root.winfo_id())
                img, info = capture_window(hwnd)
                captured["img"] = img
                captured["info"] = info
                root.destroy()

            root.after(150, _do_capture)
            root.mainloop()
        finally:
            try:
                root.destroy()
            except Exception:
                pass

        self.assertIn("img", captured)
        img = captured["img"]
        info = captured["info"]
        self.assertGreater(img.width, 200)
        self.assertGreater(img.height, 150)
        self.assertEqual(info.title, "LugonCaptureProbe")
        center = img.get(img.width // 2, img.height // 2)
        # Green felt (#12644b -> (18, 100, 75)) should dominate center area
        self.assertGreater(center[1], center[0])


if __name__ == "__main__":
    unittest.main()
