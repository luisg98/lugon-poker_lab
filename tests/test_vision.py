import tempfile
import unittest
from pathlib import Path

from lugon_poker.synthetic import _filled_rect, render_scene
from lugon_poker.vision import RGBImage, analyze_image, load_png, load_scene, save_png, compare_fixture
from lugon_poker.vision_cli import _expected


SCENE_PATH = Path(__file__).parents[1] / "examples" / "vision" / "flop_table.json"


class TableVisionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_scene(SCENE_PATH)
        cls.expected = _expected(cls.scene)

    def test_recognizes_synthetic_table_at_two_resolutions(self):
        for width, height in ((640, 360), (800, 450), (960, 540), (1280, 720), (1600, 900)):
            with self.subTest(resolution=f"{width}x{height}"):
                report = analyze_image(render_scene(self.scene, width, height))
                self.assertEqual(compare_fixture(self.expected, report), {"fields_correct": 8, "fields_total": 8})
                self.assertEqual(report["ambiguities"], [])
                self.assertEqual(report["hand_state"]["board"], ["Ks", "7d", "2c"])
                self.assertTrue(report["hero_to_act"])
                self.assertEqual(report["to_call"], 4.0)

    def test_png_round_trip_keeps_extracted_state(self):
        image = render_scene(self.scene)
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "table.png"
            save_png(image, image_path)
            report = analyze_image(load_png(image_path))
        self.assertEqual(report["hero_cards"], ["As", "Ah"])
        self.assertEqual(report["pot"], 25.0)

    def test_missing_board_card_is_left_unknown(self):
        image = render_scene(self.scene)
        # Erase the middle flop card; the detector must not shift the later cards left.
        _filled_rect(image, 393, 230, 36, 50, (18, 100, 75))
        report = analyze_image(image)
        self.assertIsNone(report["hand_state"])
        self.assertEqual(report["board"], ["Ks"])
        self.assertTrue(any("carta depois de um espaço vazio" in item for item in report["ambiguities"]))

    def test_visible_but_unreadable_turn_prevents_state_creation(self):
        scene = dict(self.scene)
        scene["board"] = [*self.scene["board"], "Jc"]
        image = render_scene(scene)
        # Leave the white card body visible but erase its rank/suit glyph.
        _filled_rect(image, 479 + 4, 230 + 5, 30, 16, (245, 245, 238))
        report = analyze_image(image)
        self.assertIsNone(report["hand_state"])
        self.assertTrue(any("carta ilegível" in item for item in report["ambiguities"]))

    def test_non_table_image_is_unknown(self):
        image = RGBImage(960, 540, bytearray((20, 20, 20) * 960 * 540))
        report = analyze_image(image)
        self.assertEqual(report["table_count"], 0)
        self.assertIsNone(report["hand_state"])
        self.assertTrue(any("layout calibrado" in item for item in report["ambiguities"]))


if __name__ == "__main__":
    unittest.main()
