"""The 169 starting-hand classes and the preflop all-in equity matrix."""

from pathlib import Path
import tempfile
import unittest

from lugon_poker.preflop import (
    CLASS_INDEX,
    HAND_CLASSES,
    PreflopMatrix,
    class_combos,
    combo_count,
    matchup_equity,
    parse_hand_class,
    upper_triangle_pairs,
)


class HandClassTests(unittest.TestCase):
    def test_there_are_exactly_169_classes_covering_1326_combos(self):
        self.assertEqual(len(HAND_CLASSES), 169)
        self.assertEqual(len(set(HAND_CLASSES)), 169)
        pairs = [c for c in HAND_CLASSES if len(c) == 2]
        suited = [c for c in HAND_CLASSES if c.endswith("s")]
        offsuit = [c for c in HAND_CLASSES if c.endswith("o")]
        self.assertEqual((len(pairs), len(suited), len(offsuit)), (13, 78, 78))
        # Every physical two-card combination belongs to exactly one class.
        self.assertEqual(sum(combo_count(c) for c in HAND_CLASSES), 1326)

    def test_classes_partition_the_deck_into_distinct_physical_combos(self):
        seen: set[tuple[int, int]] = set()
        for hand_class in HAND_CLASSES:
            combos = class_combos(hand_class)
            self.assertEqual(len(combos), combo_count(hand_class), hand_class)
            for first, second in combos:
                self.assertNotEqual(first.code, second.code)
                key = tuple(sorted((first.code, second.code)))
                self.assertNotIn(key, seen, f"{hand_class} repete um combo já usado")
                seen.add(key)
        self.assertEqual(len(seen), 1326)

    def test_parse_normalizes_order_case_and_whitespace(self):
        self.assertEqual(parse_hand_class("ka s"), "AKs")
        self.assertEqual(parse_hand_class("aKo"), "AKo")
        self.assertEqual(parse_hand_class("qq"), "QQ")
        self.assertEqual(parse_hand_class("2 2"), "22")
        for bad in ("AK", "AAs", "XYs", "A", "AKx", "AKso"):
            with self.assertRaises(ValueError, msg=bad):
                parse_hand_class(bad)

    def test_suited_and_offsuit_combos_have_the_right_suit_structure(self):
        for first, second in class_combos("AKs"):
            self.assertEqual(first.suit, second.suit)
        for first, second in class_combos("AKo"):
            self.assertNotEqual(first.suit, second.suit)
        ranks = {card.rank for combo in class_combos("77") for card in combo}
        self.assertEqual(ranks, {7})


class MatchupEquityTests(unittest.TestCase):
    def test_known_published_all_in_equities(self):
        # Classic reference numbers; 30k samples gives a standard error near 0.3 points.
        for a, b, expected in (
            ("AA", "KK", 0.82),
            ("AA", "72o", 0.88),
            ("AKs", "QQ", 0.46),
            ("AKo", "22", 0.47),
            ("JJ", "AKs", 0.54),
        ):
            value = matchup_equity(a, b, samples=30_000, seed=7)
            self.assertAlmostEqual(value, expected, delta=0.02, msg=f"{a} vs {b} = {value:.4f}")

    def test_a_class_against_itself_is_exactly_half(self):
        for hand_class in ("AA", "AKs", "72o"):
            self.assertEqual(matchup_equity(hand_class, hand_class), 0.5)

    def test_mirrored_matchups_sum_to_one(self):
        forward = matchup_equity("AKs", "QQ", samples=30_000, seed=11)
        backward = matchup_equity("QQ", "AKs", samples=30_000, seed=11)
        self.assertAlmostEqual(forward + backward, 1.0, delta=0.01)

    def test_suited_beats_its_offsuit_twin(self):
        suited = matchup_equity("AKs", "QQ", samples=30_000, seed=5)
        offsuit = matchup_equity("AKo", "QQ", samples=30_000, seed=5)
        self.assertGreater(suited, offsuit)

    def test_rejects_invalid_sample_counts(self):
        for bad in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                matchup_equity("AA", "KK", samples=bad)


class MatrixStorageTests(unittest.TestCase):
    def _fake_matrix(self) -> PreflopMatrix:
        # Deterministic synthetic values: enough to prove indexing, symmetry and round-tripping.
        pairs = upper_triangle_pairs()
        values = tuple(((i * 31 + j * 17) % 1000) / 1000.0 for i, j in pairs)
        return PreflopMatrix(values=values, samples=1, seed=0, generated_at="t", app_version="test")

    def test_builder_ordering_matches_reader_offsets(self):
        # The single invariant that keeps the stored matrix from being silently scrambled:
        # the builder writes at enumerate(upper_triangle_pairs()), the reader looks up via _offset.
        pairs = upper_triangle_pairs()
        self.assertEqual(len(pairs), 169 * 168 // 2)
        self.assertEqual(len(pairs), 14196)
        for position, (i, j) in enumerate(pairs):
            self.assertEqual(PreflopMatrix._offset(i, j), position, f"({i},{j})")

    def test_lookup_is_symmetric_and_diagonal_is_exactly_half(self):
        matrix = self._fake_matrix()
        for a, b in (("AA", "KK"), ("72o", "AKs"), ("55", "T9s")):
            self.assertAlmostEqual(matrix.equity(a, b) + matrix.equity(b, a), 1.0, places=12)
        for hand_class in ("AA", "AKs", "72o"):
            self.assertEqual(matrix.equity(hand_class, hand_class), 0.5)

    def test_save_and_load_round_trip(self):
        matrix = self._fake_matrix()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = matrix.save(Path(tmp_dir) / "m.json")
            loaded = PreflopMatrix.load(path)
            self.assertEqual(len(loaded.values), len(matrix.values))
            for a, b in (("AA", "KK"), ("QJs", "33")):
                self.assertAlmostEqual(loaded.equity(a, b), matrix.equity(a, b), places=6)

    def test_corrupt_matrix_is_rejected_rather_than_silently_used(self):
        matrix = self._fake_matrix()
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = matrix.save(Path(tmp_dir) / "m.json")
            import json

            data = json.loads(path.read_text(encoding="utf-8"))
            data["values"] = data["values"][:-5]
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                PreflopMatrix.load(path)

            data["values"] = [0.5] * 14196
            data["classes"] = list(reversed(HAND_CLASSES))
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                PreflopMatrix.load(path)

    def test_equity_vs_range_weights_classes_by_combo_count(self):
        matrix = self._fake_matrix()
        # A range of one class is just that matchup.
        self.assertAlmostEqual(matrix.equity_vs_range("AA", ["KK"]), matrix.equity("AA", "KK"), places=12)
        # AKo holds 12 combos against AKs's 4, so it must dominate the blend 3:1.
        blended = matrix.equity_vs_range("55", ["AKs", "AKo"])
        expected = (4 * matrix.equity("55", "AKs") + 12 * matrix.equity("55", "AKo")) / 16
        self.assertAlmostEqual(blended, expected, places=12)
        # Explicit weights scale on top of combo counts.
        half = matrix.equity_vs_range("55", {"AKs": 1.0, "AKo": 0.5})
        expected_half = (4 * matrix.equity("55", "AKs") + 6 * matrix.equity("55", "AKo")) / 10
        self.assertAlmostEqual(half, expected_half, places=12)
        with self.assertRaises(ValueError):
            matrix.equity_vs_range("55", {"AKs": 0.0})

    def test_class_index_covers_every_class(self):
        self.assertEqual(len(CLASS_INDEX), 169)
        for position, name in enumerate(HAND_CLASSES):
            self.assertEqual(CLASS_INDEX[name], position)


if __name__ == "__main__":
    unittest.main()
