"""Three-max ICM push/fold solver checks."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from lugon_poker.cards import evaluate, parse_cards
from lugon_poker.preflop import HAND_CLASSES
from lugon_poker.three_max import (
    _crossfit_stratified_br_gain,
    _make_stratified_deals,
    _settle,
    solve_three_max_push_fold,
    solve_three_max_push_fold_stratified,
)
from lugon_poker.three_max_model import ACTION_NAMES, ThreeMaxFrequencyModel, train_three_max_frequency_model


def _synthetic_model_rows():
    rows = []
    stack_nodes = (6.0, 10.0, 15.0)
    for hand_index, hand in enumerate(HAND_CLASSES):
        for stack in stack_nodes:
            x = (stack - 10.5) / 4.5
            targets = {}
            for action_index, action in enumerate(ACTION_NAMES):
                base = 0.45 + ((hand_index * 7 + action_index * 3) % 12) / 100
                frequency = base + (0.08 + action_index / 100) * x
                targets[action] = {
                    "mean_frequency": frequency,
                    "seed_standard_deviation": 0.01,
                    "frequency_by_seed": {"1": frequency - 0.01, "2": frequency + 0.01},
                }
            rows.append(
                {
                    "format": "lugon_3max_push_fold_label_v1",
                    "features": {
                        "hand_class": hand,
                        "stacks_bb": {"BTN": stack, "SB": stack, "BB": stack},
                        "payouts": [0.5, 0.3, 0.2],
                        "small_blind_bb": 0.5,
                        "big_blind_bb": 1.0,
                        "ante_bb_per_player": 0.0,
                    },
                    "targets": targets,
                    "label_provenance": {
                        "sampling_method": "stratified_by_seat_and_hand_class",
                        "all_seeds_within_tolerance": True,
                        "independent_stratified_validation": {
                            "method": "stratified_crossfit",
                            "samples_per_seat_hand_class": 2,
                            "mean_strategy_plugin_nash_conv": 0.001,
                            "mean_strategy_crossfit_br_gain": 0.0,
                        },
                    },
                }
            )
    return rows


class ThreeMaxSolverTests(unittest.TestCase):
    def test_uncalled_excess_is_returned_before_icm(self):
        ranks = (
            evaluate(parse_cards(("As", "Ah", "2c", "3d", "4h", "8s", "9c"))),
            evaluate(parse_cards(("Ks", "Kh", "2c", "3d", "4h", "8s", "9c"))),
            evaluate(parse_cards(("Qs", "Qh", "2c", "3d", "4h", "8s", "9c"))),
        )
        # BTN's 10 is unmatched; the effective contributions are 5, 5 and 1.
        icm = _settle((10, 10, 10), (10, 5, 1), (0, 1), ranks, (1, 0, 0))
        self.assertAlmostEqual(icm[0], 16 / 30)

    def test_folded_chips_are_dead_money_in_the_main_pot(self):
        ranks = (
            evaluate(parse_cards(("As", "Ah", "2c", "3d", "4h", "8s", "9c"))),
            evaluate(parse_cards(("Ks", "Kh", "2c", "3d", "4h", "8s", "9c"))),
            evaluate(parse_cards(("Qs", "Qh", "2c", "3d", "4h", "8s", "9c"))),
        )
        # The BB's folded 3 BB is dead money. BTN and SB have only matched 1 BB,
        # so both are eligible to win that dead money; BTN has the better hand.
        icm = _settle((10, 10, 10), (1, 2, 3), (0, 1), ranks, (1, 0, 0))
        self.assertAlmostEqual(icm[0], 14 / 30)

    def test_solver_returns_six_169_class_action_rows(self):
        solution = solve_three_max_push_fold(
            (10, 10, 10),
            (0.5, 0.3, 0.2),
            samples=40,
            max_iterations=4,
            check_interval=2,
            seed=17,
        )
        self.assertEqual(solution.iterations, 4)
        self.assertEqual(len(solution.strategies), 6)
        self.assertTrue(all(len(row) == len(HAND_CLASSES) for row in solution.strategies))
        self.assertTrue(all(0.0 <= rate <= 1.0 for row in solution.strategies for rate in row))
        self.assertEqual(solution.to_dict()["format"], "lugon_3max_push_fold_icm_v1")
        self.assertTrue(all(value >= 0.0 for value in solution.expected_icm))

    def test_solver_rejects_wrong_player_count(self):
        with self.assertRaises(ValueError):
            solve_three_max_push_fold((10, 10), (1, 0, 0), samples=1, max_iterations=1)

    def test_equal_prizes_make_every_action_equivalent(self):
        solution = solve_three_max_push_fold(
            (10, 10, 10), (1, 1, 1), samples=50, max_iterations=4, check_interval=2
        )
        self.assertAlmostEqual(solution.nash_conv, 0.0)
        self.assertTrue(all(rate == 0.5 for row in solution.strategies for rate in row))

    def test_stratified_sampler_covers_every_seat_and_hand_class(self):
        deals = _make_stratified_deals(1, 11, (8, 8, 8), (0, 0.5, 1), (1, 0, 0))
        self.assertEqual(len(deals), 3)
        self.assertTrue(all(len(seat) == len(HAND_CLASSES) for seat in deals))
        for seat_index, seat in enumerate(deals):
            for hand_index, samples in enumerate(seat):
                self.assertEqual(len(samples), 1)
                self.assertEqual(samples[0].classes[seat_index], hand_index)

    def test_crossfit_reports_no_gain_when_actions_are_identical(self):
        deals = _make_stratified_deals(4, 13, (8, 8, 8), (0, 0.5, 1), (1, 1, 1))
        indifferent = tuple(tuple(0.5 for _ in HAND_CLASSES) for _ in range(6))
        mean_gain, a_to_b, b_to_a = _crossfit_stratified_br_gain(deals, indifferent)
        self.assertAlmostEqual(mean_gain, 0.0, places=12)
        self.assertAlmostEqual(a_to_b, 0.0, places=12)
        self.assertAlmostEqual(b_to_a, 0.0, places=12)

    def test_stratified_solver_records_its_sampling_design(self):
        solution = solve_three_max_push_fold_stratified(
            (8, 8, 8), (0.5, 0.3, 0.2), samples_per_hand_class=2,
            max_iterations=2, check_interval=1, seed=23,
        )
        self.assertEqual(solution.samples, 3 * len(HAND_CLASSES) * 2)
        self.assertEqual(solution.samples_per_hand_class, 2)
        self.assertEqual(solution.sampling_method, "stratified_by_seat_and_hand_class")
        self.assertEqual(solution.to_dict()["sampling"]["samples_per_hand_class"], 2)

    def test_frequency_model_fits_and_round_trips(self):
        rows = _synthetic_model_rows()
        model = train_three_max_frequency_model(rows)
        self.assertEqual(model.training_examples, 3 * len(HAND_CLASSES))
        self.assertLess(
            model.validation["leave_one_stack_out_combo_weighted_mae"],
            model.validation["constant_per_hand_baseline_combo_weighted_mae"],
        )
        prediction = model.predict("AKs", 9)
        self.assertEqual(set(prediction), set(ACTION_NAMES))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in prediction.values()))
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model.save(path)
            loaded = ThreeMaxFrequencyModel.load(path)
        self.assertEqual(loaded.to_dict(), model.to_dict())
        with self.assertRaises(ValueError):
            loaded.predict("AKs", 5)

    def test_frequency_model_rejects_unvalidated_labels(self):
        rows = _synthetic_model_rows()
        del rows[0]["label_provenance"]["independent_stratified_validation"]
        with self.assertRaisesRegex(ValueError, "independent validation provenance"):
            train_three_max_frequency_model(rows)

    def test_frequency_model_rejects_unsupported_payouts(self):
        rows = _synthetic_model_rows()
        rows[0]["features"]["payouts"] = [0.7, 0.3, 0.0]
        with self.assertRaisesRegex(ValueError, "different payout structure"):
            train_three_max_frequency_model(rows)

    def test_frequency_model_rejects_duplicate_labels(self):
        rows = _synthetic_model_rows()
        rows.append(rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate label"):
            train_three_max_frequency_model(rows)

    def test_frequency_model_loader_checks_game_scope(self):
        model = train_three_max_frequency_model(_synthetic_model_rows())
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            payload = model.to_dict()
            payload["scope"]["players"] = 2
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unsupported game scope"):
                ThreeMaxFrequencyModel.load(path)


if __name__ == "__main__":
    unittest.main()
