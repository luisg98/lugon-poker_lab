"""Heads-up push/fold Nash solver checks."""

import unittest

from lugon_poker.preflop import HAND_CLASSES, PreflopMatrix, combo_count, upper_triangle_pairs
from lugon_poker.push_fold import solve_push_fold_nash


class PushFoldSolverTests(unittest.TestCase):
    @staticmethod
    def _matrix(value: float = 0.5) -> PreflopMatrix:
        return PreflopMatrix(
            values=tuple(value for _ in upper_triangle_pairs()),
            samples=1,
            seed=0,
            generated_at="test",
            app_version="test",
        )

    def test_symmetric_showdown_game_converges_to_push_and_call(self):
        solution = solve_push_fold_nash(
            self._matrix(), stack_bb=10, max_iterations=1_000, tolerance_bb=0.005
        )

        self.assertTrue(solution.converged)
        self.assertLessEqual(solution.nash_gap_bb, 0.005)
        self.assertGreater(min(solution.small_blind_push), 0.99)
        self.assertGreater(min(solution.big_blind_call), 0.99)

    def test_reported_nash_gap_matches_independent_best_response_calculation(self):
        values = tuple(((i * 29 + j * 7) % 101) / 100.0 for i, j in upper_triangle_pairs())
        matrix = PreflopMatrix(values, 100, 7, "test", "test")
        solution = solve_push_fold_nash(
            matrix,
            stack_bb=8,
            ante_bb=0.1,
            max_iterations=7,
            tolerance_bb=1e-12,
            check_interval=1,
        )

        chance = tuple(combo_count(hand) / 1326 for hand in HAND_CLASSES)
        fold_ev = -(solution.small_blind_bb + solution.ante_bb)
        bb_fold_win = solution.big_blind_bb + solution.ante_bb
        push_mass = [chance[i] * solution.small_blind_push[i] for i in range(len(HAND_CLASSES))]
        total_push_mass = sum(push_mass)

        best_sb_ev = fold_ev
        for i, hand_a in enumerate(HAND_CLASSES):
            push_ev = 0.0
            for j, hand_b in enumerate(HAND_CLASSES):
                called_ev = 2 * solution.stack_bb * matrix.equity(hand_a, hand_b) - solution.stack_bb
                push_ev += chance[j] * (
                    (1.0 - solution.big_blind_call[j]) * bb_fold_win
                    + solution.big_blind_call[j] * called_ev
                )
            best_sb_ev += chance[i] * max(0.0, push_ev - fold_ev)

        best_bb_ev_for_sb = fold_ev * (1.0 - total_push_mass)
        for j, hand_b in enumerate(HAND_CLASSES):
            fold_response = total_push_mass * bb_fold_win
            call_response = sum(
                push_mass[i]
                * (2 * solution.stack_bb * matrix.equity(hand_a, hand_b) - solution.stack_bb)
                for i, hand_a in enumerate(HAND_CLASSES)
            )
            best_bb_ev_for_sb += chance[j] * min(fold_response, call_response)

        self.assertAlmostEqual(
            solution.nash_gap_bb, best_sb_ev - best_bb_ev_for_sb, places=10
        )

    def test_10bb_no_ante_range_mass_matches_published_hu_chart(self):
        matrix = PreflopMatrix.load()
        solution = solve_push_fold_nash(matrix, stack_bb=10)
        with_ante = solve_push_fold_nash(matrix, stack_bb=10, ante_bb=0.1)
        total_combos = sum(combo_count(hand) for hand in HAND_CLASSES)
        shove_mass = sum(
            combo_count(hand) * frequency
            for hand, frequency in zip(HAND_CLASSES, solution.small_blind_push)
        ) / total_combos
        call_mass = sum(
            combo_count(hand) * frequency
            for hand, frequency in zip(HAND_CLASSES, solution.big_blind_call)
        ) / total_combos

        # Reference chart: 58.4% SB shove, 37.3% BB call at 10bb, no ante, chip EV.
        # https://pokermathcoach.com/tools/push-fold-chart/
        self.assertTrue(solution.converged)
        self.assertLessEqual(solution.nash_gap_bb, 0.01)
        self.assertAlmostEqual(shove_mass, 0.584, delta=0.01)
        self.assertAlmostEqual(call_mass, 0.373, delta=0.01)
        self.assertTrue(with_ante.converged)
        ante_shove_mass = sum(
            combo_count(hand) * frequency
            for hand, frequency in zip(HAND_CLASSES, with_ante.small_blind_push)
        ) / total_combos
        ante_call_mass = sum(
            combo_count(hand) * frequency
            for hand, frequency in zip(HAND_CLASSES, with_ante.big_blind_call)
        ) / total_combos
        self.assertGreater(ante_shove_mass, shove_mass)
        self.assertGreater(ante_call_mass, call_mass)

    def test_serialized_training_labels_include_both_actions_for_all_classes(self):
        solution = solve_push_fold_nash(
            self._matrix(), stack_bb=10, max_iterations=500, tolerance_bb=0.01
        )
        payload = solution.to_dict()

        self.assertEqual(payload["format"], "lugon_hu_push_fold_nash_v1")
        self.assertEqual(len(payload["strategies"]), 169)
        self.assertEqual(payload["strategies"][0]["hand_class"], HAND_CLASSES[0])
        for row in payload["strategies"]:
            self.assertAlmostEqual(row["sb_push"] + row["sb_fold"], 1.0)
            self.assertAlmostEqual(row["bb_call"] + row["bb_fold"], 1.0)

    def test_rejects_invalid_stack_and_iteration_settings(self):
        matrix = self._matrix()
        with self.assertRaises(ValueError):
            solve_push_fold_nash(matrix, stack_bb=1.0)
        with self.assertRaises(ValueError):
            solve_push_fold_nash(matrix, stack_bb=2.0, ante_bb=1.0)
        with self.assertRaises(ValueError):
            solve_push_fold_nash(matrix, stack_bb=10.0, max_iterations=0)


if __name__ == "__main__":
    unittest.main()
