"""Malmuth-Harville ICM checks."""

import unittest

from lugon_poker.icm import icm_equity


class IcmTests(unittest.TestCase):
    def test_equal_stacks_split_total_prize_equally(self):
        equity = icm_equity((10, 10, 10), (0.5, 0.3, 0.2))
        for value in equity:
            self.assertAlmostEqual(value, 1 / 3)

    def test_winner_take_all_equity_is_proportional_to_chips(self):
        equity = icm_equity((50, 30, 20), (1, 0, 0))
        self.assertEqual(equity, (0.5, 0.3, 0.2))

    def test_busted_players_split_remaining_finish_orders(self):
        equity = icm_equity((10, 0, 0), (0.5, 0.3, 0.2))
        self.assertEqual(equity, (0.5, 0.25, 0.25))

    def test_invalid_player_count_and_payout_order_are_rejected(self):
        with self.assertRaises(ValueError):
            icm_equity((10, 10), (1, 0, 0))
        with self.assertRaises(ValueError):
            icm_equity((10, 10, 10), (0.2, 0.5, 0.3))


if __name__ == "__main__":
    unittest.main()
