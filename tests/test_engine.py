import unittest

from lugon_poker.cards import Card, best_hand, describe, evaluate
from lugon_poker.equity import calculate_equity
from lugon_poker.ranges import parse_range
from lugon_poker.state import HandState
from lugon_poker.strategy import analyze_strategy


def state(data):
    return HandState.from_dict(data)


class CardEvaluationTests(unittest.TestCase):
    def cards(self, text):
        return [Card.parse(token) for token in text.split()]

    def test_royal_flush_and_straight_flush(self):
        self.assertEqual(evaluate(self.cards("As Ks Qs Js Ts")), (8, 14))
        self.assertEqual(describe(evaluate(self.cards("5c 4c 3c 2c Ac"))), "straight flush")

    def test_wheel_straight_is_five_high(self):
        self.assertEqual(evaluate(self.cards("Ac 2d 3h 4s 5c")), (4, 5))

    def test_full_house_uses_highest_trip_and_pair(self):
        self.assertEqual(evaluate(self.cards("Ah Ad Ac Kh Kd Ks 2c")), (6, 14, 13))

    def test_best_of_seven_uses_best_five(self):
        self.assertEqual(best_hand(self.cards("As Ah Kc Kd Qs 2c 3d")), (2, 14, 13, 12))

    def test_best_hand_matches_brute_force_over_all_five_card_subsets(self):
        """`best_hand` reads 6-7 cards in one pass instead of ranking all 21 subsets (~17x faster).

        This pins that shortcut against an independent brute-force reference, so a future change to
        `evaluate`'s category ordering cannot silently corrupt every equity number in the project.
        """
        import random
        from itertools import combinations

        def brute_force(hand):
            return max(evaluate(subset) for subset in combinations(hand, 5))

        edge_cases = (
            "As 2s 3s 4s 5s Kh Qd",  # wheel straight flush alongside higher cards
            "As Ks Qs Js Ts 9s 8s",  # seven to a straight flush
            "6c 7c 8c 9c Tc Jc 2d",  # six-card straight flush
            "2c 3d 4h 5s Ah Kd Qc",  # wheel using the ace low
            "7c 7d 7h 8s 8d 8h 2c",  # two sets competing for the full house
            "9c 9d 9h 9s 2c 2d 2h",  # quads over trips
            "Ac Kc Qc Jc 9c 8h 8d",  # flush must beat the pair
            "2c 2d 3c 3d 4c 4d 5h",  # three pairs, only the top two count
            "Ah Ad Kc Kd Qs Qh 2c",  # three pairs with a kicker decision
        )
        for text in edge_cases:
            hand = self.cards(text)
            self.assertEqual(best_hand(hand), brute_force(hand), text)

        deck = [Card(rank, suit) for rank in range(2, 15) for suit in "cdhs"]
        rng = random.Random(20261116)
        for size in (6, 7):
            for _ in range(1500):
                hand = rng.sample(deck, size)
                self.assertEqual(best_hand(hand), brute_force(hand), " ".join(str(c) for c in hand))

    def test_duplicate_cards_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evaluate(self.cards("As As Kc Qd Jh"))

    def test_card_parser_rejects_non_string_values_cleanly(self):
        with self.assertRaisesRegex(ValueError, "must be a string"):
            Card.parse(14)


class RangeTests(unittest.TestCase):
    def test_class_expansion_counts(self):
        self.assertEqual(len(parse_range("AA")), 6)
        self.assertEqual(len(parse_range("AKs")), 4)
        self.assertEqual(len(parse_range("AKo")), 12)
        self.assertEqual(len(parse_range("QQ+")), 18)
        self.assertEqual(len(parse_range("AJs+")), 12)

    def test_exact_combo_and_union(self):
        self.assertEqual(len(parse_range("AsKd")), 1)
        self.assertEqual(len(parse_range("AA,AA")), 6)


class StateAndEquityTests(unittest.TestCase):
    def test_state_infers_street_and_rejects_duplicate(self):
        hand = state({"hero": {"cards": ["As", "Ah"]}, "board": ["Ks", "7d", "2c"], "pot": 20, "to_call": 5, "opponents": [{"nickname": "V", "range": "*"}]})
        self.assertEqual(hand.street, "flop")
        with self.assertRaisesRegex(ValueError, "share a card"):
            state({"hero": {"cards": ["As", "Ah"]}, "board": ["As", "7d", "2c"], "pot": 20, "opponents": [{"range": "*"}]})

    def test_rejects_incomplete_flop_and_negative_chips(self):
        with self.assertRaisesRegex(ValueError, "0, 3, 4, or 5"):
            state({"hero": {"cards": ["As", "Ah"]}, "board": ["Ks", "7d"], "pot": 20, "opponents": [{"range": "*"}]})
        with self.assertRaisesRegex(ValueError, "finite, non-negative"):
            state({"hero": {"cards": ["As", "Ah"]}, "pot": -1, "opponents": [{"range": "*"}]})

    def test_exact_river_equity(self):
        hand = state({
            "hero": {"cards": ["As", "Ah"]},
            "board": ["2c", "3d", "4h", "5s", "9c"],
            "pot": 10,
            "to_call": 2,
            "opponents": [{"nickname": "V", "range": "KcKd"}],
        })
        result = calculate_equity(hand)
        self.assertEqual(result.method, "exact_river")
        self.assertEqual(result.equity, 1.0)
        self.assertEqual(result.samples, 1)
        self.assertAlmostEqual(analyze_strategy(hand, result).call_ev, 10.0)

    def test_exact_river_board_play_splits_the_pot(self):
        hand = state({
            "hero": {"cards": ["2c", "3d"]},
            "board": ["As", "Ks", "Qs", "Js", "Ts"],
            "pot": 8,
            "to_call": 2,
            "opponents": [{"range": "4c5d"}],
        })
        result = calculate_equity(hand)
        self.assertEqual(result.equity, 0.5)
        self.assertEqual(result.tie_rate, 1.0)

    def test_free_action_is_check_without_fabricated_check_ev(self):
        hand = state({"hero": {"cards": ["As", "Ah"]}, "board": ["Ks", "7d", "2c"], "pot": 20, "opponents": [{"range": "KK"}]})
        from lugon_poker.equity import EquityResult
        analysis = analyze_strategy(hand, EquityResult(0.5, 0.5, 0, 0.5, 100, "test", 1, 100))
        self.assertEqual(analysis.decision, "check")
        self.assertIsNone(analysis.call_ev)

    def test_monte_carlo_is_reproducible(self):
        hand = state({"hero": {"cards": ["As", "Ah"]}, "pot": 4, "to_call": 2, "opponents": [{"range": "KK"}]})
        first = calculate_equity(hand, iterations=400, seed=31)
        second = calculate_equity(hand, iterations=400, seed=31)
        self.assertEqual(first, second)
        self.assertEqual(first.samples, 400)

    def test_pot_odds_and_call_ev(self):
        hand = state({"hero": {"cards": ["As", "Ah"]}, "board": ["Ks", "7d", "2c"], "pot": 20, "to_call": 10, "opponents": [{"range": "KcKd"}]})
        from lugon_poker.equity import EquityResult
        equity = EquityResult(0.5, 0.5, 0.0, 0.5, 100, "test", 1, 100)
        analysis = analyze_strategy(hand, equity)
        self.assertAlmostEqual(analysis.pot_odds, 1 / 3)
        self.assertAlmostEqual(analysis.call_ev, 5.0)
        self.assertEqual(analysis.decision, "call")


if __name__ == "__main__":
    unittest.main()
