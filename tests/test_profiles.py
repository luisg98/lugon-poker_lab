import copy
from pathlib import Path
import tempfile
import unittest

from lugon_poker.history import HandHistory, diff_table_captures
from lugon_poker.persistence import ProfileStore, load_history_file
from lugon_poker.profiles import RateEstimate, build_profiles, wilson_interval_95
from lugon_poker.synthetic import render_scene
from lugon_poker.vision import analyze_image, load_scene


HISTORY_PATH = Path(__file__).parents[1] / "examples" / "histories" / "sample_hands.json"
SCENE_PATH = Path(__file__).parents[1] / "examples" / "vision" / "flop_table.json"


class PlayerIntelligenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.histories = load_history_file(HISTORY_PATH)
        cls.profiles = build_profiles(cls.histories)

    def test_vpip_pfr_and_three_bet_denominators_and_walk_handling(self):
        vitor = self.profiles["Vitor"]
        self.assertEqual(vitor.hands_observed, 3)
        self.assertEqual((vitor.vpip.successes, vitor.vpip.opportunities), (2, 3))
        self.assertEqual((vitor.pfr.successes, vitor.pfr.opportunities), (2, 3))
        self.assertEqual((vitor.three_bet.successes, vitor.three_bet.opportunities), (1, 1))
        self.assertAlmostEqual(vitor.three_bet.empirical_rate, 1.0)

        # Marta had a BB walk in H003, so 3 hands observed but only 2 VPIP/PFR opportunities.
        marta = self.profiles["Marta"]
        self.assertEqual(marta.hands_observed, 3)
        self.assertEqual((marta.vpip.successes, marta.vpip.opportunities), (1, 2))
        self.assertEqual((marta.pfr.successes, marta.pfr.opportunities), (1, 2))
        self.assertEqual((marta.three_bet.successes, marta.three_bet.opportunities), (0, 0))
        self.assertIsNone(marta.three_bet.empirical_rate)

        # Lugon cold-called a 3-bet in H001 (not a 3-bet opportunity) and folded to an open in H002 (1 opportunity).
        lugon = self.profiles["Lugon"]
        self.assertEqual((lugon.vpip.successes, lugon.vpip.opportunities), (1, 3))
        self.assertEqual((lugon.pfr.successes, lugon.pfr.opportunities), (0, 3))
        self.assertEqual((lugon.three_bet.successes, lugon.three_bet.opportunities), (0, 1))
        self.assertEqual(lugon.three_bet.empirical_rate, 0.0)

    def test_postflop_aggression_and_bet_sizing_ratios(self):
        vitor = self.profiles["Vitor"]
        pf = vitor.postflop_aggression
        # Postflop Vitor: 3 bets, 0 raises, 1 call, 2 checks, 1 fold -> AFq = 3 / (3 + 1 + 1) = 0.6, AF = 3 / 1 = 3.0
        self.assertEqual((pf.bets, pf.raises, pf.calls, pf.checks, pf.folds), (3, 0, 1, 2, 1))
        self.assertAlmostEqual(pf.aggression_frequency.empirical_rate, 3 / 5)
        self.assertAlmostEqual(pf.aggression_factor, 3.0)

        # Ana had 0 calls, so aggression_factor must be None rather than fabricated.
        ana = self.profiles["Ana"]
        self.assertIsNone(ana.overall_aggression.aggression_factor)

        # Lugon made 1 turn bet of 60 into a pot of 107.
        lugon_bs = self.profiles["Lugon"].overall_bet_sizing
        self.assertEqual(lugon_bs.observations, 1)
        self.assertAlmostEqual(lugon_bs.mean_pot_ratio, 60.0 / 107.0)
        self.assertEqual(lugon_bs.medium_bets_count, 1)

    def test_positional_breakdown(self):
        vitor = self.profiles["Vitor"]
        self.assertEqual(set(vitor.by_position.keys()), {"CO", "MP", "UTG"})
        self.assertEqual(vitor.by_position["CO"].vpip.empirical_rate, 1.0)
        self.assertEqual(vitor.by_position["UTG"].vpip.empirical_rate, 0.0)

    def test_wilson_interval_and_bayesian_posterior_math(self):
        low_0, high_0 = wilson_interval_95(0, 10)
        self.assertEqual(low_0, 0.0)
        self.assertGreater(high_0, 0.20)
        self.assertLess(high_0, 0.30)

        est_small = RateEstimate.from_counts(5, 10)
        est_large = RateEstimate.from_counts(50, 100)
        self.assertAlmostEqual(est_small.empirical_rate, 0.5)
        self.assertAlmostEqual(est_small.wilson_95_low + est_small.wilson_95_high, 1.0)
        self.assertLess(
            est_large.wilson_95_high - est_large.wilson_95_low,
            est_small.wilson_95_high - est_small.wilson_95_low,
        )
        # Beta(1,1) updated with k=5, n=10 -> Beta(6, 6), mean=0.5, std=sqrt(36 / (144 * 13))
        self.assertAlmostEqual(est_small.bayesian_mean, 0.5)
        self.assertAlmostEqual(est_small.bayesian_std, (36 / (144 * 13)) ** 0.5)

    def test_small_sample_guardrail_avoids_premature_labels(self):
        vitor = self.profiles["Vitor"]
        self.assertEqual(vitor.vpip.sample_adequacy, "insufficient_sample")
        self.assertTrue(any("nenhuma classificação de perfil é atribuída" in note for note in vitor.tendency_notes))

    def test_sqlite_persistence_and_idempotent_deduplication(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "profiles.sqlite"
            with ProfileStore(db_path) as store:
                first_import = store.import_hands(self.histories)
                second_import = store.import_hands(self.histories)
                self.assertEqual(first_import, {"imported": 3, "skipped_duplicates": 0, "total_stored": 3})
                self.assertEqual(second_import, {"imported": 0, "skipped_duplicates": 3, "total_stored": 3})
                stored_vitor = store.get_profile("Vitor")
                self.assertIsNotNone(stored_vitor)
                self.assertEqual(stored_vitor.hands_observed, 3)
                self.assertEqual(stored_vitor.vpip.successes, 2)

    def test_invalid_hand_history_is_rejected(self):
        raw = self.histories[0].to_dict()
        bad_check = copy.deepcopy(raw)
        bad_check["actions"][5] = {"street": "preflop", "player": "Lugon", "action": "check", "amount": 0.0}
        with self.assertRaisesRegex(ValueError, "cannot check when facing"):
            HandHistory.from_dict(bad_check)

        bad_fold = copy.deepcopy(raw)
        bad_fold["actions"].append({"street": "turn", "player": "Bruno", "action": "check", "amount": 0.0})
        with self.assertRaisesRegex(ValueError, "cannot act after folding"):
            HandHistory.from_dict(bad_fold)


class CaptureTransitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_scene = load_scene(SCENE_PATH)

    def test_detects_single_player_call_and_turn_card(self):
        before_report = analyze_image(render_scene(self.base_scene))

        # HERO calls 4 chips on the flop (bet goes from 0 -> 4, stack 96 -> 92, pot 25 -> 29).
        after_call_scene = copy.deepcopy(self.base_scene)
        after_call_scene["pot"] = 29
        for player in after_call_scene["players"]:
            if player["nickname"] == "HERO":
                player["stack"] = 92
                player["bet"] = 4
                player["active"] = False
            elif player["nickname"] == "CALLER":
                player["active"] = True
        after_call_report = analyze_image(render_scene(after_call_scene))

        diff = diff_table_captures(before_report, after_call_report)
        self.assertTrue(diff["valid_transition"])
        self.assertEqual(diff["previous_street"], "flop")
        self.assertEqual(diff["current_street"], "flop")
        self.assertEqual(diff["pot_delta"], 4.0)
        self.assertEqual(len(diff["inferred_events"]), 1)
        self.assertEqual(diff["inferred_events"][0]["player"], "HERO")
        self.assertEqual(diff["inferred_events"][0]["type"], "call")
        self.assertEqual(diff["inferred_events"][0]["amount_added"], 4.0)
        self.assertEqual(diff["ambiguities"], [])

    def test_flags_board_mutation_inconsistency_and_multi_actor_ambiguity(self):
        before_report = analyze_image(render_scene(self.base_scene))

        mutated_scene = copy.deepcopy(self.base_scene)
        mutated_scene["board"] = ["Kd", "7d", "2c", "Jc"]  # Ks mutated to Kd
        for player in mutated_scene["players"]:
            if player["nickname"] == "HERO":
                player["stack"] = 92
                player["bet"] = 0
            if player["nickname"] == "CALLER":
                player["folded"] = True
        mutated_report = analyze_image(render_scene(mutated_scene))

        diff = diff_table_captures(before_report, mutated_report)
        self.assertFalse(diff["valid_transition"])
        self.assertTrue(any("cartas do board mudaram" in item for item in diff["inconsistencies"]))
        self.assertTrue(any("2 jogadores alteraram estado" in item for item in diff["ambiguities"]))


if __name__ == "__main__":
    unittest.main()
