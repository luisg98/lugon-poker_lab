import json
import unittest
import urllib.error

from lugon_poker.assistant import (
    DecisionContext,
    GameAssistant,
    LegalAction,
    build_assistant_prompt,
    validate_and_clamp_action,
)
from lugon_poker.simulator import TableSimulator


class AssistantFlowTests(unittest.TestCase):
    def _sample_context(self) -> DecisionContext:
        return DecisionContext(
            hand_id="TEST_H1",
            street="flop",
            hero_nickname="HERO",
            hero_position="BTN",
            hero_cards=("As", "Ah"),
            board=("Ks", "7d", "2c"),
            hero_hand_description="one pair",
            pot=25.0,
            to_call=4.0,
            hero_stack=96.0,
            effective_stack=86.0,
            spr=3.44,
            legal_actions=(
                LegalAction("fold", 0.0, 0.0),
                LegalAction("call", 4.0, 4.0),
                LegalAction("raise", 6.0, 96.0),
            ),
            equity={
                "share": 0.78,
                "win_rate": 0.77,
                "tie_rate": 0.02,
                "loss_rate": 0.21,
                "samples": 1000,
                "method": "monte_carlo",
            },
            math_strategy={
                "decision": "call",
                "equity": 0.78,
                "pot_odds": 4.0 / 29.0,
                "call_ev": 0.78 * 29.0 - 4.0,
                "fold_ev": 0.0,
                "explanation": "Call tem EV positivo.",
            },
            recent_actions=({"street": "flop", "player": "VILLAIN", "action": "bet", "amount": 4.0},),
            opponent_profiles={},
        )

    def test_prompt_contains_grounded_math_and_legal_actions(self):
        ctx = self._sample_context()
        prompts = build_assistant_prompt(ctx)
        self.assertIn("NÃO inventes probabilidades", prompts["system"])
        user_payload = json.loads(prompts["user"])
        self.assertAlmostEqual(user_payload["equity"]["share"], 0.78)
        self.assertEqual(len(user_payload["legal_actions"]), 3)

    def test_ollama_provider_parses_structured_llm_json(self):
        ctx = self._sample_context()
        captured_requests = []

        def fake_transport(url, headers, payload, timeout):
            captured_requests.append((url, payload))
            llm_reply = {
                "recommended_action": "raise",
                "recommended_amount": 18.0,
                "alternatives": [{"action": "call", "amount": 4.0, "pros": "Mantém bluffs", "cons": "Menor valor"}],
                "reasoning": "Overpair forte com 78% de equity num board seco; subir para 18 extrai valor.",
                "risks": ["Enfrentar 3-bet no flop contra sets."],
                "assumptions": ["Adversário paga com top pair."],
            }
            return {"message": {"content": json.dumps(llm_reply, ensure_ascii=False)}}

        assistant = GameAssistant(
            provider="ollama",
            model="llama3.1",
            endpoint="http://127.0.0.1:11434",
            transport=fake_transport,
        )
        rec = assistant.recommend(ctx)
        self.assertEqual(len(captured_requests), 1)
        self.assertEqual(captured_requests[0][0], "http://127.0.0.1:11434/api/chat")
        self.assertEqual(rec.provider, "ollama")
        self.assertEqual(rec.recommended_action, "raise")
        self.assertEqual(rec.recommended_amount, 18.0)
        self.assertTrue(rec.is_legal)
        self.assertFalse(rec.fallback_used)

    def test_openai_compatible_and_gemini_providers_and_illegal_action_clamping(self):
        ctx = self._sample_context()

        def fake_openai_transport(url, headers, payload, timeout):
            # Intentionally return an oversized raise (500 > max 96) to verify clamping.
            llm_reply = {
                "recommended_action": "raise",
                "recommended_amount": 500.0,
                "reasoning": "All-in por valor.",
            }
            return {"choices": [{"message": {"content": f"```json\n{json.dumps(llm_reply)}\n```"}}]}

        assistant = GameAssistant(
            provider="openai_compatible",
            model="local-qwen",
            endpoint="http://127.0.0.1:1234/v1",
            api_key="test-key",
            transport=fake_openai_transport,
        )
        rec = assistant.recommend(ctx)
        self.assertEqual(rec.recommended_action, "raise")
        self.assertEqual(rec.recommended_amount, 96.0)
        self.assertFalse(rec.is_legal)

        # Verify completely illegal action (e.g. 'check' when facing a bet) falls back to a legal action.
        act, amt, is_legal = validate_and_clamp_action("check", 0.0, ctx.legal_actions)
        self.assertEqual((act, amt, is_legal), ("fold", 0.0, False))

    def test_unreachable_llm_endpoint_falls_back_cleanly_to_local_baseline(self):
        ctx = self._sample_context()

        def failing_transport(url, headers, payload, timeout):
            raise urllib.error.URLError("Connection refused")

        assistant = GameAssistant(provider="ollama", transport=failing_transport)
        rec = assistant.recommend(ctx, fallback_on_error=True)
        self.assertTrue(rec.fallback_used)
        self.assertIn("Connection refused", rec.error)
        self.assertIn(rec.recommended_action, {"fold", "call", "raise"})
        self.assertTrue(rec.is_legal)

    def test_ollama_empty_content_recovers_from_thinking_or_fails_cleanly(self):
        ctx = self._sample_context()

        def thinking_only_transport(url, headers, payload, timeout):
            reply = {"recommended_action": "call", "recommended_amount": 4.0, "reasoning": "OK."}
            return {"message": {"content": "", "thinking": f"Analisando... {json.dumps(reply)}"}}

        assistant = GameAssistant(provider="ollama", model="gpt-oss:20b", transport=thinking_only_transport)
        rec = assistant.recommend(ctx, fallback_on_error=True)
        self.assertFalse(rec.fallback_used)
        self.assertEqual(rec.recommended_action, "call")

        def truly_empty_transport(url, headers, payload, timeout):
            return {"message": {"content": ""}}

        empty_assistant = GameAssistant(provider="ollama", model="gpt-oss:20b", transport=truly_empty_transport)
        empty_rec = empty_assistant.recommend(ctx, fallback_on_error=True)
        self.assertTrue(empty_rec.fallback_used)
        self.assertIn("empty response content", empty_rec.error)


class TableSimulatorTests(unittest.TestCase):
    def test_single_hand_simulation_conserves_chips_and_verifies_vision(self):
        with TableSimulator(equity_iterations=400, use_vision=True) as sim:
            result = sim.simulate_hand(hand_id="SIM_TEST_1", seed=42)
            self.assertAlmostEqual(sum(result.net_chips.values()), 0.0, places=4)
            self.assertGreater(result.final_pot, 0.0)
            self.assertEqual(sim.profile_store.hand_count(), 1)
            for dec in result.hero_decisions:
                self.assertTrue(dec.vision_verified)
                self.assertGreater(dec.total_latency_ms, 0.0)
                self.assertEqual(set(dec.baseline_decisions.keys()), {"pot_odds", "tight_aggressive", "calling_station"})

    def test_multi_hand_evaluation_session_updates_profiles_and_measures_latency(self):
        with TableSimulator(equity_iterations=300, use_vision=True) as sim:
            summary = sim.evaluate_session(hands=5, seed=19)
            self.assertEqual(summary["hands_simulated"], 5)
            self.assertGreater(summary["hero_decisions_count"], 0)
            self.assertEqual(summary["vision_verification_rate"], 1.0)
            self.assertAlmostEqual(sum(summary["cumulative_net_chips"].values()), 0.0, places=4)
            self.assertIsNotNone(summary["latency_summary"]["total"]["mean_ms"])
            self.assertEqual(set(summary["profiles_snapshot"].keys()), {"HERO", "VILLAIN", "CALLER"})
            self.assertEqual(summary["profiles_snapshot"]["VILLAIN"]["hands_observed"], 5)

    def test_close_only_closes_an_internally_owned_profile_store(self):
        from lugon_poker.persistence import ProfileStore

        store = ProfileStore(":memory:")
        simulator = TableSimulator(profile_store=store)
        simulator.close()
        self.assertEqual(store.hand_count(), 0)
        store.close()


if __name__ == "__main__":
    unittest.main()
