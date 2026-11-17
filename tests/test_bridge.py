import json
from pathlib import Path
import tempfile
import unittest

from lugon_poker.agent_bridge import (
    AgentBridge,
    build_decision_context_from_inputs,
    discover_local_models,
)
from lugon_poker.companion_gui import (
    _pick_strategy_model,
    evaluate_vision_report,
)
from lugon_poker.synthetic import _filled_rect, render_scene
from lugon_poker.vision import analyze_image, load_scene


SCENE_PATH = Path(__file__).parents[1] / "examples" / "vision" / "flop_table.json"


class AgentBridgeTests(unittest.TestCase):
    def test_strategy_model_picker_prioritizes_gpt_oss(self):
        installed = [
            "hf.co/duarteocarmo/AMALIA-9B-0626-DPO-GGUF:Q4_K_M",
            "llama3.1:8b-instruct-q4_K_M",
            "qwen2.5:7b-instruct",
            "gpt-oss:20b",
        ]
        self.assertEqual(_pick_strategy_model(installed), "gpt-oss:20b")
        self.assertEqual(_pick_strategy_model(["llama3.1:8b", "qwen2.5:7b"]), "qwen2.5:7b")
        self.assertEqual(_pick_strategy_model([]), "gpt-oss:20b")

    def test_evaluate_vision_report_handles_ready_waiting_and_failed(self):
        scene = load_scene(SCENE_PATH)
        ready_report = analyze_image(render_scene(scene))
        kind, payload = evaluate_vision_report(ready_report)
        self.assertEqual(kind, "ready")
        self.assertEqual(payload["hero_cards"], ["As", "Ah"])
        self.assertEqual(payload["board"], ["Ks", "7d", "2c"])
        self.assertEqual(payload["pot"], 25.0)
        self.assertEqual(payload["to_call"], 4.0)

        # Turn active player into an opponent -> waiting state
        waiting_scene = json.loads(json.dumps(scene))
        for p in waiting_scene["players"]:
            p["active"] = p["nickname"] == "VILLAIN"
        waiting_report = analyze_image(render_scene(waiting_scene))
        kind, payload = evaluate_vision_report(waiting_report)
        self.assertEqual(kind, "waiting")
        self.assertEqual(payload["active_player"], "VILLAIN")

        # Corrupt a board card -> failed (LEITURA FALHADA)
        broken_img = render_scene(scene)
        _filled_rect(broken_img, 393, 230, 36, 50, (18, 100, 75))
        failed_report = analyze_image(broken_img)
        kind, payload = evaluate_vision_report(failed_report)
        self.assertEqual(kind, "failed")
        self.assertTrue(len(payload["reasons"]) >= 1)

    def test_publish_and_reply_round_trip_for_antigravity_bridge(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            bridge = AgentBridge(Path(tmp_dir))
            ctx = build_decision_context_from_inputs(
                hero_cards=["As", "Ah"],
                board=["Ks", "7d", "2c"],
                pot=25.0,
                to_call=4.0,
                hero_stack=96.0,
                opponents=[{"nickname": "VILLAIN", "range": "KK,AKs,KQs"}],
                equity_iterations=400,
                seed=7,
            )
            req = bridge.publish_request(ctx, agent_target="antigravity")
            pending = bridge.read_pending_request()
            self.assertIsNotNone(pending)
            self.assertEqual(pending["request_id"], req.request_id)
            self.assertNotIn("user_message", pending)
            self.assertNotIn("PERGUNTA ADICIONAL", pending["prompts"]["user"])

            resp = bridge.write_agent_response(
                request_id=req.request_id,
                recommended_action="raise",
                recommended_amount=14.0,
                reasoning="Contra uma range condensada em top pair e overpairs, o raise extrai valor imediato.",
                agent_name="Antigravity",
                risks=["Isolar contra KK."],
                assumptions=["Vilão paga com AK/KQ."],
            )
            latest = bridge.read_latest_response(expected_request_id=req.request_id)
            self.assertIsNotNone(latest)
            self.assertEqual(latest["agent_name"], "Antigravity")
            self.assertEqual(latest["recommended_action"], "raise")
            self.assertEqual(latest["recommended_amount"], 14.0)
            self.assertEqual(resp.request_id, req.request_id)

    def test_discover_local_models_and_query_local_gpt_oss(self):
        def fake_get(url: str, timeout: float):
            if "11434" in url:
                return {"models": [{"name": "gpt-oss:20b"}, {"name": "llama3.1:8b"}]}
            return {"data": [{"id": "gpt-oss-20b-gguf"}]}

        models = discover_local_models(http_get=fake_get)
        self.assertTrue(models["ollama_online"])
        self.assertIn("gpt-oss:20b", models["ollama_models"])
        self.assertTrue(models["openai_local_online"])
        self.assertIn("gpt-oss-20b-gguf", models["openai_local_models"])

        with tempfile.TemporaryDirectory() as tmp_dir:
            bridge = AgentBridge(Path(tmp_dir))
            ctx = build_decision_context_from_inputs(
                hero_cards=["Qs", "Qh"],
                board=["Jh", "8c", "3d"],
                pot=18.0,
                to_call=6.0,
                equity_iterations=300,
            )

            def fake_post(url, headers, payload, timeout):
                self.assertEqual(payload["model"], "gpt-oss:20b")
                reply = {
                    "recommended_action": "call",
                    "recommended_amount": 6.0,
                    "reasoning": "QQ tem equity claramente acima das pot odds.",
                }
                return {"message": {"content": json.dumps(reply)}}

            rec = bridge.query_local_gpt_oss(
                ctx,
                provider="ollama",
                model="gpt-oss:20b",
                transport=fake_post,
            )
            self.assertEqual(rec.model, "gpt-oss:20b")
            self.assertEqual(rec.recommended_action, "call")
            self.assertEqual(rec.recommended_amount, 6.0)
            self.assertFalse(rec.fallback_used)


if __name__ == "__main__":
    unittest.main()
