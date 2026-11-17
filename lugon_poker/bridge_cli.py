"""CLI for inspecting the live Agent Bridge (Antigravity <-> Companion App) and probing local GPT-OSS models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .agent_bridge import (
    DEFAULT_BRIDGE_DIR,
    AgentBridge,
    build_decision_context_from_inputs,
    discover_local_models,
)
from .persistence import ProfileStore


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lugon Poker Lab — Ponte de comunicação entre a App Companion e o Agente (Antigravity / Local GPT-OSS)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("status", help="mostra o pedido atual da app companion e deteta modelos locais")
    status_parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    status_parser.add_argument("--json", action="store_true")

    reply_parser = subparsers.add_parser("reply", help="envia uma resposta do agente (ex: Antigravity) para a janela da app")
    reply_parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    reply_parser.add_argument("--action", required=True, choices=("fold", "check", "call", "bet", "raise"))
    reply_parser.add_argument("--amount", type=float, default=0.0)
    reply_parser.add_argument("--reasoning", required=True, help="explicação estratégica para apresentar na janela da app")
    reply_parser.add_argument("--agent", default="Antigravity", help="identificador do agente (default: Antigravity)")
    reply_parser.add_argument("--risk", action="append", default=[], help="risco a listar (pode repetir)")
    reply_parser.add_argument("--assumption", action="append", default=[], help="pressuposto a listar (pode repetir)")

    analyze_parser = subparsers.add_parser("analyze", help="analisa um estado de mão e publica a recomendação")
    analyze_parser.add_argument("--bridge-dir", type=Path, default=DEFAULT_BRIDGE_DIR)
    analyze_parser.add_argument("--hero", nargs=2, default=["As", "Ah"], help="duas cartas do Hero (ex: As Ah)")
    analyze_parser.add_argument("--board", nargs="*", default=["Ks", "7d", "2c"], help="cartas do board")
    analyze_parser.add_argument("--pot", type=float, default=25.0)
    analyze_parser.add_argument("--to-call", type=float, default=4.0)
    analyze_parser.add_argument("--stack", type=float, default=96.0)
    analyze_parser.add_argument("--villain", type=str, default="VILLAIN")
    analyze_parser.add_argument("--range", type=str, default="*")
    analyze_parser.add_argument(
        "--target",
        choices=("antigravity", "ollama", "openai_compatible", "heuristic"),
        default="antigravity",
    )
    analyze_parser.add_argument("--model", type=str, default="gpt-oss:20b")
    analyze_parser.add_argument("--endpoint", type=str, default=None)
    analyze_parser.add_argument("--db", type=Path, default=None)
    analyze_parser.add_argument("--json", action="store_true")

    args = parser.parse_args(argv)
    try:
        bridge = AgentBridge(args.bridge_dir)

        if args.command == "status":
            pending = bridge.read_pending_request()
            latest_resp = bridge.read_latest_response()
            models = discover_local_models()
            payload = {
                "bridge_dir": str(bridge.bridge_dir),
                "pending_request": pending,
                "latest_response": latest_resp,
                "local_models": models,
            }
            if args.json:
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                print(f"Pasta da ponte: {bridge.bridge_dir}")
                print(
                    f"Modelos locais detetados — Ollama online: {models['ollama_online']} ({', '.join(models['ollama_models']) or 'nenhum'}) | "
                    f"OpenAI-local online: {models['openai_local_online']} ({', '.join(models['openai_local_models']) or 'nenhum'})"
                )
                if pending:
                    ctx = pending.get("context", {})
                    print(
                        f"Pedido ativo: {pending.get('request_id')} ({pending.get('created_at')}) | "
                        f"Alvo: {pending.get('agent_target')}"
                    )
                    print(
                        f"  Mão: Hero {' '.join(ctx.get('hero_cards', []))} | Board {' '.join(ctx.get('board', [])) or '—'} | "
                        f"Pote={ctx.get('pot')} | To Call={ctx.get('to_call')} | Equity={ctx.get('equity', {}).get('share', 0):.1%}"
                    )
                else:
                    print("Nenhum pedido pendente na ponte.")
                if latest_resp:
                    print(
                        f"Última resposta: {latest_resp.get('request_id')} por {latest_resp.get('agent_name')} -> "
                        f"{latest_resp.get('recommended_action')} ({latest_resp.get('recommended_amount')})"
                    )
            return 0

        if args.command == "reply":
            pending = bridge.read_pending_request()
            req_id = pending["request_id"] if pending and "request_id" in pending else "MANUAL_REPLY"
            resp = bridge.write_agent_response(
                request_id=req_id,
                recommended_action=args.action,
                recommended_amount=args.amount,
                reasoning=args.reasoning,
                agent_name=args.agent,
                risks=args.risk,
                assumptions=args.assumption,
            )
            print(f"Resposta gravada em {bridge.response_path} para o pedido {resp.request_id}.")
            return 0

        if args.command == "analyze":
            store = ProfileStore(args.db) if args.db else None
            try:
                ctx = build_decision_context_from_inputs(
                    hero_cards=list(args.hero),
                    board=list(args.board),
                    pot=args.pot,
                    to_call=args.to_call,
                    hero_stack=args.stack,
                    opponents=[{"nickname": args.villain, "range": args.range}],
                    profile_store=store,
                )
            finally:
                if store is not None:
                    store.close()

            if args.target == "antigravity":
                req = bridge.publish_request(ctx, agent_target="antigravity")
                if args.json:
                    print(json.dumps(req.to_dict(), ensure_ascii=False, indent=2))
                else:
                    print(f"Pedido publicado para Antigravity: {req.request_id} em {bridge.request_path}")
                    base_rec = req.baseline_recommendation
                    print(
                        f"Baseline imediata: {base_rec['recommended_action']} ({base_rec['recommended_amount']}) — "
                        f"{base_rec['reasoning']}"
                    )
                return 0

            provider = "heuristic" if args.target == "heuristic" else args.target
            rec = bridge.query_local_gpt_oss(
                ctx,
                provider=provider,
                model=args.model,
                endpoint=args.endpoint,
            )
            if args.json:
                print(json.dumps(rec.to_dict(), ensure_ascii=False, indent=2))
            else:
                print(
                    f"Recomendação [{rec.provider}:{rec.model}] (fallback={rec.fallback_used}): "
                    f"{rec.recommended_action.upper()} ({rec.recommended_amount:.2f})"
                )
                print(f"Explicação: {rec.reasoning}")
            return 0

        return 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
