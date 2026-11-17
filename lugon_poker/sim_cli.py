"""CLI for Milestone 4: Interactive Poker Simulator and LLM/Baseline strategic assistant benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from .assistant import AssistantRecommendation, DecisionContext, GameAssistant
from .persistence import ProfileStore
from .simulator import TableSimulator


def _interactive_selector(
    context: DecisionContext,
    recommendation: AssistantRecommendation,
    baselines: dict[str, tuple[str, float]],
) -> tuple[str, float]:
    print(f"\n--- Decisão Hero ({context.hero_nickname} @ {context.hero_position}) | Street: {context.street.upper()} ---")
    print(f"Cartas Hero: {' '.join(context.hero_cards)} | Board: {' '.join(context.board) or '—'}")
    print(
        f"Pote: {context.pot:.2f} | A pagar: {context.to_call:.2f} | "
        f"Stack Hero: {context.hero_stack:.2f} | SPR: {context.spr if context.spr is not None else '—'}"
    )
    eq = context.equity
    po = context.math_strategy.get("pot_odds")
    cev = context.math_strategy.get("call_ev")
    po_txt = f"{po:.1%}" if po is not None else "—"
    cev_txt = f"{cev:+.2f}" if cev is not None else "—"
    print(
        f"Motor Matemático: Equity={eq['share']:.1%} (W={eq['win_rate']:.1%}, T={eq['tie_rate']:.1%}) | "
        f"Pot Odds={po_txt} | EV Call={cev_txt}"
    )
    print(
        f"Assistente [{recommendation.provider}:{recommendation.model} | {recommendation.latency_ms:.1f}ms]: "
        f"-> {recommendation.recommended_action.upper()} ({recommendation.recommended_amount:.2f})"
    )
    print(f"  Justificação: {recommendation.reasoning}")
    b_str = ", ".join(f"{k}={v[0]}({v[1]:.1f})" for k, v in baselines.items())
    print(f"  Baselines:    {b_str}")
    legal_str = ", ".join(
        f"{item.action}"
        + (f"[{item.min_amount:.1f}..{item.max_amount:.1f}]" if item.max_amount > 0 else "")
        for item in context.legal_actions
    )
    print(f"  Ações legais: {legal_str}")
    try:
        raw = input("Escolhe ação (Enter para aceitar recomendação, ou ex: 'call', 'fold', 'raise 12'): ").strip()
    except EOFError:
        raw = ""
    if not raw:
        return recommendation.recommended_action, recommendation.recommended_amount
    parts = raw.split()
    act = parts[0].lower()
    amt = float(parts[1]) if len(parts) > 1 else recommendation.recommended_amount
    return act, amt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lugon Poker Lab — Simulador Interativo de Mesa Única e Assistente Estratégico (Baseline / LLM)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    def _add_common_args(sub: argparse.ArgumentParser) -> None:
        sub.add_argument(
            "--provider",
            choices=("heuristic", "ollama", "openai_compatible", "gemini"),
            default="heuristic",
            help="fornecedor de assistência estratégica (default: heuristic)",
        )
        sub.add_argument("--model", type=str, default=None, help="nome do modelo LLM (ex: llama3.1, gpt-4o-mini, gemini-2.5-flash)")
        sub.add_argument("--endpoint", type=str, default=None, help="URL base opcional para Ollama ou servidor OpenAI-compatible local/cloud")
        sub.add_argument("--db", type=Path, default=None, help="base SQLite opcional para persistir mãos e perfis entre sessões")
        sub.add_argument("--iterations", type=int, default=1_500, help="amostras Monte Carlo por decisão (default: 1500)")
        sub.add_argument("--seed", type=int, default=7, help="seed para reprodutibilidade")
        sub.add_argument("--no-vision", action="store_true", help="desativa renderização e leitura visual PNG em cada decisão")
        sub.add_argument("--json", action="store_true", help="imprime resultado completo em JSON")

    play_parser = subparsers.add_parser("play", help="simula uma mão completa (modo automático ou interativo)")
    _add_common_args(play_parser)
    play_parser.add_argument("--interactive", action="store_true", help="pede decisão ao utilizador no terminal após mostrar a recomendação")
    play_parser.add_argument("--show-prompt", action="store_true", help="inclui/imprime o prompt estruturado enviado ao LLM")

    eval_parser = subparsers.add_parser("evaluate", help="simula várias mãos, compara com baselines e mede latência")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--hands", type=int, default=10, help="número de mãos a simular (default: 10)")

    args = parser.parse_args(argv)
    try:
        assistant = GameAssistant(
            provider=args.provider,
            model=args.model,
            endpoint=args.endpoint,
        )
        store_target: str | Path = args.db if args.db else ":memory:"
        with ProfileStore(store_target) as store:
            sim = TableSimulator(
                equity_iterations=args.iterations,
                use_vision=not args.no_vision,
                assistant=assistant,
                profile_store=store,
            )

            if args.command == "play":
                selector = _interactive_selector if args.interactive else None
                result = sim.simulate_hand(hand_id="SIM_PLAY_001", seed=args.seed, hero_action_selector=selector)
                payload: dict[str, Any] = result.to_dict()
                if args.show_prompt and result.hero_decisions:
                    payload["llm_prompt_preview"] = result.hero_decisions[0].llm_prompt_preview
                if args.json:
                    print(json.dumps(payload, ensure_ascii=False, indent=2))
                else:
                    print(
                        f"Mão {result.hand_id} concluída | Board final: {' '.join(str(c) for c in result.history.board) or '—'} | "
                        f"Pote: {result.final_pot:.2f} | Vencedor(es): {', '.join(result.winners)}"
                    )
                    print(f"Resultado líquido (fichas): {result.net_chips}")
                    for idx, dec in enumerate(result.hero_decisions, start=1):
                        rec = dec.assistant_recommendation
                        print(
                            f"  Decisão {idx} ({dec.street}): Hero escolheu {dec.chosen_action} ({dec.chosen_amount:.2f}) | "
                            f"Recomendação [{rec.provider}:{rec.model}]={rec.recommended_action} ({rec.recommended_amount:.2f}) | "
                            f"Equity={dec.equity:.1%} | Visão OK={dec.vision_verified} | "
                            f"Latência total={dec.total_latency_ms:.1f}ms (visão={dec.vision_latency_ms:.1f}ms, "
                            f"equity={dec.equity_latency_ms:.1f}ms, assistente={dec.assistant_latency_ms:.1f}ms)"
                        )
                        print(f"    Explicação: {rec.reasoning}")
                    if args.show_prompt and result.hero_decisions:
                        print("\n--- Prompt LLM Gerado (1ª decisão) ---")
                        print(result.hero_decisions[0].llm_prompt_preview["system"])
                        print(result.hero_decisions[0].llm_prompt_preview["user"])
                return 0

            if args.command == "evaluate":
                summary = sim.evaluate_session(hands=args.hands, seed=args.seed)
                if args.json:
                    print(json.dumps(summary, ensure_ascii=False, indent=2))
                else:
                    print(
                        f"Sessão simulada: {summary['hands_simulated']} mãos | "
                        f"decisões do Hero: {summary['hero_decisions_count']} | "
                        f"assistente: {summary['assistant_provider']} ({summary['assistant_model']})"
                    )
                    v_rate = summary["vision_verification_rate"]
                    print(f"Verificação visual (Vision -> HandState): {v_rate:.1%}" if v_rate is not None else "Verificação visual: n/d")
                    agree = summary["baseline_agreement_rate"]
                    print(
                        "Concordância com baselines: "
                        + ", ".join(
                            f"{k}={v:.1%}" if v is not None else f"{k}=n/d"
                            for k, v in agree.items()
                        )
                    )
                    print(f"Saldo acumulado (fichas): {summary['cumulative_net_chips']}")
                    lat = summary["latency_summary"]
                    for stage in ("vision", "equity", "assistant", "total"):
                        st = lat[stage]
                        if st["mean_ms"] is not None:
                            print(
                                f"  Latência {stage:9s}: média={st['mean_ms']:.2f}ms | "
                                f"mediana={st['median_ms']:.2f}ms | máx={st['max_ms']:.2f}ms"
                            )
                return 0

        return 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
