"""Command line interface for importing and analyzing one hand state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .cards import best_hand, describe
from .equity import calculate_equity
from .persistence import ProfileStore, load_history_file
from .profiles import PlayerProfile
from .state import HandState
from .strategy import analyze_strategy


def _report(
    state: HandState,
    iterations: int,
    seed: int,
    profiles: dict[str, PlayerProfile] | None = None,
) -> dict[str, object]:
    equity = calculate_equity(state, iterations=iterations, seed=seed)
    analysis = analyze_strategy(state, equity)
    hero_rank = best_hand((*state.hero_cards, *state.board)) if len(state.board) >= 3 else None
    report: dict[str, object] = {
        "street": state.street,
        "hero_cards": [str(card) for card in state.hero_cards],
        "board": [str(card) for card in state.board],
        "hero_current_hand": describe(hero_rank) if hero_rank else None,
        "opponents": [{"nickname": opponent.nickname, "range": opponent.range_text or "*"} for opponent in state.opponents],
        "equity": {
            "share": equity.equity,
            "win_rate": equity.win_rate,
            "tie_rate": equity.tie_rate,
            "loss_rate": equity.loss_rate,
            "samples": equity.samples,
            "method": equity.method,
            "seed": equity.seed,
            "attempts": equity.attempts,
        },
        "strategy": analysis.to_dict(),
        "assumptions": [
            "Opponent ranges are uniformly weighted physical two-card combinations.",
            "Equity is conditional on the supplied known cards and ranges.",
            "Call EV assumes no further betting, no side pots and no rake; the current pot excludes hero's call.",
        ],
    }
    if profiles is not None:
        report["opponent_profiles"] = {
            opponent.nickname: profiles[opponent.nickname].to_dict()
            if opponent.nickname in profiles
            else None
            for opponent in state.opponents
        }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze a single Texas Hold'em hand from a JSON state.")
    parser.add_argument("state_file", type=Path, help="path to a hand-state JSON file")
    parser.add_argument("--iterations", type=int, default=10_000, help="valid Monte Carlo samples (default: 10000)")
    parser.add_argument("--seed", type=int, default=7, help="random seed for reproducible simulations")
    parser.add_argument("--db", type=Path, help="optional SQLite profile store to attach opponent statistics")
    parser.add_argument("--history", type=Path, nargs="*", help="optional hand-history JSON file(s) for opponent stats")
    parser.add_argument("--json", action="store_true", help="print the complete result as JSON")
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be positive")
    try:
        raw = json.loads(args.state_file.read_text(encoding="utf-8"))
        state = HandState.from_dict(raw)
        profiles: dict[str, PlayerProfile] | None = None
        if args.db or args.history:
            store_path: str | Path = args.db if args.db else ":memory:"
            with ProfileStore(store_path) as store:
                if args.history:
                    for file_path in args.history:
                        store.import_hands(load_history_file(file_path))
                profiles = store.all_profiles()
        report = _report(state, args.iterations, args.seed, profiles=profiles)
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0

    equity = report["equity"]
    strategy = report["strategy"]
    assert isinstance(equity, dict) and isinstance(strategy, dict)
    print(f"Lugon Poker Lab — {report['street']}")
    print(f"Hero: {' '.join(report['hero_cards'])}; board: {' '.join(report['board']) or '—'}")
    print(f"Equity: {equity['share']:.1%} (vitórias {equity['win_rate']:.1%}, empates {equity['tie_rate']:.1%})")
    print(f"Método: {equity['method']}; amostras: {equity['samples']}; seed: {equity['seed']}")
    print(f"Decisão modelo: {strategy['decision']}")
    if strategy["pot_odds"] is not None:
        print(f"Pot odds: {strategy['pot_odds']:.1%}; EV call: {strategy['call_ev']:.2f}; EV fold: {strategy['fold_ev']:.2f}")
    print(strategy["explanation"])
    print("Pressupostos: ranges uniformes; showdown sem apostas futuras, side pots ou rake.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
