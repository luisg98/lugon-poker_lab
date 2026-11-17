"""Command-line chart generator for the heads-up push/fold solver."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preflop import DEFAULT_MATRIX_PATH, HAND_CLASSES, PreflopMatrix, combo_count
from .push_fold import solve_push_fold_nash


def _combo_weighted_frequency(frequencies: tuple[float, ...]) -> float:
    total = sum(combo_count(hand) for hand in HAND_CLASSES)
    return sum(combo_count(hand) * value for hand, value in zip(HAND_CLASSES, frequencies)) / total


def _range_text(frequencies: tuple[float, ...], cutoff: float) -> str:
    return ", ".join(
        f"{hand} ({frequency:.0%})"
        for hand, frequency in zip(HAND_CLASSES, frequencies)
        if frequency >= cutoff
    ) or "(vazia)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calcula ranges heads-up de shove/fold/call em equilíbrio chip-EV."
    )
    parser.add_argument("--stack-bb", type=float, required=True, help="stack inicial igual de ambos os jogadores")
    parser.add_argument("--small-blind-bb", type=float, default=0.5)
    parser.add_argument("--big-blind-bb", type=float, default=1.0)
    parser.add_argument("--ante-bb", type=float, default=0.0, help="ante por jogador")
    parser.add_argument("--max-iterations", type=int, default=25_000)
    parser.add_argument("--tolerance-bb", type=float, default=0.01)
    parser.add_argument("--range-cutoff", type=float, default=0.5, help="frequência mínima para mostrar uma mão")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--output", type=Path, default=None, help="grava todas as frequências num JSON")
    args = parser.parse_args(argv)

    if not 0.0 <= args.range_cutoff <= 1.0:
        parser.error("--range-cutoff tem de estar entre 0 e 1")

    try:
        matrix = PreflopMatrix.load(args.matrix)
        solution = solve_push_fold_nash(
            matrix,
            stack_bb=args.stack_bb,
            small_blind_bb=args.small_blind_bb,
            big_blind_bb=args.big_blind_bb,
            ante_bb=args.ante_bb,
            max_iterations=args.max_iterations,
            tolerance_bb=args.tolerance_bb,
        )
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Heads-up chip-EV · stack {solution.stack_bb:g} BB · ante {solution.ante_bb:g} BB/jogador"
    )
    print(
        f"Nash gap: {solution.nash_gap_bb:.4f} BB · {solution.iterations:,} iterações · "
        f"{'convergido' if solution.converged else 'não convergido'}"
    )
    print(
        f"SB shove: {_combo_weighted_frequency(solution.small_blind_push):.1%} das combinações · "
        f"BB call: {_combo_weighted_frequency(solution.big_blind_call):.1%}"
    )
    print(f"SB shove >= {args.range_cutoff:.0%}: {_range_text(solution.small_blind_push, args.range_cutoff)}")
    print(f"BB call >= {args.range_cutoff:.0%}: {_range_text(solution.big_blind_call, args.range_cutoff)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(solution.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Labels completos gravados em {args.output}")

    return 0 if solution.converged else 2


if __name__ == "__main__":
    raise SystemExit(main())
