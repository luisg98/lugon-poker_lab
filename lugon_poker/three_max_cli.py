"""Command-line generator for sampled 3-max ICM push/fold charts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .preflop import HAND_CLASSES, combo_count
from .three_max import (
    ThreeMaxSolution,
    solve_three_max_push_fold,
    solve_three_max_push_fold_stratified,
)


def _triple(text: str, *, name: str) -> tuple[float, float, float]:
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{name} tem de conter três números separados por vírgulas") from exc
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"{name} tem de conter exatamente três números")
    return values  # type: ignore[return-value]


def _frequency(solution: ThreeMaxSolution, strategy: str) -> float:
    values = solution.strategy(strategy)
    total = sum(combo_count(hand) for hand in HAND_CLASSES)
    return sum(combo_count(hand) * rate for hand, rate in zip(HAND_CLASSES, values)) / total


def _range(solution: ThreeMaxSolution, strategy: str, cutoff: float) -> str:
    selected = (
        f"{hand} ({rate:.0%})"
        for hand, rate in zip(HAND_CLASSES, solution.strategy(strategy))
        if rate >= cutoff
    )
    return ", ".join(selected) or "(vazia)"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calcula charts 3-max push/fold com utilidades ICM Malmuth-Harville."
    )
    parser.add_argument("--stacks", type=lambda value: _triple(value, name="--stacks"), default=(10.0, 10.0, 10.0), help="stacks BTN,SB,BB em BB; default: 10,10,10")
    parser.add_argument("--payouts", type=lambda value: _triple(value, name="--payouts"), required=True, help="prémios do 1.º, 2.º e 3.º; ex.: 1,0,0")
    parser.add_argument("--small-blind-bb", type=float, default=0.5)
    parser.add_argument("--big-blind-bb", type=float, default=1.0)
    parser.add_argument("--ante-bb", type=float, default=0.0, help="ante por jogador")
    parser.add_argument("--sampling-mode", choices=("fixed", "stratified"), default="fixed")
    parser.add_argument("--samples", type=int, default=20_000, help="deals físicos aleatórios no modo fixed")
    parser.add_argument("--samples-per-hand-class", type=int, default=100, help="deals por lugar/classe no modo stratified")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument("--tolerance", type=float, default=0.005, help="NashConv máximo em unidades do prémio")
    parser.add_argument("--range-cutoff", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=None, help="grava todas as frequências num JSON")
    args = parser.parse_args(argv)
    if not 0.0 <= args.range_cutoff <= 1.0:
        parser.error("--range-cutoff tem de estar entre 0 e 1")

    try:
        solver = (
            solve_three_max_push_fold_stratified
            if args.sampling_mode == "stratified"
            else solve_three_max_push_fold
        )
        common = {
            "small_blind_bb": args.small_blind_bb,
            "big_blind_bb": args.big_blind_bb,
            "ante_bb": args.ante_bb,
            "seed": args.seed,
            "max_iterations": args.max_iterations,
            "tolerance": args.tolerance,
        }
        if args.sampling_mode == "stratified":
            solution = solver(
                args.stacks,
                args.payouts,
                samples_per_hand_class=args.samples_per_hand_class,
                **common,
            )
        else:
            solution = solver(args.stacks, args.payouts, samples=args.samples, **common)
    except (ArithmeticError, OSError, ValueError) as exc:
        parser.error(str(exc))

    print(f"3-max ICM · stacks BTN/SB/BB {','.join(f'{value:g}' for value in solution.stacks_bb)} BB")
    print(f"Prémios 1.º/2.º/3.º: {','.join(f'{value:g}' for value in solution.payouts)}")
    print(
        f"NashConv na amostra: {solution.nash_conv:.5f} · {solution.iterations:,} iterações · "
        f"{solution.samples:,} deals · {'dentro da tolerância' if solution.converged else 'acima da tolerância'}"
    )
    print(f"Amostragem: {solution.sampling_method} · seed {solution.seed}")
    print("Frequências de ação por número de combinações físicas:")
    for name in ("btn_push", "sb_push_after_btn_fold", "sb_call_vs_btn_shove", "bb_call_vs_sb_shove", "bb_call_vs_btn_shove", "bb_call_multiway"):
        print(f"  {name}: {_frequency(solution, name):.1%}")
    for name in ("btn_push", "sb_push_after_btn_fold", "sb_call_vs_btn_shove", "bb_call_vs_sb_shove", "bb_call_vs_btn_shove", "bb_call_multiway"):
        print(f"  {name} >= {args.range_cutoff:.0%}: {_range(solution, name, args.range_cutoff)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(solution.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Frequências completas gravadas em {args.output}")
    # The strategy is still useful when the sampled multiplayer game does not
    # reach the requested tolerance; the JSON reports that explicitly.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
