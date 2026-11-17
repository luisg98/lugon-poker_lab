"""Build a multi-stack, multi-payout 3-max chart sweep and training labels.

Run from the repository root, for example:
    py tools/build_three_max_charts.py --samples 20000 --seeds 7,19
    py tools/build_three_max_charts.py --sampling-mode stratified --samples-per-hand-class 500
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.preflop import HAND_CLASSES, combo_count
from lugon_poker.three_max import (
    ThreeMaxSolution,
    solve_three_max_push_fold,
    solve_three_max_push_fold_stratified,
)

ACTION_NAMES = (
    "btn_push",
    "sb_push_after_btn_fold",
    "sb_call_vs_btn_shove",
    "bb_call_vs_sb_shove",
    "bb_call_vs_btn_shove",
    "bb_call_multiway",
)
DEFAULT_STACKS = (6.0, 8.0, 10.0, 12.0, 15.0)
DEFAULT_PAYOUTS = ((1.0, 0.0, 0.0), (0.7, 0.3, 0.0), (0.5, 0.3, 0.2))


def parse_numbers(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("esperava números separados por vírgulas") from exc
    if not values or any(not math.isfinite(value) for value in values):
        raise argparse.ArgumentTypeError("os valores têm de ser números finitos")
    return values


def payout_slug(payouts: Sequence[float]) -> str:
    return "_".join(f"{value:g}".replace(".", "p") for value in payouts)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _combo_weighted_rate(solution: ThreeMaxSolution, action_index: int) -> float:
    return sum(
        combo_count(hand) * frequency
        for hand, frequency in zip(HAND_CLASSES, solution.strategies[action_index])
    ) / 1326.0


def _aggregate_config(
    stack_bb: float,
    payouts: tuple[float, float, float],
    solutions: Sequence[ThreeMaxSolution],
    ante_bb: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize replicate stability and make one model-label row per hand class."""
    action_stability: dict[str, Any] = {}
    for action_index, action in enumerate(ACTION_NAMES):
        rates = [_combo_weighted_rate(solution, action_index) for solution in solutions]
        class_ranges = [
            max(solution.strategies[action_index][hand_index] for solution in solutions)
            - min(solution.strategies[action_index][hand_index] for solution in solutions)
            for hand_index in range(len(HAND_CLASSES))
        ]
        action_stability[action] = {
            "combo_weighted_rate_by_seed": {
                str(solution.seed): rate for solution, rate in zip(solutions, rates)
            },
            "combo_weighted_rate_range": max(rates) - min(rates),
            "combo_weighted_mean_class_range": sum(
                combo_count(hand) * value for hand, value in zip(HAND_CLASSES, class_ranges)
            ) / 1326.0,
            "maximum_class_range": max(class_ranges),
        }

    convergence = [
        {
            "seed": solution.seed,
            "iterations": solution.iterations,
            "nash_conv": solution.nash_conv,
            "converged_on_sample": solution.converged,
        }
        for solution in solutions
    ]
    summary = {
        "stacks_bb": {"BTN": stack_bb, "SB": stack_bb, "BB": stack_bb},
        "payouts": list(payouts),
        "ante_bb_per_player": ante_bb,
        "samples_per_seed": solutions[0].samples,
        "sampling_method": solutions[0].sampling_method,
        "samples_per_hand_class": solutions[0].samples_per_hand_class,
        "seeds": [solution.seed for solution in solutions],
        "all_seeds_within_tolerance": all(solution.converged for solution in solutions),
        "maximum_sampled_nash_conv": max(solution.nash_conv for solution in solutions),
        "mean_expected_icm_by_seat": {
            seat: statistics.fmean(solution.expected_icm[index] for solution in solutions)
            for index, seat in enumerate(("BTN", "SB", "BB"))
        },
        "convergence_by_seed": convergence,
        "action_stability": action_stability,
    }

    rows: list[dict[str, Any]] = []
    for hand_index, hand_class in enumerate(HAND_CLASSES):
        targets: dict[str, Any] = {}
        for action_index, action in enumerate(ACTION_NAMES):
            values = [solution.strategies[action_index][hand_index] for solution in solutions]
            targets[action] = {
                "mean_frequency": statistics.fmean(values),
                "seed_standard_deviation": statistics.pstdev(values),
                "frequency_by_seed": {
                    str(solution.seed): solution.strategies[action_index][hand_index]
                    for solution in solutions
                },
            }
        rows.append(
            {
                "format": "lugon_3max_push_fold_label_v1",
                "features": {
                    "hand_class": hand_class,
                    "stacks_bb": {"BTN": stack_bb, "SB": stack_bb, "BB": stack_bb},
                    "payouts": list(payouts),
                    "small_blind_bb": 0.5,
                    "big_blind_bb": 1.0,
                    "ante_bb_per_player": ante_bb,
                },
                "targets": targets,
                "label_provenance": {
                    "solver": "simultaneous_regret_matching_plus",
                    "deals_per_seed": solutions[0].samples,
                    "sampling_method": solutions[0].sampling_method,
                    "samples_per_hand_class": solutions[0].samples_per_hand_class,
                    "seeds": [solution.seed for solution in solutions],
                    "nash_conv_by_seed": [solution.nash_conv for solution in solutions],
                    "all_seeds_within_tolerance": all(solution.converged for solution in solutions),
                },
            }
        )
    return summary, rows


def build_sweep(
    *,
    stacks: Sequence[float],
    payouts_grid: Sequence[tuple[float, float, float]],
    seeds: Sequence[int],
    sampling_mode: str,
    samples: int,
    samples_per_hand_class: int,
    max_iterations: int,
    tolerance: float,
    ante_bb: float,
    chart_dir: Path,
    dataset_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    total_runs = len(stacks) * len(payouts_grid) * len(seeds)
    completed_runs = 0
    summaries: list[dict[str, Any]] = []
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for stack_bb in stacks:
            for payouts in payouts_grid:
                solutions: list[ThreeMaxSolution] = []
                for seed in seeds:
                    solver_options = {
                        "ante_bb": ante_bb,
                        "seed": seed,
                        "max_iterations": max_iterations,
                        "tolerance": tolerance,
                    }
                    if sampling_mode == "stratified":
                        solution = solve_three_max_push_fold_stratified(
                            (stack_bb, stack_bb, stack_bb),
                            payouts,
                            samples_per_hand_class=samples_per_hand_class,
                            **solver_options,
                        )
                    else:
                        solution = solve_three_max_push_fold(
                            (stack_bb, stack_bb, stack_bb),
                            payouts,
                            samples=samples,
                            **solver_options,
                        )
                    solutions.append(solution)
                    completed_runs += 1
                    ante_slug = f"{ante_bb:g}".replace(".", "p")
                    physical_samples = solution.samples
                    chart_name = (
                        f"3max_{stack_bb:g}bb_{payout_slug(payouts)}_a{ante_slug}_"
                        f"n{physical_samples}_seed{seed}.json"
                    )
                    _write_json(chart_dir / chart_name, solution.to_dict())
                    print(
                        f"[{completed_runs}/{total_runs}] {stack_bb:g} BB · "
                        f"{payout_slug(payouts)} · seed {seed} · "
                        f"NashConv={solution.nash_conv:.5f} · "
                        f"{solution.iterations} iterações · "
                        f"{'OK' if solution.converged else 'acima da tolerância'}",
                        flush=True,
                    )

                summary, rows = _aggregate_config(
                    stack_bb, payouts, solutions, ante_bb
                )
                summaries.append(summary)
                for row in rows:
                    dataset_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                dataset_file.flush()
                _write_json(
                    summary_path,
                    {
                        "format": "lugon_3max_sweep_summary_v1",
                        "status": "in_progress",
                        "settings": {
                            "stacks_bb": list(stacks),
                            "payouts_grid": [list(payouts) for payouts in payouts_grid],
                            "seeds": list(seeds),
                            "samples_per_seed": solutions[0].samples,
                            "sampling_mode": sampling_mode,
                            "samples_per_hand_class": (
                                samples_per_hand_class if sampling_mode == "stratified" else None
                            ),
                            "max_iterations": max_iterations,
                            "tolerance": tolerance,
                            "ante_bb_per_player": ante_bb,
                        },
                        "completed_solver_runs": completed_runs,
                        "total_solver_runs": total_runs,
                        "configurations": summaries,
                    },
                )

    result = {
        "format": "lugon_3max_sweep_summary_v1",
        "status": "complete",
        "settings": {
            "stacks_bb": list(stacks),
            "payouts_grid": [list(payouts) for payouts in payouts_grid],
            "seeds": list(seeds),
            "samples_per_seed": summaries[0]["samples_per_seed"] if summaries else 0,
            "sampling_mode": sampling_mode,
            "samples_per_hand_class": (
                samples_per_hand_class if sampling_mode == "stratified" else None
            ),
            "max_iterations": max_iterations,
            "tolerance": tolerance,
            "ante_bb_per_player": ante_bb,
        },
        "completed_solver_runs": completed_runs,
        "total_solver_runs": total_runs,
        "training_rows": len(summaries) * len(HAND_CLASSES),
        "configurations": summaries,
    }
    _write_json(summary_path, result)
    print(f"Labels: {dataset_path} ({result['training_rows']} linhas)", flush=True)
    print(f"Resumo: {summary_path}", flush=True)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Gera charts e labels 3-max para uma grelha de stacks/prémios.")
    parser.add_argument("--stacks", type=parse_numbers, default=DEFAULT_STACKS, help="stacks iguais em BB, ex.: 6,8,10,12,15")
    parser.add_argument(
        "--payout", action="append", type=parse_numbers,
        help="estrutura 1.º,2.º,3.º; repetir para várias, ex.: --payout 1,0,0 --payout 0.5,0.3,0.2",
    )
    parser.add_argument("--seeds", type=parse_numbers, default=(7.0, 19.0), help="seeds inteiras separadas por vírgulas")
    parser.add_argument("--sampling-mode", choices=("fixed", "stratified"), default="fixed")
    parser.add_argument("--samples", type=int, default=20_000, help="deals físicos aleatórios no modo fixed")
    parser.add_argument("--samples-per-hand-class", type=int, default=100, help="deals por lugar/classe no modo stratified")
    parser.add_argument("--max-iterations", type=int, default=1_000)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--ante-bb", type=float, default=0.0)
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_sweep"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training/three_max_push_fold_labels.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_sweep_summary.json"))
    args = parser.parse_args(argv)

    payouts_grid = tuple(tuple(values) for values in (args.payout or DEFAULT_PAYOUTS))
    if any(len(values) != 3 for values in payouts_grid):
        parser.error("cada --payout tem de conter exatamente três números")
    if any(value <= 0.0 for value in args.stacks):
        parser.error("todos os stacks têm de ser positivos")
    if any(value < 0 or int(value) != value for value in args.seeds):
        parser.error("as seeds têm de ser inteiros não negativos")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        parser.error("indica pelo menos uma seed, sem duplicados")
    if args.samples < 1 or args.samples_per_hand_class < 1:
        parser.error("o número de amostras tem de ser positivo")
    try:
        build_sweep(
            stacks=args.stacks,
            payouts_grid=payouts_grid,  # type: ignore[arg-type]
            seeds=tuple(int(seed) for seed in args.seeds),
            sampling_mode=args.sampling_mode,
            samples=args.samples,
            samples_per_hand_class=args.samples_per_hand_class,
            max_iterations=args.max_iterations,
            tolerance=args.tolerance,
            ante_bb=args.ante_bb,
            chart_dir=args.chart_dir,
            dataset_path=args.dataset,
            summary_path=args.summary,
        )
    except (ArithmeticError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
