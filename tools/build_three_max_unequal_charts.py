"""Build stratified solver labels for equal and unequal 3-max stack vectors."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.preflop import HAND_CLASSES
from lugon_poker.three_max import solve_three_max_push_fold_stratified

DEFAULT_STACK_VECTORS = (
    (6.0, 6.0, 6.0),
    (8.0, 8.0, 8.0),
    (10.0, 10.0, 10.0),
    (12.0, 12.0, 12.0),
    (15.0, 15.0, 15.0),
    (6.0, 10.0, 10.0),
    (15.0, 10.0, 10.0),
    (10.0, 6.0, 10.0),
    (10.0, 15.0, 10.0),
    (10.0, 10.0, 6.0),
    (10.0, 10.0, 15.0),
)
SEATS = ("BTN", "SB", "BB")
ACTION_NAMES = (
    "btn_push",
    "sb_push_after_btn_fold",
    "sb_call_vs_btn_shove",
    "bb_call_vs_sb_shove",
    "bb_call_vs_btn_shove",
    "bb_call_multiway",
)


def parse_stack_vectors(text: str) -> tuple[tuple[float, float, float], ...]:
    vectors: list[tuple[float, float, float]] = []
    try:
        for item in text.split(";"):
            vector = tuple(float(part.strip()) for part in item.split(","))
            if len(vector) != 3 or any(not math.isfinite(value) or value <= 0.0 for value in vector):
                raise ValueError
            vectors.append(vector)  # type: ignore[arg-type]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "usa vetores BTN,SB,BB positivos separados por ponto e vírgula"
        ) from exc
    if not vectors or len(set(vectors)) != len(vectors):
        raise argparse.ArgumentTypeError("indica pelo menos um vetor de stacks, sem duplicados")
    return tuple(vectors)


def _num(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def stack_slug(stacks: Sequence[float]) -> str:
    return "_".join(f"{seat.lower()}{_num(value)}" for seat, value in zip(SEATS, stacks))


def payout_slug(payouts: Sequence[float]) -> str:
    return "_".join(_num(value) for value in payouts)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _chart_path(
    chart_dir: Path,
    stacks: Sequence[float],
    payouts: Sequence[float],
    ante: float,
    samples: int,
    seed: int,
) -> Path:
    ante_slug = _num(ante)
    name = (
        f"3max_{stack_slug(stacks)}bb_{payout_slug(payouts)}_a{ante_slug}_"
        f"n{samples}_seed{seed}.json"
    )
    return chart_dir / name


def _build_rows(
    stacks: tuple[float, float, float],
    payouts: tuple[float, float, float],
    solutions: Sequence[Any],
    ante: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
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
        "stacks_bb": dict(zip(SEATS, stacks)),
        "payouts": list(payouts),
        "ante_bb_per_player": ante,
        "samples_per_seed": solutions[0].samples,
        "sampling_method": solutions[0].sampling_method,
        "samples_per_hand_class": solutions[0].samples_per_hand_class,
        "seeds": [solution.seed for solution in solutions],
        "all_seeds_within_tolerance": all(solution.converged for solution in solutions),
        "maximum_sampled_nash_conv": max(solution.nash_conv for solution in solutions),
        "convergence_by_seed": convergence,
    }
    rows: list[dict[str, Any]] = []
    for hand_index, hand_class in enumerate(HAND_CLASSES):
        targets = {}
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
                    "stacks_bb": dict(zip(SEATS, stacks)),
                    "payouts": list(payouts),
                    "small_blind_bb": 0.5,
                    "big_blind_bb": 1.0,
                    "ante_bb_per_player": ante,
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


def build_unequal_sweep(
    *,
    stack_vectors: Sequence[tuple[float, float, float]],
    payouts: tuple[float, float, float],
    seeds: Sequence[int],
    samples_per_hand_class: int,
    max_iterations: int,
    tolerance: float,
    ante_bb: float,
    chart_dir: Path,
    dataset_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    total_runs = len(stack_vectors) * len(seeds)
    completed_runs = 0
    summaries: list[dict[str, Any]] = []
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    with dataset_path.open("w", encoding="utf-8", newline="\n") as dataset_file:
        for stacks in stack_vectors:
            solutions = []
            for seed in seeds:
                solution = solve_three_max_push_fold_stratified(
                    stacks,
                    payouts,
                    ante_bb=ante_bb,
                    seed=seed,
                    samples_per_hand_class=samples_per_hand_class,
                    max_iterations=max_iterations,
                    tolerance=tolerance,
                )
                solutions.append(solution)
                completed_runs += 1
                chart = _chart_path(chart_dir, stacks, payouts, ante_bb, solution.samples, seed)
                _write_json(chart, solution.to_dict())
                print(
                    f"[{completed_runs}/{total_runs}] {stack_slug(stacks)} BB · seed {seed} · "
                    f"NashConv={solution.nash_conv:.5f} · {solution.iterations} iterações · "
                    f"{'OK' if solution.converged else 'acima da tolerância'}",
                    flush=True,
                )

            configuration, rows = _build_rows(stacks, payouts, solutions, ante_bb)
            summaries.append(configuration)
            for row in rows:
                dataset_file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            dataset_file.flush()
            _write_json(
                summary_path,
                {
                    "format": "lugon_3max_unequal_sweep_summary_v1",
                    "status": "in_progress",
                    "settings": {
                        "stack_vectors_bb": [list(vector) for vector in stack_vectors],
                        "payouts": list(payouts),
                        "seeds": list(seeds),
                        "sampling_mode": "stratified",
                        "samples_per_hand_class": samples_per_hand_class,
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
        "format": "lugon_3max_unequal_sweep_summary_v1",
        "status": "complete",
        "settings": {
            "stack_vectors_bb": [list(vector) for vector in stack_vectors],
            "payouts": list(payouts),
            "seeds": list(seeds),
            "sampling_mode": "stratified",
            "samples_per_hand_class": samples_per_hand_class,
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stack-vectors",
        type=parse_stack_vectors,
        default=DEFAULT_STACK_VECTORS,
        help="vetores BTN,SB,BB separados por ponto e vírgula",
    )
    parser.add_argument("--payouts", type=str, default="0.5,0.3,0.2", help="prémios 1.º,2.º,3.º em BB/ICM")
    parser.add_argument("--seeds", type=str, default="131,149,167,181")
    parser.add_argument("--samples-per-hand-class", type=int, default=500)
    parser.add_argument("--max-iterations", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=0.005)
    parser.add_argument("--ante-bb", type=float, default=0.0)
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_unequal_train"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training/three_max_unequal_train.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_unequal_train_summary.json"))
    args = parser.parse_args(argv)
    try:
        payouts = tuple(float(part.strip()) for part in args.payouts.split(","))
        seeds_values = tuple(int(part.strip()) for part in args.seeds.split(","))
        if len(payouts) != 3 or any(not math.isfinite(value) or value < 0.0 for value in payouts):
            raise ValueError("payouts têm de conter três valores finitos não negativos")
        if not seeds_values or len(set(seeds_values)) != len(seeds_values) or any(seed < 0 for seed in seeds_values):
            raise ValueError("seeds têm de ser inteiros não negativos, sem duplicados")
        if args.samples_per_hand_class < 1 or args.max_iterations < 1:
            raise ValueError("amostras e iterações têm de ser positivas")
        if args.ante_bb < 0.0 or not math.isfinite(args.ante_bb):
            raise ValueError("ante tem de ser finito e não negativo")
        if not math.isfinite(args.tolerance) or args.tolerance <= 0.0:
            raise ValueError("tolerância tem de ser finita e positiva")
        if all(stacks[0] == stacks[1] == stacks[2] for stacks in args.stack_vectors):
            raise ValueError("a grelha tem de incluir pelo menos um vetor de stacks desiguais")
        build_unequal_sweep(
            stack_vectors=args.stack_vectors,
            payouts=payouts,  # type: ignore[arg-type]
            seeds=seeds_values,
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
