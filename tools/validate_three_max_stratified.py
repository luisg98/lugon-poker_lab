"""Evaluate saved 3-max profiles on balanced conditional chance samples.

The tool reports the plug-in NashConv and a cross-fitted best-response gain.
The latter chooses actions on one half of the sample and evaluates them on the
other half, reducing the upward bias from maximizing noisy action values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.three_max import (
    _crossfit_stratified_br_gain,
    _make_stratified_deals,
    _stratified_nash_conv,
)

ACTION_NAMES = (
    "btn_push",
    "sb_push_after_btn_fold",
    "sb_call_vs_btn_shove",
    "bb_call_vs_sb_shove",
    "bb_call_vs_btn_shove",
    "bb_call_multiway",
)


def _numbers(text: str) -> tuple[float, ...]:
    try:
        return tuple(float(part.strip()) for part in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("usa números separados por vírgulas") from exc


def _payout_slug(payouts: tuple[float, float, float]) -> str:
    return "_".join(f"{value:g}".replace(".", "p") for value in payouts)


def _load_strategy(path: Path) -> tuple[tuple[float, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("strategies")
    if payload.get("format") != "lugon_3max_push_fold_icm_v1" or not isinstance(rows, list):
        raise ValueError(f"chart inválido: {path}")
    if not rows:
        raise ValueError(f"chart sem estratégias: {path}")
    return tuple(tuple(float(row[action]) for row in rows) for action in ACTION_NAMES)


def validate_profiles(
    *,
    stack_bb: float,
    payouts: tuple[float, float, float],
    chart_dir: Path,
    chart_samples: int,
    chart_seeds: tuple[int, ...],
    validation_samples_per_class: int,
    validation_seed: int,
    output: Path,
) -> dict[str, Any]:
    stacks = (stack_bb, stack_bb, stack_bb)
    forced = (0.0, 0.5, 1.0)
    profiles = []
    strategies = []
    for seed in chart_seeds:
        chart = chart_dir / (
            f"3max_{stack_bb:g}bb_{_payout_slug(payouts)}_a0_n{chart_samples}_seed{seed}.json"
        )
        payload = json.loads(chart.read_text(encoding="utf-8"))
        profile = _load_strategy(chart)
        strategies.append(profile)
        profiles.append({"seed": seed, "chart": str(chart), "training_nash_conv": payload["convergence"]["nash_conv_payout_units"]})

    conditioned = _make_stratified_deals(
        validation_samples_per_class,
        validation_seed,
        stacks,
        forced,
        payouts,
    )
    evaluated = []
    for profile, strategy in zip(profiles, strategies):
        gap, expected = _stratified_nash_conv(conditioned, strategy)
        crossfit, a_to_b, b_to_a = _crossfit_stratified_br_gain(conditioned, strategy)
        evaluated.append(
            {
                **profile,
                "stratified_plugin_nash_conv": gap,
                "crossfit_br_gain_mean": crossfit,
                "crossfit_br_gain_a_to_b": a_to_b,
                "crossfit_br_gain_b_to_a": b_to_a,
                "expected_icm": {seat: expected[i] for i, seat in enumerate(("BTN", "SB", "BB"))},
            }
        )

    average = tuple(
        tuple(sum(profile[action][hand] for profile in strategies) / len(strategies) for hand in range(len(strategies[0][action])))
        for action in range(len(ACTION_NAMES))
    )
    average_gap, average_expected = _stratified_nash_conv(conditioned, average)
    average_crossfit, average_a_to_b, average_b_to_a = _crossfit_stratified_br_gain(conditioned, average)
    result = {
        "format": "lugon_3max_stratified_validation_v1",
        "parameters": {"stacks_bb": [stack_bb] * 3, "payouts": list(payouts), "ante_bb": 0.0},
        "validation": {
            "method": "balanced conditional samples by seat and private hand class",
            "samples_per_seat_hand_class": validation_samples_per_class,
            "total_physical_deals": 3 * 169 * validation_samples_per_class,
            "seed": validation_seed,
            "crossfit": "select each information-set response on one half; score on the other, then reverse",
        },
        "profiles_by_training_seed": evaluated,
        "mean_strategy_profile": {
            "stratified_plugin_nash_conv": average_gap,
            "crossfit_br_gain_mean": average_crossfit,
            "crossfit_br_gain_a_to_b": average_a_to_b,
            "crossfit_br_gain_b_to_a": average_b_to_a,
            "expected_icm": {seat: average_expected[i] for i, seat in enumerate(("BTN", "SB", "BB"))},
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida charts 3-max com chance estratificada e best response cross-fit.")
    parser.add_argument("--stack-bb", type=float, required=True)
    parser.add_argument("--payouts", type=_numbers, required=True, help="prémios 1.º,2.º,3.º")
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_sweep"))
    parser.add_argument("--chart-samples", type=int, default=20_000)
    parser.add_argument("--chart-seeds", type=_numbers, default=(7.0, 19.0))
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12011)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if len(args.payouts) != 3:
        parser.error("--payouts tem de conter três números")
    if any(value < 0 or int(value) != value for value in args.chart_seeds):
        parser.error("--chart-seeds tem de conter inteiros não negativos")
    if not args.chart_seeds or len(set(args.chart_seeds)) != len(args.chart_seeds):
        parser.error("--chart-seeds requer valores sem duplicados")
    if args.samples_per_class < 2:
        parser.error("--samples-per-class tem de ser pelo menos 2 para cross-fit")
    try:
        result = validate_profiles(
            stack_bb=args.stack_bb,
            payouts=tuple(args.payouts),  # type: ignore[arg-type]
            chart_dir=args.chart_dir,
            chart_samples=args.chart_samples,
            chart_seeds=tuple(int(seed) for seed in args.chart_seeds),
            validation_samples_per_class=args.samples_per_class,
            validation_seed=args.seed,
            output=args.output,
        )
    except (ArithmeticError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    print(
        f"Plugin NashConv={result['mean_strategy_profile']['stratified_plugin_nash_conv']:.5f}; "
        f"crossfit BR gain={result['mean_strategy_profile']['crossfit_br_gain_mean']:.5f} "
        f"({result['validation']['total_physical_deals']:,} deals)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
