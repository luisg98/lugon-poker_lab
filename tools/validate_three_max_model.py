"""Evaluate the trained 3-max model on solver charts at unseen stack depths."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lugon_poker.preflop import HAND_CLASSES, combo_count
from lugon_poker.three_max import (
    _crossfit_stratified_br_gain,
    _make_stratified_deals,
    _stratified_nash_conv,
)
from lugon_poker.three_max_model import ACTION_NAMES, ThreeMaxFrequencyModel
from validate_three_max_stratified import _load_strategy, _payout_slug


def validate_model(
    *,
    model_path: Path,
    summary_path: Path,
    chart_dir: Path,
    output_path: Path,
    samples_per_class: int,
    seed: int,
) -> dict[str, Any]:
    model = ThreeMaxFrequencyModel.load(model_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("the holdout chart sweep is incomplete")
    settings = summary["settings"]
    if settings.get("sampling_mode") != "stratified":
        raise ValueError("holdout charts must use stratified sampling")

    results: list[dict[str, Any]] = []
    all_errors: list[tuple[float, int]] = []
    errors_by_action: dict[str, list[tuple[float, int]]] = {action: [] for action in ACTION_NAMES}
    seed_disagreement_by_action: dict[str, list[tuple[float, int]]] = {
        action: [] for action in ACTION_NAMES
    }
    for config_index, config in enumerate(summary["configurations"], start=1):
        stack = float(config["stacks_bb"]["BTN"])
        payouts = tuple(float(value) for value in config["payouts"])
        if payouts != model.payouts:
            continue
        if any(float(config["stacks_bb"][seat]) != stack for seat in ("SB", "BB")):
            raise ValueError("holdout must have equal stacks")
        if float(config["ante_bb_per_player"]) != 0.0:
            raise ValueError("holdout must have zero ante")
        if stack < model.stack_min_bb or stack > model.stack_max_bb:
            raise ValueError(f"holdout stack {stack:g} BB lies outside the model's supported domain")

        physical_samples = int(config["samples_per_seed"])
        ante_slug = "0"
        profiles = []
        for chart_seed in config["seeds"]:
            chart = chart_dir / (
                f"3max_{stack:g}bb_{_payout_slug(payouts)}_a{ante_slug}_"
                f"n{physical_samples}_seed{int(chart_seed)}.json"
            )
            profiles.append(_load_strategy(chart))
        reference = tuple(
            tuple(sum(profile[action][hand] for profile in profiles) / len(profiles) for hand in range(169))
            for action in range(len(ACTION_NAMES))
        )
        predicted = tuple(
            tuple(model.predict(hand, stack, payouts=payouts)[action] for hand in HAND_CLASSES)
            for action in ACTION_NAMES
        )

        config_errors: list[tuple[float, int]] = []
        seed_disagreement: list[tuple[float, int]] = []
        for action_index, action in enumerate(ACTION_NAMES):
            for hand_index, hand in enumerate(HAND_CLASSES):
                error = abs(predicted[action_index][hand_index] - reference[action_index][hand_index])
                weighted = (error, combo_count(hand))
                all_errors.append(weighted)
                config_errors.append(weighted)
                errors_by_action[action].append(weighted)
                for left_index in range(len(profiles)):
                    for right_index in range(left_index + 1, len(profiles)):
                        disagreement = (
                            abs(
                                profiles[left_index][action_index][hand_index]
                                - profiles[right_index][action_index][hand_index]
                            ),
                            combo_count(hand),
                        )
                        seed_disagreement.append(disagreement)
                        seed_disagreement_by_action[action].append(disagreement)

        forced = (0.0, 0.5, 1.0)
        deals = _make_stratified_deals(
            samples_per_class,
            seed,
            (stack, stack, stack),
            forced,
            payouts,  # type: ignore[arg-type]
        )
        model_plugin, model_expected = _stratified_nash_conv(deals, predicted)
        model_crossfit, model_a_to_b, model_b_to_a = _crossfit_stratified_br_gain(deals, predicted)
        reference_plugin, _ = _stratified_nash_conv(deals, reference)
        reference_crossfit, reference_a_to_b, reference_b_to_a = _crossfit_stratified_br_gain(deals, reference)
        profile_crossfits = [
            _crossfit_stratified_br_gain(deals, profile)[0]
            for profile in profiles
        ]

        result = {
            "stack_bb": stack,
            "solver_seed_count": len(profiles),
            "frequency_mae_combo_weighted": sum(error * weight for error, weight in config_errors)
            / sum(weight for _, weight in config_errors),
            "solver_seed_disagreement_mae_combo_weighted": (
                sum(error * weight for error, weight in seed_disagreement)
                / sum(weight for _, weight in seed_disagreement)
                if seed_disagreement
                else 0.0
            ),
            "model_plugin_nash_conv": model_plugin,
            "model_crossfit_br_gain": model_crossfit,
            "model_crossfit_directions": [model_a_to_b, model_b_to_a],
            "solver_mean_plugin_nash_conv": reference_plugin,
            "solver_mean_crossfit_br_gain": reference_crossfit,
            "solver_mean_crossfit_directions": [reference_a_to_b, reference_b_to_a],
            "individual_solver_profile_crossfit_br_gain": profile_crossfits,
            "model_expected_icm": {seat: model_expected[i] for i, seat in enumerate(("BTN", "SB", "BB"))},
        }
        results.append(result)
        print(
            f"[{config_index}] {stack:g} BB · MAE={result['frequency_mae_combo_weighted']:.4f} · "
            f"crossfit modelo={model_crossfit:.5f} · solver médio={reference_crossfit:.5f}",
            flush=True,
        )
        # Release deal samples before constructing the next independent holdout.
        del deals, profiles, reference, predicted, profile_crossfits, seed_disagreement

    if not results:
        raise ValueError("the holdout summary contains no configuration for this model's payouts")
    total_weight = sum(weight for _, weight in all_errors)
    summary_result = {
        "format": "lugon_3max_frequency_model_holdout_validation_v1",
        "model": str(model_path),
        "training_stacks_bb": list(model.training_stacks_bb),
        "training_provenance": dict(model.training_provenance),
        "holdout_stacks_bb": [result["stack_bb"] for result in results],
        "payouts": list(model.payouts),
        "sampling": {
            "method": "stratified_crossfit",
            "samples_per_seat_hand_class": samples_per_class,
            "total_physical_deals_per_stack": 3 * 169 * samples_per_class,
            "seed": seed,
        },
        "combo_weighted_frequency_mae": sum(error * weight for error, weight in all_errors) / total_weight,
        "combo_weighted_frequency_mae_by_action": {
            action: sum(error * weight for error, weight in values) / sum(weight for _, weight in values)
            for action, values in errors_by_action.items()
        },
        "solver_seed_disagreement_mae_combo_weighted_by_action": {
            action: sum(error * weight for error, weight in values) / sum(weight for _, weight in values)
            for action, values in seed_disagreement_by_action.items()
        },
        "solver_seed_disagreement_mae_combo_weighted": (
            sum(
                result["solver_seed_disagreement_mae_combo_weighted"]
                for result in results
            )
            / len(results)
        ),
        "mean_model_plugin_nash_conv": sum(result["model_plugin_nash_conv"] for result in results) / len(results),
        "mean_solver_plugin_nash_conv": sum(result["solver_mean_plugin_nash_conv"] for result in results) / len(results),
        "mean_model_crossfit_br_gain": sum(result["model_crossfit_br_gain"] for result in results) / len(results),
        "mean_solver_crossfit_br_gain": sum(result["solver_mean_crossfit_br_gain"] for result in results) / len(results),
        "configurations": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(summary_result, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"Validação independente gravada: {output_path}", flush=True)
    return summary_result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/models/three_max_50_30_20_linear.json"))
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_holdout_summary.json"))
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_holdout"))
    parser.add_argument("--output", type=Path, default=Path("data/charts/three_max_model_holdout_validation.json"))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=99173)
    args = parser.parse_args(argv)
    if args.samples_per_class < 2:
        parser.error("samples-per-class must be at least 2 for cross-fit")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        validate_model(
            model_path=args.model,
            summary_path=args.summary,
            chart_dir=args.chart_dir,
            output_path=args.output,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    except (ArithmeticError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
