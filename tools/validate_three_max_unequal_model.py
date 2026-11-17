"""Evaluate the unequal-stack model on unseen 3-max stack vectors."""

from __future__ import annotations

import argparse
import json
import math
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
from lugon_poker.three_max_model import ACTION_NAMES, UnequalStackThreeMaxFrequencyModel
from build_three_max_unequal_charts import SEATS, _chart_path
from validate_three_max_stratified import _load_strategy


def validate_unequal_model(
    *,
    model_path: Path,
    summary_path: Path,
    chart_dir: Path,
    output_path: Path,
    samples_per_class: int,
    seed: int,
) -> dict[str, Any]:
    model = UnequalStackThreeMaxFrequencyModel.load(model_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("the unequal-stack holdout sweep is incomplete")
    if summary.get("settings", {}).get("sampling_mode") != "stratified":
        raise ValueError("holdout charts must use stratified sampling")

    training_vectors = {
        tuple(float(value) for value in vector)
        for vector in model.training_stack_vectors_bb
    }
    training_seeds_raw = model.training_provenance.get("solver_seeds")
    if not isinstance(training_seeds_raw, list) or len(training_seeds_raw) < 2:
        raise ValueError("model must record its solver seeds to validate an independent holdout")
    training_seeds = set(training_seeds_raw)
    if len(training_seeds) != len(training_seeds_raw):
        raise ValueError("model training provenance contains duplicate solver seeds")
    seen_holdout_vectors: set[tuple[float, float, float]] = set()

    all_errors: list[tuple[float, int]] = []
    errors_by_action: dict[str, list[tuple[float, int]]] = {action: [] for action in ACTION_NAMES}
    results: list[dict[str, Any]] = []
    for index, config in enumerate(summary["configurations"], start=1):
        stacks = tuple(float(config["stacks_bb"][seat]) for seat in SEATS)
        payouts = tuple(float(value) for value in config["payouts"])
        if payouts != model.payouts:
            continue
        if stacks in training_vectors:
            raise ValueError(f"holdout includes a training stack vector: {stacks}")
        if stacks in seen_holdout_vectors:
            raise ValueError(f"holdout repeats a stack vector: {stacks}")
        seen_holdout_vectors.add(stacks)
        ante = float(config["ante_bb_per_player"])
        if ante != 0.0:
            raise ValueError("this model only supports zero ante")
        raw_chart_seeds = config.get("seeds")
        if not isinstance(raw_chart_seeds, list) or len(raw_chart_seeds) < 2:
            raise ValueError("holdout configurations need at least two recorded solver seeds")
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in raw_chart_seeds):
            raise ValueError("holdout solver seeds must be nonnegative integers")
        if len(set(raw_chart_seeds)) != len(raw_chart_seeds):
            raise ValueError("holdout configuration contains duplicate solver seeds")
        if training_seeds.intersection(raw_chart_seeds):
            raise ValueError("holdout solver seeds overlap the model's training seeds")
        # The model's predict method enforces each seat's trained range.
        physical_samples = int(config["samples_per_seed"])
        profiles = []
        for chart_seed in config["seeds"]:
            chart = _chart_path(chart_dir, stacks, payouts, ante, physical_samples, int(chart_seed))
            chart_payload = json.loads(chart.read_text(encoding="utf-8"))
            parameters = chart_payload.get("parameters", {})
            sampling = chart_payload.get("sampling", {})
            convergence = chart_payload.get("convergence", {})
            if chart_payload.get("format") != "lugon_3max_push_fold_icm_v1":
                raise ValueError(f"holdout chart has an unsupported format: {chart}")
            if (
                parameters.get("stacks_bb") != dict(zip(SEATS, stacks))
                or parameters.get("payouts") != list(payouts)
                or parameters.get("ante_bb_per_player") != ante
                or parameters.get("small_blind_bb") != 0.5
                or parameters.get("big_blind_bb") != 1.0
            ):
                raise ValueError(f"holdout chart parameters do not match the summary: {chart}")
            if (
                sampling.get("seed") != int(chart_seed)
                or sampling.get("method") != "stratified_by_seat_and_hand_class"
                or sampling.get("samples_per_hand_class") != config.get("samples_per_hand_class")
                or sampling.get("deals") != 3 * len(HAND_CLASSES) * config.get("samples_per_hand_class", 0)
            ):
                raise ValueError(f"holdout chart sampling provenance is inconsistent: {chart}")
            sampled_nash_conv = convergence.get("nash_conv_payout_units")
            if (
                convergence.get("converged_on_sample") is not True
                or isinstance(sampled_nash_conv, bool)
                or not isinstance(sampled_nash_conv, (int, float))
                or not math.isfinite(sampled_nash_conv)
                or sampled_nash_conv > float(summary.get("settings", {}).get("tolerance", 0.0))
            ):
                raise ValueError(f"holdout chart did not meet the recorded solver tolerance: {chart}")
            profiles.append(_load_strategy(chart))
        if len(profiles) < 2:
            raise ValueError("holdout configurations need at least two solver seeds")
        reference = tuple(
            tuple(sum(profile[action][hand] for profile in profiles) / len(profiles) for hand in range(169))
            for action in range(len(ACTION_NAMES))
        )
        predicted = tuple(
            tuple(model.predict(hand, stacks, payouts=payouts)[action] for hand in HAND_CLASSES)
            for action in ACTION_NAMES
        )

        config_errors: list[tuple[float, int]] = []
        seed_disagreement: list[tuple[float, int]] = []
        for action_index, action in enumerate(ACTION_NAMES):
            for hand_index, hand in enumerate(HAND_CLASSES):
                weighted_error = (
                    abs(predicted[action_index][hand_index] - reference[action_index][hand_index]),
                    combo_count(hand),
                )
                all_errors.append(weighted_error)
                config_errors.append(weighted_error)
                errors_by_action[action].append(weighted_error)
                for left in range(len(profiles)):
                    for right in range(left + 1, len(profiles)):
                        disagreement = (
                            abs(profiles[left][action_index][hand_index] - profiles[right][action_index][hand_index]),
                            combo_count(hand),
                        )
                        seed_disagreement.append(disagreement)

        forced = (ante, ante + 0.5, ante + 1.0)
        deals = _make_stratified_deals(samples_per_class, seed, stacks, forced, payouts)  # type: ignore[arg-type]
        model_plugin, model_expected = _stratified_nash_conv(deals, predicted)
        model_crossfit, model_a_to_b, model_b_to_a = _crossfit_stratified_br_gain(deals, predicted)
        solver_plugin, _ = _stratified_nash_conv(deals, reference)
        solver_crossfit, solver_a_to_b, solver_b_to_a = _crossfit_stratified_br_gain(deals, reference)
        per_profile_crossfit = [
            _crossfit_stratified_br_gain(deals, profile)[0]
            for profile in profiles
        ]
        result = {
            "stacks_bb": dict(zip(SEATS, stacks)),
            "solver_seed_count": len(profiles),
            "frequency_mae_combo_weighted": sum(error * weight for error, weight in config_errors)
            / sum(weight for _, weight in config_errors),
            "solver_seed_disagreement_mae_combo_weighted": (
                sum(error * weight for error, weight in seed_disagreement)
                / sum(weight for _, weight in seed_disagreement)
            ),
            "model_plugin_nash_conv": model_plugin,
            "model_crossfit_br_gain": model_crossfit,
            "model_crossfit_directions": [model_a_to_b, model_b_to_a],
            "solver_mean_plugin_nash_conv": solver_plugin,
            "solver_mean_crossfit_br_gain": solver_crossfit,
            "solver_mean_crossfit_directions": [solver_a_to_b, solver_b_to_a],
            "individual_solver_profile_crossfit_br_gain": per_profile_crossfit,
            "model_expected_icm": {seat: model_expected[seat_index] for seat_index, seat in enumerate(SEATS)},
        }
        results.append(result)
        print(
            f"[{index}/{len(summary['configurations'])}] "
            f"BTN/SB/BB={','.join(f'{value:g}' for value in stacks)} BB · "
            f"MAE={result['frequency_mae_combo_weighted']:.4f} · "
            f"crossfit modelo={model_crossfit:.5f} · solver={solver_crossfit:.5f}",
            flush=True,
        )
        del deals, profiles, reference, predicted, per_profile_crossfit, seed_disagreement

    if not results:
        raise ValueError("the holdout summary contains no configuration for this model's payout structure")
    total_weight = sum(weight for _, weight in all_errors)
    output = {
        "format": "lugon_3max_unequal_frequency_model_holdout_validation_v1",
        "model": str(model_path),
        "interaction_terms": model.interaction_terms,
        "training_stack_vectors_bb": [list(vector) for vector in model.training_stack_vectors_bb],
        "training_provenance": dict(model.training_provenance),
        "holdout_stack_vectors_bb": [result["stacks_bb"] for result in results],
        "payouts": list(model.payouts),
        "sampling": {
            "method": "stratified_crossfit",
            "samples_per_seat_hand_class": samples_per_class,
            "total_physical_deals_per_config": 3 * 169 * samples_per_class,
            "seed": seed,
        },
        "combo_weighted_frequency_mae": sum(error * weight for error, weight in all_errors) / total_weight,
        "combo_weighted_frequency_mae_by_action": {
            action: sum(error * weight for error, weight in values) / sum(weight for _, weight in values)
            for action, values in errors_by_action.items()
        },
        "solver_seed_disagreement_mae_combo_weighted": sum(
            result["solver_seed_disagreement_mae_combo_weighted"] for result in results
        ) / len(results),
        "mean_model_plugin_nash_conv": sum(result["model_plugin_nash_conv"] for result in results) / len(results),
        "mean_solver_plugin_nash_conv": sum(result["solver_mean_plugin_nash_conv"] for result in results) / len(results),
        "mean_model_crossfit_br_gain": sum(result["model_crossfit_br_gain"] for result in results) / len(results),
        "mean_solver_crossfit_br_gain": sum(result["solver_mean_crossfit_br_gain"] for result in results) / len(results),
        "configurations": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    print(f"Validação guardada: {output_path}", flush=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/models/three_max_unequal_50_30_20_linear.json"))
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_unequal_holdout_summary.json"))
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_unequal_holdout"))
    parser.add_argument("--output", type=Path, default=Path("data/charts/three_max_unequal_model_holdout_validation.json"))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=99241)
    args = parser.parse_args(argv)
    if args.samples_per_class < 2:
        parser.error("samples-per-class must be at least two for cross-fit")
    if args.seed < 0:
        parser.error("seed must be nonnegative")
    try:
        validate_unequal_model(
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
