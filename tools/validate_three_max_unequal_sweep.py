"""Cross-fit validate labels from an unequal-stack 3-max sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lugon_poker.three_max import (
    _crossfit_stratified_br_gain,
    _make_stratified_deals,
    _stratified_nash_conv,
)
from build_three_max_unequal_charts import ACTION_NAMES, SEATS, _chart_path
from validate_three_max_stratified import _load_strategy


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_unequal_sweep(
    *,
    summary_path: Path,
    chart_dir: Path,
    dataset_path: Path,
    output_path: Path,
    samples_per_class: int,
    seed: int,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("the unequal-stack sweep is incomplete")
    if summary.get("settings", {}).get("sampling_mode") != "stratified":
        raise ValueError("the sweep must use stratified sampling")

    results: list[dict[str, Any]] = []
    validation_by_key: dict[tuple[tuple[float, float, float], tuple[float, ...]], dict[str, Any]] = {}
    for index, config in enumerate(summary["configurations"], start=1):
        stacks = tuple(float(config["stacks_bb"][seat]) for seat in SEATS)
        payouts = tuple(float(value) for value in config["payouts"])
        ante = float(config["ante_bb_per_player"])
        forced = (ante, ante + 0.5, ante + 1.0)
        physical_samples = int(config["samples_per_seed"])
        chart_seeds = [int(value) for value in config["seeds"]]
        deals = _make_stratified_deals(
            samples_per_class,
            seed,
            stacks,
            forced,
            payouts,  # type: ignore[arg-type]
        )

        strategies = []
        seed_profiles = []
        for chart_seed in chart_seeds:
            chart = _chart_path(chart_dir, stacks, payouts, ante, physical_samples, chart_seed)
            payload = json.loads(chart.read_text(encoding="utf-8"))
            strategy = _load_strategy(chart)
            strategies.append(strategy)
            plugin_gap, _ = _stratified_nash_conv(deals, strategy)
            crossfit, a_to_b, b_to_a = _crossfit_stratified_br_gain(deals, strategy)
            seed_profiles.append(
                {
                    "seed": chart_seed,
                    "training_plugin_nash_conv": payload["convergence"]["nash_conv_payout_units"],
                    "validation_plugin_nash_conv": plugin_gap,
                    "crossfit_br_gain": crossfit,
                    "crossfit_a_to_b": a_to_b,
                    "crossfit_b_to_a": b_to_a,
                }
            )

        mean_strategy = tuple(
            tuple(
                sum(profile[action][hand] for profile in strategies) / len(strategies)
                for hand in range(len(strategies[0][action]))
            )
            for action in range(len(ACTION_NAMES))
        )
        mean_plugin, expected = _stratified_nash_conv(deals, mean_strategy)
        mean_crossfit, mean_a_to_b, mean_b_to_a = _crossfit_stratified_br_gain(deals, mean_strategy)
        result = {
            "stacks_bb": dict(zip(SEATS, stacks)),
            "payouts": list(payouts),
            "ante_bb_per_player": ante,
            "profiles_by_seed": seed_profiles,
            "mean_strategy_profile": {
                "validation_plugin_nash_conv": mean_plugin,
                "crossfit_br_gain": mean_crossfit,
                "crossfit_a_to_b": mean_a_to_b,
                "crossfit_b_to_a": mean_b_to_a,
                "expected_icm": {seat: expected[i] for i, seat in enumerate(SEATS)},
            },
        }
        results.append(result)
        validation_by_key[(stacks, payouts)] = {
            "method": "stratified_crossfit",
            "samples_per_seat_hand_class": samples_per_class,
            "total_physical_deals": 3 * 169 * samples_per_class,
            "seed": seed,
            "mean_strategy_plugin_nash_conv": mean_plugin,
            "mean_strategy_crossfit_br_gain": mean_crossfit,
            "maximum_seed_crossfit_br_gain": max(profile["crossfit_br_gain"] for profile in seed_profiles),
        }
        print(
            f"[{index}/{len(summary['configurations'])}] "
            f"BTN/SB/BB={','.join(f'{value:g}' for value in stacks)} BB · "
            f"plugin={mean_plugin:.5f} · crossfit={mean_crossfit:.5f}",
            flush=True,
        )
        del deals, strategies, seed_profiles, mean_strategy

    audit = {
        "format": "lugon_3max_unequal_sweep_validation_v1",
        "validation_method": "balanced conditional deals; best-response choice and value use opposite halves",
        "validation_samples_per_seat_hand_class": samples_per_class,
        "total_deals_per_configuration": 3 * 169 * samples_per_class,
        "seed": seed,
        "configurations": results,
    }
    _write_json(output_path, audit)

    enriched_rows: list[str] = []
    for line_number, line in enumerate(dataset_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        features = row["features"]
        stacks = tuple(float(features["stacks_bb"][seat]) for seat in SEATS)
        payouts = tuple(float(value) for value in features["payouts"])
        key = (stacks, payouts)
        if key not in validation_by_key:
            raise ValueError(f"dataset line {line_number} has no matching validation configuration")
        row["label_provenance"]["independent_stratified_validation"] = validation_by_key[key]
        enriched_rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    if not enriched_rows:
        raise ValueError("training dataset is empty")
    temporary_dataset = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    temporary_dataset.write_text("\n".join(enriched_rows) + "\n", encoding="utf-8", newline="\n")
    temporary_dataset.replace(dataset_path)

    summary["independent_stratified_validation"] = {
        "path": str(output_path),
        "samples_per_seat_hand_class": samples_per_class,
        "seed": seed,
        "label_rows_enriched": len(enriched_rows),
    }
    _write_json(summary_path, summary)
    print(f"Validação gravada: {output_path} · etiquetas enriquecidas: {len(enriched_rows)}", flush=True)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_unequal_train_summary.json"))
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_unequal_train"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training/three_max_unequal_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/charts/three_max_unequal_train_validation.json"))
    parser.add_argument("--samples-per-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=12031)
    args = parser.parse_args(argv)
    if args.samples_per_class < 2:
        parser.error("samples-per-class must be at least 2 for cross-fit")
    if args.seed < 0 or not math.isfinite(args.seed):
        parser.error("seed must be a nonnegative integer")
    try:
        validate_unequal_sweep(
            summary_path=args.summary,
            chart_dir=args.chart_dir,
            dataset_path=args.dataset,
            output_path=args.output,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    except (ArithmeticError, OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
