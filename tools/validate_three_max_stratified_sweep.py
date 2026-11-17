"""Cross-fit validate a stratified 3-max chart grid on balanced deal samples."""

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
from validate_three_max_stratified import ACTION_NAMES, _load_strategy, _payout_slug


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def validate_sweep(
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
        raise ValueError("o resumo do sweep não está completo")
    settings = summary["settings"]
    if settings.get("sampling_mode") != "stratified":
        raise ValueError("o sweep tem de ter sido gerado com --sampling-mode stratified")
    configurations = summary["configurations"]
    results: list[dict[str, Any]] = []
    validation_by_key: dict[tuple[float, tuple[float, ...]], dict[str, Any]] = {}

    for index, configuration in enumerate(configurations, start=1):
        stack = float(configuration["stacks_bb"]["BTN"])
        payouts = [float(value) for value in configuration["payouts"]]
        ante = float(configuration["ante_bb_per_player"])
        forced = (ante, ante + 0.5, ante + 1.0)
        physical_samples = int(configuration["samples_per_seed"])
        chart_seeds = [int(value) for value in configuration["seeds"]]
        deals = _make_stratified_deals(
            samples_per_class,
            seed,
            (stack, stack, stack),
            forced,
            tuple(payouts),  # type: ignore[arg-type]
        )

        strategies = []
        seed_profiles = []
        ante_slug = f"{ante:g}".replace(".", "p")
        for chart_seed in chart_seeds:
            chart = chart_dir / (
                f"3max_{stack:g}bb_{_payout_slug(tuple(payouts))}_a{ante_slug}_"
                f"n{physical_samples}_seed{chart_seed}.json"
            )
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

        averaged = tuple(
            tuple(
                sum(profile[action][hand] for profile in strategies) / len(strategies)
                for hand in range(len(strategies[0][action]))
            )
            for action in range(len(ACTION_NAMES))
        )
        mean_plugin, expected = _stratified_nash_conv(deals, averaged)
        mean_crossfit, mean_a_to_b, mean_b_to_a = _crossfit_stratified_br_gain(deals, averaged)
        result = {
            "stacks_bb": {"BTN": stack, "SB": stack, "BB": stack},
            "payouts": payouts,
            "ante_bb_per_player": ante,
            "profiles_by_seed": seed_profiles,
            "mean_strategy_profile": {
                "validation_plugin_nash_conv": mean_plugin,
                "crossfit_br_gain": mean_crossfit,
                "crossfit_a_to_b": mean_a_to_b,
                "crossfit_b_to_a": mean_b_to_a,
                "expected_icm": {seat: expected[i] for i, seat in enumerate(("BTN", "SB", "BB"))},
            },
        }
        results.append(result)
        validation_by_key[(stack, tuple(payouts))] = {
            "method": "stratified_crossfit",
            "samples_per_seat_hand_class": samples_per_class,
            "total_physical_deals": 3 * 169 * samples_per_class,
            "seed": seed,
            "mean_strategy_plugin_nash_conv": mean_plugin,
            "mean_strategy_crossfit_br_gain": mean_crossfit,
            "maximum_seed_crossfit_br_gain": max(profile["crossfit_br_gain"] for profile in seed_profiles),
        }
        print(
            f"[{index}/{len(configurations)}] {stack:g} BB · {_payout_slug(tuple(payouts))} · "
            f"plugin={mean_plugin:.5f} · crossfit={mean_crossfit:.5f}",
            flush=True,
        )
        # Drop the million-deal working set before building the next configuration.
        del deals, strategies, seed_profiles, averaged

    audit = {
        "format": "lugon_3max_stratified_sweep_validation_v1",
        "validation_method": "balanced conditional deals; best-response choice and value use opposite halves",
        "validation_samples_per_seat_hand_class": samples_per_class,
        "total_deals_per_configuration": 3 * 169 * samples_per_class,
        "seed": seed,
        "configurations": results,
    }
    _write_json(output_path, audit)

    rows: list[str] = []
    for line in dataset_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        features = row["features"]
        key = (float(features["stacks_bb"]["BTN"]), tuple(float(v) for v in features["payouts"]))
        row["label_provenance"]["independent_stratified_validation"] = validation_by_key[key]
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    temporary_dataset = dataset_path.with_suffix(dataset_path.suffix + ".tmp")
    temporary_dataset.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    temporary_dataset.replace(dataset_path)

    summary["independent_stratified_validation"] = {
        "path": str(output_path),
        "samples_per_seat_hand_class": samples_per_class,
        "seed": seed,
        "label_rows_enriched": len(rows),
    }
    _write_json(summary_path, summary)
    print(f"Validação gravada: {output_path} · etiquetas enriquecidas: {len(rows)}", flush=True)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida uma grelha 3-max estratificada com cross-fit independente.")
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_stratified_3paid_summary.json"))
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_stratified_3paid"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training/three_max_push_fold_3paid_stratified.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/charts/three_max_stratified_3paid_validation.json"))
    parser.add_argument("--samples-per-class", type=int, default=500)
    parser.add_argument("--seed", type=int, default=12011)
    args = parser.parse_args(argv)
    if args.samples_per_class < 2:
        parser.error("samples-per-class tem de ser pelo menos 2 para cross-fit")
    if args.seed < 0 or not math.isfinite(args.seed):
        parser.error("seed tem de ser não negativo")
    try:
        validate_sweep(
            summary_path=args.summary,
            chart_dir=args.chart_dir,
            dataset_path=args.dataset,
            output_path=args.output,
            samples_per_class=args.samples_per_class,
            seed=args.seed,
        )
    except (ArithmeticError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
