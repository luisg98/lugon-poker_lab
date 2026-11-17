"""Evaluate saved 3-max chart profiles on shared, independent chance samples."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.preflop import HAND_CLASSES
from lugon_poker.three_max import _make_deals, _sampled_nash_conv

ACTION_NAMES = (
    "btn_push",
    "sb_push_after_btn_fold",
    "sb_call_vs_btn_shove",
    "bb_call_vs_sb_shove",
    "bb_call_vs_btn_shove",
    "bb_call_multiway",
)


def _payout_slug(payouts: list[float]) -> str:
    return "_".join(f"{value:g}".replace(".", "p") for value in payouts)


def _chart_path(
    chart_dir: Path,
    stack_bb: float,
    payouts: list[float],
    ante_bb: float,
    samples: int,
    seed: int,
) -> Path:
    ante_slug = f"{ante_bb:g}".replace(".", "p")
    name = f"3max_{stack_bb:g}bb_{_payout_slug(payouts)}_a{ante_slug}_n{samples}_seed{seed}.json"
    return chart_dir / name


def _read_strategy(path: Path) -> tuple[tuple[float, ...], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "lugon_3max_push_fold_icm_v1":
        raise ValueError(f"formato de chart inesperado: {path}")
    rows = payload.get("strategies")
    if not isinstance(rows, list) or len(rows) != len(HAND_CLASSES):
        raise ValueError(f"chart sem as 169 classes de mãos: {path}")
    strategies = tuple(
        tuple(float(hand_row[action]) for hand_row in rows) for action in ACTION_NAMES
    )
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for row in strategies for value in row):
        raise ValueError(f"frequência de ação inválida: {path}")
    return strategies


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _enrich_dataset(path: Path, validation_by_key: dict[tuple[float, tuple[float, ...]], dict[str, Any]]) -> int:
    rows: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        features = row["features"]
        stack_bb = float(features["stacks_bb"]["BTN"])
        payouts = tuple(float(value) for value in features["payouts"])
        validation = validation_by_key[(stack_bb, payouts)]
        row["label_provenance"]["independent_validation"] = validation
        rows.append(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(path)
    return len(rows)


def validate_sweep(
    *,
    summary_path: Path,
    chart_dir: Path,
    dataset_path: Path,
    output_path: Path,
    validation_samples: int,
    validation_seed: int,
    tolerance: float,
) -> dict[str, Any]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "complete":
        raise ValueError("o resumo do sweep não está completo")
    settings = summary["settings"]
    train_samples = int(settings["samples_per_seed"])
    ante = float(settings["ante_bb_per_player"])
    results: list[dict[str, Any]] = []
    validation_by_key: dict[tuple[float, tuple[float, ...]], dict[str, Any]] = {}

    for index, configuration in enumerate(summary["configurations"], start=1):
        stack = float(configuration["stacks_bb"]["BTN"])
        payouts = [float(value) for value in configuration["payouts"]]
        stacks = (stack, stack, stack)
        forced = (ante, ante + 0.5, ante + 1.0)
        deals = _make_deals(
            validation_samples,
            validation_seed,
            stacks,
            forced,
            tuple(payouts),  # type: ignore[arg-type]
        )
        profiles: list[dict[str, Any]] = []
        strategy_profiles: list[tuple[tuple[float, ...], ...]] = []
        for training_result in configuration["convergence_by_seed"]:
            seed = int(training_result["seed"])
            chart = _chart_path(chart_dir, stack, payouts, ante, train_samples, seed)
            strategies = _read_strategy(chart)
            strategy_profiles.append(strategies)
            holdout_gap, expected_icm = _sampled_nash_conv(deals, strategies)
            profiles.append(
                {
                    "seed": seed,
                    "training_nash_conv": float(training_result["nash_conv"]),
                    "holdout_nash_conv": holdout_gap,
                    "within_tolerance_on_holdout": holdout_gap <= tolerance,
                    "expected_icm_on_holdout": {
                        seat: expected_icm[seat_index]
                        for seat_index, seat in enumerate(("BTN", "SB", "BB"))
                    },
                    "chart": str(chart),
                }
            )

        averaged = tuple(
            tuple(
                sum(profile[action_index][hand_index] for profile in strategy_profiles)
                / len(strategy_profiles)
                for hand_index in range(len(HAND_CLASSES))
            )
            for action_index in range(len(ACTION_NAMES))
        )
        average_gap, average_expected_icm = _sampled_nash_conv(deals, averaged)
        result = {
            "stacks_bb": {"BTN": stack, "SB": stack, "BB": stack},
            "payouts": payouts,
            "validation_samples": validation_samples,
            "validation_seed": validation_seed,
            "profiles_by_training_seed": profiles,
            "mean_strategy_profile": {
                "holdout_nash_conv": average_gap,
                "within_tolerance_on_holdout": average_gap <= tolerance,
                "expected_icm_on_holdout": {
                    seat: average_expected_icm[seat_index]
                    for seat_index, seat in enumerate(("BTN", "SB", "BB"))
                },
            },
        }
        results.append(result)
        validation_by_key[(stack, tuple(payouts))] = {
            "samples": validation_samples,
            "seed": validation_seed,
            "mean_strategy_holdout_nash_conv": average_gap,
            "mean_strategy_within_tolerance": average_gap <= tolerance,
            "maximum_seed_profile_holdout_nash_conv": max(
                profile["holdout_nash_conv"] for profile in profiles
            ),
        }
        print(
            f"[{index}/{len(summary['configurations'])}] {stack:g} BB · "
            f"{_payout_slug(payouts)} · holdout mean-profile NashConv={average_gap:.5f} · "
            f"seed max={max(profile['holdout_nash_conv'] for profile in profiles):.5f}",
            flush=True,
        )

    maximum_holdout = max(
        profile["holdout_nash_conv"]
        for result in results
        for profile in result["profiles_by_training_seed"]
    )
    audit = {
        "format": "lugon_3max_holdout_validation_v1",
        "validation_method": "shared independent physical-card deal sample per configuration",
        "tolerance": tolerance,
        "validation_samples_per_configuration": validation_samples,
        "validation_seed": validation_seed,
        "maximum_seed_profile_holdout_nash_conv": maximum_holdout,
        "all_mean_profiles_within_tolerance": all(
            result["mean_strategy_profile"]["within_tolerance_on_holdout"]
            for result in results
        ),
        "configurations": results,
    }
    _write_json(output_path, audit)
    row_count = _enrich_dataset(dataset_path, validation_by_key)
    summary["holdout_validation"] = {
        "path": str(output_path),
        "validation_samples_per_configuration": validation_samples,
        "validation_seed": validation_seed,
        "maximum_seed_profile_holdout_nash_conv": maximum_holdout,
        "all_mean_profiles_within_tolerance": audit["all_mean_profiles_within_tolerance"],
        "training_rows_enriched": row_count,
    }
    _write_json(summary_path, summary)
    print(f"Validação independente: {output_path}", flush=True)
    print(f"Dataset atualizado: {dataset_path} ({row_count} linhas)", flush=True)
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Valida charts 3-max numa amostra independente partilhada.")
    parser.add_argument("--summary", type=Path, default=Path("data/charts/three_max_sweep_summary.json"))
    parser.add_argument("--chart-dir", type=Path, default=Path("data/charts/three_max_sweep"))
    parser.add_argument("--dataset", type=Path, default=Path("data/training/three_max_push_fold_labels.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/charts/three_max_holdout_validation.json"))
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=12011)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args(argv)
    try:
        if args.samples < 1:
            raise ValueError("samples tem de ser positivo")
        if args.seed < 0:
            raise ValueError("seed tem de ser não negativo")
        if args.tolerance <= 0 or not math.isfinite(args.tolerance):
            raise ValueError("tolerance tem de ser positivo e finito")
        validate_sweep(
            summary_path=args.summary,
            chart_dir=args.chart_dir,
            dataset_path=args.dataset,
            output_path=args.output,
            validation_samples=args.samples,
            validation_seed=args.seed,
            tolerance=args.tolerance,
        )
    except (ArithmeticError, OSError, ValueError, KeyError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
