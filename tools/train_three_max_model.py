"""Train the narrow 3-max push/fold frequency model from validated labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.three_max_model import train_three_max_frequency_model


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_number} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"line {line_number} of {path} is not a JSON object")
            rows.append(row)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/training/three_max_push_fold_3paid_stratified_v2.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--payouts",
        nargs=3,
        type=float,
        default=(0.5, 0.3, 0.2),
        metavar=("FIRST", "SECOND", "THIRD"),
        help="estrutura de prémios da grelha de treino (default: 0.5 0.3 0.2)",
    )
    args = parser.parse_args(argv)

    payouts = tuple(args.payouts)
    model = train_three_max_frequency_model(_read_jsonl(args.dataset), payouts=payouts)
    if args.output is not None:
        output_path = args.output
    elif payouts == (0.5, 0.3, 0.2):
        output_path = Path("data/models/three_max_50_30_20_linear.json")
    else:
        payout_slug = "_".join(f"{value * 100:g}".replace(".", "p") for value in payouts)
        output_path = Path(f"data/models/three_max_{payout_slug}_linear.json")
    model.save(output_path)
    print(f"Modelo experimental gravado em {output_path}")
    print(f"Exemplos: {model.training_examples}; stacks: {list(model.training_stacks_bb)} BB")
    print(json.dumps(model.validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
