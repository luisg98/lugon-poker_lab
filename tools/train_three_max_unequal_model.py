"""Train the experimental 3-max model with BTN/SB/BB stack features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.three_max_model import train_unequal_stack_three_max_frequency_model


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
    parser.add_argument("--dataset", type=Path, default=Path("data/training/three_max_unequal_train.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/models/three_max_unequal_50_30_20_linear.json"))
    parser.add_argument("--payouts", nargs=3, type=float, default=(0.5, 0.3, 0.2))
    parser.add_argument("--interactions", action="store_true", help="inclui interações BTN×SB, BTN×BB e SB×BB")
    args = parser.parse_args(argv)
    model = train_unequal_stack_three_max_frequency_model(
        _read_jsonl(args.dataset),
        payouts=tuple(args.payouts),  # type: ignore[arg-type]
        interaction_terms=args.interactions,
    )
    model.save(args.output)
    print(f"Modelo experimental gravado em {args.output}")
    print(f"Exemplos: {model.training_examples}; vetores de stacks: {len(model.training_stack_vectors_bb)}")
    print(json.dumps(model.validation, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
