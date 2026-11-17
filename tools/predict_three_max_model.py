"""Inspect one prediction from the experimental 3-max push/fold model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.three_max_model import ThreeMaxFrequencyModel
from lugon_poker.preflop import parse_hand_class


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/models/three_max_50_30_20_linear.json"))
    parser.add_argument("--hand", required=True, help="classe pré-flop, ex.: AA, AKo ou 76s")
    parser.add_argument("--stack-bb", required=True, type=float, help="stack efetivo igual dos três jogadores")
    args = parser.parse_args(argv)

    try:
        model = ThreeMaxFrequencyModel.load(args.model)
        predictions = model.predict(args.hand, args.stack_bb)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    result = {
        "model": "experimental_3max_push_fold_frequency",
        "hand_class": parse_hand_class(args.hand),
        "stack_bb": args.stack_bb,
        "payouts": list(model.payouts),
        "scope": {
            "players": 3,
            "equal_stacks_only": True,
            "training_stack_range_bb": [model.stack_min_bb, model.stack_max_bb],
            "ante_bb": 0.0,
            "blinds_bb": [0.5, 1.0],
            "game_tree": "preflop_push_fold",
        },
        "training_provenance": dict(model.training_provenance),
        "frequency_unit": "probability from 0 to 1",
        "action_frequencies": {action: round(rate, 6) for action, rate in predictions.items()},
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
