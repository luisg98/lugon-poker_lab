"""Inspect an unequal-stack prediction from the experimental 3-max model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker.preflop import parse_hand_class
from lugon_poker.three_max_model import UnequalStackThreeMaxFrequencyModel


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("data/models/three_max_unequal_50_30_20_linear.json"))
    parser.add_argument("--hand", required=True, help="classe pré-flop, ex.: AA, AKo ou 76s")
    parser.add_argument("--stacks-bb", required=True, help="BTN,SB,BB em BB, ex.: 8,10,12")
    args = parser.parse_args(argv)
    try:
        stacks = tuple(float(part.strip()) for part in args.stacks_bb.split(","))
        if len(stacks) != 3:
            raise ValueError("stacks-bb must contain BTN, SB and BB")
        model = UnequalStackThreeMaxFrequencyModel.load(args.model)
        predictions = model.predict(args.hand, stacks)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    result = {
        "model": "experimental_3max_unequal_stack_push_fold_frequency",
        "feature_model": "pairwise_stack_interactions" if model.interaction_terms else "additive_stack_slopes",
        "hand_class": parse_hand_class(args.hand),
        "stacks_bb": dict(zip(("BTN", "SB", "BB"), stacks)),
        "payouts": list(model.payouts),
        "scope": {
            "players": 3,
            "equal_stacks_only": False,
            "stack_range_by_seat_bb": {
                seat: [model.stack_min_by_seat_bb[index], model.stack_max_by_seat_bb[index]]
                for index, seat in enumerate(("BTN", "SB", "BB"))
            },
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
