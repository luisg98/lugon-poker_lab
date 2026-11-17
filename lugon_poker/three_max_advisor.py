"""Solver-backed and ML-model-backed single-hand advice for the 3-max push/fold tree."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cards import parse_cards
from .preflop import CLASS_INDEX
from .three_max import ThreeMaxSolution, solve_three_max_push_fold_stratified
from .three_max_model import UnequalStackThreeMaxFrequencyModel

DEFAULT_PAYOUTS = (0.5, 0.3, 0.2)
DEFAULT_SAMPLES_PER_HAND_CLASS = 250
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_TOLERANCE = 0.0075

_MODELS_DIR = Path(__file__).resolve().parents[1] / "data" / "models"
_DEFAULT_UNEQUAL_MODEL_CANDIDATES = (
    "three_max_unequal_50_30_20_interactions_v3.json",
    "three_max_unequal_50_30_20_interactions_v2.json",
    "three_max_unequal_50_30_20_linear.json",
)

SOLVER_NODES: dict[str, dict[str, str]] = {
    "btn_push": {"position": "BTN", "strategy": "btn_push", "action": "push"},
    "sb_push_after_btn_fold": {
        "position": "SB",
        "strategy": "sb_push_after_btn_fold",
        "action": "push",
    },
    "sb_call_vs_btn_shove": {
        "position": "SB",
        "strategy": "sb_call_vs_btn_shove",
        "action": "call",
    },
    "bb_call_vs_sb_shove": {
        "position": "BB",
        "strategy": "bb_call_vs_sb_shove",
        "action": "call",
    },
    "bb_call_vs_btn_shove": {
        "position": "BB",
        "strategy": "bb_call_vs_btn_shove",
        "action": "call",
    },
    "bb_call_multiway": {
        "position": "BB",
        "strategy": "bb_call_multiway",
        "action": "call",
    },
}

NODE_LABELS = {
    "btn_push": "BTN: shove or fold",
    "sb_push_after_btn_fold": "SB after BTN folds: shove or fold",
    "sb_call_vs_btn_shove": "SB vs BTN shove: call or fold",
    "bb_call_vs_sb_shove": "BB vs SB shove: call or fold",
    "bb_call_vs_btn_shove": "BB vs BTN shove: call or fold",
    "bb_call_multiway": "BB vs BTN shove and SB call: call or fold",
}


def _hand_class(cards: Sequence[str]) -> str:
    parsed = parse_cards(cards)
    if len(parsed) != 2:
        raise ValueError("The 3-max solver requires exactly two cards")
    first, second = parsed
    rank_chars = "23456789TJQKA"
    high, low = sorted((first.rank, second.rank), reverse=True)
    high_char, low_char = rank_chars[high - 2], rank_chars[low - 2]
    if high == low:
        return high_char + low_char
    return high_char + low_char + ("s" if first.suit == second.suit else "o")


@lru_cache(maxsize=1)
def load_default_unequal_model() -> UnequalStackThreeMaxFrequencyModel | None:
    """Load the best available unequal-stack 3-max frequency model if present on disk."""
    for filename in _DEFAULT_UNEQUAL_MODEL_CANDIDATES:
        candidate = _MODELS_DIR / filename
        if candidate.exists():
            try:
                return UnequalStackThreeMaxFrequencyModel.load(candidate)
            except (OSError, ValueError, KeyError):
                continue
    return None


def predict_hand_advice_from_model(
    *,
    hero_cards: Sequence[str],
    node: str,
    stacks_bb: Sequence[float],
    payouts: Sequence[float] = DEFAULT_PAYOUTS,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
    ante_bb: float = 0.0,
    model: UnequalStackThreeMaxFrequencyModel | None = None,
) -> dict[str, Any] | None:
    """Fast (<1ms) frequency prediction when the spot is inside the trained unequal model domain."""
    if node not in SOLVER_NODES or len(stacks_bb) != 3 or len(payouts) != 3:
        return None
    loaded = model or load_default_unequal_model()
    if loaded is None:
        return None

    stacks = tuple(round(float(value), 4) for value in stacks_bb)
    prizes = tuple(round(float(value), 6) for value in payouts)
    if prizes != loaded.payouts or ante_bb != 0.0 or small_blind_bb != 0.5 or big_blind_bb != 1.0:
        return None
    if any(
        val < low or val > high
        for val, low, high in zip(stacks, loaded.stack_min_by_seat_bb, loaded.stack_max_by_seat_bb)
    ):
        return None

    hand_class = _hand_class(hero_cards)
    definition = SOLVER_NODES[node]
    freqs = loaded.predict(
        hand_class,
        stacks,
        payouts=prizes,  # type: ignore[arg-type]
        ante_bb=ante_bb,
        small_blind_bb=small_blind_bb,
        big_blind_bb=big_blind_bb,
    )
    action_frequency = float(freqs[definition["strategy"]])
    return {
        "engine": "lugon_three_max_unequal_model",
        "status": "ready",
        "node": node,
        "node_label": NODE_LABELS[node],
        "position": definition["position"],
        "hand_class": hand_class,
        "solver_action": definition["action"],
        "action_frequency": action_frequency,
        "fold_frequency": 1.0 - action_frequency,
        "recommended_action": definition["action"] if action_frequency >= 0.5 else "fold",
        "stacks_bb": {seat: stacks[index] for index, seat in enumerate(("BTN", "SB", "BB"))},
        "payouts": list(prizes),
        "small_blind_bb": float(small_blind_bb),
        "big_blind_bb": float(big_blind_bb),
        "ante_bb_per_player": float(ante_bb),
        "sampling": {
            "method": "precomputed_ml_regression",
            "training_examples": loaded.training_examples,
            "interaction_terms": loaded.interaction_terms,
        },
        "convergence": {
            "converged_on_sample": True,
            "nash_conv_payout_units": None,
        },
        "assumptions": [
            "Model trained for a 3-max pre-flop push/fold tree (50/30/20 payouts, 0.5/1 BB blinds, no ante)",
            "Stacks are within the trained range (6–15 BB)",
            "Strategy uses 169 hand classes and assumes opponents play close to equilibrium",
        ],
    }


def infer_live_three_max_request(
    *,
    game_type: str,
    board: Sequence[str],
    hero_position: str,
    hero_stack: float,
    to_call: float,
    big_blind: float,
    opponents: Sequence[Mapping[str, Any]],
) -> dict[str, Any] | None:
    """Infer a 3-max push/fold request from a live/manual table snapshot when all seat stacks are known."""
    if game_type != "spin_and_go" or board or big_blind <= 0.0:
        return None
    hero_pos = str(hero_position or "").strip().upper()
    if hero_pos not in ("BTN", "SB", "BB"):
        return None

    seats_bb: dict[str, float] = {hero_pos: float(hero_stack) / float(big_blind)}
    opp_by_pos: dict[str, Mapping[str, Any]] = {}
    for opp in opponents:
        pos = str(opp.get("position") or "").strip().upper()
        stack_raw = opp.get("stack")
        if pos not in ("BTN", "SB", "BB") or pos == hero_pos or stack_raw is None:
            continue
        stack_bb = float(stack_raw) / float(big_blind)
        if stack_bb <= 0.0:
            continue
        seats_bb[pos] = stack_bb
        opp_by_pos[pos] = opp

    if set(seats_bb) != {"BTN", "SB", "BB"}:
        return None

    stacks_bb = (seats_bb["BTN"], seats_bb["SB"], seats_bb["BB"])
    hero_bb = seats_bb[hero_pos]
    to_call_bb = float(to_call) / float(big_blind)

    # Only trigger push/fold model automatically when stacks/action actually represent a push/fold spot.
    if to_call_bb <= 1e-6:
        if hero_bb > 15.0:
            return None
        if hero_pos == "BTN":
            node = "btn_push"
        elif hero_pos == "SB":
            node = "sb_push_after_btn_fold"
        else:
            return None
    else:
        max_opp_bb = max((seats_bb[pos] for pos in opp_by_pos), default=0.0)
        is_shove_facing = to_call_bb >= min(hero_bb, max_opp_bb) * 0.70 or to_call_bb >= 5.0
        if not is_shove_facing:
            return None
        if hero_pos == "SB":
            node = "sb_call_vs_btn_shove"
        elif hero_pos == "BB":
            btn_bet = float(opp_by_pos["BTN"].get("bet") or 0.0) if "BTN" in opp_by_pos else 0.0
            sb_bet = float(opp_by_pos["SB"].get("bet") or 0.0) if "SB" in opp_by_pos else 0.0
            if btn_bet > 0.0 and sb_bet > float(big_blind) * 0.6:
                node = "bb_call_multiway"
            elif btn_bet > 0.0 and sb_bet <= float(big_blind) * 0.6:
                node = "bb_call_vs_btn_shove"
            elif sb_bet > 0.0:
                node = "bb_call_vs_sb_shove"
            else:
                return None
        else:
            return None

    return {
        "node": node,
        "stacks_bb": stacks_bb,
        "payouts": DEFAULT_PAYOUTS,
        "small_blind_bb": 0.5,
        "big_blind_bb": 1.0,
        "ante_bb": 0.0,
    }


@lru_cache(maxsize=24)
def _solve_cached(
    stacks_bb: tuple[float, float, float],
    payouts: tuple[float, float, float],
    small_blind_bb: float,
    big_blind_bb: float,
    ante_bb: float,
    samples_per_hand_class: int,
    seed: int,
    max_iterations: int,
    tolerance: float,
) -> ThreeMaxSolution:
    return solve_three_max_push_fold_stratified(
        stacks_bb,
        payouts,
        small_blind_bb=small_blind_bb,
        big_blind_bb=big_blind_bb,
        ante_bb=ante_bb,
        samples_per_hand_class=samples_per_hand_class,
        seed=seed,
        max_iterations=max_iterations,
        tolerance=tolerance,
        check_interval=25,
    )


def solve_hand_advice(
    *,
    hero_cards: Sequence[str],
    node: str,
    stacks_bb: Sequence[float],
    payouts: Sequence[float] = DEFAULT_PAYOUTS,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
    ante_bb: float = 0.0,
    samples_per_hand_class: int = DEFAULT_SAMPLES_PER_HAND_CLASS,
    seed: int = 7,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
    prefer_model: bool = True,
) -> dict[str, Any]:
    """Solve or predict one 3-max ICM spot and return frequency plus diagnostics."""
    if node not in SOLVER_NODES:
        raise ValueError(f"Unknown solver node: {node!r}")
    if len(stacks_bb) != 3 or len(payouts) != 3:
        raise ValueError("The 3-max solver requires three stacks and three payouts")
    stacks = tuple(round(float(value), 4) for value in stacks_bb)
    prizes = tuple(round(float(value), 6) for value in payouts)
    if any(value <= 0.0 for value in stacks):
        raise ValueError("All three stacks must be positive and measured in BB")
    if not any(value > 0.0 for value in prizes):
        raise ValueError("At least one payout must be positive")

    if prefer_model:
        predicted = predict_hand_advice_from_model(
            hero_cards=hero_cards,
            node=node,
            stacks_bb=stacks,
            payouts=prizes,
            small_blind_bb=float(small_blind_bb),
            big_blind_bb=float(big_blind_bb),
            ante_bb=float(ante_bb),
        )
        if predicted is not None:
            return predicted

    hand_class = _hand_class(hero_cards)
    definition = SOLVER_NODES[node]
    solution = _solve_cached(
        stacks,  # type: ignore[arg-type]
        prizes,  # type: ignore[arg-type]
        float(small_blind_bb),
        float(big_blind_bb),
        float(ante_bb),
        int(samples_per_hand_class),
        int(seed),
        int(max_iterations),
        float(tolerance),
    )
    action_frequency = float(solution.strategy(definition["strategy"])[CLASS_INDEX[hand_class]])
    return {
        "engine": "lugon_three_max_icm_cfr_plus",
        "status": "ready",
        "node": node,
        "node_label": NODE_LABELS[node],
        "position": definition["position"],
        "hand_class": hand_class,
        "solver_action": definition["action"],
        "action_frequency": action_frequency,
        "fold_frequency": 1.0 - action_frequency,
        "recommended_action": definition["action"] if action_frequency >= 0.5 else "fold",
        "stacks_bb": {seat: stacks[index] for index, seat in enumerate(("BTN", "SB", "BB"))},
        "payouts": list(prizes),
        "small_blind_bb": float(small_blind_bb),
        "big_blind_bb": float(big_blind_bb),
        "ante_bb_per_player": float(ante_bb),
        "sampling": {
            "method": solution.sampling_method,
            "samples_per_hand_class": solution.samples_per_hand_class,
            "seed": solution.seed,
            "iterations": solution.iterations,
        },
        "convergence": {
            "converged_on_sample": solution.converged,
            "nash_conv_payout_units": solution.nash_conv,
            "expected_icm": {seat: solution.expected_icm[index] for index, seat in enumerate(("BTN", "SB", "BB"))},
        },
        "assumptions": [
            "Pre-flop push/fold tree; limps and post-flop decisions are not included",
            "ICM Malmuth-Harville; sem rake",
            "Strategy is based on 169 hand classes, not physical card combinations",
            "NashConv is estimated by Monte Carlo sampling, so the strategy is approximate",
            "The action is mixed when its frequency is between 0% and 100%",
        ],
    }
