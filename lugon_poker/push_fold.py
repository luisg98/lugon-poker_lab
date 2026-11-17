"""Heads-up chip-EV push/fold equilibrium using the preflop class equity matrix.

This first solver is deliberately limited to equal starting stacks, no rake and a
class-level hand abstraction. The small blind (button) acts first; after a shove,
the big blind chooses call or fold. CFR+ returns average strategies, and the
exploitability report is the zero-sum Nash gap in big blinds.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .preflop import HAND_CLASSES, PreflopMatrix, combo_count

_TOTAL_COMBOS = 1326
_TIE_EPSILON = 1e-12


@dataclass(frozen=True)
class PushFoldSolution:
    """Average heads-up equilibrium strategies indexed by ``HAND_CLASSES``."""

    stack_bb: float
    small_blind_bb: float
    big_blind_bb: float
    ante_bb: float
    iterations: int
    nash_gap_bb: float
    converged: bool
    small_blind_push: tuple[float, ...]
    big_blind_call: tuple[float, ...]
    matrix_samples: int
    matrix_seed: int

    def __post_init__(self) -> None:
        if len(self.small_blind_push) != len(HAND_CLASSES):
            raise ValueError("small-blind strategy must contain one frequency per hand class")
        if len(self.big_blind_call) != len(HAND_CLASSES):
            raise ValueError("big-blind strategy must contain one frequency per hand class")

    def range_at_least(self, position: str, frequency: float = 0.5) -> tuple[str, ...]:
        """Return classes played at least ``frequency`` of the time from a position."""
        if not 0.0 <= frequency <= 1.0:
            raise ValueError("frequency must be in [0, 1]")
        normalized = position.strip().lower()
        if normalized in ("sb", "small_blind", "small blind", "button", "btn"):
            strategy = self.small_blind_push
        elif normalized in ("bb", "big_blind", "big blind"):
            strategy = self.big_blind_call
        else:
            raise ValueError("position must be SB/Button or BB")
        return tuple(hand for hand, rate in zip(HAND_CLASSES, strategy) if rate >= frequency)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete frequency labels for review or model-data generation."""
        return {
            "format": "lugon_hu_push_fold_nash_v1",
            "game": "heads_up_chip_ev",
            "assumptions": [
                "equal starting stacks",
                "no rake",
                "small blind/button acts first",
                "class-level ranges use combo-count chance weights",
                "card removal between the two private hand classes is ignored",
                "each ante_bb is posted by both players before action",
            ],
            "parameters": {
                "stack_bb": self.stack_bb,
                "small_blind_bb": self.small_blind_bb,
                "big_blind_bb": self.big_blind_bb,
                "ante_bb_per_player": self.ante_bb,
            },
            "convergence": {
            "algorithm": "cfr_plus",
                "iterations": self.iterations,
                "nash_gap_bb": self.nash_gap_bb,
                "converged": self.converged,
            },
            "equity_matrix": {"samples_per_matchup": self.matrix_samples, "seed": self.matrix_seed},
            "strategies": [
                {
                    "hand_class": hand,
                    "sb_push": push,
                    "sb_fold": 1.0 - push,
                    "bb_call": call,
                    "bb_fold": 1.0 - call,
                }
                for hand, push, call in zip(HAND_CLASSES, self.small_blind_push, self.big_blind_call)
            ],
        }


def _validate_number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    return number


def _best_small_blind_response(
    average_call: list[float],
    chance: tuple[float, ...],
    centered_payoffs: tuple[tuple[float, ...], ...],
    fold_ev: float,
    bb_fold_win: float,
) -> tuple[list[float], list[float]]:
    """Best response to BB's average call strategy; returns actions and EV deltas."""
    n = len(HAND_CLASSES)
    call_mass = [chance[j] * average_call[j] for j in range(n)]
    push: list[float] = []
    deltas: list[float] = []
    dead_money_gain = bb_fold_win - fold_ev
    for i in range(n):
        delta = dead_money_gain
        row = centered_payoffs[i]
        for j in range(n):
            delta += call_mass[j] * row[j]
        deltas.append(delta)
        if delta > _TIE_EPSILON:
            push.append(1.0)
        elif delta < -_TIE_EPSILON:
            push.append(0.0)
        else:
            push.append(0.5)
    return push, deltas


def _big_blind_call_advantages(
    average_push: list[float],
    chance: tuple[float, ...],
    centered_columns: tuple[tuple[float, ...], ...],
) -> list[float]:
    """BB's call-minus-fold counterfactual utility at each private-hand node."""
    n = len(HAND_CLASSES)
    push_mass = [chance[i] * average_push[i] for i in range(n)]
    advantages: list[float] = []
    for j in range(n):
        # BB gains (bb + ante - SB's called payoff) by calling instead of folding.
        advantage = 0.0
        column = centered_columns[j]
        for i in range(n):
            advantage -= push_mass[i] * column[i]
        advantages.append(advantage)
    return advantages


def _regret_matching(action_regret: float, other_regret: float) -> float:
    """Probability of the first action under regret matching+ at a binary node."""
    positive_action = max(0.0, action_regret)
    positive_other = max(0.0, other_regret)
    total = positive_action + positive_other
    return positive_action / total if total > _TIE_EPSILON else 0.5


def _nash_gap(
    average_push: list[float],
    average_call: list[float],
    chance: tuple[float, ...],
    centered_payoffs: tuple[tuple[float, ...], ...],
    fold_ev: float,
    bb_fold_win: float,
) -> float:
    """Return max-BR minus min-BR utility for the average strategies."""
    _, sb_deltas = _best_small_blind_response(
        average_call, chance, centered_payoffs, fold_ev, bb_fold_win
    )
    max_sb_ev = fold_ev + sum(chance[i] * max(0.0, sb_deltas[i]) for i in range(len(HAND_CLASSES)))

    push_mass = [chance[i] * average_push[i] for i in range(len(HAND_CLASSES))]
    total_push_mass = sum(push_mass)
    min_sb_ev = fold_ev * (1.0 - total_push_mass)
    fold_response_value = total_push_mass * bb_fold_win
    for j in range(len(HAND_CLASSES)):
        call_response_value = 0.0
        for i in range(len(HAND_CLASSES)):
            call_response_value += push_mass[i] * (
                bb_fold_win + centered_payoffs[i][j]
            )
        # BB chooses one response per private hand after observing a shove. It
        # cannot condition its action on the SB's hidden hand class.
        min_sb_ev += chance[j] * min(fold_response_value, call_response_value)
    return max(0.0, max_sb_ev - min_sb_ev)


def solve_push_fold_nash(
    matrix: PreflopMatrix,
    stack_bb: float,
    *,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
    ante_bb: float = 0.0,
    max_iterations: int = 25_000,
    tolerance_bb: float = 0.01,
    check_interval: int = 25,
) -> PushFoldSolution:
    """Solve the heads-up shove/fold/call chip-EV game with equal starting stacks.

    Chance weights come from the physical combo counts (6 per pair, 4 suited,
    12 offsuit). At showdown the stored matrix's class-to-class equity is used.
    The equilibrium is computed with counterfactual regret minimization (CFR+), and its average
    strategies are returned. ``nash_gap_bb`` is the difference between the
    small blind's best-response EV and the big blind's best-response EV against
    those averages; convergence is reported only when it is within tolerance.
    """
    stack = _validate_number("stack_bb", stack_bb, positive=True)
    sb = _validate_number("small_blind_bb", small_blind_bb, positive=True)
    bb = _validate_number("big_blind_bb", big_blind_bb, positive=True)
    ante = _validate_number("ante_bb", ante_bb)
    tol = _validate_number("tolerance_bb", tolerance_bb, positive=True)
    if ante < 0:
        raise ValueError("ante_bb cannot be negative")
    if sb >= bb:
        raise ValueError("small_blind_bb must be smaller than big_blind_bb")
    if stack <= bb + ante:
        raise ValueError("stack_bb must exceed the big blind plus ante")
    if not isinstance(max_iterations, int) or isinstance(max_iterations, bool) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")
    if not isinstance(check_interval, int) or isinstance(check_interval, bool) or check_interval < 1:
        raise ValueError("check_interval must be a positive integer")
    if len(matrix.values) != len(HAND_CLASSES) * (len(HAND_CLASSES) - 1) // 2:
        raise ValueError("equity matrix has an invalid number of matchup values")
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for value in matrix.values):
        raise ValueError("equity matrix values must be finite numbers in [0, 1]")

    n = len(HAND_CLASSES)
    chance = tuple(combo_count(hand) / _TOTAL_COMBOS for hand in HAND_CLASSES)
    bb_fold_win = bb + ante
    fold_ev = -(sb + ante)

    # Centering around the SB's reward when BB folds makes both best responses
    # simple dot products over the same payoff matrix.
    centered_rows: list[tuple[float, ...]] = []
    for hand_a in HAND_CLASSES:
        row = []
        for hand_b in HAND_CLASSES:
            called_ev = 2.0 * stack * matrix.equity(hand_a, hand_b) - stack
            row.append(called_ev - bb_fold_win)
        centered_rows.append(tuple(row))
    centered_payoffs = tuple(centered_rows)
    centered_columns = tuple(
        tuple(centered_payoffs[i][j] for i in range(n)) for j in range(n)
    )

    average_push = [0.0] * n
    average_call = [0.0] * n
    sum_push = [0.0] * n
    sum_call = [0.0] * n
    sb_push_regret = [0.0] * n
    sb_fold_regret = [0.0] * n
    bb_call_regret = [0.0] * n
    bb_fold_regret = [0.0] * n
    gap = math.inf
    converged = False
    completed = 0

    # Each private-hand class is an information set. Counterfactual regrets for
    # BB are weighted by the probability that SB shoves; averages are kept per
    # class so the output remains useful as a set of training labels.
    for iteration in range(1, max_iterations + 1):
        strategy_push = [
            _regret_matching(sb_push_regret[i], sb_fold_regret[i]) for i in range(n)
        ]
        strategy_call = [
            _regret_matching(bb_call_regret[j], bb_fold_regret[j]) for j in range(n)
        ]
        _, push_deltas = _best_small_blind_response(
            strategy_call, chance, centered_payoffs, fold_ev, bb_fold_win
        )
        call_deltas = _big_blind_call_advantages(strategy_push, chance, centered_columns)

        for i, (frequency, delta) in enumerate(zip(strategy_push, push_deltas)):
            sb_push_regret[i] = max(0.0, sb_push_regret[i] + (1.0 - frequency) * delta)
            sb_fold_regret[i] = max(0.0, sb_fold_regret[i] - frequency * delta)
            sum_push[i] += frequency
        for j, (frequency, delta) in enumerate(zip(strategy_call, call_deltas)):
            bb_call_regret[j] = max(0.0, bb_call_regret[j] + (1.0 - frequency) * delta)
            bb_fold_regret[j] = max(0.0, bb_fold_regret[j] - frequency * delta)
            sum_call[j] += frequency

        inverse_iteration = 1.0 / iteration
        average_push = [total * inverse_iteration for total in sum_push]
        average_call = [total * inverse_iteration for total in sum_call]
        completed = iteration

        if iteration == 1 or iteration % check_interval == 0 or iteration == max_iterations:
            gap = _nash_gap(
                average_push, average_call, chance, centered_payoffs, fold_ev, bb_fold_win
            )
            if gap <= tol:
                converged = True
                break

    return PushFoldSolution(
        stack_bb=stack,
        small_blind_bb=sb,
        big_blind_bb=bb,
        ante_bb=ante,
        iterations=completed,
        nash_gap_bb=gap,
        converged=converged,
        small_blind_push=tuple(average_push),
        big_blind_call=tuple(average_call),
        matrix_samples=matrix.samples,
        matrix_seed=matrix.seed,
    )
