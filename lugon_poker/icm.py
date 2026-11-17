"""Exact Malmuth-Harville Independent Chip Model (ICM) equity."""

from __future__ import annotations

from itertools import permutations
import math
from typing import Sequence


def icm_equity(stacks: Sequence[float], payouts: Sequence[float]) -> tuple[float, ...]:
    """Return each player's expected prize share under Malmuth-Harville ICM.

    First-place probability is proportional to stack size. Conditional on each
    finish, the same rule is applied to the remaining positive stacks. If two
    or more players have zero chips, the remaining prizes are split evenly over
    their possible finish orders, avoiding an arbitrary tie-break.
    """
    if isinstance(stacks, (str, bytes)) or isinstance(payouts, (str, bytes)):
        raise ValueError("stacks and payouts must be numeric sequences")
    stack_values = tuple(_finite_nonnegative("stack", value) for value in stacks)
    payout_values = tuple(_finite_nonnegative("payout", value) for value in payouts)
    if not stack_values or len(stack_values) != len(payout_values):
        raise ValueError("stacks and payouts must have the same nonzero length")
    if sum(stack_values) <= 0.0:
        raise ValueError("at least one player must have chips")
    if sum(payout_values) <= 0.0:
        raise ValueError("at least one payout must be positive")
    if any(payout_values[i] < payout_values[i + 1] for i in range(len(payout_values) - 1)):
        raise ValueError("payouts must be ordered from first place to last place")

    equities = [0.0] * len(stack_values)
    positive = tuple(i for i, stack in enumerate(stack_values) if stack > 0.0)
    busted = tuple(i for i, stack in enumerate(stack_values) if stack == 0.0)

    def assign_positive(remaining: tuple[int, ...], place: int, probability: float) -> None:
        if not remaining:
            if not busted:
                return
            orders = tuple(permutations(busted))
            order_probability = probability / len(orders)
            for order in orders:
                for offset, player in enumerate(order):
                    equities[player] += order_probability * payout_values[place + offset]
            return

        chips_left = sum(stack_values[i] for i in remaining)
        for player in remaining:
            finish_probability = stack_values[player] / chips_left
            equities[player] += probability * finish_probability * payout_values[place]
            rest = tuple(i for i in remaining if i != player)
            assign_positive(rest, place + 1, probability * finish_probability)

    assign_positive(positive, 0, 1.0)
    return tuple(equities)


def _finite_nonnegative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if number < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return number
