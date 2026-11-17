"""Reproducible Monte Carlo equity estimation with exact river enumeration."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

from .cards import Card, best_hand
from .ranges import Combo
from .state import HandState


@dataclass(frozen=True)
class EquityResult:
    equity: float
    win_rate: float
    tie_rate: float
    loss_rate: float
    samples: int
    method: str
    seed: int | None
    attempts: int


def _card_from_code(code: int) -> Card:
    return Card(code // 4, "cdhs"[code % 4])


def _result(wins: int, ties: int, shares: float, samples: int, method: str, seed: int | None, attempts: int) -> EquityResult:
    return EquityResult(shares / samples, wins / samples, ties / samples, (samples - wins - ties) / samples, samples, method, seed, attempts)


def calculate_equity(
    state: HandState,
    iterations: int = 10_000,
    seed: int | None = 7,
    max_attempts: int | None = None,
) -> EquityResult:
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        raise ValueError("iterations must be a positive integer")
    known = {card.code for card in (*state.hero_cards, *state.board)}
    deck = [_card_from_code(code) for code in range(8, 60) if code not in known]
    missing_board = 5 - len(state.board)
    opponent_ranges = [opponent.hand_range for opponent in state.opponents]

    if missing_board == 0 and len(opponent_ranges) == 1:
        return _exact_river(state, deck, opponent_ranges[0], seed)

    rng = random.Random(seed)
    wins = ties = 0
    shares = 0.0
    samples = attempts = 0
    draw_count = 2 * len(opponent_ranges) + missing_board
    attempt_limit = max_attempts if max_attempts is not None else max(20_000, iterations * 500)

    while samples < iterations and attempts < attempt_limit:
        attempts += 1
        drawn = rng.sample(deck, draw_count)
        opponent_hands = [drawn[2 * i : 2 * i + 2] for i in range(len(opponent_ranges))]
        if any(tuple(sorted((hand[0].code, hand[1].code))) not in combo_range for hand, combo_range in zip(opponent_hands, opponent_ranges)):
            continue

        completed_board = (*state.board, *drawn[2 * len(opponent_ranges) :])
        hero_rank = best_hand((*state.hero_cards, *completed_board))
        opponent_ranks = [best_hand((*hand, *completed_board)) for hand in opponent_hands]
        best_rank = max(hero_rank, *opponent_ranks)
        winners = int(hero_rank == best_rank) + sum(rank == best_rank for rank in opponent_ranks)
        if hero_rank == best_rank:
            if winners == 1:
                wins += 1
            else:
                ties += 1
            shares += 1 / winners
        samples += 1

    if samples < iterations:
        raise RuntimeError(
            f"only {samples}/{iterations} valid deals found after {attempts} attempts; "
            "ranges may be too narrow or mutually incompatible. Use broader ranges or raise max_attempts."
        )
    return _result(wins, ties, shares, samples, "monte_carlo", seed, attempts)


def _exact_river(state: HandState, deck: list[Card], combo_range: frozenset[Combo], seed: int | None) -> EquityResult:
    wins = ties = samples = 0
    shares = 0.0
    hero_rank = best_hand((*state.hero_cards, *state.board))
    for hand in combinations(deck, 2):
        if tuple(sorted((hand[0].code, hand[1].code))) not in combo_range:
            continue
        opponent_rank = best_hand((*hand, *state.board))
        samples += 1
        if hero_rank > opponent_rank:
            wins += 1
            shares += 1.0
        elif hero_rank == opponent_rank:
            ties += 1
            shares += 0.5
    if samples == 0:
        raise ValueError("opponent range has no legal river combos")
    return _result(wins, ties, shares, samples, "exact_river", seed, samples)
