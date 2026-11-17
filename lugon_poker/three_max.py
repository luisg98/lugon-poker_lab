"""Sampled three-player push/fold solver with Malmuth-Harville ICM utilities.

The chance model deals physical cards, so blockers and three-way showdowns are
represented directly. Strategies are abstracted to the 169 starting-hand
classes. The solver uses simultaneous regret matching+ on a fixed Monte Carlo
chance sample; in a three-player game this is an approximate search procedure,
not a guarantee of Nash convergence. The reported NashConv is measured against
the sampled game and the returned average strategy profile.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Sequence

from .cards import Card, RANKS, SUITS, evaluate
from .icm import icm_equity
from .preflop import CLASS_INDEX, HAND_CLASSES, class_combos, combo_count

_DECK = tuple(Card(rank, suit) for rank in range(2, 15) for suit in SUITS)
_LEAF_NAMES = (
    "walk",
    "sb_shove_bb_fold",
    "sb_shove_bb_call",
    "btn_shove_sb_fold_bb_fold",
    "btn_shove_sb_fold_bb_call",
    "btn_shove_sb_call_bb_fold",
    "btn_shove_sb_call_bb_call",
)
_STRATEGY_NAMES = (
    "btn_push",
    "sb_push_after_btn_fold",
    "sb_call_vs_btn_shove",
    "bb_call_vs_sb_shove",
    "bb_call_vs_btn_shove",
    "bb_call_multiway",
)
_ACTION_DEFINITIONS = {
    "btn_push": "BTN shoves; alternative action: fold",
    "sb_push_after_btn_fold": "SB open-shoves after BTN folds; alternative action: fold",
    "sb_call_vs_btn_shove": "SB calls BTN's shove; alternative action: fold",
    "bb_call_vs_sb_shove": "BB calls SB's shove after BTN folds; alternative action: fold",
    "bb_call_vs_btn_shove": "BB calls BTN's shove when SB folds; alternative action: fold",
    "bb_call_multiway": "BB calls BTN's shove after SB has called; alternative action: fold",
}


@dataclass(frozen=True)
class _Deal:
    classes: tuple[int, int, int]
    ranks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    # Each leaf stores ICM equities in BTN, SB, BB order.
    utilities: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]


@dataclass(frozen=True)
class ThreeMaxSolution:
    """Average 3-max strategies plus sample-based convergence diagnostics."""

    stacks_bb: tuple[float, float, float]
    payouts: tuple[float, float, float]
    small_blind_bb: float
    big_blind_bb: float
    ante_bb: float
    samples: int
    seed: int
    iterations: int
    nash_conv: float
    converged: bool
    strategies: tuple[tuple[float, ...], ...]
    expected_icm: tuple[float, float, float]
    sampling_method: str = "fixed_deal_sample"
    samples_per_hand_class: int | None = None

    def __post_init__(self) -> None:
        if len(self.strategies) != len(_STRATEGY_NAMES):
            raise ValueError("three-max solution must contain six strategy rows")
        if any(len(row) != len(HAND_CLASSES) for row in self.strategies):
            raise ValueError("each strategy row must contain 169 hand classes")

    def strategy(self, name: str) -> tuple[float, ...]:
        """Return an action-frequency row by its documented strategy name."""
        try:
            return self.strategies[_STRATEGY_NAMES.index(name)]
        except ValueError as exc:
            raise ValueError(f"unknown strategy {name!r}; expected one of {_STRATEGY_NAMES}") from exc

    def range_at_least(self, name: str, frequency: float = 0.5) -> tuple[str, ...]:
        if not 0.0 <= frequency <= 1.0:
            raise ValueError("frequency must be in [0, 1]")
        values = self.strategy(name)
        return tuple(hand for hand, rate in zip(HAND_CLASSES, values) if rate >= frequency)

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete action frequencies for charts and review."""
        return {
            "format": "lugon_3max_push_fold_icm_v1",
            "game": "three_player_push_fold_icm",
            "players": ["BTN", "SB", "BB"],
            "assumptions": [
                "preflop push/fold tree; no limps or postflop decisions",
                "physical card deals preserve blockers and multiway outcomes",
                "strategies are shared by the 169 starting-hand classes",
                "Malmuth-Harville ICM prizes; no rake",
                "Monte Carlo chance deals using the reported sampling method and seed",
                "simultaneous regret matching+ is approximate in a three-player game",
                "NashConv is measured on the sampled game, not the full card tree",
                "uncalled chips are returned before side pots are awarded",
            ],
            "parameters": {
                "stacks_bb": {"BTN": self.stacks_bb[0], "SB": self.stacks_bb[1], "BB": self.stacks_bb[2]},
                "payouts": list(self.payouts),
                "small_blind_bb": self.small_blind_bb,
                "big_blind_bb": self.big_blind_bb,
                "ante_bb_per_player": self.ante_bb,
            },
            "sampling": {
                "deals": self.samples,
                "seed": self.seed,
                "method": self.sampling_method,
                "samples_per_hand_class": self.samples_per_hand_class,
            },
            "convergence": {
                "algorithm": "simultaneous_regret_matching_plus",
                "iterations": self.iterations,
                "nash_conv_payout_units": self.nash_conv,
                "converged_on_sample": self.converged,
            },
            "action_definitions": _ACTION_DEFINITIONS,
            "expected_icm": {"BTN": self.expected_icm[0], "SB": self.expected_icm[1], "BB": self.expected_icm[2]},
            "strategies": [
                {
                    "hand_class": hand,
                    **{name: self.strategies[row][index] for row, name in enumerate(_STRATEGY_NAMES)},
                }
                for index, hand in enumerate(HAND_CLASSES)
            ],
        }


def _number(name: str, value: float, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if result < 0.0 or (positive and result == 0.0):
        requirement = "positive" if positive else "nonnegative"
        raise ValueError(f"{name} must be {requirement}")
    return result


def _class_index(cards: tuple[Card, Card]) -> int:
    first, second = cards
    if first.rank == second.rank:
        name = RANKS[first.rank - 2] + RANKS[second.rank - 2]
    else:
        high, low = sorted((first.rank, second.rank), reverse=True)
        suffix = "s" if first.suit == second.suit else "o"
        name = RANKS[high - 2] + RANKS[low - 2] + suffix
    return CLASS_INDEX[name]


def _settle(
    stacks: tuple[float, float, float],
    commitments: tuple[float, float, float],
    active: tuple[int, ...],
    ranks: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    payouts: tuple[float, float, float],
) -> tuple[float, float, float]:
    """Settle a terminal state, return uncalled excess, then convert stacks to ICM."""
    committed = list(commitments)
    if len(active) > 1:
        # Folded chips are dead money. They cannot make an active player's
        # shove matched, so compare active contributions only when refunding.
        ordered = sorted((committed[i] for i in active), reverse=True)
        if ordered[0] > ordered[1]:
            largest = next(i for i in active if committed[i] == ordered[0])
            committed[largest] = ordered[1]

    final_stacks = [stacks[i] - committed[i] for i in range(3)]
    if len(active) == 1:
        winner = active[0]
        final_stacks[winner] += sum(committed)
    else:
        folded_dead_money = sum(committed[i] for i in range(3) if i not in active)
        levels = sorted({committed[i] for i in active if committed[i] > 0.0})
        previous = 0.0
        for level_index, level in enumerate(levels):
            contributors = [i for i in active if committed[i] >= level]
            pot = (level - previous) * len(contributors)
            if level_index == 0:
                # All folded contributions go into the main pot, where every
                # live player is eligible, even if a folded blind exceeded a
                # short stack's total contribution.
                pot += folded_dead_money
            eligible = contributors
            best_rank = max(ranks[i] for i in eligible)
            winners = [i for i in eligible if ranks[i] == best_rank]
            share = pot / len(winners)
            for winner in winners:
                final_stacks[winner] += share
            previous = level

    # Small floating-point noise can produce -1e-15 after pot splitting.
    final = tuple(0.0 if -1e-10 < value < 0.0 else value for value in final_stacks)
    if any(value < 0.0 for value in final):
        raise ArithmeticError("terminal settlement produced a negative stack")
    return icm_equity(final, payouts)


def _make_deal(
    hands: tuple[tuple[Card, Card], tuple[Card, Card], tuple[Card, Card]],
    board: tuple[Card, ...],
    stacks: tuple[float, float, float],
    forced: tuple[float, float, float],
    payouts: tuple[float, float, float],
) -> _Deal:
    classes = tuple(_class_index(hand) for hand in hands)
    ranks = tuple(evaluate((*hand, *board)) for hand in hands)

    def utility(active: tuple[int, ...], committed: tuple[float, float, float]) -> tuple[float, float, float]:
        return _settle(stacks, committed, active, ranks, payouts)

    # Fold terminals only commit the antes and blinds: an uncalled shove is returned.
    walk = utility((2,), forced)
    sb_fold = utility((1,), forced)
    btn_fold = utility((0,), forced)

    sb_vs_bb = (
        forced[0],
        stacks[1],
        min(stacks[2], stacks[1]),
    )
    btn_vs_bb = (
        stacks[0],
        forced[1],
        min(stacks[2], stacks[0]),
    )
    btn_vs_sb = (
        stacks[0],
        min(stacks[1], stacks[0]),
        forced[2],
    )
    three_way = (
        stacks[0],
        min(stacks[1], stacks[0]),
        min(stacks[2], stacks[0]),
    )
    utilities = (
        walk,
        sb_fold,
        utility((1, 2), sb_vs_bb),
        btn_fold,
        utility((0, 2), btn_vs_bb),
        utility((0, 1), btn_vs_sb),
        utility((0, 1, 2), three_way),
    )
    return _Deal(classes, ranks, utilities)


def _make_deals(
    samples: int,
    seed: int,
    stacks: tuple[float, float, float],
    forced: tuple[float, float, float],
    payouts: tuple[float, float, float],
) -> tuple[_Deal, ...]:
    rng = random.Random(seed)
    deals: list[_Deal] = []
    for _ in range(samples):
        drawn = rng.sample(_DECK, 11)
        hands = ((drawn[0], drawn[1]), (drawn[2], drawn[3]), (drawn[4], drawn[5]))
        deals.append(_make_deal(hands, drawn[6:], stacks, forced, payouts))
    return tuple(deals)


def _make_stratified_deals(
    samples_per_hand_class: int,
    seed: int,
    stacks: tuple[float, float, float],
    forced: tuple[float, float, float],
    payouts: tuple[float, float, float],
) -> tuple[tuple[tuple[_Deal, ...], ...], ...]:
    """Sample each seat's private hand class equally, conditional on that class.

    Each row is a seat, each item in the row is one of the 169 classes, and
    each class contains ``samples_per_hand_class`` physical deals. Opponent
    hands and board cards are drawn uniformly from cards remaining after the
    conditioned player's physical combo has been selected.
    """
    rng = random.Random(seed)
    conditioned: list[tuple[tuple[_Deal, ...], ...]] = []
    class_combos_by_index = tuple(class_combos(hand_class) for hand_class in HAND_CLASSES)
    for conditioned_seat in range(3):
        seat_rows: list[tuple[_Deal, ...]] = []
        for hand_class_index, combos in enumerate(class_combos_by_index):
            deals: list[_Deal] = []
            for _ in range(samples_per_hand_class):
                private = rng.choice(combos)
                private_codes = {card.code for card in private}
                remaining = tuple(card for card in _DECK if card.code not in private_codes)
                drawn = rng.sample(remaining, 9)
                hands: list[tuple[Card, Card] | None] = [None, None, None]
                hands[conditioned_seat] = private
                cursor = 0
                for seat in range(3):
                    if seat == conditioned_seat:
                        continue
                    hands[seat] = (drawn[cursor], drawn[cursor + 1])
                    cursor += 2
                completed_hands = (hands[0], hands[1], hands[2])
                # The loop fills all seats; the cast-free assertion documents the invariant.
                if any(hand is None for hand in completed_hands):
                    raise AssertionError("conditioned deal is missing a player's cards")
                deals.append(
                    _make_deal(
                        completed_hands,  # type: ignore[arg-type]
                        drawn[4:],
                        stacks,
                        forced,
                        payouts,
                    )
                )
            if any(deal.classes[conditioned_seat] != hand_class_index for deal in deals):
                raise AssertionError("stratified deals do not match the conditioned hand class")
            seat_rows.append(tuple(deals))
        conditioned.append(tuple(seat_rows))
    return tuple(conditioned)


def _mix(
    first: tuple[float, float, float],
    second: tuple[float, float, float],
    probability: float,
) -> tuple[float, float, float]:
    other = 1.0 - probability
    return (
        other * first[0] + probability * second[0],
        other * first[1] + probability * second[1],
        other * first[2] + probability * second[2],
    )


def _regret_probability(first: float, second: float) -> float:
    a, b = max(0.0, first), max(0.0, second)
    return a / (a + b) if a + b > 1e-15 else 0.5


def _sampled_nash_conv(
    deals: tuple[_Deal, ...],
    strategies: tuple[tuple[float, ...], ...],
) -> tuple[float, tuple[float, float, float]]:
    """Compute unilateral best-response gains and profile ICM on the fixed sample."""
    n = len(HAND_CLASSES)
    fold_values = [[0.0] * n for _ in _STRATEGY_NAMES]
    action_values = [[0.0] * n for _ in _STRATEGY_NAMES]
    expected = [0.0, 0.0, 0.0]

    for deal in deals:
        btn, sb, bb = deal.classes
        p = strategies[0][btn]
        q = strategies[1][sb]
        r = strategies[2][sb]
        t = strategies[3][bb]
        s = strategies[4][bb]
        u = strategies[5][bb]
        leaf = deal.utilities
        u_fold_btn = _mix(leaf[0], _mix(leaf[1], leaf[2], t), q)
        u_push_btn = _mix(_mix(leaf[3], leaf[4], s), _mix(leaf[5], leaf[6], u), r)

        # BTN acts at the root.
        fold_values[0][btn] += u_fold_btn[0]
        action_values[0][btn] += u_push_btn[0]
        # SB's two mutually exclusive information sets.
        reach_sb_open = 1.0 - p
        fold_values[1][sb] += reach_sb_open * leaf[0][1]
        action_values[1][sb] += reach_sb_open * _mix(leaf[1], leaf[2], t)[1]
        fold_values[2][sb] += p * _mix(leaf[3], leaf[4], s)[1]
        action_values[2][sb] += p * _mix(leaf[5], leaf[6], u)[1]
        # BB's three response information sets.
        fold_values[3][bb] += (1.0 - p) * q * leaf[1][2]
        action_values[3][bb] += (1.0 - p) * q * leaf[2][2]
        fold_values[4][bb] += p * (1.0 - r) * leaf[3][2]
        action_values[4][bb] += p * (1.0 - r) * leaf[4][2]
        fold_values[5][bb] += p * r * leaf[5][2]
        action_values[5][bb] += p * r * leaf[6][2]

        for seat in range(3):
            expected[seat] += (1.0 - p) * u_fold_btn[seat] + p * u_push_btn[seat]

    gap = 0.0
    for group, row in enumerate(strategies):
        for hand_index, action_probability in enumerate(row):
            fold_value = fold_values[group][hand_index]
            action_value = action_values[group][hand_index]
            profile_value = (1.0 - action_probability) * fold_value + action_probability * action_value
            gap += max(fold_value, action_value) - profile_value
    scale = 1.0 / len(deals)
    return max(0.0, gap * scale), tuple(value * scale for value in expected)


def _stratified_action_values(
    conditioned_deals: tuple[tuple[tuple[_Deal, ...], ...], ...],
    strategies: tuple[tuple[float, ...], ...],
    *,
    start: int = 0,
    stop: int | None = None,
) -> tuple[list[list[float]], list[list[float]], tuple[float, float, float]]:
    """Return weighted fold/action values for a slice of balanced samples."""
    n = len(HAND_CLASSES)
    fold_values = [[0.0] * n for _ in _STRATEGY_NAMES]
    action_values = [[0.0] * n for _ in _STRATEGY_NAMES]
    expected = [0.0, 0.0, 0.0]

    for hand_index, hand_class in enumerate(HAND_CLASSES):
        chance_weight = combo_count(hand_class) / 1326.0
        btn_deals = conditioned_deals[0][hand_index]
        sb_deals = conditioned_deals[1][hand_index]
        bb_deals = conditioned_deals[2][hand_index]
        end = len(btn_deals) if stop is None else stop
        btn_deals = btn_deals[start:end]
        sb_deals = sb_deals[start:end]
        bb_deals = bb_deals[start:end]
        if not btn_deals or len(btn_deals) != len(sb_deals) or len(btn_deals) != len(bb_deals):
            raise ValueError("stratified chance data must have equal nonzero samples per hand class")
        scale = chance_weight / len(btn_deals)

        for deal in btn_deals:
            btn, sb, bb = deal.classes
            p, q, r = strategies[0][btn], strategies[1][sb], strategies[2][sb]
            t, s, u = strategies[3][bb], strategies[4][bb], strategies[5][bb]
            leaf = deal.utilities
            u_fold = _mix(leaf[0], _mix(leaf[1], leaf[2], t), q)
            u_push = _mix(_mix(leaf[3], leaf[4], s), _mix(leaf[5], leaf[6], u), r)
            fold_values[0][hand_index] += scale * u_fold[0]
            action_values[0][hand_index] += scale * u_push[0]
            profile = _mix(u_fold, u_push, p)
            for seat in range(3):
                expected[seat] += scale * profile[seat]

        for deal in sb_deals:
            btn, sb, bb = deal.classes
            p = strategies[0][btn]
            q = strategies[1][hand_index]
            r = strategies[2][hand_index]
            t, s, u = strategies[3][bb], strategies[4][bb], strategies[5][bb]
            leaf = deal.utilities
            reach_open = 1.0 - p
            reach_call = p
            fold_values[1][hand_index] += scale * reach_open * leaf[0][1]
            action_values[1][hand_index] += scale * reach_open * _mix(leaf[1], leaf[2], t)[1]
            fold_values[2][hand_index] += scale * reach_call * _mix(leaf[3], leaf[4], s)[1]
            action_values[2][hand_index] += scale * reach_call * _mix(leaf[5], leaf[6], u)[1]

        for deal in bb_deals:
            btn, sb, bb = deal.classes
            p = strategies[0][btn]
            q, r = strategies[1][sb], strategies[2][sb]
            t = strategies[3][hand_index]
            s = strategies[4][hand_index]
            u = strategies[5][hand_index]
            leaf = deal.utilities
            fold_values[3][hand_index] += scale * (1.0 - p) * q * leaf[1][2]
            action_values[3][hand_index] += scale * (1.0 - p) * q * leaf[2][2]
            fold_values[4][hand_index] += scale * p * (1.0 - r) * leaf[3][2]
            action_values[4][hand_index] += scale * p * (1.0 - r) * leaf[4][2]
            fold_values[5][hand_index] += scale * p * r * leaf[5][2]
            action_values[5][hand_index] += scale * p * r * leaf[6][2]

    return fold_values, action_values, tuple(expected)


def _stratified_nash_conv(
    conditioned_deals: tuple[tuple[tuple[_Deal, ...], ...], ...],
    strategies: tuple[tuple[float, ...], ...],
) -> tuple[float, tuple[float, float, float]]:
    """Estimate NashConv with a balanced conditional sample for every hand class.

    Best-response gains are averaged within the acting player's hand class and
    then weighted by its exact physical-combo probability. This avoids the
    random under-representation of rare classes in a flat deal sample.
    """
    fold_values, action_values, expected = _stratified_action_values(
        conditioned_deals, strategies
    )

    gap = 0.0
    for group, row in enumerate(strategies):
        for hand_index, action_probability in enumerate(row):
            fold_value = fold_values[group][hand_index]
            action_value = action_values[group][hand_index]
            profile_value = (1.0 - action_probability) * fold_value + action_probability * action_value
            gap += max(fold_value, action_value) - profile_value
    return max(0.0, gap), expected


def _crossfit_stratified_br_gain(
    conditioned_deals: tuple[tuple[tuple[_Deal, ...], ...], ...],
    strategies: tuple[tuple[float, ...], ...],
) -> tuple[float, float, float]:
    """Select best responses on one half and value them on the other half.

    The ordinary plug-in NashConv maximizes noisy action values on the same
    sample, which biases it upward. This diagnostic reports both directional
    cross-fitted gains so sample noise is visible instead of hidden.
    """
    sample_count = len(conditioned_deals[0][0])
    midpoint = sample_count // 2
    if midpoint < 1:
        raise ValueError("cross-fit validation needs at least two samples per class")
    fold_a, action_a, _ = _stratified_action_values(
        conditioned_deals, strategies, start=0, stop=midpoint
    )
    fold_b, action_b, _ = _stratified_action_values(
        conditioned_deals, strategies, start=midpoint, stop=sample_count
    )
    gain_a_to_b = 0.0
    gain_b_to_a = 0.0
    for group, row in enumerate(strategies):
        for hand_index, action_probability in enumerate(row):
            profile_a = (1.0 - action_probability) * fold_a[group][hand_index] + action_probability * action_a[group][hand_index]
            profile_b = (1.0 - action_probability) * fold_b[group][hand_index] + action_probability * action_b[group][hand_index]
            if action_a[group][hand_index] > fold_a[group][hand_index]:
                gain_a_to_b += action_b[group][hand_index] - profile_b
            else:
                gain_a_to_b += fold_b[group][hand_index] - profile_b
            if action_b[group][hand_index] > fold_b[group][hand_index]:
                gain_b_to_a += action_a[group][hand_index] - profile_a
            else:
                gain_b_to_a += fold_a[group][hand_index] - profile_a
    return (gain_a_to_b + gain_b_to_a) / 2.0, gain_a_to_b, gain_b_to_a


def solve_three_max_push_fold(
    stacks_bb: Sequence[float],
    payouts: Sequence[float],
    *,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
    ante_bb: float = 0.0,
    samples: int = 20_000,
    seed: int = 7,
    max_iterations: int = 1_000,
    tolerance: float = 0.005,
    check_interval: int = 25,
) -> ThreeMaxSolution:
    """Approximate 3-max push/fold ranges under ICM.

    ``stacks_bb`` is ordered BTN, SB, BB. Action order is BTN, SB, BB; after a
    BTN shove, the SB and BB can each call/fold, and the BB's final node is
    separate when facing a heads-up shove versus a multiway call. Each player
    has one decision at most in a terminal path.
    """
    if isinstance(stacks_bb, (str, bytes)) or isinstance(payouts, (str, bytes)):
        raise ValueError("stacks_bb and payouts must be numeric sequences")
    if len(stacks_bb) != 3 or len(payouts) != 3:
        raise ValueError("three players require exactly three stacks and three payouts")
    stacks = tuple(_number(f"stack {seat}", value, positive=True) for seat, value in zip(("BTN", "SB", "BB"), stacks_bb))
    prize_values = tuple(_number(f"payout {place}", value) for place, value in enumerate(payouts, 1))
    if sum(prize_values) <= 0.0:
        raise ValueError("at least one payout must be positive")
    if any(prize_values[i] < prize_values[i + 1] for i in range(2)):
        raise ValueError("payouts must be ordered from first place to last place")
    sb = _number("small_blind_bb", small_blind_bb, positive=True)
    bb = _number("big_blind_bb", big_blind_bb, positive=True)
    ante = _number("ante_bb", ante_bb)
    tol = _number("tolerance", tolerance, positive=True)
    if sb >= bb:
        raise ValueError("small_blind_bb must be smaller than big_blind_bb")
    forced = (ante, ante + sb, ante + bb)
    if any(stack <= post for stack, post in zip(stacks, forced)):
        raise ValueError("each stack must exceed its ante and forced blind")
    for name, value in (("samples", samples), ("max_iterations", max_iterations), ("check_interval", check_interval)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    deals = _make_deals(samples, seed, stacks, forced, prize_values)
    n = len(HAND_CLASSES)
    regret_first = [[0.0] * n for _ in _STRATEGY_NAMES]
    regret_second = [[0.0] * n for _ in _STRATEGY_NAMES]
    strategy_sums = [[0.0] * n for _ in _STRATEGY_NAMES]
    strategy_weight = 0.0
    nash_conv = math.inf
    expected_icm = (0.0, 0.0, 0.0)
    completed = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        current = tuple(
            tuple(_regret_probability(regret_first[group][i], regret_second[group][i]) for i in range(n))
            for group in range(len(_STRATEGY_NAMES))
        )
        # Linear CFR averaging; there is no previous own action at these nodes,
        # so each information-set iteration has unit own-reach weight.
        weight = float(iteration)
        strategy_weight += weight
        for group in range(len(_STRATEGY_NAMES)):
            for hand_index, probability in enumerate(current[group]):
                strategy_sums[group][hand_index] += weight * probability

        increments_first = [[0.0] * n for _ in _STRATEGY_NAMES]
        increments_second = [[0.0] * n for _ in _STRATEGY_NAMES]
        for deal in deals:
            btn, sb_class, bb_class = deal.classes
            p, q, r = current[0][btn], current[1][sb_class], current[2][sb_class]
            t, s, u = current[3][bb_class], current[4][bb_class], current[5][bb_class]
            leaf = deal.utilities

            btn_fold = _mix(leaf[0], _mix(leaf[1], leaf[2], t), q)[0]
            btn_push = _mix(_mix(leaf[3], leaf[4], s), _mix(leaf[5], leaf[6], u), r)[0]
            delta = btn_push - btn_fold
            increments_first[0][btn] += (1.0 - p) * delta
            increments_second[0][btn] -= p * delta

            sb_open_fold = leaf[0][1]
            sb_open_push = _mix(leaf[1], leaf[2], t)[1]
            delta = sb_open_push - sb_open_fold
            reach = 1.0 - p
            increments_first[1][sb_class] += reach * (1.0 - q) * delta
            increments_second[1][sb_class] -= reach * q * delta

            sb_call_fold = _mix(leaf[3], leaf[4], s)[1]
            sb_call = _mix(leaf[5], leaf[6], u)[1]
            delta = sb_call - sb_call_fold
            increments_first[2][sb_class] += p * (1.0 - r) * delta
            increments_second[2][sb_class] -= p * r * delta

            delta = leaf[2][2] - leaf[1][2]
            reach = (1.0 - p) * q
            increments_first[3][bb_class] += reach * (1.0 - t) * delta
            increments_second[3][bb_class] -= reach * t * delta

            delta = leaf[4][2] - leaf[3][2]
            reach = p * (1.0 - r)
            increments_first[4][bb_class] += reach * (1.0 - s) * delta
            increments_second[4][bb_class] -= reach * s * delta

            delta = leaf[6][2] - leaf[5][2]
            reach = p * r
            increments_first[5][bb_class] += reach * (1.0 - u) * delta
            increments_second[5][bb_class] -= reach * u * delta

        for group in range(len(_STRATEGY_NAMES)):
            for hand_index in range(n):
                regret_first[group][hand_index] = max(
                    0.0, regret_first[group][hand_index] + increments_first[group][hand_index]
                )
                regret_second[group][hand_index] = max(
                    0.0, regret_second[group][hand_index] + increments_second[group][hand_index]
                )

        completed = iteration
        if iteration % check_interval == 0 or iteration == max_iterations:
            average = tuple(
                tuple(value / strategy_weight for value in row) for row in strategy_sums
            )
            nash_conv, expected_icm = _sampled_nash_conv(deals, average)
            if nash_conv <= tol:
                converged = True
                break

    average = tuple(tuple(value / strategy_weight for value in row) for row in strategy_sums)
    return ThreeMaxSolution(
        stacks_bb=stacks,
        payouts=prize_values,
        small_blind_bb=sb,
        big_blind_bb=bb,
        ante_bb=ante,
        samples=samples,
        seed=seed,
        iterations=completed,
        nash_conv=nash_conv,
        converged=converged,
        strategies=average,
        expected_icm=expected_icm,
    )


def solve_three_max_push_fold_stratified(
    stacks_bb: Sequence[float],
    payouts: Sequence[float],
    *,
    samples_per_hand_class: int = 100,
    small_blind_bb: float = 0.5,
    big_blind_bb: float = 1.0,
    ante_bb: float = 0.0,
    seed: int = 7,
    max_iterations: int = 1_000,
    tolerance: float = 0.005,
    check_interval: int = 25,
) -> ThreeMaxSolution:
    """Solve with balanced conditional chance samples for all 169 classes.

    ``samples_per_hand_class`` deals are generated for every seat/class pair,
    so the physical sample count is ``3 * 169 * samples_per_hand_class``. The
    class probability is applied exactly when calculating NashConv; regret
    matching may omit that fixed per-information-set scale.
    """
    if isinstance(stacks_bb, (str, bytes)) or isinstance(payouts, (str, bytes)):
        raise ValueError("stacks_bb and payouts must be numeric sequences")
    if len(stacks_bb) != 3 or len(payouts) != 3:
        raise ValueError("three players require exactly three stacks and three payouts")
    stacks = tuple(
        _number(f"stack {seat}", value, positive=True)
        for seat, value in zip(("BTN", "SB", "BB"), stacks_bb)
    )
    prize_values = tuple(_number(f"payout {place}", value) for place, value in enumerate(payouts, 1))
    if sum(prize_values) <= 0.0:
        raise ValueError("at least one payout must be positive")
    if any(prize_values[i] < prize_values[i + 1] for i in range(2)):
        raise ValueError("payouts must be ordered from first place to last place")
    sb = _number("small_blind_bb", small_blind_bb, positive=True)
    bb = _number("big_blind_bb", big_blind_bb, positive=True)
    ante = _number("ante_bb", ante_bb)
    tol = _number("tolerance", tolerance, positive=True)
    if sb >= bb:
        raise ValueError("small_blind_bb must be smaller than big_blind_bb")
    forced = (ante, ante + sb, ante + bb)
    if any(stack <= post for stack, post in zip(stacks, forced)):
        raise ValueError("each stack must exceed its ante and forced blind")
    for name, value in (
        ("samples_per_hand_class", samples_per_hand_class),
        ("max_iterations", max_iterations),
        ("check_interval", check_interval),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")

    conditioned_deals = _make_stratified_deals(
        samples_per_hand_class, seed, stacks, forced, prize_values
    )
    n = len(HAND_CLASSES)
    total_samples = 3 * n * samples_per_hand_class
    regret_first = [[0.0] * n for _ in _STRATEGY_NAMES]
    regret_second = [[0.0] * n for _ in _STRATEGY_NAMES]
    strategy_sums = [[0.0] * n for _ in _STRATEGY_NAMES]
    strategy_weight = 0.0
    nash_conv = math.inf
    expected_icm = (0.0, 0.0, 0.0)
    completed = 0
    converged = False

    for iteration in range(1, max_iterations + 1):
        current = tuple(
            tuple(
                _regret_probability(regret_first[group][i], regret_second[group][i])
                for i in range(n)
            )
            for group in range(len(_STRATEGY_NAMES))
        )
        weight = float(iteration)
        strategy_weight += weight
        for group in range(len(_STRATEGY_NAMES)):
            for hand_index, probability in enumerate(current[group]):
                strategy_sums[group][hand_index] += weight * probability

        increments_first = [[0.0] * n for _ in _STRATEGY_NAMES]
        increments_second = [[0.0] * n for _ in _STRATEGY_NAMES]
        inverse_class_samples = 1.0 / samples_per_hand_class

        for hand_index in range(n):
            p_btn = current[0][hand_index]
            p_sb_open = current[1][hand_index]
            p_sb_call = current[2][hand_index]
            p_bb_sb = current[3][hand_index]
            p_bb_btn = current[4][hand_index]
            p_bb_multi = current[5][hand_index]

            # BTN regrets, conditional on BTN's private class.
            btn_first = btn_second = 0.0
            for deal in conditioned_deals[0][hand_index]:
                btn, sb_class, bb_class = deal.classes
                p, q, r = p_btn, current[1][sb_class], current[2][sb_class]
                t, s, u = current[3][bb_class], current[4][bb_class], current[5][bb_class]
                leaf = deal.utilities
                btn_fold = _mix(leaf[0], _mix(leaf[1], leaf[2], t), q)[0]
                btn_push = _mix(_mix(leaf[3], leaf[4], s), _mix(leaf[5], leaf[6], u), r)[0]
                delta = btn_push - btn_fold
                btn_first += (1.0 - p) * delta
                btn_second -= p * delta
            increments_first[0][hand_index] = btn_first * inverse_class_samples
            increments_second[0][hand_index] = btn_second * inverse_class_samples

            # SB has two public-history information sets, both conditional on SB's class.
            sb_open_first = sb_open_second = 0.0
            sb_call_first = sb_call_second = 0.0
            for deal in conditioned_deals[1][hand_index]:
                btn, _sb_class, bb_class = deal.classes
                p = current[0][btn]
                t, s, u = current[3][bb_class], current[4][bb_class], current[5][bb_class]
                leaf = deal.utilities

                delta = _mix(leaf[1], leaf[2], t)[1] - leaf[0][1]
                reach = 1.0 - p
                sb_open_first += reach * (1.0 - p_sb_open) * delta
                sb_open_second -= reach * p_sb_open * delta

                delta = _mix(leaf[5], leaf[6], u)[1] - _mix(leaf[3], leaf[4], s)[1]
                reach = p
                sb_call_first += reach * (1.0 - p_sb_call) * delta
                sb_call_second -= reach * p_sb_call * delta

            increments_first[1][hand_index] = sb_open_first * inverse_class_samples
            increments_second[1][hand_index] = sb_open_second * inverse_class_samples
            increments_first[2][hand_index] = sb_call_first * inverse_class_samples
            increments_second[2][hand_index] = sb_call_second * inverse_class_samples

            # BB's three nodes, conditional on BB's own private class.
            bb_sb_first = bb_sb_second = 0.0
            bb_btn_first = bb_btn_second = 0.0
            bb_multi_first = bb_multi_second = 0.0
            for deal in conditioned_deals[2][hand_index]:
                btn, sb_class, _bb_class = deal.classes
                p = current[0][btn]
                q, r = current[1][sb_class], current[2][sb_class]
                leaf = deal.utilities

                delta = leaf[2][2] - leaf[1][2]
                reach = (1.0 - p) * q
                bb_sb_first += reach * (1.0 - p_bb_sb) * delta
                bb_sb_second -= reach * p_bb_sb * delta

                delta = leaf[4][2] - leaf[3][2]
                reach = p * (1.0 - r)
                bb_btn_first += reach * (1.0 - p_bb_btn) * delta
                bb_btn_second -= reach * p_bb_btn * delta

                delta = leaf[6][2] - leaf[5][2]
                reach = p * r
                bb_multi_first += reach * (1.0 - p_bb_multi) * delta
                bb_multi_second -= reach * p_bb_multi * delta

            increments_first[3][hand_index] = bb_sb_first * inverse_class_samples
            increments_second[3][hand_index] = bb_sb_second * inverse_class_samples
            increments_first[4][hand_index] = bb_btn_first * inverse_class_samples
            increments_second[4][hand_index] = bb_btn_second * inverse_class_samples
            increments_first[5][hand_index] = bb_multi_first * inverse_class_samples
            increments_second[5][hand_index] = bb_multi_second * inverse_class_samples

        for group in range(len(_STRATEGY_NAMES)):
            for hand_index in range(n):
                regret_first[group][hand_index] = max(
                    0.0, regret_first[group][hand_index] + increments_first[group][hand_index]
                )
                regret_second[group][hand_index] = max(
                    0.0, regret_second[group][hand_index] + increments_second[group][hand_index]
                )

        completed = iteration
        if iteration % check_interval == 0 or iteration == max_iterations:
            average = tuple(
                tuple(value / strategy_weight for value in row) for row in strategy_sums
            )
            nash_conv, expected_icm = _stratified_nash_conv(conditioned_deals, average)
            if nash_conv <= tol:
                converged = True
                break

    average = tuple(tuple(value / strategy_weight for value in row) for row in strategy_sums)
    return ThreeMaxSolution(
        stacks_bb=stacks,
        payouts=prize_values,
        small_blind_bb=sb,
        big_blind_bb=bb,
        ante_bb=ante,
        samples=total_samples,
        seed=seed,
        iterations=completed,
        nash_conv=nash_conv,
        converged=converged,
        strategies=average,
        expected_icm=expected_icm,
        sampling_method="stratified_by_seat_and_hand_class",
        samples_per_hand_class=samples_per_hand_class,
    )


__all__ = [
    "ThreeMaxSolution",
    "solve_three_max_push_fold",
    "solve_three_max_push_fold_stratified",
]
