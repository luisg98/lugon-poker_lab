"""Card parsing and standard high-hand evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

RANKS = "23456789TJQKA"
SUITS = "cdhs"
HAND_NAMES = (
    "high card",
    "one pair",
    "two pair",
    "three of a kind",
    "straight",
    "flush",
    "full house",
    "four of a kind",
    "straight flush",
)


@dataclass(frozen=True, order=True)
class Card:
    rank: int
    suit: str

    def __post_init__(self) -> None:
        if self.rank < 2 or self.rank > 14 or self.suit not in SUITS:
            raise ValueError(f"invalid card: rank={self.rank}, suit={self.suit!r}")

    @classmethod
    def parse(cls, value: str) -> "Card":
        if not isinstance(value, str):
            raise ValueError(f"card must be a string, received {type(value).__name__}")
        token = value.strip()
        if len(token) != 2:
            raise ValueError(f"card must use two characters, such as As: {value!r}")
        rank_text, suit = token[0].upper(), token[1].lower()
        if rank_text not in RANKS or suit not in SUITS:
            raise ValueError(f"invalid card: {value!r}; ranks are 2-9,T,J,Q,K,A and suits c,d,h,s")
        return cls(RANKS.index(rank_text) + 2, suit)

    @property
    def code(self) -> int:
        return self.rank * 4 + SUITS.index(self.suit)

    def __str__(self) -> str:
        return f"{RANKS[self.rank - 2]}{self.suit}"


def parse_cards(values: Iterable[str]) -> tuple[Card, ...]:
    cards = tuple(Card.parse(value) for value in values)
    if len({card.code for card in cards}) != len(cards):
        raise ValueError("the same card cannot appear more than once")
    return cards


def _straight_high(ranks: set[int]) -> int | None:
    if 14 in ranks:
        ranks = set(ranks) | {1}
    for high in range(14, 4, -1):
        if all(rank in ranks for rank in range(high - 4, high + 1)):
            return high
    return None


def evaluate(cards: Iterable[Card]) -> tuple[int, ...]:
    """Return a lexicographically comparable rank tuple for 5 to 7 cards."""
    hand = tuple(cards)
    if not 5 <= len(hand) <= 7:
        raise ValueError("hand evaluation requires five, six, or seven cards")
    if len({card.code for card in hand}) != len(hand):
        raise ValueError("a hand cannot contain duplicate cards")

    counts: dict[int, int] = {}
    suits: dict[str, list[int]] = {}
    for card in hand:
        counts[card.rank] = counts.get(card.rank, 0) + 1
        suits.setdefault(card.suit, []).append(card.rank)

    rank_set = set(counts)
    flush_suits = [suit for suit, suit_ranks in suits.items() if len(suit_ranks) >= 5]
    for suit in flush_suits:
        straight_flush = _straight_high(set(suits[suit]))
        if straight_flush is not None:
            return (8, straight_flush)

    groups = sorted(((count, rank) for rank, count in counts.items()), reverse=True)
    quads = [rank for count, rank in groups if count == 4]
    if quads:
        quad = quads[0]
        kicker = max(rank for rank in rank_set if rank != quad)
        return (7, quad, kicker)

    trips = sorted((rank for count, rank in groups if count >= 3), reverse=True)
    if trips:
        pair_candidates = [rank for count, rank in groups if rank != trips[0] and count >= 2]
        if pair_candidates:
            return (6, trips[0], max(pair_candidates))

    if flush_suits:
        top_five = sorted(suits[flush_suits[0]], reverse=True)[:5]
        return (5, *top_five)

    straight = _straight_high(rank_set)
    if straight is not None:
        return (4, straight)

    if trips:
        kickers = sorted((rank for rank in rank_set if rank != trips[0]), reverse=True)[:2]
        return (3, trips[0], *kickers)

    pairs = sorted((rank for count, rank in groups if count >= 2), reverse=True)
    if len(pairs) >= 2:
        kicker = max(rank for rank in rank_set if rank not in pairs[:2])
        return (2, pairs[0], pairs[1], kicker)
    if pairs:
        kickers = sorted((rank for rank in rank_set if rank != pairs[0]), reverse=True)[:3]
        return (1, pairs[0], *kickers)

    return (0, *sorted(rank_set, reverse=True)[:5])


def best_hand(cards: Iterable[Card]) -> tuple[int, ...]:
    """Evaluate the best five-card hand from five to seven available cards.

    ``evaluate`` already picks the best five out of six or seven: it counts ranks and suits over the whole
    hand and applies the category checks in priority order. Iterating the 21 five-card subsets returned the
    same rank tuple on 300k random 7-card hands, 150k 6-card hands and the usual edge cases (wheel, 6-card
    straight flush, two trips, quads over trips), and was ~17x slower. See tests/test_engine.py.
    """
    hand = tuple(cards)
    if not 5 <= len(hand) <= 7:
        raise ValueError("best-hand evaluation requires five, six, or seven cards")
    return evaluate(hand)


def describe(rank: tuple[int, ...]) -> str:
    return HAND_NAMES[rank[0]]
