"""A small, explicit Texas Hold'em starting-hand range parser."""

from __future__ import annotations

from itertools import combinations
import re

from .cards import Card, RANKS, SUITS

Combo = tuple[int, int]
_CLASS = re.compile(r"^([2-9TJQKA])([2-9TJQKA])([so]?)$", re.IGNORECASE)
_EXACT = re.compile(r"^[2-9TJQKA][cdhs][2-9TJQKA][cdhs]$", re.IGNORECASE)


def _combo(first: Card, second: Card) -> Combo:
    return tuple(sorted((first.code, second.code)))  # type: ignore[return-value]


def _pair_combos(rank: int) -> set[Combo]:
    cards = [Card(rank, suit) for suit in SUITS]
    return {_combo(a, b) for a, b in combinations(cards, 2)}


def _non_pair_combos(high: int, low: int, suitedness: str = "") -> set[Combo]:
    result: set[Combo] = set()
    for high_suit in SUITS:
        for low_suit in SUITS:
            if suitedness == "s" and high_suit != low_suit:
                continue
            if suitedness == "o" and high_suit == low_suit:
                continue
            result.add(_combo(Card(high, high_suit), Card(low, low_suit)))
    return result


def _expand_class(token: str) -> set[Combo]:
    plus = token.endswith("+")
    core = token[:-1] if plus else token
    match = _CLASS.fullmatch(core)
    if not match:
        raise ValueError(f"unsupported range token {token!r}; use AA, AKs, AKo, exact cards, or *")

    first_char, second_char, suitedness = match.groups()
    first_rank = RANKS.index(first_char.upper()) + 2
    second_rank = RANKS.index(second_char.upper()) + 2
    suitedness = suitedness.lower()
    if first_rank == second_rank:
        if suitedness:
            raise ValueError(f"pair range cannot have a suitedness suffix: {token!r}")
        ranks = range(first_rank, 15) if plus else (first_rank,)
        return set().union(*(_pair_combos(rank) for rank in ranks))

    if first_rank < second_rank:
        raise ValueError(f"range ranks must be ordered high to low: {token!r}")
    if plus:
        if first_rank != 14:
            raise ValueError("+ expansion for non-pairs is supported only for ace-high hands (for example AJs+)")
        lows = range(second_rank, 14)
    else:
        lows = (second_rank,)
    return set().union(*(_non_pair_combos(first_rank, low, suitedness) for low in lows))


def parse_range(text: str | None) -> frozenset[Combo]:
    """Expand comma-separated classes into uniformly weighted physical combos.

    Supported examples: ``*``, ``AA``, ``AKs``, ``AKo``, ``AK``, ``QQ+``,
    ``AJs+``, and explicit two-card combos such as ``AsKd``. Overlapping
    terms are unioned; weights and advanced interval syntax are not supported.
    """
    if text is None or not text.strip() or text.strip() == "*":
        all_cards = [Card(rank, suit) for rank in range(2, 15) for suit in SUITS]
        return frozenset(_combo(a, b) for a, b in combinations(all_cards, 2))

    result: set[Combo] = set()
    for raw in text.split(","):
        token = raw.strip()
        if not token:
            raise ValueError("range contains an empty token")
        if token == "*":
            return parse_range(None)
        if _EXACT.fullmatch(token):
            first, second = Card.parse(token[:2]), Card.parse(token[2:])
            if first == second:
                raise ValueError(f"range combo contains the same card twice: {token!r}")
            result.add(_combo(first, second))
        else:
            result.update(_expand_class(token))
    if not result:
        raise ValueError("range contains no card combinations")
    return frozenset(result)
