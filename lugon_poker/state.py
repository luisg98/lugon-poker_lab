"""Shared hand-state model and input validation."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

from .cards import Card, parse_cards
from .ranges import Combo, parse_range


@dataclass(frozen=True)
class Opponent:
    nickname: str
    range_text: str | None = None
    hand_range: frozenset[Combo] = frozenset()


@dataclass(frozen=True)
class HandState:
    hero_cards: tuple[Card, Card]
    board: tuple[Card, ...]
    pot: float
    to_call: float
    opponents: tuple[Opponent, ...]

    @property
    def street(self) -> str:
        return {0: "preflop", 3: "flop", 4: "turn", 5: "river"}[len(self.board)]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandState":
        if not isinstance(data, dict):
            raise ValueError("hand state must be a JSON object")
        hero_raw = data.get("hero")
        if not isinstance(hero_raw, dict) or not isinstance(hero_raw.get("cards"), list):
            raise ValueError("hero.cards must be a list containing exactly two cards")
        hero_cards = parse_cards(hero_raw["cards"])
        if len(hero_cards) != 2:
            raise ValueError("hero must have exactly two hole cards")

        board_raw = data.get("board", [])
        if not isinstance(board_raw, list):
            raise ValueError("board must be a list of cards")
        board = parse_cards(board_raw)
        if len(board) not in (0, 3, 4, 5):
            raise ValueError("board must contain 0, 3, 4, or 5 cards")
        if set(hero_cards) & set(board):
            raise ValueError("hero cards and board cannot share a card")

        pot = _amount(data.get("pot"), "pot")
        to_call = _amount(data.get("to_call", 0), "to_call")

        opponents_raw = data.get("opponents")
        if not isinstance(opponents_raw, list) or not 1 <= len(opponents_raw) <= 9:
            raise ValueError("opponents must list between one and nine active opponents")
        opponents: list[Opponent] = []
        known_codes = {card.code for card in (*hero_cards, *board)}
        for index, opponent_data in enumerate(opponents_raw, start=1):
            if not isinstance(opponent_data, dict):
                raise ValueError(f"opponent {index} must be an object")
            nickname = opponent_data.get("nickname", f"Opponent {index}")
            range_text = opponent_data.get("range")
            if not isinstance(nickname, str) or not nickname.strip():
                raise ValueError(f"opponent {index} nickname must be a non-empty string")
            if range_text is not None and not isinstance(range_text, str):
                raise ValueError(f"opponent {index} range must be a string")
            combo_range = parse_range(range_text)
            legal_range = frozenset(combo for combo in combo_range if not known_codes.intersection(combo))
            if not legal_range:
                raise ValueError(f"opponent {nickname!r} range has no combos after known-card blockers")
            opponents.append(Opponent(nickname.strip(), range_text, legal_range))

        return cls((hero_cards[0], hero_cards[1]), board, pot, to_call, tuple(opponents))


def _amount(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    return float(value)
