"""Preflop all-in equity between the 169 starting-hand classes.

The 169x169 matrix is the primitive every short-stack tool needs: push/fold equilibria, ICM-adjusted
shove ranges and hand-vs-range equity all reduce to weighted lookups in it. It is computed once
(see ``tools/build_preflop_matrix.py``) and stored in ``data/preflop_equity_169.json``.

Only the strict upper triangle is stored: ``equity(a, b)`` for ``a < b``. The diagonal is exactly 0.5 by
symmetry (a class against itself), and the lower triangle is derived as ``1 - equity(b, a)``, so the matrix
is consistent by construction rather than by luck of two independent Monte Carlo runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
import json
from pathlib import Path
import random
from typing import Any, Iterable

from . import __version__
from .cards import Card, RANKS, SUITS, evaluate

DEFAULT_MATRIX_PATH = Path(__file__).resolve().parents[1] / "data" / "preflop_equity_169.json"

# Ranks high to low, the order poker charts are drawn in.
_RANKS_DESC = tuple(range(14, 1, -1))


def _rank_char(rank: int) -> str:
    return RANKS[rank - 2]


def _build_hand_classes() -> tuple[str, ...]:
    """All 169 starting-hand classes: 13 pairs, 78 suited, 78 offsuit, in canonical chart order."""
    classes: list[str] = []
    for high in _RANKS_DESC:
        for low in _RANKS_DESC:
            if high == low:
                classes.append(f"{_rank_char(high)}{_rank_char(low)}")
            elif high > low:
                classes.append(f"{_rank_char(high)}{_rank_char(low)}s")
            else:
                classes.append(f"{_rank_char(low)}{_rank_char(high)}o")
    # The 13x13 grid above lists each suited/offsuit class once; keep chart order, drop nothing.
    seen: set[str] = set()
    ordered: list[str] = []
    for item in classes:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return tuple(ordered)


HAND_CLASSES: tuple[str, ...] = _build_hand_classes()
CLASS_INDEX: dict[str, int] = {name: i for i, name in enumerate(HAND_CLASSES)}


def parse_hand_class(text: str) -> str:
    """Normalize a class token ('aks', 'AKS', 'KA s') to canonical form ('AKs'); pairs carry no suffix."""
    token = "".join(str(text).split()).upper()
    if len(token) == 2:
        first, second = token[0], token[1]
        if first not in RANKS or second not in RANKS:
            raise ValueError(f"classe de mão inválida: {text!r}")
        if first != second:
            raise ValueError(f"classe de mão não-par precisa de sufixo 's' ou 'o': {text!r}")
        return f"{first}{second}"
    if len(token) != 3:
        raise ValueError(f"classe de mão inválida: {text!r}")
    first, second, suffix = token[0], token[1], token[2].lower()
    if first not in RANKS or second not in RANKS or suffix not in ("s", "o"):
        raise ValueError(f"classe de mão inválida: {text!r}")
    if first == second:
        raise ValueError(f"par não pode ter sufixo de naipe: {text!r}")
    hi, lo = (first, second) if RANKS.index(first) > RANKS.index(second) else (second, first)
    return f"{hi}{lo}{suffix}"


def class_combos(hand_class: str) -> tuple[tuple[Card, Card], ...]:
    """Expand a class into its physical combos: 6 for a pair, 4 suited, 12 offsuit."""
    name = parse_hand_class(hand_class)
    high = RANKS.index(name[0]) + 2
    low = RANKS.index(name[1]) + 2
    if len(name) == 2:
        cards = [Card(high, suit) for suit in SUITS]
        return tuple((a, b) for a, b in combinations(cards, 2))
    suited = name[2] == "s"
    out: list[tuple[Card, Card]] = []
    for hs in SUITS:
        for ls in SUITS:
            if suited != (hs == ls):
                continue
            out.append((Card(high, hs), Card(low, ls)))
    return tuple(out)


def combo_count(hand_class: str) -> int:
    """Number of physical combos in a class, used to weight classes inside a range."""
    name = parse_hand_class(hand_class)
    if len(name) == 2:
        return 6
    return 4 if name[2] == "s" else 12


_FULL_DECK: tuple[Card, ...] = tuple(Card(rank, suit) for rank in range(2, 15) for suit in SUITS)


def _valid_pairings(class_a: str, class_b: str) -> tuple[tuple[tuple[Card, Card], tuple[Card, Card], tuple[Card, ...]], ...]:
    """Every card-disjoint (combo_a, combo_b) pairing, each with its 48-card remaining deck."""
    out = []
    for combo_a in class_combos(class_a):
        codes_a = {card.code for card in combo_a}
        for combo_b in class_combos(class_b):
            codes_b = {card.code for card in combo_b}
            if codes_a & codes_b:
                continue
            used = codes_a | codes_b
            remaining = tuple(card for card in _FULL_DECK if card.code not in used)
            out.append((combo_a, combo_b, remaining))
    return tuple(out)


def matchup_equity(class_a: str, class_b: str, samples: int = 20_000, seed: int = 7) -> float:
    """Monte Carlo all-in equity of ``class_a`` against ``class_b`` over random boards.

    A sample draws one card-disjoint combo pairing uniformly, then five board cards. Ties split the pot.
    Identical classes return exactly 0.5 by symmetry without sampling.
    """
    name_a = parse_hand_class(class_a)
    name_b = parse_hand_class(class_b)
    if name_a == name_b:
        return 0.5
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError("samples must be a positive integer")

    pairings = _valid_pairings(name_a, name_b)
    if not pairings:
        raise ValueError(f"sem confrontos possíveis entre {name_a} e {name_b}")

    rng = random.Random(seed)
    pick = rng.choice
    draw = rng.sample
    shares = 0.0
    for _ in range(samples):
        combo_a, combo_b, deck = pick(pairings)
        board = draw(deck, 5)
        rank_a = evaluate((*combo_a, *board))
        rank_b = evaluate((*combo_b, *board))
        if rank_a > rank_b:
            shares += 1.0
        elif rank_a == rank_b:
            shares += 0.5
    return shares / samples


@dataclass(frozen=True)
class PreflopMatrix:
    """169x169 all-in equity lookup, stored as its strict upper triangle."""

    values: tuple[float, ...]
    samples: int
    seed: int
    generated_at: str
    app_version: str

    @staticmethod
    def _offset(i: int, j: int) -> int:
        # Index into the flattened strict upper triangle for i < j.
        n = len(HAND_CLASSES)
        return i * n - (i * (i + 1)) // 2 + (j - i - 1)

    def equity(self, class_a: str, class_b: str) -> float:
        """All-in equity of ``class_a`` against ``class_b``, in [0, 1]."""
        i = CLASS_INDEX[parse_hand_class(class_a)]
        j = CLASS_INDEX[parse_hand_class(class_b)]
        if i == j:
            return 0.5
        if i < j:
            return self.values[self._offset(i, j)]
        return 1.0 - self.values[self._offset(j, i)]

    def equity_vs_range(self, hand_class: str, opponent: Iterable[str] | dict[str, float]) -> float:
        """Equity of one class against a range, weighted by each class's physical combo count.

        ``opponent`` is either an iterable of class names (uniform) or a mapping of class name to a
        weight in [0, 1] (the fraction of that class the opponent plays this way). Combo counts are
        applied on top, so AKs counts 4 and AKo counts 12. Card removal between the hero's exact cards
        and the opponent's range is not modelled: this is class-level, matching standard push/fold tools.
        """
        weights = (
            {name: float(w) for name, w in opponent.items()}
            if isinstance(opponent, dict)
            else {name: 1.0 for name in opponent}
        )
        total = 0.0
        acc = 0.0
        for name, weight in weights.items():
            if weight <= 0.0:
                continue
            mass = weight * combo_count(name)
            acc += mass * self.equity(hand_class, name)
            total += mass
        if total <= 0.0:
            raise ValueError("a range do oponente não tem combinações com peso positivo")
        return acc / total

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "lugon_preflop_equity_169_upper_triangle",
            "classes": list(HAND_CLASSES),
            "samples_per_matchup": self.samples,
            "seed": self.seed,
            "generated_at": self.generated_at,
            "app_version": self.app_version,
            "note": "Diagonal is exactly 0.5; lower triangle is 1 - upper.",
            "values": [round(v, 6) for v in self.values],
        }

    def save(self, path: Path = DEFAULT_MATRIX_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path = DEFAULT_MATRIX_PATH) -> "PreflopMatrix":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        classes = tuple(data.get("classes") or ())
        if classes != HAND_CLASSES:
            raise ValueError("matriz gravada não corresponde à ordem canónica de classes desta versão")
        expected = len(HAND_CLASSES) * (len(HAND_CLASSES) - 1) // 2
        values = tuple(float(v) for v in data["values"])
        if len(values) != expected:
            raise ValueError(f"matriz corrompida: esperava {expected} valores, encontrei {len(values)}")
        return cls(
            values=values,
            samples=int(data.get("samples_per_matchup", 0)),
            seed=int(data.get("seed", 0)),
            generated_at=str(data.get("generated_at", "")),
            app_version=str(data.get("app_version", "")),
        )


def upper_triangle_pairs() -> tuple[tuple[int, int], ...]:
    """Every (i, j) with i < j: the 14,196 matchups that actually need computing."""
    n = len(HAND_CLASSES)
    return tuple((i, j) for i in range(n) for j in range(i + 1, n))


def compute_chunk(task: tuple[int, int, int, int]) -> tuple[int, int, float]:
    """Worker entry point: compute one matchup. Module-level and picklable for multiprocessing."""
    i, j, samples, seed = task
    # Seed per matchup so the result is reproducible regardless of worker scheduling.
    value = matchup_equity(HAND_CLASSES[i], HAND_CLASSES[j], samples=samples, seed=seed + i * 1009 + j)
    return i, j, value
