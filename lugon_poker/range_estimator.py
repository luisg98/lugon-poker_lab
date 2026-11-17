"""Context-aware opponent range estimation when explicit ranges are unavailable."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .cards import Card
from .ranges import Combo, parse_range

# Ordered from tightest (0) to widest (5).
# NOTE: ranges.py supports '+' only on pairs (e.g. '77+') and ace-high hands (e.g. 'A7s+').
RANGE_TIERS: tuple[tuple[str, str, str], ...] = (
    (
        "tight",
        "Top ~6% (premium hands / call vs a large shove)",
        "99+,AQs+,AQo+,KQs",
    ),
    (
        "strong",
        "Top ~11% (strong open / tight defense)",
        "77+,ATs+,AJo+,KQs,KJs,QJs",
    ),
    (
        "standard",
        "Top ~18% (standard open / short-stack shove)",
        "55+,A7s+,ATo+,KQs,KJs,KTs,KQo,QJs,QTs,JTs,T9s",
    ),
    (
        "active",
        "Top ~28% (active BTN/SB or medium call)",
        "33+,A2s+,A8o+,KQs,KJs,KTs,K9s,K8s,KQo,KJo,QJs,QTs,Q9s,QJo,JTs,J9s,JTo,T9s,T8s,98s,87s",
    ),
    (
        "wide",
        "Top ~42% (wide Spin & Go / limp / check-defend)",
        "22+,A2s+,A2o+,KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,KQo,KJo,KTo,K9o,QJs,QTs,Q9s,Q8s,Q7s,QJo,QTo,JTs,J9s,J8s,JTo,T9s,T8s,T7s,T9o,98s,97s,87s,86s,76s,65s",
    ),
    (
        "very_wide",
        "Top ~58% (BB defense / very wide SB shove)",
        "22+,A2s+,A2o+,KQs,KJs,KTs,K9s,K8s,K7s,K6s,K5s,K4s,K3s,K2s,KQo,KJo,KTo,K9o,K8o,K7o,QJs,QTs,Q9s,Q8s,Q7s,Q6s,Q5s,Q4s,QJo,QTo,Q9o,JTs,J9s,J8s,J7s,J6s,JTo,J9o,T9s,T8s,T7s,T6s,T9o,T8o,98s,97s,96s,98o,87s,86s,85s,87o,76s,75s,65s,64s,54s",
    ),
)


@dataclass(frozen=True)
class RangeEstimate:
    range_text: str
    tier: str
    label: str
    reason: str
    is_explicit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "range_text": self.range_text,
            "tier": self.tier,
            "label": self.label,
            "reason": self.reason,
            "is_explicit": self.is_explicit,
        }


def _has_unblocked_combos(range_text: str, dead_cards: Sequence[str]) -> bool:
    if not dead_cards:
        return True
    dead_codes = {Card.parse(token).code for token in dead_cards}
    combos: frozenset[Combo] = parse_range(range_text)
    return any(first not in dead_codes and second not in dead_codes for first, second in combos)


def _profile_rate(profile: Mapping[str, Any] | None, metric: str, min_opportunities: int = 20) -> float | None:
    if not profile:
        return None
    block = profile.get(metric)
    if not isinstance(block, Mapping):
        return None
    opportunities = block.get("opportunities", 0)
    rate = block.get("empirical_rate")
    if isinstance(opportunities, int) and opportunities >= min_opportunities and isinstance(rate, (int, float)):
        return float(rate)
    return None


def estimate_opponent_range(
    *,
    street: str,
    pot: float,
    to_call: float,
    hero_stack: float,
    big_blind: float = 1.0,
    game_type: str = "spin_and_go",
    opp_position: str | None = None,
    opp_bet: float | None = None,
    opp_stack: float | None = None,
    num_opponents: int = 1,
    opponent_profile: Mapping[str, Any] | None = None,
    explicit_range: str | None = None,
    dead_cards: Sequence[str] = (),
) -> RangeEstimate:
    """Infer a realistic opponent range from table context when range is '*' or omitted."""
    cleaned = (explicit_range or "").strip()
    if cleaned and cleaned != "*":
        return RangeEstimate(
            range_text=cleaned,
            tier="explicit",
            label="Explicit range",
            reason="Range entered manually for this opponent.",
            is_explicit=True,
        )

    bb = max(float(big_blind), 1e-6)
    pos = str(opp_position or "").strip().upper()
    bet_val = float(opp_bet) if opp_bet is not None and opp_bet > 0 else float(to_call)
    pressure_bb = bet_val / bb
    pot_before = max(1e-6, float(pot) - float(to_call))
    bet_pot_ratio = float(to_call) / pot_before if to_call > 0 else 0.0
    commits_stack = (hero_stack > 0 and to_call >= hero_stack * 0.60) or (
        opp_stack is not None and opp_stack <= bb * 1.5 and bet_val > 0
    )
    is_spin = game_type != "tournament"
    multiway = num_opponents >= 2

    reasons: list[str] = []

    if street == "preflop":
        if to_call <= 1e-6:
            tier_idx = 4 if is_spin else 3
            if pos == "BB":
                tier_idx = 5 if is_spin else 4
            reasons.append("pre-flop with no heavy open bet")
        else:
            if pressure_bb >= 8.0 or commits_stack:
                if is_spin and not multiway and pos == "SB":
                    tier_idx = 4
                    reasons.append("3-max SB shove/pressure (widened range)")
                elif is_spin and not multiway and pos == "BTN":
                    tier_idx = 3
                    reasons.append("BTN shove/pressure")
                elif multiway:
                    tier_idx = 1 if is_spin else 0
                    reasons.append("all-in/high pressure in a multiway pot")
                else:
                    tier_idx = 2 if is_spin else 1
                    reasons.append("pre-flop all-in/high pressure")
            elif pressure_bb >= 3.5:
                tier_idx = 2 if is_spin else 1
                reasons.append(f"open-raise/3-bet of ~{pressure_bb:.1f} BB")
            else:
                tier_idx = 3 if is_spin else 2
                if pos in ("BTN", "SB") and not multiway:
                    tier_idx += 1
                reasons.append(f"moderate pre-flop bet (~{pressure_bb:.1f} BB)")
    else:
        if to_call <= 1e-6:
            tier_idx = 4 if street == "flop" else 3
            reasons.append(f"{street} with no bet faced (widened/capped range)")
        else:
            if bet_pot_ratio >= 0.75 or commits_stack:
                tier_idx = 2 if street == "flop" else 1
                reasons.append(f"large bet/all-in on the {street} ({bet_pot_ratio:.0%} pot)")
            elif bet_pot_ratio >= 0.35:
                tier_idx = 3 if street == "flop" else 2
                reasons.append(f"medium bet on the {street} ({bet_pot_ratio:.0%} pot)")
            else:
                tier_idx = 3 if street == "flop" else 2
                reasons.append(f"small bet on the {street} ({bet_pot_ratio:.0%} pot)")
            if multiway:
                tier_idx -= 1
                reasons.append("multiway adjustment")

    vpip = _profile_rate(opponent_profile, "vpip")
    pfr = _profile_rate(opponent_profile, "pfr")
    if vpip is not None:
        if vpip >= 0.45:
            tier_idx += 1
            reasons.append(f"perfil loose (VPIP {vpip:.0%})")
        elif vpip <= 0.18:
            tier_idx -= 1
            reasons.append(f"perfil tight (VPIP {vpip:.0%})")
    if to_call > 0 and pfr is not None and pfr <= 0.12:
        tier_idx -= 1
        reasons.append(f"low pre-flop aggression (PFR {pfr:.0%})")

    tier_idx = max(0, min(len(RANGE_TIERS) - 1, tier_idx))
    while tier_idx < len(RANGE_TIERS) - 1 and not _has_unblocked_combos(RANGE_TIERS[tier_idx][2], dead_cards):
        tier_idx += 1

    tier_key, label, range_text = RANGE_TIERS[tier_idx]
    return RangeEstimate(
        range_text=range_text,
        tier=tier_key,
        label=label,
        reason="; ".join(reasons) if reasons else label,
        is_explicit=False,
    )
