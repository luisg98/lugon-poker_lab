"""Transparent one-decision pot-odds and showdown-EV analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .equity import EquityResult
from .state import HandState


@dataclass(frozen=True)
class StrategyAnalysis:
    decision: str
    equity: float
    pot_odds: float | None
    call_ev: float | None
    fold_ev: float
    explanation: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def analyze_strategy(state: HandState, equity: EquityResult) -> StrategyAnalysis:
    if state.to_call == 0:
        return StrategyAnalysis(
            decision="check",
            equity=equity.equity,
            pot_odds=None,
            call_ev=None,
            fold_ev=0.0,
            explanation=(
                "There is no additional amount to call: checking keeps the option to see the next card. "
                "Equity estimates the showdown share under the stated ranges, but it is not the EV of checking "
                "because more bets may occur."
            ),
        )

    final_pot = state.pot + state.to_call
    pot_odds = state.to_call / final_pot if final_pot else 1.0
    call_ev = equity.equity * final_pot - state.to_call
    if call_ev > 1e-9:
        decision = "call"
        verdict = "Calling has positive EV under the stated ranges and showdown assumption."
    elif call_ev < -1e-9:
        decision = "fold"
        verdict = "Folding has higher immediate EV than calling under the stated ranges and showdown assumption."
    else:
        decision = "indifferent"
        verdict = "Calling and folding have approximately equal immediate EV in this model."
    explanation = (
        f"{verdict} Estimated equity: {equity.equity:.1%}; minimum equity to call: {pot_odds:.1%}; "
        f"incremental call EV: {call_ev:.2f} chips. This calculation assumes there are no more bets, "
        "side pots, or rake, and that current equity holds through showdown."
    )
    return StrategyAnalysis(decision, equity.equity, pot_odds, call_ev, 0.0, explanation)
