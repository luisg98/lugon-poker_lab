"""Pluggable strategic game assistant supporting deterministic baseline, local LLMs (Ollama/OpenAI-compatible), and cloud LLMs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from .cards import best_hand, parse_cards


_GAME_STRATEGY_PROFILES: dict[str, dict[str, Any]] = {
    "spin_and_go": {
        "label": "Spin & Go (3-max hyper-turbo)",
        "bet_eq": 0.56,
        "bet_pot_frac": 0.60,
        "raise_eq": 0.66,
        "raise_edge": 0.14,
        "raise_pot_frac": 0.75,
        "icm_low": 0.0,
        "icm_high": 0.0,
        "prompt_note": (
            "Current format: Spin & Go (3-max hyper-turbo, winner-take-all / top-heavy structure). "
            "Prioritize initiative, thin value, and suitable defense and pressure against wider 3-max ranges and short stacks."
        ),
    },
    "tournament": {
        "label": "Tournament (MTT)",
        "bet_eq": 0.60,
        "bet_pot_frac": 0.50,
        "raise_eq": 0.70,
        "raise_edge": 0.18,
        "raise_pot_frac": 0.65,
        "icm_low": 0.02,
        "icm_high": 0.035,
        "prompt_note": (
            "Current format: MTT tournament (6-max / 9-max). "
            "Account for ICM pressure and survival: require extra equity above pot odds for expensive or marginal calls that commit a large part of the stack, "
            "and use more controlled post-flop sizings."
        ),
    },
}


def normalize_game_type(game_type: str | None) -> str:
    """Normalize supported game types ('spin_and_go' or 'tournament')."""
    low = str(game_type or "spin_and_go").strip().lower()
    if low in ("tournament", "torneio", "torneios", "mtt"):
        return "tournament"
    return "spin_and_go"


_PLAY_STYLE_PROMPTS: dict[str, dict[str, str]] = {
    "balanced": {
        "label": "Balanced",
        "prompt_note": (
            "Play style: Balanced (Default). Follow the best mathematical EV and the engine/solver's standard frequencies "
            "without artificial risk deviations; use balanced sizings (~40–60% pot depending on texture and SPR)."
        ),
    },
    "passive": {
        "label": "Passive",
        "prompt_note": (
            "Play style: Passive (Lower Variance and Pot Control). Accept giving up marginal EV in exchange for lower risk "
            "and stack preservation: prefer checking or controlled calls with medium-strength hands, avoid expensive bluffs and marginal all-ins, "
            "and use smaller sizings (~25–40% pot or restrained min-raises) to risk fewer chips on each street."
        ),
    },
    "aggressive": {
        "label": "Aggressive",
        "prompt_note": (
            "Play style: Aggressive (Continuous Pressure). Pressure opponents to deny free equity "
            "and make each card expensive: prefer betting or raising over passive calls, put opponents to difficult stack decisions, "
            "and use larger sizings (~65–100% pot) on coordinated boards or when holding a range advantage."
        ),
    },
    "creative": {
        "label": "Creative",
        "prompt_note": (
            "Play style: Creative (Advanced Plays with a Theoretical Basis). Use complex, asymmetric, and risky lines "
            "when theory or solver evidence supports them (key blockers, nut advantage, range interaction, or board texture): "
            "consider traps and slow plays on dry boards, check-raises as semi-bluffs, donk or probe bets against weakness, "
            "and polarized overbets (120–150%+), accepting the risk and explaining the theoretical basis."
        ),
    },
}


def normalize_play_style(play_style: str | None) -> str:
    """Normalize supported play styles ('balanced', 'passive', 'aggressive', 'creative')."""
    low = str(play_style or "balanced").strip().lower()
    if low in ("passive", "passivo", "conservador"):
        return "passive"
    if low in ("aggressive", "agressivo", "lag", "pressao", "pressão"):
        return "aggressive"
    if low in ("creative", "criativo", "tricky"):
        return "creative"
    return "balanced"


@dataclass(frozen=True)
class LegalAction:
    action: str
    min_amount: float
    max_amount: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DecisionContext:
    hand_id: str
    street: str
    hero_nickname: str
    hero_position: str
    hero_cards: tuple[str, str]
    board: tuple[str, ...]
    hero_hand_description: str | None
    pot: float
    to_call: float
    hero_stack: float
    effective_stack: float
    spr: float | None
    legal_actions: tuple[LegalAction, ...]
    equity: dict[str, Any]
    math_strategy: dict[str, Any]
    recent_actions: tuple[dict[str, Any], ...]
    opponent_profiles: dict[str, dict[str, Any]]
    game_type: str = "spin_and_go"
    play_style: str = "balanced"
    three_max_solver: dict[str, Any] | None = None
    assumed_ranges: dict[str, dict[str, Any]] | None = None
    hand_analysis: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "game_type": normalize_game_type(self.game_type),
            "play_style": normalize_play_style(self.play_style),
            "street": self.street,
            "hero_nickname": self.hero_nickname,
            "hero_position": self.hero_position,
            "hero_cards": list(self.hero_cards),
            "board": list(self.board),
            "hero_hand_description": self.hero_hand_description,
            "pot": self.pot,
            "to_call": self.to_call,
            "hero_stack": self.hero_stack,
            "effective_stack": self.effective_stack,
            "spr": self.spr,
            "legal_actions": [item.to_dict() for item in self.legal_actions],
            "equity": self.equity,
            "math_strategy": self.math_strategy,
            "recent_actions": list(self.recent_actions),
            "opponent_profiles": self.opponent_profiles,
            "three_max_solver": self.three_max_solver,
            "assumed_ranges": self.assumed_ranges,
            "hand_analysis": self.hand_analysis,
        }


@dataclass(frozen=True)
class AssistantRecommendation:
    provider: str
    model: str
    recommended_action: str
    recommended_amount: float
    is_legal: bool
    alternatives: tuple[dict[str, Any], ...]
    reasoning: str
    risks: tuple[str, ...]
    assumptions: tuple[str, ...]
    latency_ms: float
    numbers_source: str = "deterministic_engine"
    fallback_used: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "recommended_action": self.recommended_action,
            "recommended_amount": self.recommended_amount,
            "is_legal": self.is_legal,
            "alternatives": list(self.alternatives),
            "reasoning": self.reasoning,
            "risks": list(self.risks),
            "assumptions": list(self.assumptions),
            "latency_ms": round(self.latency_ms, 3),
            "numbers_source": self.numbers_source,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


def build_assistant_prompt(context: DecisionContext) -> dict[str, str]:
    """Build the system and user prompts for a local or cloud LLM assistant."""
    profile = _GAME_STRATEGY_PROFILES[normalize_game_type(context.game_type)]
    style_note = _PLAY_STYLE_PROMPTS[normalize_play_style(context.play_style)]["prompt_note"]
    system_prompt = (
        "You are the Lugon Poker Lab strategic assistant for No-Limit Texas Hold'em (Single Table). "
        f"{profile['prompt_note']}\n"
        f"{style_note}\n"
        "REQUIRED RULES:\n"
        "1. Do not invent probabilities, equity, pot odds, EV, or sample sizes. Use only the values "
        "provided by the mathematical engine in the input JSON.\n"
        "2. Choose only an action listed in 'legal_actions' and a 'recommended_amount' within [min_amount, max_amount]. The Hero's remaining stack and SPR are hard limits: never recommend investing more than the stack; when a bet or raise uses the full stack, identify it as ALL-IN/SHOVE in the explanation.\n"
        "3. Compare the recommended action with at least one legal alternative, explaining benefits, risks, and assumptions.\n"
        "4. If opponent profiles have 'insufficient_sample' (<20 opportunities), do not make aggressive exploitative adjustments "
        "based on those frequencies.\n"
        "5. 'pot' is the current total and includes blinds, antes, and chips already committed; never add the ante again. "
        "'to_call' is only the additional amount the Hero must put in now, not an ante or chips already committed. "
        "If 'hero_position' is UNKNOWN, do not invent a position.\n"
        "6. Respond ONLY with a valid JSON object using these keys: "
        "'recommended_action' (string), 'recommended_amount' (number), "
        "'alternatives' (list of objects with 'action', 'amount', 'pros', 'cons'), "
        "'reasoning' (string in English), 'risks' (list of strings), and 'assumptions' (list of strings). "
        "Write every user-facing explanation in English."
    )
    if context.three_max_solver:
        system_prompt += (
            "\n\n3-MAX ICM SOLVER MODE IS ACTIVE: 'three_max_solver' contains the authoritative numerical solution "
            "for this hand, position, node, stacks, and payouts. Do not recalculate or alter its frequencies. Explain the solver's recommended action "
            "and mixing frequency; do not present a mixed action as 100% pure. The program will set the final action and amount from the solver, "
            "so focus on the explanation, risks, and assumptions. If the solution did not converge on the sample, state clearly that it is approximate."
        )
    user_prompt = json.dumps(context.to_dict(), ensure_ascii=False, indent=2)
    return {"system": system_prompt, "user": user_prompt}


def validate_and_clamp_action(
    action: str,
    amount: float,
    legal_actions: tuple[LegalAction, ...],
) -> tuple[str, float, bool]:
    """Check whether (action, amount) is legal; if not, clamp or select the safest legal fallback."""
    legal_map = {item.action: item for item in legal_actions}
    clean_action = str(action).strip().lower()
    if clean_action in {"all-in", "all in", "all_in", "allin", "shove", "jam"}:
        push_action = next((name for name in ("raise", "bet", "all_in") if name in legal_map), None)
        if push_action is not None:
            return push_action, legal_map[push_action].max_amount, True
        if "call" in legal_map:
            # In a short-stack spot an all-in response may only be a legal call.
            return "call", legal_map["call"].max_amount, False
    if clean_action in legal_map:
        spec = legal_map[clean_action]
        if clean_action in ("fold", "check"):
            return clean_action, 0.0, abs(amount) <= 1e-6
        if clean_action == "call":
            is_exact = abs(amount - spec.min_amount) <= 1e-4 or amount == 0.0
            return "call", spec.min_amount, is_exact
        # bet or raise
        if spec.min_amount - 1e-6 <= amount <= spec.max_amount + 1e-6:
            clamped = min( max(amount, spec.min_amount), spec.max_amount )
            return clean_action, round(clamped, 2), True
        clamped = min(max(amount, spec.min_amount), spec.max_amount)
        return clean_action, round(clamped, 2), False

    # Fallback if action itself is not legal
    if "check" in legal_map:
        return "check", 0.0, False
    if "fold" in legal_map:
        return "fold", 0.0, False
    first = legal_actions[0]
    return first.action, first.min_amount, False


def _analyze_board_and_hand(
    hero_cards: tuple[str, str],
    board: tuple[str, ...],
    hero_position: str,
) -> dict[str, Any]:
    """Classify board texture, relative position, made hand tier, and draws."""
    pos = str(hero_position or "").strip().upper()
    is_ip = pos in ("BTN", "CO", "HJ")
    if len(board) < 3:
        return {
            "street": "preflop",
            "is_ip": is_ip,
            "board_texture": "preflop",
            "hand_tier": "preflop",
            "has_flush_draw": False,
            "has_straight_draw": False,
        }

    try:
        hero = parse_cards(hero_cards)
        bd = parse_cards(board)
    except ValueError:
        return {
            "street": "postflop",
            "is_ip": is_ip,
            "board_texture": "unknown",
            "hand_tier": "unknown",
            "has_flush_draw": False,
            "has_straight_draw": False,
        }

    bd_ranks = [card.rank for card in bd]
    bd_suits = [card.suit for card in bd]
    max_bd_rank = max(bd_ranks)
    bd_rank_counts: dict[int, int] = {}
    for rank in bd_ranks:
        bd_rank_counts[rank] = bd_rank_counts.get(rank, 0) + 1
    bd_suit_counts: dict[str, int] = {}
    for suit in bd_suits:
        bd_suit_counts[suit] = bd_suit_counts.get(suit, 0) + 1

    is_monotone = max(bd_suit_counts.values()) >= 3
    is_two_tone = max(bd_suit_counts.values()) == 2
    is_paired_board = max(bd_rank_counts.values()) >= 2
    unique_bd_ranks = sorted(set(bd_ranks))
    is_connected = any(
        unique_bd_ranks[idx + 2] - unique_bd_ranks[idx] <= 4
        for idx in range(len(unique_bd_ranks) - 2)
    )

    if is_monotone:
        texture = "monotone"
    elif is_connected and is_two_tone:
        texture = "wet"
    elif is_paired_board:
        texture = "paired"
    elif is_two_tone or is_connected:
        texture = "two_tone"
    else:
        texture = "dry"

    rank_tuple = best_hand((*hero, *bd))
    cat = rank_tuple[0]
    hero_ranks = {card.rank for card in hero}
    bd_rank_set = set(bd_ranks)

    if cat >= 3:
        # Three of a kind or better: check if trips/full are entirely on the board.
        if cat == 3 and max(bd_rank_counts.values()) >= 3:
            hand_tier = "board_pair_only"
        else:
            hand_tier = "monster"
    elif cat == 2:
        # Two pair: if board is paired and hero only holds 1 pair on board with no pocket pair/hit, still moderate
        if is_paired_board and not (hero[0].rank == hero[1].rank or len(hero_ranks & bd_rank_set) >= 1):
            hand_tier = "board_pair_only"
        elif is_paired_board and len(hero_ranks & bd_rank_set) == 1 and hero[0].rank != hero[1].rank:
            pair_rank = next(iter(hero_ranks & bd_rank_set))
            hand_tier = "strong_pair" if pair_rank >= max_bd_rank else "medium_pair"
        else:
            hand_tier = "monster"
    elif cat == 1:
        pair_rank = rank_tuple[1]
        if hero[0].rank == hero[1].rank and hero[0].rank == pair_rank:
            hand_tier = "strong_pair" if pair_rank > max_bd_rank else "medium_pair"
        elif pair_rank in hero_ranks and pair_rank in bd_rank_set:
            hand_tier = "strong_pair" if pair_rank == max_bd_rank else "medium_pair"
        else:
            hand_tier = "board_pair_only"
    else:
        hand_tier = "high_card"

    has_flush_draw = False
    has_straight_draw = False
    if len(bd) in (3, 4):
        all_cards = (*hero, *bd)
        for suit in ("c", "d", "h", "s"):
            suit_cards = [c for c in all_cards if c.suit == suit]
            hero_suit_cards = [c for c in hero if c.suit == suit]
            if len(suit_cards) == 4 and len(hero_suit_cards) >= 1:
                has_flush_draw = True
                break

        all_ranks = {c.rank for c in all_cards}
        if 14 in all_ranks:
            all_ranks.add(1)
        hero_draw_ranks = set(hero_ranks)
        if 14 in hero_draw_ranks:
            hero_draw_ranks.add(1)
        for low in range(1, 11):
            window = set(range(low, low + 5))
            present = all_ranks & window
            if len(present) == 4 and (hero_draw_ranks & window):
                # 4-to-a-straight with at least one hero card; check open-ended vs gutshot
                missing = next(iter(window - present))
                if missing in (low, low + 4) and low not in (1, 10):
                    has_straight_draw = True
                    break

    return {
        "street": "flop" if len(bd) == 3 else ("turn" if len(bd) == 4 else "river"),
        "is_ip": is_ip,
        "board_texture": texture,
        "hand_tier": hand_tier,
        "has_flush_draw": has_flush_draw,
        "has_straight_draw": has_straight_draw,
    }


def _heuristic_recommendation(context: DecisionContext, start_time: float) -> AssistantRecommendation:
    legal_map = {item.action: item for item in context.legal_actions}
    eq = float(context.equity.get("share", 0.0))
    pot_odds = context.math_strategy.get("pot_odds")
    call_ev = context.math_strategy.get("call_ev")
    game_key = normalize_game_type(context.game_type)
    profile = _GAME_STRATEGY_PROFILES[game_key]
    game_label = profile["label"]

    hand_info = context.hand_analysis or _analyze_board_and_hand(
        context.hero_cards, context.board, context.hero_position
    )
    is_postflop = hand_info["street"] not in ("preflop",)
    is_ip = bool(hand_info.get("is_ip"))
    texture = str(hand_info.get("board_texture", "preflop"))
    hand_tier = str(hand_info.get("hand_tier", "preflop"))
    has_draw = bool(hand_info.get("has_flush_draw") or hand_info.get("has_straight_draw"))

    alternatives: list[dict[str, Any]] = []
    if context.assumed_ranges:
        range_summaries = [
            f"{name}: {info.get('label', info.get('range_text', '*'))}"
            for name, info in context.assumed_ranges.items()
        ]
        risks: list[str] = [
            "Contextually estimated ranges (" + "; ".join(range_summaries) + ").",
        ]
    else:
        risks = [
            "Calculated equity assumes uniform ranges by physical combination for the active opponents.",
        ]
    assumptions: list[str] = [
        f"Active strategy mode: {game_label}.",
        "Immediate EV does not subtract rake or model complete future re-raise trees.",
    ]

    # Check opponent profile notes
    profile_notes: list[str] = []
    for opp_name, prof in context.opponent_profiles.items():
        if not prof:
            continue
        vpip_info = prof.get("vpip", {})
        n_opp = vpip_info.get("opportunities", 0)
        adequacy = vpip_info.get("sample_adequacy", "no_observations")
        if n_opp >= 20 and vpip_info.get("empirical_rate") is not None:
            ci = vpip_info.get("wilson_95_ci") or [None, None]
            profile_notes.append(
                f"{opp_name}: VPIP {vpip_info['empirical_rate']:.0%} (95% CI {ci[0]:.0%}–{ci[1]:.0%}, n={n_opp})."
            )
        elif n_opp > 0:
            assumptions.append(
                f"The sample for {opp_name} ({n_opp} hands, {adequacy}) is too small for strong exploitative adjustments."
            )

    if context.to_call == 0:
        # Dynamic c-bet / value bet threshold & sizing
        bet_threshold = float(profile["bet_eq"])
        if is_postflop and is_ip and texture == "dry" and hand_tier in ("strong_pair", "monster"):
            bet_threshold -= 0.04
        if hand_tier == "board_pair_only":
            bet_threshold = max(bet_threshold, 0.64)

        semi_bluff = (
            "bet" in legal_map
            and is_postflop
            and hand_info["street"] in ("flop", "turn")
            and has_draw
            and is_ip
            and eq >= 0.36
            and (context.spr is None or context.spr >= 1.5)
        )

        if "bet" in legal_map and (eq >= bet_threshold or semi_bluff):
            bet_spec = legal_map["bet"]
            if texture in ("wet", "monotone"):
                target_frac = max(profile["bet_pot_frac"], 0.70)
            elif texture == "dry" and is_ip and hand_tier != "monster":
                target_frac = 0.40
            elif semi_bluff and eq < bet_threshold:
                target_frac = 0.50
            else:
                target_frac = profile["bet_pot_frac"]
            pot_pct = int(round(target_frac * 100))

            if context.spr is not None and context.spr <= 1.0:
                target_bet = bet_spec.max_amount
            else:
                target_bet = round(
                    max(bet_spec.min_amount, min(bet_spec.max_amount, context.pot * target_frac)),
                    2,
                )
            rec_action = "bet"
            rec_amount = target_bet
            all_in_bet = rec_amount >= context.hero_stack - 0.01
            sizing_text = (
                f" The remaining stack ({context.hero_stack:.2f}) is short relative to the pot (SPR {context.spr:.2f}), so this bet commits the full stack."
                if all_in_bet and context.spr is not None and context.spr <= 1.0
                else (" The maximum legal size equals the remaining stack." if all_in_bet else "")
            )
            if semi_bluff and eq < bet_threshold:
                reasoning = (
                    f"[{game_label}] With no bet to call, in position, and holding a strong draw (equity {eq:.1%}, {texture} board), "
                    f"betting {rec_amount:.2f} (~{pot_pct}% of the {context.pot:.2f} pot) works as a semi-bluff with immediate fold equity."
                    f"{sizing_text}"
                )
            else:
                board_note = f" em board {texture}" if is_postflop else ""
                reasoning = (
                    f"[{game_label}] With no bet to call and high estimated equity ({eq:.1%}{board_note}), "
                    f"betting {rec_amount:.2f} (~{pot_pct}% of the {context.pot:.2f} pot) extracts value and denies free equity realization."
                    f"{sizing_text}"
                )
            alternatives.append(
                {
                    "action": "check",
                    "amount": 0.0,
                    "pros": "Controls the pot size and avoids facing a check-raise.",
                    "cons": "Gives drawing hands a free card.",
                }
            )
            risks.append("Betting reopens the action and exposes the hand to an opponent's raise.")
        else:
            rec_action = "check"
            rec_amount = 0.0
            extra_note = (
                " The pair is on the board only, so its showdown value is fragile."
                if hand_tier == "board_pair_only"
                else ""
            )
            reasoning = (
                f"[{game_label}] There is no cost to see the next card (to_call=0), and estimated equity is {eq:.1%}. "
                f"Checking realizes equity for free without inflating the pot.{extra_note}"
            )
            if "bet" in legal_map:
                bet_spec = legal_map["bet"]
                probe = round(max(bet_spec.min_amount, min(bet_spec.max_amount, context.pot * 0.45)), 2)
                alternatives.append(
                    {
                        "action": "bet",
                        "amount": probe,
                        "pros": "Creates immediate fold equity against marginal hands.",
                        "cons": "Invests chips with moderate or low equity if called.",
                    }
                )
    else:
        # Facing a bet/raise
        po_val = float(pot_odds) if pot_odds is not None else 1.0
        cev_val = float(call_ev) if call_ev is not None else -context.to_call
        stack_commit = (context.to_call / context.hero_stack) if context.hero_stack > 0 else 1.0
        icm_buffer = 0.0
        if game_key == "tournament":
            if stack_commit >= 0.35:
                icm_buffer = profile["icm_high"]
            elif context.hero_stack <= 20.0 and stack_commit >= 0.20:
                icm_buffer = profile["icm_low"]
        elif game_key == "spin_and_go":
            # In Spin & Go multiway all-in spots (without solver) or OOP postflop with future streets,
            # add a realistic survival / equity-realization buffer.
            if not is_postflop and len(context.opponent_profiles) >= 2 and stack_commit >= 0.85:
                icm_buffer = 0.02
            elif (
                is_postflop
                and hand_info["street"] in ("flop", "turn")
                and not is_ip
                and context.hero_stack > context.to_call * 1.5
                and hand_tier in ("high_card", "board_pair_only", "medium_pair")
                and not has_draw
            ):
                icm_buffer = 0.015

        can_value_raise = (
            "raise" in legal_map
            and eq >= profile["raise_eq"]
            and eq > po_val + profile["raise_edge"]
            and not (hand_tier == "board_pair_only" and eq < 0.72)
        )

        if can_value_raise:
            raise_spec = legal_map["raise"]
            raise_frac = max(profile["raise_pot_frac"], 0.85) if texture in ("wet", "monotone") else profile["raise_pot_frac"]
            if context.spr is not None and context.spr <= 1.0:
                target_raise = raise_spec.max_amount
            else:
                target_raise = round(
                    max(raise_spec.min_amount, min(raise_spec.max_amount, context.to_call + context.pot * raise_frac)),
                    2,
                )
            rec_action = "raise"
            rec_amount = target_raise
            all_in_raise = rec_amount >= context.hero_stack - 0.01
            sizing_text = (
                f" The remaining stack ({context.hero_stack:.2f}) is short relative to the pot (SPR {context.spr:.2f}), so this raise commits the full stack."
                if all_in_raise and context.spr is not None and context.spr <= 1.0
                else (" The maximum legal size equals the remaining stack." if all_in_raise else "")
            )
            reasoning = (
                f"[{game_label}] Estimated equity ({eq:.1%}) is well above the pot odds ({po_val:.1%}, call EV {cev_val:+.2f}). "
                f"Raising (+{rec_amount:.2f}) maximizes value against the opponent's range.{sizing_text}"
            )
            alternatives.append(
                {
                    "action": "call",
                    "amount": legal_map["call"].min_amount,
                    "pros": f"Keeps worse hands and bluffs in the pot with immediate EV of {cev_val:+.2f}.",
                    "cons": "Allows the opponent to realize equity at no additional cost.",
                }
            )
            risks.append("An opponent's re-raise isolates the Hero against the very top of their range.")
        elif cev_val > 1e-9 and eq >= po_val + icm_buffer and "call" in legal_map:
            rec_action = "call"
            rec_amount = legal_map["call"].min_amount
            icm_text = f" (ICM/equity-realization buffer of +{icm_buffer:.1%} satisfied)" if icm_buffer > 0 else ""
            reasoning = (
                f"[{game_label}] Estimated equity ({eq:.1%}) is above the pot odds ({po_val:.1%}){icm_text}, "
                f"giving the call an incremental EV of {cev_val:+.2f} chips."
            )
            alternatives.append(
                {
                    "action": "fold",
                    "amount": 0.0,
                    "pros": "Zero variance and immediate stack preservation.",
                    "cons": f"Gives up a decision with positive estimated EV ({cev_val:+.2f}).",
                }
            )
            risks.append("Future bets on later streets may reduce realized equity.")
        else:
            rec_action = "fold" if "fold" in legal_map else context.legal_actions[0].action
            rec_amount = 0.0
            if cev_val > 1e-9 and icm_buffer > 0:
                reasoning = (
                    f"[{game_label}] Although immediate chip EV is slightly positive ({cev_val:+.2f}), "
                    f"equity ({eq:.1%}) does not meet the required survival/ICM margin ({po_val + icm_buffer:.1%}) "
                    "for committing this portion of the stack."
                )
            else:
                reasoning = (
                    f"[{game_label}] Estimated equity ({eq:.1%}) is below the required pot odds ({po_val:.1%}), "
                    f"resulting in negative immediate call EV ({cev_val:+.2f} chips)."
                )
            if "call" in legal_map:
                alternatives.append(
                    {
                        "action": "call",
                        "amount": legal_map["call"].min_amount,
                        "pros": "Keeps the possibility of improving on later streets (implied odds).",
                        "cons": f"Commits chips with insufficient EV ({cev_val:+.2f}) for the {game_label} format.",
                    }
                )

    if profile_notes:
        reasoning += " Player profile context: " + " ".join(profile_notes)

    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    return AssistantRecommendation(
        provider="heuristic",
        model="lugon-baseline-v2",
        recommended_action=rec_action,
        recommended_amount=rec_amount,
        is_legal=True,
        alternatives=tuple(alternatives),
        reasoning=reasoning,
        risks=tuple(risks),
        assumptions=tuple(assumptions),
        latency_ms=elapsed_ms,
    )


def _three_max_solver_recommendation(
    context: DecisionContext,
    start_time: float,
    *,
    llm_reasoning: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    extra_risks: tuple[str, ...] = (),
) -> AssistantRecommendation:
    advice = context.three_max_solver or {}
    engine_name = str(advice.get("engine", "lugon_three_max_icm_cfr_plus"))
    resolved_provider = provider or (
        "three_max_unequal_model" if engine_name == "lugon_three_max_unequal_model" else "three_max_icm_solver"
    )
    solver_action = str(advice.get("solver_action", "push"))
    action_frequency = float(advice.get("action_frequency", 0.0))
    fold_frequency = float(advice.get("fold_frequency", 1.0))
    selected_solver_action = solver_action if action_frequency >= 0.5 else "fold"
    legal_map = {item.action: item for item in context.legal_actions}
    risks = list(extra_risks)

    if selected_solver_action == "push":
        app_action = next((name for name in ("raise", "bet", "all_in") if name in legal_map), None)
        if app_action is None:
            action, amount, is_legal = validate_and_clamp_action("fold", 0.0, context.legal_actions)
            risks.append("The solver's shove is not available among this capture's legal actions; a legal fallback was used.")
        else:
            action, amount, is_legal = validate_and_clamp_action(
                app_action,
                legal_map[app_action].max_amount,
                context.legal_actions,
            )
    elif selected_solver_action == "call":
        if "call" in legal_map:
            action, amount, is_legal = validate_and_clamp_action(
                "call",
                legal_map["call"].min_amount,
                context.legal_actions,
            )
        else:
            action, amount, is_legal = validate_and_clamp_action("fold", 0.0, context.legal_actions)
            risks.append("The solver's call is not available among this capture's legal actions; a legal fallback was used.")
    else:
        action, amount, is_legal = validate_and_clamp_action("fold", 0.0, context.legal_actions)

    node_label = str(advice.get("node_label", advice.get("node", "3-max node")))
    hand_class = str(advice.get("hand_class", "hand"))
    selected_frequency = action_frequency if selected_solver_action != "fold" else fold_frequency
    source_title = "3-Max ML Model" if engine_name == "lugon_three_max_unequal_model" else "3-Max ICM Solver"
    reasoning = (
        f"{source_title} ({node_label}): {hand_class} — {solver_action} {action_frequency:.0%}, "
        f"fold {fold_frequency:.0%}. The selected action is {action} ({selected_frequency:.0%} strategy frequency)."
    )
    if llm_reasoning:
        reasoning += f"\n\nAgent explanation: {llm_reasoning.strip()}"

    convergence = advice.get("convergence") if isinstance(advice.get("convergence"), dict) else {}
    if not convergence.get("converged_on_sample", False):
        risks.append("The solver did not reach its tolerance on this sample; treat these frequencies as an approximation.")
    assumptions_raw = advice.get("assumptions")
    assumptions = tuple(str(value) for value in assumptions_raw) if isinstance(assumptions_raw, list) else ()

    if solver_action == "push":
        push_spec = next((legal_map[name] for name in ("raise", "bet", "all_in") if name in legal_map), None)
        mix_action = push_spec.action if push_spec else "push"
        mix_amount = push_spec.max_amount if push_spec else 0.0
    else:
        mix_action = "call"
        mix_amount = legal_map["call"].min_amount if "call" in legal_map else 0.0
    alternatives = (
        {"action": mix_action, "solver_action": solver_action, "amount": mix_amount, "frequency": action_frequency},
        {"action": "fold", "amount": 0.0, "frequency": fold_frequency},
    )
    return AssistantRecommendation(
        provider=resolved_provider,
        model=model or engine_name,
        recommended_action=action,
        recommended_amount=amount,
        is_legal=is_legal,
        alternatives=alternatives,
        reasoning=reasoning,
        risks=tuple(risks),
        assumptions=assumptions,
        latency_ms=(time.perf_counter() - start_time) * 1000.0,
        numbers_source="three_max_icm_solver",
        fallback_used=False,
        error=None,
    )


def _parse_llm_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if not text:
        raise ValueError("LLM returned empty response content")
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            data = json.loads(text[start : end + 1])
        else:
            raise
    if not isinstance(data, dict):
        raise ValueError("LLM response must be a JSON object")
    return data


class GameAssistant:
    """Unified strategic assistant supporting heuristic baseline, local LLMs, and cloud LLMs."""

    def __init__(
        self,
        provider: str = "heuristic",
        model: str | None = None,
        endpoint: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 45.0,
        transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
    ) -> None:
        valid_providers = {"heuristic", "ollama", "openai_compatible", "gemini"}
        if provider not in valid_providers:
            raise ValueError(f"provider must be one of {sorted(valid_providers)}")
        self.provider = provider
        default_models = {
            "heuristic": "lugon-baseline-v1",
            "ollama": "llama3.1",
            "openai_compatible": "gpt-4o-mini",
            "gemini": "gemini-2.5-flash",
        }
        self.model = model or default_models[provider]
        self.endpoint = endpoint
        self.api_key = api_key
        self.timeout_seconds = timeout_seconds
        self._transport = transport or self._default_http_post

    @staticmethod
    def _default_http_post(
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        timeout: float,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read().decode("utf-8")
            return json.loads(raw)

    def _call_provider(self, prompts: dict[str, str]) -> dict[str, Any]:
        if self.provider == "ollama":
            base = (self.endpoint or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
            url = f"{base}/api/chat"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                "format": "json",
                "stream": False,
                "keep_alive": "10m",
            }
            resp = self._transport(url, {"Content-Type": "application/json"}, payload, self.timeout_seconds)
            msg = resp.get("message", {}) if isinstance(resp, dict) else {}
            content = str(msg.get("content") or "").strip()
            if not content:
                content = str(msg.get("thinking") or "").strip()
            return _parse_llm_json(content)

        if self.provider == "openai_compatible":
            base = (self.endpoint or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
            url = f"{base}/chat/completions"
            token = self.api_key if self.api_key is not None else os.environ.get("OPENAI_API_KEY", "")
            headers = {"Content-Type": "application/json"}
            if token:
                headers["Authorization"] = f"Bearer {token}"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": prompts["system"]},
                    {"role": "user", "content": prompts["user"]},
                ],
                "temperature": 0.1,
                "response_format": {"type": "json_object"},
            }
            resp = self._transport(url, headers, payload, self.timeout_seconds)
            choices = resp.get("choices") or []
            if not choices:
                raise ValueError("OpenAI-compatible endpoint returned no choices")
            content = choices[0].get("message", {}).get("content", "")
            return _parse_llm_json(str(content))

        if self.provider == "gemini":
            key = self.api_key if self.api_key is not None else os.environ.get("GEMINI_API_KEY", "")
            if not key:
                raise ValueError("GEMINI_API_KEY is required for the gemini provider")
            base = (
                self.endpoint
                or f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"
            )
            url = f"{base}?key={key}" if "?" not in base else base
            payload = {
                "systemInstruction": {"parts": [{"text": prompts["system"]}]},
                "contents": [{"role": "user", "parts": [{"text": prompts["user"]}]}],
                "generationConfig": {"temperature": 0.1, "responseMimeType": "application/json"},
            }
            resp = self._transport(url, {"Content-Type": "application/json"}, payload, self.timeout_seconds)
            candidates = resp.get("candidates") or []
            if not candidates:
                raise ValueError("Gemini endpoint returned no candidates")
            parts = candidates[0].get("content", {}).get("parts") or []
            content = parts[0].get("text", "") if parts else ""
            return _parse_llm_json(str(content))

        raise ValueError(f"unsupported provider {self.provider!r}")

    def recommend(self, context: DecisionContext, fallback_on_error: bool = True) -> AssistantRecommendation:
        start_time = time.perf_counter()
        if self.provider == "heuristic":
            if context.three_max_solver:
                return _three_max_solver_recommendation(context, start_time)
            return _heuristic_recommendation(context, start_time)

        prompts = build_assistant_prompt(context)
        try:
            parsed = self._call_provider(prompts)
            raw_action = str(parsed.get("recommended_action", ""))
            raw_amount = float(parsed.get("recommended_amount", 0.0) or 0.0)
            action, amount, is_legal = validate_and_clamp_action(raw_action, raw_amount, context.legal_actions)
            alternatives_raw = parsed.get("alternatives")
            alternatives = (
                tuple(item for item in alternatives_raw if isinstance(item, dict))
                if isinstance(alternatives_raw, list)
                else ()
            )
            reasoning = str(parsed.get("reasoning") or "Recommendation generated by the language model.")
            risks_raw = parsed.get("risks")
            risks = tuple(str(x) for x in risks_raw) if isinstance(risks_raw, list) else ()
            assumptions_raw = parsed.get("assumptions")
            assumptions = tuple(str(x) for x in assumptions_raw) if isinstance(assumptions_raw, list) else ()
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            if context.three_max_solver:
                solver_rec = _three_max_solver_recommendation(
                    context,
                    start_time,
                    llm_reasoning=reasoning,
                    provider=self.provider,
                    model=self.model,
                    extra_risks=risks,
                )
                return AssistantRecommendation(
                    provider=solver_rec.provider,
                    model=solver_rec.model,
                    recommended_action=solver_rec.recommended_action,
                    recommended_amount=solver_rec.recommended_amount,
                    is_legal=solver_rec.is_legal,
                    alternatives=solver_rec.alternatives,
                    reasoning=solver_rec.reasoning,
                    risks=solver_rec.risks,
                    assumptions=solver_rec.assumptions,
                    latency_ms=elapsed_ms,
                    numbers_source="three_max_icm_solver",
                    fallback_used=False,
                    error=None,
                )
            return AssistantRecommendation(
                provider=self.provider,
                model=self.model,
                recommended_action=action,
                recommended_amount=amount,
                is_legal=is_legal,
                alternatives=alternatives,
                reasoning=reasoning,
                risks=risks,
                assumptions=assumptions,
                latency_ms=elapsed_ms,
                fallback_used=False,
                error=None,
            )
        except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            if not fallback_on_error:
                raise
            baseline = (
                _three_max_solver_recommendation(context, start_time)
                if context.three_max_solver
                else _heuristic_recommendation(context, start_time)
            )
            return AssistantRecommendation(
                provider=self.provider,
                model=self.model,
                recommended_action=baseline.recommended_action,
                recommended_amount=baseline.recommended_amount,
                is_legal=baseline.is_legal,
                alternatives=baseline.alternatives,
                reasoning=f"[Fallback local ativado: {exc}] {baseline.reasoning}",
                risks=baseline.risks,
                assumptions=baseline.assumptions,
                latency_ms=(time.perf_counter() - start_time) * 1000.0,
                numbers_source=baseline.numbers_source,
                fallback_used=True,
                error=str(exc),
            )
