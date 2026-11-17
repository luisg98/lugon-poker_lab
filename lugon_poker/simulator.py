"""Interactive Single-Table No-Limit Texas Hold'em Simulator integrating Vision, State, Profiles, Math, and Assistant."""

from __future__ import annotations

from dataclasses import dataclass
import random
from statistics import median
import time
from typing import Any, Callable

from .assistant import (
    AssistantRecommendation,
    DecisionContext,
    GameAssistant,
    LegalAction,
    build_assistant_prompt,
    validate_and_clamp_action,
)
from .cards import Card, RANKS, SUITS, best_hand, describe
from .equity import calculate_equity
from .history import ActionRecord, HandHistory, SeatInfo
from .persistence import ProfileStore
from .state import HandState
from .strategy import analyze_strategy
from .synthetic import render_scene
from .vision import analyze_image

POSITIONS_BY_COUNT: dict[int, tuple[str, ...]] = {
    2: ("BTN", "BB"),
    3: ("BTN", "SB", "BB"),
    4: ("BTN", "SB", "BB", "UTG"),
    5: ("BTN", "SB", "BB", "UTG", "CO"),
    6: ("BTN", "SB", "BB", "UTG", "MP", "CO"),
}

VISUAL_SEAT_MAP: dict[int, tuple[int, ...]] = {
    2: (4, 0),
    3: (4, 0, 2),
    4: (4, 5, 1, 3),
    5: (4, 5, 0, 2, 3),
    6: (4, 5, 0, 1, 2, 3),
}


@dataclass(frozen=True)
class SimulatedDecisionRecord:
    hand_id: str
    street: str
    pot: float
    to_call: float
    equity: float
    pot_odds: float | None
    call_ev: float | None
    chosen_action: str
    chosen_amount: float
    assistant_recommendation: AssistantRecommendation
    baseline_decisions: dict[str, tuple[str, float]]
    vision_verified: bool
    vision_latency_ms: float
    equity_latency_ms: float
    assistant_latency_ms: float
    total_latency_ms: float
    llm_prompt_preview: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "street": self.street,
            "pot": self.pot,
            "to_call": self.to_call,
            "equity": self.equity,
            "pot_odds": self.pot_odds,
            "call_ev": self.call_ev,
            "chosen_action": self.chosen_action,
            "chosen_amount": self.chosen_amount,
            "assistant_recommendation": self.assistant_recommendation.to_dict(),
            "baseline_decisions": {
                name: {"action": act, "amount": amt} for name, (act, amt) in self.baseline_decisions.items()
            },
            "vision_verified": self.vision_verified,
            "latency_ms": {
                "vision": round(self.vision_latency_ms, 3),
                "equity": round(self.equity_latency_ms, 3),
                "assistant": round(self.assistant_latency_ms, 3),
                "total": round(self.total_latency_ms, 3),
            },
        }


@dataclass(frozen=True)
class SimulatedHandResult:
    hand_id: str
    history: HandHistory
    winners: tuple[str, ...]
    final_pot: float
    net_chips: dict[str, float]
    hero_decisions: tuple[SimulatedDecisionRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "winners": list(self.winners),
            "final_pot": self.final_pot,
            "net_chips": self.net_chips,
            "history": self.history.to_dict(),
            "hero_decisions": [dec.to_dict() for dec in self.hero_decisions],
        }


def calling_station_baseline(
    legal_actions: tuple[LegalAction, ...],
    to_call: float,
    pot: float,
    equity: float,
) -> tuple[str, float]:
    legal_map = {item.action: item for item in legal_actions}
    if to_call > 0 and "call" in legal_map:
        return "call", legal_map["call"].min_amount
    if "check" in legal_map:
        return "check", 0.0
    return legal_actions[0].action, legal_actions[0].min_amount


def pot_odds_baseline(
    legal_actions: tuple[LegalAction, ...],
    to_call: float,
    pot: float,
    equity: float,
) -> tuple[str, float]:
    legal_map = {item.action: item for item in legal_actions}
    if to_call <= 1e-6:
        if "check" in legal_map:
            return "check", 0.0
        return legal_actions[0].action, legal_actions[0].min_amount
    call_amount = legal_map["call"].min_amount if "call" in legal_map else to_call
    call_ev = equity * (pot + call_amount) - call_amount
    if call_ev > 1e-9 and "call" in legal_map:
        return "call", call_amount
    if "fold" in legal_map:
        return "fold", 0.0
    return legal_actions[0].action, legal_actions[0].min_amount


def tight_aggressive_baseline(
    legal_actions: tuple[LegalAction, ...],
    to_call: float,
    pot: float,
    equity: float,
) -> tuple[str, float]:
    legal_map = {item.action: item for item in legal_actions}
    if to_call <= 1e-6:
        if "bet" in legal_map and equity >= 0.58:
            spec = legal_map["bet"]
            amt = round(max(spec.min_amount, min(spec.max_amount, pot * 0.6)), 2)
            return "bet", amt
        if "raise" in legal_map and equity >= 0.65:
            spec = legal_map["raise"]
            amt = round(max(spec.min_amount, min(spec.max_amount, pot * 0.6)), 2)
            return "raise", amt
        if "check" in legal_map:
            return "check", 0.0
    else:
        call_amount = legal_map["call"].min_amount if "call" in legal_map else to_call
        pot_odds = call_amount / (pot + call_amount) if (pot + call_amount) > 0 else 1.0
        if "raise" in legal_map and equity >= 0.68 and equity > pot_odds + 0.15:
            spec = legal_map["raise"]
            amt = round(max(spec.min_amount, min(spec.max_amount, call_amount + pot * 0.75)), 2)
            return "raise", amt
        if "call" in legal_map and equity >= pot_odds:
            return "call", call_amount
        if "fold" in legal_map:
            return "fold", 0.0
    return legal_actions[0].action, legal_actions[0].min_amount


def _preflop_hand_strength(first: Card, second: Card) -> float:
    """Fast deterministic preflop strength heuristic in [0, 1] for simulated villain bots."""
    high = max(first.rank, second.rank)
    low = min(first.rank, second.rank)
    if high == low:
        return min(0.95, 0.50 + (high - 2) * 0.036)
    suited_bonus = 0.04 if first.suit == second.suit else 0.0
    gap = high - low
    conn_bonus = 0.03 if gap == 1 else (0.015 if gap == 2 else 0.0)
    raw = 0.18 + (high - 2) * 0.028 + (low - 2) * 0.014 + suited_bonus + conn_bonus
    return max(0.05, min(0.88, raw))


def _bot_strength(hole: tuple[Card, Card], board: tuple[Card, ...]) -> float:
    if len(board) < 3:
        return _preflop_hand_strength(hole[0], hole[1])
    rank_tuple = best_hand((*hole, *board))
    category = rank_tuple[0]
    primary = rank_tuple[1] if len(rank_tuple) > 1 else 2
    base_by_cat = {
        0: 0.22,
        1: 0.46,
        2: 0.66,
        3: 0.78,
        4: 0.85,
        5: 0.90,
        6: 0.95,
        7: 0.98,
        8: 0.99,
    }
    return min(0.99, base_by_cat.get(category, 0.3) + (primary - 2) * 0.008)


class TableSimulator:
    """Single-table NLHE simulator; context-manage it when it owns its profile store."""

    def __init__(
        self,
        hero_nickname: str = "HERO",
        opponent_nicknames: tuple[str, ...] = ("VILLAIN", "CALLER"),
        starting_stack: float = 100.0,
        small_blind: float = 1.0,
        big_blind: float = 2.0,
        equity_iterations: int = 2_000,
        use_vision: bool = True,
        assistant: GameAssistant | None = None,
        profile_store: ProfileStore | None = None,
    ) -> None:
        if not 1 <= len(opponent_nicknames) <= 5:
            raise ValueError("TableSimulator supports between 1 and 5 opponents (2 to 6 total seats)")
        self.hero_nickname = hero_nickname
        self.opponent_nicknames = tuple(opponent_nicknames)
        self.all_nicknames = (hero_nickname, *opponent_nicknames)
        if len(set(self.all_nicknames)) != len(self.all_nicknames):
            raise ValueError("player nicknames must be unique")
        self.starting_stack = float(starting_stack)
        self.small_blind = float(small_blind)
        self.big_blind = float(big_blind)
        self.equity_iterations = equity_iterations
        self.use_vision = use_vision
        self.assistant = assistant or GameAssistant(provider="heuristic")
        self._owns_profile_store = profile_store is None
        self.profile_store = profile_store if profile_store is not None else ProfileStore(":memory:")
        self._closed = False

    def close(self) -> None:
        """Close the internally owned profile database, leaving injected stores alone."""
        if self._closed:
            return
        self._closed = True
        if self._owns_profile_store:
            self.profile_store.close()

    def __enter__(self) -> "TableSimulator":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        # Fallback for callers that do not use the context manager or call close().
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def _compute_legal_actions(
        to_call: float,
        remaining_stack: float,
        highest_bet: float,
        big_blind: float,
        raises_on_street: int,
        max_raises: int = 3,
    ) -> tuple[LegalAction, ...]:
        actions: list[LegalAction] = []
        if to_call > 1e-6:
            actions.append(LegalAction("fold", 0.0, 0.0))
            call_amt = round(min(to_call, remaining_stack), 2)
            if call_amt > 0:
                actions.append(LegalAction("call", call_amt, call_amt))
            if remaining_stack > to_call + 1e-6 and raises_on_street < max_raises:
                min_raise_inc = round(min(remaining_stack, to_call + big_blind), 2)
                max_raise_inc = round(remaining_stack, 2)
                if min_raise_inc > call_amt + 1e-6:
                    actions.append(LegalAction("raise", min_raise_inc, max_raise_inc))
        else:
            actions.append(LegalAction("check", 0.0, 0.0))
            if remaining_stack > 1e-6 and raises_on_street < max_raises:
                min_inc = round(min(remaining_stack, big_blind), 2)
                max_inc = round(remaining_stack, 2)
                verb = "bet" if highest_bet <= 1e-6 else "raise"
                actions.append(LegalAction(verb, min_inc, max_inc))
        return tuple(actions)

    def _build_scene(
        self,
        seat_order: list[str],
        positions: dict[str, str],
        stacks: dict[str, float],
        street_bets: dict[str, float],
        folded: set[str],
        active_player: str,
        pot_before_street_bets: float,
        board: list[Card],
        hero_cards: tuple[Card, Card],
    ) -> dict[str, Any]:
        visual_seats = VISUAL_SEAT_MAP[len(seat_order)]
        players_scene = []
        for idx, name in enumerate(seat_order):
            players_scene.append(
                {
                    "nickname": name,
                    "position": positions[name],
                    "stack": int(round(stacks[name])),
                    "bet": int(round(street_bets[name])),
                    "seat": visual_seats[idx],
                    "hero": name == self.hero_nickname,
                    "active": name == active_player,
                    "folded": name in folded,
                }
            )
        return {
            "pot": int(round(pot_before_street_bets)),
            "small_blind": int(round(self.small_blind)),
            "big_blind": int(round(self.big_blind)),
            "board": [str(c) for c in board],
            "hero_cards": [str(hero_cards[0]), str(hero_cards[1])],
            "players": players_scene,
        }

    def simulate_hand(
        self,
        hand_id: str = "SIM_001",
        seed: int = 7,
        button_offset: int = 0,
        hero_action_selector: Callable[
            [DecisionContext, AssistantRecommendation, dict[str, tuple[str, float]]],
            tuple[str, float],
        ]
        | None = None,
    ) -> SimulatedHandResult:
        rng = random.Random(seed)
        deck = [Card(rank, suit) for rank in range(2, 15) for suit in SUITS]
        rng.shuffle(deck)

        n_players = len(self.all_nicknames)
        rotated = [
            self.all_nicknames[(i + button_offset) % n_players]
            for i in range(n_players)
        ]
        pos_names = POSITIONS_BY_COUNT[n_players]
        positions = {rotated[i]: pos_names[i] for i in range(n_players)}

        hole_cards: dict[str, tuple[Card, Card]] = {}
        cursor = 0
        for name in rotated:
            hole_cards[name] = (deck[cursor], deck[cursor + 1])
            cursor += 2
        full_board = tuple(deck[cursor : cursor + 5])

        stacks = {name: self.starting_stack for name in rotated}
        folded: set[str] = set()
        actions: list[ActionRecord] = []
        hero_decisions: list[SimulatedDecisionRecord] = []
        pot = 0.0

        # Identify SB and BB seats
        if n_players == 2:
            sb_player = rotated[0]  # BTN posts SB in heads-up
            bb_player = rotated[1]
            preflop_order = [rotated[0], rotated[1]]
            postflop_order = [rotated[1], rotated[0]]
        else:
            sb_player = rotated[1]
            bb_player = rotated[2]
            preflop_order = rotated[3:] + rotated[:3]
            postflop_order = rotated[1:] + rotated[:1]

        street_bets = {name: 0.0 for name in rotated}

        # Post blinds
        sb_amt = round(min(self.small_blind, stacks[sb_player]), 2)
        stacks[sb_player] -= sb_amt
        street_bets[sb_player] += sb_amt
        pot += sb_amt
        actions.append(ActionRecord("preflop", sb_player, "post_sb", sb_amt))

        bb_amt = round(min(self.big_blind, stacks[bb_player]), 2)
        stacks[bb_player] -= bb_amt
        street_bets[bb_player] += bb_amt
        pot += bb_amt
        actions.append(ActionRecord("preflop", bb_player, "post_bb", bb_amt))

        highest_bet = max(street_bets.values())
        raises_on_street = 0

        board_len_by_street = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}
        final_board: tuple[Card, ...] = ()

        for street in ("preflop", "flop", "turn", "river"):
            active_players = [p for p in rotated if p not in folded]
            if len(active_players) <= 1:
                break

            if street != "preflop":
                street_bets = {name: 0.0 for name in rotated}
                highest_bet = 0.0
                raises_on_street = 0
                final_board = full_board[: board_len_by_street[street]]

            order = preflop_order if street == "preflop" else postflop_order
            needs_to_act = {p for p in order if p not in folded and stacks[p] > 1e-6}

            while needs_to_act:
                progressed = False
                for actor in order:
                    if actor not in needs_to_act or actor in folded or stacks[actor] <= 1e-6:
                        needs_to_act.discard(actor)
                        continue
                    if len([p for p in rotated if p not in folded]) <= 1:
                        needs_to_act.clear()
                        break

                    to_call = round(max(0.0, highest_bet - street_bets[actor]), 2)
                    legal = self._compute_legal_actions(
                        to_call=to_call,
                        remaining_stack=stacks[actor],
                        highest_bet=highest_bet,
                        big_blind=self.big_blind,
                        raises_on_street=raises_on_street,
                    )

                    if actor == self.hero_nickname:
                        t_start = time.perf_counter()
                        current_board = list(final_board)
                        pot_before_street = max(0.0, pot - sum(street_bets.values()))

                        # 1. Vision step
                        t_vis0 = time.perf_counter()
                        vision_verified = False
                        if self.use_vision:
                            scene = self._build_scene(
                                seat_order=rotated,
                                positions=positions,
                                stacks=stacks,
                                street_bets=street_bets,
                                folded=folded,
                                active_player=actor,
                                pot_before_street_bets=pot_before_street,
                                board=current_board,
                                hero_cards=hole_cards[actor],
                            )
                            vis_report = analyze_image(render_scene(scene))
                            vision_verified = (
                                vis_report.get("hand_state") is not None
                                and vis_report.get("active_player") == actor
                                and vis_report.get("board") == [str(c) for c in current_board]
                            )
                        vision_ms = (time.perf_counter() - t_vis0) * 1000.0

                        # 2. Equity & Math Strategy step
                        t_eq0 = time.perf_counter()
                        active_opps = [
                            {"nickname": p, "range": "*"}
                            for p in rotated
                            if p != actor and p not in folded
                        ]
                        pot_excluding_hero_call = round(pot, 2)
                        hand_state = HandState.from_dict(
                            {
                                "hero": {"cards": [str(hole_cards[actor][0]), str(hole_cards[actor][1])]},
                                "board": [str(c) for c in current_board],
                                "pot": pot_excluding_hero_call,
                                "to_call": to_call,
                                "opponents": active_opps,
                            }
                        )
                        eq_res = calculate_equity(
                            hand_state,
                            iterations=self.equity_iterations,
                            seed=seed + len(actions),
                        )
                        strat_res = analyze_strategy(hand_state, eq_res)
                        equity_ms = (time.perf_counter() - t_eq0) * 1000.0

                        # 3. Opponent Profiles & Decision Context
                        stored_profiles = self.profile_store.all_profiles()
                        opp_prof_dict = {
                            opp["nickname"]: stored_profiles[opp["nickname"]].to_dict()
                            if opp["nickname"] in stored_profiles
                            else {}
                            for opp in active_opps
                        }
                        max_opp_stack = max(
                            (stacks[opp["nickname"]] for opp in active_opps),
                            default=0.0,
                        )
                        eff_stack = round(min(stacks[actor], max_opp_stack), 2)
                        spr = round(eff_stack / pot, 2) if pot > 0 else None
                        hero_desc = (
                            describe(best_hand((*hole_cards[actor], *final_board)))
                            if len(final_board) >= 3
                            else None
                        )

                        context = DecisionContext(
                            hand_id=hand_id,
                            street=street,
                            hero_nickname=actor,
                            hero_position=positions[actor],
                            hero_cards=(str(hole_cards[actor][0]), str(hole_cards[actor][1])),
                            board=tuple(str(c) for c in current_board),
                            hero_hand_description=hero_desc,
                            pot=round(pot, 2),
                            to_call=to_call,
                            hero_stack=round(stacks[actor], 2),
                            effective_stack=eff_stack,
                            spr=spr,
                            legal_actions=legal,
                            equity={
                                "share": eq_res.equity,
                                "win_rate": eq_res.win_rate,
                                "tie_rate": eq_res.tie_rate,
                                "loss_rate": eq_res.loss_rate,
                                "samples": eq_res.samples,
                                "method": eq_res.method,
                            },
                            math_strategy=strat_res.to_dict(),
                            recent_actions=tuple(
                                {
                                    "street": a.street,
                                    "player": a.player,
                                    "action": a.action,
                                    "amount": a.amount,
                                }
                                for a in actions[-8:]
                            ),
                            opponent_profiles=opp_prof_dict,
                        )

                        # 4. Strategic Assistant & Baselines
                        recommendation = self.assistant.recommend(context)
                        baselines = {
                            "pot_odds": pot_odds_baseline(legal, to_call, pot, eq_res.equity),
                            "tight_aggressive": tight_aggressive_baseline(legal, to_call, pot, eq_res.equity),
                            "calling_station": calling_station_baseline(legal, to_call, pot, eq_res.equity),
                        }
                        if hero_action_selector is not None:
                            chosen_act, chosen_amt = hero_action_selector(context, recommendation, baselines)
                            chosen_act, chosen_amt, _ = validate_and_clamp_action(chosen_act, chosen_amt, legal)
                        else:
                            chosen_act = recommendation.recommended_action
                            chosen_amt = recommendation.recommended_amount

                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        hero_decisions.append(
                            SimulatedDecisionRecord(
                                hand_id=hand_id,
                                street=street,
                                pot=round(pot, 2),
                                to_call=to_call,
                                equity=eq_res.equity,
                                pot_odds=strat_res.pot_odds,
                                call_ev=strat_res.call_ev,
                                chosen_action=chosen_act,
                                chosen_amount=chosen_amt,
                                assistant_recommendation=recommendation,
                                baseline_decisions=baselines,
                                vision_verified=vision_verified,
                                vision_latency_ms=vision_ms,
                                equity_latency_ms=equity_ms,
                                assistant_latency_ms=recommendation.latency_ms,
                                total_latency_ms=total_ms,
                                llm_prompt_preview=build_assistant_prompt(context),
                            )
                        )
                    else:
                        # Opponent bot policy: seat-dependent mix of TAG and calling-station tendencies
                        strength = _bot_strength(hole_cards[actor], final_board)
                        if "CALL" in actor.upper():
                            # Looser calling tendency
                            adj_strength = min(0.95, strength + 0.12)
                            chosen_act, chosen_amt = pot_odds_baseline(legal, to_call, pot, adj_strength)
                        else:
                            chosen_act, chosen_amt = tight_aggressive_baseline(legal, to_call, pot, strength)

                    # Apply chosen action to table state
                    needs_to_act.discard(actor)
                    progressed = True
                    if chosen_act == "fold":
                        folded.add(actor)
                        actions.append(ActionRecord(street, actor, "fold", 0.0))
                    elif chosen_act == "check":
                        actions.append(ActionRecord(street, actor, "check", 0.0))
                    elif chosen_act == "call":
                        stacks[actor] = round(stacks[actor] - chosen_amt, 2)
                        street_bets[actor] = round(street_bets[actor] + chosen_amt, 2)
                        pot = round(pot + chosen_amt, 2)
                        actions.append(ActionRecord(street, actor, "call", chosen_amt))
                    elif chosen_act in ("bet", "raise"):
                        stacks[actor] = round(stacks[actor] - chosen_amt, 2)
                        street_bets[actor] = round(street_bets[actor] + chosen_amt, 2)
                        pot = round(pot + chosen_amt, 2)
                        highest_bet = max(highest_bet, street_bets[actor])
                        raises_on_street += 1
                        actions.append(ActionRecord(street, actor, chosen_act, chosen_amt))
                        needs_to_act = {
                            p
                            for p in order
                            if p != actor and p not in folded and stacks[p] > 1e-6
                        }
                        break

                if not progressed:
                    break

        # Settle pot and determine winners
        remaining = [p for p in rotated if p not in folded]
        if len(remaining) == 1:
            winners = (remaining[0],)
        else:
            if len(final_board) < 5:
                final_board = full_board
            ranks = {p: best_hand((*hole_cards[p], *final_board)) for p in remaining}
            best_rank = max(ranks.values())
            winners = tuple(p for p in remaining if ranks[p] == best_rank)

        share = pot / len(winners)
        net_chips: dict[str, float] = {}
        for name in rotated:
            invested = self.starting_stack - stacks[name]
            won = share if name in winners else 0.0
            net_chips[name] = round(won - invested, 2)

        seat_infos = tuple(
            SeatInfo(
                nickname=name,
                position=positions[name],
                starting_stack=self.starting_stack,
                hole_cards=hole_cards[name] if (name == self.hero_nickname or len(remaining) > 1 and name in remaining) else None,
            )
            for name in rotated
        )
        history = HandHistory(
            hand_id=hand_id,
            small_blind=self.small_blind,
            big_blind=self.big_blind,
            players=seat_infos,
            board=final_board,
            actions=tuple(actions),
        )
        # Validate history via replay and store in ProfileStore
        history.replay()
        self.profile_store.import_hands([history])

        return SimulatedHandResult(
            hand_id=hand_id,
            history=history,
            winners=winners,
            final_pot=round(pot, 2),
            net_chips=net_chips,
            hero_decisions=tuple(hero_decisions),
        )

    def evaluate_session(self, hands: int = 10, seed: int = 7) -> dict[str, Any]:
        """Simulate multiple hands, comparing assistant recommendations against baselines and measuring latency."""
        if hands < 1:
            raise ValueError("hands must be at least 1")

        hand_results: list[SimulatedHandResult] = []
        all_decisions: list[SimulatedDecisionRecord] = []
        cumulative_net = {name: 0.0 for name in self.all_nicknames}

        for idx in range(hands):
            hand_id = f"SIM_{idx + 1:04d}"
            res = self.simulate_hand(
                hand_id=hand_id,
                seed=seed + idx * 101,
                button_offset=idx,
            )
            hand_results.append(res)
            all_decisions.extend(res.hero_decisions)
            for name, delta in res.net_chips.items():
                cumulative_net[name] = round(cumulative_net[name] + delta, 2)

        n_dec = len(all_decisions)
        agreement_counts = {"pot_odds": 0, "tight_aggressive": 0, "calling_station": 0}
        vision_ok_count = 0
        vis_lat: list[float] = []
        eq_lat: list[float] = []
        ast_lat: list[float] = []
        tot_lat: list[float] = []

        for dec in all_decisions:
            if dec.vision_verified:
                vision_ok_count += 1
            vis_lat.append(dec.vision_latency_ms)
            eq_lat.append(dec.equity_latency_ms)
            ast_lat.append(dec.assistant_latency_ms)
            tot_lat.append(dec.total_latency_ms)
            for b_name in agreement_counts:
                if dec.baseline_decisions[b_name][0] == dec.assistant_recommendation.recommended_action:
                    agreement_counts[b_name] += 1

        def _lat_stats(values: list[float]) -> dict[str, float | None]:
            if not values:
                return {"mean_ms": None, "median_ms": None, "max_ms": None}
            return {
                "mean_ms": round(sum(values) / len(values), 3),
                "median_ms": round(float(median(values)), 3),
                "max_ms": round(max(values), 3),
            }

        return {
            "hands_simulated": hands,
            "hero_decisions_count": n_dec,
            "assistant_provider": self.assistant.provider,
            "assistant_model": self.assistant.model,
            "vision_verification_rate": (vision_ok_count / n_dec) if n_dec > 0 else None,
            "baseline_agreement_rate": {
                b_name: (count / n_dec) if n_dec > 0 else None
                for b_name, count in agreement_counts.items()
            },
            "cumulative_net_chips": cumulative_net,
            "latency_summary": {
                "vision": _lat_stats(vis_lat),
                "equity": _lat_stats(eq_lat),
                "assistant": _lat_stats(ast_lat),
                "total": _lat_stats(tot_lat),
            },
            "profiles_snapshot": {
                name: prof.to_dict() for name, prof in self.profile_store.all_profiles().items()
            },
        }
