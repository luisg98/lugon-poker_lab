"""Hand history representation, validation, action replay, and capture transition tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

from .cards import Card, parse_cards

STREETS = ("preflop", "flop", "turn", "river")
VALID_POSITIONS = frozenset({"BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "LJ", "HJ", "MP", "CO"})
BLIND_ACTIONS = frozenset({"post_sb", "post_bb", "post_ante"})
VOLUNTARY_ACTIONS = frozenset({"fold", "check", "call", "bet", "raise"})
ALL_ACTIONS = BLIND_ACTIONS | VOLUNTARY_ACTIONS


def _finite_non_negative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    return float(value)


def _finite_positive(value: Any, name: str) -> float:
    amount = _finite_non_negative(value, name)
    if amount <= 0:
        raise ValueError(f"{name} must be positive")
    return amount


@dataclass(frozen=True)
class SeatInfo:
    nickname: str
    position: str
    starting_stack: float
    hole_cards: tuple[Card, Card] | None = None


@dataclass(frozen=True)
class ActionRecord:
    street: str
    player: str
    action: str
    amount: float = 0.0


@dataclass(frozen=True)
class AnnotatedAction:
    street: str
    player: str
    position: str
    action: str
    amount: float
    pot_before: float
    to_call_before: float
    raises_before: int
    pot_ratio: float | None


@dataclass(frozen=True)
class PlayerHandSummary:
    hand_id: str
    nickname: str
    position: str
    vpip_opportunity: bool
    did_vpip: bool
    pfr_opportunity: bool
    did_pfr: bool
    three_bet_opportunity: bool
    did_three_bet: bool
    bets: int
    raises: int
    calls: int
    checks: int
    folds: int
    postflop_bets: int
    postflop_raises: int
    postflop_calls: int
    postflop_checks: int
    postflop_folds: int
    saw_flop: bool
    went_to_showdown: bool


@dataclass(frozen=True)
class HandReplay:
    hand_id: str
    final_pot: float
    annotated_actions: tuple[AnnotatedAction, ...]
    player_summaries: dict[str, PlayerHandSummary]


@dataclass(frozen=True)
class HandHistory:
    hand_id: str
    small_blind: float
    big_blind: float
    players: tuple[SeatInfo, ...]
    board: tuple[Card, ...]
    actions: tuple[ActionRecord, ...]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HandHistory":
        if not isinstance(data, dict):
            raise ValueError("hand history must be a JSON object")
        hand_id = data.get("hand_id")
        if not isinstance(hand_id, str) or not hand_id.strip():
            raise ValueError("hand_id must be a non-empty string")

        small_blind = _finite_positive(data.get("small_blind"), "small_blind")
        big_blind = _finite_positive(data.get("big_blind"), "big_blind")
        if big_blind < small_blind:
            raise ValueError("big_blind must be greater than or equal to small_blind")

        players_raw = data.get("players")
        if not isinstance(players_raw, list) or not 2 <= len(players_raw) <= 10:
            raise ValueError("players must be a list of 2 to 10 seated players")

        board_raw = data.get("board", [])
        if not isinstance(board_raw, list):
            raise ValueError("board must be a list of card strings")
        board = parse_cards(board_raw)
        if len(board) not in (0, 3, 4, 5):
            raise ValueError("board must contain 0, 3, 4, or 5 cards")

        seen_cards: set[Card] = set(board)
        seen_names: set[str] = set()
        players: list[SeatInfo] = []
        for index, raw_player in enumerate(players_raw, start=1):
            if not isinstance(raw_player, dict):
                raise ValueError(f"player {index} must be an object")
            nickname = raw_player.get("nickname")
            if not isinstance(nickname, str) or not nickname.strip():
                raise ValueError(f"player {index} nickname must be a non-empty string")
            nickname = nickname.strip()
            if nickname in seen_names:
                raise ValueError(f"duplicate player nickname {nickname!r}")
            seen_names.add(nickname)

            position = raw_player.get("position")
            if not isinstance(position, str) or position.strip().upper() not in VALID_POSITIONS:
                raise ValueError(f"player {nickname!r} has unsupported position {position!r}")
            position = position.strip().upper()

            starting_stack = _finite_positive(raw_player.get("starting_stack"), f"player {nickname!r} starting_stack")
            hole_raw = raw_player.get("hole_cards")
            hole_cards: tuple[Card, Card] | None = None
            if hole_raw is not None:
                if not isinstance(hole_raw, list) or len(hole_raw) != 2:
                    raise ValueError(f"player {nickname!r} hole_cards must be null or a 2-card list")
                parsed_hole = parse_cards(hole_raw)
                if seen_cards.intersection(parsed_hole):
                    raise ValueError(f"player {nickname!r} hole_cards duplicate another known card")
                seen_cards.update(parsed_hole)
                hole_cards = (parsed_hole[0], parsed_hole[1])
            players.append(SeatInfo(nickname, position, starting_stack, hole_cards))

        actions_raw = data.get("actions")
        if not isinstance(actions_raw, list) or not actions_raw:
            raise ValueError("actions must be a non-empty list")

        actions: list[ActionRecord] = []
        for index, raw_action in enumerate(actions_raw, start=1):
            if not isinstance(raw_action, dict):
                raise ValueError(f"action {index} must be an object")
            street = raw_action.get("street")
            if not isinstance(street, str) or street not in STREETS:
                raise ValueError(f"action {index} street must be one of {STREETS}")
            player = raw_action.get("player")
            if not isinstance(player, str) or player.strip() not in seen_names:
                raise ValueError(f"action {index} references unknown player {player!r}")
            action_type = raw_action.get("action")
            if not isinstance(action_type, str) or action_type not in ALL_ACTIONS:
                raise ValueError(f"action {index} has unsupported action type {action_type!r}")
            amount = _finite_non_negative(raw_action.get("amount", 0.0), f"action {index} amount")
            actions.append(ActionRecord(street, player.strip(), action_type, amount))

        history = cls(hand_id.strip(), small_blind, big_blind, tuple(players), board, tuple(actions))
        history.replay()
        return history

    def to_dict(self) -> dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "players": [
                {
                    "nickname": seat.nickname,
                    "position": seat.position,
                    "starting_stack": seat.starting_stack,
                    "hole_cards": [str(card) for card in seat.hole_cards] if seat.hole_cards else None,
                }
                for seat in self.players
            ],
            "board": [str(card) for card in self.board],
            "actions": [asdict(action) for action in self.actions],
        }

    def replay(self) -> HandReplay:
        """Validate action sequence consistency and extract per-player statistical summaries."""
        seat_by_name = {seat.nickname: seat for seat in self.players}
        remaining_stack = {seat.nickname: seat.starting_stack for seat in self.players}
        folded: set[str] = set()
        saw_flop: set[str] = set()

        current_street_idx = 0
        street_contrib = {seat.nickname: 0.0 for seat in self.players}
        highest_street_contrib = 0.0
        raises_on_street = 0
        pot = 0.0

        vpip_opp = {seat.nickname: False for seat in self.players}
        did_vpip = {seat.nickname: False for seat in self.players}
        pfr_opp = {seat.nickname: False for seat in self.players}
        did_pfr = {seat.nickname: False for seat in self.players}
        three_bet_opp = {seat.nickname: False for seat in self.players}
        did_three_bet = {seat.nickname: False for seat in self.players}

        counts = {
            seat.nickname: {
                "bets": 0,
                "raises": 0,
                "calls": 0,
                "checks": 0,
                "folds": 0,
                "postflop_bets": 0,
                "postflop_raises": 0,
                "postflop_calls": 0,
                "postflop_checks": 0,
                "postflop_folds": 0,
            }
            for seat in self.players
        }

        annotated: list[AnnotatedAction] = []
        required_board_len = {"preflop": 0, "flop": 3, "turn": 4, "river": 5}

        for index, record in enumerate(self.actions, start=1):
            street_idx = STREETS.index(record.street)
            if street_idx < current_street_idx:
                raise ValueError(f"action {index} moves backward from {STREETS[current_street_idx]} to {record.street}")
            if len(self.board) < required_board_len[record.street]:
                raise ValueError(
                    f"action {index} occurs on {record.street}, which requires at least "
                    f"{required_board_len[record.street]} board cards (got {len(self.board)})"
                )
            if street_idx > current_street_idx:
                current_street_idx = street_idx
                street_contrib = {seat.nickname: 0.0 for seat in self.players}
                highest_street_contrib = 0.0
                raises_on_street = 0
                for seat in self.players:
                    if seat.nickname not in folded:
                        saw_flop.add(seat.nickname)

            player = record.player
            if player in folded:
                raise ValueError(f"action {index}: player {player!r} cannot act after folding")

            to_call = max(0.0, highest_street_contrib - street_contrib[player])
            pot_before = pot
            raises_before = raises_on_street

            if record.action in BLIND_ACTIONS:
                if record.street != "preflop":
                    raise ValueError(f"action {index}: {record.action} is only valid on preflop")
                if record.amount <= 0:
                    raise ValueError(f"action {index}: {record.action} amount must be positive")
            elif record.action == "fold":
                if record.amount != 0:
                    raise ValueError(f"action {index}: fold must have amount 0")
            elif record.action == "check":
                if record.amount != 0:
                    raise ValueError(f"action {index}: check must have amount 0")
                if to_call > 1e-6:
                    raise ValueError(f"action {index}: {player!r} cannot check when facing {to_call:.2f} to call")
            elif record.action == "call":
                if record.amount <= 0:
                    raise ValueError(f"action {index}: call must have positive amount")
                if to_call <= 1e-6:
                    raise ValueError(f"action {index}: {player!r} cannot call when to_call is 0")
                if record.amount - to_call > 1e-6:
                    raise ValueError(f"action {index}: call amount {record.amount} exceeds to_call {to_call}")
            elif record.action == "bet":
                if record.street == "preflop" and highest_street_contrib > 0:
                    raise ValueError(f"action {index}: use 'raise' instead of 'bet' on preflop after blinds")
                if highest_street_contrib > 1e-6:
                    raise ValueError(f"action {index}: {player!r} cannot bet when a bet of {highest_street_contrib} exists")
                if record.amount <= 0:
                    raise ValueError(f"action {index}: bet amount must be positive")
            elif record.action == "raise":
                if highest_street_contrib <= 1e-6:
                    raise ValueError(f"action {index}: {player!r} cannot raise when no prior bet exists on {record.street}")
                if street_contrib[player] + record.amount <= highest_street_contrib + 1e-6:
                    raise ValueError(
                        f"action {index}: raise by {player!r} must increase total street commitment above {highest_street_contrib}"
                    )

            if record.amount > remaining_stack[player] + 1e-6:
                raise ValueError(
                    f"action {index}: {player!r} committed {record.amount}, exceeding remaining stack {remaining_stack[player]}"
                )

            # Update player preflop statistical opportunities before mutating state
            if record.street == "preflop" and record.action in VOLUNTARY_ACTIONS:
                vpip_opp[player] = True
                pfr_opp[player] = True
                if raises_before == 1:
                    three_bet_opp[player] = True
                if record.action in ("call", "bet", "raise"):
                    did_vpip[player] = True
                if record.action in ("bet", "raise"):
                    did_pfr[player] = True
                    if raises_before == 1:
                        did_three_bet[player] = True

            # Update voluntary action counters
            if record.action in VOLUNTARY_ACTIONS:
                key_Map = {
                    "bet": "bets",
                    "raise": "raises",
                    "call": "calls",
                    "check": "checks",
                    "fold": "folds",
                }
                plural = key_Map[record.action]
                counts[player][plural] += 1
                if record.street != "preflop":
                    counts[player][f"postflop_{plural}"] += 1

            pot_ratio: float | None = None
            if record.action in ("bet", "raise") and pot_before > 0:
                pot_ratio = record.amount / pot_before

            # Mutate running state
            remaining_stack[player] = max(0.0, remaining_stack[player] - record.amount)
            if record.action != "post_ante":
                street_contrib[player] += record.amount
                if street_contrib[player] > highest_street_contrib:
                    highest_street_contrib = street_contrib[player]
            pot += record.amount
            if record.action == "fold":
                folded.add(player)
            elif record.action in ("bet", "raise"):
                raises_on_street += 1

            annotated.append(
                AnnotatedAction(
                    street=record.street,
                    player=player,
                    position=seat_by_name[player].position,
                    action=record.action,
                    amount=record.amount,
                    pot_before=pot_before,
                    to_call_before=to_call,
                    raises_before=raises_before,
                    pot_ratio=pot_ratio,
                )
            )

        active_at_end = [seat.nickname for seat in self.players if seat.nickname not in folded]
        reached_river_end = current_street_idx == STREETS.index("river") and len(active_at_end) >= 2
        summaries: dict[str, PlayerHandSummary] = {}
        for seat in self.players:
            name = seat.nickname
            c = counts[name]
            summaries[name] = PlayerHandSummary(
                hand_id=self.hand_id,
                nickname=name,
                position=seat.position,
                vpip_opportunity=vpip_opp[name],
                did_vpip=did_vpip[name],
                pfr_opportunity=pfr_opp[name],
                did_pfr=did_pfr[name],
                three_bet_opportunity=three_bet_opp[name],
                did_three_bet=did_three_bet[name],
                bets=c["bets"],
                raises=c["raises"],
                calls=c["calls"],
                checks=c["checks"],
                folds=c["folds"],
                postflop_bets=c["postflop_bets"],
                postflop_raises=c["postflop_raises"],
                postflop_calls=c["postflop_calls"],
                postflop_checks=c["postflop_checks"],
                postflop_folds=c["postflop_folds"],
                saw_flop=name in saw_flop,
                went_to_showdown=reached_river_end and name in active_at_end,
            )

        return HandReplay(
            hand_id=self.hand_id,
            final_pot=pot,
            annotated_actions=tuple(annotated),
            player_summaries=summaries,
        )


def diff_table_captures(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Compare two table vision reports, detecting transitions without fabricating unseen actions."""
    inconsistencies: list[str] = []
    ambiguities: list[str] = []
    events: list[dict[str, Any]] = []

    if not previous.get("table_count") or not current.get("table_count"):
        ambiguities.append("uma ou ambas as capturas não contêm uma mesa calibrada válida")
        return {
            "valid_transition": False,
            "previous_street": None,
            "current_street": None,
            "pot_delta": None,
            "inferred_events": [],
            "inconsistencies": inconsistencies,
            "ambiguities": ambiguities,
        }

    prev_ambiguities = previous.get("ambiguities") or []
    curr_ambiguities = current.get("ambiguities") or []
    if prev_ambiguities or curr_ambiguities:
        ambiguities.append("capturas contêm ambiguidades visuais de origem; transição parcial")

    prev_board = list(previous.get("board") or [])
    curr_board = list(current.get("board") or [])
    street_by_len = {0: "preflop", 3: "flop", 4: "turn", 5: "river"}
    prev_street = street_by_len.get(len(prev_board))
    curr_street = street_by_len.get(len(curr_board))

    if prev_street is None or curr_street is None:
        inconsistencies.append(f"número inválido de cartas no board ({len(prev_board)} -> {len(curr_board)})")
    elif len(curr_board) < len(prev_board):
        inconsistencies.append(f"board encolheu de {prev_board} para {curr_board} (possível nova mão)")
    elif curr_board[: len(prev_board)] != prev_board:
        inconsistencies.append(f"cartas do board mudaram entre capturas: {prev_board} -> {curr_board}")

    prev_pot = previous.get("pot")
    curr_pot = current.get("pot")
    pot_delta = (float(curr_pot) - float(prev_pot)) if (prev_pot is not None and curr_pot is not None) else None
    if pot_delta is not None and pot_delta < -1e-6:
        inconsistencies.append(f"pote diminuiu de {prev_pot} para {curr_pot} dentro da mesma mão")

    prev_players = {p["seat"]: p for p in (previous.get("players") or []) if "seat" in p}
    curr_players = {p["seat"]: p for p in (current.get("players") or []) if "seat" in p}

    changed_seats = 0
    same_street = prev_street is not None and prev_street == curr_street

    for seat, prev_p in sorted(prev_players.items()):
        curr_p = curr_players.get(seat)
        if curr_p is None:
            inconsistencies.append(f"lugar {seat} desapareceu na segunda captura")
            continue
        name = curr_p.get("nickname") or prev_p.get("nickname") or f"seat_{seat}"
        if prev_p.get("nickname") and curr_p.get("nickname") and prev_p["nickname"] != curr_p["nickname"]:
            inconsistencies.append(f"nickname no lugar {seat} mudou de {prev_p['nickname']!r} para {curr_p['nickname']!r}")

        prev_folded = bool(prev_p.get("folded"))
        curr_folded = bool(curr_p.get("folded"))
        if prev_folded and not curr_folded:
            inconsistencies.append(f"jogador {name!r} estava folded e reapareceu ativo na mesma mão")
            continue

        prev_stack = prev_p.get("stack")
        curr_stack = curr_p.get("stack")
        prev_bet = prev_p.get("bet")
        curr_bet = curr_p.get("bet")

        if not prev_folded and curr_folded:
            changed_seats += 1
            events.append({"player": name, "seat": seat, "type": "fold", "amount_added": 0.0})
            continue

        if None in (prev_stack, curr_stack, prev_bet, curr_bet):
            ambiguities.append(f"stack ou bet desconhecido para {name!r}")
            continue

        stack_drop = round(float(prev_stack) - float(curr_stack), 4)
        bet_rise = round(float(curr_bet) - float(prev_bet), 4)

        if stack_drop < -1e-6:
            inconsistencies.append(f"stack de {name!r} aumentou de {prev_stack} para {curr_stack} durante a mão")
            continue

        if same_street and abs(stack_drop - bet_rise) > 1e-4:
            inconsistencies.append(
                f"inconsistência de fichas para {name!r}: stack caiu {stack_drop}, mas bet subiu {bet_rise}"
            )

        if stack_drop > 1e-6:
            changed_seats += 1
            if same_street:
                prev_max_bet = max(
                    (float(p["bet"]) for p in prev_players.values() if not p.get("folded") and p.get("bet") is not None),
                    default=0.0,
                )
                if float(curr_bet) > prev_max_bet + 1e-6:
                    action_kind = "bet" if prev_max_bet <= 1e-6 else "raise"
                else:
                    action_kind = "call"
                events.append({"player": name, "seat": seat, "type": action_kind, "amount_added": stack_drop})
            else:
                events.append(
                    {
                        "player": name,
                        "seat": seat,
                        "type": "chips_committed_across_street",
                        "amount_added": stack_drop,
                    }
                )
                ambiguities.append(
                    f"{name!r} investiu {stack_drop} fichas entre {prev_street} e {curr_street}; "
                    "a ação exata (call/bet/raise) não é inferida sem a captura intermédia"
                )

    if not same_street and prev_street != curr_street and prev_street is not None and curr_street is not None:
        events.append(
            {
                "type": "street_change",
                "from_street": prev_street,
                "to_street": curr_street,
                "new_board_cards": curr_board[len(prev_board) :],
            }
        )

    if changed_seats > 1:
        ambiguities.append(
            f"{changed_seats} jogadores alteraram estado entre as duas capturas; "
            "a ordem exata ou ações intermédias de check não são observáveis apenas pelos dois frames"
        )

    return {
        "valid_transition": len(inconsistencies) == 0,
        "previous_street": prev_street,
        "current_street": curr_street,
        "pot_delta": pot_delta,
        "inferred_events": events,
        "inconsistencies": inconsistencies,
        "ambiguities": ambiguities,
    }
