"""Live bridge between the side-by-side companion app and the strategic agent (Antigravity workspace bridge or local GPT-OSS)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
from typing import Any, Callable
import urllib.error
import urllib.request

from .assistant import (
    AssistantRecommendation,
    DecisionContext,
    GameAssistant,
    LegalAction,
    _analyze_board_and_hand,
    build_assistant_prompt,
    normalize_game_type,
    normalize_play_style,
)
from .cards import best_hand, describe
from .equity import calculate_equity
from .persistence import ProfileStore
from .range_estimator import estimate_opponent_range
from .simulator import TableSimulator
from .state import HandState
from .strategy import analyze_strategy

DEFAULT_BRIDGE_DIR = Path(__file__).resolve().parents[1] / "data" / "live_bridge"


@dataclass(frozen=True)
class BridgeRequest:
    request_id: str
    created_at: str
    agent_target: str
    context: dict[str, Any]
    baseline_recommendation: dict[str, Any]
    prompts: dict[str, str]
    capture_info: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BridgeResponse:
    request_id: str
    responded_at: str
    agent_name: str
    recommended_action: str
    recommended_amount: float
    reasoning: str
    alternatives: list[dict[str, Any]]
    risks: list[str]
    assumptions: list[str]
    detected_state: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def discover_local_models(
    ollama_url: str = "http://127.0.0.1:11434",
    openai_url: str = "http://127.0.0.1:1234/v1",
    timeout: float = 1.2,
    http_get: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Probe local Ollama and OpenAI-compatible servers (e.g. LM Studio / vLLM / llama.cpp) for available local models."""

    def _default_get(url: str, to: float) -> dict[str, Any]:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=to) as resp:
            return json.loads(resp.read().decode("utf-8"))

    getter = http_get or _default_get
    ollama_models: list[str] = []
    openai_models: list[str] = []

    try:
        data = getter(f"{ollama_url.rstrip('/')}/api/tags", timeout)
        for item in data.get("models") or []:
            if isinstance(item, dict) and item.get("name"):
                ollama_models.append(str(item["name"]))
    except (OSError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError):
        pass

    try:
        data = getter(f"{openai_url.rstrip('/')}/models", timeout)
        for item in data.get("data") or []:
            if isinstance(item, dict) and item.get("id"):
                openai_models.append(str(item["id"]))
    except (OSError, urllib.error.URLError, ValueError, KeyError, json.JSONDecodeError):
        pass

    return {
        "ollama_online": len(ollama_models) > 0,
        "ollama_models": ollama_models,
        "openai_local_online": len(openai_models) > 0,
        "openai_local_models": openai_models,
    }


def build_decision_context_from_inputs(
    hero_cards: list[str],
    board: list[str],
    pot: float,
    to_call: float,
    hero_stack: float = 100.0,
    effective_stack: float | None = None,
    hero_position: str = "BTN",
    hero_nickname: str = "HERO",
    big_blind: float = 2.0,
    opponents: list[dict[str, Any]] | None = None,
    recent_actions: list[dict[str, Any]] | None = None,
    equity_iterations: int = 2_000,
    seed: int = 7,
    profile_store: ProfileStore | None = None,
    hand_id: str = "LIVE_001",
    game_type: str = "spin_and_go",
    play_style: str = "balanced",
) -> DecisionContext:
    """Validate manual or visual table inputs, run the deterministic math engine, and build a DecisionContext."""
    opp_list = opponents or [{"nickname": "VILLAIN", "range": "*"}]
    effective_to_call = max(0.0, min(float(to_call), max(0.0, float(hero_stack))))
    canonical_game_type = normalize_game_type(game_type)
    street_guess = (
        "preflop"
        if not board
        else ("flop" if len(board) == 3 else ("turn" if len(board) == 4 else "river"))
    )

    opp_profiles: dict[str, dict[str, Any]] = {}
    all_profs = profile_store.all_profiles() if profile_store is not None else {}
    for opp in opp_list:
        nick = str(opp.get("nickname") or "VILLAIN")
        opp_profiles[nick] = all_profs[nick].to_dict() if nick in all_profs else {}

    resolved_opponents: list[dict[str, str]] = []
    assumed_ranges: dict[str, dict[str, Any]] = {}
    known_opp_stacks: list[float] = []
    for opp in opp_list:
        nick = str(opp.get("nickname") or "VILLAIN")
        raw_stack = opp.get("stack")
        opp_stack_val = float(raw_stack) if isinstance(raw_stack, (int, float)) and raw_stack > 0 else None
        if opp_stack_val is not None:
            known_opp_stacks.append(opp_stack_val)
        raw_bet = opp.get("bet")
        opp_bet_val = float(raw_bet) if isinstance(raw_bet, (int, float)) and raw_bet >= 0 else None
        est = estimate_opponent_range(
            street=street_guess,
            pot=float(pot),
            to_call=effective_to_call,
            hero_stack=float(hero_stack),
            big_blind=float(big_blind),
            game_type=canonical_game_type,
            opp_position=opp.get("position"),
            opp_bet=opp_bet_val,
            opp_stack=opp_stack_val,
            num_opponents=len(opp_list),
            opponent_profile=opp_profiles.get(nick),
            explicit_range=opp.get("range"),
            dead_cards=[*hero_cards, *board],
        )
        resolved_opponents.append({"nickname": nick, "range": est.range_text})
        assumed_ranges[nick] = est.to_dict()

    state = HandState.from_dict(
        {
            "hero": {"cards": hero_cards},
            "board": board,
            "pot": pot,
            "to_call": effective_to_call,
            "opponents": resolved_opponents,
        }
    )
    eq_res = calculate_equity(state, iterations=equity_iterations, seed=seed)
    strat_res = analyze_strategy(state, eq_res)

    if effective_stack is not None:
        eff = float(effective_stack)
    elif known_opp_stacks:
        eff = min(float(hero_stack), max(known_opp_stacks))
    else:
        eff = float(hero_stack)
    spr = round(eff / pot, 2) if pot > 0 else None
    highest_bet = effective_to_call
    legal = TableSimulator._compute_legal_actions(
        to_call=effective_to_call,
        remaining_stack=hero_stack,
        highest_bet=highest_bet,
        big_blind=big_blind,
        raises_on_street=1 if to_call > 0 else 0,
    )
    hero_desc = describe(best_hand((*state.hero_cards, *state.board))) if len(state.board) >= 3 else None
    hand_analysis = _analyze_board_and_hand(
        (str(state.hero_cards[0]), str(state.hero_cards[1])),
        tuple(str(c) for c in state.board),
        hero_position,
    )

    return DecisionContext(
        hand_id=hand_id,
        street=state.street,
        hero_nickname=hero_nickname,
        hero_position=hero_position,
        hero_cards=(str(state.hero_cards[0]), str(state.hero_cards[1])),
        board=tuple(str(c) for c in state.board),
        hero_hand_description=hero_desc,
        pot=round(pot, 2),
        to_call=round(effective_to_call, 2),
        hero_stack=round(hero_stack, 2),
        effective_stack=round(eff, 2),
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
        recent_actions=tuple(recent_actions or ()),
        opponent_profiles=opp_profiles,
        game_type=canonical_game_type,
        play_style=normalize_play_style(play_style),
        assumed_ranges=assumed_ranges,
        hand_analysis=hand_analysis,
    )


class AgentBridge:
    """Manages communication between the companion UI and either Antigravity (via workspace bridge files) or a local GPT-OSS LLM."""

    def __init__(self, bridge_dir: Path = DEFAULT_BRIDGE_DIR) -> None:
        self.bridge_dir = Path(bridge_dir)
        self.bridge_dir.mkdir(parents=True, exist_ok=True)
        self.request_path = self.bridge_dir / "request.json"
        self.response_path = self.bridge_dir / "response.json"
        self.history_log_path = self.bridge_dir / "bridge_history.jsonl"

    def publish_request(
        self,
        context: DecisionContext,
        agent_target: str = "antigravity",
        capture_info: dict[str, Any] | None = None,
    ) -> BridgeRequest:
        """Write the current hand state and computed prompt to data/live_bridge/request.json."""
        baseline_rec = GameAssistant(provider="heuristic").recommend(context)
        prompts = build_assistant_prompt(context)
        req_id = f"REQ_{int(time.time() * 1000)}"
        now_iso = datetime.now(timezone.utc).isoformat()
        request = BridgeRequest(
            request_id=req_id,
            created_at=now_iso,
            agent_target=agent_target,
            context=context.to_dict(),
            baseline_recommendation=baseline_rec.to_dict(),
            prompts=prompts,
            capture_info=capture_info,
        )
        self.request_path.write_text(
            json.dumps(request.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.history_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({"type": "request", **request.to_dict()}, ensure_ascii=False) + "\n")
        return request

    def read_pending_request(self) -> dict[str, Any] | None:
        if not self.request_path.exists():
            return None
        try:
            return json.loads(self.request_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def write_agent_response(
        self,
        request_id: str,
        recommended_action: str,
        recommended_amount: float,
        reasoning: str,
        agent_name: str = "Antigravity",
        alternatives: list[dict[str, Any]] | None = None,
        risks: list[str] | None = None,
        assumptions: list[str] | None = None,
        detected_state: dict[str, Any] | None = None,
    ) -> BridgeResponse:
        now_iso = datetime.now(timezone.utc).isoformat()
        response = BridgeResponse(
            request_id=request_id,
            responded_at=now_iso,
            agent_name=agent_name,
            recommended_action=recommended_action.lower().strip(),
            recommended_amount=float(recommended_amount),
            reasoning=reasoning.strip(),
            alternatives=alternatives or [],
            risks=risks or [],
            assumptions=assumptions or [],
            detected_state=detected_state,
        )
        self.response_path.write_text(
            json.dumps(response.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        with self.history_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps({"type": "response", **response.to_dict()}, ensure_ascii=False) + "\n")
        return response

    def read_latest_response(self, expected_request_id: str | None = None) -> dict[str, Any] | None:
        if not self.response_path.exists():
            return None
        try:
            data = json.loads(self.response_path.read_text(encoding="utf-8"))
            if expected_request_id and data.get("request_id") != expected_request_id:
                return None
            return data
        except (OSError, json.JSONDecodeError):
            return None

    def query_local_gpt_oss(
        self,
        context: DecisionContext,
        provider: str = "ollama",
        model: str = "gpt-oss:20b",
        endpoint: str | None = None,
        api_key: str | None = None,
        transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
        timeout_seconds: float = 45.0,
    ) -> AssistantRecommendation:
        """Query a local open-source LLM (e.g. gpt-oss via Ollama or LM Studio/vLLM) and also log to the bridge."""
        self.publish_request(context, agent_target=f"{provider}:{model}")
        assistant = GameAssistant(
            provider=provider,
            model=model,
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            transport=transport,
        )
        rec = assistant.recommend(context, fallback_on_error=True)
        return rec
