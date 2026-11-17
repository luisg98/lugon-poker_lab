"""Hybrid live table vision: deterministic calibrated parser + local/cloud Vision-LLM extraction."""

from __future__ import annotations

import base64
import json
import math
import os
import re
import time
from typing import Any, Callable
import urllib.error

from . import __version__
from .agent_bridge import discover_local_models
from .assistant import GameAssistant, _parse_llm_json, normalize_game_type
from .blast_vision import (
    analyze_blast_table,
    build_blast_collage_prompt,
    build_blast_numeric_collage,
)
from .cards import Card
from .mercury_vision import (
    COLLAGE_HEIGHT,
    COLLAGE_WIDTH,
    analyze_mercury_table,
    build_mercury_numeric_collage,
    classify_mercury_suit_rgb,
)
from .screen_capture import resize_rgb_image
from .state import HandState
from .vision import RGBImage, analyze_image, encode_png_bytes

DEFAULT_VLM_TIMEOUT_SECONDS = 10.0
DEFAULT_TABLE_PRESET = "pokerstars_mercury_4color"

_VISION_MODEL_KEYWORDS = (
    "qwen3-vl",
    "qwen2.5vl",
    "qwen2-vl",
    "llama3.2-vision",
    "minicpm-v",
    "llava",
    "gemma3",
    "moondream",
    "pixtral",
    "internvl",
    "-vl",
    ":vl",
    "vision",
)

# Prefer smaller instruct VLMs before larger or thinking-heavy variants.
_VISION_MODEL_PRIORITY = (
    "qwen3-vl:2b-instruct",
    "2b-instruct",
    "qwen3-vl:4b-instruct",
    "4b-instruct",
    "qwen3-vl:8b-instruct",
    "3b-instruct",
    "qwen2.5vl:3b",
    "instruct",
    "minicpm-v",
    "moondream",
    "qwen2.5vl",
    "qwen2-vl",
    "qwen3-vl",
    "llama3.2-vision",
    "llava",
    "gemma3",
)

_VALID_POSITIONS = {"BTN", "SB", "BB", "UTG", "UTG+1", "UTG+2", "MP", "LJ", "HJ", "CO"}

_UNICODE_SUIT_MAP = {
    "♠": "s",
    "♥": "h",
    "♦": "d",
    "♣": "c",
}


def normalize_card_token(value: Any) -> str:
    """Normalize common VLM card notations (e.g. '10s', 'A♠', 'kH') into canonical 'Ts', 'As', 'Kh'."""
    if not isinstance(value, str):
        raise ValueError(f"Card must be text, received {type(value).__name__}")
    raw = value.strip()
    for sym, letter in _UNICODE_SUIT_MAP.items():
        raw = raw.replace(sym, letter)
    raw = raw.replace(" ", "")
    if raw.upper().startswith("10") and len(raw) == 3:
        raw = "T" + raw[2]
    card = Card.parse(raw)
    return str(card)


def is_vision_model_name(name: str) -> bool:
    """Heuristic check for local Vision-Language Model tags (Ollama / LM Studio)."""
    low = name.lower().strip()
    return any(kw in low for kw in _VISION_MODEL_KEYWORDS)


def pick_vision_model(models: list[str]) -> str | None:
    """Select the preferred local Vision-Language Model, favoring fast non-thinking instruct models."""
    vision_models = [m for m in models if is_vision_model_name(m)]
    if not vision_models:
        return None
    for kw in _VISION_MODEL_PRIORITY:
        for m in vision_models:
            if kw in m.lower():
                return m
    return vision_models[0]


def discover_vision_models(
    ollama_url: str = "http://127.0.0.1:11434",
    openai_url: str = "http://127.0.0.1:1234/v1",
    timeout: float = 1.2,
    http_get: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Detect local and configured cloud Vision-LLM providers/models."""
    local = discover_local_models(
        ollama_url=ollama_url,
        openai_url=openai_url,
        timeout=timeout,
        http_get=http_get,
    )
    ollama_vision = [m for m in local["ollama_models"] if is_vision_model_name(m)]
    openai_local_vision = [m for m in local["openai_local_models"] if is_vision_model_name(m)]
    gemini_configured = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    openai_cloud_configured = bool(os.environ.get("OPENAI_API_KEY", "").strip())

    rec_provider: str | None = None
    rec_model: str | None = None
    if ollama_vision:
        rec_provider = "ollama"
        rec_model = pick_vision_model(ollama_vision)
    elif openai_local_vision:
        rec_provider = "openai_compatible"
        rec_model = pick_vision_model(openai_local_vision)
    elif gemini_configured:
        rec_provider = "gemini"
        rec_model = "gemini-2.5-flash"
    elif openai_cloud_configured:
        rec_provider = "openai_compatible"
        rec_model = "gpt-4o-mini"

    return {
        "ollama_online": local["ollama_online"],
        "ollama_vision_models": ollama_vision,
        "openai_local_online": local["openai_local_online"],
        "openai_local_vision_models": openai_local_vision,
        "gemini_configured": gemini_configured,
        "openai_cloud_configured": openai_cloud_configured,
        "recommended_provider": rec_provider,
        "recommended_model": rec_model,
    }


_IGNORED_OPPONENT_LABELS = {
    "hero",
    "luisfg98",
    "lugon98",
    "desisto",
    "desistir",
    "passo",
    "passar",
    "pago",
    "igualar",
    "aposto",
    "aposta",
    "subo",
    "subo para",
    "subir",
    "fold",
    "check",
    "call",
    "bet",
    "raise",
    "min",
    "mín",
    "max",
    "máx",
    "pote",
    "pot",
    "all-in",
    "all in",
    "ausente",
    "small blind",
    "big blind",
    "dealer",
}

_IGNORED_OPPONENT_PREFIXES = (
    "pago ",
    "pagando ",
    "igualar ",
    "aposto ",
    "aposta ",
    "subo ",
    "desisto ",
    "desistir ",
    "passar ",
    "subir ",
    "all-in ",
    "all in ",
)


def build_vision_prompt(preset: str = DEFAULT_TABLE_PRESET, game_type: str = "spin_and_go") -> str:
    """Build a schema-ordered, concise JSON extraction prompt tailored to the table preset and game format."""
    if normalize_game_type(game_type) == "tournament":
        opponents_note = (
            "opponents: nicknames of EVERY other seated player still in the hand, at any seat around the oval "
            "(NEVER Hero 'Luisfg98' or action words Desisto/Passo/Pago/Aposto/Subir)."
        )
    else:
        opponents_note = (
            "opponents: include ONLY real nicknames on the top-left / top-right seats "
            "(NEVER Hero 'Luisfg98' or action words Desisto/Passo/Pago/Aposto/Subir)."
        )
    # The Spin & Go prize banner ("€5" beside five spade icons) was read as board cards 5s5s5s5s5s in real captures.
    banner_note = (
        "IGNORE the prize/multiplier banner above the table (e.g. '€5' next to spade icons) and any blind-level or "
        "timer notice: they are NOT cards. Board cards are ONLY the large face-up cards in the middle of the oval felt. "
    )
    if preset == "pokerstars_mercury_4color":
        preset_notes = (
            "PokerStars Mercury 4-color card rule (USE CARD BACKGROUND COLOR FOR SUIT): "
            "red card=h, blue card=d, green card=c, dark-grey/black card=s; rank is bold white text ('10'->'T'). "
            "Hero cards sit right above Hero's bottom name plate (ignore the small circular icon on the avatar). "
            "Portuguese buttons: Desisto=Fold, Passo=Check (when Passo shows alongside Aposto/Subo without Desisto, to_call MUST be 0), "
            "Pago X=Call X, Aposto X=Bet, Subo para X=Raise. "
            "Keep exact numbers: replace comma decimals with dot and strip 'BB'/'$' (e.g. '1,5 BB'->1.5, '37 BB'->37, '6,67 BB'->6.67, '1.500 $'->1500); NEVER multiply BB decimals by 1000. "
            f"{opponents_note}"
        )
    elif preset == "888_blast":
        preset_notes = (
            "888 Poker BLAST Sit & Go table, usually a three-player layout on a green oval felt. "
            "You receive one composite image. The full table is at the top; a magnified crop of Hero's bottom seat "
            "is at the lower left, and a magnified crop of the CENTER FELT and pot label is at the lower right. "
            "The two lower panels are details from this same table. Use the lower-left panel only for Hero cards, "
            "and the lower-right panel only for community cards and the Pote total label. Read action buttons from "
            "the full table at the top. Never copy Hero cards into the board. "
            "The lower-left zoom is the authoritative close-up for Hero's cards; read it even when Hero is not to act. "
            "This 888 BLAST layout uses a FOUR-COLOR deck: diamonds are blue, clubs green, hearts red, spades black. "
            "Read the printed suit glyph first; if an action badge hides that glyph, use this card-fill color mapping. "
            "Rank is the large card letter/number. Distinguish Q from 2 carefully: Q has a closed round loop and a short tail; "
            "2 has an open top curve and a diagonal lower stroke. Read the actual glyph, especially on a green club card. "
            "Hero cards are the two face-up cards directly above the bottom Hero nameplate. Read the rank at the upper-left "
            "of each card and the printed suit glyph below it; the rank and suit glyph remain usable if an action badge "
            "covers the lower part of a card. If both rank and suit glyph are visible, return both cards instead of []. "
            "Return [] only when Hero's cards are actually hidden or face-down; never guess an unreadable rank or suit. "
            "NEVER copy cards exposed beside an opponent, including a winner after an all-in/showdown. Board cards are only the large "
            "face-up cards in the center of the felt. Ignore the central '1 €' logo, prize/multiplier graphics, "
            "blind/timer text, chips, avatars and action buttons as possible cards. "
            "pot is ONLY the number immediately after the center label 'Pote total:'; that displayed total already "
            "includes posted blinds, antes and bets. Never use an ante, a chip pile, a player's stack, a blind level, "
            "the € prize, or a sum of visible chips as pot; never add ante to the displayed total. If 'Pote total:' "
            "is absent or unreadable, set pot=null and pot_source='not_visible'; otherwise pot_source='center_total'. "
            "Also transcribe the full visible center label, e.g. pot_label_text='Pote total: 9,4 BB'; if absent use null. "
            "Portuguese controls: DESISTIR=Fold, IGUALAR X=Call X, CHECK/VER=Check (to_call=0), "
            "SUBIR A X=Raise to X. Read to_call ONLY from an explicit IGUALAR/Call amount; a small chip/ante label "
            "beside any seat is never to_call. If Hero faces an opponent all-in and the only response buttons are "
            "DESISTIR and ALL-IN X, X is Hero's total all-in amount, not to_call; when Hero's remaining stack is "
            "clearly the short stack, to_call is hero_stack and to_call_source='short_stack_allin', otherwise set "
            "to_call=null and to_call_source='unknown'. Use to_call_source='call_button' only for IGUALAR/Call, "
            "'check_button' only for a real check/pass action, and 'none' when Hero is not to act. "
            "Transcribe the exact labels of the real move buttons in action_buttons; never include ante, stack, or chip labels. "
            "hero_to_act is true ONLY when the standard rectangular move-button row is visible at the bottom-right of the table. "
            "Require at least two separate rectangular move buttons before setting hero_to_act=true. The rounded "
            "PASSAR/DESISTIR pill with a small circle underneath Hero is an auto-check/fold preference, NOT a move button; "
            "do not list it in action_buttons and do not use it to set hero_to_act=true. A DESISTIR/ALL-IN/FOLD label over Hero's "
            "cards or avatar is a past action, not a button. A zero-stack reading or an ALL-IN badge means hero_to_act=false "
            "only when the real rectangular move-button row is absent. WINNER/showdown screens are never Hero's turn. "
            "For 888, read hero_stack only from the remaining stack beside Hero's own nameplate; the amount printed on an ALL-IN "
            "button can include chips already committed and is not hero_stack. Also transcribe the exact stack label into "
            "hero_stack_text, e.g. '5,6 BB'; convert that comma decimal to 5.6 and NEVER turn it into 5600. Read the number "
            "below Hero's own nickname, not the yellow bet/ante marker above it. If the remaining stack is unreadable, use null. "
            "Every board card must include rank and suit (e.g. 7s, Kd, 3s); on 888 the face colors are black=spades, "
            "red=hearts, blue=diamonds, green=clubs. Never return rank-only board entries. "
            "Only assign BTN/SB/BB when the dealer marker or clear seat labels establish the position; otherwise null. "
            "Convert comma decimals to dot and strip 'BB'/'€'. "
            f"{opponents_note}"
        )
    else:
        preset_notes = "4-color deck: black=s, red=h, blue=d, green=c. Dot decimals."

    return (
        "Read this poker screenshot into ONE JSON object in exact key order:\n"
        "1. table_visible (boolean)\n"
        "2. center_felt_has_cards (boolean: true ONLY if 3, 4, or 5 playing cards are dealt in the middle of the oval felt; "
        "false if the middle only shows table logo/pot)\n"
        "3. board (array of card strings ONLY from the middle of the oval felt; MUST be [] if center_felt_has_cards is false)\n"
        "4. hero_cards (two canonical strings such as 'Kd' and 'Ts'; on 888 return [] only if the cards are hidden or face-down; never guess)\n"
        "5. pot (number or null; on 888 use only the center 'Pote total:' label)\n"
        "6. pot_source ('center_total' only if that exact label is visible, otherwise 'not_visible')\n"
        "7. pot_label_text (exact visible text of the center pot label, or null if absent)\n"
        "8. to_call (number or null: amount still to add now; use an explicit call button, never ante/chips; 0 if checking is available)\n"
        "9. to_call_source ('call_button', 'check_button', 'short_stack_allin', 'none', or 'unknown')\n"
        "10. action_buttons (array of exact visible move-button labels, such as 'IGUALAR 1 BB' or 'ALL-IN 7,3 BB')\n"
        "11. hero_stack (number or null, in BB)\n"
        "12. hero_stack_text (exact text beside Hero's own nameplate, e.g. '5,6 BB', or null)\n"
        "13. hero_position (string or null: BTN/SB/BB/UTG/MP/CO or null)\n"
        "14. hero_to_act (boolean: true ONLY if the actual action-button row is visible; seat labels, past-action badges and showdown overlays do not count)\n"
        f"15. opponents (array of visible opponent nicknames{'' if normalize_game_type(game_type) == 'tournament' else ' at top seats only'})\n"
        "16. ambiguities (array of strings ONLY if any card or required number is unreadable, else [])\n"
        f"{banner_note}{preset_notes} NEVER invent cards or names."
    )


def _crop_rgb_image(image: RGBImage, left: int, top: int, right: int, bottom: int) -> RGBImage:
    """Copy a clipped rectangle from an RGB image."""
    left = max(0, min(image.width - 1, left))
    top = max(0, min(image.height - 1, top))
    right = max(left + 1, min(image.width, right))
    bottom = max(top + 1, min(image.height, bottom))
    width = right - left
    height = bottom - top
    stride = width * 3
    src_stride = image.width * 3
    pixels = bytearray(stride * height)
    for row in range(height):
        src_start = (top + row) * src_stride + left * 3
        dst_start = row * stride
        pixels[dst_start : dst_start + stride] = image.pixels[src_start : src_start + stride]
    return RGBImage(width, height, pixels)


def _build_888_focus_frame(image: RGBImage, max_width: int, max_height: int) -> RGBImage:
    """Build one image with the full table plus enlarged Hero-card and center-pot details."""
    main = resize_rgb_image(image, max_width=round(max_width * 0.69), max_height=max_height)
    focus = _crop_rgb_image(
        image,
        round(image.width * 0.41),
        round(image.height * 0.74),
        round(image.width * 0.60),
        round(image.height * 0.97),
    )
    focus = resize_rgb_image(focus, max_width=round(max_width * 0.46), max_height=round(max_height * 0.50))
    center = _crop_rgb_image(
        image,
        round(image.width * 0.30),
        round(image.height * 0.28),
        round(image.width * 0.70),
        round(image.height * 0.45),
    )
    center = resize_rgb_image(center, max_width=round(max_width * 0.50), max_height=round(max_height * 0.32))

    gap = 10
    width = max(main.width, focus.width + gap + center.width)
    details_height = max(focus.height, center.height)
    height = main.height + gap + details_height
    canvas = RGBImage(width, height, bytearray([16, 24, 32]) * (width * height))

    def paste(source: RGBImage, x: int, y: int) -> None:
        row_bytes = source.width * 3
        canvas_stride = canvas.width * 3
        source_stride = source.width * 3
        for row in range(source.height):
            src_start = row * source_stride
            dst_start = (y + row) * canvas_stride + x * 3
            canvas.pixels[dst_start : dst_start + row_bytes] = source.pixels[src_start : src_start + row_bytes]

    paste(main, (width - main.width) // 2, 0)
    details_x = (width - (focus.width + gap + center.width)) // 2
    paste(focus, details_x, main.height + gap)
    paste(center, details_x + focus.width + gap, main.height + gap)
    return canvas


def _raw_888_card_rank(value: Any) -> str | None:
    if isinstance(value, dict):
        value = value.get("rank") or value.get("card") or value.get("value")
    if not isinstance(value, str):
        return None
    match = re.match(r"\s*(10|[2-9TJQKA])", value.upper())
    if not match:
        return None
    rank = match.group(1)
    return "T" if rank == "10" else rank


def _is_normalizable_card(value: Any) -> bool:
    try:
        normalize_card_token(value)
    except (TypeError, ValueError):
        return False
    return True


def _sample_888_hero_suit(image: RGBImage, card_index: int) -> str | None:
    """Infer a face-up 888 card's suit from its four-color face fill."""
    slots = ((0.44, 0.50), (0.50, 0.565))
    x0, x1 = slots[card_index]
    counts = {"h": 0, "c": 0, "d": 0, "s": 0}
    # Sample near the card tops. The lower region is often covered by a green call/check badge and can
    # look black to the sampler even when the card itself is red or blue.
    for y in range(round(image.height * 0.765), round(image.height * 0.795), 2):
        for x in range(round(image.width * x0), round(image.width * x1), 2):
            red, green, blue = image.get(x, y)
            if red > 85 and red > green * 1.35 and red > blue * 1.2:
                counts["h"] += 1
            elif green > 85 and green > red * 1.25 and green > blue * 1.1:
                counts["c"] += 1
            elif blue > 85 and blue > red * 1.2 and blue > green * 1.15:
                counts["d"] += 1
            elif max(red, green, blue) < 75:
                counts["s"] += 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = sum(counts.values())
    if total < 100 or ranked[0][1] < total * 0.50 or ranked[0][1] < ranked[1][1] * 1.25:
        return None
    return ranked[0][0]


def _sample_888_board_suit(image: RGBImage, card_index: int) -> str | None:
    """Infer a community card's suit from its colored face when the VLM returns only its rank."""
    center_x = 0.354 + 0.073 * card_index
    counts = {"h": 0, "c": 0, "d": 0, "s": 0}
    for y in range(round(image.height * 0.43), round(image.height * 0.48), 2):
        for x in range(round(image.width * (center_x - 0.019)), round(image.width * (center_x + 0.019)), 2):
            red, green, blue = image.get(x, y)
            if red > 85 and red > green * 1.35 and red > blue * 1.2:
                counts["h"] += 1
            elif green > 85 and green > red * 1.25 and green > blue * 1.1:
                counts["c"] += 1
            elif blue > 85 and blue > red * 1.2 and blue > green * 1.15:
                counts["d"] += 1
            elif max(red, green, blue) < 75:
                counts["s"] += 1

    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = sum(counts.values())
    if total < 100 or ranked[0][1] < total * 0.50 or ranked[0][1] < ranked[1][1] * 1.25:
        return None
    return ranked[0][0]


def _count_888_visual_board_cards(image: RGBImage) -> int:
    """Count bright rank glyphs in the five fixed 888 community-card slots."""
    count = 0
    for card_index in range(5):
        center_x = 0.354 + 0.073 * card_index
        white_rank_pixels = 0
        for y in range(round(image.height * 0.43), round(image.height * 0.47), 2):
            for x in range(
                round(image.width * (center_x - 0.031)),
                round(image.width * (center_x - 0.012)),
                2,
            ):
                red, green, blue = image.get(x, y)
                if min(red, green, blue) >= 205 and max(red, green, blue) - min(red, green, blue) <= 40:
                    white_rank_pixels += 1
        if white_rank_pixels >= 25:
            count += 1
    return count


def _apply_888_hero_suits(raw: dict[str, Any], image: RGBImage) -> dict[str, Any]:
    tokens = raw.get("hero_cards")
    if not isinstance(tokens, list) or len(tokens) != 2:
        return raw
    normalized: list[Any] = []
    for index, token in enumerate(tokens):
        rank = _raw_888_card_rank(token)
        suit = _sample_888_hero_suit(image, index)
        normalized.append(f"{rank}{suit}" if rank and suit else token)
    raw["hero_cards"] = normalized
    return raw


def _apply_888_board_suits(raw: dict[str, Any], image: RGBImage) -> dict[str, Any]:
    tokens = raw.get("board")
    if not isinstance(tokens, list) or len(tokens) not in (3, 4, 5):
        return raw
    normalized: list[Any] = []
    for index, token in enumerate(tokens):
        rank = _raw_888_card_rank(token)
        suit = _sample_888_board_suit(image, index)
        normalized.append(f"{rank}{suit}" if rank and suit else token)
    raw["board"] = normalized
    return raw


def _clean_888_action_buttons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    buttons = [
        re.sub(r"(?i)^apposta\b", "APOSTA", str(button).strip())
        for button in value
        if str(button).strip()
    ]
    buttons = [
        button
        for button in buttons
        if not re.fullmatch(r"(?i)\s*passar\s*/\s*desistir\s*", button)
    ]
    unique_buttons: list[str] = []
    seen_buttons: set[str] = set()
    for button in buttons:
        folded = button.casefold()
        if folded not in seen_buttons:
            unique_buttons.append(button)
            seen_buttons.add(folded)
    buttons = unique_buttons
    labels = {button.casefold() for button in buttons}
    if "passar" in labels and "desistir" in labels:
        has_real_action = any(
            re.match(r"(?i)\s*(?:igualar|call|aposta|apostar|apposta|bet|subir|raise)\b", button)
            for button in buttons
        )
        if not has_real_action:
            buttons = [button for button in buttons if button.casefold() not in {"passar", "desistir"}]
    return buttons


def _has_888_action_button_row(value: Any) -> bool:
    buttons = _clean_888_action_buttons(value)
    labels = [button.casefold() for button in buttons]
    has_fold = any(re.fullmatch(r"\s*(?:desistir|fold)\s*", label) for label in labels)
    has_call = any(re.match(r"\s*(?:igualar|call)\b", label) for label in labels)
    has_check = any(re.match(r"\s*(?:check|passar|ver)\b", label) for label in labels)
    has_bet = any(re.match(r"\s*(?:aposta|apostar|apposta|bet)\b", label) for label in labels)
    has_raise = any(re.match(r"\s*(?:subir|raise)\b", label) for label in labels)
    has_all_in = any(re.match(r"\s*all[ -]?in\b", label) for label in labels)
    return (
        (has_fold and (has_call or has_bet or has_raise or has_all_in))
        or (has_call and (has_raise or has_all_in))
        or (has_check and (has_bet or has_raise))
        or has_all_in
    )


def _count_888_visual_action_buttons(image: RGBImage) -> int:
    """Count separate rectangular controls in the 888 table's lower-right action area."""
    x_start = round(image.width * 0.55)
    min_run = max(32, round(image.width * 0.08))
    max_buttons = 0
    for y in range(round(image.height * 0.90), image.height):
        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for x in range(x_start, image.width - 3):
            red, green, blue = image.get(x, y)
            is_neutral_border = min(red, green, blue) >= 100 and max(red, green, blue) - min(red, green, blue) <= 65
            if is_neutral_border and run_start is None:
                run_start = x
            elif not is_neutral_border and run_start is not None:
                if x - run_start >= min_run:
                    runs.append((run_start, x - 1))
                run_start = None
        if run_start is not None and image.width - 3 - run_start >= min_run:
            runs.append((run_start, image.width - 4))

        separated = 0
        previous_end: int | None = None
        for start, end in runs:
            if previous_end is None or start - previous_end >= 3:
                separated += 1
            previous_end = end
        max_buttons = max(max_buttons, separated)
    return max_buttons


def _has_888_visual_action_button_row(image: RGBImage) -> bool:
    return _count_888_visual_action_buttons(image) >= 2


def _needs_888_center_detail(raw: dict[str, Any]) -> bool:
    hero = raw.get("hero_cards")
    board = raw.get("board")
    hero_ranks = [_raw_888_card_rank(card) for card in hero] if isinstance(hero, list) else []
    board_ranks = [_raw_888_card_rank(card) for card in board] if isinstance(board, list) else []
    legal_count = isinstance(board, list) and len(board) in (0, 3, 4, 5)
    normalized_cards = [
        normalize_card_token(card)
        for card in (hero if isinstance(hero, list) else []) + (board if isinstance(board, list) else [])
        if _is_normalizable_card(card)
    ]
    duplicate_cards = len(normalized_cards) != len(set(normalized_cards))
    copied_hero = (
        len(hero_ranks) == 2
        and len(board_ranks) in (2, 4)
        and all(hero_ranks)
        and board_ranks == hero_ranks * (len(board_ranks) // 2)
    )
    hero_board_rank_overlap = bool(
        set(rank for rank in hero_ranks if rank) & set(rank for rank in board_ranks if rank)
    )
    invalid_cards = any(
        not _is_normalizable_card(card)
        for card in (board if isinstance(board, list) else [])
    )
    pot_label = str(raw.get("pot_label_text") or "").strip().casefold()
    pot_label_ok = pot_label.startswith("pote total:") and "bb" in pot_label
    board_missing = bool(raw.get("center_felt_has_cards")) and not board
    # In this 888 four-color layout, Qc and 2c are easy to confuse at the
    # composite-frame scale. Confirm only that rare green-card ambiguity with
    # the existing close center crop, rather than adding a second read to every board.
    club_deuce_needs_confirmation = any(
        normalize_card_token(card) == "2c"
        for card in (board if isinstance(board, list) else [])
        if _is_normalizable_card(card)
    )
    return (
        not legal_count
        or copied_hero
        or duplicate_cards
        or hero_board_rank_overlap
        or invalid_cards
        or board_missing
        or club_deuce_needs_confirmation
        or not pot_label_ok
    )


def _needs_888_hero_detail(raw: dict[str, Any]) -> bool:
    hero = raw.get("hero_cards")
    board = raw.get("board")
    if not isinstance(hero, list) or len(hero) != 2:
        return True
    if any(not _is_normalizable_card(card) for card in hero):
        return True

    hero_cards = [normalize_card_token(card) for card in hero]
    board_cards = [
        normalize_card_token(card)
        for card in (board if isinstance(board, list) else [])
        if _is_normalizable_card(card)
    ]
    if len(set(hero_cards)) != len(hero_cards) or set(hero_cards) & set(board_cards):
        return True

    hero_ranks = {card[0] for card in hero_cards}
    board_ranks = {
        rank
        for rank in (_raw_888_card_rank(card) for card in (board if isinstance(board, list) else []))
        if rank
    }
    return bool(hero_ranks & board_ranks)


def _read_888_center_detail(
    image: RGBImage,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    timeout_seconds: float,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None,
) -> dict[str, Any]:
    center = _crop_rgb_image(
        image,
        round(image.width * 0.28),
        round(image.height * 0.37),
        round(image.width * 0.68),
        round(image.height * 0.57),
    )
    image_b64 = base64.b64encode(encode_png_bytes(center, compress_level=1)).decode("ascii")
    prompt = (
        "Read only this close crop of the center felt. List every large community card from left to right, without "
        "reordering them. The legal board is empty preflop or has 3, 4, or 5 cards. The rank is the large white "
        "glyph at the upper-left of each card; suits by face color: blue diamonds d, green clubs c, red hearts h, "
        "black spades s. Also copy the visible label beginning exactly 'Pote total:'; ignore chips and the large "
        "prize logo. Return concise JSON only: "
        '{"board_cards":[],"pot_label_text":"Pote total: 3,3 BB"}. '
        "If there are no community cards return an empty board. If the exact pot label is unreadable use null."
    )
    return _call_vlm_provider(
        image_b64=image_b64,
        prompt=prompt,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        # Some Qwen 2B builds expand each card into a JSON object despite the compact prompt;
        # leave enough room so the final card and pot label are not cut off.
        num_predict=320,
        transport=transport,
    )


def _read_888_hero_cards_detail(
    image: RGBImage,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    timeout_seconds: float,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Read only the two face-up cards directly above Hero's nameplate on 888."""
    hero = _crop_rgb_image(
        image,
        round(image.width * 0.40),
        round(image.height * 0.73),
        round(image.width * 0.60),
        round(image.height * 0.90),
    )
    image_b64 = base64.b64encode(encode_png_bytes(hero, compress_level=1)).decode("ascii")
    prompt = (
        "This crop shows the two face-up hole cards directly above the bottom player's nameplate. Read exactly "
        "those two cards from left to right. Ignore the board, chips, badges, stack and action text. The rank is "
        "the large white glyph at the upper-left of each card; suits by card color: blue diamonds d, green clubs c, "
        "red hearts h, black spades s. Return canonical card codes only as JSON: "
        '{"hero_cards":["8s","5d"]}. If either card is genuinely hidden or unreadable, return [].'
    )
    return _call_vlm_provider(
        image_b64=image_b64,
        prompt=prompt,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        num_predict=60,
        transport=transport,
    )


def _read_888_pot_detail(
    image: RGBImage,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    timeout_seconds: float,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Read only the high-contrast center total label on 888."""
    label = _crop_rgb_image(
        image,
        round(image.width * 0.35),
        round(image.height * 0.37),
        round(image.width * 0.65),
        round(image.height * 0.43),
    )
    image_b64 = base64.b64encode(encode_png_bytes(label, compress_level=1)).decode("ascii")
    prompt = (
        "Read only the exact text in the dark rounded center label. It begins 'Pote total:'. Copy the number and "
        "decimal comma exactly, then BB. Do not estimate or round. Return JSON only: "
        '{"pot_label_text":"Pote total: 2,8 BB"}'
    )
    return _call_vlm_provider(
        image_b64=image_b64,
        prompt=prompt,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        num_predict=80,
        transport=transport,
    )


def _read_888_action_detail(
    image: RGBImage,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    timeout_seconds: float,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Read only the separate rectangular move buttons on 888."""
    buttons = _crop_rgb_image(
        image,
        round(image.width * 0.55),
        round(image.height * 0.90),
        round(image.width * 0.995),
        image.height,
    )
    image_b64 = base64.b64encode(encode_png_bytes(buttons, compress_level=1)).decode("ascii")
    prompt = (
        "Read only the labels inside the separate rectangular move buttons in the bottom-right action row. "
        "Exclude rounded preference pills, chips, stack labels and percentage controls. Preserve each amount exactly. "
        "Return JSON only: {\"action_buttons\":[\"DESISTIR\",\"IGUALAR 4,3 BB\"]}."
    )
    return _call_vlm_provider(
        image_b64=image_b64,
        prompt=prompt,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        num_predict=80,
        transport=transport,
    )


def _read_888_hero_stack_detail(
    image: RGBImage,
    provider: str,
    model: str,
    endpoint: str | None,
    api_key: str | None,
    timeout_seconds: float,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Read only the remaining stack printed below the Hero's nickname on 888."""
    hero = _crop_rgb_image(
        image,
        round(image.width * 0.40),
        round(image.height * 0.73),
        round(image.width * 0.60),
        round(image.height * 0.98),
    )
    image_b64 = base64.b64encode(encode_png_bytes(hero, compress_level=1)).decode("ascii")
    prompt = (
        "Read the remaining stack printed directly below the bottom player's own nickname. Ignore the yellow circular bet/ante "
        "marker above the nameplate, chip piles, pot, cards, and action-button amounts. Transcribe the visible label exactly, "
        "including its comma and BB, such as '5,6 BB'. This is a decimal comma: 5,6 BB means 5.6 BB, never 5600. "
        "If the Hero's remaining stack cannot be read, return null. Return JSON only: {\"hero_stack_text\":\"5,6 BB\"}."
    )
    return _call_vlm_provider(
        image_b64=image_b64,
        prompt=prompt,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        num_predict=60,
        transport=transport,
    )


def build_mercury_collage_prompt(need_to_call: bool = True) -> str:
    """Build a short 4/5-key numeric OCR prompt for the Mercury 896x240 composite strip."""
    schema = (
        '{"pot": number|null, "to_call": number|null, "hero_stack": number|null, "opponents": [string], "ambiguities": []}'
        if need_to_call
        else '{"pot": number|null, "hero_stack": number|null, "opponents": [string], "ambiguities": []}'
    )
    call_note = (
        "Bottom-right red box shows ONLY 'Pago X BB' -> to_call=X (NEVER copy opponent chip numbers from Top band). "
        if need_to_call
        else ""
    )
    return (
        f"Read this PokerStars image into ONE JSON object: {schema}\n"
        "Top band above divider: left/right opponent names and center 'Pote: X BB'. "
        "Bottom-left box ALWAYS shows Hero ('Luisfg98' or action badge) with hero_stack in BB at the bottom of that box — read that BB value for hero_stack and NEVER read stacks from Top band. "
        f"{call_note}"
        "Convert comma decimal to dot and strip 'BB'/'$' (e.g. '1,5 BB'->1.5, '6,67 BB'->6.67, '37 BB'->37); NEVER multiply BB by 1000. "
        "opponents: top-left/top-right player names ONLY (ignore 'Luisfg98', 'Pago', 'Passo', 'Desisto', 'pagando ...')."
    )


def _normalize_numeric_string(raw: str) -> str:
    had_bb = bool(re.search(r"(?i)\bbb\b", raw))
    s = re.sub(r"(?i)\b(?:bb|chips?)\b", "", raw).strip()
    s = s.lstrip("$€£").rstrip("$€£").replace(" ", "")
    if "," in s and "." in s:
        if s.rfind(",") > s.rfind("."):
            s = s.replace(".", "").replace(",", ".")
        else:
            s = s.replace(",", "")
    elif "," in s:
        if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s):
            s = s.replace(",", "")
        else:
            s = s.replace(",", ".")
    elif "." in s and not had_bb:
        if re.fullmatch(r"\d{1,3}(?:\.\d{3})+", s):
            s = s.replace(".", "")
    return s


def _parse_optional_nonneg_float(value: Any, field: str, ambiguities: list[str]) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        ambiguities.append(f"{field}: invalid boolean value")
        return None
    if isinstance(value, str):
        cleaned = _normalize_numeric_string(value)
        if not cleaned:
            return None
        try:
            num = float(cleaned)
        except ValueError:
            ambiguities.append(f"{field}: non-numeric value ({value!r})")
            return None
    elif isinstance(value, (int, float)):
        num = float(value)
    else:
        ambiguities.append(f"{field}: invalid type ({type(value).__name__})")
        return None

    if not math.isfinite(num) or num < 0:
        ambiguities.append(f"{field}: invalid value ({num})")
        return None
    return round(num, 4)


def _parse_888_pot_label(value: Any) -> float | None:
    """Read the pot only from a transcribed 888 center label, never from nearby chips or stacks."""
    if not isinstance(value, str):
        return None
    match = re.search(r"(?i)^\s*pote\s+total\s*:\s*([0-9]+(?:[.,][0-9]+)?)\s*(?:bb|b|e|€|\$|chips?)?\s*$", value)
    if not match:
        return None
    cleaned = _normalize_numeric_string(match.group(1) + " BB")
    if not cleaned:
        return None
    try:
        number = float(cleaned)
    except ValueError:
        return None
    if not math.isfinite(number) or number < 0:
        return None
    return round(number, 4)


def _parse_888_stack_label(value: Any) -> float | None:
    """Parse the Hero nameplate label so comma decimals cannot become thousand-fold stack errors."""
    if not isinstance(value, str) or not value.strip():
        return None
    ambiguities: list[str] = []
    return _parse_optional_nonneg_float(value, "hero_stack_text", ambiguities)


def _needs_888_stack_detail(raw: dict[str, Any]) -> bool:
    label_value = _parse_888_stack_label(raw.get("hero_stack_text"))
    if label_value is not None and label_value > 0:
        return False
    try:
        stack_value = float(raw.get("hero_stack")) if raw.get("hero_stack") is not None else None
    except (TypeError, ValueError):
        stack_value = None
    return stack_value is None or stack_value <= 0 or stack_value >= 200


def _call_amount_from_888_buttons(value: Any) -> float | None:
    """Extract a call amount only when an exact move-button label says IGUALAR/Call."""
    if not isinstance(value, list):
        return None
    for button in value:
        if not isinstance(button, str):
            continue
        match = re.search(r"(?i)\b(?:igualar|call)\b[^0-9]*([0-9]+(?:[.,][0-9]+)?)", button)
        if match:
            try:
                number = float(match.group(1).replace(",", "."))
            except ValueError:
                continue
            if math.isfinite(number) and number >= 0:
                return round(number, 4)
    return None


def _needs_888_action_detail(raw: dict[str, Any], visual_button_count: int) -> bool:
    """Recheck a call button when OCR says the call consumes exactly the remaining stack."""
    labels = [str(button).casefold() for button in _clean_888_action_buttons(raw.get("action_buttons"))]
    has_fold = any(re.fullmatch(r"(?:desistir|fold)", label.strip()) for label in labels)
    has_allin = any(re.match(r"all[ -]?in\b", label) for label in labels)
    has_call = any(re.match(r"(?:igualar|call)\b", label) for label in labels)
    if visual_button_count > 0 and len(labels) > visual_button_count:
        return True
    if visual_button_count == 2 and has_fold and has_allin and not has_call:
        return True

    call_amount = _call_amount_from_888_buttons(raw.get("action_buttons"))
    if call_amount is None:
        return False
    stack = _parse_888_stack_label(raw.get("hero_stack_text"))
    if stack is None:
        try:
            stack = float(raw.get("hero_stack"))
        except (TypeError, ValueError):
            return False
    if stack <= 0 or call_amount < stack - 0.03:
        return False
    has_check = any(re.match(r"(?:check|passar|ver)\b", label) for label in labels)
    return not has_check and not has_allin


def _is_thinking_vlm(model_name: str) -> bool:
    low = model_name.lower().strip()
    if "instruct" in low:
        return False
    return "thinking" in low or "qwen3-vl" in low


def _call_vlm_provider(
    image_b64: str,
    prompt: str,
    provider: str,
    model: str,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_VLM_TIMEOUT_SECONDS,
    num_predict: int | None = None,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    post = transport or GameAssistant._default_http_post
    if provider == "ollama":
        base = (endpoint or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        url = f"{base}/api/chat"
        predict_limit = num_predict if num_predict is not None else 512
        ollama_options: dict[str, Any] = (
            {"temperature": 0.0, "num_ctx": 8192}
            if _is_thinking_vlm(model)
            else {"temperature": 0.0, "num_ctx": 3072, "num_predict": predict_limit}
        )
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [image_b64],
                }
            ],
            "format": "json",
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": ollama_options,
        }
        resp = post(url, {"Content-Type": "application/json"}, payload, timeout_seconds)
        msg = resp.get("message", {}) if isinstance(resp, dict) else {}
        content = str(msg.get("content") or "").strip()
        if not content:
            content = str(msg.get("thinking") or "").strip()
        return _parse_llm_json(content)

    if provider == "openai_compatible":
        base = (endpoint or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        url = f"{base}/chat/completions"
        token = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }
        resp = post(url, headers, payload, timeout_seconds)
        choices = resp.get("choices") or []
        if not choices:
            raise ValueError("OpenAI-compatible endpoint returned no choices")
        content = choices[0].get("message", {}).get("content", "")
        return _parse_llm_json(str(content))

    if provider == "gemini":
        key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError("GEMINI_API_KEY is required for the Gemini provider")
        base = (
            endpoint
            or f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )
        url = f"{base}?key={key}" if "?" not in base else base
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"inlineData": {"mimeType": "image/png", "data": image_b64}},
                    ],
                }
            ],
            "generationConfig": {"temperature": 0.0, "responseMimeType": "application/json"},
        }
        resp = post(url, {"Content-Type": "application/json"}, payload, timeout_seconds)
        candidates = resp.get("candidates") or []
        if not candidates:
            raise ValueError("Gemini endpoint returned no candidates")
        parts = candidates[0].get("content", {}).get("parts") or []
        content = parts[0].get("text", "") if parts else ""
        return _parse_llm_json(str(content))

    raise ValueError(f"Unsupported Vision-LLM provider: {provider!r}")


def _is_ignored_opponent_name(name: str) -> bool:
    low = " ".join(name.strip().lower().split())
    if not low:
        return True
    if low in _IGNORED_OPPONENT_LABELS:
        return True
    return any(low.startswith(prefix) for prefix in _IGNORED_OPPONENT_PREFIXES)


def _apply_slot_suit_overrides(tokens: list[Any], slots: list[dict[str, Any]]) -> list[Any]:
    if len(tokens) != len(slots):
        return tokens
    out: list[Any] = []
    for token, slot in zip(tokens, slots):
        suit = slot.get("suit")
        if suit and isinstance(token, str):
            try:
                norm = normalize_card_token(token)
                out.append(f"{norm[0]}{suit}")
                continue
            except ValueError:
                pass
        out.append(token)
    return out


def _merge_mercury_with_vlm_raw(
    raw: dict[str, Any], mercury: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_snapshot = dict(raw) if isinstance(raw, dict) else {}
    merged = dict(raw_snapshot)
    merged["table_visible"] = True

    if mercury.get("hero_complete"):
        merged["hero_cards"] = list(mercury["hero_cards"])
    else:
        merged["hero_cards"] = _apply_slot_suit_overrides(
            list(merged.get("hero_cards") or []),
            list(mercury.get("hero_slots") or []),
        )

    if mercury.get("board_complete"):
        merged["board"] = list(mercury["board"])
        merged["center_felt_has_cards"] = len(merged["board"]) > 0
    elif mercury.get("board_slot_count", 0) == 0:
        merged["board"] = []
        merged["center_felt_has_cards"] = False
    else:
        vlm_board = list(merged.get("board") or [])
        board_slots = list(mercury.get("board_slots") or [])
        if len(vlm_board) == len(board_slots):
            merged["board"] = _apply_slot_suit_overrides(vlm_board, board_slots)
        else:
            # The pixel scan found N card bodies on the felt but the VLM returned a different count (typically
            # phantom cards copied from the prize banner). Trust the pixel count: keep only cards read from pixels
            # and leave the rest unreadable (None) instead of inheriting fabricated ones.
            merged["board"] = [slot.get("card") for slot in board_slots]

    merged["hero_to_act"] = bool(mercury["hero_to_act"])
    if merged["hero_to_act"] and mercury.get("check_available"):
        merged["to_call"] = 0.0
    if mercury.get("hero_position"):
        merged["hero_position"] = mercury["hero_position"]

    # If Mercury read all visible cards cleanly, drop spurious card-unreadable notes from VLM.
    if mercury.get("hero_complete") and mercury.get("board_complete"):
        kept_amb = [
            item
            for item in (merged.get("ambiguities") or [])
            if "card" not in str(item).lower() and "carta" not in str(item).lower()
        ]
        merged["ambiguities"] = kept_amb

    return merged, raw_snapshot


def _merge_blast_with_vlm_raw(
    raw: dict[str, Any], blast: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_snapshot = dict(raw) if isinstance(raw, dict) else {}
    merged = dict(raw_snapshot)
    merged["table_visible"] = True

    if blast.get("hero_complete"):
        merged["hero_cards"] = list(blast["hero_cards"])
    else:
        merged["hero_cards"] = _apply_slot_suit_overrides(
            list(merged.get("hero_cards") or []),
            list(blast.get("hero_slots") or []),
        )

    if blast.get("board_complete"):
        merged["board"] = list(blast["board"])
        merged["center_felt_has_cards"] = len(merged["board"]) > 0
    elif blast.get("board_slot_count", 0) == 0:
        merged["board"] = []
        merged["center_felt_has_cards"] = False
    else:
        vlm_board = list(merged.get("board") or [])
        board_slots = list(blast.get("board_slots") or [])
        if len(vlm_board) == len(board_slots):
            merged["board"] = _apply_slot_suit_overrides(vlm_board, board_slots)
        else:
            merged["board"] = [slot.get("card") for slot in board_slots]

    merged["hero_to_act"] = bool(blast["hero_to_act_visual"])
    if blast.get("hero_position"):
        merged["hero_position"] = blast["hero_position"]

    hero_nick = str(merged.get("hero_nickname") or "").strip().lower()
    has_collage_seats = "seat_left" in raw_snapshot or "seat_right" in raw_snapshot
    if has_collage_seats:
        structured_opps: list[dict[str, Any]] = []
        for side, key, fallback_label in (
            ("left", "seat_left", "Left Opponent"),
            ("right", "seat_right", "Right Opponent"),
        ):
            if not blast.get(f"seat_{side}_occupied"):
                continue
            seat_obj = raw_snapshot.get(key)
            nick = str(seat_obj.get("nickname") or "").strip() if isinstance(seat_obj, dict) else ""
            if _is_ignored_opponent_name(nick) or (hero_nick and nick.lower() == hero_nick):
                nick = fallback_label
            stack_val: float | None = None
            if isinstance(seat_obj, dict):
                stack_val = _parse_888_stack_label(seat_obj.get("stack_text"))
                if stack_val is None and isinstance(seat_obj.get("stack"), (int, float)):
                    stack_val = round(float(seat_obj["stack"]), 4)
            structured_opps.append(
                {
                    "nickname": nick,
                    "position": blast.get(f"seat_{side}_position"),
                    "stack": stack_val,
                    "bet": None,
                    "folded": not bool(blast.get(f"seat_{side}_active")),
                }
            )
        if structured_opps:
            merged["opponents"] = structured_opps

    if blast.get("hero_complete") and blast.get("board_complete"):
        kept_amb = [
            item
            for item in (merged.get("ambiguities") or [])
            if "card" not in str(item).lower() and "carta" not in str(item).lower()
        ]
        merged["ambiguities"] = kept_amb

    return merged, raw_snapshot


def _normalize_vlm_report(
    raw: dict[str, Any],
    image: RGBImage,
    vision_source: str,
    latency_ms: float,
    mercury: dict[str, Any] | None = None,
    blast: dict[str, Any] | None = None,
    preset: str = DEFAULT_TABLE_PRESET,
) -> dict[str, Any]:
    raw_snapshot = dict(raw) if isinstance(raw, dict) else {}
    working = raw_snapshot
    if mercury is not None:
        working, raw_snapshot = _merge_mercury_with_vlm_raw(raw_snapshot, mercury)
    elif blast is not None:
        working, raw_snapshot = _merge_blast_with_vlm_raw(raw_snapshot, blast)

    ambiguities: list[str] = [
        str(item).strip()
        for item in (working.get("ambiguities") or [])
        if str(item).strip()
    ]
    if preset == "888_blast":
        ambiguities = [
            item
            for item in ambiguities
            if not re.fullmatch(
                r"(?i)(?:[€$]\s*)?\d+(?:[.,]\d+)?\s*(?:bb|€|eur|euro)",
                item,
            )
        ]
    table_visible = bool(working.get("table_visible"))
    if not table_visible:
        if not ambiguities:
            ambiguities.append("table: no poker table visible in the captured area")
        return {
            "image": {"width": image.width, "height": image.height},
            "table_count": 0,
            "table_count_source": f"vision_llm ({vision_source})",
            "vision_source": vision_source,
            "vision_latency_ms": round(latency_ms, 2),
            "hero_cards": [],
            "board": [],
            "pot": None,
            "blinds": {"small": None, "big": None},
            "players": [],
            "active_player": None,
            "hero_to_act": False,
            "to_call": None,
            "hand_state": None,
            "opponent_ranges": "unknown; uniform combinations used only if the M1 analyzer is called",
            "ambiguities": ambiguities,
            "raw_vlm_response": raw_snapshot,
        }

    hero_cards: list[str] = []
    for idx, token in enumerate(working.get("hero_cards") or []):
        try:
            hero_cards.append(normalize_card_token(token))
        except ValueError as exc:
            ambiguities.append(f"hero.cards[{idx}]: {exc}")

    board: list[str] = []
    for idx, token in enumerate(working.get("board") or []):
        if token is None:
            ambiguities.append(f"board[{idx}]: visible but unreadable card")
            continue
        try:
            board.append(normalize_card_token(token))
        except ValueError as exc:
            ambiguities.append(f"board[{idx}]: {exc}")

    if len(board) not in (0, 3, 4, 5):
        ambiguities.append(f"board: esperava 0, 3, 4 ou 5 cartas, recebi {len(board)}")
    if len(set(hero_cards) | set(board)) != len(hero_cards) + len(board):
        ambiguities.append("cards: cartas duplicadas detetadas entre hero_cards e board")

    pot = _parse_optional_nonneg_float(working.get("pot"), "pot", ambiguities)
    to_call = _parse_optional_nonneg_float(working.get("to_call"), "to_call", ambiguities)
    stack_from_label = _parse_888_stack_label(working.get("hero_stack_text")) if preset == "888_blast" else None
    hero_stack = _parse_optional_nonneg_float(
        stack_from_label if stack_from_label is not None and stack_from_label > 0 else working.get("hero_stack"),
        "hero_stack",
        ambiguities,
    )
    if preset == "888_blast" and hero_stack is not None and hero_stack >= 200:
        ambiguities.append("hero_stack: implausible BB scale; stack blocked until the exact label can be read")
        hero_stack = None

    raw_hero_to_act = working.get("hero_to_act")
    if raw_hero_to_act is None:
        hero_to_act = False
        ambiguities.append("hero_to_act: could not confirm whether it is the Hero's turn")
    else:
        hero_to_act = bool(raw_hero_to_act)

    pot_label_text = working.get("pot_label_text")
    pot_source = str(working.get("pot_source") or "").strip().lower() or None
    to_call_source = str(working.get("to_call_source") or "").strip().lower() or None
    action_buttons = working.get("action_buttons")
    if preset == "888_blast":
        action_buttons = _clean_888_action_buttons(action_buttons)
        visual_button_count = _count_888_visual_action_buttons(image)
        row_confirmed = _has_888_action_button_row(action_buttons) and visual_button_count >= 2
        if row_confirmed:
            hero_to_act = True
        elif hero_to_act:
            hero_to_act = False
            ambiguities.append("hero_to_act: rectangular action button row was not confirmed in the table image")
        if hero_to_act and hero_stack == 0:
            hero_stack = None
            ambiguities.append("hero_stack: zero conflicts with the visible action row; stack remains unreadable")
        if not hero_to_act:
            action_buttons = []
        pot = _parse_888_pot_label(pot_label_text)
        pot_source = "center_total" if pot is not None else "not_visible"
        if not hero_to_act:
            pot = None
            pot_source = "not_visible"
            to_call = None
            to_call_source = "none"
        else:
            call_from_button = _call_amount_from_888_buttons(action_buttons)
            button_labels = [str(button).strip().casefold() for button in action_buttons]
            fold_button_visible = any(re.fullmatch(r"(?:desistir|fold)", label) for label in button_labels)
            allin_button_visible = any(re.match(r"all[ -]?in\b", label) for label in button_labels)
            if allin_button_visible and visual_button_count == 2:
                action_buttons = [
                    button
                    for button in action_buttons
                    if re.fullmatch(r"(?i)\s*(?:desistir|fold)\s*", str(button))
                    or re.match(r"(?i)\s*all[ -]?in\b", str(button))
                ]
                call_from_button = None
                button_labels = [str(button).strip().casefold() for button in action_buttons]
            check_button_visible = any(
                re.fullmatch(r"(?i)\s*(?:check|passar|ver)(?:\s*/\s*desistir)?\s*", str(button))
                for button in action_buttons
            )
            fold_button_visible = any(re.fullmatch(r"(?:desistir|fold)", label) for label in button_labels)
            allin_button_visible = any(re.match(r"all[ -]?in\b", label) for label in button_labels)
            if check_button_visible:
                to_call = 0.0
                to_call_source = "check_button"
                action_buttons = [
                    button
                    for button in action_buttons
                    if _call_amount_from_888_buttons([button]) is None
                ]
            elif (
                allin_button_visible
                and visual_button_count == 2
                and hero_stack is not None
            ):
                to_call = hero_stack
                to_call_source = "short_stack_allin"
            elif call_from_button is not None:
                to_call = call_from_button
                to_call_source = "call_button"
            elif fold_button_visible and allin_button_visible and hero_stack is not None:
                to_call = hero_stack
                to_call_source = "short_stack_allin"
            elif (
                to_call_source == "short_stack_allin"
                and hero_stack is not None
                and allin_button_visible
            ):
                to_call = hero_stack
            else:
                to_call = None
                to_call_source = "unknown"

    raw_pos = str(working.get("hero_position") or "").strip().upper()
    hero_pos = raw_pos if raw_pos in _VALID_POSITIONS else "UNKNOWN"
    active_player = str(working.get("active_player")).strip() if working.get("active_player") else None

    players: list[dict[str, Any]] = []
    hero_nick_lower = str(working.get("hero_nickname") or "").strip().lower()
    opponents_raw = working.get("opponents") or []
    active_opponents: list[dict[str, Any]] = []
    seen_nicks: set[str] = set()
    for idx, item in enumerate(opponents_raw, start=1):
        if isinstance(item, str):
            nick = item.strip()
            if (
                _is_ignored_opponent_name(nick)
                or (hero_nick_lower and nick.lower() == hero_nick_lower)
                or nick.lower() in seen_nicks
            ):
                continue
            seen_nicks.add(nick.lower())
            opp_entry = {
                "nickname": nick,
                "position": None,
                "stack": None,
                "bet": None,
                "hero": False,
                "active": active_player == nick,
                "folded": False,
                "seat": len(players),
            }
        elif isinstance(item, dict):
            nick = str(item.get("nickname") or "").strip()
            if (
                _is_ignored_opponent_name(nick)
                or (hero_nick_lower and nick.lower() == hero_nick_lower)
                or nick.lower() in seen_nicks
            ):
                continue
            seen_nicks.add(nick.lower())
            opp_stack = _parse_optional_nonneg_float(item.get("stack"), f"opponents[{idx}].stack", ambiguities)
            opp_bet = _parse_optional_nonneg_float(item.get("bet"), f"opponents[{idx}].bet", ambiguities)
            opp_pos = str(item.get("position")).strip().upper() if item.get("position") else None
            folded = bool(item.get("folded", False))
            opp_entry = {
                "nickname": nick,
                "position": opp_pos,
                "stack": opp_stack,
                "bet": opp_bet,
                "hero": False,
                "active": active_player == nick,
                "folded": folded,
                "seat": len(players),
            }
        else:
            continue
        players.append(opp_entry)
        if not opp_entry["folded"]:
            active_opponents.append(opp_entry)

    # When deterministic seat pixels (Mercury or 888 BLAST) confirm active opponent card backs whose
    # nameplate text was obscured or missed, fill seat placeholders truthfully.
    pixel_seats = mercury if mercury is not None else blast
    if pixel_seats is not None and pixel_seats.get("active_opponents_count") is not None:
        needed = int(pixel_seats["active_opponents_count"]) - len(active_opponents)
        if needed > 0:
            candidate_seats = []
            if pixel_seats.get("seat_left_active"):
                candidate_seats.append(("Left Opponent", pixel_seats.get("seat_left_position")))
            if pixel_seats.get("seat_right_active"):
                candidate_seats.append(("Right Opponent", pixel_seats.get("seat_right_position")))
            for label, seat_pos in candidate_seats:
                if needed <= 0:
                    break
                if label.lower() in seen_nicks:
                    continue
                seen_nicks.add(label.lower())
                opp_entry = {
                    "nickname": label,
                    "position": seat_pos,
                    "stack": None,
                    "bet": None,
                    "hero": False,
                    "active": False,
                    "folded": False,
                    "seat": len(players),
                }
                players.append(opp_entry)
                active_opponents.append(opp_entry)
                needed -= 1

    players.append(
        {
            "nickname": "HERO",
            "position": hero_pos,
            "stack": hero_stack,
            "bet": None,
            "hero": True,
            "active": hero_to_act,
            "folded": False,
            "seat": len(players),
        }
    )

    if hero_to_act:
        if len(hero_cards) != 2:
            ambiguities.append(f"hero_cards: expected 2 readable cards, found {len(hero_cards)}")
        if pot is None:
            ambiguities.append("pot: unreadable pot amount")
        if to_call is None:
            ambiguities.append("to_call: unreadable amount to call")
        if not 1 <= len(active_opponents) <= 9:
            ambiguities.append(f"opponents: esperava 1 a 9 oponentes ativos, encontrei {len(active_opponents)}")

    if hero_to_act and hero_stack is None:
        ambiguities.append("hero_stack: remaining Hero stack is unreadable; advice blocked")

    hand_state: dict[str, Any] | None = None
    if (
        not ambiguities
        and len(hero_cards) == 2
        and len(board) in (0, 3, 4, 5)
        and pot is not None
        and to_call is not None
        and 1 <= len(active_opponents) <= 9
    ):
        candidate = {
            "hero": {"nickname": "HERO", "cards": hero_cards},
            "board": board,
            "pot": pot,
            "to_call": to_call,
            "opponents": [{"nickname": opp["nickname"]} for opp in active_opponents],
        }
        try:
            HandState.from_dict(candidate)
            hand_state = candidate
        except ValueError as exc:
            ambiguities.append(f"hand_state: {exc}")

    if mercury is not None:
        final_source = f"{vision_source}+mercury"
    elif blast is not None:
        final_source = f"{vision_source}+blast" if vision_source != "blast_pixel" else "blast_pixel"
    else:
        final_source = vision_source
    out = {
        "image": {"width": image.width, "height": image.height},
        "table_count": 1,
        "table_count_source": f"vision_llm ({final_source})",
        "vision_source": final_source,
        "vision_latency_ms": round(latency_ms, 2),
        "hero_cards": hero_cards,
        "hero_stack_text": working.get("hero_stack_text"),
        "board": board,
        "pot": pot,
        "blinds": {"small": None, "big": None},
        "players": players,
        "active_player": active_player,
        "hero_to_act": hero_to_act,
        "to_call": to_call,
        "pot_source": pot_source,
        "pot_label_text": pot_label_text,
        "to_call_source": to_call_source,
        "action_buttons": action_buttons,
        "hand_state": hand_state,
        "opponent_ranges": "unknown; uniform combinations used only if the M1 analyzer is called",
        "ambiguities": ambiguities,
        "raw_vlm_response": raw_snapshot,
    }
    if mercury is not None:
        out["mercury_vision"] = {
            "table_rect": mercury["table_rect"],
            "hero_cards": mercury["hero_cards"],
            "hero_complete": mercury["hero_complete"],
            "hero_slots": mercury["hero_slots"],
            "board": mercury["board"],
            "board_complete": mercury["board_complete"],
            "board_slots": mercury["board_slots"],
            "hero_to_act": mercury["hero_to_act"],
            "check_available": mercury["check_available"],
            "hero_position": mercury["hero_position"],
            "active_opponents_count": mercury["active_opponents_count"],
        }
    if blast is not None:
        out["blast_vision"] = {
            "hero_cards": blast["hero_cards"],
            "hero_complete": blast["hero_complete"],
            "hero_slots": blast["hero_slots"],
            "board": blast["board"],
            "board_complete": blast["board_complete"],
            "board_slots": blast["board_slots"],
            "hero_to_act_visual": blast["hero_to_act_visual"],
            "visual_button_count": blast["visual_button_count"],
            "dealer_seat": blast["dealer_seat"],
            "hero_position": blast["hero_position"],
            "seat_left_position": blast["seat_left_position"],
            "seat_right_position": blast["seat_right_position"],
            "seat_left_occupied": blast["seat_left_occupied"],
            "seat_left_active": blast["seat_left_active"],
            "seat_right_occupied": blast["seat_right_occupied"],
            "seat_right_active": blast["seat_right_active"],
            "active_opponents_count": blast["active_opponents_count"],
        }
    return out


def extract_table_with_vlm(
    image: RGBImage,
    provider: str = "ollama",
    model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_VLM_TIMEOUT_SECONDS,
    max_width: int = 1024,
    max_height: int = 640,
    preset: str = DEFAULT_TABLE_PRESET,
    game_type: str = "spin_and_go",
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Extract and validate a live poker table state from an RGBImage using a Vision-LLM."""
    default_models = {
        "ollama": "qwen3-vl:2b-instruct",
        "openai_compatible": "gpt-4o-mini",
        "gemini": "gemini-2.5-flash",
    }
    if provider not in default_models:
        raise ValueError(f"Unsupported Vision-LLM provider: {provider!r}")
    chosen_model = model or default_models[provider]
    source_label = f"{provider}:{chosen_model}"
    game_type = normalize_game_type(game_type)

    t0 = time.perf_counter()
    mercury = analyze_mercury_table(image, game_type=game_type) if preset == "pokerstars_mercury_4color" else None
    blast = analyze_blast_table(image, game_type=game_type) if preset == "888_blast" else None

    # Fast deterministic off-turn shortcut on 888 BLAST: skip VLM completely when it's not Hero's turn.
    if blast is not None and not blast["hero_to_act_visual"] and transport is None:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return _normalize_vlm_report(
            {"table_visible": True, "hero_to_act": False, "opponents": []},
            image=image,
            vision_source="blast_pixel",
            latency_ms=elapsed_ms,
            blast=blast,
            preset=preset,
        )

    use_collage = bool(
        game_type == "spin_and_go"
        and mercury
        and mercury.get("hero_complete")
        and mercury.get("board_complete")
    )
    use_blast_collage = bool(
        game_type == "spin_and_go"
        and blast
        and blast.get("hero_complete")
        and blast.get("board_complete")
        and blast.get("hero_to_act_visual")
    )
    if use_collage:
        assert mercury is not None
        prepared = build_mercury_numeric_collage(image, mercury)
        need_to_call = bool(mercury.get("hero_to_act") and not mercury.get("check_available"))
        prompt = build_mercury_collage_prompt(need_to_call=need_to_call)
        predict_limit: int | None = 140
    elif use_blast_collage:
        assert blast is not None
        prepared = build_blast_numeric_collage(image, blast)
        prompt = build_blast_collage_prompt(blast)
        predict_limit = 220
    else:
        source_for_vlm = mercury["vlm_crop"] if mercury is not None else image
        if preset == "888_blast":
            prepared = _build_888_focus_frame(source_for_vlm, max_width, max_height)
        else:
            prepared = resize_rgb_image(source_for_vlm, max_width=max_width, max_height=max_height)
        prompt = build_vision_prompt(preset=preset, game_type=game_type)
        predict_limit = None
    image_b64 = base64.b64encode(encode_png_bytes(prepared, compress_level=1)).decode("ascii")

    try:
        raw = _call_vlm_provider(
            image_b64=image_b64,
            prompt=prompt,
            provider=provider,
            model=chosen_model,
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            num_predict=predict_limit,
            transport=transport,
        )
    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        err_msg = str(exc)
        if isinstance(exc, TimeoutError) or "timed out" in err_msg.lower():
            reason = (
                f"vision_llm ({source_label}) excedeu o timeout de {timeout_seconds:g}s; "
                "make sure Qwen3-VL 2B Instruct is loaded and the capture contains only the table."
            )
        else:
            reason = f"vision_llm ({source_label}) falhou: {exc}"
        return {
            "image": {"width": image.width, "height": image.height},
            "table_count": 0,
            "table_count_source": f"vision_llm ({source_label})",
            "vision_source": source_label,
            "vision_latency_ms": round(elapsed_ms, 2),
            "hero_cards": [],
            "board": [],
            "pot": None,
            "blinds": {"small": None, "big": None},
            "players": [],
            "active_player": None,
            "hero_to_act": False,
            "to_call": None,
            "hand_state": None,
            "opponent_ranges": "unknown; uniform combinations used only if the M1 analyzer is called",
            "ambiguities": [reason],
        }

    if preset == "888_blast":
        if use_blast_collage and blast is not None:
            raw["hero_cards"] = list(blast["hero_cards"])
            raw["board"] = list(blast["board"])
            raw["center_felt_has_cards"] = bool(blast["board"])
            visual_board_count = len(blast["board"])
            visual_turn = True
            if (
                provider == "ollama"
                and transport is None
                and _parse_888_pot_label(raw.get("pot_label_text")) is None
            ):
                try:
                    pot_detail = _read_888_pot_detail(
                        image=image,
                        provider=provider,
                        model=chosen_model,
                        endpoint=endpoint,
                        api_key=api_key,
                        timeout_seconds=min(timeout_seconds, 8.0),
                        transport=transport,
                    )
                    pot_detail_label = pot_detail.get("pot_label_text")
                    if (
                        isinstance(pot_detail_label, str)
                        and "bb" in pot_detail_label.casefold()
                        and (pot_detail_value := _parse_888_pot_label(pot_detail_label)) is not None
                    ):
                        raw["pot_label_text"] = pot_detail_label
                        raw["pot"] = pot_detail_value
                        raw["pot_source"] = "center_total"
                    raw["pot_detail_response"] = pot_detail
                except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raw["pot_detail_error"] = str(exc)
        else:
            if blast is not None and blast.get("hero_complete"):
                raw["hero_cards"] = list(blast["hero_cards"])
            if blast is not None and blast.get("board_complete"):
                raw["board"] = list(blast["board"])
            raw = _apply_888_hero_suits(raw, image)
            visual_board_count = _count_888_visual_board_cards(image)
            board_value = raw.get("board")
            if visual_board_count == 0:
                raw["board"] = []
                raw["center_felt_has_cards"] = False
                raw["ambiguities"] = [
                    item
                    for item in (raw.get("ambiguities") or [])
                    if "board" not in str(item).casefold()
                    and not ("duplicat" in str(item).casefold() and "card" in str(item).casefold())
                    and not ("duplicat" in str(item).casefold() and "carta" in str(item).casefold())
                ]
            elif visual_board_count in (3, 4, 5):
                raw["center_felt_has_cards"] = True
                if isinstance(board_value, list) and len(board_value) > visual_board_count:
                    raw["board"] = board_value[:visual_board_count]
            raw = _apply_888_board_suits(raw, image)
            visual_turn = _has_888_visual_action_button_row(image)
            if (
                visual_turn
                and provider == "ollama"
                and transport is None
                and _needs_888_hero_detail(raw)
            ):
                try:
                    hero_detail = _read_888_hero_cards_detail(
                        image=image,
                        provider=provider,
                        model=chosen_model,
                        endpoint=endpoint,
                        api_key=api_key,
                        timeout_seconds=min(timeout_seconds, 8.0),
                        transport=transport,
                    )
                    hero_cards_detail = hero_detail.get("hero_cards")
                    if (
                        isinstance(hero_cards_detail, list)
                        and len(hero_cards_detail) == 2
                        and all(_is_normalizable_card(card) for card in hero_cards_detail)
                    ):
                        candidate_cards = [normalize_card_token(card) for card in hero_cards_detail]
                        board_cards = [
                            normalize_card_token(card)
                            for card in (raw.get("board") or [])
                            if _is_normalizable_card(card)
                        ]
                        if len(set(candidate_cards)) == 2 and not (set(candidate_cards) & set(board_cards)):
                            raw["hero_cards"] = candidate_cards
                    raw["hero_detail_response"] = hero_detail
                except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raw["hero_detail_error"] = str(exc)
            raw = _apply_888_hero_suits(raw, image)

            if (
                visual_turn
                and visual_board_count > 0
                and provider == "ollama"
                and transport is None
                and _needs_888_center_detail(raw)
            ):
                center_error: str | None = None
                center_detail_ok = False
                for attempt in range(2):
                    try:
                        center_detail = _read_888_center_detail(
                            image=image,
                            provider=provider,
                            model=chosen_model,
                            endpoint=endpoint,
                            api_key=api_key,
                            timeout_seconds=min(timeout_seconds, 8.0),
                            transport=transport,
                        )
                        center_board = center_detail.get("board_cards")
                        center_board_ranks = (
                            [_raw_888_card_rank(card) for card in center_board]
                            if isinstance(center_board, list)
                            else None
                        )
                        board_ok = (
                            center_board_ranks is not None
                            and len(center_board_ranks) in (0, 3, 4, 5)
                            and (not center_board_ranks or all(center_board_ranks))
                        )
                        if board_ok:
                            raw["board"] = center_board_ranks
                            raw["center_felt_has_cards"] = bool(center_board_ranks)

                        center_pot_label = center_detail.get("pot_label_text")
                        center_pot = (
                            _parse_888_pot_label(center_pot_label)
                            if isinstance(center_pot_label, str) and "bb" in center_pot_label.casefold()
                            else None
                        )
                        if center_pot is not None:
                            raw["pot_label_text"] = center_pot_label
                            raw["pot"] = center_pot
                            raw["pot_source"] = "center_total"
                        raw["center_detail_response"] = center_detail

                        existing_pot_ok = _parse_888_pot_label(raw.get("pot_label_text")) is not None
                        if board_ok and (center_pot is not None or existing_pot_ok):
                            center_detail_ok = True
                            break
                        center_error = "center detail did not return a valid board and pot label"
                    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                        center_error = str(exc)
                if center_detail_ok:
                    raw.pop("center_detail_error", None)
                elif center_error is not None:
                    raw["center_detail_error"] = center_error

                try:
                    pot_detail = _read_888_pot_detail(
                        image=image,
                        provider=provider,
                        model=chosen_model,
                        endpoint=endpoint,
                        api_key=api_key,
                        timeout_seconds=min(timeout_seconds, 8.0),
                        transport=transport,
                    )
                    pot_detail_label = pot_detail.get("pot_label_text")
                    if (
                        isinstance(pot_detail_label, str)
                        and "bb" in pot_detail_label.casefold()
                        and (pot_detail_value := _parse_888_pot_label(pot_detail_label)) is not None
                    ):
                        raw["pot_label_text"] = pot_detail_label
                        raw["pot"] = pot_detail_value
                        raw["pot_source"] = "center_total"
                    raw["pot_detail_response"] = pot_detail
                except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raw["pot_detail_error"] = str(exc)
            elif (
                visual_turn
                and visual_board_count == 0
                and provider == "ollama"
                and transport is None
                and _parse_888_pot_label(raw.get("pot_label_text")) is None
            ):
                try:
                    pot_detail = _read_888_pot_detail(
                        image=image,
                        provider=provider,
                        model=chosen_model,
                        endpoint=endpoint,
                        api_key=api_key,
                        timeout_seconds=min(timeout_seconds, 8.0),
                        transport=transport,
                    )
                    pot_detail_label = pot_detail.get("pot_label_text")
                    if (
                        isinstance(pot_detail_label, str)
                        and "bb" in pot_detail_label.casefold()
                        and (pot_detail_value := _parse_888_pot_label(pot_detail_label)) is not None
                    ):
                        raw["pot_label_text"] = pot_detail_label
                        raw["pot"] = pot_detail_value
                        raw["pot_source"] = "center_total"
                    raw["pot_detail_response"] = pot_detail
                except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                    raw["pot_detail_error"] = str(exc)
        if (
            visual_turn
            and provider == "ollama"
            and transport is None
            and _needs_888_action_detail(raw, blast["visual_button_count"] if blast is not None else _count_888_visual_action_buttons(image))
        ):
            try:
                action_detail = _read_888_action_detail(
                    image=image,
                    provider=provider,
                    model=chosen_model,
                    endpoint=endpoint,
                    api_key=api_key,
                    timeout_seconds=min(timeout_seconds, 8.0),
                    transport=transport,
                )
                detail_buttons = action_detail.get("action_buttons")
                if isinstance(detail_buttons, list) and _has_888_action_button_row(detail_buttons):
                    raw["action_buttons"] = _clean_888_action_buttons(detail_buttons)
                raw["action_detail_response"] = action_detail
            except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raw["action_detail_error"] = str(exc)
        if (
            visual_turn
            and provider == "ollama"
            and transport is None
            and _needs_888_stack_detail(raw)
        ):
            try:
                stack_detail = _read_888_hero_stack_detail(
                    image=image,
                    provider=provider,
                    model=chosen_model,
                    endpoint=endpoint,
                    api_key=api_key,
                    timeout_seconds=min(timeout_seconds, 8.0),
                    transport=transport,
                )
                if isinstance(stack_detail.get("hero_stack_text"), str):
                    raw["hero_stack_text"] = stack_detail["hero_stack_text"]
                raw["stack_detail_response"] = stack_detail
            except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                raw["stack_detail_error"] = str(exc)
        if not use_blast_collage:
            raw = _apply_888_board_suits(raw, image)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return _normalize_vlm_report(
        raw,
        image=image,
        vision_source=source_label,
        latency_ms=elapsed_ms,
        mercury=mercury,
        blast=blast,
        preset=preset,
    )


def warmup_vision_model(
    provider: str,
    model: str,
    endpoint: str | None = None,
    timeout_seconds: float = 15.0,
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
) -> bool:
    """Preload a local Ollama Vision-LLM (weights + CUDA vision projector) into VRAM."""
    if provider != "ollama" or not model:
        return False
    post = transport or GameAssistant._default_http_post
    base = (endpoint or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
    url = f"{base}/api/chat"
    warm_img = RGBImage(
        COLLAGE_WIDTH,
        COLLAGE_HEIGHT,
        bytearray((18, 22, 28) * COLLAGE_WIDTH * COLLAGE_HEIGHT),
    )
    b64 = base64.b64encode(encode_png_bytes(warm_img, compress_level=1)).decode("ascii")
    opts = (
        {"temperature": 0.0, "num_ctx": 8192, "num_predict": 1}
        if _is_thinking_vlm(model)
        else {"temperature": 0.0, "num_ctx": 3072, "num_predict": 1}
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "ok", "images": [b64]}],
        "stream": False,
        "keep_alive": "10m",
        "options": opts,
    }
    try:
        post(url, {"Content-Type": "application/json"}, payload, timeout_seconds)
        return True
    except (OSError, urllib.error.URLError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def analyze_live_image(
    image: RGBImage,
    mode: str = "hybrid",
    vlm_provider: str | None = None,
    vlm_model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_VLM_TIMEOUT_SECONDS,
    preset: str = DEFAULT_TABLE_PRESET,
    game_type: str = "spin_and_go",
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
    http_get: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a live captured RGBImage and stamp the report with the app version and game format used."""
    game_key = normalize_game_type(game_type)
    report = _analyze_live_image(
        image,
        mode=mode,
        vlm_provider=vlm_provider,
        vlm_model=vlm_model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        preset=preset,
        game_type=game_key,
        transport=transport,
        http_get=http_get,
    )
    report["game_type"] = game_key
    report["app_version"] = __version__
    return report


def _analyze_live_image(
    image: RGBImage,
    mode: str = "hybrid",
    vlm_provider: str | None = None,
    vlm_model: str | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    timeout_seconds: float = DEFAULT_VLM_TIMEOUT_SECONDS,
    preset: str = DEFAULT_TABLE_PRESET,
    game_type: str = "spin_and_go",
    transport: Callable[[str, dict[str, str], dict[str, Any], float], dict[str, Any]] | None = None,
    http_get: Callable[[str, float], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze a live captured RGBImage using 'hybrid', 'calibrated', or 'vlm' mode."""
    if mode not in ("hybrid", "calibrated", "vlm"):
        raise ValueError(f"Invalid vision mode: {mode!r}")

    if mode in ("calibrated", "hybrid"):
        t0 = time.perf_counter()
        cal = analyze_image(image)
        cal["vision_source"] = "calibrated"
        cal["vision_latency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
        cal_clean = (
            cal.get("table_count") == 1
            and not cal.get("ambiguities")
            and (cal.get("hand_state") is not None or not cal.get("hero_to_act"))
        )
        if mode == "calibrated" or cal_clean:
            return cal

    provider = vlm_provider
    model = vlm_model
    if not provider:
        if transport is not None:
            provider = "ollama"
            model = model or "qwen3-vl:2b-instruct"
        else:
            found = discover_vision_models(http_get=http_get)
            provider = found["recommended_provider"]
            model = model or found["recommended_model"]

    if not provider:
        if mode == "hybrid":
            cal["ambiguities"] = [
                *list(cal.get("ambiguities") or []),
                "vision_llm: nenhum modelo visual local (ex.: qwen3-vl:2b-instruct no Ollama) ou API cloud configurado.",
            ]
            return cal
        return {
            "image": {"width": image.width, "height": image.height},
            "table_count": 0,
            "table_count_source": "vision_llm (unavailable)",
            "vision_source": "unavailable",
            "vision_latency_ms": 0.0,
            "hero_cards": [],
            "board": [],
            "pot": None,
            "blinds": {"small": None, "big": None},
            "players": [],
            "active_player": None,
            "hero_to_act": False,
            "to_call": None,
            "hand_state": None,
            "opponent_ranges": "unknown; uniform combinations used only if the M1 analyzer is called",
            "ambiguities": ["vision_llm: no local vision model (e.g. qwen3-vl:2b-instruct) or cloud API available."],
        }

    return extract_table_with_vlm(
        image=image,
        provider=provider,
        model=model,
        endpoint=endpoint,
        api_key=api_key,
        timeout_seconds=timeout_seconds,
        preset=preset,
        game_type=game_type,
        transport=transport,
    )
