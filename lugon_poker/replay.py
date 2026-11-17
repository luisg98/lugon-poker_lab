"""Re-run archived diagnostic captures through the current code and compare with what was read at the time.

Captures live in ``data/captures/v<version>/`` next to a ``*_meta.json`` that records what the build that took them
understood. Replaying them with today's code answers "did this release actually fix what we saw in that release?".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .live_vision import DEFAULT_TABLE_PRESET, analyze_live_image
from .mercury_vision import analyze_mercury_table
from .screen_capture import mask_screen_rect_on_image
from .vision import load_png


def _cards(value: Any) -> list[str]:
    return [str(card) for card in (value or []) if card]


def _crop_to_target(image: Any, target: Any) -> Any:
    """Crop a full-screen archive image to the target window/region used by the live analyzer."""
    if not isinstance(target, dict) or target.get("kind") not in ("window", "region"):
        return image
    rect = target.get("rect")
    if not isinstance(rect, list) or len(rect) != 4:
        return image
    try:
        x, y, width, height = (int(value) for value in rect)
    except (TypeError, ValueError):
        return image
    left = max(0, min(image.width - 1, x))
    top = max(0, min(image.height - 1, y))
    right = max(left + 1, min(image.width, x + width))
    bottom = max(top + 1, min(image.height, y + height))
    cropped_width = right - left
    cropped_height = bottom - top
    row_bytes = cropped_width * 3
    src_stride = image.width * 3
    pixels = bytearray(row_bytes * cropped_height)
    for row in range(cropped_height):
        src_start = (top + row) * src_stride + left * 3
        dst_start = row * row_bytes
        pixels[dst_start : dst_start + row_bytes] = image.pixels[src_start : src_start + row_bytes]
    return type(image)(cropped_width, cropped_height, pixels)


def replay_captures(
    directory: Path,
    mask: tuple[int, int, int, int] | None = None,
    use_vlm: bool = False,
    game_type: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Replay every ``cap_*_full.png`` in ``directory``; returns one old-vs-new record per capture.

    With ``use_vlm=False`` only the deterministic Mercury pixel reader runs (fast, reproducible, no Ollama).
    A capture's own ``companion_rect`` is masked before analysis, exactly as the live app does; ``mask`` is the
    fallback for captures taken before that field existed.
    """
    from .companion_gui import evaluate_vision_report  # lazy: pulls in tkinter

    records: list[dict[str, Any]] = []
    pngs = sorted(Path(directory).glob("cap_*_full.png"))
    if limit is not None:
        pngs = pngs[-limit:]
    for png in pngs:
        meta_path = png.with_name(png.name.replace("_full.png", "_meta.json"))
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            meta = {}

        image = load_png(png)
        rect = meta.get("companion_rect")
        mask_rect = tuple(int(v) for v in rect) if isinstance(rect, list) and len(rect) == 4 else mask
        if mask_rect is not None:
            mask_screen_rect_on_image(image, (0, 0, image.width, image.height), mask_rect)
        target = meta.get("target")
        image = _crop_to_target(image, target)
        game = game_type or meta.get("game_type") or "spin_and_go"

        if use_vlm:
            preset = meta.get("preset") or DEFAULT_TABLE_PRESET
            mode = meta.get("mode")
            if mode not in ("hybrid", "calibrated", "vlm"):
                mode = "hybrid"
            report = analyze_live_image(image, mode=mode, game_type=game, preset=preset)
            status, _ = evaluate_vision_report(report)
            hero_player = next(
                (player for player in (report.get("players") or []) if player.get("hero")),
                None,
            )
            new = {
                "status": status,
                "hero_to_act": bool(report.get("hero_to_act")),
                "hero": _cards(report.get("hero_cards")),
                "board": _cards(report.get("board")),
                "pot": report.get("pot"),
                "to_call": report.get("to_call"),
                "hero_stack": hero_player.get("stack") if hero_player else None,
                "pot_source": report.get("pot_source"),
                "pot_label_text": report.get("pot_label_text"),
                "to_call_source": report.get("to_call_source"),
                "action_buttons": report.get("action_buttons"),
                "problems": list(report.get("ambiguities") or []),
            }
        else:
            mercury = analyze_mercury_table(image, game_type=game)
            if mercury is None:
                new = {"status": "no_table", "hero": [], "board": [], "problems": ["mesa Mercury não detetada"]}
            else:
                problems = []
                if not mercury["hero_complete"]:
                    problems.append("cartas do Hero incompletas")
                if not mercury["board_complete"]:
                    problems.append("board incompleto")
                new = {
                    "status": "complete" if not problems else "partial",
                    "hero": _cards(mercury["hero_cards"]),
                    "board": _cards(mercury["board"]),
                    "problems": problems,
                }

        records.append(
            {
                "file": png.name,
                "old_version": meta.get("app_version") or png.parent.name,
                "old": {
                    "status": meta.get("evaluate_status"),
                    "hero": _cards(meta.get("hero_cards")),
                    "board": _cards(meta.get("board")),
                    "pot": meta.get("pot"),
                    "to_call": meta.get("to_call"),
                    "problems": list(meta.get("ambiguities") or []),
                },
                "new": new,
            }
        )
    return records


def format_replay(records: list[dict[str, Any]]) -> str:
    """Human-readable old-vs-new table plus a one-line summary."""
    lines: list[str] = []
    changed = 0
    for rec in records:
        old, new = rec["old"], rec["new"]
        same = (
            old["status"] == new["status"]
            and old["hero"] == new["hero"]
            and old["board"] == new["board"]
            and all(old.get(key) == new[key] for key in ("pot", "to_call") if key in new)
        )
        changed += 0 if same else 1
        stamp = rec["file"][4:19]
        values = (
            f" pot={new['pot']}[{new.get('pot_source') or '?'}] "
            f"call={new['to_call']}[{new.get('to_call_source') or '?'}]"
            if "pot" in new
            else ""
        )
        buttons = new.get("action_buttons") or []
        if buttons:
            values += f" buttons={' / '.join(str(button) for button in buttons)}"
        lines.append(
            f"{stamp} {'==' if same else '!='} "
            f"antes[{old['status'] or '?'}] hero={' '.join(old['hero']) or '-'} board={' '.join(old['board']) or '-'}  |  "
            f"agora[{new['status']}] hero={' '.join(new['hero']) or '-'} board={' '.join(new['board']) or '-'}{values}"
        )
        for problem in new["problems"]:
            lines.append(f"            ! {problem}")
    lines.append(f"{len(records)} capturas reanalisadas, {changed} com leitura diferente da original")
    return "\n".join(lines)
