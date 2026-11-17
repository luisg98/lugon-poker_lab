"""Render synthetic single-table screenshots for repeatable vision tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .cards import Card
from .vision import (
    BASE_HEIGHT,
    BASE_WIDTH,
    BOARD_X,
    BOARD_Y,
    CARD_HEIGHT,
    CARD_WIDTH,
    FONT,
    HERO_CARD_X,
    HERO_CARD_Y,
    RGBImage,
    SEATS,
    _draw_text,
    _scale_x,
    _scale_y,
    load_scene,
    save_png,
)


def _filled_rect(image: RGBImage, x: float, y: float, width: float, height: float, color: tuple[int, int, int]) -> None:
    sx, sy = _scale_x(image), _scale_y(image)
    x0, y0 = round(x * sx), round(y * sy)
    x1, y1 = round((x + width) * sx), round((y + height) * sy)
    for py in range(y0, y1):
        for px in range(x0, x1):
            image.set(px, py, color)


def _ellipse(image: RGBImage, cx: float, cy: float, rx: float, ry: float, color: tuple[int, int, int]) -> None:
    sx, sy = _scale_x(image), _scale_y(image)
    center_x, center_y = round(cx * sx), round(cy * sy)
    radius_x, radius_y = round(rx * sx), round(ry * sy)
    for y in range(max(0, center_y - radius_y), min(image.height, center_y + radius_y + 1)):
        for x in range(max(0, center_x - radius_x), min(image.width, center_x + radius_x + 1)):
            if ((x - center_x) / max(1, radius_x)) ** 2 + ((y - center_y) / max(1, radius_y)) ** 2 <= 1:
                image.set(x, y, color)


def _chip_text(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"


def _draw_card(image: RGBImage, card_text: str, x: float, y: float) -> None:
    _filled_rect(image, x, y, CARD_WIDTH, CARD_HEIGHT, (245, 245, 238))
    _draw_text(image, card_text, x + 4, y + 5, (22, 25, 28), scale=2)


def render_scene(scene: dict[str, Any], width: int = BASE_WIDTH, height: int = BASE_HEIGHT) -> RGBImage:
    if width < 640 or height < 360 or width / height != BASE_WIDTH / BASE_HEIGHT:
        raise ValueError("synthetic captures require at least 640x360 and a 16:9 aspect ratio")
    if len(scene.get("players", [])) > len(SEATS):
        raise ValueError(f"synthetic layout supports at most {len(SEATS)} players")
    hero_cards = [Card.parse(value) for value in scene.get("hero_cards", [])]
    board = [Card.parse(value) for value in scene.get("board", [])]
    if len(hero_cards) != 2 or len(board) not in (0, 3, 4, 5):
        raise ValueError("synthetic scene needs two hero cards and 0, 3, 4, or 5 board cards")
    if len({card.code for card in (*hero_cards, *board)}) != len(hero_cards) + len(board):
        raise ValueError("hero and board cannot repeat a card")
    players = scene.get("players", [])
    if not isinstance(players, list) or not players or sum(bool(player.get("hero")) for player in players) != 1:
        raise ValueError("scene must have players and exactly one hero")
    if sum(bool(player.get("active")) for player in players) != 1:
        raise ValueError("scene must have exactly one active player")
    used_seats = [player.get("seat") for player in players]
    if len(set(used_seats)) != len(used_seats) or any(not isinstance(seat, int) or not 0 <= seat < len(SEATS) for seat in used_seats):
        raise ValueError("player seat indexes must be unique integers from zero to five")

    image = RGBImage(width, height, bytearray((14, 19, 25) * width * height))
    _ellipse(image, 480, 270, 420, 225, (107, 76, 42))
    _ellipse(image, 480, 270, 410, 215, (18, 100, 75))

    if scene.get("small_blind") is not None:
        _draw_text(image, "SB", 420, 173, (245, 245, 238), scale=2)
        _draw_text(image, _chip_text(float(scene["small_blind"])), 474, 173, (245, 245, 238), scale=2)
    if scene.get("big_blind") is not None:
        _draw_text(image, "BB", 525, 173, (245, 245, 238), scale=2)
        _draw_text(image, _chip_text(float(scene["big_blind"])), 560, 173, (245, 245, 238), scale=2)

    _draw_text(image, "POT", 426, 198, (245, 245, 238), scale=2)
    _draw_text(image, _chip_text(float(scene["pot"])), 480, 198, (245, 245, 238), scale=2)

    for index, card in enumerate(board):
        _draw_card(image, str(card).upper(), BOARD_X[index], BOARD_Y)
    for index, card in enumerate(hero_cards):
        _draw_card(image, str(card).upper(), HERO_CARD_X[index], HERO_CARD_Y)

    for player in players:
        seat_index = player["seat"]
        x, y = SEATS[seat_index]
        _filled_rect(image, x, y, 8, 10, (37, 114, 215) if player.get("hero") else (14, 19, 25))
        if player.get("active"):
            _filled_rect(image, x + 9, y, 8, 10, (218, 54, 48))
        elif player.get("folded"):
            _filled_rect(image, x + 9, y, 8, 10, (145, 145, 145))

        name = str(player["nickname"]).upper()
        if not name or any(character not in FONT and character != " " for character in name):
            raise ValueError(f"synthetic nickname uses an unsupported character: {name!r}")
        _draw_text(image, name, x + 20, y, (250, 250, 246), scale=2)
        _draw_text(image, "POS", x + 20, y + 17, (182, 207, 196), scale=2)
        _draw_text(image, str(player["position"]).upper(), x + 66, y + 17, (250, 250, 246), scale=2)
        _draw_text(image, "STACK", x + 20, y + 34, (182, 207, 196), scale=2)
        _draw_text(image, _chip_text(float(player["stack"])), x + 95, y + 34, (250, 250, 246), scale=2)
        _draw_text(image, "BET", x + 20, y + 51, (182, 207, 196), scale=2)
        _draw_text(image, _chip_text(float(player["bet"])), x + 62, y + 51, (250, 250, 246), scale=2)

    return image


def render_file(scene_path: str | Path, output_path: str | Path, width: int = BASE_WIDTH, height: int = BASE_HEIGHT) -> None:
    scene = load_scene(scene_path)
    image = render_scene(scene, width, height)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_png(image, output_path)
