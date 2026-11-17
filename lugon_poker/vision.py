"""Dependency-free PNG reader and calibrated single-table visual parser.

The first vision baseline is deliberately scoped to one known table layout. It
uses normalized regions and a small bitmap glyph recognizer; it is a measurable
prototype for synthetic/authorized captures, not a general poker-site OCR model.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import struct
import zlib
from typing import Any

from .cards import Card
from .state import HandState

BASE_WIDTH = 960
BASE_HEIGHT = 540
BOARD_X = (350, 393, 436, 479, 522)
BOARD_Y = 230
HERO_CARD_X = (430, 473)
HERO_CARD_Y = 350
CARD_WIDTH = 36
CARD_HEIGHT = 50
SEATS = ((70, 105), (380, 65), (665, 105), (690, 320), (380, 430), (55, 320))


@dataclass
class RGBImage:
    width: int
    height: int
    pixels: bytearray

    def get(self, x: int, y: int) -> tuple[int, int, int]:
        index = (y * self.width + x) * 3
        return self.pixels[index], self.pixels[index + 1], self.pixels[index + 2]

    def set(self, x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            index = (y * self.width + x) * 3
            self.pixels[index : index + 3] = bytes(color)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else (b if pb <= pc else c)


def load_png(path: str | Path) -> RGBImage:
    """Read non-interlaced 8-bit PNGs (RGB, RGBA, grayscale, or palette)."""
    data = Path(path).read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("input must be a PNG image")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    compressed = bytearray()
    palette: bytes | None = None
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG chunk")
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        kind = data[offset + 4 : offset + 8]
        chunk = data[offset + 8 : offset + 8 + length]
        if len(chunk) != length:
            raise ValueError("truncated PNG data")
        if kind == b"IHDR":
            width, height, bit_depth, color_type, compression, filtering, interlace = struct.unpack(">IIBBBBB", chunk)
            if compression != 0 or filtering != 0:
                raise ValueError("unsupported PNG compression or filter method")
        elif kind == b"IDAT":
            compressed.extend(chunk)
        elif kind == b"PLTE":
            palette = bytes(chunk)
        elif kind == b"IEND":
            break
        offset += 12 + length

    if not width or not height or bit_depth != 8 or interlace != 0:
        raise ValueError("PNG must be non-interlaced, 8-bit, and have non-zero dimensions")
    channel_counts = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if color_type not in channel_counts:
        raise ValueError(f"unsupported PNG color type: {color_type}")
    channels = channel_counts[color_type]
    stride = width * channels
    raw = zlib.decompress(compressed)
    expected = height * (stride + 1)
    if len(raw) != expected:
        raise ValueError("PNG decompressed data length does not match its dimensions")

    rows: list[bytearray] = []
    prior = bytearray(stride)
    cursor = 0
    for _ in range(height):
        filter_type = raw[cursor]
        source = raw[cursor + 1 : cursor + 1 + stride]
        cursor += stride + 1
        row = bytearray(stride)
        for i, byte in enumerate(source):
            left = row[i - channels] if i >= channels else 0
            above = prior[i]
            upper_left = prior[i - channels] if i >= channels else 0
            if filter_type == 0:
                predictor = 0
            elif filter_type == 1:
                predictor = left
            elif filter_type == 2:
                predictor = above
            elif filter_type == 3:
                predictor = (left + above) // 2
            elif filter_type == 4:
                predictor = _paeth(left, above, upper_left)
            else:
                raise ValueError(f"unsupported PNG row filter: {filter_type}")
            row[i] = (byte + predictor) & 255
        rows.append(row)
        prior = row

    rgb = bytearray(width * height * 3)
    out = 0
    for row in rows:
        if color_type in (2, 6):
            source_channels = 3 if color_type == 2 else 4
            for x in range(width):
                start = x * source_channels
                rgb[out : out + 3] = row[start : start + 3]
                out += 3
        elif color_type in (0, 4):
            source_channels = 1 if color_type == 0 else 2
            for x in range(width):
                shade = row[x * source_channels]
                rgb[out : out + 3] = bytes((shade, shade, shade))
                out += 3
        else:
            if palette is None:
                raise ValueError("palette PNG has no PLTE chunk")
            for index in row:
                start = index * 3
                if start + 2 >= len(palette):
                    raise ValueError("PNG palette index is out of range")
                rgb[out : out + 3] = palette[start : start + 3]
                out += 3
    return RGBImage(width, height, rgb)


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def encode_png_bytes(image: RGBImage, compress_level: int = 6) -> bytes:
    """Encode an RGBImage into 8-bit RGB PNG bytes."""
    if len(image.pixels) != image.width * image.height * 3:
        raise ValueError("image pixel buffer has the wrong size")
    stride = image.width * 3
    rows = bytearray((stride + 1) * image.height)
    src = memoryview(image.pixels)
    dst = memoryview(rows)
    row_span = stride + 1
    for y in range(image.height):
        out_start = y * row_span
        rows[out_start] = 0
        in_start = y * stride
        dst[out_start + 1 : out_start + row_span] = src[in_start : in_start + stride]
    ihdr = struct.pack(">IIBBBBB", image.width, image.height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + _png_chunk(b"IHDR", ihdr) + _png_chunk(b"IDAT", zlib.compress(rows, compress_level)) + _png_chunk(b"IEND", b"")


def save_png(image: RGBImage, path: str | Path) -> None:
    Path(path).write_bytes(encode_png_bytes(image))


def _scale_x(image: RGBImage) -> float:
    return image.width / BASE_WIDTH


def _scale_y(image: RGBImage) -> float:
    return image.height / BASE_HEIGHT


def _rect(image: RGBImage, x: float, y: float, width: float, height: float) -> tuple[int, int, int, int]:
    sx, sy = _scale_x(image), _scale_y(image)
    x0, y0 = round(x * sx), round(y * sy)
    return x0, y0, round((x + width) * sx), round((y + height) * sy)


def _font_scale(image: RGBImage) -> int:
    return max(1, round(2 * _scale_y(image)))


FONT: dict[str, tuple[str, ...]] = {
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01111"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("111", "010", "010", "010", "010", "010", "111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("010", "110", "010", "010", "010", "010", "111"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "-": ("000", "000", "000", "111", "000", "000", "000"),
    ".": ("000", "000", "000", "000", "000", "010", "010"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "+": ("000", "010", "010", "111", "010", "010", "000"),
    "?": ("11110", "00001", "00010", "00100", "00100", "00000", "00100"),
}


def _trim_glyph(rows: tuple[str, ...]) -> tuple[str, ...]:
    lit_columns = [x for x in range(len(rows[0])) if any(row[x] == "1" for row in rows)]
    if not lit_columns:
        return rows
    left, right = min(lit_columns), max(lit_columns) + 1
    return tuple(row[left:right] for row in rows)


GLYPHS = {_trim_glyph(rows): char for char, rows in FONT.items()}


def _draw_text(image: RGBImage, text: str, x: float, y: float, color: tuple[int, int, int], scale: int) -> None:
    sx, sy = _scale_x(image), _scale_y(image)
    cursor = round(x * sx)
    top = round(y * sy)
    pixel_scale = max(1, round(scale * sy))
    for char in text.upper():
        if char == " ":
            cursor += 4 * pixel_scale
            continue
        glyph = FONT.get(char, FONT["?"])
        for row_index, row in enumerate(glyph):
            for col_index, value in enumerate(row):
                if value == "1":
                    for dy in range(pixel_scale):
                        for dx in range(pixel_scale):
                            image.set(cursor + col_index * pixel_scale + dx, top + row_index * pixel_scale + dy, color)
        cursor += (len(glyph[0]) + 1) * pixel_scale


def _read_text(image: RGBImage, rect: tuple[float, float, float, float], dark_ink: bool = False) -> str | None:
    x, y, width, height = rect
    x0, y0, x1, y1 = _rect(image, x, y, width, height)
    scale = _font_scale(image)
    if x1 <= x0 or y1 <= y0:
        return None
    mask: list[list[bool]] = []
    for row in range(7):
        line: list[bool] = []
        cell_y0 = y0 + row * scale
        cell_y1 = min(y1, cell_y0 + scale)
        for col in range(max(0, (x1 - x0 + scale - 1) // scale)):
            cell_x0 = x0 + col * scale
            cell_x1 = min(x1, cell_x0 + scale)
            hits = 0
            total = 0
            for sample_y in range(cell_y0, cell_y1):
                for sample_x in range(cell_x0, cell_x1):
                    red, green, blue = image.get(sample_x, sample_y)
                    ink = max(red, green, blue) < 110 if dark_ink else min(red, green, blue) > 170
                    hits += int(ink)
                    total += 1
            line.append(total > 0 and hits / total >= 0.2)
        mask.append(line)

    occupied = [any(mask[row][col] for row in range(7)) for col in range(len(mask[0]))]
    groups: list[tuple[int, int]] = []
    cursor = 0
    while cursor < len(occupied):
        while cursor < len(occupied) and not occupied[cursor]:
            cursor += 1
        start = cursor
        while cursor < len(occupied) and occupied[cursor]:
            cursor += 1
        if start < cursor:
            groups.append((start, cursor))

    if not groups:
        return None
    decoded: list[str] = []
    prior_end: int | None = None
    for start, end in groups:
        if prior_end is not None and start - prior_end > 2:
            decoded.append(" ")
        rows = tuple("".join("1" if mask[row][col] else "0" for col in range(start, end)) for row in range(7))
        glyph = GLYPHS.get(rows)
        decoded.append(glyph or "?")
        prior_end = end
    result = "".join(decoded).strip()
    return result or None


def _is_marker(image: RGBImage, x: float, y: float, color: str) -> bool:
    x0, y0, x1, y1 = _rect(image, x, y, 8, 10)
    pixels = [image.get(px, py) for py in range(y0, y1) for px in range(x0, x1)]
    if not pixels:
        return False
    if color == "hero":
        return sum(blue > 170 and red < 80 and green > 100 for red, green, blue in pixels) > len(pixels) * 0.35
    if color == "active":
        return sum(red > 180 and green < 90 and blue < 90 for red, green, blue in pixels) > len(pixels) * 0.35
    return sum(red > 100 and abs(red - green) < 20 and abs(green - blue) < 20 for red, green, blue in pixels) > len(pixels) * 0.35


def _has_calibrated_table(image: RGBImage) -> bool:
    sample_points = ((160, 270), (800, 270), (480, 85), (480, 455))
    green_samples = 0
    for x, y in sample_points:
        px = min(image.width - 1, round(x * _scale_x(image)))
        py = min(image.height - 1, round(y * _scale_y(image)))
        red, green, blue = image.get(px, py)
        green_samples += int(green > 65 and green > red * 1.3 and green > blue * 1.1)
    return green_samples >= 3


def _has_card(image: RGBImage, x: float, y: float) -> bool:
    x0, y0, x1, y1 = _rect(image, x, y, CARD_WIDTH, CARD_HEIGHT)
    points = [image.get(px, py) for py in range(y0 + 2, y1 - 2, max(1, round(_scale_y(image) * 3))) for px in range(x0 + 2, x1 - 2, max(1, round(_scale_x(image) * 3)))]
    if not points:
        return False
    bright = sum(min(red, green, blue) > 210 for red, green, blue in points)
    return bright / len(points) > 0.72


def _read_card(image: RGBImage, x: float, y: float, field: str, ambiguities: list[str]) -> str | None:
    if not _has_card(image, x, y):
        return None
    text = _read_text(image, (x + 4, y + 5, CARD_WIDTH - 6, 16), dark_ink=True)
    if not text or "?" in text or len(text.replace(" ", "")) != 2:
        ambiguities.append(f"{field}: symbol is present but the card is unreadable")
        return None
    token = text.replace(" ", "").lower()
    token = token[0].upper() + token[1]
    try:
        Card.parse(token)
    except ValueError:
        ambiguities.append(f"{field}: invalid card reading ({text})")
        return None
    return token


def _read_amount(image: RGBImage, x: float, y: float, field: str, ambiguities: list[str], width: float = 92) -> float | None:
    text = _read_text(image, (x, y, width, 16))
    if text is None:
        ambiguities.append(f"{field}: value is not visible")
        return None
    cleaned = text.replace(" ", "")
    if not re.fullmatch(r"\d+(?:\.\d+)?", cleaned):
        ambiguities.append(f"{field}: ambiguous value ({text})")
        return None
    return float(cleaned)


def analyze_table(image_path: str | Path) -> dict[str, Any]:
    return analyze_image(load_png(image_path))


def analyze_image(image: RGBImage) -> dict[str, Any]:
    ambiguities: list[str] = []
    table_detected = _has_calibrated_table(image)
    if not table_detected:
        ambiguities.append("table: calibrated single-table layout not recognized")
    board: list[str] = []
    seen_empty_board_slot = False
    board_order_ambiguous = False
    board_card_ambiguous = False
    for index, x in enumerate(BOARD_X):
        card_present = _has_card(image, x, BOARD_Y)
        card = _read_card(image, x, BOARD_Y, f"board[{index}]", ambiguities)
        if card_present and card is None:
            board_card_ambiguous = True
        if card is None:
            seen_empty_board_slot = True
        elif seen_empty_board_slot:
            ambiguities.append("board: a card appears after an empty slot")
            board_order_ambiguous = True
        else:
            board.append(card)

    hero_cards = [_read_card(image, x, HERO_CARD_Y, f"hero.cards[{index}]", ambiguities) for index, x in enumerate(HERO_CARD_X)]
    players: list[dict[str, Any]] = []
    for seat_index, (x, y) in enumerate(SEATS):
        nickname = _read_text(image, (x + 20, y, 190, 16))
        if nickname is None:
            continue
        if "?" in nickname:
            ambiguities.append(f"players[{seat_index}].nickname: unrecognized characters")
            nickname = None
        position = _read_text(image, (x + 66, y + 17, 80, 16))
        stack = _read_amount(image, x + 95, y + 34, f"players[{seat_index}].stack", ambiguities, width=80)
        bet = _read_amount(image, x + 62, y + 51, f"players[{seat_index}].bet", ambiguities)
        is_hero = _is_marker(image, x, y, "hero")
        active = _is_marker(image, x + 9, y, "active")
        folded = _is_marker(image, x + 9, y, "folded")
        players.append({
            "nickname": nickname,
            "position": position,
            "stack": stack,
            "bet": bet,
            "hero": is_hero,
            "active": active,
            "folded": folded,
            "seat": seat_index,
        })

    pot = _read_amount(image, 480, 198, "pot", ambiguities)
    small_blind = _read_amount(image, 474, 173, "small_blind", ambiguities, width=36)
    big_blind = _read_amount(image, 560, 173, "big_blind", ambiguities)
    heroes = [player for player in players if player["hero"]]
    active_players = [player for player in players if player["active"]]
    if len(heroes) != 1:
        ambiguities.append(f"hero: expected one marker, found {len(heroes)}")
    if len(active_players) != 1:
        ambiguities.append(f"active_player: expected one marker, found {len(active_players)}")

    hero = heroes[0] if len(heroes) == 1 else None
    active_player = active_players[0] if len(active_players) == 1 else None
    opponents = [player for player in players if not player["hero"] and not player["folded"]]
    to_call: float | None = None
    if hero and hero["bet"] is not None and all(player["bet"] is not None for player in players if not player["folded"]):
        highest_bet = max((player["bet"] for player in players if not player["folded"]), default=hero["bet"])
        to_call = max(0.0, highest_bet - hero["bet"])
    else:
        ambiguities.append("to_call: could not determine the amount from visible bets")

    hand_state: dict[str, Any] | None = None
    if table_detected and hero and not board_order_ambiguous and not board_card_ambiguous and all(hero_cards) and pot is not None and to_call is not None and 1 <= len(opponents) <= 9 and len(board) in (0, 3, 4, 5):
        candidate = {
            "hero": {"nickname": hero["nickname"], "cards": hero_cards},
            "board": board,
            "pot": pot,
            "to_call": to_call,
            "opponents": [{"nickname": player["nickname"] or f"Opponent {index}"} for index, player in enumerate(opponents, start=1)],
        }
        try:
            HandState.from_dict(candidate)
            hand_state = candidate
        except ValueError as error:
            ambiguities.append(f"hand_state: {error}")
    elif len(board) not in (0, 3, 4, 5):
        ambiguities.append(f"board: expected 0, 3, 4, or 5 cards, found {len(board)}")

    return {
        "image": {"width": image.width, "height": image.height},
        "table_count": 1 if table_detected else 0,
        "table_count_source": "calibrated proof-of-concept layout; does not detect additional tables",
        "hero_cards": hero_cards,
        "board": board,
        "pot": pot,
        "blinds": {"small": small_blind, "big": big_blind},
        "players": players,
        "active_player": active_player["nickname"] if active_player else None,
        "hero_to_act": bool(hero and active_player and hero["seat"] == active_player["seat"]),
        "to_call": to_call,
        "hand_state": hand_state,
        "opponent_ranges": "unknown; uniform combinations used only if the M1 analyzer is called",
        "ambiguities": ambiguities,
    }


def compare_fixture(expected: dict[str, Any], actual: dict[str, Any]) -> dict[str, int]:
    """Count exact field matches for a synthetic fixture; not a real-image metric."""
    checks: list[tuple[Any, Any]] = [
        ([str(Card.parse(card)) for card in expected["hero_cards"]], actual["hero_cards"]),
        ([str(Card.parse(card)) for card in expected["board"]], actual["board"]),
        (float(expected["pot"]), actual["pot"]),
        (float(expected["blinds"]["small"]), actual["blinds"]["small"]),
        (float(expected["blinds"]["big"]), actual["blinds"]["big"]),
        (expected["active_player"], actual["active_player"]),
        (float(expected["to_call"]), actual["to_call"]),
        (expected["players"], [
            {key: item[key] for key in ("nickname", "position", "stack", "bet", "hero", "active", "folded", "seat")}
            for item in actual["players"]
        ]),
    ]
    passed = sum(wanted == received for wanted, received in checks)
    return {"fields_correct": passed, "fields_total": len(checks)}


def load_scene(path: str | Path) -> dict[str, Any]:
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(scene, dict):
        raise ValueError("scene must be a JSON object")
    return scene
