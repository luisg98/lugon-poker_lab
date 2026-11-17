"""Deterministic 888 BLAST 4-color card, seat, dealer, and numeric-collage extractor."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from .screen_capture import crop_rgb_image, resize_rgb_image
from .vision import RGBImage

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
TEMPLATE_WIDTH = 12
TEMPLATE_HEIGHT = 16

# Canonical 12x16 normalized rank templates and width/height aspect ratios extracted from 888 BLAST 4-color cards.
_BLAST_RANK_SPECS: dict[str, tuple[str, float]] = {
    "2": (
        "....####....|..#########.|.##########.|.###....####|..#.....####|........####|.......####.|......#####.|......####..|.....####...|....####....|..####......|.#####......|###########.|###########.|.##########.",
        0.567,
    ),
    "3": (
        ".##########.|.##########.|.....#####..|.....####...|....####....|....###.....|...#######..|......#####.|........####|.........###|.........###|........####|.##.....####|###########.|.#########..|....###.....",
        0.554,
    ),
    "4": (
        "........###.|.......####.|......#####.|......#####.|....#######.|....##..###.|...###..###.|..###...###.|..##....###.|.###....##..|.##########.|############|...#########|.......###..|.......###..|.......###..",
        0.633,
    ),
    "5": (
        "...#######..|.#########..|.#########..|.###........|.###........|.########...|.#########..|.####..####.|...#....####|........####|.........###|........####|.##.....####|###########.|.#########..|...#####....",
        0.558,
    ),
    "6": (
        "......##....|.....###....|....####....|....###.....|...###......|..#####.....|.#########..|.##########.|.###....###.|####....####|###......###|####....####|.###....###.|.####..####.|..########..|....####....",
        0.586,
    ),
    "7": (
        ".#########..|.###########|.###########|.......####.|.......####.|......#####.|......####..|.....####...|.....####...|....####....|....####....|...####.....|...###......|..####......|.#####......|..###.......",
        0.565,
    ),
    "8": (
        "...#####....|.#########..|.####..####.|####...####.|###....####.|.###...####.|.#########..|...######...|..#########.|.###....###.|###......###|###......###|####....####|.##########.|..########..|.....##.....",
        0.583,
    ),
    "9": (
        "....####....|..########..|.##########.|####....####|###.....####|###......###|####....####|.###....###.|.####.#####.|.##########.|....######..|......###...|.....###....|....####....|....###.....|...###......",
        0.554,
    ),
    "T": (
        "..#....###..|.##...#####.|###..###.##.|###..##..##.|..#..##...##|..#..##...##|..#..##...##|..#..##...##|..#..##...##|..#..##...##|..#..##...##|..#..##...##|..#..###..#.|..##..#####.|..##..#####.|..##....#...",
        1.133,
    ),
    "J": (
        ".......#####|.......#####|.......#####|.......#####|.......#####|.......#####|.......#####|.......#####|.......#####|.......#####|.......#####|.......#####|......######|.##########.|##########..|.#######....",
        0.448,
    ),
    "Q": (
        ".....##.....|...######...|..########..|.###...####.|###.....###.|###......##.|###......###|###......###|###......###|###......###|###...#####.|###...#####.|.###...####.|.##########.|..#########.|....###..##.",
        0.800,
    ),
    "K": (
        "........##..|###....###..|###...###...|###..####...|.##.####....|.##.###.....|.######.....|.#######....|.#######....|.########...|.###.####...|.###..####..|.###...###..|.###....###.|.###....###.|.###........",
        0.705,
    ),
    "A": (
        "....###.....|....####....|...#####....|...#####....|...##.###...|...##.###...|..###..##...|..###..##...|..##...###..|..##...###..|.#########..|.#########..|.##########.|.##.....###.|###......##.|###......###",
        0.747,
    ),
}


def _decode_pattern(pattern: str) -> tuple[int, ...]:
    return tuple(1 if ch == "#" else 0 for ch in pattern.replace("|", ""))


_BLAST_TEMPLATES: dict[str, tuple[int, ...]] = {
    rank: _decode_pattern(spec[0]) for rank, spec in _BLAST_RANK_SPECS.items()
}
_BLAST_ASPECTS: dict[str, float] = {
    rank: spec[1] for rank, spec in _BLAST_RANK_SPECS.items()
}

# Fixed normalized boxes on the 888 BLAST table window (~1326x1025).
_HERO_GLYPH_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (0.440, 0.474, 0.768, 0.806),
    (0.494, 0.528, 0.765, 0.803),
)
_HERO_SUIT_BOXES: tuple[tuple[float, float, float, float], ...] = (
    (0.440, 0.500, 0.765, 0.795),
    (0.500, 0.565, 0.765, 0.795),
)


@dataclass(frozen=True)
class BlastCardSlot:
    present: bool
    suit: str | None
    rank: str | None
    score: float
    margin: float
    clipped: bool = False

    @property
    def card(self) -> str | None:
        if self.present and self.rank and self.suit:
            return f"{self.rank}{self.suit}"
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "present": self.present,
            "suit": self.suit,
            "rank": self.rank,
            "card": self.card,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "clipped": self.clipped,
        }


def _is_white_glyph(r: int, g: int, b: int) -> bool:
    return min(r, g, b) >= 195 and (max(r, g, b) - min(r, g, b)) <= 45


def classify_blast_suit_rgb(r: int, g: int, b: int) -> str | None:
    if r > 85 and r > g * 1.35 and r > b * 1.2:
        return "h"
    if g > 85 and g > r * 1.25 and g > b * 1.1:
        return "c"
    if b > 85 and b > r * 1.2 and b > g * 1.15:
        return "d"
    if max(r, g, b) < 75:
        return "s"
    return None


def _sample_suit_in_box(image: RGBImage, rx0: float, rx1: float, ry0: float, ry1: float) -> str | None:
    counts = {"h": 0, "c": 0, "d": 0, "s": 0}
    y_start, y_end = max(0, round(image.height * ry0)), min(image.height, round(image.height * ry1))
    x_start, x_end = max(0, round(image.width * rx0)), min(image.width, round(image.width * rx1))
    for y in range(y_start, y_end, 2):
        for x in range(x_start, x_end, 2):
            suit = classify_blast_suit_rgb(*image.get(x, y))
            if suit is not None:
                counts[suit] += 1
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    total = sum(counts.values())
    if total < 100 or ranked[0][1] < total * 0.50 or ranked[0][1] < ranked[1][1] * 1.25:
        return None
    return ranked[0][0]


def _extract_raw_glyph(
    image: RGBImage,
    rx0: float,
    rx1: float,
    ry0: float,
    ry1: float,
) -> list[list[int]] | None:
    x0, x1 = max(0, round(image.width * rx0)), min(image.width, round(image.width * rx1))
    y0, y1 = max(0, round(image.height * ry0)), min(image.height, round(image.height * ry1))
    if x1 - x0 < 5 or y1 - y0 < 10:
        return None
    grid = [
        [1 if _is_white_glyph(*image.get(x, y)) else 0 for x in range(x0, x1)]
        for y in range(y0, y1)
    ]
    row_sums = [sum(row) for row in grid]
    first_y = next((idx for idx, total in enumerate(row_sums) if total >= 2), None)
    if first_y is None:
        return None
    last_y = first_y
    for idx in range(first_y, len(grid)):
        if row_sums[idx] >= 2:
            last_y = idx
        elif idx - first_y >= max(10, round(image.height * 0.0115)):
            # Horizontal gap between the rank glyph and the small suit pip directly underneath.
            break
    sub = grid[first_y : last_y + 1]
    width = len(sub[0])
    col_sums = [sum(sub[r][c] for r in range(len(sub))) for c in range(width)]
    first_x = next((idx for idx, total in enumerate(col_sums) if total >= 1), None)
    if first_x is None:
        return None
    last_x = max(idx for idx, total in enumerate(col_sums) if total >= 1)
    cropped = [row[first_x : last_x + 1] for row in sub]
    if len(cropped) < 9 or len(cropped[0]) < 4:
        return None
    return cropped


def _resize_binary(mat: list[list[int]], w_out: int, h_out: int) -> tuple[int, ...]:
    gh, gw = len(mat), len(mat[0])
    return tuple(
        mat[min(gh - 1, int((oy + 0.5) * gh / h_out))][min(gw - 1, int((ox + 0.5) * gw / w_out))]
        for oy in range(h_out)
        for ox in range(w_out)
    )


def _rotate_template(tpl: tuple[int, ...], angle_deg: float) -> tuple[int, ...]:
    """Rotate a canonical 12x16 template by angle_deg and re-normalize to tight bounding box."""
    if abs(angle_deg) < 1e-6:
        return tpl
    rad = math.radians(angle_deg)
    ca, sa = math.cos(rad), math.sin(rad)
    # Render into an oversampled padded grid, then tightly crop and resize back to 12x16
    scale = 2
    pw, ph = TEMPLATE_WIDTH * scale + 12, TEMPLATE_HEIGHT * scale + 12
    cx_out, cy_out = (pw - 1) / 2.0, (ph - 1) / 2.0
    cx_in, cy_in = (TEMPLATE_WIDTH - 1) / 2.0, (TEMPLATE_HEIGHT - 1) / 2.0
    mat: list[list[int]] = []
    for y in range(ph):
        dy = (y - cy_out) / scale
        row: list[int] = []
        for x in range(pw):
            dx = (x - cx_out) / scale
            sx = round(cx_in + dx * ca + dy * sa)
            sy = round(cy_in - dx * sa + dy * ca)
            val = tpl[sy * TEMPLATE_WIDTH + sx] if (0 <= sx < TEMPLATE_WIDTH and 0 <= sy < TEMPLATE_HEIGHT) else 0
            row.append(val)
        mat.append(row)
    ys = [y for y in range(ph) if any(mat[y])]
    if not ys:
        return tpl
    xs = [x for x in range(pw) if any(mat[y][x] for y in ys)]
    tight = [mat[y][min(xs) : max(xs) + 1] for y in range(min(ys), max(ys) + 1)]
    return _resize_binary(tight, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)


_BLAST_TEMPLATE_VARIANTS: dict[str, tuple[tuple[int, ...], ...]] = {
    rank: (
        tpl,
        _rotate_template(tpl, -6.5),
        _rotate_template(tpl, 6.5),
        _rotate_template(tpl, 11.0),
    )
    for rank, tpl in _BLAST_TEMPLATES.items()
}


def _match_blast_rank(mat: list[list[int]], image_height: int) -> tuple[str | None, float, float, bool]:
    gh, gw = len(mat), len(mat[0])
    rel_h = gh / max(1, image_height)
    expected_full_h = max(12.0, image_height * 0.0283)
    clipped = rel_h < 0.024

    if not clipped:
        norm = _resize_binary(mat, TEMPLATE_WIDTH, TEMPLATE_HEIGHT)
        aspect = gw / max(1, gh)
        scored: list[tuple[float, str, float]] = []
        for rank, variants in _BLAST_TEMPLATE_VARIANTS.items():
            pix = max(
                sum(1 for a, b in zip(norm, tpl) if a == b) / (TEMPLATE_WIDTH * TEMPLATE_HEIGHT)
                for tpl in variants
            )
            asp_pen = min(0.15, abs(aspect - _BLAST_ASPECTS[rank]) * 0.12)
            scored.append((pix - asp_pen, rank, pix))
        scored.sort(reverse=True)
        best_adj, best_rank, best_pix = scored[0]
        margin = best_adj - scored[1][0]
        if best_pix >= 0.80 and margin >= 0.04:
            return best_rank, best_pix, margin, False
        return None, best_pix, margin, False

    # Bottom of glyph clipped by an action/blind badge (e.g. PASSAR, SMALL BLIND).
    # Compare against the proportional top portion of each canonical template + width aspect.
    frac = max(0.50, min(0.95, gh / expected_full_h))
    keep_rows = max(8, min(TEMPLATE_HEIGHT - 1, round(TEMPLATE_HEIGHT * frac)))
    norm_top = _resize_binary(mat, TEMPLATE_WIDTH, keep_rows)
    aspect_vs_full = gw / expected_full_h
    scored_top: list[tuple[float, str, float]] = []
    for rank, variants in _BLAST_TEMPLATE_VARIANTS.items():
        best_pix_r = 0.0
        best_top_width_frac = 1.0
        for tpl in variants:
            tpl_rows = [list(tpl[y * TEMPLATE_WIDTH : (y + 1) * TEMPLATE_WIDTH]) for y in range(keep_rows)]
            col_sums = [sum(tpl_rows[y][x] for y in range(keep_rows)) for x in range(TEMPLATE_WIDTH)]
            fx = next((x for x, s in enumerate(col_sums) if s > 0), 0)
            lx = max((x for x, s in enumerate(col_sums) if s > 0), default=TEMPLATE_WIDTH - 1)
            tpl_sub = [row[fx : lx + 1] for row in tpl_rows]
            tpl_norm = _resize_binary(tpl_sub, TEMPLATE_WIDTH, keep_rows)
            pix = sum(1 for a, b in zip(norm_top, tpl_norm) if a == b) / (TEMPLATE_WIDTH * keep_rows)
            if pix > best_pix_r:
                best_pix_r = pix
                best_top_width_frac = (lx - fx + 1) / TEMPLATE_WIDTH
        expected_top_aspect = _BLAST_ASPECTS[rank] * best_top_width_frac
        asp_pen = min(0.18, abs(aspect_vs_full - expected_top_aspect) * 0.25)
        scored_top.append((best_pix_r - asp_pen, rank, best_pix_r))
    scored_top.sort(reverse=True)
    best_adj, best_rank, best_pix = scored_top[0]
    margin = best_adj - scored_top[1][0]
    if best_pix >= 0.80 and margin >= 0.045:
        return best_rank, best_pix, margin, True
    return None, best_pix, margin, True


def _scan_hero_slots(image: RGBImage) -> list[BlastCardSlot]:
    slots: list[BlastCardSlot] = []
    for idx in range(2):
        suit = _sample_suit_in_box(image, *_HERO_SUIT_BOXES[idx])
        mat = _extract_raw_glyph(image, *_HERO_GLYPH_BOXES[idx])
        if mat is None:
            slots.append(BlastCardSlot(present=False, suit=suit, rank=None, score=0.0, margin=0.0))
            continue
        rank, score, margin, clipped = _match_blast_rank(mat, image.height)
        slots.append(
            BlastCardSlot(
                present=True,
                suit=suit,
                rank=rank,
                score=score,
                margin=margin,
                clipped=clipped,
            )
        )
    return slots


def _scan_board_slots(image: RGBImage) -> list[BlastCardSlot]:
    slots: list[BlastCardSlot] = []
    for idx in range(5):
        cx = 0.354 + 0.0734 * idx
        mat = _extract_raw_glyph(image, cx - 0.036, cx - 0.004, 0.428, 0.466)
        if mat is None:
            break
        suit = _sample_suit_in_box(image, cx - 0.019, cx + 0.019, 0.430, 0.480)
        rank, score, margin, clipped = _match_blast_rank(mat, image.height)
        slots.append(
            BlastCardSlot(
                present=True,
                suit=suit,
                rank=rank,
                score=score,
                margin=margin,
                clipped=clipped,
            )
        )
    return slots


def count_blast_visual_action_buttons(image: RGBImage) -> int:
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


def detect_blast_seats_and_dealer(image: RGBImage) -> dict[str, Any]:
    """Detect Top-Left / Top-Right seat occupancy, active cards, yellow Dealer button, and exact positions."""

    def _box_counts(rx0: float, rx1: float, ry0: float, ry1: float) -> tuple[int, int, int]:
        white = dark = yellow = 0
        for y in range(max(0, round(image.height * ry0)), min(image.height, round(image.height * ry1)), 2):
            for x in range(max(0, round(image.width * rx0)), min(image.width, round(image.width * rx1)), 2):
                r, g, b = image.get(x, y)
                if min(r, g, b) >= 195 and max(r, g, b) - min(r, g, b) <= 25:
                    white += 1
                if max(r, g, b) <= 45:
                    dark += 1
                if r >= 180 and g >= 150 and b <= 100:
                    yellow += 1
        return white, dark, yellow

    tl_cards_w, _, _ = _box_counts(0.215, 0.310, 0.220, 0.315)
    _, tl_plate_dark, tl_plate_yel = _box_counts(0.208, 0.320, 0.325, 0.375)
    tr_cards_w, _, _ = _box_counts(0.685, 0.780, 0.220, 0.315)
    _, tr_plate_dark, tr_plate_yel = _box_counts(0.678, 0.790, 0.325, 0.375)

    # Scale-independent pixel thresholds (at 1326x1025, occupied plate has ~1500 dark px and ~30+ yellow px;
    # active card backs have ~1150-1800 white px vs 0 when folded/empty).
    scale = (image.width * image.height) / (1326.0 * 1025.0)
    seat_left_occupied = tl_plate_dark >= round(400 * scale) and tl_plate_yel >= round(8 * scale)
    seat_right_occupied = tr_plate_dark >= round(400 * scale) and tr_plate_yel >= round(8 * scale)
    seat_left_active = seat_left_occupied and tl_cards_w >= round(300 * scale)
    seat_right_active = seat_right_occupied and tr_cards_w >= round(300 * scale)

    dealer_regions = {
        "left": (0.31, 0.39, 0.36, 0.44),
        "right": (0.61, 0.69, 0.36, 0.44),
        "hero": (0.38, 0.46, 0.70, 0.77),
    }
    dealer_scores: dict[str, int] = {}
    for name, (rx0, rx1, ry0, ry1) in dealer_regions.items():
        cnt = 0
        for y in range(max(0, round(image.height * ry0)), min(image.height, round(image.height * ry1)), 2):
            for x in range(max(0, round(image.width * rx0)), min(image.width, round(image.width * rx1)), 2):
                r, g, b = image.get(x, y)
                if r >= 200 and g >= 165 and b <= 85 and r - b >= 130:
                    cnt += 1
        dealer_scores[name] = cnt

    ranked_d = sorted(dealer_scores.items(), key=lambda item: item[1], reverse=True)
    dealer_seat: str | None = None
    if ranked_d[0][1] >= max(20, round(35 * scale)) and ranked_d[0][1] >= ranked_d[1][1] * 2.5:
        dealer_seat = ranked_d[0][0]

    # Clockwise order on 888 BLAST: left -> right -> hero -> left.
    hero_position: str | None = None
    seat_left_position: str | None = None
    seat_right_position: str | None = None
    if dealer_seat is not None:
        if seat_left_occupied and seat_right_occupied:
            if dealer_seat == "hero":
                hero_position, seat_left_position, seat_right_position = "BTN", "SB", "BB"
            elif dealer_seat == "left":
                seat_left_position, seat_right_position, hero_position = "BTN", "SB", "BB"
            elif dealer_seat == "right":
                seat_right_position, hero_position, seat_left_position = "BTN", "SB", "BB"
        elif seat_left_occupied or seat_right_occupied:
            # Heads-up: Dealer is BTN/SB, other seat is BB.
            if dealer_seat == "hero":
                hero_position = "BTN"
                if seat_left_occupied:
                    seat_left_position = "BB"
                if seat_right_occupied:
                    seat_right_position = "BB"
            elif dealer_seat == "left" and seat_left_occupied:
                seat_left_position, hero_position = "BTN", "BB"
            elif dealer_seat == "right" and seat_right_occupied:
                seat_right_position, hero_position = "BTN", "BB"

    active_opponents_count = int(seat_left_active) + int(seat_right_active)
    return {
        "seat_left_occupied": seat_left_occupied,
        "seat_left_active": seat_left_active,
        "seat_right_occupied": seat_right_occupied,
        "seat_right_active": seat_right_active,
        "dealer_seat": dealer_seat,
        "hero_position": hero_position,
        "seat_left_position": seat_left_position,
        "seat_right_position": seat_right_position,
        "active_opponents_count": active_opponents_count,
    }


def analyze_blast_table(image: RGBImage, game_type: str = "spin_and_go") -> dict[str, Any] | None:
    """Extract deterministic 888 BLAST cards, seats, dealer/positions, and turn indicator (<5ms)."""
    if image.width < 600 or image.height < 450:
        return None
    aspect = image.width / max(1, image.height)
    # 888 BLAST table windows are ~1.25..1.40 aspect ratio (e.g. 1326x1025 = 1.294).
    # Reject lobby windows (~1302x752 = 1.73) or uncropped 16:9 desktops (~1.78).
    if not (1.18 <= aspect <= 1.45):
        return None

    hero_slots = _scan_hero_slots(image)
    board_slots = _scan_board_slots(image)
    visual_button_count = count_blast_visual_action_buttons(image)
    seats = detect_blast_seats_and_dealer(image) if game_type == "spin_and_go" else {
        "seat_left_occupied": False,
        "seat_left_active": False,
        "seat_right_occupied": False,
        "seat_right_active": False,
        "dealer_seat": None,
        "hero_position": None,
        "seat_left_position": None,
        "seat_right_position": None,
        "active_opponents_count": None,
    }

    has_signal = (
        any(slot.present for slot in hero_slots)
        or len(board_slots) > 0
        or visual_button_count >= 2
        or seats.get("dealer_seat") is not None
        or bool(seats.get("seat_left_occupied"))
        or bool(seats.get("seat_right_occupied"))
    )
    if not has_signal:
        return None

    hero_cards = [slot.card for slot in hero_slots if slot.card is not None]
    board_cards = [slot.card for slot in board_slots if slot.card is not None]
    hero_complete = len(hero_slots) == 2 and all(slot.card is not None for slot in hero_slots) and len(set(hero_cards)) == 2
    board_complete = (
        len(board_slots) in (0, 3, 4, 5)
        and all(slot.card is not None for slot in board_slots)
        and len(set(board_cards)) == len(board_cards)
    )
    if hero_complete and board_complete and (set(hero_cards) & set(board_cards)):
        # Impossible duplicate: mark incomplete so VLM fallback investigates.
        hero_complete = False
        board_complete = False

    return {
        "game_type": game_type,
        "hero_slots": [slot.to_dict() for slot in hero_slots],
        "board_slots": [slot.to_dict() for slot in board_slots],
        "hero_cards": hero_cards,
        "hero_complete": hero_complete,
        "board": board_cards,
        "board_complete": board_complete,
        "board_slot_count": len(board_slots),
        "visual_button_count": visual_button_count,
        "hero_to_act_visual": visual_button_count >= 2,
        **seats,
    }


def _paste_rgb(canvas: RGBImage, source: RGBImage, dst_x: int, dst_y: int) -> None:
    row_bytes = source.width * 3
    canvas_stride = canvas.width * 3
    source_stride = source.width * 3
    for row in range(source.height):
        if dst_y + row >= canvas.height:
            break
        src_start = row * source_stride
        dst_start = (dst_y + row) * canvas_stride + dst_x * 3
        canvas.pixels[dst_start : dst_start + row_bytes] = source.pixels[src_start : src_start + row_bytes]


def _fill_rect(canvas: RGBImage, x0: int, y0: int, x1: int, y1: int, rgb: tuple[int, int, int]) -> None:
    r, g, b = rgb
    px = bytes((r, g, b))
    stride = canvas.width * 3
    x0 = max(0, x0)
    x1 = min(canvas.width, x1)
    y0 = max(0, y0)
    y1 = min(canvas.height, y1)
    row_data = px * (x1 - x0)
    for y in range(y0, y1):
        start = y * stride + x0 * 3
        canvas.pixels[start : start + len(row_data)] = row_data


def build_blast_numeric_collage(image: RGBImage, blast: dict[str, Any]) -> RGBImage:
    """Build a compact, clearly separated 3-band numeric/text collage for fast, zero-confusion OCR."""
    w, h = image.width, image.height

    pot_crop = crop_rgb_image(image, round(w * 0.375), round(h * 0.376), round(w * 0.255), round(h * 0.044))
    hero_crop = crop_rgb_image(image, round(w * 0.422), round(h * 0.855), round(w * 0.156), round(h * 0.076))
    btn_crop = crop_rgb_image(image, round(w * 0.592), round(h * 0.924), round(w * 0.404), round(h * 0.070))

    pot_img = resize_rgb_image(pot_crop, max_width=420, max_height=76)
    hero_img = resize_rgb_image(hero_crop, max_width=280, max_height=86)
    btn_img = resize_rgb_image(btn_crop, max_width=700, max_height=88)

    tl_img = None
    if blast.get("seat_left_occupied"):
        tl_crop = crop_rgb_image(image, round(w * 0.200), round(h * 0.318), round(w * 0.130), round(h * 0.068))
        tl_img = resize_rgb_image(tl_crop, max_width=280, max_height=84)

    tr_img = None
    if blast.get("seat_right_occupied"):
        tr_crop = crop_rgb_image(image, round(w * 0.672), round(h * 0.318), round(w * 0.130), round(h * 0.068))
        tr_img = resize_rgb_image(tr_crop, max_width=280, max_height=84)

    divider = 6
    pad = 8
    band1_h = max(pot_img.height, hero_img.height) + pad * 2
    has_mid = tl_img is not None or tr_img is not None
    band2_h = (max(tl_img.height if tl_img else 0, tr_img.height if tr_img else 0) + pad * 2) if has_mid else 0
    band3_h = btn_img.height + pad * 2

    canvas_w = max(720, pot_img.width + hero_img.width + pad * 3 + divider, btn_img.width + pad * 2)
    canvas_h = band1_h + divider + (band2_h + divider if has_mid else 0) + band3_h
    canvas = RGBImage(canvas_w, canvas_h, bytearray((16, 20, 28)) * (canvas_w * canvas_h))

    # Band 1: TOP-LEFT = Pot Total | vertical cyan divider | TOP-RIGHT = Hero nameplate & stack
    _paste_rgb(canvas, pot_img, pad, pad + (band1_h - pad * 2 - pot_img.height) // 2)
    split_x = pad + pot_img.width + pad
    _fill_rect(canvas, split_x, 0, split_x + divider, band1_h, (56, 189, 248))
    _paste_rgb(canvas, hero_img, split_x + divider + pad, pad + (band1_h - pad * 2 - hero_img.height) // 2)

    y_cur = band1_h
    _fill_rect(canvas, 0, y_cur, canvas_w, y_cur + divider, (250, 204, 21))
    y_cur += divider

    # Band 2: MIDDLE-LEFT = Seat TL | vertical divider | MIDDLE-RIGHT = Seat TR
    if has_mid:
        mid_split = canvas_w // 2
        if tl_img is not None:
            _paste_rgb(canvas, tl_img, pad, y_cur + pad)
        _fill_rect(canvas, mid_split, y_cur, mid_split + divider, y_cur + band2_h, (56, 189, 248))
        if tr_img is not None:
            _paste_rgb(canvas, tr_img, mid_split + divider + pad, y_cur + pad)
        y_cur += band2_h
        _fill_rect(canvas, 0, y_cur, canvas_w, y_cur + divider, (250, 204, 21))
        y_cur += divider

    # Band 3: BOTTOM = Action buttons
    _paste_rgb(canvas, btn_img, pad, y_cur + pad)
    return canvas


def build_blast_collage_prompt(blast: dict[str, Any]) -> str:
    """Build a tiny, focused OCR prompt for the 3-band 888 BLAST numeric collage."""
    has_left = bool(blast.get("seat_left_occupied"))
    has_right = bool(blast.get("seat_right_occupied"))
    if has_left or has_right:
        mid_desc = (
            "MIDDLE BAND (between the two yellow dividers): "
            f"left of blue divider = seat_left ({'visible nameplate+stack' if has_left else 'empty, return null'}); "
            f"right of blue divider = seat_right ({'visible nameplate+stack' if has_right else 'empty, return null'}). "
        )
    else:
        mid_desc = "There is no middle opponent band; set seat_left=null and seat_right=null. "

    return (
        "Read this 888 BLAST poker collage into ONE JSON object only:\n"
        '{"pot_label_text": "Pote total: X,Y BB", "hero_nickname": string|null, "hero_stack_text": "X,Y BB", '
        '"seat_left": {"nickname": string, "stack_text": "X,Y BB"}|null, '
        '"seat_right": {"nickname": string, "stack_text": "X,Y BB"}|null, '
        '"action_buttons": [string]}\n'
        "TOP BAND (above first yellow divider): left of blue divider = exact 'Pote total: ... BB' text for pot_label_text; "
        "right of blue divider = Hero's nickname and yellow stack text ('X,Y BB') for hero_nickname and hero_stack_text. "
        "NEVER copy pot or opponent stacks into hero_stack_text.\n"
        f"{mid_desc}\n"
        "BOTTOM BAND (below last yellow divider): copy every action button label from left to right verbatim "
        "(e.g. 'DESISTIR', 'PASSAR', 'IGUALAR 1 BB', 'APOSTA 1 BB', 'SUBIR A 2 BB', 'ALL-IN 4,2 BB').\n"
        "Return strict JSON only."
    )
