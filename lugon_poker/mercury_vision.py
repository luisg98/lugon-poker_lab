"""Deterministic PokerStars Mercury 4-color table locator and visual extractor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .screen_capture import crop_rgb_image, resize_rgb_image
from .vision import RGBImage

COLLAGE_WIDTH = 896
COLLAGE_HEIGHT = 240

RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
TEMPLATE_WIDTH = 12
TEMPLATE_HEIGHT = 16

# Canonical 12x16 normalized rank templates captured from PokerStars Mercury 4-color deck.
_MERCURY_RANK_PATTERNS: dict[str, str] = {
    "2": "..########..|.##########.|############|####.....###|###......###|###......###|.........###|.......#####|...#########|.#########..|######......|####........|###.........|############|############|############",
    "3": "..########..|.##########.|############|###......###|###......###|.##......###|........###.|.....#####..|.....######.|.........###|###......###|###......###|###......###|############|.###########|..########..",
    "4": "......####..|.....#####..|.....#####..|....###.##..|...###..##..|...###..##..|..###...##..|.###....##..|.###....##..|###.....##..|############|############|############|.......###..|.......###..|.......###..",
    "5": "###########.|###########.|###########.|####........|####........|##########..|###########.|######.#####|.##......###|.........###|.........###|###......###|####.....###|############|.##########.|..########..",
    "6": "..########..|.##########.|.###########|###......###|###.......#.|###.........|###..####...|###########.|############|###......###|###......###|###......###|###......###|############|.##########.|..########..",
    "7": "############|############|############|.........###|........###.|........###.|.......###..|......####..|......###...|......###...|.....###....|.....###....|....###.....|...####.....|...###......|...###......",
    "8": "..########..|.##########.|############|###......###|###......###|###......###|.###....###.|..########..|..#########.|.###.....###|###......###|###......###|###......###|############|.###########|..########..",
    "9": "..########..|.##########.|############|###......###|###......###|###......###|###......###|############|.###########|...####..###|.........###|###......###|###......###|############|.##########.|..########..",
    "T": "##...######.|##..########|##..########|##.###....##|##.###....##|##.###....##|##.###....##|##.###....##|##.###....##|##.###....##|##.###....##|##.###....##|##.###....##|##..########|##..########|##...######.",
    "J": ".........###|.........###|.........###|.........###|.........###|.........###|.........###|.........###|.........###|.........###|####.....###|####.....###|####.....###|############|.###########|..########..",
    "Q": "..#######...|.#########..|.##########.|###.....###.|###.....###.|###......##.|##.......##.|##.......##.|##.......##.|##...##..##.|###..######.|###...#####.|###....####.|.###########|.###########|..######....",
    "K": "###.....###.|###....####.|###....###..|###...###...|###..###....|###..###....|#######.....|######......|######......|###.###.....|###..###....|###..####...|###...####..|###....####.|###.....###.|###.....####",
    "A": "....####....|....####....|....####....|....##.##...|...###.##...|...###.##...|...##..###..|...##..###..|..###..###..|..###...##..|..###...###.|..#########.|.##########.|.##......###|.##......###|###.......##",
}


def _decode_pattern(pattern: str) -> tuple[int, ...]:
    return tuple(1 if ch == "#" else 0 for ch in pattern.replace("|", ""))


_MERCURY_RANK_TEMPLATES: dict[str, tuple[int, ...]] = {
    rank: _decode_pattern(pattern) for rank, pattern in _MERCURY_RANK_PATTERNS.items()
}

# Secondary scaled variants observed at smaller PokerStars client widths (e.g. 1317x917).
_MERCURY_RANK_EXTRA_PATTERNS: dict[str, tuple[str, ...]] = {
    "8": (
        "..#########.|.###########|############|###......###|###......###|###......###|####....####|.##########.|.##########.|############|###.......##|###.......##|###......###|####.....###|############|.##########.",
    ),
    # Real 1323x917 client captures (v0.2.0 diagnostics): the canonical '9' scored 0.92-0.93 against the
    # runner-up '8' (0.90), a margin of ~0.03 below the 0.04 acceptance threshold, so every 9 was rejected.
    "9": (
        "..########..|.##########.|############|###......###|###.......##|###.......##|###......###|############|.###########|..######.###|.........###|###......###|###......###|###......###|############|.##########.",
    ),
}

_MERCURY_RANK_VARIANTS: dict[str, tuple[tuple[int, ...], ...]] = {
    rank: (
        _MERCURY_RANK_TEMPLATES[rank],
        *(_decode_pattern(pat) for pat in _MERCURY_RANK_EXTRA_PATTERNS.get(rank, ())),
    )
    for rank in _MERCURY_RANK_PATTERNS
}


def _compute_template_weights(templates: dict[str, tuple[int, ...]]) -> tuple[tuple[float, ...], float]:
    n_ranks = len(templates)
    weights: list[float] = []
    for idx in range(TEMPLATE_WIDTH * TEMPLATE_HEIGHT):
        p = sum(tpl[idx] for tpl in templates.values()) / n_ranks
        weights.append(0.25 + 3.0 * (p * (1.0 - p)))
    return tuple(weights), sum(weights)


_RANK_CELL_WEIGHTS, _RANK_WEIGHT_SUM = _compute_template_weights(_MERCURY_RANK_TEMPLATES)

# Normalized Mercury slot bounds (relative to client table rect).
_BOARD_SLOT_RANGES: tuple[tuple[float, float], ...] = (
    (0.3350, 0.3972),
    (0.4021, 0.4643),
    (0.4700, 0.5322),
    (0.5378, 0.6000),
    (0.6049, 0.6678),
)
_HERO_SLOT_RANGES: tuple[tuple[float, float], ...] = (
    (0.4382, 0.5004),
    (0.5025, 0.5647),
)


@dataclass(frozen=True)
class CardSlotResult:
    present: bool
    suit: str | None
    rank: str | None
    score: float
    margin: float

    @property
    def card(self) -> str | None:
        if self.present and self.rank and self.suit:
            return f"{self.rank}{self.suit}"
        return None


def classify_mercury_suit_rgb(r: int, g: int, b: int) -> str | None:
    """Classify a PokerStars Mercury 4-color card background RGB pixel into 'h', 'd', 'c', 's', or None."""
    if r > g + 35 and r > b + 35:
        return "h"  # red card body
    if b > r + 25 and b >= g + 8:
        return "d"  # blue card body
    if g > r + 10 and g > b + 10:
        return "c"  # green card body
    if max(r, g, b) - min(r, g, b) <= 22 and 15 <= max(r, g, b) <= 135:
        return "s"  # dark grey / charcoal card body
    return None


def _is_light_grey(r: int, g: int, b: int, min_val: int = 225, max_spread: int = 16) -> bool:
    return min(r, g, b) >= min_val and (max(r, g, b) - min(r, g, b)) <= max_spread


def _is_red_ui(r: int, g: int, b: int) -> bool:
    return r > 120 and r > g + 45 and r > b + 45


def _looks_like_mercury_felt_and_bg(image: RGBImage, tx: int, ty: int, tw: int, th: int) -> bool:
    if tw < 500 or th < 350:
        return False

    # Check dark neutral charcoal oval felt on left and right sides away from cards/pot.
    felt_samples = []
    for rel_x in (0.21, 0.25, 0.75, 0.79):
        for rel_y in (0.37, 0.44):
            sx = min(image.width - 1, max(0, tx + round(rel_x * tw)))
            sy = min(image.height - 1, max(0, ty + round(rel_y * th)))
            felt_samples.append(image.get(sx, sy))

    dark_felt = sum(
        1 for r, g, b in felt_samples if max(r, g, b) <= 45 and (max(r, g, b) - min(r, g, b)) <= 12
    )
    if dark_felt < 6:
        return False

    # Check outer Mercury table radial background on the left margin (blue €1, green €5, gold/purple, or slate).
    bg_samples = []
    for rel_x in (0.04, 0.08):
        for rel_y in (0.45, 0.55, 0.65):
            sx = min(image.width - 1, max(0, tx + round(rel_x * tw)))
            sy = min(image.height - 1, max(0, ty + round(rel_y * th)))
            bg_samples.append(image.get(sx, sy))

    table_bg = sum(1 for r, g, b in bg_samples if 35 <= max(r, g, b) <= 195)
    return table_bg >= 4


def _same_color(a: tuple[int, int, int], b: tuple[int, int, int], tol: int = 12) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol and abs(a[2] - b[2]) <= tol


def _find_titlebar_spans(image: RGBImage) -> list[tuple[int, int, int, tuple[int, int, int]]]:
    """Collect candidate horizontal Windows titlebars (sorted widest-first) across the upper screen."""
    w, h = image.width, image.height
    candidates: list[tuple[int, int, int, tuple[int, int, int]]] = []
    seen_buckets: set[tuple[int, int]] = set()
    for y in range(2, min(140, h)):
        run_start: int | None = None
        ref_rgb: tuple[int, int, int] = (0, 0, 0)
        for x in range(w):
            rgb = image.get(x, y)
            if run_start is None:
                if max(rgb) >= 75:
                    run_start = x
                    ref_rgb = rgb
            elif _same_color(rgb, ref_rgb, tol=6):
                continue
            else:
                if x - run_start >= 420:
                    bucket = (run_start // 24, x // 24)
                    if bucket not in seen_buckets:
                        seen_buckets.add(bucket)
                        candidates.append((run_start, x, y, ref_rgb))
                if max(rgb) >= 75:
                    run_start = x
                    ref_rgb = rgb
                else:
                    run_start = None
        if run_start is not None and (w - run_start) >= 420:
            bucket = (run_start // 24, w // 24)
            if bucket not in seen_buckets:
                seen_buckets.add(bucket)
                candidates.append((run_start, w, y, ref_rgb))
    candidates.sort(key=lambda item: item[1] - item[0], reverse=True)
    return candidates


def detect_mercury_table_rect(image: RGBImage) -> tuple[int, int, int, int] | None:
    """Locate the PokerStars Mercury client rect (x, y, w, h) inside a full or cropped image."""
    w, h = image.width, image.height
    if w < 500 or h < 350:
        return None

    # 1. Try each detected Windows titlebar candidate (widest first) so side Companion/IDE bars never shadow the table.
    for tb_x0, tb_x1, tb_y, tb_rgb in _find_titlebar_spans(image):
        client_w = max(1, tb_x1 - tb_x0)
        tb_probes = [min(w - 1, tb_x0 + round(client_w * frac)) for frac in (0.35, 0.55)]
        client_y0 = tb_y + 1
        for y in range(tb_y, min(h, tb_y + 55)):
            if all(not _same_color(image.get(px, y), tb_rgb, tol=28) for px in tb_probes):
                client_y0 = y
                break

        # Find bottom window edge before another window, active accent border, or Windows taskbar.
        client_y1 = h
        if h >= 700 and w > client_w + 80:
            probe_xs = [
                min(w - 1, tb_x0 + max(80, round(client_w * offset)))
                for offset in (0.15, 0.35, 0.55)
            ]
            y_scan_start = min(h - 1, max(client_y0 + 300, client_y0 + round(client_w * 0.675)))
            y_scan_end = min(h, client_y0 + round(client_w * 0.725))
            for y in range(y_scan_start, y_scan_end):
                accent_hits = 0
                grey_hits = 0
                for px in probe_xs:
                    r, g, b = image.get(px, y)
                    if _same_color((r, g, b), tb_rgb, tol=20) and not _is_light_grey(r, g, b):
                        accent_hits += 1
                    elif 44 <= max(r, g, b) <= 95 and abs(r - g) <= 8 and abs(g - b) <= 8:
                        grey_hits += 1
                if accent_hits >= 2 or grey_hits >= 2:
                    client_y1 = y - 1
                    break
            if client_y1 == h:
                client_y1 = min(h, client_y0 + round(client_w * 0.709))

        tw = max(1, client_w)
        th = max(1, client_y1 - client_y0)
        if _looks_like_mercury_felt_and_bg(image, tb_x0, client_y0, tw, th):
            return (tb_x0, client_y0, tw, th)

    # 2. Fallback: image is already a window client capture or tight table capture.
    if _looks_like_mercury_felt_and_bg(image, 0, 0, w, h):
        return (0, 0, w, h)
    return None


def prepare_mercury_vlm_crop(image: RGBImage, table_rect: tuple[int, int, int, int]) -> RGBImage:
    """Crop tightly around the PokerStars Mercury table (removing banner, titlebar, and desktop clutter).

    The Spin & Go prize banner ("€5" next to spade icons) spans ~5%-12.5% of the table height and was being read by
    the VLM as five spade board cards, so the crop must start below it (opponent hole cards begin at ~15%).
    """
    tx, ty, tw, th = table_rect
    top_trim = round(th * 0.135)
    bottom_trim = round(th * 0.996)
    crop_x = max(0, tx)
    crop_y = max(0, ty + top_trim)
    crop_w = min(image.width - crop_x, tw)
    crop_h = min(image.height - crop_y, max(1, bottom_trim - top_trim))
    if crop_w <= 0 or crop_h <= 0:
        return image
    return crop_rgb_image(image, crop_x, crop_y, crop_w, crop_h)


def _blit_rgb(dst: RGBImage, src: RGBImage, dx: int, dy: int) -> None:
    copy_w = min(src.width, dst.width - dx)
    copy_h = min(src.height, dst.height - dy)
    if copy_w <= 0 or copy_h <= 0 or dx < 0 or dy < 0:
        return
    row_bytes = copy_w * 3
    src_stride = src.width * 3
    dst_stride = dst.width * 3
    for row in range(copy_h):
        s0 = row * src_stride
        d0 = (dy + row) * dst_stride + dx * 3
        dst.pixels[d0 : d0 + row_bytes] = src.pixels[s0 : s0 + row_bytes]


def build_mercury_numeric_collage(image: RGBImage, mercury: dict[str, Any]) -> RGBImage:
    """Build a crisp fixed-size (896x240) numeric/seat collage for fast VLM OCR (<1.5s)."""
    tx, ty, tw, th = mercury["table_rect"]
    canvas = RGBImage(COLLAGE_WIDTH, COLLAGE_HEIGHT, bytearray((18, 22, 28) * COLLAGE_WIDTH * COLLAGE_HEIGHT))

    # Top row: opponents seat plates + center pot text ("Pote: X BB").
    top = crop_rgb_image(image, tx, ty + round(0.155 * th), tw, max(1, round(0.165 * th)))
    top_resized = resize_rgb_image(top, COLLAGE_WIDTH, 144)
    _blit_rgb(canvas, top_resized, (COLLAGE_WIDTH - top_resized.width) // 2, 0)

    # Clear horizontal divider separating opponent seats/pot from Hero & action button panels.
    divider = bytes((72, 92, 118)) * COLLAGE_WIDTH
    for dy in (146, 147):
        start = dy * COLLAGE_WIDTH * 3
        canvas.pixels[start : start + COLLAGE_WIDTH * 3] = divider

    # Bottom-left: Hero stack box only (clean vertical padding around 'Luisfg98' + BB stack).
    hero_box = crop_rgb_image(
        image,
        tx + round(0.340 * tw),
        ty + round(0.665 * th),
        max(1, round(0.245 * tw)),
        max(1, round(0.095 * th)),
    )
    hero_resized = resize_rgb_image(hero_box, 340, 86)
    _blit_rgb(canvas, hero_resized, 80, 152)

    # Bottom-right: Middle call button ("Pago X BB") ONLY when Hero faces an active bet.
    if mercury.get("hero_to_act") and not mercury.get("check_available"):
        btn_box = crop_rgb_image(
            image,
            tx + round(0.668 * tw),
            ty + round(0.912 * th),
            max(1, round(0.156 * tw)),
            max(1, round(0.080 * th)),
        )
        btn_resized = resize_rgb_image(btn_box, 260, 86)
        _blit_rgb(canvas, btn_resized, 540, 152)

    return canvas


def _extract_rank_from_slot(
    image: RGBImage,
    x0: int,
    x1: int,
    y0: int,
    y1: int,
    white_thr: int = 205,
) -> tuple[str | None, float, float]:
    pts: list[tuple[int, int]] = []
    # Restrict search horizontally to the left half of the card where the rank sits.
    gx_limit = min(x1, x0 + max(24, round((x1 - x0) * 0.58)))
    for y in range(max(0, y0), min(image.height, y1)):
        for x in range(max(0, x0 + 2), gx_limit):
            r, g, b = image.get(x, y)
            if min(r, g, b) >= white_thr and (max(r, g, b) - min(r, g, b)) <= 32:
                pts.append((x, y))

    if len(pts) < 35:
        return (None, 0.0, 0.0)

    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    rx0, rx1 = min(xs), max(xs)
    ry0, ry1 = min(ys), max(ys)
    gw = rx1 - rx0 + 1
    gh = ry1 - ry0 + 1
    if gw < 8 or gh < 18:
        return (None, 0.0, 0.0)

    pset = set(pts)
    bits: list[int] = []
    for dy in range(TEMPLATE_HEIGHT):
        sy0 = ry0 + (dy * gh) // TEMPLATE_HEIGHT
        sy1 = max(sy0 + 1, ry0 + ((dy + 1) * gh) // TEMPLATE_HEIGHT)
        for dx in range(TEMPLATE_WIDTH):
            sx0 = rx0 + (dx * gw) // TEMPLATE_WIDTH
            sx1 = max(sx0 + 1, rx0 + ((dx + 1) * gw) // TEMPLATE_WIDTH)
            hits = sum(1 for yy in range(sy0, sy1) for xx in range(sx0, sx1) if (xx, yy) in pset)
            total = (sy1 - sy0) * (sx1 - sx0)
            bits.append(1 if hits * 2 >= total else 0)

    scores: list[tuple[float, str]] = []
    for rank, variants in _MERCURY_RANK_VARIANTS.items():
        best_variant = max(
            sum(weight for a, b, weight in zip(bits, tpl, _RANK_CELL_WEIGHTS) if a == b)
            / _RANK_WEIGHT_SUM
            for tpl in variants
        )
        scores.append((best_variant, rank))
    scores.sort(reverse=True)
    best_score, best_rank = scores[0]
    second_score = scores[1][0]
    margin = best_score - second_score
    if best_score < 0.85 or margin < 0.04:
        return (None, best_score, margin)
    return (best_rank, best_score, margin)


def _sample_card_suit(image: RGBImage, sx0: int, sx1: int, sample_y0: int, sample_y1: int) -> str | None:
    """Sample solid card body in the upper-right area away from the white rank glyph."""
    rx0 = sx0 + max(10, round((sx1 - sx0) * 0.56))
    rx1 = max(rx0 + 4, sx1 - max(4, round((sx1 - sx0) * 0.10)))
    rs = gs = bs = count = 0
    step_x = max(2, (rx1 - rx0) // 6)
    step_y = max(2, (sample_y1 - sample_y0) // 5)
    for y in range(max(0, sample_y0), min(image.height, sample_y1), step_y):
        for x in range(max(0, rx0), min(image.width, rx1), step_x):
            r, g, b = image.get(x, y)
            if max(r, g, b) >= 58:
                rs += r
                gs += g
                bs += b
                count += 1
    if count < 4:
        return None
    return classify_mercury_suit_rgb(round(rs / count), round(gs / count), round(bs / count))


def _find_card_vertical_bounds(
    image: RGBImage,
    sx0: int,
    sx1: int,
    scan_y0: int,
    scan_y1: int,
    min_run_h: int,
) -> tuple[int, int] | None:
    """Locate the contiguous colored card body rows [card_y0, card_y1] inside a horizontal slot."""
    sw = sx1 - sx0
    x_probes = (
        min(image.width - 1, max(0, sx0 + max(6, round(sw * 0.60)))),
        min(image.width - 1, max(0, sx0 + max(8, round(sw * 0.74)))),
        min(image.width - 1, max(0, sx0 + max(10, round(sw * 0.85)))),
    )
    best_run: tuple[int, int] | None = None
    cur_start: int | None = None
    for y in range(max(0, scan_y0), min(image.height, scan_y1)):
        solid_hits = 0
        for px in x_probes:
            r, g, b = image.get(px, y)
            if max(r, g, b) >= 58 and classify_mercury_suit_rgb(r, g, b) is not None:
                solid_hits += 1
        if solid_hits >= 2:
            if cur_start is None:
                cur_start = y
        elif cur_start is not None:
            if best_run is None or (y - cur_start) > (best_run[1] - best_run[0]):
                best_run = (cur_start, y - 1)
            cur_start = None
    if cur_start is not None:
        end_y = min(image.height, scan_y1) - 1
        if best_run is None or (end_y - cur_start + 1) > (best_run[1] - best_run[0]):
            best_run = (cur_start, end_y)
    if best_run is None or (best_run[1] - best_run[0] + 1) < min_run_h:
        return None
    return best_run


def _scan_card_slots(
    image: RGBImage,
    table_rect: tuple[int, int, int, int],
    x_ranges: tuple[tuple[float, float], ...],
    scan_y_range: tuple[float, float],
    stop_on_missing: bool,
) -> list[CardSlotResult]:
    tx, ty, tw, th = table_rect
    scan_y0 = ty + round(scan_y_range[0] * th)
    scan_y1 = ty + round(scan_y_range[1] * th)
    min_run_h = max(20, round(0.032 * th))
    suit_top_off = max(6, round(0.008 * th))
    suit_bot_off = max(22, round(0.030 * th))
    rank_max_h = max(42, round(0.052 * th))

    results: list[CardSlotResult] = []
    for u0, u1 in x_ranges:
        sx0 = tx + round(u0 * tw)
        sx1 = tx + round(u1 * tw)
        bounds = _find_card_vertical_bounds(image, sx0, sx1, scan_y0, scan_y1, min_run_h=min_run_h)
        if bounds is None:
            if stop_on_missing:
                break
            results.append(CardSlotResult(False, None, None, 0.0, 0.0))
            continue
        cy0, cy1 = bounds
        suit = _sample_card_suit(image, sx0, sx1, cy0 + suit_top_off, min(cy1 + 1, cy0 + suit_bot_off))
        if suit is None:
            if stop_on_missing:
                break
            results.append(CardSlotResult(False, None, None, 0.0, 0.0))
            continue
        gy0 = cy0 + 3
        gy1 = min(cy1 - 2, cy0 + rank_max_h) if (cy1 - cy0) <= round(0.075 * th) else (cy0 + rank_max_h)
        rank, score, margin = _extract_rank_from_slot(image, sx0, sx1, gy0, gy1)
        results.append(CardSlotResult(True, suit, rank, round(score, 3), round(margin, 3)))
    return results


def _detect_action_buttons(
    image: RGBImage, table_rect: tuple[int, int, int, int]
) -> dict[str, bool]:
    tx, ty, tw, th = table_rect
    scan_ys = (
        min(image.height - 1, max(0, ty + round(0.925 * th))),
        min(image.height - 1, max(0, ty + round(0.948 * th))),
    )

    def _slot_is_red(u0: float, u1: float) -> bool:
        x0 = max(0, tx + round(u0 * tw))
        x1 = min(image.width, tx + round(u1 * tw))
        if x1 <= x0:
            return False
        hits = 0
        total = 0
        step = max(4, (x1 - x0) // 16)
        for y in scan_ys:
            for x in range(x0, x1, step):
                total += 1
                if _is_red_ui(*image.get(x, y)):
                    hits += 1
        return total > 0 and (hits / total) >= 0.45

    has_left = _slot_is_red(0.515, 0.635)
    has_mid = _slot_is_red(0.680, 0.805)
    has_right = _slot_is_red(0.845, 0.945)
    hero_to_act = has_mid or has_right
    check_available = has_mid and not has_left and has_right
    return {
        "has_fold_btn": has_left,
        "has_mid_btn": has_mid,
        "has_right_btn": has_right,
        "hero_to_act": hero_to_act,
        "check_available": check_available,
    }


def _count_red_pixels(image: RGBImage, x0: int, y0: int, x1: int, y1: int, step: int = 2) -> int:
    hits = 0
    for y in range(max(0, y0), min(image.height, y1), step):
        for x in range(max(0, x0), min(image.width, x1), step):
            if _is_red_ui(*image.get(x, y)):
                hits += 1
    return hits


def _count_white_pixels(image: RGBImage, x0: int, y0: int, x1: int, y1: int, step: int = 2) -> int:
    hits = 0
    for y in range(max(0, y0), min(image.height, y1), step):
        for x in range(max(0, x0), min(image.width, x1), step):
            r, g, b = image.get(x, y)
            if min(r, g, b) >= 180:
                hits += 1
    return hits


def _detect_seats_and_dealer(
    image: RGBImage, table_rect: tuple[int, int, int, int]
) -> dict[str, Any]:
    tx, ty, tw, th = table_rect
    sl_cards = _count_red_pixels(
        image,
        tx + round(0.095 * tw),
        ty + round(0.115 * th),
        tx + round(0.195 * tw),
        ty + round(0.175 * th),
    )
    sr_cards = _count_red_pixels(
        image,
        tx + round(0.760 * tw),
        ty + round(0.115 * th),
        tx + round(0.875 * tw),
        ty + round(0.175 * th),
    )
    sl_plate = _count_white_pixels(
        image,
        tx + round(0.056 * tw),
        ty + round(0.169 * th),
        tx + round(0.156 * tw),
        ty + round(0.249 * th),
    )
    sr_plate = _count_white_pixels(
        image,
        tx + round(0.805 * tw),
        ty + round(0.169 * th),
        tx + round(0.920 * tw),
        ty + round(0.249 * th),
    )
    seat_l_active = sl_cards >= 25
    seat_r_active = sr_cards >= 25
    seat_l_seated = seat_l_active or sl_plate >= 20
    seat_r_seated = seat_r_active or sr_plate >= 20
    active_opponents = int(seat_l_active) + int(seat_r_active)

    btn_l = _count_red_pixels(
        image,
        tx + round(0.184 * tw),
        ty + round(0.299 * th),
        tx + round(0.226 * tw),
        ty + round(0.354 * th),
    )
    btn_r = _count_red_pixels(
        image,
        tx + round(0.774 * tw),
        ty + round(0.299 * th),
        tx + round(0.816 * tw),
        ty + round(0.354 * th),
    )
    btn_h = _count_red_pixels(
        image,
        tx + round(0.353 * tw),
        ty + round(0.513 * th),
        tx + round(0.410 * tw),
        ty + round(0.583 * th),
    )

    dealer_seat: str | None = None
    hero_position: str | None = None
    if btn_l >= 35 and btn_r < 20:
        dealer_seat = "left"
        hero_position = "BB"
    elif btn_r >= 35 and btn_l < 20:
        dealer_seat = "right"
        hero_position = "SB" if seat_l_seated else "BB"
    elif btn_l < 15 and btn_r < 15 and (btn_h >= 3 or active_opponents >= 1):
        dealer_seat = "hero"
        hero_position = "BTN"

    return {
        "seat_left_active": seat_l_active,
        "seat_right_active": seat_r_active,
        "seat_left_seated": seat_l_seated,
        "seat_right_seated": seat_r_seated,
        "active_opponents_count": active_opponents,
        "dealer_seat": dealer_seat,
        "hero_position": hero_position,
    }


def analyze_mercury_table(image: RGBImage, game_type: str = "spin_and_go") -> dict[str, Any] | None:
    """Extract deterministic Mercury 4-color card/action/seat facts if the image matches PokerStars Mercury."""
    rect = detect_mercury_table_rect(image)
    if rect is None:
        return None

    hero_slots = _scan_card_slots(
        image=image,
        table_rect=rect,
        x_ranges=_HERO_SLOT_RANGES,
        scan_y_range=(0.57, 0.70),
        stop_on_missing=False,
    )
    board_slots = _scan_card_slots(
        image=image,
        table_rect=rect,
        x_ranges=_BOARD_SLOT_RANGES,
        scan_y_range=(0.29, 0.48),
        stop_on_missing=True,
    )
    buttons = _detect_action_buttons(image, rect)
    if game_type == "spin_and_go":
        seats: dict[str, Any] = _detect_seats_and_dealer(image, rect)
    else:
        # MTT tables (6-max / 9-max) arrange opponents around the whole table; let VLM extract seats/position.
        seats = {
            "seat_left_active": False,
            "seat_right_active": False,
            "seat_left_seated": False,
            "seat_right_seated": False,
            "active_opponents_count": None,
            "dealer_seat": None,
            "hero_position": None,
        }

    # Guard against accidental false positives if neither cards, buttons, nor active opponents exist.
    has_signal = (
        any(slot.present for slot in hero_slots)
        or len(board_slots) > 0
        or buttons["hero_to_act"]
        or bool(seats.get("active_opponents_count"))
    )
    if not has_signal:
        return None

    hero_cards = [s.card for s in hero_slots if s.card is not None]
    board_cards = [s.card for s in board_slots if s.card is not None]
    hero_complete = len(hero_slots) == 2 and all(s.card is not None for s in hero_slots)
    board_complete = len(board_slots) in (0, 3, 4, 5) and all(s.card is not None for s in board_slots)

    return {
        "table_rect": rect,
        "game_type": game_type,
        "vlm_crop": prepare_mercury_vlm_crop(image, rect),
        "hero_slots": [
            {"present": s.present, "suit": s.suit, "rank": s.rank, "card": s.card, "score": s.score, "margin": s.margin}
            for s in hero_slots
        ],
        "board_slots": [
            {"present": s.present, "suit": s.suit, "rank": s.rank, "card": s.card, "score": s.score, "margin": s.margin}
            for s in board_slots
        ],
        "hero_cards": hero_cards,
        "hero_complete": hero_complete,
        "board": board_cards,
        "board_complete": board_complete,
        "board_slot_count": len(board_slots),
        **buttons,
        **seats,
    }
