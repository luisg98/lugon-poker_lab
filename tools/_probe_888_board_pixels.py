from pathlib import Path

from lugon_poker.vision import load_png

root = Path(r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6")
for name in [
    "cap_20261117_172011_864679_a9126b6f_full.png",  # no board
    "cap_20261117_174317_347940_2b553b14_full.png",  # no board, other table skin
    "cap_20261117_172353_300260_262b71fa_full.png",  # flop
    "cap_20261117_174407_237507_ec38da42_full.png",  # four-card board
    "cap_20261117_172219_316145_88df8b6e_full.png",  # river
]:
    image = load_png(root / name)
    print(name)
    for i in range(5):
        cx = 0.354 + 0.073 * i
        whites = 0
        for y in range(round(image.height * 0.43), round(image.height * 0.47), 2):
            for x in range(round(image.width * (cx - 0.031)), round(image.width * (cx - 0.012)), 2):
                red, green, blue = image.get(x, y)
                if min(red, green, blue) >= 205 and max(red, green, blue) - min(red, green, blue) <= 40:
                    whites += 1
        rows = []
        for y0, y1 in ((0.42, 0.45), (0.45, 0.48), (0.48, 0.51), (0.51, 0.54)):
            counts = {"h": 0, "c": 0, "d": 0, "s": 0}
            for y in range(round(image.height * y0), round(image.height * y1), 2):
                for x in range(round(image.width * (cx - 0.019)), round(image.width * (cx + 0.019)), 2):
                    red, green, blue = image.get(x, y)
                    if red > 85 and red > green * 1.35 and red > blue * 1.2:
                        counts["h"] += 1
                    elif green > 85 and green > red * 1.25 and green > blue * 1.1:
                        counts["c"] += 1
                    elif blue > 85 and blue > red * 1.2 and blue > green * 1.15:
                        counts["d"] += 1
                    elif max(red, green, blue) < 75:
                        counts["s"] += 1
            total = sum(counts.values())
            best = max(counts, key=counts.get) if total else None
            rows.append((best, counts.get(best, 0) / total if total else 0))
        print(i, "white_rank", whites, rows)
