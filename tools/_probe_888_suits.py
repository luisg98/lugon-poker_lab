from pathlib import Path

from lugon_poker.vision import load_png

root = Path(r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6")
images = [
    "cap_20261117_172406_634965_7d7ae35c_full.png",
    "cap_20261117_174321_901512_b9325064_full.png",
    "cap_20261117_174523_987953_7c69484b_full.png",
]
bands = [(0.765, 0.795), (0.765, 0.805), (0.77, 0.80), (0.77, 0.81), (0.78, 0.80), (0.80, 0.88)]
for name in images:
    image = load_png(root / name)
    print(name)
    for y0, y1 in bands:
        slots = []
        for x0, x1 in ((0.44, 0.50), (0.50, 0.565)):
            counts = {"h": 0, "c": 0, "d": 0, "s": 0}
            for y in range(round(image.height * y0), round(image.height * y1), 2):
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
            slots.append(counts)
        print((y0, y1), slots)
