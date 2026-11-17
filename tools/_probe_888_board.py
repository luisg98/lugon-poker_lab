import base64
from pathlib import Path

from lugon_poker.live_vision import _call_vlm_provider, _crop_rgb_image
from lugon_poker.screen_capture import encode_png_bytes
from lugon_poker.vision import load_png

root = Path(r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6")
prompt = (
    "Read only the community cards and the exact Pote total label in this close crop of the center felt. "
    "List every large community card from left to right. "
    "There are exactly 0, 3, 4, or 5 board cards. The rank is the large white glyph at each card's upper-left; "
    "suits by face color: blue diamonds d, green clubs c, red hearts h, black spades s. "
    "Do not reorder cards or use the table logo, chips, hero hand, or other screen text. "
    "Also copy the visible Pote total label exactly. Return JSON only: "
    "{\"board_cards\":[\"8d\",\"5c\",\"Qc\"],\"pot_label_text\":\"Pote total: 2,2 BB\"}."
)
for name in [
    "cap_20261117_172353_300260_262b71fa_full.png",
    "cap_20261117_174321_901512_b9325064_full.png",
    "cap_20261117_174333_598363_3540c978_full.png",
]:
    image = load_png(root / name)
    crop = _crop_rgb_image(
        image,
        round(image.width * 0.28),
        round(image.height * 0.37),
        round(image.width * 0.68),
        round(image.height * 0.57),
    )
    image_b64 = base64.b64encode(encode_png_bytes(crop, compress_level=1)).decode("ascii")
    result = _call_vlm_provider(
        image_b64=image_b64,
        prompt=prompt,
        provider="ollama",
        model="qwen3-vl:2b-instruct",
        endpoint=None,
        api_key=None,
        timeout_seconds=8,
        num_predict=80,
        transport=None,
    )
    print(name, result)
