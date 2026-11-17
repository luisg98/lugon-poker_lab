import base64
from pathlib import Path

from lugon_poker.live_vision import _call_vlm_provider, _crop_rgb_image
from lugon_poker.screen_capture import encode_png_bytes
from lugon_poker.vision import load_png

root = Path(r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6")
prompt = (
    "Read only the exact text in the dark rounded center label. It begins 'Pote total:'. "
    "Copy the number and decimal comma exactly, then BB. Do not estimate or round. "
    "Return JSON only: {\"pot_label_text\":\"Pote total: 2,8 BB\"}."
)
for name in [
    "cap_20261117_174523_987953_7c69484b_full.png",
    "cap_20261117_172353_300260_262b71fa_full.png",
    "cap_20261117_174333_598363_3540c978_full.png",
]:
    image = load_png(root / name)
    crop = _crop_rgb_image(
        image,
        round(image.width * 0.35),
        round(image.height * 0.37),
        round(image.width * 0.65),
        round(image.height * 0.43),
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
