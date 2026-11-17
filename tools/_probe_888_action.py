import base64
from pathlib import Path

from lugon_poker.live_vision import _call_vlm_provider, _crop_rgb_image
from lugon_poker.screen_capture import encode_png_bytes
from lugon_poker.vision import load_png

root = Path(r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6")
prompt = (
    "Read only the labels inside the separate rectangular move buttons in the bottom-right action row. "
    "Exclude rounded preference pills, chips, stack labels and percentage controls. Preserve each amount exactly. "
    "Return JSON only: {\"action_buttons\":[\"DESISTIR\",\"IGUALAR 4,3 BB\"]}."
)
for name in [
    "cap_20261117_174817_193246_e85cb387_full.png",
    "cap_20261117_174645_460490_ffcfa63e_full.png",
    "cap_20261117_174842_861427_91a33921_full.png",
]:
    image = load_png(root / name)
    crop = _crop_rgb_image(
        image,
        round(image.width * 0.55),
        round(image.height * 0.90),
        round(image.width * 0.995),
        image.height,
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
