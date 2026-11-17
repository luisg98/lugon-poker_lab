import base64
from pathlib import Path
from lugon_poker.live_vision import _call_vlm_provider, _crop_rgb_image
from lugon_poker.screen_capture import encode_png_bytes
from lugon_poker.vision import load_png

root = Path(r'C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6')
prompt = ('This crop shows ONLY the two face-up hole cards directly above the bottom player nameplate. '
          'Read exactly those two cards, left to right; do not read board cards, chips, badges, stack or action text. '
          'Four-color deck: blue=d, green=c, red=h, black=s. Rank is the large white glyph at the upper-left of each card. '
          'Return only JSON: {"hero_cards":["8s","5d"]}.')
for name in ['cap_20261117_174321_901512_b9325064_full.png',
             'cap_20261117_174523_987953_7c69484b_full.png',
             'cap_20261117_172406_634965_7d7ae35c_full.png']:
    image = load_png(root / name)
    crop = _crop_rgb_image(image, round(image.width*.40), round(image.height*.73), round(image.width*.60), round(image.height*.90))
    image_b64 = base64.b64encode(encode_png_bytes(crop, compress_level=1)).decode('ascii')
    result = _call_vlm_provider(image_b64=image_b64, prompt=prompt, provider='ollama', model='qwen3-vl:2b-instruct', endpoint=None, api_key=None, timeout_seconds=8, num_predict=60, transport=None)
    print(name, result)
