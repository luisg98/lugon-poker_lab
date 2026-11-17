from pathlib import Path

from lugon_poker.assistant import GameAssistant
from lugon_poker.live_vision import _read_888_center_detail
from lugon_poker.vision import load_png


def show_response(url, headers, payload, timeout):
    response = GameAssistant._default_http_post(url, headers, payload, timeout)
    content = response.get("message", {}).get("content", "")
    print("MODEL_RESPONSE:", content.encode("ascii", "backslashreplace").decode("ascii"))
    return response


path = Path(
    r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6\cap_20261117_172353_300260_262b71fa_full.png"
)
try:
    print(
        "PARSED:",
        _read_888_center_detail(
            load_png(path), "ollama", "qwen3-vl:2b-instruct", None, None, 8, show_response
        ),
    )
except Exception as exc:
    print("PARSE_ERROR:", repr(exc))
