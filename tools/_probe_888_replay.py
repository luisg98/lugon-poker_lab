import json
from pathlib import Path

from lugon_poker.live_vision import analyze_live_image
from lugon_poker.companion_gui import evaluate_vision_report
from lugon_poker.vision import load_png

root = Path(r"C:\Users\luisg\Desktop\lugon-poker_lab\data\captures\v1.0.6")
names = [
    "cap_20261117_172142_044651_483387d5_full.png",
    "cap_20261117_174645_460490_ffcfa63e_full.png",
    "cap_20261117_174751_413854_29f9880a_full.png",
    "cap_20261117_174817_193246_e85cb387_full.png",
]
for name in names:
    report = analyze_live_image(
        load_png(root / name), mode="vlm", game_type="spin_and_go", preset="888_blast"
    )
    status, _ = evaluate_vision_report(report)
    hero = next((player for player in report.get("players", []) if player.get("hero")), {})
    print(
        name[4:19],
        json.dumps(
            {
                "hero_to_act": report.get("hero_to_act"),
                "hero_cards": report.get("hero_cards"),
                "board": report.get("board"),
                "pot": report.get("pot"),
                "to_call": report.get("to_call"),
                "hero_stack": hero.get("stack"),
                "status": report.get("status"),
                "evaluation": status,
                "action_buttons": report.get("action_buttons"),
                "ambiguities": report.get("ambiguities"),
            },
            ensure_ascii=False,
        ),
    )
