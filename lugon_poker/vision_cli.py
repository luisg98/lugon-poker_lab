"""CLI for rendering fixtures and inspecting one calibrated table image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .equity import calculate_equity
from .live_vision import DEFAULT_VLM_TIMEOUT_SECONDS, analyze_live_image
from .replay import format_replay, replay_captures
from .screen_capture import CaptureTarget, capture_target, enable_dpi_awareness, list_windows
from .state import HandState
from .strategy import analyze_strategy
from .synthetic import render_scene, render_file
from .vision import analyze_image, analyze_table, compare_fixture, load_scene, save_png


def _expected(scene: dict) -> dict:
    hero = next(player for player in scene["players"] if player["hero"])
    active = next(player for player in scene["players"] if player["active"])
    in_hand = [player for player in scene["players"] if not player.get("folded", False)]
    highest_bet = max(float(player["bet"]) for player in in_hand)
    return {
        "hero_cards": [str(card).lower() for card in scene["hero_cards"]],
        "board": [str(card).lower() for card in scene["board"]],
        "pot": float(scene["pot"]),
        "blinds": {"small": float(scene["small_blind"]), "big": float(scene["big_blind"])},
        "active_player": active["nickname"],
        "to_call": max(0.0, highest_bet - float(hero["bet"])),
        "players": [
            {
                "nickname": player["nickname"],
                "position": player["position"],
                "stack": float(player["stack"]),
                "bet": float(player["bet"]),
                "hero": bool(player["hero"]),
                "active": bool(player["active"]),
                "folded": bool(player.get("folded", False)),
                "seat": player["seat"],
            }
            for player in scene["players"]
        ],
    }


def _attach_m1_analysis(report: dict, iterations: int, seed: int) -> None:
    if report["hand_state"] is not None and report["hero_to_act"]:
        hand_state = HandState.from_dict(report["hand_state"])
        equity = calculate_equity(hand_state, iterations=iterations, seed=seed)
        report["strategy_analysis"] = analyze_strategy(hand_state, equity).to_dict()
        report["equity"] = {
            "share": equity.equity,
            "win_rate": equity.win_rate,
            "tie_rate": equity.tie_rate,
            "loss_rate": equity.loss_rate,
            "samples": equity.samples,
            "method": equity.method,
            "seed": equity.seed,
            "attempts": equity.attempts,
        }
    else:
        report["strategy_analysis"] = None
        report["ambiguities"].append("análise M1 disponível apenas quando hero está identificado e é o jogador ativo")


def _print_vision_report(report: dict) -> None:
    src = report.get("vision_source", "calibrated")
    print(f"Imagem: {report['image']['width']}x{report['image']['height']} | fonte: {src} | mesa reconhecida: {bool(report['table_count'])}")
    print(f"Hero: {' '.join(card for card in report['hero_cards'] if card) or 'desconhecido'}; board: {' '.join(report['board']) or '—'}")
    print(f"Pote: {report['pot']}; blinds: {report['blinds']}; ativo: {report['active_player']}; a pagar: {report['to_call']}")
    for player in report["players"]:
        print(f"{player['nickname'] or '?'} ({player['position'] or '?'}) stack={player['stack']} bet={player['bet']} hero={player['hero']} ativo={player['active']} folded={player['folded']}")
    print(f"Hero a agir: {report['hero_to_act']}; ambiguidades: {len(report['ambiguities'])}")
    for ambiguity in report["ambiguities"]:
        print(f"- {ambiguity}")
    if report.get("strategy_analysis"):
        print(f"M1: {report['strategy_analysis']['decision']} — {report['strategy_analysis']['explanation']}")


def _parse_region(text: str) -> tuple[int, int, int, int]:
    parts = [int(item.strip()) for item in text.split(",")]
    if len(parts) != 4:
        raise ValueError("região deve usar formato X,Y,LARGURA,ALTURA")
    return parts[0], parts[1], parts[2], parts[3]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspeciona capturas PNG ou captura o ecrã em tempo real.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    render_parser = subparsers.add_parser("render", help="renderiza uma captura sintética de referência")
    render_parser.add_argument("--scene", type=Path, required=True)
    render_parser.add_argument("--output", type=Path, required=True)
    render_parser.add_argument("--width", type=int, default=960)
    render_parser.add_argument("--height", type=int, default=540)

    scan_parser = subparsers.add_parser("scan", help="extrai estado visível de uma imagem PNG")
    scan_parser.add_argument("image", type=Path)
    scan_parser.add_argument("--mode", choices=("calibrated", "hybrid", "vlm"), default="calibrated")
    scan_parser.add_argument("--vlm-provider", choices=("ollama", "openai_compatible", "gemini"), default=None)
    scan_parser.add_argument("--vlm-model", type=str, default=None)
    scan_parser.add_argument("--timeout", type=float, default=DEFAULT_VLM_TIMEOUT_SECONDS)
    scan_parser.add_argument("--game-type", choices=("spin_and_go", "tournament"), default="spin_and_go")
    scan_parser.add_argument("--json", action="store_true", help="imprime JSON completo")
    scan_parser.add_argument("--analyze", action="store_true", help="executa também o motor M1 se o estado for válido")
    scan_parser.add_argument("--iterations", type=int, default=10_000)
    scan_parser.add_argument("--seed", type=int, default=7)

    win_parser = subparsers.add_parser("windows", help="lista janelas abertas disponíveis para captura em tempo real")
    win_parser.add_argument("--json", action="store_true")

    cap_parser = subparsers.add_parser("capture", help="captura ecrã/região/janela em direto e analisa o estado")
    cap_parser.add_argument("--region", type=str, default=None, help="região X,Y,W,H em píxeis")
    cap_parser.add_argument("--window", type=str, default=None, help="parte do título da janela alvo")
    cap_parser.add_argument("--hwnd", type=int, default=None, help="HWND exato da janela alvo")
    cap_parser.add_argument("--mode", choices=("hybrid", "calibrated", "vlm"), default="hybrid")
    cap_parser.add_argument("--vlm-provider", choices=("ollama", "openai_compatible", "gemini"), default=None)
    cap_parser.add_argument("--vlm-model", type=str, default=None)
    cap_parser.add_argument("--timeout", type=float, default=DEFAULT_VLM_TIMEOUT_SECONDS)
    cap_parser.add_argument("--game-type", choices=("spin_and_go", "tournament"), default="spin_and_go")
    cap_parser.add_argument("--output", type=Path, default=None, help="opcional: guardar captura em PNG")
    cap_parser.add_argument("--json", action="store_true")
    cap_parser.add_argument("--analyze", action="store_true")
    cap_parser.add_argument("--iterations", type=int, default=10_000)
    cap_parser.add_argument("--seed", type=int, default=7)

    replay_parser = subparsers.add_parser(
        "replay",
        help="reanalisa capturas arquivadas (data/captures/v<versão>) com o código atual e compara com a leitura original",
    )
    replay_parser.add_argument("directory", type=Path, help="pasta de capturas, ex.: data/captures/v1.0.6")
    replay_parser.add_argument("--mask", type=str, default=None, help="X,Y,W,H a apagar (ex.: janela do Companion) se o meta não o regista")
    replay_parser.add_argument("--vlm", action="store_true", help="inclui o Vision-LLM (lento); por omissão só o leitor Mercury determinístico")
    replay_parser.add_argument("--game-type", choices=("spin_and_go", "tournament"), default=None)
    replay_parser.add_argument("--last", type=int, default=None, help="apenas as últimas N capturas")

    eval_parser = subparsers.add_parser("evaluate", help="mede correspondência em um fixture sintético conhecido")
    eval_parser.add_argument("--scene", type=Path, required=True)
    eval_parser.add_argument("--sizes", default="960x540,1280x720")

    args = parser.parse_args(argv)
    try:
        if args.command == "render":
            render_file(args.scene, args.output, args.width, args.height)
            print(f"Captura sintética gravada em {args.output}")
            return 0

        if args.command == "windows":
            enable_dpi_awareness()
            windows = list_windows()
            if args.json:
                print(json.dumps([w.to_dict() for w in windows], ensure_ascii=False, indent=2))
            else:
                for w in windows:
                    tag = " [minimizada]" if w.minimized else f" {w.width}x{w.height} @ ({w.left},{w.top})"
                    print(f"[{w.hwnd}] {w.title}{tag}")
            return 0

        if args.command == "replay":
            records = replay_captures(
                args.directory,
                mask=_parse_region(args.mask) if args.mask else None,
                use_vlm=args.vlm,
                game_type=args.game_type,
                limit=args.last,
            )
            print(format_replay(records))
            return 0

        if args.command == "capture":
            enable_dpi_awareness()
            if args.region:
                target = CaptureTarget(kind="region", region=_parse_region(args.region))
            elif args.hwnd:
                target = CaptureTarget(kind="window", hwnd=args.hwnd, title=f"HWND {args.hwnd}")
            elif args.window:
                needle = args.window.lower()
                matches = [w for w in list_windows() if needle in w.title.lower() and not w.minimized]
                if not matches:
                    raise RuntimeError(f"Nenhuma janela visível corresponde a {args.window!r}")
                target = CaptureTarget(kind="window", hwnd=matches[0].hwnd, title=matches[0].title)
            else:
                target = CaptureTarget(kind="fullscreen", title="Ecrã Inteiro")

            image, cap_meta = capture_target(target)
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                save_png(image, args.output)
            report = analyze_live_image(
                image,
                mode=args.mode,
                vlm_provider=args.vlm_provider,
                vlm_model=args.vlm_model,
                timeout_seconds=args.timeout,
                game_type=args.game_type,
            )
            report["capture_info"] = cap_meta
            if args.analyze:
                _attach_m1_analysis(report, args.iterations, args.seed)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                print(f"Captura: {cap_meta['title']} ({cap_meta['width']}x{cap_meta['height']}, {cap_meta['capture_ms']}ms)")
                _print_vision_report(report)
            return 0 if report["hand_state"] is not None else 1

        if args.command == "scan":
            if args.mode == "calibrated":
                report = analyze_table(args.image)
            else:
                from .vision import load_png

                report = analyze_live_image(
                    load_png(args.image),
                    mode=args.mode,
                    vlm_provider=args.vlm_provider,
                    vlm_model=args.vlm_model,
                    timeout_seconds=args.timeout,
                    game_type=args.game_type,
                )
            if args.analyze:
                _attach_m1_analysis(report, args.iterations, args.seed)
            if args.json:
                print(json.dumps(report, ensure_ascii=False, indent=2))
            else:
                _print_vision_report(report)
            return 0 if report["hand_state"] is not None else 1

        scene = load_scene(args.scene)
        expected = _expected(scene)
        results = []
        for size in args.sizes.split(","):
            width_text, height_text = size.lower().split("x", 1)
            width, height = int(width_text), int(height_text)
            actual = analyze_image(render_scene(scene, width, height))
            comparison = compare_fixture(expected, actual)
            results.append({"resolution": size, **comparison, "ambiguities": actual["ambiguities"]})
        report = {"scope": "synthetic fixture only; not real-capture accuracy", "results": results}
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if all(item["fields_correct"] == item["fields_total"] and not item["ambiguities"] for item in results) else 1
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
