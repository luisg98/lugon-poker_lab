"""CLI for Milestone 3: importing hand histories, querying player profiles, and diffing captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .history import diff_table_captures
from .persistence import ProfileStore, load_history_file
from .profiles import PlayerProfile, RateEstimate
from .vision import analyze_table


def _fmt_rate(est: RateEstimate) -> str:
    if est.opportunities == 0 or est.empirical_rate is None:
        return f"n/d (0/{est.opportunities}, {est.sample_adequacy})"
    return (
        f"{est.empirical_rate:.1%} ({est.successes}/{est.opportunities}, "
        f"IC95% Wilson [{est.wilson_95_low:.1%}, {est.wilson_95_high:.1%}], "
        f"Bayes {est.bayesian_prior}={est.bayesian_mean:.1%}±{est.bayesian_std:.1%}, "
        f"{est.sample_adequacy})"
    )


def _print_profile(profile: PlayerProfile) -> None:
    print(f"=== Perfil: {profile.nickname} (mãos observadas: {profile.hands_observed}) ===")
    print(f"  VPIP:          {_fmt_rate(profile.vpip)}")
    print(f"  PFR:           {_fmt_rate(profile.pfr)}")
    print(f"  3-Bet:         {_fmt_rate(profile.three_bet)}")
    print(f"  Saw Flop:      {_fmt_rate(profile.saw_flop)}")
    print(f"  WTSD:          {_fmt_rate(profile.went_to_showdown)}")
    pf = profile.postflop_aggression
    af_str = f"{pf.aggression_factor:.2f}" if pf.aggression_factor is not None else "n/d (0 calls)"
    print(f"  Agressão Pós:  AFq={_fmt_rate(pf.aggression_frequency)} | AF={af_str}")
    bs = profile.overall_bet_sizing
    if bs.observations > 0 and bs.mean_pot_ratio is not None and bs.median_pot_ratio is not None:
        print(
            f"  Bet Sizing:    n={bs.observations}, média={bs.mean_pot_ratio:.2f}x pote, "
            f"mediana={bs.median_pot_ratio:.2f}x pote "
            f"(pequenas={bs.small_bets_count}, médias={bs.medium_bets_count}, grandes={bs.large_bets_count})"
        )
    else:
        print("  Bet Sizing:    sem apostas/raises observados")
    if profile.by_position:
        pos_parts = []
        for pos, pprof in profile.by_position.items():
            v_str = f"{pprof.vpip.empirical_rate:.0%}" if pprof.vpip.empirical_rate is not None else "n/d"
            p_str = f"{pprof.pfr.empirical_rate:.0%}" if pprof.pfr.empirical_rate is not None else "n/d"
            pos_parts.append(f"{pos}: n={pprof.hands} (VPIP {v_str}, PFR {p_str})")
        print(f"  Por posição:   {'; '.join(pos_parts)}")
    for note in profile.tendency_notes:
        print(f"  Nota: {note}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lugon Poker Lab — Player Intelligence (perfis estatísticos e transição entre capturas)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    import_parser = subparsers.add_parser("import", help="importa históricos de mãos JSON para base SQLite")
    import_parser.add_argument("files", type=Path, nargs="+", help="ficheiro(s) JSON com mãos observadas")
    import_parser.add_argument("--db", type=Path, required=True, help="caminho para a base SQLite de perfis")
    import_parser.add_argument("--json", action="store_true", help="imprime resultado em JSON")

    show_parser = subparsers.add_parser("show", help="apresenta estatísticas e incerteza por nickname")
    show_parser.add_argument("--db", type=Path, help="caminho para a base SQLite de perfis")
    show_parser.add_argument("--history", type=Path, nargs="*", help="ficheiro(s) JSON de mãos para análise direta")
    show_parser.add_argument("--player", type=str, help="filtrar por nickname específico")
    show_parser.add_argument("--json", action="store_true", help="imprime perfis em JSON")

    diff_parser = subparsers.add_parser("diff", help="deteta alterações de estado entre duas capturas PNG")
    diff_parser.add_argument("before", type=Path, help="primeira captura PNG")
    diff_parser.add_argument("after", type=Path, help="segunda captura PNG")
    diff_parser.add_argument("--json", action="store_true", help="imprime relatório de transição em JSON")

    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            all_histories = []
            for file_path in args.files:
                all_histories.extend(load_history_file(file_path))
            with ProfileStore(args.db) as store:
                summary = store.import_hands(all_histories)
                nicknames = store.list_nicknames()
            result = {**summary, "db": str(args.db), "players_tracked": nicknames}
            if args.json:
                print(json.dumps(result, ensure_ascii=False, indent=2))
            else:
                print(
                    f"Importadas: {result['imported']} | duplicadas ignoradas: {result['skipped_duplicates']} | "
                    f"total na base: {result['total_stored']} | jogadores: {', '.join(nicknames)}"
                )
            return 0

        if args.command == "show":
            if not args.db and not args.history:
                parser.error("show requer --db e/ou --history")
            store_path: str | Path = args.db if args.db else ":memory:"
            with ProfileStore(store_path) as store:
                if args.history:
                    for file_path in args.history:
                        store.import_hands(load_history_file(file_path))
                if args.player:
                    prof = store.get_profile(args.player)
                    if prof is None:
                        print(f"Erro: jogador {args.player!r} não encontrado.", file=sys.stderr)
                        return 1
                    profiles = {prof.nickname: prof}
                else:
                    profiles = store.all_profiles()

            if args.json:
                payload = {
                    "players": {name: prof.to_dict() for name, prof in profiles.items()},
                }
                print(json.dumps(payload, ensure_ascii=False, indent=2))
            else:
                for idx, prof in enumerate(profiles.values()):
                    if idx > 0:
                        print()
                    _print_profile(prof)
            return 0

        if args.command == "diff":
            prev_report = analyze_table(args.before)
            curr_report = analyze_table(args.after)
            transition = diff_table_captures(prev_report, curr_report)
            if args.json:
                print(json.dumps(transition, ensure_ascii=False, indent=2))
            else:
                print(
                    f"Transição válida: {transition['valid_transition']} | "
                    f"street: {transition['previous_street']} -> {transition['current_street']} | "
                    f"delta pote: {transition['pot_delta']}"
                )
                for event in transition["inferred_events"]:
                    print(f"  evento: {event}")
                for inc in transition["inconsistencies"]:
                    print(f"  inconsistência: {inc}")
                for amb in transition["ambiguities"]:
                    print(f"  ambiguidade: {amb}")
            return 0 if transition["valid_transition"] else 1

        return 2
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"Erro: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
