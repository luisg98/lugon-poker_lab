"""Compute the 169x169 preflop all-in equity matrix across CPU cores.

One-time precomputation. The result lands in data/preflop_equity_169.json (~250KB) and is committed, so
every downstream tool (push/fold Nash, ICM, range equity) reads it instantly instead of recomputing.

    py tools/build_preflop_matrix.py --samples 100000 --workers 24
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lugon_poker import __version__
from lugon_poker.preflop import (
    DEFAULT_MATRIX_PATH,
    HAND_CLASSES,
    PreflopMatrix,
    compute_chunk,
    upper_triangle_pairs,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Constrói a matriz de equity pré-flop all-in 169x169.")
    parser.add_argument("--samples", type=int, default=100_000, help="amostras Monte Carlo por confronto")
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 4))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_MATRIX_PATH)
    parser.add_argument("--progress", type=Path, default=None, help="ficheiro JSON de progresso")
    args = parser.parse_args(argv)

    pairs = upper_triangle_pairs()
    tasks = [(i, j, args.samples, args.seed) for i, j in pairs]
    total = len(tasks)
    expected_se = (0.25 / args.samples) ** 0.5

    print(f"Lugon Poker Lab v{__version__}")
    print(f"{len(HAND_CLASSES)} classes -> {total:,} confrontos (triângulo superior estrito)")
    print(f"{args.samples:,} amostras/confronto -> erro-padrão ~{expected_se*100:.3f} pontos percentuais")
    print(f"{args.workers} processos; a diagonal é 0.5 exata e o triângulo inferior é 1 - superior")
    print(flush=True)

    values = [0.0] * total
    offsets = {(i, j): k for k, (i, j) in enumerate(pairs)}
    started = time.perf_counter()
    done = 0

    with mp.Pool(processes=args.workers) as pool:
        for i, j, value in pool.imap_unordered(compute_chunk, tasks, chunksize=16):
            values[offsets[(i, j)]] = value
            done += 1
            if done % 250 == 0 or done == total:
                elapsed = time.perf_counter() - started
                rate = done / elapsed
                eta = (total - done) / rate if rate > 0 else 0.0
                line = (
                    f"\r{done:,}/{total:,} ({done/total:6.1%})  "
                    f"{rate:6.1f} confrontos/s  decorrido {elapsed/60:5.1f}min  faltam {eta/60:5.1f}min"
                )
                print(line, end="", flush=True)
                if args.progress:
                    args.progress.parent.mkdir(parents=True, exist_ok=True)
                    args.progress.write_text(
                        json.dumps(
                            {
                                "done": done,
                                "total": total,
                                "fraction": done / total,
                                "elapsed_s": round(elapsed, 1),
                                "eta_s": round(eta, 1),
                            }
                        ),
                        encoding="utf-8",
                    )
    print(flush=True)

    matrix = PreflopMatrix(
        values=tuple(values),
        samples=args.samples,
        seed=args.seed,
        generated_at=datetime.now(timezone.utc).isoformat(),
        app_version=__version__,
    )
    path = matrix.save(args.output)
    elapsed = time.perf_counter() - started

    # Sanity gate: published all-in equities. A corrupt or mis-indexed matrix fails here, not silently later.
    checks = [("AA", "KK", 0.82), ("AA", "72o", 0.88), ("AKs", "QQ", 0.46), ("AKo", "22", 0.47)]
    print(f"\nGravada em {path} ({path.stat().st_size/1024:.0f}KB) em {elapsed/60:.1f}min")
    reloaded = PreflopMatrix.load(path)
    failures = 0
    for a, b, expected in checks:
        got = reloaded.equity(a, b)
        mirrored = reloaded.equity(b, a)
        ok = abs(got - expected) < 0.02 and abs(got + mirrored - 1.0) < 1e-9
        failures += not ok
        print(f"  {'OK ' if ok else 'FALHA'} {a} vs {b} = {got:.4f} (ref ~{expected:.2f}); espelho soma {got+mirrored:.6f}")
    if failures:
        print("\nVerificação falhou: matriz NÃO é de confiança.", file=sys.stderr)
        return 1
    print("\nMatriz validada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
