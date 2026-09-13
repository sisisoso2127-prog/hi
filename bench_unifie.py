#!/usr/bin/env python3
"""
bench_unifie.py
===============
LA FORME UNIFIEE DE LA BORNE, MESUREE.

Les deux enonces -- seuil libre (Th. 8) et denominateur restreint (Th. 5) --
ne sont pas incomparables : ils se composent. Pour tout seuil t et tout
v >= max_R (N - tD) avec v >= 0,

    D_t+ = min { D(x) : x dans R, N(x) - t D(x) >= 0 },
    max_R f <= t + v / D_t+.

Th. 5 est le cas t = q ; la forme a Dmin est celle ou D_t+ est remplace par
le minorant plus faible Dmin <= D_t+. La forme unifiee DOMINE donc les deux
et non seulement leur minimum, au prix d'UN programme de plus, au seuil
retenu.

Ce banc mesure ce que ce programme achete, sur les lignes ou la seconde
route ne converge pas -- les seules ou le denominateur intervienne, puisque
a convergence la borne est exacte. Lecture appariee : meme instance, meme
graine, seul `seuil_dplus` change.

Usage :  python bench_unifie.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P

TAILLES = [20, 30, 40]
CORRS = [0.0, 0.5]


def une(inst, cap, g, dplus):
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=g,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True, pool_alterne=True,
                       seuil_dplus=dplus)
    return (float(r.q_lb),
            r.q_ub if r.q_ub is not None else float("inf"),
            r.ilp_calls, r.cert.get("geom"), bool(r.proved_optimal))


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    # tailles surchargeables : le gain de D_t+ ne peut se voir que la ou la
    # route s'arrete AVANT de converger, donc sur des instances assez
    # grandes pour que chaque appel entier coute cher.
    global TAILLES
    if len(sys.argv) > 3:
        TAILLES = [int(v) for v in sys.argv[3].split(",")]
    print("=" * 96)
    print(f"FORME UNIFIEE : D_t+ au seuil retenu, contre Dmin. "
          f"plafond {cap}, {n_gr} graines")
    print("=" * 96)
    print(f"{'n':>4}{'corr':>6}{'gr':>4}{'statut':>9}"
          f"{'q_ub Dmin':>12}{'q_ub D_t+':>12}{'gain %':>9}"
          f"{'ILP s/a':>11}")
    print("-" * 96)
    gains, mieux, pire, egal, viole = [], 0, 0, 0, 0
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in range(1, n_gr + 1):
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                lb0, ub0, i0, st0, _ = une(inst, cap, g, False)
                lb1, ub1, i1, st1, _ = une(inst, cap, g, True)
                # controle : le minorant ne doit pas bouger, et la borne
                # unifiee ne doit JAMAIS depasser l'ancienne
                if ub1 > ub0 + 1e-9:
                    viole += 1
                gp = (ub0 - ub1) / abs(ub0) * 100 if ub0 not in (0, float("inf")) else 0.0
                gains.append(gp)
                if gp > 1e-9:
                    mieux += 1
                elif gp < -1e-9:
                    pire += 1
                else:
                    egal += 1
                print(f"{n:>4}{corr:>6.1f}{g:>4}{str(st0):>9}"
                      f"{ub0:>12.4f}{ub1:>12.4f}{gp:>+9.2f}"
                      f"{f'{i0}/{i1}':>11}")
    print("-" * 96)
    print(f"  resserrement : median {statistics.median(gains):+.2f} %  "
          f"moyen {statistics.mean(gains):+.2f} %  max {max(gains):+.2f} %")
    print(f"  mieux {mieux}   egal {egal}   pire {pire}")
    print(f"  BORNE UNIFIEE PLUS LACHE QUE L'ANCIENNE : {viole} fois "
          f"(doit valoir 0 -- elle la domine par construction)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
