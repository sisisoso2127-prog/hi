#!/usr/bin/env python3
"""
bench_unifie_coupe.py
=====================
LA FORME UNIFIEE LA OU LA ROUTE EST INTERROMPUE.

`bench_unifie.py` a rendu un gain EXACTEMENT NUL sur 54 executions, et la
raison etait structurelle : le denominateur restreint D_t+ n'agit que si la
seconde route s'arrete AVANT d'avoir converge -- a convergence v <= 0, la
borne vaut t exactement -- et la route converge partout ou elle s'engage, en
deux a quatre appels entiers. Il n'y avait donc aucune ligne ou le gain
PUISSE se voir. Les instances a n = 50 et 60 convergent aussi.

Ce banc construit le regime manquant de la seule facon honnete : en
reduisant l'allocation PROPRE de la route (`geom_iters`) au-dessous de ce
qu'il lui faut. Ce n'est pas un artifice -- c'est exactement la situation
que le memoire decrit ailleurs, celle d'une route affamee par le plafond --
et cela parametre proprement « a quelle distance de la convergence » on se
place.

A geom_iters = 1 la route ne fait qu'un tour : elle n'a pas le temps
d'avancer son seuil, et la borne se lit au seuil initial. C'est le cas ou
D_t+ doit valoir exactement le D+ du Th. 5. A 2 et 3 tours, le seuil a
avance mais la convergence n'est pas atteinte : c'est la que la forme
unifiee gagne sur les DEUX enonces anterieurs a la fois.

Usage :  python bench_unifie_coupe.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P

TAILLES = [20, 30, 40]
CORRS = [0.0, 0.5]


def une(inst, cap, g, dplus, iters):
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=g,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True, pool_alterne=True,
                       seuil_dplus=dplus, geom_iters=iters)
    return (r.q_ub if r.q_ub is not None else float("inf"),
            r.ilp_calls, r.cert.get("geom"))


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print("=" * 100)
    print(f"FORME UNIFIEE, ROUTE INTERROMPUE -- plafond {cap}, {n_gr} graines")
    print("=" * 100)
    for iters in (1, 2, 3):
        gains, mieux, pire, viole, n_lim = [], 0, 0, 0, 0
        print(f"\n--- geom_iters = {iters} "
              f"(la route ne peut faire que {iters} tour(s)) ---")
        print(f"{'n':>4}{'corr':>6}{'gr':>4}{'statut':>9}"
              f"{'q_ub Dmin':>12}{'q_ub D_t+':>12}{'gain %':>9}{'ILP s/a':>11}")
        for n in TAILLES:
            m = max(3, n // 2 + 1)
            for corr in CORRS:
                for g in range(1, n_gr + 1):
                    inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                    corr=corr)
                    ub0, i0, st0 = une(inst, cap, g, False, iters)
                    ub1, i1, st1 = une(inst, cap, g, True, iters)
                    if st0 == "limit":
                        n_lim += 1
                    if ub1 > ub0 + 1e-9:
                        viole += 1
                    gp = ((ub0 - ub1) / abs(ub0) * 100
                          if ub0 not in (0, float("inf")) else 0.0)
                    gains.append(gp)
                    if gp > 1e-9:
                        mieux += 1
                    elif gp < -1e-9:
                        pire += 1
                    print(f"{n:>4}{corr:>6.1f}{g:>4}{str(st0):>9}"
                          f"{ub0:>12.4f}{ub1:>12.4f}{gp:>+9.2f}"
                          f"{f'{i0}/{i1}':>11}", flush=True)
        print(f"  lignes 'limit' : {n_lim}/{len(gains)}   "
              f"resserrement median {statistics.median(gains):+.2f} %  "
              f"moyen {statistics.mean(gains):+.2f} %  "
              f"max {max(gains):+.2f} %")
        print(f"  mieux {mieux}   pire {pire}   "
              f"borne unifiee plus lache : {viole} (doit valoir 0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
