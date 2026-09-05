"""
bench_scale.py
==============
Passage a l'echelle : ou la methode exacte cesse de conclure, et ce que la
matheuristique certifie au-dela.

Aucune enumeration. Les deux methodes recoivent le MEME budget. Trois
quantites seulement sont rapportees, toutes calculables sans connaitre q* :

  * `exact`   valeur rendue par `solve_P` et son statut. Un statut 'limit'
              signale une valeur NON prouvee : elle ne borne rien.
  * `q_lb`    valeur de la matheuristique, certifiee efficace par le Th. 2 :
              c'est une borne INFERIEURE sure de q*, a toute taille.
  * `q_ub`    meilleure borne superieure disponible, c'est-a-dire
              min(borne du Th. 5', max_S f). `max_S f` est exacte et coute un
              seul Dinkelbach : ne pas la prendre en compte reviendrait a
              rapporter un ecart plus mauvais que necessaire.

L'ecart rapporte est donc un ecart GARANTI, pas un ecart a une verite
inconnue.

Usage :  python bench_scale.py [budget_s]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from molfp_core import (ORACLE_CALLS, efficiency_test, max_f_over_S,
                        reset_oracle_counter)
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P
from molfp_oracle import solve_P

SIZES = [10, 12, 15, 20, 25, 30, 40]
INSTANCE_SEEDS = [1, 2]
P = 3


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 30.0

    print("=" * 114)
    print(f"PASSAGE A L'ECHELLE - budget identique {budget:.0f} s par methode,"
          f" aucune enumeration")
    print("=" * 114)
    print(f"{'instance':<22}{'n':>4}{'m':>4}"
          f"{'exact':>22}{'matheuristique':>34}{'gagnant':>16}")
    print(f"{'':<22}{'':>4}{'':>4}{'valeur':>12}{'statut':>10}"
          f"{'q_lb':>12}{'q_ub':>12}{'ecart garanti':>10}{'ILP':>7}{'':>16}")
    print("-" * 114)

    rows = []
    for n in SIZES:
        m = max(3, n // 2 + 1)
        for s in INSTANCE_SEEDS:
            inst = generate(n=n, m=m, p=P, seed=s, rhs_scale=1.0)

            reset_oracle_counter()
            try:
                e = solve_P(inst, time_limit=budget)
                e_val = float(e.q_star) if e.q_star is not None else float("nan")
                e_st = e.status
            except Exception as exc:                   # pragma: no cover
                e_val, e_st = float("nan"), type(exc).__name__

            reset_oracle_counter()
            mh = matheuristic_P(inst, time_budget=budget * 0.6,
                                bound_budget=budget * 0.4, seed=0)
            ilp = ORACLE_CALLS["ilp"]
            lo = float(mh.q_lb)

            rS = max_f_over_S(inst)
            max_S = float(rS.q_star) if rS.q_star is not None else float("inf")
            hi = min(mh.q_ub if mh.q_ub is not None else np.inf, max_S)
            gap = (hi - lo) / max(1e-12, abs(hi)) * 100 if np.isfinite(hi) else np.nan

            # le LB doit etre certifie : sans cela rien de ce qui precede ne tient
            assert efficiency_test(inst, mh.x_best).efficient, "LB NON CERTIFIE"

            # une valeur exacte non prouvee ne borne rien : on ne la declare
            # gagnante que si elle est prouvee
            if e_st == "optimal":
                who = "exact (prouve)"
            elif np.isnan(e_val) or lo > e_val + 1e-9:
                who = "matheuristique"
            elif abs(lo - e_val) <= 1e-9:
                who = "egalite"
            else:
                who = "exact (non prouve)"

            rows.append((n, e_st, who, gap))
            print(f"{inst.name:<22}{n:>4}{m:>4}"
                  f"{e_val:>12.4f}{e_st:>10}"
                  f"{lo:>12.4f}{hi:>12.4f}{gap:>9.1f}%{ilp:>7}{who:>16}",
                  flush=True)

    print("-" * 114)
    n_proved = sum(1 for r in rows if r[1] == "optimal")
    n_mh = sum(1 for r in rows if r[2] == "matheuristique")
    print(f"exact prouve l'optimalite   : {n_proved}/{len(rows)} instances")
    print(f"matheuristique strictement meilleure : {n_mh}/{len(rows)}")
    by_n = {}
    for n, st, who, gap in rows:
        by_n.setdefault(n, []).append((st == "optimal", gap))
    print(f"\n{'n':>5}{'exact prouve':>15}{'ecart garanti median':>24}")
    for n in sorted(by_n):
        v = by_n[n]
        print(f"{n:>5}{sum(1 for a, _ in v if a):>8}/{len(v):<6}"
              f"{np.nanmedian([g for _, g in v]):>23.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
