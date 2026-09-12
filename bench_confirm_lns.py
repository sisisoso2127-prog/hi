#!/usr/bin/env python3
"""
bench_confirm_lns.py
====================
DEUX FACTEURS QUE LE BALAYAGE N'A PAS TRANCHES.

`bench_sensibilite.py` a laisse deux reglages dans une position inconfortable :
la fraction LNS (0,4 -> 0,6 en diversification) et `max_stall`. Sur sa strate
B, l'ecart MEDIAN s'ameliore nettement quand on les change -- -7,56 points
pour une fraction constante a 0,4, -7,21 points pour max_stall=1 -- alors que
le compte APPARIE dit l'inverse : plus de lignes se degradent qu'elles ne
s'ameliorent. Une mediane qui baisse pendant que la majorite des lignes monte
est un signal d'echantillon trop petit, pas un resultat : la strate B ne
compte que 8 executions.

Ce banc ne fait qu'une chose : reprendre ces deux facteurs sur un lot
ELARGI -- huit instances a n=20 et n=30, trois graines, soit 24 executions
par configuration au lieu de 8 -- pour voir si la mediane tient ou si elle
etait un accident. Nous ne changeons aucun defaut avant de le savoir.

Usage :  python bench_confirm_lns.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys
from typing import Dict, List

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P
from bench_sensibilite import PROD

LOT = [dict(n=n, m=max(3, n // 2 + 1), p=3, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for c in (0.00, 0.50) for s in (1, 2)]

CONFIGS = [
    ("production        lns=(0,4 ; 0,6)  max_stall=3", {}),
    ("lns constante     lns=(0,4 ; 0,4)", dict(lns_frac=(0.4, 0.4))),
    ("lns basse         lns=(0,2 ; 0,4)", dict(lns_frac=(0.2, 0.4))),
    ("arret immediat    max_stall=1",     dict(max_stall=1)),
]


def campagne(graines: List[int], cap: int, **surcharge) -> Dict:
    cfg = dict(PROD)
    cfg.update(surcharge)
    ecarts, preuves = [], 0
    for spec in LOT:
        inst = generate(**spec)
        for g in graines:
            reset_oracle_counter()
            r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                               seed=g, ilp_budget=cap, **cfg)
            ecarts.append((r.gap * 100) if r.gap is not None
                          else float("nan"))
            preuves += int(r.proved_optimal)
    return dict(ecarts=ecarts, preuves=preuves, n=len(ecarts))


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print("=" * 96)
    print(f"CONFIRMATION -- lot elargi : {len(LOT)} instances "
          f"(n = 20 et 30) x {len(graines)} graines, plafond {cap}")
    print("=" * 96)

    base = campagne(graines, cap)
    e0 = base["ecarts"]
    fin0 = [x for x in e0 if x == x]
    print(f"\n{'configuration':<46}{'ecart med':>11}{'moyen':>9}"
          f"{'delta med':>11}{'mieux/pire':>12}{'preuves':>9}")
    print("-" * 96)
    for nom, sur in CONFIGS:
        r = base if not sur else campagne(graines, cap, **sur)
        e = r["ecarts"]
        fin = [x for x in e if x == x]
        paires = [(a, b) for a, b in zip(e, e0) if a == a and b == b]
        mieux = sum(1 for a, b in paires if a < b - 1e-9)
        pire = sum(1 for a, b in paires if a > b + 1e-9)
        dm = statistics.median(fin) - statistics.median(fin0)
        print(f"{nom:<46}{statistics.median(fin):>11.2f}"
              f"{statistics.mean(fin):>9.2f}{dm:>+11.2f}"
              f"{f'{mieux}/{pire}':>12}{r['preuves']:>4}/{r['n']:<4}")
    print("-" * 96)
    print("Lecture : une mediane et un compte apparie qui disent la meme chose")
    print("tranchent ; s'ils se contredisent encore sur 24 executions, le")
    print("facteur reste non tranche et le defaut ne bouge pas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
