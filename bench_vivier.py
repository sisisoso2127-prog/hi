#!/usr/bin/env python3
"""
bench_vivier.py
===============
L'ALTERNANCE GUERIT-ELLE LA NON-MONOTONIE EN BUDGET ?

LE DEFAUT, tel que le diagnostic l'a etabli. En triplant le plafond, sur
six executions a n = 20, 30, 40 : l'archive double, le nombre total de
coupes augmente, et les coupes D'ARCHIVE s'effondrent -- 28->12, 26->0,
20->2, 33->10, 16->0, 22->15. Or seules les coupes d'archive peuvent VIDER
le relache ; les coupes de dominance preservent E par construction. Les
affamer supprime la seule preuve forte, et l'ecart garanti empire sur cinq
lignes sur six.

LA CAUSE n'est pas le plafond unique mais l'ECHELLE unique : les deux
sources sont classees sur le meme critere w^T x, alors que leurs capacites
different en nature. Quand la recherche s'allonge, les points domines
deviennent nombreux ET bien places, et ils raflent le vivier.

LA CORRECTION mesuree ici : classer a l'interieur de chaque source, puis
ALTERNER. Aucune fraction a regler.

CE QUE LE BANC DOIT ETABLIR, et il peut conclure contre l'alternance :

  1. a plafond 300, l'alternance ne DEGRADE pas -- sinon elle achete la
     monotonie au prix du regime nominal, ce qui n'est pas un marche ;
  2. a plafond 900, elle empeche l'effondrement des coupes d'archive ;
  3. et le passage 300 -> 900 cesse d'etre une perte.

Usage :  python bench_vivier.py [plafonds separes par virgules]
"""

import sys
import time

import numpy as np

from molfp_core import reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

# les six lignes du diagnostic : celles qui portent le defaut
LIGNES = [(20, 0.0, 1), (20, 0.0, 2), (30, 0.0, 1), (30, 0.0, 2),
          (40, 0.0, 2), (40, 0.5, 2)]


def une(inst, cap, seed, mS, alterne):
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True, pool_alterne=alterne)
    ub = min(r.q_ub if r.q_ub is not None else np.inf, mS)
    lb = float(r.q_lb)
    return {"lb": lb, "ub": float(ub), "arch": r.cert.get("archive_cuts", 0),
            "cuts": r.cert.get("n_cuts", 0), "ilp": r.ilp_calls,
            "ecart": (ub - lb) / max(1e-12, abs(ub)) * 100,
            "t": time.time() - t0}


def main() -> int:
    caps = [int(v) for v in (sys.argv[1] if len(sys.argv) > 1
                             else "300,900").split(",")]
    c0, c1 = caps[0], caps[-1]

    entete = (f"{'n':>4}{'corr':>6}{'gr':>4}{'vivier':>10}"
              f"{'ecart@' + str(c0):>11}{'ecart@' + str(c1):>11}"
              f"{'monotone':>10}"
              f"{'arch@' + str(c0):>10}{'arch@' + str(c1):>10}"
              f"{'coupes s/a':>12}")
    print("=" * len(entete))
    print(f"VIVIER : CLASSEMENT COMMUN CONTRE ALTERNANCE - plafonds {caps}")
    print("=" * len(entete))
    print(entete)
    print("-" * len(entete))

    res = {}
    for (n, corr, g) in LIGNES:
        m = max(3, n // 2 + 1)
        inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0, corr=corr)
        mS = upper_bound_over_S(inst)
        for alt, nom in ((False, "commun"), (True, "alterne")):
            a = une(inst, c0, g, mS, alt)
            b = une(inst, c1, g, mS, alt)
            res[(n, corr, g, alt)] = (a, b)
            mono = "oui" if b["ecart"] <= a["ecart"] + 1e-9 else "NON"
            print(f"{n:>4}{corr:>6.2f}{g:>4}{nom:>10}"
                  f"{a['ecart']:>10.1f}%{b['ecart']:>10.1f}%{mono:>10}"
                  f"{a['arch']:>10}{b['arch']:>10}"
                  f"{str(a['cuts']) + '/' + str(b['cuts']):>12}", flush=True)
        print("-" * len(entete))

    for cap, idx in ((c0, 0), (c1, 1)):
        d = [res[(n, c, g, False)][idx]["ecart"] for (n, c, g) in LIGNES]
        a = [res[(n, c, g, True)][idx]["ecart"] for (n, c, g) in LIGNES]
        diff = np.asarray(d) - np.asarray(a)      # >0 : alternance meilleure
        mieux = int((diff > 1e-9).sum())
        pire = int((diff < -1e-9).sum())
        print(f"  PLAFOND {cap} : alternance MIEUX {mieux}  PIRE {pire}  "
              f"EGAL {len(diff) - mieux - pire}   "
              f"gain median {np.median(diff):+.2f} pt   "
              f"moyen {diff.mean():+.2f} pt")
    for alt, nom in ((False, "classement commun"), (True, "alternance")):
        mono = sum(1 for L in LIGNES
                   if res[(L[0], L[1], L[2], alt)][1]["ecart"]
                   <= res[(L[0], L[1], L[2], alt)][0]["ecart"] + 1e-9)
        arch0 = [res[(L[0], L[1], L[2], alt)][1]["arch"] for L in LIGNES]
        print(f"  {nom:<20} : monotone sur {mono}/{len(LIGNES)} lignes   "
              f"coupes d'archive a {c1} : {arch0}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
