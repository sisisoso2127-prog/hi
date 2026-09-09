#!/usr/bin/env python3
"""
bench_budget.py
===============
TRIPLER LE PLAFOND : QUE DEVIENT L'ECART GARANTI, ET PAR QUEL TERME ?

`bench_recherche.py` etablit que la recherche n'est PAS a son plafond : en
triplant son budget, l'incumbent progresse sur 5 lignes sur 12, jusqu'a
+49,6 % -- et notamment sur n = 40, corr = 0, graine 1, la ligne meme ou la
seconde route converge (la borne EST max_R f) sans que l'ecart bouge.

Mais un incumbent qui monte de moitie ne resserre pas forcement l'ecart de
moitie : l'ecart vaut (q_ub - q_lb)/|q_ub|, et si q_ub est grand, un gain
sur q_lb s'y dilue. La question posee ici est donc la suivante, et elle ne
se deduit pas du banc precedent :

    en triplant le plafond de la METHODE COMPLETE, de combien de POINTS
    l'ecart garanti descend-il, et lequel des deux termes le doit-on ?

On mesure les deux termes separement a chaque plafond. La decomposition
est exacte :

    ecart(300) - ecart(900) = [ce que gagne q_lb] + [ce que gagne q_ub],

chaque part etant evaluee en gelant l'autre terme. Aucune verite terrain
n'est requise -- on compare la methode a elle-meme.

Usage :  python bench_budget.py [graines] [plafonds separes par virgules]
"""

import os
import sys
import time

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [int(v) for v in os.environ.get("MOLFP_TAILLES", "20,30,40").split(",")]
CORRS = [0.0, 0.5]


def une(inst, cap, seed, mS):
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True)
    ub = min(r.q_ub if r.q_ub is not None else np.inf, mS)
    lb = float(r.q_lb)
    return {"lb": lb, "ub": float(ub), "ilp": r.ilp_calls,
            "geom": r.cert.get("geom"), "t": time.time() - t0,
            "ecart": (ub - lb) / max(1e-12, abs(ub)) * 100}


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    caps = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2
                             else "300,900").split(",")]
    c0, c1 = caps[0], caps[-1]

    entete = (f"{'n':>4}{'corr':>6}{'gr':>4}"
              f"{'q_lb@' + str(c0):>11}{'q_lb@' + str(c1):>11}"
              f"{'q_ub@' + str(c0):>11}{'q_ub@' + str(c1):>11}"
              f"{'ecart@' + str(c0):>11}{'ecart@' + str(c1):>11}"
              f"{'gain':>8}{'du LB':>8}{'du UB':>8}")
    print("=" * len(entete))
    print(f"METHODE COMPLETE, plafonds {caps} - d'ou vient le gain ?")
    print("=" * len(entete))
    print(entete)
    print("-" * len(entete))

    parts_lb, parts_ub, gains = [], [], []
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in range(1, n_gr + 1):
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                mS = upper_bound_over_S(inst)
                a, b = une(inst, c0, g, mS), une(inst, c1, g, mS)

                def ec(lb, ub):
                    return (ub - lb) / max(1e-12, abs(ub)) * 100

                # decomposition exacte : on fait varier un terme a la fois.
                # part du LB   : ecart(lb0,ub0) - ecart(lb1,ub0)
                # part du UB   : ecart(lb1,ub0) - ecart(lb1,ub1)
                # leur somme vaut exactement ecart(300) - ecart(900).
                e0 = ec(a["lb"], a["ub"])
                e1 = ec(b["lb"], b["ub"])
                inter = ec(b["lb"], a["ub"])
                p_lb, p_ub = e0 - inter, inter - e1
                gains.append(e0 - e1)
                parts_lb.append(p_lb)
                parts_ub.append(p_ub)
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      f"{a['lb']:>11.4f}{b['lb']:>11.4f}"
                      f"{a['ub']:>11.3f}{b['ub']:>11.3f}"
                      f"{e0:>10.1f}%{e1:>10.1f}%"
                      f"{e0 - e1:>+8.2f}{p_lb:>+8.2f}{p_ub:>+8.2f}",
                      flush=True)

    print("-" * len(entete))
    G, L, U = map(np.asarray, (gains, parts_lb, parts_ub))
    print(f"  points d'ecart gagnes en triplant le plafond : "
          f"median {np.median(G):+.2f}   moyen {G.mean():+.2f}   "
          f"max {G.max():+.2f}")
    print(f"  dont imputables a l'INCUMBENT : median {np.median(L):+.2f}   "
          f"moyen {L.mean():+.2f}   max {L.max():+.2f}")
    print(f"  dont imputables a la BORNE    : median {np.median(U):+.2f}   "
          f"moyen {U.mean():+.2f}   max {U.max():+.2f}")
    tot = abs(L).sum() + abs(U).sum()
    if tot > 0:
        print(f"  repartition de l'effort utile : "
              f"incumbent {100*abs(L).sum()/tot:.0f} %   "
              f"borne {100*abs(U).sum()/tot:.0f} %")
    return 0


if __name__ == "__main__":
    sys.exit(main())
