#!/usr/bin/env python3
"""
bench_budgets.py
================
UN SEUL PLAFOND POUR LES DEUX SOURCES, OU UN PLAFOND CHACUNE ?

La regle du vivier unique repose sur une mesure ancienne : « poser 40
coupes d'archive EN PLUS des 40 de dominance fait tomber l'optimalite
prouvee de 84 a 57 sur 90 instances ». Ce chiffre pose trois problemes.

  1. Le code en porte 57, l'article 56. L'un des deux est recopie de
     travers, et rien ne permet de dire lequel.
  2. Il a ete obtenu sous un budget en TEMPS, dont nous etablissons par
     ailleurs qu'il fait varier le resultat d'une execution a l'autre dans
     un rapport atteignant 125 %.
  3. Il date d'une configuration que trois corrections ont depuis
     modifiee.

Un chiffre qu'on ne peut ni verifier ni reproduire ne doit pas servir a
justifier une decision de conception. Ce banc le remesure sous le protocole
deterministe, et le remplace.

DEUX BRAS, memes instances, meme plafond d'appels entiers :

  commun   les deux sources concourent sous UN plafond de `cut_batch`
           coupes par tour -- la regle en vigueur ;
  separes  chaque source recoit SON plafond de `cut_batch`, soit deux fois
           plus de coupes posees par tour.

Si le bras `separes` degrade, la regle du vivier unique est fondee et l'on
saura de combien. S'il ne degrade pas, la justification tombe et il faudra
le dire.

Usage :  python bench_budgets.py [plafond] [graines]
"""

import os
import sys
import time

import numpy as np

from molfp_core import reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [int(v) for v in os.environ.get("MOLFP_TAILLES", "20,30,40").split(",")]
CORRS = [0.0, 0.5]


def une(inst, cap, seed, separes, mS):
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True, budgets_separes=separes)
    ub = min(r.q_ub if r.q_ub is not None else np.inf, mS)
    lb = float(r.q_lb)
    return {"lb": lb, "ecart": (ub - lb) / max(1e-12, abs(ub)) * 100,
            "cuts": r.cert.get("n_cuts", 0),
            "arch": r.cert.get("archive_cuts", 0),
            "prouve": bool(r.proved_optimal), "ilp": r.ilp_calls,
            "t": time.time() - t0}


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_gr = int(sys.argv[2]) if len(sys.argv) > 2 else 3

    entete = (f"{'n':>4}{'corr':>6}{'gr':>4}"
              f"{'ecart commun':>14}{'ecart separes':>15}{'gain':>8}"
              f"{'coupes c/s':>13}{'prouve c/s':>12}")
    print("=" * len(entete))
    print(f"VIVIER : UN PLAFOND OU DEUX - plafond {cap} appels, {n_gr} graines")
    print("=" * len(entete))
    print(entete)
    print("-" * len(entete))

    ec, es, pc, ps = [], [], 0, 0
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in range(1, n_gr + 1):
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                mS = upper_bound_over_S(inst)
                a = une(inst, cap, g, False, mS)
                b = une(inst, cap, g, True, mS)
                ec.append(a["ecart"]); es.append(b["ecart"])
                pc += a["prouve"]; ps += b["prouve"]
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      f"{a['ecart']:>13.1f}%{b['ecart']:>14.1f}%"
                      f"{a['ecart'] - b['ecart']:>+8.2f}"
                      f"{str(a['cuts']) + '/' + str(b['cuts']):>13}"
                      f"{str(a['prouve'])[0] + '/' + str(b['prouve'])[0]:>12}",
                      flush=True)

    print("-" * len(entete))
    A, B = np.asarray(ec), np.asarray(es)
    d = A - B                                  # > 0 : `separes` est meilleur
    mieux = int((d > 1e-9).sum()); pire = int((d < -1e-9).sum())
    print(f"  ecart median   commun {np.median(A):5.1f}%   "
          f"separes {np.median(B):5.1f}%")
    print(f"  ecart moyen    commun {A.mean():5.1f}%   separes {B.mean():5.1f}%")
    print(f"  optimalite prouvee   commun {pc}/{len(A)}   separes {ps}/{len(B)}")
    print(f"  `separes` contre `commun` : MIEUX {mieux}  PIRE {pire}  "
          f"EGAL {len(d) - mieux - pire}")
    print()
    print("  Si `separes` degrade, la regle du vivier unique est fondee.")
    print("  Sinon, la justification publiee tombe et il faut le dire.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
