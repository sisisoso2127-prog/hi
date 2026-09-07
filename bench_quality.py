#!/usr/bin/env python3
"""
bench_quality.py
================
A/B de la QUALITE DE SOLUTION a grande taille.

Pourquoi ce banc n'existait pas, et pourquoi il fallait le creer. Toute la
mesure jusqu'ici portait sur la BORNE : ecart garanti, optimalite prouvee,
appels au solveur. Sur le lot fige c'etait suffisant, la recherche y trouvant
l'optimum sur 90 instances sur 90 -- il n'y avait tout simplement pas de
place pour un progres sur la solution. A n >= 20 la situation s'inverse : la
borne ne bouge pas (92-98 % d'ecart garanti quoi qu'on fasse du budget) et
c'est la SOLUTION qui laisse de la marge.

Ce que compare ce banc, a budget identique et par PAIRES (meme instance,
meme graine) :

    A  budget de certification integralement consacre a la borne (avant)
    B  sonde d'un tour ; si la borne ne ferme pas, le reste va a la
       DIVERSIFICATION PAR COUPES D'EFFICACITE (apres)

Lecture. Les deux valeurs rendues sont des points EFFICACES CERTIFIES, donc
comparables sans reserve : q_lb plus grand = strictement meilleur. Aucune
verite terrain n'est necessaire, et c'est heureux, car a ces tailles
l'enumeration de E est hors de portee.

Controle de surete a chaque execution : q_lb <= q_ub, et les deux
encadrements A et B doivent se recouper (ils encadrent le meme q*).

Usage :  python bench_quality.py [budget_recherche] [budget_borne]
"""

import sys
import time

import numpy as np

from molfp_core import ORACLE_CALLS, max_f_over_S, reset_oracle_counter
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [20, 30, 40, 50]
GRAINES = [0, 1, 2]
CORRS = [0.0, 0.5]


def une(inst, t_rech, t_borne, seed, diversify):
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=t_rech, bound_budget=t_borne,
                       seed=seed, archive_cuts=True,
                       cut_diversify=diversify)
    return {"q": float(r.q_lb), "ub": r.q_ub, "ilp": r.ilp_calls,
            "t": time.time() - t0, "arch": len(r.archive),
            "neufs": r.cert.get("diversify_new", 0),
            "prouve": r.proved_optimal}


def main() -> int:
    t_rech = float(sys.argv[1]) if len(sys.argv) > 1 else 18.0
    t_borne = float(sys.argv[2]) if len(sys.argv) > 2 else 12.0

    print("=" * 106)
    print(f"QUALITE DE SOLUTION A GRANDE TAILLE - budget {t_rech:g} + "
          f"{t_borne:g} s par variante, comparaison APPARIEE")
    print("=" * 106)
    print(f"{'n':>4}{'corr':>6}{'graine':>7}"
          f"{'q_lb A':>12}{'q_lb B':>12}{'gain %':>9}"
          f"{'neufs':>7}{'|arch| A/B':>12}{'ILP A/B':>12}{'coherent':>10}")
    print("-" * 106)

    gains, mieux, pire, egal, tout_ok = [], 0, 0, 0, True
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in GRAINES:
                inst = generate(n=n, m=m, p=3, seed=1 + g,
                                rhs_scale=1.0, corr=corr)
                mS = float(max_f_over_S(inst).q_star)
                A = une(inst, t_rech, t_borne, g, diversify=False)
                B = une(inst, t_rech, t_borne, g, diversify=True)
                # les deux encadrements doivent contenir le meme q*
                lo = max(A["q"], B["q"])
                hi = min(min(A["ub"] or np.inf, mS), min(B["ub"] or np.inf, mS))
                ok = lo <= hi + 1e-9
                tout_ok &= ok
                gain = (B["q"] - A["q"]) / max(1e-12, abs(A["q"])) * 100
                gains.append(gain)
                if B["q"] > A["q"] + 1e-12:
                    mieux += 1
                elif B["q"] < A["q"] - 1e-12:
                    pire += 1
                else:
                    egal += 1
                print(f"{n:>4}{corr:>6.2f}{g:>7}"
                      f"{A['q']:>12.4f}{B['q']:>12.4f}{gain:>+8.1f}%"
                      f"{B['neufs']:>7}"
                      f"{A['arch']:>6}/{B['arch']:<5}"
                      f"{A['ilp']:>6}/{B['ilp']:<5}"
                      f"{'ok' if ok else 'KO':>10}", flush=True)

    print("-" * 106)
    d = np.asarray(gains)
    print(f"  gain sur q_lb : median {np.median(d):+.2f} %   "
          f"moyen {d.mean():+.2f} %   max {d.max():+.1f} %   min {d.min():+.1f} %")
    print(f"  MIEUX {mieux}   PIRE {pire}   EGAL {egal}   sur {len(gains)} paires")
    print(f"  COHERENCE des encadrements : "
          f"{'TOUT VALIDE' if tout_ok else 'ECHEC'}")
    return 0 if tout_ok else 1


if __name__ == "__main__":
    sys.exit(main())
