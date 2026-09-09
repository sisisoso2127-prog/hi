#!/usr/bin/env python3
"""
bench_cglp.py
=============
La coupe par programme generateur deplace-t-elle les 92-98 % ?

C'EST LA SEULE QUESTION. La section d'echelle mesure des ecarts garantis de
98,3 / 97,1 / 92,5 % a n = 20, 30, 40, inchanges par tout ce qui a ete
essaye jusqu'ici. Le diagnostic est etabli : le plafond de coupes sature,
et il sature parce que chaque coupe disjonctive coute p binaires. La coupe
generee par programme lineaire ne coute AUCUNE binaire. Si le diagnostic
est juste, elle doit mordre ici et nulle part ailleurs.

PROTOCOLE. Budget DETERMINISTE en appels au solveur entier, identique dans
les deux bras. Comparaison appariee, chaque configuration jouee DEUX FOIS
pour verifier la reproductibilite plutot que la supposer.

UNE PRECAUTION DE LECTURE QUI COMPTE. Le budget est compte en appels
ENTIERS, or la coupe generee ne coute que des PROGRAMMES LINEAIRES. Sous
cette unite elle parait donc gratuite, ce qu'elle n'est pas. La colonne des
appels lineaires est rapportee a cote pour que le cout reel reste visible :
un PL est d'un ordre de grandeur moins cher qu'un ILP, pas gratuit.

Usage :  python bench_cglp.py [plafond_appels] [nb_cglp]
"""

import os
import sys
import time

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

# ---------------------------------------------------------------------------
# CONFIGURATION MESUREE
# ---------------------------------------------------------------------------
# Ce banc a d'abord ete lance en configuration d'origine : vivier a
# classement commun, sans seconde route. Son ecart de reference etait alors
# 98,6 % a n = 20 ; la configuration courante donne 77,2 % sur la meme
# instance. La conclusion du banc -- lever le plafond ne suffit pas -- doit
# donc etre reexaminee sur la base ou elle porte reellement.
#
#   MOLFP_CFG=ooo   vivier commun, sans seconde route  (config d'origine)
#   MOLFP_CFG=eee   vivier alterne, seconde route      (defaut, courant)
CFG = os.environ.get("MOLFP_CFG", "eee").lower()
CFG_ALT = CFG[0:1] in ("e", "*", "1")
CFG_GEOM = CFG[1:2] in ("e", "*", "1")


def marqueur() -> str:
    return f"[V{'*' if CFG_ALT else 'o'} S{'*' if CFG_GEOM else 'o'} D*]"


TAILLES = [20, 30, 40]
CORRS = [0.0, 0.5]
GRAINES = [1]


def une(inst, cap, seed, n_cglp):
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap,
                       cut_diversify=False, cglp_extra=n_cglp,
                       pool_alterne=CFG_ALT, geom_bound=CFG_GEOM)
    return {"lb": r.q_lb, "ub": r.q_ub, "ilp": r.ilp_calls,
            "lp": ORACLE_CALLS["lp"], "cglp": r.cert.get("cglp_cuts", 0),
            "cuts": r.cert.get("n_cuts", 0), "t": time.time() - t0}


def _sig(d):
    return (str(d["lb"]), None if d["ub"] is None else round(d["ub"], 9),
            d["ilp"], d["cuts"], d["cglp"])


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 900
    n_cglp = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    # GRAINES restait a 1 : six lignes pour un resultat negatif, c'est peu.
    # Le troisieme argument permet d'en ajouter sans changer le protocole.
    global GRAINES
    GRAINES = list(range(1, (int(sys.argv[3]) if len(sys.argv) > 3 else 1) + 1))

    print("=" * 108)
    print(f"COUPE PAR PROGRAMME GENERATEUR - plafond {cap} appels entiers, "
          f"jusqu'a {n_cglp} coupes generees")
    print(f"configuration mesuree : {marqueur()}   graines : {len(GRAINES)}")
    print("=" * 108)
    print(f"{'n':>4}{'corr':>6}{'gr':>4}{'reprod':>8}"
          f"{'ecart sans':>12}{'ecart avec':>12}{'gain pts':>10}"
          f"{'cglp':>6}{'ILP s/a':>12}{'PL s/a':>14}{'coherent':>10}")
    print("-" * 108)

    gains, mieux, pire, egal, alea, ok_tout = [], 0, 0, 0, 0, True
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in GRAINES:
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                mS = upper_bound_over_S(inst)
                A, A2 = une(inst, cap, g, 0), une(inst, cap, g, 0)
                B, B2 = une(inst, cap, g, n_cglp), une(inst, cap, g, n_cglp)
                reprod = (_sig(A) == _sig(A2) and _sig(B) == _sig(B2))
                alea += 0 if reprod else 1

                def ecart(d):
                    hi = min(d["ub"] if d["ub"] is not None else np.inf, mS)
                    lo = float(d["lb"])
                    return (hi - lo) / max(1e-12, abs(hi)) * 100 \
                        if np.isfinite(hi) else np.nan

                ea, eb = ecart(A), ecart(B)
                coherent = (A["lb"] == B["lb"]) or True
                lo = max(float(A["lb"]), float(B["lb"]))
                hi = min(min(A["ub"] or np.inf, mS), min(B["ub"] or np.inf, mS))
                coherent = lo <= hi + 1e-9
                ok_tout &= coherent
                d = ea - eb                    # points d'ecart gagnes
                if reprod:
                    gains.append(d)
                    if d > 1e-9:
                        mieux += 1
                    elif d < -1e-9:
                        pire += 1
                    else:
                        egal += 1
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      f"{('oui' if reprod else 'NON'):>8}"
                      f"{ea:>11.1f}%{eb:>11.1f}%{d:>+9.2f}"
                      f"{B['cglp']:>6}"
                      f"{A['ilp']:>6}/{B['ilp']:<5}"
                      f"{A['lp']:>7}/{B['lp']:<6}"
                      f"{'ok' if coherent else 'KO':>10}", flush=True)

    print("-" * 108)
    print(f"  REPRODUCTIBILITE : {len(TAILLES)*len(CORRS)*len(GRAINES)-alea}"
          f"/{len(TAILLES)*len(CORRS)*len(GRAINES)} lignes stables")
    if gains:
        d = np.asarray(gains)
        print(f"  points d'ecart garanti gagnes : median {np.median(d):+.2f}   "
              f"moyen {d.mean():+.2f}   max {d.max():+.2f}   min {d.min():+.2f}")
    print(f"  MIEUX {mieux}   PIRE {pire}   EGAL {egal}")
    print(f"  COHERENCE des encadrements : "
          f"{'TOUT VALIDE' if ok_tout else 'ECHEC'}")
    print()
    print("  Rappel : le budget est compte en appels ENTIERS, or la coupe")
    print("  generee ne coute que des PROGRAMMES LINEAIRES. Sous cette unite")
    print("  elle parait gratuite ; la colonne « PL » dit son cout reel.")
    return 0 if ok_tout else 1


if __name__ == "__main__":
    sys.exit(main())
