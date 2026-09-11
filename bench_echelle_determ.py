"""
bench_echelle_determ.py
=======================
LE SOMMET DE L'ECHELLE (n = 30, 40, 50), SOUS PLAFOND D'APPELS.

Pourquoi la lecture change ici. Aux deux lots precedents, la quantite
informative etait le nombre de PREUVES. A n >= 30 et front epais, la methode
ne prouve presque rien : compter les preuves y renverrait 0 contre 0 et ne
dirait rien du tout. L'axe informatif devient l'ECART GARANTI, et ses deux
termes separement -- un minorant certifie et un majorant valide.

Ce banc mesure donc, par paire (meme instance, meme graine) :
  * l'ecart garanti de chaque bras et le delta apparie ;
  * les trois axes de RECUL du protocole : preuve perdue, valeur plus faible,
    borne plus lache ;
  * la coherence interne q_lb <= q_ub, seul controle disponible sans verite
    terrain -- l'enumeration est hors de portee a ces tailles.

Grille identique a celle de l'etude d'echelle du memoire : m = n/2 + 1,
p = 3, rhs_scale = 1, corr dans {0 ; 0,5 ; 0,9}, graines 1 et 2. Elle est
prolongee a n = 50, que le memoire n'atteignait pas.

Les deux bras portent la configuration courante ; seule la coupe
d'efficacite les separe. Chaque configuration est jouee deux fois et toute
ligne non reproductible est exclue de la conclusion.

Usage :  python bench_echelle_determ.py [cap1,cap2,...] [n1,n2,...]
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

CORRS = [0.00, 0.50, 0.90]
SEEDS = [1, 2]
P = 3
BRAS = [("sans", False), ("avec", True)]


def une(inst, cap: int, arch: bool) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=0,
                       archive_cuts=arch, ilp_budget=cap,
                       pool_alterne=True, geom_bound=True)
    return dict(prouve=bool(r.proved_optimal),
                lb=float(r.q_lb) if r.q_lb is not None else None,
                ub=r.q_ub, ilp=r.ilp_calls,
                gap=(r.gap * 100) if r.gap is not None else float("nan"))


def campagne(cap: int, sizes: List[int]) -> Dict:
    print("=" * 112)
    print(f"ECHELLE SOUS PLAFOND DETERMINISTE DE {cap} APPELS  [V* S* D*]")
    print("=" * 112)
    print(f"{'n':>4}{'corr':>7}{'gr':>4}"
          f"{'ecart sans':>13}{'ecart avec':>13}{'delta':>9}"
          f"{'lb sans':>11}{'lb avec':>11}{'ub sans':>12}{'ub avec':>12}"
          f"{'det':>5}")
    print("-" * 112)

    mieux = pire = egal = 0
    p_perdue = r_val = r_borne = 0
    prv = {k: 0 for _, k in BRAS}
    incoherent = 0
    non_reprod: List[str] = []
    deltas: List[float] = []
    n_paires = 0
    alertes: List[str] = []

    for n in sizes:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for s in SEEDS:
                inst = generate(n=n, m=m, p=P, seed=s, rhs_scale=1.0,
                                corr=corr)
                res, reprod = {}, True
                for _, arch in BRAS:
                    a = une(inst, cap, arch)
                    b = une(inst, cap, arch)
                    if (a["prouve"], a["ilp"], a["lb"]) != (b["prouve"],
                                                            b["ilp"], b["lb"]):
                        reprod = False
                    res[arch] = a
                    prv[arch] += int(a["prouve"])
                    if (a["lb"] is not None and a["ub"] is not None
                            and a["lb"] > a["ub"] + 1e-9):
                        incoherent += 1
                        alertes.append(f"  INCOHERENT q_lb > q_ub : n={n} "
                                       f"corr={corr} graine={s}")
                A, B = res[False], res[True]
                tag = f"n{n}_c{corr}_s{s}"
                if not reprod:
                    non_reprod.append(tag)
                    print(f"{n:>4}{corr:>7.2f}{s:>4}"
                          f"{'--':>13}{'--':>13}{'--':>9}"
                          f"{'':>11}{'':>11}{'':>12}{'':>12}{'KO':>5}",
                          flush=True)
                    continue
                n_paires += 1
                d = A["gap"] - B["gap"]          # > 0 : « avec » est meilleur
                deltas.append(d)
                if d > 1e-9:
                    mieux += 1
                elif d < -1e-9:
                    pire += 1
                else:
                    egal += 1
                if A["prouve"] and not B["prouve"]:
                    p_perdue += 1
                    alertes.append(f"  PREUVE PERDUE : n={n} corr={corr} gr={s}")
                if (A["lb"] is not None and B["lb"] is not None
                        and B["lb"] < A["lb"] - 1e-12):
                    r_val += 1
                    alertes.append(f"  VALEUR PLUS FAIBLE : n={n} corr={corr} "
                                   f"gr={s}  {A['lb']:.4f} -> {B['lb']:.4f}")
                if (A["ub"] is not None and B["ub"] is not None
                        and B["ub"] > A["ub"] + 1e-9):
                    r_borne += 1
                    alertes.append(f"  BORNE PLUS LACHE : n={n} corr={corr} "
                                   f"gr={s}  {A['ub']:.3f} -> {B['ub']:.3f}")
                fu = lambda v: ("--" if v is None else f"{v:.2f}")
                print(f"{n:>4}{corr:>7.2f}{s:>4}"
                      f"{A['gap']:>12.1f}%{B['gap']:>12.1f}%{d:>+9.2f}"
                      f"{A['lb']:>11.3f}{B['lb']:>11.3f}"
                      f"{fu(A['ub']):>12}{fu(B['ub']):>12}{'ok':>5}",
                      flush=True)

    print("-" * 112)
    print(f"  paires retenues        : {n_paires}"
          + (f"   exclues : {', '.join(non_reprod)}" if non_reprod else ""))
    print(f"  ecart garanti          : mieux {mieux}   pire {pire}   "
          f"egal {egal}")
    if deltas:
        d = np.asarray(deltas)
        print(f"  delta d'ecart (points) : median {np.median(d):+.2f}   "
              f"moyen {d.mean():+.2f}   min {d.min():+.2f}   max {d.max():+.2f}")
    print(f"  optimalite prouvee     : sans {prv[False]}   avec {prv[True]}")
    print(f"  recul : preuve {p_perdue}   valeur {r_val}   borne {r_borne}")
    print(f"  q_lb > q_ub            : {incoherent}")
    for a in alertes:
        print(a)
    return dict(cap=cap, n_paires=n_paires, mieux=mieux, pire=pire, egal=egal,
                med=float(np.median(deltas)) if deltas else float("nan"),
                moy=float(np.mean(deltas)) if deltas else float("nan"),
                pp=p_perdue, rv=r_val, rb=r_borne)


if __name__ == "__main__":
    caps = ([int(c) for c in sys.argv[1].split(",")]
            if len(sys.argv) > 1 else [100, 300])
    sizes = ([int(x) for x in sys.argv[2].split(",")]
             if len(sys.argv) > 2 else [30, 40, 50])
    t0 = time.time()
    bilan = []
    for cap in caps:
        bilan.append(campagne(cap, sizes))
        print()
    print("=" * 112)
    print("BILAN")
    for b in bilan:
        print(f"  plafond {b['cap']:>4} : ecart mieux {b['mieux']} / pire "
              f"{b['pire']} / egal {b['egal']} sur {b['n_paires']}   "
              f"median {b['med']:+.2f} pt   "
              f"reculs {b['pp']}+{b['rv']}+{b['rb']}")
    print(f"  ({time.time()-t0:.0f} s)")
    print("\n=== TERMINE ===")
