#!/usr/bin/env python3
"""
bench_stable.py
===============
LE VIVIER STABLE PAR PREFIXE : monotonie contre qualite de coupe.

Le diagnostic de non-monotonie a etabli que le relache n'est PAS emboite
dans le budget : l'archive grossit, de nouveaux centres s'intercalent dans
le prefixe des beta coupes posees, et l'on pose d'autres coupes plutot que
les memes et davantage. D'ou une borne qui peut reculer quand on donne plus
de temps.

La correction de principe est de classer le vivier sur un ordre MONOTONE en
budget. L'ordre d'insertion l'est : la trajectoire de la phase 1 ne depend
pas du plafond total, donc l'archive a plafond B est un prefixe de celle a
plafond B'. Le relache devient emboite, et la borne monotone par
construction.

Elle a un prix, et c'est tout l'enjeu du banc : on renonce a poser d'abord
les centres qui tirent U le plus haut. Deux mesures, donc, et pas une :

  1. MONOTONIE : sur les six lignes du diagnostic, l'ecart garanti
     empire-t-il encore quand on triple le plafond ?
  2. QUALITE : a plafond fixe, que coute l'ordre stable sur l'ecart garanti
     et sur les preuves ?

Une correction qui achete la monotonie en degradant partout la borne n'est
pas une correction, c'est un troc -- et il faut le chiffrer avant de le
proposer.

Usage :  python bench_stable.py [graines]
"""

from __future__ import annotations

import statistics
import sys

from molfp_core import reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_hybride import matheuristic_P

LIGNES = [(20, 0.0, 1), (20, 0.0, 2), (30, 0.0, 1), (30, 0.0, 2),
          (40, 0.0, 2), (40, 0.5, 2)]
TAILLES, CORRS = [20, 30, 40], [0.0, 0.5]


def une(inst, cap, g, stable, mS=None):
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=g,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True, pool_alterne=True,
                       vivier_stable=stable)
    ub = r.q_ub if r.q_ub is not None else float("inf")
    if mS is not None:
        ub = min(ub, mS)
    lb = float(r.q_lb)
    return dict(lb=lb, ub=ub, prouve=bool(r.proved_optimal),
                ecart=(ub - lb) / max(1e-12, abs(ub)) * 100,
                cuts=r.cert.get("n_cuts", 0), ilp=r.ilp_calls)


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 3

    print("=" * 92)
    print("1. MONOTONIE EN BUDGET -- les six lignes du diagnostic, 300 -> 900")
    print("=" * 92)
    print(f"{'n':>4}{'corr':>6}{'gr':>4}  {'vivier':<12}"
          f"{'ecart@300':>11}{'ecart@900':>11}{'monotone':>10}")
    print("-" * 92)
    bilan = {}
    for (n, corr, g) in LIGNES:
        m = max(3, n // 2 + 1)
        inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0, corr=corr)
        mS = upper_bound_over_S(inst)
        for stable, nom in ((False, "surrogat"), (True, "insertion")):
            a = une(inst, 300, g, stable, mS)
            b = une(inst, 900, g, stable, mS)
            mono = "oui" if b["ecart"] <= a["ecart"] + 1e-9 else "NON"
            bilan.setdefault(nom, []).append(mono)
            print(f"{n:>4}{corr:>6.1f}{g:>4}  {nom:<12}"
                  f"{a['ecart']:>10.1f}%{b['ecart']:>10.1f}%{mono:>10}",
                  flush=True)
        print("-" * 92)
    for nom, L in bilan.items():
        print(f"  {nom:<12} monotone sur {L.count('oui')}/{len(L)} lignes")

    print()
    print("=" * 92)
    print(f"2. CE QUE L'ORDRE STABLE COUTE -- plafond 300, {n_gr} graines")
    print("=" * 92)
    print(f"{'n':>4}{'corr':>6}{'gr':>4}"
          f"{'ecart surrogat':>16}{'ecart insertion':>17}{'delta':>9}"
          f"{'coupes s/i':>13}")
    print("-" * 92)
    d, p0, p1 = [], 0, 0
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in range(1, n_gr + 1):
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                a = une(inst, 300, g, False)
                b = une(inst, 300, g, True)
                p0 += int(a["prouve"]); p1 += int(b["prouve"])
                d.append(b["ecart"] - a["ecart"])
                print(f"{n:>4}{corr:>6.1f}{g:>4}"
                      f"{a['ecart']:>15.1f}%{b['ecart']:>16.1f}%"
                      f"{d[-1]:>+9.2f}"
                      f"{f'{a[chr(99)+chr(117)+chr(116)+chr(115)]}/{b[chr(99)+chr(117)+chr(116)+chr(115)]}':>13}",
                      flush=True)
    print("-" * 92)
    print(f"  delta d'ecart (insertion - surrogat) : median "
          f"{statistics.median(d):+.2f} pt   moyen {statistics.mean(d):+.2f} pt"
          f"   pire {max(d):+.2f} pt")
    print(f"  preuves : surrogat {p0}/{len(d)}   insertion {p1}/{len(d)}")
    print("\n  Un delta POSITIF est une degradation : l'ordre stable donne un")
    print("  ecart garanti plus grand. C'est le prix de la monotonie.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
