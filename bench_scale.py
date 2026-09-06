"""
bench_scale.py
==============
Passage a l'echelle : ou la methode exacte cesse de conclure, et ce que la
matheuristique certifie au-dela.

Aucune enumeration. Les deux methodes recoivent le MEME budget.

STRATIFICATION PAR `corr`. La campagne de difficulte a etabli que `corr` est
le levier structurel : a `(n, m, p)` fixe, `|E|` varie d'un facteur 40 selon
ce parametre, et le taux d'echec de la methode exacte suit. Une etude de
passage a l'echelle menee a `corr` fixe ne mesure donc qu'une tranche du
probleme, et le projet s'est lui-meme donne pour regle qu'une comparaison
ignorant `corr` est ininterpretable. La grille croise donc `n` et `corr`.

Le plan est CONTROLE et non observationnel : le generateur construit `A` et
`b` independamment de `corr`, donc le domaine `S` est rigoureusement
identique d'un niveau de `corr` a l'autre pour un meme `(n, m, seed)`. On
manipule la cause supposee, on ne l'observe pas.

Trois quantites seulement sont rapportees, toutes calculables sans connaitre
q* :

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

SIZES = [10, 15, 20, 25, 30, 40]
CORRS = [0.00, 0.50, 0.90]     # E epais -> E mince, a domaine S identique
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
        for corr in CORRS:
         for s in INSTANCE_SEEDS:
            inst = generate(n=n, m=m, p=P, seed=s, rhs_scale=1.0, corr=corr)

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

            rows.append((n, corr, e_st, who, gap))
            print(f"{inst.name:<22}{n:>4}{m:>4}"
                  f"{e_val:>12.4f}{e_st:>10}"
                  f"{lo:>12.4f}{hi:>12.4f}{gap:>9.1f}%{ilp:>7}{who:>16}",
                  flush=True)

    print("-" * 114)
    n_proved = sum(1 for r in rows if r[2] == "optimal")
    n_mh = sum(1 for r in rows if r[3] == "matheuristique")
    print(f"exact prouve l'optimalite   : {n_proved}/{len(rows)} instances")
    print(f"matheuristique strictement meilleure : {n_mh}/{len(rows)}")

    cell = {}
    for n, corr, st, who, gap in rows:
        cell.setdefault((n, corr), []).append((st == "optimal", gap))

    print("\nECART GARANTI MEDIAN  (lignes : n ; colonnes : corr)")
    print(f"{'n':>5}" + "".join(f"{c:>12.2f}" for c in CORRS) + f"{'toutes':>12}")
    for n in SIZES:
        line = f"{n:>5}"
        for c in CORRS:
            v = cell.get((n, c), [])
            line += (f"{np.nanmedian([g for _, g in v]):>11.1f}%"
                     if v else f"{'-':>12}")
        allv = [g for c in CORRS for _, g in cell.get((n, c), [])]
        line += f"{np.nanmedian(allv):>11.1f}%" if allv else f"{'-':>12}"
        print(line)

    print("\nOPTIMALITE PROUVEE PAR LA METHODE EXACTE  (lignes : n ; colonnes : corr)")
    print(f"{'n':>5}" + "".join(f"{c:>12.2f}" for c in CORRS))
    for n in SIZES:
        line = f"{n:>5}"
        for c in CORRS:
            v = cell.get((n, c), [])
            line += (f"{sum(1 for a, _ in v if a):>8}/{len(v):<4}"
                     if v else f"{'-':>12}")
        print(line)

    print("\nPar niveau de corr, toutes tailles confondues :")
    print(f"{'corr':>6}{'exact prouve':>15}{'ecart garanti median':>24}")
    for c in CORRS:
        v = [x for (n, cc), lst in cell.items() if cc == c for x in lst]
        print(f"{c:>6.2f}{sum(1 for a, _ in v if a):>8}/{len(v):<6}"
              f"{np.nanmedian([g for _, g in v]):>23.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
