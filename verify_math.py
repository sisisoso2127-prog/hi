"""
verify_math.py
==============
Validation de la matheuristique, puis comparaison frontale avec la methode
exacte SUR SON REGIME CIBLE (|E| gros, corr faible).

  V8   LB sur : x_best est reellement efficace (Th. 2) et q_lb <= q*.
       Une matheuristique qui renvoie un LB optimiste est inutilisable :
       c'est le test le plus important.
  V9   BORNE VALIDE (Th. 5) : q_ub >= q*. Un seul contre-exemple invalide
       le theoreme.
  V10  Archive : aucun faux positif.
  V11  Qualite : frequence de q_lb == q*, et ecart garanti annonce.

Puis : matheuristique vs exact a BUDGET DE TEMPS EGAL sur corr in {0, 0.25},
ou la campagne a localise les 10 echecs de la methode exacte sur 17.

Usage :  python verify_math.py
"""

from __future__ import annotations

import time
from fractions import Fraction

import numpy as np

from molfp_core import ORACLE_CALLS, efficiency_test, reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P
from molfp_oracle import solve_P


def check(inst, gt, r) -> dict:
    """Verifie les quatre proprietes sur un resultat de matheuristique."""
    out = {}

    # V8 : incumbent reellement efficace et LB non optimiste
    eff = efficiency_test(inst, r.x_best).efficient
    out["V8"] = bool(eff and r.q_lb <= gt.q_star)
    out["V8_eff"] = eff
    out["V8_le"] = bool(r.q_lb <= gt.q_star)

    # V9 : borne superieure valide
    out["V9"] = (r.q_ub is None) or (r.q_ub >= float(gt.q_star) - 1e-9)

    # V10 : archive sans faux positif
    bad = sum(1 for x in r.archive if not efficiency_test(inst, x).efficient)
    out["V10"] = (bad == 0)
    out["V10_bad"] = bad

    # V11 : qualite
    out["exact"] = (r.q_lb == gt.q_star)
    out["ecart_reel"] = float((gt.q_star - r.q_lb) / abs(gt.q_star)) if gt.q_star else 0.0
    out["couverture"] = 100.0 * len(r.archive) / max(1, len(gt.E))
    return out


def main() -> None:
    # instances SUR LESQUELLES LA METHODE EXACTE A ECHOUE dans la campagne
    # (campaign.csv, statut 'limit'), triees par |E| decroissant.
    # m et rhs_scale reprennent exactement la grille de campaign.py :
    # m = max(3, n//2+1) et rhs = {5:1.8, 6:1.5, 7:1.2, 8:1.0}
    configs = [
        dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.00),   # |E| = 218
        dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.25),   # |E| = 198
        dict(n=6, m=4, p=4, seed=3, rhs_scale=1.5, corr=0.00),   # |E| = 185
        dict(n=7, m=4, p=3, seed=1, rhs_scale=1.2, corr=0.25),   # |E| = 112
        dict(n=5, m=3, p=4, seed=2, rhs_scale=1.8, corr=0.00),   # |E| =  54
        dict(n=8, m=5, p=3, seed=1, rhs_scale=1.0, corr=0.00),   # |E| =  46
        dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.50),   # |E| =  30
        dict(n=6, m=4, p=2, seed=1, rhs_scale=1.5, corr=0.25),   # |E| =  26
    ]

    print("=" * 96)
    print("PARTIE A - VALIDATION CONTRE LA VERITE TERRAIN")
    print("=" * 96)
    print(f"{'instance':<26}{'|E|':>6}{'V8':>5}{'V9':>5}{'V10':>5}"
          f"{'exact':>7}{'ecart reel':>12}{'ecart garanti':>15}{'couv%':>8}")
    print("-" * 96)

    rows = []
    for cfg in configs:
        inst = generate(**cfg)
        gt = ground_truth(inst, limit=200_000)
        reset_oracle_counter()
        r = matheuristic_P(inst, time_budget=8.0, bound_budget=6.0, seed=1)
        c = check(inst, gt, r)
        c["name"], c["E"] = inst.name, len(gt.E)
        c["gap"] = r.gap
        rows.append(c)
        g = f"{r.gap*100:>13.2f} %" if r.gap is not None else f"{'-':>15}"
        print(f"{inst.name:<26}{len(gt.E):>6}"
              f"{'ok' if c['V8'] else 'KO':>5}{'ok' if c['V9'] else 'KO':>5}"
              f"{'ok' if c['V10'] else 'KO':>5}"
              f"{'oui' if c['exact'] else 'non':>7}"
              f"{c['ecart_reel']*100:>11.2f} %{g}{c['couverture']:>8.1f}")

    allok = all(r["V8"] and r["V9"] and r["V10"] for r in rows)
    n_ex = sum(1 for r in rows if r["exact"])
    print("-" * 96)
    print(f"V8/V9/V10 : {'TOUT VALIDE' if allok else 'ECHEC'}   |   "
          f"optimum atteint : {n_ex}/{len(rows)}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("PARTIE B - MATHEURISTIQUE vs EXACT A BUDGET EGAL (regime |E| gros)")
    print("=" * 96)
    BUDGET = 12.0
    print(f"{'instance':<26}{'|E|':>6}{'q* (verite)':>13}"
          f"{'exact':>20}{'matheuristique':>22}")
    print(f"{'':<26}{'':>6}{'':>13}{'valeur':>11}{'statut':>9}"
          f"{'valeur':>11}{'ecart garanti':>11}")
    print("-" * 96)

    for cfg in configs:
        inst = generate(**cfg)
        gt = ground_truth(inst, limit=200_000)

        reset_oracle_counter()
        try:
            e = solve_P(inst, time_limit=BUDGET)
            e_val = float(e.q_star) if e.q_star is not None else float("nan")
            e_st = e.status
        except Exception as ex:
            e_val, e_st = float("nan"), type(ex).__name__

        reset_oracle_counter()
        m = matheuristic_P(inst, time_budget=BUDGET * 0.6,
                           bound_budget=BUDGET * 0.4, seed=1)
        m_val = float(m.q_lb)
        g = f"{m.gap*100:>9.2f} %" if m.gap is not None else f"{'-':>11}"

        mark = "" if abs(m_val - float(gt.q_star)) < 1e-9 else "  <- sous-optimal"
        print(f"{inst.name:<26}{len(gt.E):>6}{float(gt.q_star):>13.5f}"
              f"{e_val:>11.5f}{e_st:>9}{m_val:>11.5f}{g}{mark}")

    print("-" * 96)
    print("Lecture : le statut 'limit' de la colonne exacte signale une valeur")
    print("NON prouvee ; la matheuristique, elle, annonce toujours un ecart.")


if __name__ == "__main__":
    main()
