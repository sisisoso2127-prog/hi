#!/usr/bin/env python3
"""
illustration.py
===============
Reproduit INTEGRALEMENT les deux illustrations numeriques de `doc/hybride.tex`
(la coupe d'efficacite) : instances I2 (2 variables) et I3 (3 variables).

Tout chiffre imprime ici figure tel quel dans l'article. Rien n'est recopie a
la main : les tableaux, les trajectoires de borne et les coordonnees des
figures sortent de ce script.

Sortie :
  * S, E, q*, max_S f par enumeration exacte (verite terrain)
  * l'ordre du vivier, tel que `build_cut_pool` le produit
  * pour chaque coupe posee : |R|, U, D+, et la borne du Th. 2
  * la condition de cloture, un ILP par point d'archive
  * l'execution reelle de la methode, avec et sans coupes d'archive

Usage :  python illustration.py
"""

import sys
from fractions import Fraction

import numpy as np

from molfp_instance import MOILFP, FracObj
from molfp_enum import ground_truth
from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_matheuristic import (matheuristic_P, same_criteria_improves,
                                surrogate)


# ----------------------------------------------------------------------------
# Les deux instances de l'article
# ----------------------------------------------------------------------------

def _build(A, b, Zs, fo, name):
    return MOILFP(
        A=np.array(A), b=np.array(b),
        Z=[FracObj(np.array(c), a, np.array(d), bb) for c, a, d, bb in Zs],
        f=FracObj(np.array(fo[0]), fo[1], np.array(fo[2]), fo[3]),
        name=name, seed=0)


def instance_I2() -> MOILFP:
    """
    max Z1 = (3 x2 + 2) / (x1 + x2 + 3),  Z2 = (x1 + 3 x2) / (2 x2 + 3)
    max f  = (3 x1 + x2) / (2 x2 + 1)   sur E
    s.c.   3 x1 +   x2 <= 14
           2 x1 + 3 x2 <= 15,  x entier >= 0
    |S| = 22, |E| = 5, q* = 14/5 en (4,2), max_S f = 12 en (4,0) -- domine.
    """
    return _build([[3, 1], [2, 3]], [14, 15],
                  [([0, 3], 2, [1, 1], 3), ([1, 3], 0, [0, 2], 3)],
                  ([3, 1], 0, [0, 2], 1), "I2")


def instance_I3() -> MOILFP:
    """
    max Z1 = 3 x3 / (2 x2 + x3 + 2)
        Z2 = (3 x1 + 3 x2 + x3 + 1) / (x1 + 2 x2 + x3 + 3)
    max f  = (2 x1 + 2 x2 + x3 + 1) / (x1 + x3 + 1)   sur E
    s.c.   3 x1 + x2 +   x3 <= 8
             x1 + 2 x2 + 2 x3 <= 9,  x entier >= 0
    |S| = 36, |E| = 5, q* = 3 en (2,2,0), max_S f = 9 en (0,4,0) -- domine.
    """
    return _build([[3, 1, 1], [1, 2, 2]], [8, 9],
                  [([0, 0, 3], 0, [0, 2, 1], 2), ([3, 3, 1], 1, [1, 2, 1], 3)],
                  ([2, 2, 1], 1, [1, 0, 1], 1), "I3")


# ----------------------------------------------------------------------------
# Outils d'enumeration : ce que la METHODE ne fait pas, mais qui permet de la
# verifier point par point sur une instance minuscule.
# ----------------------------------------------------------------------------

def in_relaxation(inst: MOILFP, x, cuts) -> bool:
    """x survit-il a toutes les coupes posees ? (disjonction du Th. 1 / brique 3)"""
    zx = inst.criteria(x)
    for a in cuts:
        za = inst.criteria(np.asarray(a, dtype=int))
        if not any(zx[k] > za[k] for k in range(inst.p)):
            return False
    return True


def bound_th2(inst: MOILFP, R, q: Fraction):
    """
    Les quantites EXACTES du Th. 2 sur un relache R enumere :
        U  = max_{R} { Q N(x) - P D(x) },   D+ = min D sur la zone active,
        borne = q + U / (Q D+).
    Renvoie (U, D+, borne) ; borne vaut q si aucun point de R ne depasse q.
    """
    P, Q = q.numerator, q.denominator
    if not R:
        return None, None, None                      # relache VIDE
    vals = [(Q * inst.f.numerator(x) - P * inst.f.denominator(x),
             inst.f.denominator(x)) for x in R]
    U = max(0, max(v for v, _ in vals))
    pos = [d for v, d in vals if v >= 0]
    if not pos:
        return U, None, q
    Dp = min(pos)
    return U, Dp, q + Fraction(U, Q * Dp)


def _fmt(x) -> str:
    return "(" + ",".join(str(int(v)) for v in x) + ")"


# ----------------------------------------------------------------------------
# Le deroule de l'article, section par section
# ----------------------------------------------------------------------------

def illustrate(inst: MOILFP) -> None:
    print("=" * 78)
    print(f"INSTANCE {inst.name}   n = {inst.n}, m = {inst.m}, p = {inst.p}")
    print("=" * 78)
    inst.check_assumptions()

    gt = ground_truth(inst, limit=50_000)
    q_star = gt.q_star
    E_keys = {tuple(int(v) for v in x) for x in gt.E}
    print(f"  |S| = {len(gt.S)}   |E| = {len(gt.E)}   "
          f"q* = {q_star} = {float(q_star):.4f} en {_fmt(gt.x_star)}")
    x_maxS = max(gt.S, key=lambda x: inst.f.value(x))
    print(f"  max_S f = {gt.q_max_S} en {_fmt(x_maxS)} "
          f"({'EFFICACE' if tuple(int(v) for v in x_maxS) in E_keys else 'DOMINE'})"
          f"  -> ecart de la borne naive : "
          f"{float((gt.q_max_S - q_star) / gt.q_max_S) * 100:.0f} %")

    # -- etape 1 : la recherche ---------------------------------------------
    print("\n-- Etape 1 : la recherche --------------------------------------")
    res = matheuristic_P(inst, time_budget=3.0, bound_budget=2.0, seed=0,
                         archive_cuts=True, cut_batch=40, cert_rounds=2)
    archive = list(res.archive)
    print(f"  archive rendue ({len(archive)} points, dans l'ordre) : "
          + "  ".join(_fmt(x) for x in archive))
    print(f"  incumbent q = {res.q_lb}")
    print(f"  points DOMINES collectes par la recherche : "
          f"{len(res.cut_points)}   <- le verrou de la section 3")

    # -- etape 2 : l'ordre du vivier ----------------------------------------
    q = q_star
    w, _, _, _ = surrogate(inst, q)
    pool = sorted(archive, key=lambda x: -float(w @ x))
    dominated_all = sorted([x for x in gt.S
                            if tuple(int(v) for v in x) not in E_keys],
                           key=lambda x: -float(w @ x))
    print("\n-- Etape 2 : l'ordre du vivier (substitut decroissant) ---------")
    print(f"  w = Qc - Pd = {[int(v) for v in w]}")
    print("  archive  : " + "  ".join(_fmt(x) for x in pool))

    # -- etape 3 : la cloture ------------------------------------------------
    print("\n-- Etape 3 : la cloture, UN ILP par point ----------------------")
    reset_oracle_counter()
    for a in pool:
        r = same_criteria_improves(inst, np.asarray(a, dtype=int), q)
        verdict = "FAISABLE (on empoche le point)" if r is not None \
            else "infaisable -> cloture etablie, coupe licite"
        print(f"  a = {_fmt(a):>10} : {verdict}")
    print(f"  cout total : {ORACLE_CALLS['ilp']} appels au solveur entier")

    # -- etapes 4 et 5 : les deux trajectoires -------------------------------
    for titre, source, dispo in (
            ("COUPE D'EFFICACITE (source : archive)", pool, len(pool)),
            ("coupe de dominance (source : reparations)", dominated_all, 0)):
        print(f"\n-- {titre} --")
        print(f"   {'#':>2} {'coupe en':>10} {'|R|':>4} {'U':>5} {'D+':>4} "
              f"{'borne':>10}")
        cuts, R = [], list(gt.S)
        U, Dp, bd = bound_th2(inst, R, q)
        print(f"   {0:>2} {'--':>10} {len(R):>4} {U:>5} {str(Dp):>4} {str(bd):>10}")
        for i, a in enumerate(source, 1):
            cuts.append(a)
            R = [x for x in gt.S if in_relaxation(inst, x, cuts)]
            if not R:
                print(f"   {i:>2} {_fmt(a):>10} {0:>4}   RELACHE VIDE -> "
                      f"q* = q = {q}   (Proposition 3)")
                break
            U, Dp, bd = bound_th2(inst, R, q)
            flag = "  <- U = 0, optimalite prouvee" if U == 0 and i and \
                bd == q else ""
            print(f"   {i:>2} {_fmt(a):>10} {len(R):>4} {U:>5} {str(Dp):>4} "
                  f"{str(bd):>10}{flag}")
            if i >= 8:
                print("   ...")
                break
        print(f"   points reellement DISPONIBLES pour cette source : {dispo}")

    # -- ce que la methode rend ---------------------------------------------
    print("\n-- Ce que la methode rend reellement --------------------------")
    for arch_cuts in (False, True):
        r = matheuristic_P(inst, time_budget=3.0, bound_budget=2.0, seed=0,
                           archive_cuts=arch_cuts, cut_batch=40, cert_rounds=2)
        assert r.q_lb <= q_star, "borne inferieure INVALIDE"
        assert r.q_ub is None or r.q_ub >= float(q_star) - 1e-9, \
            "borne superieure INVALIDE"
        print(f"  archive_cuts={str(arch_cuts):<5} : q_lb = {r.q_lb}   "
              f"q_ub = {r.q_ub}   prouve = {r.proved_optimal}   "
              f"coupes d'efficacite posees = {r.cert.get('archive_cuts', 0)}")
    print("  VALIDITE q_lb <= q* <= q_ub : verifiee sur les deux variantes")
    print()


def main() -> None:
    for inst in (instance_I2(), instance_I3()):
        illustrate(inst)


if __name__ == "__main__":
    sys.exit(main())
