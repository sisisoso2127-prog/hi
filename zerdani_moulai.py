#!/usr/bin/env python3
"""
zerdani_moulai.py
=================
Reimplementation de l'algorithme de Zerdani & Moulai (2011),
« Optimization over an Integer Efficient Set of a Multiple Objective Linear
Fractional Problem », Applied Mathematical Sciences 5(50), 2451-2466.

    (P_E)  max  phi = d x    sur E(P), l'ensemble EFFICACE ENTIER d'un
                              MOILFP a criteres FRACTIONNAIRES.

POURQUOI LA REIMPLEMENTER. C'est la seule facon de comparer honnetement :
leur methode ne resout pas notre probleme (leur phi est LINEAIRE, notre f
est fractionnaire), donc la comparaison ne peut se faire que sur LEUR
terrain, et sur leur terrain il faut leur methode. Elle ne se lit pas dans
une valeur mais dans un TABLEAU du simplexe -- ensemble hors base, gradient
reduit, colonnes, aretes -- qu'aucun solveur industriel n'expose. D'ou
`simplex_rationnel.py`.

CE QUI EST REPRIS TEL QUEL
  Th. 2.1   optimalite de (P1(S)) : gamma_{k,j} <= 0 pour tout j hors base.
  Cor. 2.3  si J_k = {j : gamma_{k,j} = 0} est VIDE, l'optimum de (P1(S))
            est unique, donc EFFICACE. C'est le test le moins cher de la
            methode, et il ne coute rien : il se lit dans le tableau.
  Def. 3.1  aretes E_{jk} incidentes au sommet, et pas entier
            theta^0 = partie entiere de min_i { x_{k,i} / y_{k,i,jk} }.
  Eq. (4)   upsilon_k = d_{jk} - somme_i d_i y_{k,i,jk}, le cout reduit de
            phi le long de l'arete. Sur Gamma_k on a upsilon_k >= 0, donc
            phi croit le long de l'arete et atteint son maximum en theta^0.
  Th. 3.4   coupe de Dantzig  somme_{j hors base} x_j >= 1.
  Etapes 0 a 3, terminaison sur infaisabilite.

LES DEUX ENDROITS OU NOUS AVONS DU TRANCHER, ET COMMENT
  1. TEST D'EFFICACITE. Les auteurs invoquent le theoreme 3.6 (egalite de
     l'intersection des ensembles de niveau et des courbes de niveau). Nous
     employons notre test integral, qui decide le MEME predicat -- « x
     est-il Pareto optimal ? » -- de facon exacte et sans tolerance. Ce
     n'est pas une approximation : c'est le meme enonce, calcule autrement.
  2. SOMMET FRACTIONNAIRE. Aux etapes 3.2 et 3.3 les auteurs ecrivent
     « using the dual simplex method and Gomory cuts, if necessary ». Le
     detail de ces coupes n'est pas donne. Plutot que d'inventer une
     variante et de la leur attribuer, nous nous ARRETONS avec le statut
     `sommet_fractionnaire` et le comptons comme tel dans les mesures. Le
     taux d'instances menees a terme est ainsi rapporte au lieu d'etre
     masque.

Une lecture a fixer, l'OCR de l'article etant ambigu a l'etape 3(b) : il y
est ecrit deux fois « si upsilon_k = 0 » pour deux branches opposees. Comme
upsilon_k >= 0 sur Gamma_k et que phi croit le long de l'arete a raison de
upsilon_k par pas, la seule lecture coherente est : upsilon_k > 0 -> explorer
l'arete ; upsilon_k = 0 -> phi est constante dessus, donc rien a y gagner si
le sommet est deja efficace et au moins aussi bon que l'incumbent -> passer
a l'arete suivante. C'est la lecture retenue, et elle est signalee ici.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Tuple

import numpy as np

from molfp_core import ORACLE_CALLS, efficiency_test, feasibility_rows, solve_ilp
from molfp_instance import MOILFP
from simplex_rationnel import (Tableau, _F, gradient_reduit_fractionnaire,
                               resoudre_fractionnaire, tableau_phase_un)

Frac = Fraction


@dataclass
class ZMResult:
    """Sortie de l'algorithme. `status` dit ce qui s'est reellement passe."""
    phi_opt: Optional[Frac] = None
    x_opt: Optional[np.ndarray] = None
    status: str = "optimal"          # optimal | sommet_fractionnaire |
                                     # infaisable | limite
    iterations: int = 0
    coupes: int = 0
    tests_efficacite: int = 0
    ilp_calls: int = 0
    arretes_explorees: int = 0


def _phi(d: np.ndarray, x) -> Frac:
    return sum(_F(int(d[j])) * _F(x[j]) for j in range(len(d)))


def _est_efficace(inst: MOILFP, x: np.ndarray) -> Optional[bool]:
    """Th. 3.6 des auteurs, decide par notre test integral exact."""
    r = efficiency_test(inst, np.asarray(x, dtype=int))
    return r.efficient


def _point_arete(tab: Tableau, jk: int, theta: Frac, n: int) -> List[Frac]:
    """
    Point de l'arete E_{jk} au pas theta : la variable hors base jk prend la
    valeur theta, les autres hors base restent nulles, et chaque variable de
    base i devient x_{k,i} - theta * y_{k,i,jk} (definition 3.1).
    """
    v = [Frac(0)] * tab.nvar
    v[jk] = theta
    for i, jb in enumerate(tab.basis):
        v[jb] = tab.rhs[i] - theta * tab.T[i][jk]
    return v[:n]


def zerdani_moulai(inst: MOILFP, max_iter: int = 200,
                   verbose: bool = False) -> ZMResult:
    """
    phi doit etre LINEAIRE : c'est le probleme que traitent les auteurs.
    On la lit dans `inst.f`, dont le denominateur doit etre constant.
    """
    if np.any(inst.f.den != 0):
        raise ValueError("Zerdani & Moulai : phi doit etre LINEAIRE "
                         "(denominateur constant).")
    d = np.asarray(inst.f.num, dtype=int)
    n = inst.n
    calls0 = ORACLE_CALLS["ilp"]
    res = ZMResult()

    # ---- Etape 0 : relaxation, max phi sur S (entier) -------------------
    A = np.array(inst.A, dtype=int)
    b = np.array(inst.b, dtype=int)
    r0 = solve_ilp(d.astype(float), feasibility_rows(inst),
                   inst.var_upper_bounds(), maximize=True)
    if not r0.ok:
        res.status = "infaisable"
        res.ilp_calls = ORACLE_CALLS["ilp"] - calls0
        return res
    x0 = np.asarray(r0.x, dtype=int)

    # ---- Etape 1 : x0 efficace ? -> c'est fini --------------------------
    res.tests_efficacite += 1
    if _est_efficace(inst, x0):
        res.phi_opt, res.x_opt = _phi(d, [_F(v) for v in x0]), x0
        res.ilp_calls = ORACLE_CALLS["ilp"] - calls0
        return res

    phi_opt: Optional[Frac] = None
    x_opt: Optional[np.ndarray] = None
    Z1 = inst.Z[0]

    for k in range(1, max_iter + 1):
        res.iterations = k
        # ---- Etape 2 : resoudre (P1(S)) sur la region courante ----------
        tab = tableau_phase_un(A, b)
        if tab is None:
            break                       # etape terminale : region vide
        p = [_F(v) for v in Z1.num] + [Frac(0)] * (tab.nvar - n)
        q = [_F(v) for v in Z1.den] + [Frac(0)] * (tab.nvar - n)
        st = resoudre_fractionnaire(tab, p, _F(Z1.a), q, _F(Z1.b))
        if st != "optimal":
            res.status = "limite"
            break
        if not tab.est_entiere():
            # les auteurs prescrivent ici « dual simplex et coupes de Gomory
            # si necessaire », sans en donner le detail : on s'arrete et on
            # le dit, plutot que d'inventer une variante et la leur attribuer
            res.status = "sommet_fractionnaire"
            break

        gamma, _, _ = gradient_reduit_fractionnaire(tab, p, _F(Z1.a),
                                                    q, _F(Z1.b))
        hors_base = tab.nonbasic()
        x1 = np.array([int(v) for v in tab.x()], dtype=int)
        Jk = [j for j in hors_base if gamma[j] == 0]

        # 2.1 / 2.2 : J_k vide => unique => efficace (cor. 2.3), sinon test
        if not Jk:
            efficace = True             # lu dans le tableau, sans ILP
        else:
            res.tests_efficacite += 1
            efficace = bool(_est_efficace(inst, x1))
        if efficace:
            v = _phi(d, [_F(int(t)) for t in x1])
            if phi_opt is None or v > phi_opt:
                phi_opt, x_opt = v, x1

        # ---- Etape 3 : aretes issues du sommet -------------------------
        de = np.concatenate([d, np.zeros(tab.nvar - n, dtype=int)])
        ups = {j: _F(int(de[j])) - sum(_F(int(de[jb])) * tab.T[i][j]
                                       for i, jb in enumerate(tab.basis))
               for j in hors_base}
        Gamma = [j for j in hors_base if gamma[j] <= 0 and ups[j] >= 0]

        ameliore = False
        candidats = list(Gamma)
        while candidats:
            jk = candidats.pop(0)
            ratios = [tab.rhs[i] / tab.T[i][jk]
                      for i in range(tab.m) if tab.T[i][jk] > 0]
            if not ratios:
                continue                # arete non bornee
            theta0 = int(min(ratios))   # partie entiere
            if theta0 == 0:
                continue                # aucun point entier sur cette arete
            # lecture de l'etape 3(b), cf. en-tete
            if efficace and phi_opt is not None and \
                    _phi(d, [_F(int(t)) for t in x1]) >= phi_opt and \
                    ups[jk] == 0:
                continue                # phi constante le long de l'arete
            for theta in range(theta0, 0, -1):
                pt = _point_arete(tab, jk, Frac(theta), n)
                if any(v.denominator != 1 or v < 0 for v in pt):
                    continue
                xu = np.array([int(v) for v in pt], dtype=int)
                res.arretes_explorees += 1
                res.tests_efficacite += 1
                if _est_efficace(inst, xu):
                    v = _phi(d, [_F(int(t)) for t in xu])
                    if phi_opt is None or v > phi_opt:
                        phi_opt, x_opt, ameliore = v, xu, True
                    break
            if ameliore:
                break

        # ---- 3.2 / 3.3 : couper, puis recommencer ----------------------
        # somme_{j hors base} x_j >= 1, exprimee dans les variables x en
        # substituant les ecarts s_i = b_i - A_i x.
        coef = np.zeros(n, dtype=int)
        const = 0
        for j in hors_base:
            if j < n:
                coef[j] += 1
            else:
                i = j - n
                if i < A.shape[0]:
                    coef -= A[i]
                    const += int(b[i])
        # coef . x + const >= 1   <=>   -coef . x <= const - 1
        A = np.vstack([A, -coef])
        b = np.append(b, const - 1)
        res.coupes += 1
        if verbose:
            print(f"  k={k} x1={x1.tolist()} phi_opt={phi_opt} "
                  f"|Gamma|={len(Gamma)} coupes={res.coupes}")
    else:
        res.status = "limite"

    res.phi_opt, res.x_opt = phi_opt, x_opt
    res.ilp_calls = ORACLE_CALLS["ilp"] - calls0
    return res


if __name__ == "__main__":
    from literature import zerdani_moulai_example
    inst = zerdani_moulai_example()
    r = zerdani_moulai(inst, verbose=True)
    print(f"\nstatut {r.status}   phi_opt = {r.phi_opt}   "
          f"x_opt = {None if r.x_opt is None else r.x_opt.tolist()}")
    print(f"iterations {r.iterations}  coupes {r.coupes}  "
          f"tests d'efficacite {r.tests_efficacite}  ILP {r.ilp_calls}")
    print(f"PUBLIE : phi_opt = 6 en (3, 0)   -> "
          f"{'CONFORME' if r.phi_opt == 6 else '*** ECART ***'}")
