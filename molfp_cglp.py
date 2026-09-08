#!/usr/bin/env python3
"""
molfp_cglp.py
=============
COUPE DISJONCTIVE PAR PROGRAMME GENERATEUR (lift-and-project), sans binaire.

LE VERROU QU'ELLE VISE. La coupe des theoremes 4 et 6 retranche la region
{ x : Z(x) <= Z(a) } par la disjonction

        il existe k tel que  e_k^a(x) >= 1

modelisee par p variables BINAIRES. Ce cout est ce qui impose un plafond :
la mesure montre qu'au-dela d'une quarantaine de coupes, chacune alourdit le
programme entier plus qu'elle ne resserre le relache. A n >= 20 le plafond
sature, et les ecarts garantis restent a 92-98 % quoi qu'on fasse. Ce n'est
donc plus la DISPONIBILITE des coupes qui bloque, c'est leur COUT.

Une coupe sans binaire n'a pas de plafond. La forme agregee (cf.
`ECutModel.add_aggregated_cut`) en est une, mais elle est faible : mesuree,
elle retranche le quart au tiers de ce que retranche la disjonction sur les
memes points. La voie serieuse est le programme generateur de coupes.

--------------------------------------------------------------------------
LA CONSTRUCTION
--------------------------------------------------------------------------
Soit P = { x : A x <= b, 0 <= x <= u } le relache courant en espace x, et
soient les p disjoints

        P_k = { x dans P : gamma_k^T x >= 1 - delta_k }

ou e_k^a(x) = gamma_k^T x + delta_k. Tout point efficace autre que ceux de
meme vecteur criteres que `a` appartient a la REUNION des P_k. Une
inegalite valide pour cette reunion l'est donc pour E.

On cherche alpha^T x >= beta valide sur CHAQUE P_k. Par le lemme de Farkas,
en ecrivant P_k sous forme « >= » -- -A x >= -b et gamma_k^T x >= 1 -
delta_k, avec x >= 0 -- cela equivaut a l'existence de multiplicateurs
lambda^k >= 0 et mu^k >= 0 tels que

        alpha  >=  -lambda^k A + mu^k gamma_k          (composante par composante)
        beta   <=  -lambda^k b + mu^k (1 - delta_k)

Ces conditions portent sur le MEME couple (alpha, beta) pour tous les k :
c'est ce qui rend l'inegalite valide pour la reunion. On choisit alors
(alpha, beta) le plus VIOLE possible au point courant x*, ce qui donne le
programme lineaire

        max   beta - alpha^T x*
        s.c.  alpha_j + (lambda^k A)_j - mu^k gamma_{k,j} >= 0   pour tout k, j
              beta + lambda^k b - mu^k (1 - delta_k) <= 0        pour tout k
              somme des multiplicateurs = 1                      (normalisation)
              lambda^k >= 0,  mu^k >= 0,  alpha et beta libres

Si l'optimum est strictement positif, alpha^T x >= beta est une inegalite
valide qui COUPE x*. Une ligne, aucune binaire, un seul PL pour l'obtenir.

--------------------------------------------------------------------------
CE QU'ELLE NE PEUT PAS FAIRE
--------------------------------------------------------------------------
Elle ne domine pas la forme disjonctive : celle-ci retranche exactement la
region, la coupe generee n'en retranche qu'un demi-espace. Son interet est
ailleurs -- elle ne consomme aucune binaire, donc aucune place sous le
plafond, et peut s'ajouter en nombre. C'est le seul levier disponible dans
le regime ou le plafond sature.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog

from molfp_core import INF, ORACLE_CALLS, feasibility_rows
from molfp_instance import MOILFP

Row = Tuple[np.ndarray, float, float]


def _rows_en_Ax_le_b(rows: Sequence[Row], n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Met les lignes (coef, lo, hi) sous la forme unique A x <= b."""
    A, b = [], []
    for coef, lo, hi in rows:
        c = np.asarray(coef, dtype=float)[:n]
        if np.isfinite(hi):
            A.append(c)
            b.append(float(hi))
        if np.isfinite(lo):
            A.append(-c)
            b.append(-float(lo))
    return (np.array(A) if A else np.zeros((0, n))), np.array(b)


def optimum_relaxation(coef: np.ndarray, rows: Sequence[Row],
                       ub: np.ndarray) -> Optional[np.ndarray]:
    """argmax de coef^T x sur la relaxation continue. None si echec."""
    n = len(coef)
    A, b = _rows_en_Ax_le_b(rows, n)
    ORACLE_CALLS["lp"] += 1
    res = linprog(c=-np.asarray(coef, dtype=float),
                  A_ub=A if A.size else None, b_ub=b if A.size else None,
                  bounds=[(0.0, float(u)) for u in ub], method="highs")
    return np.asarray(res.x, dtype=float) if res.success else None


def cglp_cut(inst: MOILFP, a: np.ndarray, rows: Sequence[Row],
             ub: np.ndarray, x_star: np.ndarray,
             tol: float = 1e-7) -> Optional[Tuple[np.ndarray, float]]:
    """
    Programme generateur de coupes. Renvoie (alpha, beta) tel que
    alpha^T x >= beta soit valide pour la reunion des P_k et VIOLEE en
    x_star, ou None si aucune coupe violee n'existe.

    Le cout est d'UN programme lineaire. Sa taille : n + 1 + p(m+1)
    variables, p*n + p + 1 contraintes -- soit une centaine de chaque a
    n = 40, m = 21, p = 3.
    """
    from molfp_matheuristic import e_row

    n, p = inst.n, inst.p
    A, b = _rows_en_Ax_le_b(rows, n)
    # les bornes de boite font partie du polyedre : x <= u
    A = np.vstack([A, np.eye(n)]) if A.size else np.eye(n)
    b = np.concatenate([b, ub.astype(float)])
    m = A.shape[0]

    gam, dlt = [], []
    for k in range(p):
        g, d = e_row(inst, a, k)
        gam.append(np.asarray(g, dtype=float))
        dlt.append(float(d))

    # variables : alpha (n, libre) | beta (1, libre) | lambda^k (m) | mu^k (1)
    nv = n + 1 + p * (m + 1)
    def i_lam(k): return n + 1 + k * (m + 1)
    def i_mu(k): return n + 1 + k * (m + 1) + m

    A_ub, b_ub = [], []
    for k in range(p):
        # -alpha_j - (lambda^k A)_j + mu^k gamma_{k,j} <= 0
        for j in range(n):
            row = np.zeros(nv)
            row[j] = -1.0
            row[i_lam(k):i_lam(k) + m] = -A[:, j]
            row[i_mu(k)] = gam[k][j]
            A_ub.append(row)
            b_ub.append(0.0)
        # beta + lambda^k b - mu^k (1 - delta_k) <= 0
        row = np.zeros(nv)
        row[n] = 1.0
        row[i_lam(k):i_lam(k) + m] = b
        row[i_mu(k)] = -(1.0 - dlt[k])
        A_ub.append(row)
        b_ub.append(0.0)

    # normalisation : somme des multiplicateurs = 1 (sinon non borne)
    A_eq = np.zeros((1, nv))
    for k in range(p):
        A_eq[0, i_lam(k):i_lam(k) + m + 1] = 1.0

    c = np.zeros(nv)
    c[:n] = np.asarray(x_star, dtype=float)     # on MINIMISE alpha^T x* - beta
    c[n] = -1.0
    bounds = [(None, None)] * (n + 1) + [(0.0, None)] * (p * (m + 1))

    ORACLE_CALLS["lp"] += 1
    res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=np.array([1.0]), bounds=bounds,
                  method="highs")
    if not res.success:
        return None
    violation = -float(res.fun)                 # = beta - alpha^T x*
    if violation <= tol:
        return None
    alpha = np.asarray(res.x[:n], dtype=float)
    beta = float(res.x[n])
    return alpha, beta


def verifier_validite(inst: MOILFP, a: np.ndarray,
                      alpha: np.ndarray, beta: float,
                      points: Sequence[np.ndarray],
                      tol: float = 1e-6) -> int:
    """
    Compte les points de `points` qui VIOLENT alpha^T x >= beta alors qu'ils
    devraient la satisfaire, c'est-a-dire ceux qui verifient la disjonction
    (il existe k tel que e_k^a(x) >= 1). Doit valoir 0.

    C'est le controle de surete de la coupe : on ne peut pas se contenter de
    la derivation, il faut la confronter aux points que l'on pretend garder.
    """
    from molfp_matheuristic import e_row

    gam, dlt = [], []
    for k in range(inst.p):
        g, d = e_row(inst, a, k)
        gam.append(np.asarray(g, dtype=float))
        dlt.append(float(d))
    viol = 0
    for x in points:
        xv = np.asarray(x, dtype=float)
        dans_union = any(gam[k] @ xv + dlt[k] >= 1 - 1e-9
                         for k in range(inst.p))
        if dans_union and float(alpha @ xv) < beta - tol:
            viol += 1
    return viol
