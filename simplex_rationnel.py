#!/usr/bin/env python3
"""
simplex_rationnel.py
====================
Simplexe primal en arithmetique RATIONNELLE EXACTE, exposant son tableau.

POURQUOI IL FAUT L'ECRIRE. La methode de Zerdani & Moulai (2011) ne se lit
pas dans la valeur d'un optimum : elle se lit dans le TABLEAU. Ses objets
sont l'ensemble hors base N_k, le gradient reduit gamma_{k,j}, les colonnes
y_{k,ij}, les aretes E_{jk} qu'elles definissent, et le pas entier
theta_{jk} le long de ces aretes. Aucun solveur industriel n'expose cela --
HiGHS rend une solution, pas une base. Comparer honnetement a leur methode
exige donc de la reimplementer, et la reimplementer exige ce simplexe.

POURQUOI EN RATIONNELS. Les tests de la methode portent sur des egalites :
« gamma_{k,j} = 0 » decide si l'optimum est unique (donc efficace par le
corollaire 2.3), « theta entier » decide s'il existe un point entier sur une
arete. En flottant ces tests demandent un epsilon, et un epsilon mal choisi
change le resultat de l'algorithme, pas seulement sa precision. Les donnees
etant entieres, tout le tableau reste rationnel : on garde l'exactitude sans
rien supposer.

OBJECTIF FRACTIONNAIRE. Pour max (p x + alpha) / (q x + beta) sur un
polyedre, le gradient reduit est celui de Martos, et c'est exactement la
forme employee par les auteurs (section 2) :

    gamma_j = D * (p_j - zp_j)  -  N * (q_j - zq_j)

ou N et D sont les valeurs du numerateur et du denominateur au sommet
courant, et zp_j, zq_j les couts reduits usuels des deux formes lineaires
prises separement. On entre la variable j des que gamma_j > 0 ; on est
optimal quand gamma_j <= 0 pour tout j hors base (leur theoreme 2.1).

REGLE ANTI-CYCLAGE. Regle de Bland : plus petit indice eligible a l'entree
comme a la sortie. Plus lente que la regle du plus grand gain, mais elle
garantit la terminaison sans perturbation -- et en rationnels exacts la
degenerescence est frequente sur ces petites instances a donnees entieres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

Frac = Fraction


def _F(x) -> Frac:
    return x if isinstance(x, Frac) else Frac(int(x))


@dataclass
class Tableau:
    """
    Tableau du simplexe pour  max c x  s.c.  A x <= b,  x >= 0,  b >= 0.

    Les variables 0..n-1 sont les variables de decision, n..n+m-1 les
    variables d'ecart. `T[i][j]` est la colonne j exprimee dans la base
    courante -- c'est le y_{k,ij} des auteurs -- et `rhs[i]` la valeur de la
    i-eme variable de base.
    """
    n: int                      # variables de decision
    m: int                      # contraintes
    T: List[List[Frac]]         # m x (n+m)
    rhs: List[Frac]             # m
    basis: List[int]            # m indices de variables de base

    # -- lecture ----------------------------------------------------------
    @property
    def nvar(self) -> int:
        return len(self.T[0]) if self.T else self.n + self.m

    def nonbasic(self) -> List[int]:
        """N_k : indices hors base, dans l'ordre croissant."""
        en_base = set(self.basis)
        return [j for j in range(self.nvar) if j not in en_base]

    def solution(self) -> List[Frac]:
        """Valeurs de TOUTES les variables (decision puis ecarts)."""
        x = [Frac(0)] * self.nvar
        for i, jb in enumerate(self.basis):
            x[jb] = self.rhs[i]
        return x

    def x(self) -> List[Frac]:
        """Valeurs des seules variables de decision."""
        return self.solution()[:self.n]

    def est_entiere(self) -> bool:
        return all(v.denominator == 1 for v in self.x())

    # -- pivotage ---------------------------------------------------------
    def pivot(self, ligne: int, col: int) -> None:
        p = self.T[ligne][col]
        if p == 0:
            raise ZeroDivisionError("pivot nul")
        self.T[ligne] = [v / p for v in self.T[ligne]]
        self.rhs[ligne] = self.rhs[ligne] / p
        for i in range(self.m):
            if i == ligne:
                continue
            f = self.T[i][col]
            if f != 0:
                self.T[i] = [a - f * b for a, b in zip(self.T[i],
                                                       self.T[ligne])]
                self.rhs[i] = self.rhs[i] - f * self.rhs[ligne]
        self.basis[ligne] = col

    def couts_reduits(self, c: Sequence[Frac]) -> List[Frac]:
        """c_j - z_j pour toute variable, avec c de taille nvar."""
        cb = [c[jb] for jb in self.basis]
        out = []
        for j in range(self.nvar):
            zj = sum(cb[i] * self.T[i][j] for i in range(self.m))
            out.append(c[j] - zj)
        return out

    def ratio_min(self, col: int) -> Optional[int]:
        """Regle de Bland cote sortie : plus petit indice de base eligible."""
        best, best_ratio = None, None
        for i in range(self.m):
            if self.T[i][col] > 0:
                r = self.rhs[i] / self.T[i][col]
                if best_ratio is None or r < best_ratio or \
                        (r == best_ratio and self.basis[i] < self.basis[best]):
                    best, best_ratio = i, r
        return best


def tableau_initial(A: np.ndarray, b: np.ndarray) -> Tableau:
    """
    Base initiale des variables d'ecart. Exige b >= 0 -- c'est le cas de
    toutes les instances traitees ici, et cela evite une phase I.
    """
    m, n = A.shape
    if any(int(v) < 0 for v in b):
        raise ValueError("b >= 0 requis (pas de phase I dans cette version)")
    T = [[_F(A[i][j]) for j in range(n)] +
         [Frac(1) if k == i else Frac(0) for k in range(m)]
         for i in range(m)]
    return Tableau(n=n, m=m, T=T, rhs=[_F(v) for v in b],
                   basis=[n + i for i in range(m)])


def tableau_phase_un(A: np.ndarray, b: np.ndarray) -> Optional[Tableau]:
    """
    Base realisable initiale pour A x <= b, x >= 0, avec b DE SIGNE
    QUELCONQUE. Renvoie None si le polyedre est vide.

    POURQUOI ELLE EST INDISPENSABLE ICI. La methode de Zerdani & Moulai
    ajoute des coupes de la forme « somme x_j >= 1 », soit apres passage en
    forme <= un second membre NEGATIF. La base des ecarts n'est alors plus
    realisable, et les auteurs prescrivent le simplexe DUAL. Une phase I
    resout le meme probleme et se verifie plus simplement : on rend une base
    realisable, ou la preuve que le polyedre est vide.

    Procede : les lignes a b_i < 0 sont multipliees par -1 (le second membre
    devient positif, l'ecart prend le coefficient -1), et l'on adjoint a
    chacune une variable artificielle de coefficient +1 pour former la base.
    On minimise leur somme ; l'optimum vaut 0 si et seulement si le polyedre
    est non vide. Les artificielles restant en base a valeur nulle sont
    ensuite pivotees hors base, ou leur ligne est redondante et supprimee.
    """
    m, n = A.shape
    lignes, rhs, signes = [], [], []
    for i in range(m):
        if int(b[i]) < 0:
            lignes.append([-_F(A[i][j]) for j in range(n)])
            rhs.append(-_F(b[i]))
            signes.append(-1)
        else:
            lignes.append([_F(A[i][j]) for j in range(n)])
            rhs.append(_F(b[i]))
            signes.append(1)

    # colonnes : n decision, m ecarts (coefficient = signe), k artificielles
    art = [i for i in range(m) if signes[i] < 0]
    nvar = n + m + len(art)
    T = []
    for i in range(m):
        row = lignes[i] + [Frac(0)] * (m + len(art))
        row[n + i] = _F(signes[i])
        T.append(row)
    basis = []
    for pos, i in enumerate(art):
        T[i][n + m + pos] = Frac(1)
    for i in range(m):
        basis.append(n + m + art.index(i) if signes[i] < 0 else n + i)

    tab = Tableau(n=n, m=m, T=T, rhs=rhs, basis=basis)
    tab.n = nvar - m if False else n          # `n` reste le nombre de x
    # objectif de phase I : minimiser la somme des artificielles
    cout = [Frac(0)] * nvar
    for pos in range(len(art)):
        cout[n + m + pos] = Frac(-1)          # on MAXIMISE -somme
    st = resoudre_lineaire(tab, cout)
    if st != "optimal":
        return None
    x = tab.solution()
    if sum(x[n + m + pos] for pos in range(len(art))) != 0:
        return None                            # polyedre VIDE

    # chasser les artificielles restees en base (a valeur nulle)
    for i in range(m):
        if tab.basis[i] >= n + m:
            remplacant = next((j for j in range(n + m)
                               if tab.T[i][j] != 0), None)
            if remplacant is None:
                continue                        # ligne redondante
            tab.pivot(i, remplacant)
    # retirer les colonnes artificielles
    tab.T = [row[:n + m] for row in tab.T]
    return tab


def resoudre_lineaire(tab: Tableau, c: Sequence[Frac],
                      max_iter: int = 10_000) -> str:
    """
    max c x sur le tableau donne. Renvoie 'optimal' ou 'non_borne'.
    Regle de Bland des deux cotes : terminaison garantie.
    """
    for _ in range(max_iter):
        red = tab.couts_reduits(c)
        entrant = next((j for j in tab.nonbasic() if red[j] > 0), None)
        if entrant is None:
            return "optimal"
        sortant = tab.ratio_min(entrant)
        if sortant is None:
            return "non_borne"
        tab.pivot(sortant, entrant)
    raise RuntimeError("simplexe lineaire : max_iter atteint")


def gradient_reduit_fractionnaire(tab: Tableau,
                                  p: Sequence[Frac], alpha: Frac,
                                  q: Sequence[Frac], beta: Frac
                                  ) -> Tuple[List[Frac], Frac, Frac]:
    """
    Gradient reduit de Martos pour max (p x + alpha) / (q x + beta), sous la
    forme exacte de la section 2 de Zerdani & Moulai :

        gamma_j = D * (p_j - zp_j) - N * (q_j - zq_j)

    Renvoie (gamma, N, D). D doit rester > 0, ce que garantit l'hypothese
    des auteurs sur le domaine.
    """
    x = tab.solution()
    N = alpha + sum(p[j] * x[j] for j in range(tab.nvar))
    D = beta + sum(q[j] * x[j] for j in range(tab.nvar))
    if D <= 0:
        raise ValueError("denominateur <= 0 : hypothese violee")
    rp = tab.couts_reduits(p)
    rq = tab.couts_reduits(q)
    return [D * rp[j] - N * rq[j] for j in range(tab.nvar)], N, D


def resoudre_fractionnaire(tab: Tableau,
                           p: Sequence[Frac], alpha: Frac,
                           q: Sequence[Frac], beta: Frac,
                           max_iter: int = 10_000) -> str:
    """
    max (p x + alpha) / (q x + beta) sur le polyedre du tableau, par la
    methode de Martos. Optimalite : gamma_j <= 0 pour tout j hors base
    (theoreme 2.1 des auteurs). Regle de Bland a l'entree.
    """
    for _ in range(max_iter):
        gamma, _, _ = gradient_reduit_fractionnaire(tab, p, alpha, q, beta)
        entrant = next((j for j in tab.nonbasic() if gamma[j] > 0), None)
        if entrant is None:
            return "optimal"
        sortant = tab.ratio_min(entrant)
        if sortant is None:
            return "non_borne"
        tab.pivot(sortant, entrant)
    raise RuntimeError("simplexe fractionnaire : max_iter atteint")


# ----------------------------------------------------------------------------
# Auto-verification
# ----------------------------------------------------------------------------

def _auto_test() -> int:
    """
    Controle du simplexe contre une enumeration exhaustive, sur des
    polyedres a donnees entieres dont tous les sommets sont enumerables.
    On ne verifie pas seulement la VALEUR mais aussi que la solution rendue
    est bien un SOMMET realisable et que le critere d'optimalite tient.
    """
    rng = np.random.default_rng(20260908)
    echecs = 0
    for essai in range(60):
        n = int(rng.integers(2, 5))
        m = int(rng.integers(2, 5))
        A = rng.integers(1, 6, size=(m, n))
        b = rng.integers(4, 20, size=m)
        ub = [min(int(b[i] // A[i][j]) for i in range(m) if A[i][j] > 0)
              for j in range(n)]
        pts = []
        idx = np.indices([u + 1 for u in ub]).reshape(n, -1).T
        for v in idx:
            if all(int(A[i] @ v) <= int(b[i]) for i in range(m)):
                pts.append(v)

        # -- objectif LINEAIRE ---------------------------------------------
        c = rng.integers(1, 9, size=n)
        tab = tableau_initial(A, b)
        st = resoudre_lineaire(tab, [_F(v) for v in c] + [Frac(0)] * m)
        xs = tab.x()
        val_lp = sum(_F(c[j]) * xs[j] for j in range(n))
        val_ent = max(sum(int(c[j]) * int(v[j]) for j in range(n))
                      for v in pts)
        ok = (st == "optimal" and val_lp >= val_ent
              and all(sum(_F(A[i][j]) * xs[j] for j in range(n))
                      <= _F(b[i]) for i in range(m))
              and all(v >= 0 for v in xs))
        echecs += 0 if ok else 1

        # -- objectif FRACTIONNAIRE ----------------------------------------
        p = rng.integers(1, 9, size=n)
        qd = rng.integers(0, 4, size=n)
        beta = int(rng.integers(1, 5))
        tab2 = tableau_initial(A, b)
        st2 = resoudre_fractionnaire(
            tab2, [_F(v) for v in p] + [Frac(0)] * m, Frac(0),
            [_F(v) for v in qd] + [Frac(0)] * m, _F(beta))
        xs2 = tab2.x()
        val2 = (sum(_F(p[j]) * xs2[j] for j in range(n))
                / (_F(beta) + sum(_F(qd[j]) * xs2[j] for j in range(n))))
        val2_ent = max(Fraction(int(p @ v), int(beta + qd @ v)) for v in pts)
        gamma, _, _ = gradient_reduit_fractionnaire(
            tab2, [_F(v) for v in p] + [Frac(0)] * m, Frac(0),
            [_F(v) for v in qd] + [Frac(0)] * m, _F(beta))
        ok2 = (st2 == "optimal" and val2 >= val2_ent
               and all(gamma[j] <= 0 for j in tab2.nonbasic()))
        echecs += 0 if ok2 else 1

    print(f"auto-test simplexe rationnel : {120 - echecs}/120 controles passes")
    print("  (valeur du sommet >= optimum ENTIER, realisabilite, et critere")
    print("   d'optimalite gamma_j <= 0 hors base pour le cas fractionnaire)")

    # -- phase I : second membre de signe QUELCONQUE ----------------------
    # C'est le cas qu'imposent les coupes « somme x_j >= 1 » de Zerdani &
    # Moulai, qui rendent la base des ecarts irrealisable.
    rng2 = np.random.default_rng(7)
    ok2, tot2, vides = 0, 0, 0
    for _ in range(80):
        n = int(rng2.integers(2, 4))
        mm = int(rng2.integers(2, 5))
        A = rng2.integers(-3, 6, size=(mm, n))
        b = rng2.integers(-4, 12, size=mm)
        pts = [v for v in np.indices([13] * n).reshape(n, -1).T
               if all(int(A[i] @ v) <= int(b[i]) for i in range(mm))]
        tab = tableau_phase_un(A, b)
        tot2 += 1
        if tab is None:
            # annoncer VIDE est une affirmation forte : elle doit au moins
            # etre compatible avec l'absence de point entier realisable
            ok2 += 1 if not pts else 0
            vides += 1
            continue
        c = rng2.integers(1, 6, size=n)
        st = resoudre_lineaire(tab, [_F(v) for v in c]
                               + [Frac(0)] * (tab.nvar - n))
        xs = tab.x()
        faisable = (all(sum(_F(A[i][j]) * xs[j] for j in range(n)) <= _F(b[i])
                        for i in range(mm)) and all(v >= 0 for v in xs))
        # un polyedre continu NON VIDE peut ne contenir aucun point entier :
        # la comparaison a l'optimum entier n'a alors pas lieu d'etre.
        borne = (st == "non_borne" or not pts
                 or sum(_F(c[j]) * xs[j] for j in range(n))
                 >= max(sum(int(c[j]) * int(v[j]) for j in range(n))
                        for v in pts))
        ok2 += 1 if (faisable and borne) else 0
    print(f"auto-test phase I : {ok2}/{tot2} controles passes "
          f"({vides} polyedres vides detectes)")
    echecs += tot2 - ok2
    return 0 if echecs == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_auto_test())
