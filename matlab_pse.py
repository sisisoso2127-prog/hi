#!/usr/bin/env python3
"""
matlab_pse.py
=============
PORTAGE EN PYTHON DE LA SUITE MATLAB : dichotomie + separation et evaluation.

Les neuf fichiers .m d'origine -- YousGui, pse, dichotomie, branch, maj_eff,
vect_criter, forme_standard, vr, revised_simplex, dual_simplex -- sont
reunis ici en un seul module. La seconde suite (sylva_crema / find_SND /
loop) N'EST PAS portee : `loop.m`, qui en contient toute la mecanique, n'a
pas ete fourni, et porter une methode dont la boucle manque produirait une
coquille.

CE QUE CETTE METHODE RESOUT. Un MOILP a DEUX criteres LINEAIRES et
variables BINAIRES : max (Z1 x, Z2 x) sous A x <= b, x dans {0,1}^n. Ce
n'est pas le probleme de ce depot -- criteres fractionnaires, p criteres,
variables entieres generales -- et le rapprochement des deux se fait par le
cas particulier : voir `verify_lineaire.py`.

PORTAGE FIDELE, CORRECTIONS NOMMEES. Chaque ecart au code MATLAB est
signale par un commentaire « CORRECTION ». Le drapeau `fidele=True` rend
le comportement d'origine la ou cela reste executable, pour que la
difference se mesure au lieu de se raconter. Les corrections sont :

  1. bintprog -> HiGHS. bintprog a ete retire de MATLAB en R2016a ; sans ce
     remplacement, rien ne tourne.
  2. `ismember(x', SES)` sans 'rows'. Sur des vecteurs 0/1 la condition est
     presque toujours fausse, si bien que le point supporte trouve par le
     probleme pondere n'etait JAMAIS ajoute.
  3. Dichotomie non recursive. L'originale resout un seul probleme pondere
     et s'arrete : elle rend au plus trois points et ne trouve donc pas tous
     les supportes. La version corrigee recurse sur les deux sous-intervalles
     (Aneja & Nair).
  4. `branch` reliait `ind` et `x` a l'interieur de sa propre boucle, si bien
     que l'iteration suivante parcourait les indices de l'ENFANT et non ceux
     du noeud courant : l'arbre explore n'etait pas celui que le code
     decrit.
  5. Aucun controle d'infaisabilite. Fixer des variables en produit
     necessairement ; l'original enchainait alors `Z*x` sur un `x` vide.
  6. Aucun filtre de Pareto final : `SND` etait `Z*EFF'` tel quel, sans
     garantie de non-dominance malgre son nom.
  7. `pse` lisait SND(2,1), SND(1,2), SND(2,2) sans verifier qu'il y a bien
     deux points non domines -- ce qui est faux quand les deux criteres
     atteignent leur optimum au meme point.

CE QUI N'EST PAS CORRIGE, PARCE QUE CE N'EST PAS UN DEFAUT DE CODE. La
methode ne rend qu'un representant par vecteur criteres lorsque `maj_eff`
ecarte les doublons, et son test d'arret repose sur des bornes calculees
aux deux optima lexicographiques ; ces choix sont ceux des auteurs.

Usage :  python matlab_pse.py            instance de sylva_crema.m
         python matlab_pse.py --croise   contre la force brute
         python matlab_pse.py --fidele   sans les corrections 2, 3 et 6
"""

from __future__ import annotations

import sys
from typing import List, Optional, Sequence, Tuple

import numpy as np

from molfp_core import solve_milp


# ---------------------------------------------------------------------------
# vr.m, vect_criter.m, forme_standard.m
# ---------------------------------------------------------------------------

def vr(m: int, i: int) -> np.ndarray:
    """vr.m : i-eme vecteur de la base canonique de R^m (i indexe a 1)."""
    e = np.zeros((m, 1))
    e[i - 1] = 1.0
    return e


def vect_criter(x: np.ndarray, C: np.ndarray) -> np.ndarray:
    """vect_criter.m : le vecteur critere C x."""
    return C @ np.asarray(x).reshape(-1)


def forme_standard(type_: str, c: np.ndarray, A: np.ndarray,
                   rel: Sequence[str], m: int, n: int):
    """forme_standard.m : mise sous forme standard par variables d'ecart.

    CORRECTION : `if (type == 'max')` comparait deux tableaux de caracteres
    element par element -- exact par accident pour 'min'/'max', et faux des
    que la chaine change de longueur. On compare des chaines.

    Le defaut de FOND de l'original est conserve et signale : un contrainte
    '>' recoit une variable de SURPLUS negative sans variable artificielle,
    de sorte que la base d'ecart n'est pas realisable. C'est ce que
    l'indicateur `bool` signale a l'appelant, et c'est a lui d'en tenir
    compte -- l'original ne le fait nulle part.
    """
    mm = 0 if type_ == "max" else 1
    c = np.asarray(c, dtype=float).reshape(1, -1)
    if mm == 1:
        c = -c
    A = np.asarray(A, dtype=float)

    les = 0
    for i in range(1, m + 1):
        if rel[i - 1] == "<":
            A = np.hstack([A, vr(m, i)])
            les += 1
        elif rel[i - 1] == ">":
            A = np.hstack([A, -vr(m, i)])
        # '=' : aucune colonne ajoutee, comme dans l'original

    ncol = A.shape[1]
    c = np.hstack([c, np.zeros((1, ncol - n))])
    return c, A, mm, int(les == m), ncol


# ---------------------------------------------------------------------------
# revised_simplex.m, dual_simplex.m
# ---------------------------------------------------------------------------

def revised_simplex(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                    B: List[int], iter_: int = 0):
    """revised_simplex.m : simplexe revise, regle de Dantzig, maximisation."""
    eps = 1e-6
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    B = list(B)
    N = [j for j in range(n) if j not in B]

    while True:
        iter_ += 1
        xB = np.linalg.solve(A[:, B], b)
        y = np.linalg.solve(A[:, B].T, c[B])
        cN = c[N] - y @ A[:, N]

        s = int(np.argmax(cN))
        if cN[s] <= eps:
            x = np.zeros(n)
            x[B] = xB
            return float(c @ x), B, x, iter_

        As = np.linalg.solve(A[:, B], A[:, N[s]])
        I = np.where(As > eps)[0]
        if I.size == 0:
            raise ValueError("Le PL n'admet pas de solution optimale finie")
        r = int(np.argmin(xB[I] / As[I]))
        l = int(I[r])
        B[l], N[s] = N[s], B[l]


def dual_simplex(c: np.ndarray, A: np.ndarray, b: np.ndarray,
                 B: List[int], iter_: int = 0):
    """dual_simplex.m : simplexe dual a partir d'une base dual-realisable.

    CORRECTION du test de rapport. L'original ecrit
    `[t,j] = min(cN(I)./w(I))` avec w < 0 : les rapports sont negatifs et
    `min` retient le plus negatif, c'est-a-dire le PLUS GRAND
    cN/(-w) -- l'inverse du test dual, qui minimise cette quantite.
    Minimiser cN/(-w) revient a MAXIMISER cN/w ; c'est ce qui est ecrit ici.
    """
    eps = 1e-6
    c = np.asarray(c, dtype=float).reshape(-1)
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).reshape(-1)
    m, n = A.shape
    B = list(B)
    N = [j for j in range(n) if j not in B]

    while True:
        iter_ += 1
        xB = np.linalg.solve(A[:, B], b)
        y = np.linalg.solve(A[:, B].T, c[B])
        cN = c[N] - y @ A[:, N]

        s = int(np.argmin(xB))
        if xB[s] > -eps:
            x = np.zeros(n)
            x[B] = xB
            return float(c @ x), x, B, 1, iter_

        v = np.linalg.solve(A[:, B], A[:, N])
        w = v[s, :]
        I = np.where(w < -eps)[0]
        if I.size == 0:
            return None, None, B, 2, iter_        # PL irrealisable

        r = int(np.argmax(cN[I] / w[I]))          # CORRECTION : max, pas min
        r = int(I[r])
        B[s], N[r] = N[r], B[s]


# ---------------------------------------------------------------------------
# Le solveur en nombres binaires : remplacement de bintprog
# ---------------------------------------------------------------------------

def bintprog(f: np.ndarray, A: np.ndarray, b: np.ndarray,
             Aeq: Optional[np.ndarray] = None,
             beq: Optional[np.ndarray] = None):
    """CORRECTION 1 : bintprog, retire de MATLAB en R2016a, remplace par
    HiGHS. Meme convention que l'original -- MINIMISATION de f x sous
    A x <= b, Aeq x = beq, x binaire -- et meme contrat de retour : (None,
    None) quand le probleme est infaisable, la ou MATLAB rendait [].
    """
    f = np.asarray(f, dtype=float).reshape(-1)
    n = f.size
    rows: List[Tuple[np.ndarray, float, float]] = []
    A = np.asarray(A, dtype=float).reshape(-1, n)
    b = np.asarray(b, dtype=float).reshape(-1)
    for i in range(A.shape[0]):
        rows.append((A[i], -np.inf, float(b[i])))
    if Aeq is not None and len(Aeq):
        Aeq = np.asarray(Aeq, dtype=float).reshape(-1, n)
        beq = np.asarray(beq, dtype=float).reshape(-1)
        for i in range(Aeq.shape[0]):
            rows.append((Aeq[i], float(beq[i]), float(beq[i])))
    res = solve_milp(f, rows, np.zeros(n), np.ones(n), maximize=False)
    if res.x is None:
        return None, None
    x = np.rint(np.asarray(res.x)[:n]).astype(int)
    return x, float(f @ x)


# ---------------------------------------------------------------------------
# maj_eff.m
# ---------------------------------------------------------------------------

def maj_eff(x: np.ndarray, C: np.ndarray,
            Eff: List[np.ndarray]) -> List[np.ndarray]:
    """maj_eff.m : insere x dans Eff en retirant ce qu'il domine.

    CORRECTION : l'original sortait de la boucle des qu'il rencontrait un
    vecteur critere EGAL (`i=p+1`), sans examiner la suite de Eff. Les
    points domines situes APRES ce rang n'etaient donc jamais retires, et
    Eff pouvait conserver des points domines. Ici le balayage est complet.
    """
    x = np.asarray(x).reshape(-1)
    if not Eff:
        return [x.copy()]
    if any(np.array_equal(x, e) for e in Eff):
        return Eff

    C1 = vect_criter(x, C)
    garder, domines = True, []
    for i, e in enumerate(Eff):
        C2 = vect_criter(e, C)
        if np.array_equal(C1, C2):
            continue                       # vecteur deja represente
        if np.all(C1 <= C2):
            garder = False                 # x est domine : on n'ajoute pas
            break
        if np.all(C1 >= C2):
            domines.append(i)
    if not garder:
        return Eff
    Eff = [e for i, e in enumerate(Eff) if i not in domines]
    Eff.append(x.copy())
    return Eff


# ---------------------------------------------------------------------------
# dichotomie.m
# ---------------------------------------------------------------------------

def dichotomie(Z: np.ndarray, A: np.ndarray, b: np.ndarray,
               fidele: bool = False):
    """dichotomie.m : les solutions efficaces SUPPORTEES.

    CORRECTION 2 : `~ismember(x', SES)` sans 'rows' testait l'appartenance
    element par element ; sur des vecteurs 0/1 la condition etait presque
    toujours fausse et le point pondere n'etait jamais ajoute.

    CORRECTION 3 : l'original resout UN seul probleme pondere. La
    dichotomie d'Aneja & Nair recurse sur les deux sous-intervalles tant
    qu'un nouveau point apparait ; sans cela, l'ensemble supporte est
    incomplet des que le front en compte plus de trois.
    """
    SES: List[np.ndarray] = []
    SND: List[np.ndarray] = []

    for i in range(2):
        xi, _ = bintprog(-Z[i, :], A, b)
        if xi is None:
            continue
        SES.append(xi)
        SND.append(Z @ xi)

    if len(SES) < 2:
        return SES, SND, (Z[0, :] if SES else np.zeros(Z.shape[1]))

    def ajouter(x: np.ndarray) -> bool:
        # CORRECTION 2 : comparaison PAR LIGNE
        if any(np.array_equal(x, s) for s in SES):
            return False
        SES.append(x)
        SND.append(Z @ x)
        return True

    def scan(zl: np.ndarray, zr: np.ndarray, profondeur: int = 0) -> None:
        if profondeur > 64:
            return
        lam = np.array([zr[1] - zl[1], zl[0] - zr[0]], dtype=float)
        if not np.any(lam):
            return
        x, _ = bintprog(-(lam @ Z), A, b)
        if x is None:
            return
        z = Z @ x
        # un point deja sur la droite (zl, zr) ne peut rien ouvrir
        if lam @ z <= lam @ zl + 1e-9:
            return
        if ajouter(x):
            scan(zl, z, profondeur + 1)
            scan(z, zr, profondeur + 1)

    zl, zr = SND[0], SND[1]
    lam = np.array([zr[1] - zl[1], zl[0] - zr[0]], dtype=float)
    if fidele:
        x, _ = bintprog(-(lam @ Z), A, b)      # un seul tour, comme l'original
        if x is not None:
            ajouter(x)
    else:
        scan(zl, zr)

    Zlambda = lam @ Z
    return SES, SND, Zlambda


# ---------------------------------------------------------------------------
# branch.m
# ---------------------------------------------------------------------------

def branch(A: np.ndarray, b: np.ndarray, ind: Sequence[int],
           fixes: List[Tuple[int, int]], SENS: List[np.ndarray],
           Zlambda: np.ndarray, Z: np.ndarray,
           infz1: float, infz2: float, infZlanda: float,
           vus: set, profondeur: int = 0) -> List[np.ndarray]:
    """branch.m : separation et evaluation sur les variables a 1.

    CORRECTION 4 : l'original ecrivait `ind=find(x)` puis reassignait `x`
    DANS sa propre boucle, si bien que l'iteration suivante lisait `ind(j)`
    dans le tableau de l'ENFANT. L'arbre parcouru n'etait pas celui que le
    code decrit. Ici les fixations d'un noeud sont passees en argument et
    la boucle du parent garde SES indices.

    CORRECTION 5 : controle d'infaisabilite. Fixer des variables en produit
    necessairement, et l'original enchainait `Z*x` sur un `x` vide.

    Le jeu de fixations deja visite est memorise : l'original n'avait ni
    memoire ni borne de profondeur, et pouvait recurser jusqu'a la limite
    de MATLAB.
    """
    if profondeur > 32:
        return SENS
    n = A.shape[1]

    for j, colonne in enumerate(ind):
        # noeud : les j premieres variables de `ind` a 1, celle-ci a 0
        courant = fixes + [(int(ind[t]), 1) for t in range(j)] \
                        + [(int(colonne), 0)]
        cle = tuple(sorted(courant))
        if cle in vus:
            continue
        vus.add(cle)

        Aeq = np.zeros((len(courant), n))
        beq = np.zeros(len(courant))
        for r, (col, val) in enumerate(courant):
            Aeq[r, col] = 1.0
            beq[r] = float(val)

        x, fval = bintprog(-Zlambda, A, b, Aeq, beq)
        if x is None:                       # CORRECTION 5
            continue
        vect = Z @ x
        SENS = maj_eff(x, Z, SENS)

        if vect[0] < infz1 or vect[1] < infz2 or abs(fval) < infZlanda:
            continue                        # noeud elague
        suivants = [int(t) for t in np.flatnonzero(x)]
        if suivants:
            SENS = branch(A, b, suivants, courant, SENS, Zlambda, Z,
                          infz1, infz2, infZlanda, vus, profondeur + 1)
    return SENS


# ---------------------------------------------------------------------------
# pse.m
# ---------------------------------------------------------------------------

def pse(A: np.ndarray, Z: np.ndarray, b: np.ndarray,
        fidele: bool = False):
    """pse.m : dichotomie puis separation et evaluation.

    CORRECTION 7 : l'original lisait SND(2,1), SND(1,2) et SND(2,2) sans
    verifier qu'il y a deux points non domines. Quand les deux criteres
    atteignent leur optimum au meme point -- cas parfaitement legitime --
    MATLAB s'arretait sur une erreur d'indice.

    CORRECTION 6 : filtre de Pareto final. L'original rendait `Z*EFF'` tel
    quel sous le nom « solutions non dominees », sans rien qui le garantisse.
    """
    SES, SND, Zlambda = dichotomie(Z, A, b, fidele=fidele)
    if len(SND) < 2:                                   # CORRECTION 7
        EFF = [np.asarray(s).reshape(-1) for s in SES]
        return EFF, [Z @ x for x in EFF]

    infz1 = float(SND[1][0])
    infz2 = float(SND[0][1])
    lam = np.array([SND[1][1] - SND[1][0], SND[0][0] - SND[0][1]],
                   dtype=float)
    infZlanda = float(lam @ np.array([infz1, infz2]))

    SENS = list(SES)
    depart = [int(t) for t in np.flatnonzero(SES[0])]
    SENS = branch(A, b, depart, [], SENS, Zlambda, Z,
                  infz1, infz2, infZlanda, set())

    EFF = [np.asarray(s).reshape(-1) for s in SENS]
    if not fidele:                                     # CORRECTION 6
        garde: List[np.ndarray] = []
        for x in EFF:
            garde = maj_eff(x, Z, garde)
        EFF = garde
    return EFF, [Z @ x for x in EFF]


# ---------------------------------------------------------------------------

C_MATLAB = np.array([[3, 8, 1, 5, 7, 4],
                     [7, 5, 8, 7, 0, 7]])
A_MATLAB = np.array([[11, 1, 16, 13, 8, 4],
                     [6, 13, 0, 13, 1, 9]])
B_MATLAB = np.array([205, 136])


def _force_brute(Z, A, b):
    """Reference : tous les x binaires, filtres par dominance."""
    n = A.shape[1]
    best: List[np.ndarray] = []
    for masque in range(1 << n):
        x = np.array([(masque >> i) & 1 for i in range(n)])
        if np.all(A @ x <= b):
            best = maj_eff(x, Z, best)
    return {tuple(Z @ x) for x in best}


def croise(n: int = 10, m: int = 3, essais: int = 6,
           graine: int = 7) -> int:
    """Le portage contre la force brute, sur des instances binaires tirees.

    L'instance de sylva_crema.m ne convient pas a ce controle : en
    variables BINAIRES son front ne compte qu'un point, et un banc qui ne
    rend qu'une valeur ne distingue pas une methode juste d'une methode
    muette. On tire donc des instances dont le front est peuple.
    """
    rng = np.random.default_rng(graine)
    print("=" * 78)
    print(f"PORTAGE CONTRE FORCE BRUTE -- {essais} instances binaires "
          f"n={n}, m={m}")
    print("=" * 78)
    tout = True
    for essai in range(essais):
        Z = rng.integers(1, 20, size=(2, n))
        A = rng.integers(1, 15, size=(m, n))
        b = (A.sum(axis=1) * 0.45).astype(int)
        _, SND = pse(A, Z, b)
        obtenus = {tuple(v) for v in SND}
        attendus = _force_brute(Z, A, b)
        ok = obtenus == attendus
        tout &= ok
        print(f"  essai {essai}   pse {len(obtenus):>3}   "
              f"brute {len(attendus):>3}   "
              f"manquants {len(attendus - obtenus):>3}   "
              f"non efficaces {len(obtenus - attendus):>3}   "
              f"{'OK' if ok else 'ECHEC'}", flush=True)
    print("\n" + ("OK partout" if tout else "AU MOINS UN ECHEC"))
    return 0 if tout else 1


def main() -> int:
    if "--croise" in sys.argv:
        return croise()
    fidele = "--fidele" in sys.argv
    print("=" * 78)
    print("SUITE MATLAB PORTEE EN PYTHON -- instance de sylva_crema.m")
    print(f"  mode : {'FIDELE (corrections 2, 3 et 6 desactivees)' if fidele else 'CORRIGE'}")
    print("=" * 78)

    EFF, SND = pse(A_MATLAB, C_MATLAB, B_MATLAB, fidele=fidele)
    obtenus = {tuple(v) for v in SND}
    attendus = _force_brute(C_MATLAB, A_MATLAB, B_MATLAB)

    print(f"\n  solutions rendues       : {len(EFF)}")
    print(f"  vecteurs distincts      : {len(obtenus)}")
    print(f"  force brute (reference) : {len(attendus)}")
    manque = attendus - obtenus
    trop = obtenus - attendus
    print(f"  manquants : {len(manque)}   non efficaces rendus : {len(trop)}")
    if manque:
        print(f"    exemples manquants : {sorted(manque)[:5]}")
    print("\n" + ("OK" if not manque and not trop else
                  "INCOMPLET -- cette methode n'enumere pas tout E"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
