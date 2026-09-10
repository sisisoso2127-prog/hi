#!/usr/bin/env python3
"""
indicateurs.py
==============
INDICATEURS MULTIOBJECTIF POUR L'ARCHIVE, mesuree comme un LIVRABLE.

Pourquoi ce module existe. Le probleme pose demande « un sous-ensemble de
solutions efficaces engendre par des algorithmes hybrides ». Notre methode
en produit un -- l'archive A, incluse dans E par construction -- et le
memoire ne le mesurait nulle part : ni |A|/|E|, ni couverture, ni
hypervolume, ni epsilon-indicateur. Le front approche etait traite comme un
sous-produit servant a couper, jamais comme un resultat.

Tout ici est en MAXIMISATION, et les vecteurs criteres sont exacts
(Fraction). Les indicateurs sont ceux qu'une lecture multiobjectif attend :

  PURETE       |A inter E| / |A|. Vaut 1 par construction chez nous --
               chaque point est certifie efficace -- et c'est precisement
               ce qu'une metaheuristique usuelle ne garantit pas. On le
               VERIFIE plutot que de l'affirmer.
  CARDINALITE  |Z(A)| / |Z(E)|, en vecteurs criteres DISTINCTS.
  COUVERTURE   C(A,E) : part de E faiblement dominee par au moins un point
               de A. C'est la mesure de ce que l'archive « represente ».
  HYPERVOLUME  HV(A)/HV(E), exact, par balayage dimensionnel recursif --
               pas d'estimation. Le point de reference est le nadir de E
               ABAISSE d'une marge (10 % de l'etendue de chaque critere).
               La marge n'est pas cosmetique : avec la reference posee
               exactement sur le nadir, tout point extreme de E a une boite
               d'epaisseur nulle sur au moins un critere, donc contribue
               zero ; une archive reduite aux extremes serait notee 0.0
               alors qu'elle tient les deux bouts du front. La marge est la
               meme pour A et pour E, donc le rapport reste comparable.
  EPSILON+     I(A,E) = max_{z dans E} min_{a dans A} max_k (z_k - a_k),
               rapporte a l'etendue de chaque critere. C'est le decalage
               qu'il faudrait ajouter a A pour qu'elle domine E tout
               entier ; zero signifie que A couvre E exactement.

Aucun de ces indicateurs ne remplace les autres : la cardinalite ignore la
position, l'hypervolume favorise le centre du front, l'epsilon-indicateur
en mesure le pire point. On les rapporte ensemble.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

Vecteur = Sequence[float]


# ---------------------------------------------------------------------------
# Dominance (maximisation)
# ---------------------------------------------------------------------------

def domine_faiblement(u: Vecteur, v: Vecteur) -> bool:
    """u >= v composante par composante."""
    return all(ui >= vi for ui, vi in zip(u, v))


def filtre_non_domines(pts: Sequence[Vecteur]) -> List[tuple]:
    uniq = sorted({tuple(p) for p in pts}, reverse=True)
    garde: List[tuple] = []
    for v in uniq:
        if not any(domine_faiblement(u, v) and u != v for u in garde):
            garde.append(v)
    return garde


# ---------------------------------------------------------------------------
# Hypervolume EXACT, par balayage dimensionnel
# ---------------------------------------------------------------------------

def hypervolume(pts: Sequence[Vecteur], ref: Vecteur) -> float:
    """
    Volume de l'union des boites [ref, z] pour z dans `pts`, en
    MAXIMISATION. Exact : recursion par tranches sur la derniere
    coordonnee, sans echantillonnage.

    Les points sous la reference sont ignores ; c'est le comportement
    usuel, la reference etant le nadir.
    """
    P = [tuple(float(c) for c in z) for z in pts]
    R = [float(c) for c in ref]
    P = [z for z in P if all(zi >= ri for zi, ri in zip(z, R))]
    if not P:
        return 0.0
    return _hv(filtre_non_domines(P), R)


def _hv(P: List[tuple], R: List[float]) -> float:
    d = len(R)
    if d == 1:
        return max(z[0] for z in P) - R[0]
    # tranches sur la derniere coordonnee, de la plus haute vers le bas
    Q = sorted(P, key=lambda z: -z[-1])
    total, i = 0.0, 0
    while i < len(Q):
        haut = Q[i][-1]
        j = i
        while j < len(Q) and Q[j][-1] == haut:
            j += 1
        bas = Q[j][-1] if j < len(Q) else R[-1]
        if haut > bas:
            proj = filtre_non_domines([z[:-1] for z in Q[:j]])
            total += (haut - bas) * _hv(proj, R[:-1])
        i = j
    return total


# ---------------------------------------------------------------------------
# Epsilon-indicateur additif
# ---------------------------------------------------------------------------

def epsilon_additif(A: Sequence[Vecteur], E: Sequence[Vecteur],
                    etendues: Sequence[float] | None = None) -> float:
    """
    I(A,E) = max_{z dans E} min_{a dans A} max_k (z_k - a_k).

    Decalage minimal a ajouter a chaque point de A pour que A domine
    faiblement E tout entier. Vaut 0 si et seulement si A couvre E.
    `etendues` normalise critere par critere ; sans elle, l'indicateur est
    dans l'unite des criteres et ne se compare pas d'une instance a l'autre.
    """
    if not A:
        return float("inf")
    p = len(next(iter(E)))
    ech = [float(e) if e else 1.0 for e in (etendues or [1.0] * p)]
    pire = 0.0
    for z in E:
        best = min(
            max((float(z[k]) - float(a[k])) / ech[k] for k in range(p))
            for a in A)
        pire = max(pire, best)
    return pire


# ---------------------------------------------------------------------------
# Couverture et purete
# ---------------------------------------------------------------------------

def couverture(A: Sequence[Vecteur], B: Sequence[Vecteur]) -> float:
    """C(A,B) : part de B faiblement dominee par au moins un point de A."""
    if not B:
        return 1.0
    n = sum(1 for b in B if any(domine_faiblement(a, b) for a in A))
    return n / len(B)


def purete(A: Sequence[Vecteur], E: Sequence[Vecteur]) -> float:
    """Part des points de A qui sont REELLEMENT efficaces."""
    if not A:
        return 1.0
    SE = {tuple(z) for z in E}
    return sum(1 for a in A if tuple(a) in SE) / len(A)


def indicateurs(ZA: Sequence[Vecteur], ZE: Sequence[Vecteur]) -> dict:
    """Tous les indicateurs d'un coup, avec le nadir de E pour reference."""
    p = len(next(iter(ZE)))
    nadir = [min(float(z[k]) for z in ZE) for k in range(p)]
    ideal = [max(float(z[k]) for z in ZE) for k in range(p)]
    etendues = [i - n for i, n in zip(ideal, nadir)]
    # reference abaissee : voir l'entete, sans marge les extremes valent 0
    ref = [n - (0.1 * e if e > 0 else 1.0) for n, e in zip(nadir, etendues)]
    hvE = hypervolume(ZE, ref)
    hvA = hypervolume(ZA, ref)
    dZA = {tuple(z) for z in ZA}
    dZE = {tuple(z) for z in ZE}
    return {
        "purete": purete(ZA, ZE),
        "card": len(dZA) / max(1, len(dZE)),
        "nA": len(dZA), "nE": len(dZE),
        "couverture": couverture(ZA, ZE),
        "hv": (hvA / hvE) if hvE > 0 else 1.0,
        "eps": epsilon_additif(ZA, ZE, etendues),
    }


if __name__ == "__main__":
    # controles : un front connu, et les cas limites
    E = [(3, 1), (2, 2), (1, 3)]
    assert abs(hypervolume(E, (0, 0)) - 6.0) < 1e-9, hypervolume(E, (0, 0))
    assert abs(hypervolume([(3, 1), (1, 3)], (0, 0)) - 5.0) < 1e-9
    d = indicateurs(E, E)
    assert d["purete"] == 1.0 and d["couverture"] == 1.0
    assert abs(d["hv"] - 1.0) < 1e-12 and d["eps"] == 0.0
    part = indicateurs([(3, 1)], E)
    assert part["couverture"] < 1.0 and part["eps"] > 0.0
    # un extreme de E doit compter POUR QUELQUE CHOSE : c'est le role de la
    # marge sur la reference. Sans elle ce rapport vaudrait exactement 0.
    assert 0.0 < part["hv"] < 1.0, part["hv"]
    assert hypervolume([(3, 1)], [1.0, 1.0]) == 0.0  # le piege, documente
    print("indicateurs.py : tous les controles passent")
