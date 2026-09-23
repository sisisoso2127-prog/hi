#!/usr/bin/env python3
"""
molfp_regions.py
================
ENUMERER Z(E) SANS LAISSER LE MODELE GROSSIR.

`molfp_enumere.py` enumere par coupes d'efficacite accumulees dans UN
modele. Le diagnostic (`diag_enum.py`) a montre ou cela casse : entre le
premier et le dernier quart d'une execution, le temps par programme entier
est multiplie par HUIT -- 0,957 s puis 7,766 s -- pendant que la chaine de
reparation ne bouge pas et pese 0,4 % du total. Chaque coupe ajoute p
binaires et une disjonction big-M ; le modele porte tout ce qu'il a trouve,
et le cout devient quadratique en |Z(E)|.

CE QUE FAIT CELUI-CI. Il decoupe l'espace des criteres en REGIONS, et
resout dans chacune un programme de taille CONSTANTE : les contraintes de
l'instance, plus au plus p lignes de borne inferieure. Rien ne s'accumule.
Le nombre de sous-problemes croit, mais chacun reste aussi facile que le
premier.

LA DECOMPOSITION. Une region est un vecteur de bornes L = (l_1..l_p),
chaque l_k etant une borne inferieure sur Z_k (ou l'absence de borne). On
y cherche un point efficace ; s'il n'y en a pas, la region est vide et
close. Sinon on trouve z, et tout ce qui reste d'efficace dans la region
est couvert par p sous-regions : pour chaque k, la meme region avec
l_k remplace par « strictement plus grand que z_k ». Aucun vecteur
efficace n'echappe, puisque le seul point ainsi retire est z lui-meme et
ce qu'il domine -- et rien d'efficace n'est domine par un efficace.

CE QUI RESTE FRACTIONNAIRE. La borne Z_k(x) >= P/Q n'est pas lineaire,
mais D(x) > 0 sous (A1) permet le meme produit croise que la coupe :

    Q (c_k x + a_k) - P (d_k x + b_k)  >=  0

est une ligne a coefficients entiers. Et comme cette quantite est ENTIERE
pour x entier, « strictement superieur » s'ecrit « >= 1 » -- sans epsilon,
sans tolerance, et c'est la seule raison pour laquelle ce decoupage est
exact dans le cadre fractionnaire.

CE QUI GARANTIT LA CORRECTION. La reparation ne sort jamais de la region :
elle suit une chaine de dominance, donc chaque pas AMELIORE le vecteur
critere composante par composante, et un point qui domine un point
respectant L respecte L. Un point rendu est donc a la fois efficace et
dans sa region.

TERMINAISON. Chaque sous-region releve strictement une borne, les criteres
sont bornes sur un domaine fini, donc la profondeur l'est aussi. Les
regions deja traitees sont memorisees : le decoupage se recouvre, et sans
cette memoire le meme travail serait refait.

Usage :  python molfp_regions.py            comparaison avec les coupes
         python molfp_regions.py n p corr    une instance precise
"""

from __future__ import annotations

import sys
import time
from fractions import Fraction
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter, set_ilp_budget
from molfp_instance import MOILFP, generate
from molfp_oracle import ECutModel, repair_to_efficient

INF = float("inf")
# Une borne retient sa VALEUR et sa STRICTESSE. Sans le second element,
# la sous-region engendree par un point z garde z lui-meme : elle rend le
# meme vecteur, se retrouve deja visitee, et toute la branche se referme.
# C'est la faute qui a fait rendre 2 vecteurs la ou il y en a 70.
Borne = Optional[Tuple[Fraction, bool]]


class Resultat(NamedTuple):
    vecteurs: List[tuple]
    points: List[np.ndarray]
    complet: bool
    motif: str
    ilp: int
    regions: int
    secondes: float


def _ligne_borne(inst: MOILFP, k: int, l: Fraction,
                 strict: bool) -> Tuple[np.ndarray, float, float]:
    """La ligne entiere qui exprime Z_k(x) >= l, ou > l si `strict`.

    Q N_k(x) - P D_k(x) >= 0, ou >= 1 pour le strict -- licite parce que
    cette quantite est entiere des que x l'est.
    """
    z = inst.Z[k]
    P, Q = l.numerator, l.denominator
    coef = Q * np.asarray(z.num, dtype=float) - P * np.asarray(z.den,
                                                               dtype=float)
    const = Q * float(z.a) - P * float(z.b)
    seuil = (1.0 if strict else 0.0) - const
    return (coef, seuil, INF)


def _poids(inst: MOILFP) -> np.ndarray:
    """Objectif de sondage. La correction n'en depend pas."""
    w = np.zeros(inst.n)
    for z in inst.Z:
        w = w + np.asarray(z.num, dtype=float)
    return w


def enumere_regions(inst: MOILFP,
                    ilp_budget: Optional[int] = None,
                    time_budget: float = 1e9,
                    verbose: bool = False) -> Resultat:
    t0 = time.time()
    reset_oracle_counter()
    if ilp_budget is not None:
        set_ilp_budget(ilp_budget)

    modele = ECutModel(inst)          # construit UNE fois, jamais augmente
    w = _poids(inst)

    vecteurs: List[tuple] = []
    points: List[np.ndarray] = []
    vus: set = set()
    regions_vues: set = set()
    pile: List[Tuple[Borne, ...]] = [tuple([None] * inst.p)]
    n_regions = 0

    while pile:
        if time.time() - t0 > time_budget:
            return Resultat(vecteurs, points, False, "budget de temps",
                            ORACLE_CALLS["ilp"], n_regions,
                            time.time() - t0)
        L = pile.pop()
        if L in regions_vues:
            continue
        regions_vues.add(L)
        n_regions += 1

        extra = [_ligne_borne(inst, k, val, strict=strict)
                 for k, b in enumerate(L) if b is not None
                 for val, strict in (b,)]
        reste = time_budget - (time.time() - t0)
        res = modele.optimize(w, 0.0, maximize=True, extra_rows=extra,
                              time_limit=reste)
        if res.status == "infeasible":
            continue                          # region close
        if res.x is None:
            return Resultat(vecteurs, points, False,
                            f"solveur interrompu ({res.status})",
                            ORACLE_CALLS["ilp"], n_regions,
                            time.time() - t0)

        x = np.rint(np.asarray(res.x[:inst.n])).astype(int)
        a = repair_to_efficient(inst, x, deadline=t0 + time_budget)
        if a is None:
            return Resultat(vecteurs, points, False,
                            "reparation non terminee dans le budget",
                            ORACLE_CALLS["ilp"], n_regions,
                            time.time() - t0)

        z = tuple(inst.criteria(a))
        if z not in vus:
            vus.add(z)
            vecteurs.append(z)
            points.append(a)
            if verbose:
                print(f"  {len(vecteurs):>5}  {z}", flush=True)

        # p sous-regions : la k-ieme exige de depasser STRICTEMENT z_k.
        # La strictesse est portee par la borne elle-meme, et elle est
        # exacte sans epsilon : la quantite croisee Q N_k - P D_k est
        # ENTIERE pour x entier, donc « > 0 » s'ecrit « >= 1 ».
        for k in range(inst.p):
            neuf = list(L)
            zk = Fraction(z[k])
            ancienne = neuf[k]
            if ancienne is None or zk >= ancienne[0]:
                neuf[k] = (zk, True)
                pile.append(tuple(neuf))
    return Resultat(vecteurs, points, True, "toutes regions closes",
                    ORACLE_CALLS["ilp"], n_regions, time.time() - t0)


def main() -> int:
    from molfp_enumere import enumere
    if len(sys.argv) > 3:
        cas = [(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]))]
    else:
        cas = [(6, 3, 0.00), (7, 3, 0.25), (10, 3, 0.00)]

    print("=" * 84)
    print("REGIONS (modele constant) CONTRE COUPES (modele cumulatif)")
    print("=" * 84)
    print(f"\n{'instance':<20}{'regions':>9}{'s':>8}{'ilp':>7}"
          f"{'coupes':>9}{'s':>8}{'ilp':>7}   accord")
    print("-" * 84)
    for n, p, corr in cas:
        inst = generate(n=n, m=max(3, n // 2 + 1), p=p, seed=1,
                        rhs_scale=1.0, corr=corr)
        r1 = enumere_regions(inst, time_budget=900)
        r2 = enumere(inst, time_budget=900)
        d = "OK" if (r1.complet and r2.complet
                     and set(r1.vecteurs) == set(r2.vecteurs)) else "A VOIR"
        print(f"n={n} p={p} c={corr:.2f}{'':<5}"
              f"{len(r1.vecteurs):>9}{r1.secondes:>8.1f}{r1.ilp:>7}"
              f"{len(r2.vecteurs):>9}{r2.secondes:>8.1f}{r2.ilp:>7}"
              f"   {d}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
