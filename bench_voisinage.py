#!/usr/bin/env python3
"""
bench_voisinage.py
==================
LE VOLET METAHEURISTIQUE, MESURE PLUTOT QUE DECRIT.

Pourquoi ce banc existe. L'article decrivait la phase de recherche par un
seul encadre -- « VNS dans l'espace des criteres » -- alors que le sujet est
precisement l'hybridation d'une metaheuristique avec une methode exacte. Un
lecteur ne pouvait donc ni reproduire la recherche, ni juger laquelle de ses
pieces travaille. Ce banc produit les chiffres qui manquaient.

Ce qu'il mesure, par structure de voisinage :

  * TENTATIVES   combien de fois le mouvement est tire ;
  * SUCCES       combien de fois il ameliore STRICTEMENT l'incumbent ;
  * RENDEMENT    succes / tentatives -- la seule facon de savoir si un
                 mouvement merite sa place dans le menu ;

et, pour la recherche dans son ensemble :

  * REDEMARRAGES le nombre de tours declenches par stagnation (le
                 remplacant du `shaking` : on repart des points d'archive
                 les MOINS utilises, non d'un point aleatoire) ;
  * ARCHIVE      sa taille finale ;
  * CHAINES      la distribution des profondeurs de chaine de reparation.

Ce dernier point n'est pas decoratif. Le diagnostic central de l'article --
« la coupe de dominance n'a presque rien a mordre » -- repose entierement
sur l'affirmation que ces chaines sont courtes. Une affirmation pareille se
mesure. Profondeur 0 signifie que le point produit par le mouvement etait
DEJA efficace, donc qu'aucun point domine n'a ete traverse, donc qu'aucune
coupe du Th. 4 n'a ete recoltee.

Usage :  python bench_voisinage.py [graines] [plafond_appels]
"""

import os
import sys
from collections import Counter

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P
from molfp_oracle import CHAINES, reset_chaines

TAILLES = [int(v) for v in os.environ.get("MOLFP_TAILLES", "10,20,30").split(",")]
CORRS = [0.0, 0.5]


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    entete = (f"{'n':>4}{'corr':>6}{'gr':>4}"
              f"{'A tent/succ':>14}{'B tent/succ':>14}{'C tent/succ':>14}"
              f"{'redem.':>8}{'|A|':>6}{'chaines 0/1/2+':>16}")
    print("=" * len(entete))
    print(f"VOLET METAHEURISTIQUE - plafond {cap} appels entiers, "
          f"{n_gr} graines")
    print("=" * len(entete))
    print(entete)
    print("-" * len(entete))

    tot_m = Counter()
    tot_h = Counter()
    prof = Counter()
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in range(1, n_gr + 1):
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                reset_oracle_counter()
                reset_chaines()
                r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                                   seed=g, archive_cuts=True, ilp_budget=cap,
                                   cut_diversify=False, geom_bound=True)
                mv = r.cert.get("moves", {})
                hi = r.cert.get("hits", {})
                for k in "ABC":
                    tot_m[k] += mv.get(k, 0)
                    tot_h[k] += hi.get(k, 0)
                c = Counter(min(d, 2) for d in CHAINES)
                for k, v in c.items():
                    prof[k] += v
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      + "".join(f"{str(mv.get(k, 0)) + '/' + str(hi.get(k, 0)):>14}"
                                for k in "ABC")
                      + f"{r.cert.get('restarts', 0):>8}{len(r.archive):>6}"
                      + f"{str(c.get(0,0)) + '/' + str(c.get(1,0)) + '/' + str(c.get(2,0)):>16}",
                      flush=True)

    print("-" * len(entete))
    print("  RENDEMENT PAR STRUCTURE DE VOISINAGE (tous lots confondus)")
    for k, nom in (("A", "epsilon partiel"), ("B", "plancher absolu"),
                   ("C", "LNS / fix-and-optimize")):
        t, h = tot_m[k], tot_h[k]
        print(f"    {k}  {nom:<24} {h:>5} succes / {t:>5} tentatives"
              f"   = {100*h/max(1,t):5.1f} %")
    tot = sum(prof.values())
    if tot:
        print(f"  PROFONDEUR DES CHAINES DE REPARATION sur {tot} reparations :")
        print(f"    0 (deja efficace) {prof[0]:>6}  = {100*prof[0]/tot:5.1f} %")
        print(f"    1 pas             {prof[1]:>6}  = {100*prof[1]/tot:5.1f} %")
        print(f"    2 pas ou plus     {prof[2]:>6}  = {100*prof[2]/tot:5.1f} %")
        print()
        print("    Une profondeur 0 ne fournit AUCUN point domine, donc")
        print("    aucune coupe du Th. 4. C'est la mesure sur laquelle")
        print("    repose le diagnostic du verrou.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
