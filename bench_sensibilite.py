#!/usr/bin/env python3
"""
bench_sensibilite.py
====================
LES REGLAGES SONT-ILS DES NOMBRES MAGIQUES ?

Le memoire annonce des constantes sans les justifier : 8 bases d'elite plus
4 tirees au hasard, un menu de mouvements {A,A,B,C}, une fraction LNS qui
passe de 0,4 a 0,6 n en diversification, `max_stall`, `cut_batch` et
`cert_rounds`. Une valeur qu'on ne fait jamais varier n'est pas un reglage,
c'est une hypothese cachee : ou bien elle compte, et il faut dire pourquoi
celle-la, ou bien elle ne compte pas, et il faut le montrer.

PROTOCOLE. Un facteur a la fois (OAT) autour de la configuration de
production, sous PLAFOND DETERMINISTE d'appels au solveur entier, chaque
instance rejouee sur plusieurs graines d'ALGORITHME. La lecture est
APPARIEE : meme instance, meme graine, on retranche.

DEUX STRATES, et le choix du plafond n'est pas neutre. Sous un plafond
genereux la methode prouve tout et AUCUN reglage ne peut rien montrer : un
banc sature ne mesure pas une insensibilite, il mesure son propre plafond.
Nous avons donc mesure ou se trouve la marge avant de balayer.

  STRATE A  regime cible (n de 5 a 8), plafond 80. La production y prouve
            environ la moitie des lignes : c'est le NOMBRE DE PREUVES qui
            se lit, et il peut monter comme descendre.
  STRATE B  n = 20, corr 0 et 0,5, plafond 300. La production y laisse un
            ecart garanti median d'environ 11 % : c'est l'ECART qui se lit.

A 300 appels sur la strate A l'ecart median vaut 0,00 % et les preuves sont
24/24 ; ce regime est exclu du balayage pour cette raison, et nous le
disons plutot que de publier neuf facteurs a effet nul.

CE QUE LE BANC PEUT CONCLURE, ET CE QU'IL NE PEUT PAS. Il dit si la valeur
de production est defendable dans son voisinage. Il ne dit pas qu'elle est
optimale : un balayage a un facteur ignore les interactions, et huit
instances ne font pas une preuve. Nous rapportons donc l'effet mesure et sa
dispersion, sans en tirer plus.

Usage :  python bench_sensibilite.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys
from typing import Dict, List, Tuple

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P

# les huit instances du regime cible, identiques a bench_cible_determ.py
CIBLE = [
    dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.00),
    dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.25),
    dict(n=6, m=4, p=4, seed=3, rhs_scale=1.5, corr=0.00),
    dict(n=7, m=4, p=3, seed=1, rhs_scale=1.2, corr=0.25),
    dict(n=5, m=3, p=4, seed=2, rhs_scale=1.8, corr=0.00),
    dict(n=8, m=5, p=3, seed=1, rhs_scale=1.0, corr=0.00),
    dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.50),
    dict(n=6, m=4, p=2, seed=1, rhs_scale=1.5, corr=0.25),
]

# configuration de PRODUCTION : c'est le point autour duquel on balaie
PROD = dict(pool_alterne=True, geom_bound=True, archive_cuts=True,
            k_elite=8, k_rand=4,
            menu_normal=("A", "A", "B", "C"),
            menu_divers=("B", "B", "C", "A"),
            lns_frac=(0.4, 0.6), keep_prob=(0.6, 0.4),
            max_stall=3, cut_batch=40, cert_rounds=2)

# un facteur a la fois : nom affiche -> (cle, liste de valeurs)
FACTEURS: List[Tuple[str, str, list]] = [
    ("bases d'elite      k_elite", "k_elite", [2, 4, 8, 16, 32]),
    ("bases au hasard    k_rand",  "k_rand",  [0, 2, 4, 8, 16]),
    ("menu normal",               "menu_normal",
     [("A", "B", "C"), ("A", "A", "B", "C"), ("A", "A", "A", "B", "C"),
      ("A", "B", "B", "C"), ("A", "A", "B", "C", "C")]),
    ("menu diversification",      "menu_divers",
     [("A", "B", "C"), ("B", "B", "C", "A"), ("B", "C"),
      ("B", "B", "B", "C", "A")]),
    ("fraction LNS       lns_frac", "lns_frac",
     [(0.2, 0.4), (0.3, 0.5), (0.4, 0.6), (0.5, 0.7), (0.6, 0.8),
      (0.4, 0.4)]),
    ("garde epsilon      keep_prob", "keep_prob",
     [(0.4, 0.2), (0.6, 0.4), (0.8, 0.6), (0.6, 0.6)]),
    ("stagnation         max_stall", "max_stall", [1, 2, 3, 5, 8]),
    ("lot de coupes      cut_batch", "cut_batch", [10, 20, 40, 80, 160]),
    ("tours de cert.     cert_rounds", "cert_rounds", [1, 2, 4, 6]),
]


# strate B : le regime ou l'ecart reste lisible
DURS = [dict(n=20, m=11, p=3, seed=s, rhs_scale=1.0, corr=c)
        for c in (0.00, 0.50) for s in (1, 2)]

CAP_A, CAP_B = 80, 300
GRAINES_B = 2          # la strate B coute plus cher par execution


def une(inst, cap: int, graine: int, **kw) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=graine, ilp_budget=cap, **kw)
    ecart = (r.gap * 100) if r.gap is not None else float("nan")
    return dict(ecart=ecart, prouve=bool(r.proved_optimal), ilp=r.ilp_calls)


def campagne(graines: List[int], **surcharge) -> Dict:
    """Une configuration, jouee sur les deux strates. Rend les listes
    APPARIEES, dans un ordre fixe."""
    cfg = dict(PROD)
    cfg.update(surcharge)
    out = {}
    for nom, specs, cap, gr in (("A", CIBLE, CAP_A, graines),
                                ("B", DURS, CAP_B, graines[:GRAINES_B])):
        ecarts, preuves = [], 0
        for spec in specs:
            inst = generate(**spec)
            for g in gr:
                d = une(inst, cap, g, **cfg)
                ecarts.append(d["ecart"])
                preuves += int(d["prouve"])
        out[nom] = dict(ecarts=ecarts, preuves=preuves, n=len(ecarts))
    return out


def med(v: List[float]) -> float:
    v = [x for x in v if x == x]
    return statistics.median(v) if v else float("nan")


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    # reprise : « 5- » reprend au sixieme facteur, « 0-3 » n'en fait que
    # quatre. Le balayage est long et un banc qu'on ne peut pas reprendre
    # est un banc qu'on finit par ne plus relancer.
    facteurs = FACTEURS
    if len(sys.argv) > 2:
        a, _, b = sys.argv[2].partition("-")
        facteurs = FACTEURS[int(a or 0): int(b) if b else None]

    print("=" * 108)
    print("SENSIBILITE AUX REGLAGES -- un facteur a la fois, "
          "plafond DETERMINISTE d'appels au solveur entier")
    print(f"  STRATE A  regime cible, n de 5 a 8, plafond {CAP_A}, "
          f"{len(CIBLE)} instances x {len(graines)} graines "
          f"-> on lit les PREUVES")
    print(f"  STRATE B  n = 20, plafond {CAP_B}, "
          f"{len(DURS)} instances x {len(graines[:GRAINES_B])} graines "
          f"-> on lit l'ECART garanti")
    print("=" * 108)

    base = campagne(graines)
    eA0, eB0 = base["A"]["ecarts"], base["B"]["ecarts"]
    print(f"\nPRODUCTION   A : preuves {base['A']['preuves']}/{base['A']['n']}"
          f"   ecart med {med(eA0):6.2f} %"
          f"      B : ecart med {med(eB0):6.2f} %"
          f"   preuves {base['B']['preuves']}/{base['B']['n']}\n")

    entete = (f"{'facteur':<30}{'valeur':>22}"
              f"{'A preuves':>11}{'A ecart':>9}"
              f"{'B ecart':>9}{'B delta':>9}{'B mieux/pire':>14}")
    print(entete)
    print("-" * 108)

    resume = []
    for nom, cle, valeurs in facteurs:
        ampA, ampB = [], []
        for v in valeurs:
            prod = (v == PROD[cle])
            r = base if prod else campagne(graines, **{cle: v})
            eA, eB = r["A"]["ecarts"], r["B"]["ecarts"]
            paires = [(a, b) for a, b in zip(eB, eB0) if a == a and b == b]
            mieux = sum(1 for a, b in paires if a < b - 1e-9)
            pire = sum(1 for a, b in paires if a > b + 1e-9)
            # un ecart NON DEFINI (q_ub infini) n'est pas comparable : le
            # taire ferait passer une paire manquante pour une egalite
            perdues = len(eB) - len(paires)
            ampA.append(r["A"]["preuves"])
            ampB.append(med(eB))
            marque = "  <- production" if prod else ""
            if perdues:
                marque += f"  ({perdues} non comparables)"
            print(f"{nom if v == valeurs[0] else '':<30}{str(v):>22}"
                  f"{r['A']['preuves']:>5}/{r['A']['n']:<5}"
                  f"{med(eA):>9.2f}{med(eB):>9.2f}{med(eB) - med(eB0):>+9.2f}"
                  f"{f'{mieux}/{pire}':>14}{marque}")
        resume.append((nom, max(ampA) - min(ampA), max(ampB) - min(ampB)))
        print("-" * 108)

    print("\nAMPLITUDE DE L'EFFET, par facteur : preuves gagnees ou perdues "
          "sur la strate A,")
    print("points d'ecart entre la meilleure et la pire valeur sur la "
          "strate B.")
    print(f"\n    {'facteur':<34}{'A (preuves)':>13}{'B (points)':>13}")
    for nom, aA, aB in sorted(resume, key=lambda t: -(t[1] + t[2])):
        print(f"    {nom:<34}{aA:>13}{aB:>13.2f}")
    print("\nUn facteur d'amplitude nulle sur les DEUX strates ne pilote rien")
    print("dans les regimes mesures : sa valeur n'a pas besoin d'etre")
    print("justifiee, elle a besoin d'etre DITE -- ce que le memoire ne")
    print("faisait pas. Un facteur d'amplitude non nulle demande, lui, une")
    print("justification ou un reglage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
