#!/usr/bin/env python3
"""
bench_filter.py
===============
A/B du LEMME DE CLOTURE PAR LA HAUTEUR sur le lot fige.

    A = coupes d'archive, cloture etablie par un ILP  (avant)
    B = coupes d'archive, cloture etablie par un PL quand h <= 0  (apres)

Le lemme ne change RIEN au jeu de coupes : meme vivier, meme ordre, memes
coupes posees. Seul le moyen d'etablir la cloture change. Le nombre d'appels
au solveur entier ne peut donc que baisser, la valeur rendue et les preuves
sont necessairement identiques -- et c'est ce que ce banc verifie plutot que
de le supposer.

Ce que le filtre change, et pourquoi il fallait le mesurer. La lecture
APPARIEE du lot avait etabli que les coupes d'archive coutent la ou E est
epais : la mediane du delta d'appels au solveur entier valait +1, avec +14
a corr = 0. Le mecanisme etait identifie : le vivier etant classe par
substitut decroissant, et un point EFFICACE ayant par construction un
substitut <= 0, les points d'archive se rangeaient en queue, derriere des
points domines -- mais pas assez loin pour ne jamais etre atteints. Chacun
coutait alors un ILP de cloture et une place sous le plafond, pour une coupe
qui, la ou E est epais, ne retranchait rien qui contraigne la borne.

Le filtre remplace la cle de classement par la HAUTEUR DE REGION
(`region_height`), qui met les deux sources sur la meme echelle, et rejette
les candidats de hauteur <= 0 -- mais seulement quand les candidats utiles
suffisent a saturer le plafond. La ou E est mince, ou le relache doit se
VIDER pour que la proposition du relache vide s'applique, rien n'est rejete.

Sortie : comparaison APPARIEE, instance par instance, entre
    A = coupes d'archive, classement par substitut   (avant)
    B = coupes d'archive, classement par hauteur + filtre   (apres)

Usage :  python bench_filter.py [budget] [--all]
         sans --all, se limite aux instances ou l'ecart mesure etait le plus
         defavorable, plus un temoin a front mince.
"""

import os
import statistics as st
import sys
import time
from fractions import Fraction

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_instance import MOILFP
from molfp_matheuristic import matheuristic_P

LOT = os.path.join("instances", "lot_v1")

# instances ou la lecture appariee du lot donnait le plus fort surcout,
# plus quatre temoins a front mince ou la methode gagnait deja.
CIBLES = [
    "molfp_n8_m5_p4_c000_s2", "molfp_n8_m5_p4_c000_s1",
    "molfp_n10_m6_p3_c000_s2", "molfp_n15_m8_p3_c000_s1",
    "molfp_n5_m3_p4_c000_s1", "molfp_n6_m4_p4_c000_s1",
    "molfp_n20_m11_p3_c050_s2", "molfp_n6_m4_p3_c000_s1",
    # temoins : front mince, la ou le filtre ne doit RIEN changer
    "molfp_n5_m3_p2_c090_s2", "molfp_n5_m3_p3_c090_s1",
    "molfp_n7_m4_p2_c090_s1", "molfp_n8_m5_p3_c090_s2",
]


def charger(nom: str) -> MOILFP:
    return MOILFP.load(os.path.join(LOT, nom + ".json"))


def une(inst: MOILFP, budget: float, filtre: bool) -> dict:
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=budget * 0.6,
                       bound_budget=budget * 0.4, seed=0,
                       archive_cuts=True, closure_lemma=filtre,
                       cut_batch=40, cert_rounds=2)
    sel = r.cert.get("select") or {}
    return {"q": r.q_lb, "ub": r.q_ub, "prouve": r.proved_optimal,
            "ilp": r.ilp_calls, "lp": ORACLE_CALLS["lp"],
            "t": time.time() - t0,
            "coupes": r.cert.get("n_cuts", 0),
            "arch": r.cert.get("archive_cuts", 0),
            "vains": sel.get("vains_arch", 0),
            "lemme": r.cert.get("cloture_lemme", 0),
            "filtre_actif": sel.get("filtre_actif", False)}


def main() -> None:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 and \
        not sys.argv[1].startswith("-") else 12.0
    if "--all" in sys.argv:
        noms = sorted(f[:-5] for f in os.listdir(LOT)
                      if f.endswith(".json") and f != "manifest.json")
    else:
        noms = CIBLES

    print("=" * 104)
    print(f"A/B DU LEMME DE CLOTURE - {len(noms)} instances, budget "
          f"{budget:g} s par variante, comparaison APPARIEE")
    print("=" * 104)
    print(f"{'instance':<26}{'q identique':>12}"
          f"{'prouve A/B':>12}{'ILP A':>8}{'ILP B':>8}{'dILP':>7}"
          f"{'LP B':>7}{'coupes A/B':>12}{'arch A/B':>10}"
          f"{'lemme':>7}{'filtre':>8}")
    print("-" * 104)

    d_ilp, gagne, perdu, egal, tout_valide = [], 0, 0, 0, True
    for nom in noms:
        inst = charger(nom)
        A = une(inst, budget, filtre=False)
        B = une(inst, budget, filtre=True)
        # la valeur rendue ne doit pas changer : le filtre ne touche pas la
        # recherche, seulement le choix des coupes.
        meme = (A["q"] == B["q"])
        ok = True
        for r in (A, B):
            if r["ub"] is not None and r["q"] is not None:
                ok &= r["ub"] >= float(r["q"]) - 1e-9
        tout_valide &= ok
        if B["prouve"] and not A["prouve"]:
            gagne += 1
        elif A["prouve"] and not B["prouve"]:
            perdu += 1
        else:
            egal += 1
        d = B["ilp"] - A["ilp"]
        d_ilp.append(d)
        print(f"{nom:<26}{'oui' if meme else 'NON':>12}"
              f"{str(A['prouve'])[0]}/{str(B['prouve'])[0]:>10}"
              f"{A['ilp']:>8}{B['ilp']:>8}{d:>+7}"
              f"{B['lp']:>7}"
              f"{A['coupes']:>6}/{B['coupes']:<5}"
              f"{A['arch']:>5}/{B['arch']:<4}"
              f"{B['lemme']:>7}"
              f"{('oui' if B['filtre_actif'] else 'non'):>8}", flush=True)

    print("-" * 104)
    d = np.asarray(d_ilp)
    print(f"  delta ILP par paire : mediane {np.median(d):+.0f}   "
          f"moyenne {d.mean():+.1f}   total {d.sum():+.0f}   "
          f"baisses {(d < 0).sum()}/{len(d)}")
    print(f"  preuves : gagnees {gagne}   perdues {perdu}   inchangees {egal}")
    print(f"  VALIDITE q_lb <= q_ub : "
          f"{'TOUT VALIDE' if tout_valide else 'ECHEC'}")


if __name__ == "__main__":
    sys.exit(main())
