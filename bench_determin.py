#!/usr/bin/env python3
"""
bench_determin.py
=================
Banc a budget DETERMINISTE.

Pourquoi il fallait le construire. Toute la mesure precedente etait a budget
de TEMPS. C'est le cadre realiste, mais il rend inverifiable une affirmation
du type « aucune instance en recul » : deux executions du meme code n'y
rendent pas le meme nombre d'appels, et un ecart de trois appels ne se
distingue pas du bruit. L'isolement du lemme sur 90 instances avait ainsi
donne 1 instance en hausse une fois, 3 une autre fois, toutes avec ZERO
cloture etablie par le lemme -- donc sans que le lemme y fasse quoi que ce
soit. Ce n'etait pas un cout, c'etait du bruit, et rien dans ce cadre ne
permettait de le dire.

Plafonner en NOMBRE D'APPELS au solveur entier retire l'horloge de la boucle
de decision. Deux consequences, et la seconde change la question posee :

  1. une execution devient REPRODUCTIBLE -- ce banc le VERIFIE, il ne le
     suppose pas : chaque configuration est jouee DEUX FOIS et les deux
     resultats doivent etre identiques, sans quoi la ligne est signalee et
     la conclusion suspendue ;
  2. a budget d'appels EGAL, le lemme ne peut plus se traduire par « moins
     d'appels » : les appels qu'il epargne sont aussitot depenses ailleurs.
     La question devient donc la bonne : a ressource egale, ACHETE-T-IL PLUS
     DE PREUVES ?

Recul = perdre une preuve, ou rendre une valeur plus faible, ou une borne
superieure plus lache. Les trois sont verifies.

Usage :  python bench_determin.py [plafond] [--all]
"""

import os
import sys
from fractions import Fraction

import numpy as np

from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_instance import MOILFP
from molfp_matheuristic import matheuristic_P

LOT = os.path.join("instances", "lot_v1")

VARIANTES = [
    ("sans lemme", dict(archive_cuts=True, closure_lemma=False,
                        height_rank=False, cut_diversify=False)),
    ("avec lemme", dict(archive_cuts=True, closure_lemma=True,
                        height_rank=False, cut_diversify=False)),
]


def une(inst: MOILFP, cap: int, kw: dict) -> dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=0,
                       ilp_budget=cap, cut_batch=40, cert_rounds=2, **kw)
    return {"q": r.q_lb, "ub": r.q_ub, "prouve": r.proved_optimal,
            "ilp": r.ilp_calls, "cuts": r.cert.get("n_cuts", 0),
            "lemme": r.cert.get("cloture_lemme", 0)}


def _signature(d: dict) -> tuple:
    """Ce qui doit etre identique d'une execution a l'autre."""
    return (str(d["q"]), None if d["ub"] is None else round(d["ub"], 12),
            d["prouve"], d["ilp"], d["cuts"])


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 and \
        not sys.argv[1].startswith("-") else 250
    noms = sorted(f[:-5] for f in os.listdir(LOT)
                  if f.endswith(".json") and f != "manifest.json")
    if "--all" not in sys.argv:
        noms = noms[::4]

    print("=" * 104)
    print(f"BANC DETERMINISTE - plafond {cap} appels au solveur ENTIER, "
          f"{len(noms)} instances")
    print("=" * 104)
    print(f"{'instance':<26}{'reproductible':>14}"
          f"{'prouve A/B':>12}{'q_lb A/B':>22}{'q_ub A':>11}{'q_ub B':>11}"
          f"{'lemme':>7}{'verdict':>9}")
    print("-" * 104)

    gagne = perdu = egal = 0
    recul_valeur = recul_borne = 0
    non_reprod = 0
    for nom in noms:
        inst = MOILFP.load(os.path.join(LOT, nom + ".json"))
        res, reprod = {}, True
        for lbl, kw in VARIANTES:
            a = une(inst, cap, kw)
            b = une(inst, cap, kw)          # meme configuration, deux fois
            reprod &= (_signature(a) == _signature(b))
            res[lbl] = a
        non_reprod += 0 if reprod else 1

        A, B = res["sans lemme"], res["avec lemme"]
        if B["prouve"] and not A["prouve"]:
            gagne += 1
        elif A["prouve"] and not B["prouve"]:
            perdu += 1
        else:
            egal += 1
        # recul sur la valeur ou sur la borne
        rv = B["q"] < A["q"]
        rb = (A["ub"] is not None and
              (B["ub"] is None or B["ub"] > A["ub"] + 1e-9))
        recul_valeur += int(rv)
        recul_borne += int(rb)
        verdict = "ok"
        if not reprod:
            verdict = "ALEA"
        elif rv or rb or (A["prouve"] and not B["prouve"]):
            verdict = "RECUL"

        ub_a = "-" if A["ub"] is None else f"{A['ub']:.4f}"
        ub_b = "-" if B["ub"] is None else f"{B['ub']:.4f}"
        print(f"{nom:<26}{'oui' if reprod else 'NON':>14}"
              f"{str(A['prouve'])[0]}/{str(B['prouve'])[0]:>10}"
              f"{float(A['q']):>11.4f}{float(B['q']):>11.4f}"
              f"{ub_a:>11}{ub_b:>11}{B['lemme']:>7}{verdict:>9}", flush=True)

    print("-" * 104)
    print(f"  REPRODUCTIBILITE : {len(noms) - non_reprod}/{len(noms)} "
          f"instances identiques d'une execution a l'autre"
          f"{'' if non_reprod == 0 else '   *** le banc n est pas deterministe ***'}")
    print(f"  preuves : gagnees {gagne}   perdues {perdu}   inchangees {egal}")
    print(f"  reculs   : valeur {recul_valeur}   borne superieure {recul_borne}")
    if non_reprod == 0 and perdu == 0 and recul_valeur == 0 and recul_borne == 0:
        print(f"  => a budget d'appels EGAL, aucune instance en recul, "
              f"sur les {len(noms)} testees.")
    else:
        print("  => l'affirmation « aucune instance en recul » est REFUTEE "
              "sur ce lot.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
