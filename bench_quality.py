#!/usr/bin/env python3
"""
bench_quality.py
================
A/B de la QUALITE DE SOLUTION a grande taille, a budget DETERMINISTE.

POURQUOI CETTE VERSION EXISTE. La premiere etait a budget de TEMPS, et ses
chiffres etaient sans valeur. Controle direct, meme code, meme graine, meme
instance (n = 40, corr = 0, graine 2), trois executions a 18 + 12 s :

    q_lb = 26,9286   puis   60,5000   puis   60,5000

Soit 125 % d'ecart entre deux executions de la MEME chose. Tout ce que ce
banc rapportait -- un gain de +170 %, une perte de -55,5 %, un decompte de
3 gains contre 2 pertes -- tenait dans ce bruit. Les memes trois executions
a budget d'APPELS (700) donnent 14,8500 trois fois, avec le meme nombre
d'appels et la meme archive.

La lecon avait pourtant deja ete tiree pour le banc des coupes ; elle
n'avait pas ete appliquee ici. Elle l'est maintenant, et ce banc VERIFIE son
propre determinisme au lieu de le supposer : chaque configuration est jouee
DEUX FOIS, et une ligne dont les deux executions different est marquee ALEA
et exclue de la conclusion.

CE QUI EST COMPARE, par paires (meme instance, meme graine, meme plafond) :

    A  budget de certification integralement consacre a la borne
    B  diversification par coupes d'efficacite FORCEE (gap_hopeless = 0),
       c'est-a-dire declenchee des que l'optimalite n'est pas prouvee

Le bras B n'est PAS le reglage de production (0,5) et ce choix est
deliberate : avec le reglage de production la sonde ne declenche presque
jamais -- sur une instance temoin elle rend un ecart de 23 % la ou le seuil
est a 50 % -- et le banc ne mesurait donc RIEN. On separe ici les deux
questions : ce banc mesure la MECANIQUE (si l'on tire, cela vaut-il le
budget ?), la GACHETTE se calibrant ensuite sur sa reponse.

Lecture. Les deux valeurs sont des points EFFICACES CERTIFIES : q_lb plus
grand = strictement meilleur, sans verite terrain, ce qui est indispensable
a ces tailles ou l'enumeration de E est hors de portee.

Usage :  python bench_quality.py [plafond_appels]
"""

import sys
import time

import numpy as np

from molfp_core import (ORACLE_CALLS, reset_oracle_counter,
                        upper_bound_over_S)
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [20, 30, 40, 50]
GRAINES = [0, 1, 2]
CORRS = [0.0, 0.5]


def une(inst, cap, seed, diversify, gap_hopeless=0.0):
    """
    `gap_hopeless = 0` force la diversification des que l'optimalite n'est pas
    PROUVEE. Ce n'est pas le reglage de production (0,5) mais celui qui permet
    de mesurer le MECANISME plutot que le DECLENCHEUR : avec le reglage de
    production, un premier banc n'a jamais rien mesure du tout, la sonde
    rendant un ecart de 23 % la ou le seuil est a 50 % -- la diversification
    n'etait donc, tres correctement, jamais tentee. Un declencheur qui ne
    tire pas ne dit rien sur l'arme.
    """
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=seed, archive_cuts=True,
                       cut_diversify=diversify, gap_hopeless=gap_hopeless,
                       ilp_budget=cap)
    return {"q": r.q_lb, "ub": r.q_ub, "ilp": r.ilp_calls,
            "arch": len(r.archive),
            "neufs": r.cert.get("diversify_new", 0),
            "prouve": r.proved_optimal}


def _signature(d):
    """Ce qui doit etre identique d'une execution a l'autre."""
    return (str(d["q"]), None if d["ub"] is None else round(d["ub"], 12),
            d["prouve"], d["ilp"], d["arch"], d["neufs"])


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 700

    print("=" * 110)
    print(f"QUALITE DE SOLUTION A GRANDE TAILLE - plafond {cap} appels au "
          f"solveur ENTIER, comparaison APPARIEE")
    print("=" * 110)
    print(f"{'n':>4}{'corr':>6}{'graine':>7}{'reprod':>8}"
          f"{'q_lb A':>12}{'q_lb B':>12}{'gain %':>9}"
          f"{'neufs':>7}{'|arch| A/B':>12}{'ILP A/B':>12}{'coherent':>10}")
    print("-" * 110)

    gains, mieux, pire, egal, tout_ok, alea = [], 0, 0, 0, True, 0
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in GRAINES:
                inst = generate(n=n, m=m, p=3, seed=1 + g,
                                rhs_scale=1.0, corr=corr)
                mS = upper_bound_over_S(inst)
                A, A2 = une(inst, cap, g, False), une(inst, cap, g, False)
                B, B2 = une(inst, cap, g, True), une(inst, cap, g, True)
                reprod = (_signature(A) == _signature(A2)
                          and _signature(B) == _signature(B2))
                alea += 0 if reprod else 1
                qa, qb = float(A["q"]), float(B["q"])
                lo = max(qa, qb)
                hi = min(min(A["ub"] or np.inf, mS), min(B["ub"] or np.inf, mS))
                ok = lo <= hi + 1e-9
                tout_ok &= ok
                gain = (qb - qa) / max(1e-12, abs(qa)) * 100
                if reprod:
                    gains.append(gain)
                    if qb > qa + 1e-12:
                        mieux += 1
                    elif qb < qa - 1e-12:
                        pire += 1
                    else:
                        egal += 1
                print(f"{n:>4}{corr:>6.2f}{g:>7}"
                      f"{('oui' if reprod else 'NON'):>8}"
                      f"{qa:>12.4f}{qb:>12.4f}{gain:>+8.1f}%"
                      f"{B['neufs']:>7}"
                      f"{A['arch']:>6}/{B['arch']:<5}"
                      f"{A['ilp']:>6}/{B['ilp']:<5}"
                      f"{'ok' if ok else 'KO':>10}", flush=True)

    print("-" * 110)
    print(f"  REPRODUCTIBILITE : {len(TAILLES)*len(CORRS)*len(GRAINES)-alea}"
          f"/{len(TAILLES)*len(CORRS)*len(GRAINES)} lignes stables"
          f"{'' if alea == 0 else '   *** lignes ALEA exclues de la conclusion ***'}")
    if gains:
        d = np.asarray(gains)
        print(f"  gain sur q_lb : median {np.median(d):+.2f} %   "
              f"moyen {d.mean():+.2f} %   max {d.max():+.1f} %   "
              f"min {d.min():+.1f} %")
    print(f"  MIEUX {mieux}   PIRE {pire}   EGAL {egal}   "
          f"sur {len(gains)} paires exploitables")
    print(f"  COHERENCE des encadrements : "
          f"{'TOUT VALIDE' if tout_ok else 'ECHEC'}")
    return 0 if tout_ok else 1


if __name__ == "__main__":
    sys.exit(main())
