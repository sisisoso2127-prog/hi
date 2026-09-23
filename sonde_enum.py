#!/usr/bin/env python3
"""
sonde_enum.py
=============
JUSQU'OU L'ENUMERATION PAR COUPES VA-T-ELLE, UNE INSTANCE A LA FOIS.

La question : l'enumeration par coupes d'efficacite atteint-elle des
tailles ou la force brute renonce ? Sur des instances CORRELEES la reponse
est oui et elle est nette -- n = 14, 16 et 20 enumeres en secondes pendant
que la force brute bute sur ses deux millions de points realisables. Sur des
instances NON correlees, dont le front est bien plus peuple, la question
restait ouverte : quatre tentatives de sondage ont ete tuees par des
redemarrages de conteneur avant d'avoir pu repondre.

D'OU LA FORME DE CE FICHIER. Une instance par execution, un fichier par
instance, et un saut de ce qui est deja ecrit. Une interruption ne coute
plus que l'instance en cours, et le sondage reprend ou il en etait. C'est
le dispositif de `reproduire.py`, applique ici parce qu'il a fait ses
preuves : avant lui, trois campagnes entieres perdues ; apres, un seul
morceau.

CE QU'UN DEPASSEMENT DE DELAI SIGNIFIE. Que l'instance est hors de portee
DU BUDGET, ce qui est le resultat cherche -- pas une panne du banc. Le
fichier le consigne comme tel, avec le nombre de vecteurs deja enumeres,
qui reste un sous-ensemble correct de Z(E).

Usage :  python sonde_enum.py [delai_s]            moteur a coupes
         python sonde_enum.py [delai_s] --regions  moteur a regions
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from molfp_instance import generate
from molfp_enumere import enumere
from molfp_regions import enumere_regions

RACINE = Path(__file__).resolve().parent

# (n, p, corr) -- les correlees servent de temoin, elles sont deja connues
CAS = [(10, 3, 0.00), (12, 3, 0.00), (14, 3, 0.00),
       (16, 3, 0.00), (20, 3, 0.00),
       (10, 4, 0.00), (12, 4, 0.00), (20, 4, 0.00)]


def brute(inst, limite):
    """Reference. Rend None quand l'enumeration de S est hors de portee."""
    from molfp_enum import ground_truth
    t0 = time.time()
    try:
        gt = ground_truth(inst, limit=2_000_000)
    except Exception as e:                      # limite de S depassee
        return None, time.time() - t0, str(e)[:60]
    return (len({tuple(inst.criteria(x)) for x in gt.E}),
            time.time() - t0, "")


def main() -> int:
    # Deux moteurs, deux repertoires. Les melanger ferait passer un
    # resultat de l'un pour un resultat de l'autre au premier saut de
    # reprise, et c'est precisement ce que la reprise ne doit pas faire.
    regions = "--regions" in sys.argv
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    delai = float(argv[0]) if argv else 1200.0
    moteur = enumere_regions if regions else enumere
    SORTIE = RACINE / "results" / ("sonde_regions" if regions
                                   else "sonde_enum")
    SORTIE.mkdir(parents=True, exist_ok=True)
    print(f"### moteur : {'regions' if regions else 'coupes'}   "
          f"delai {delai:.0f} s", flush=True)

    for n, p, corr in CAS:
        nom = f"n{n}_p{p}_c{int(corr * 100):03d}"
        cible = SORTIE / f"{nom}.json"
        if cible.exists():
            print(f"### {nom} : deja fait, passe", flush=True)
            continue
        print(f"### {nom} : debut", flush=True)

        inst = generate(n=n, m=max(3, n // 2 + 1), p=p, seed=1,
                        rhs_scale=1.0, corr=corr)
        t0 = time.time()
        r = moteur(inst, time_budget=delai)
        tc = time.time() - t0
        nb, tb, err = brute(inst, delai)

        # ECRITURE ATOMIQUE : un fichier a moitie ecrit serait relu comme
        # un resultat complet par le saut ci-dessus.
        part = cible.with_suffix(".part")
        part.write_text(json.dumps(dict(
            n=n, p=p, corr=corr,
            moteur=("regions" if regions else "coupes"),
            coupes_vecteurs=len(r.vecteurs), coupes_complet=r.complet,
            coupes_motif=r.motif, coupes_s=round(tc, 1), coupes_ilp=r.ilp,
            brute_vecteurs=nb, brute_s=round(tb, 1), brute_erreur=err,
            # Trois etats et non deux. « desaccord » ne doit designer que
            # le cas ou les DEUX methodes ont fini et ne s'accordent pas ;
            # une enumeration interrompue rend un sous-ensemble, ce qui
            # n'est pas une contradiction mais un budget epuise.
            verdict=("accord" if (nb is not None and r.complet
                                  and nb == len(r.vecteurs))
                     else "desaccord" if (nb is not None and r.complet)
                     else "coupes incompletes" if nb is not None
                     else "aucune des deux n'a fini"),
        ), ensure_ascii=False, indent=1), encoding="utf-8")
        part.replace(cible)
        print(f"### {nom} : {len(r.vecteurs)} vecteurs, "
              f"complet={r.complet}, {tc:.1f} s "
              f"| brute {nb if nb is not None else '--'}", flush=True)
    print("### TOUS LES CAS TRAITES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
