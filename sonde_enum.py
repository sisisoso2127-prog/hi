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
    CACHE = RACINE / "results" / "cache_ZE"

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
        # POINT DE REPRISE pour le moteur a regions. Sans lui, une
        # instance de vingt minutes tuee par un redemarrage de conteneur
        # repart de zero -- et c'est exactement ce qui avait rendu n=14
        # inatteignable avant que la reprise existe. Le moteur a coupes
        # n'en a pas, donc on ne lui en passe pas.
        if regions:
            r = moteur(inst, time_budget=delai,
                       reprise=SORTIE / f"{nom}.reprise.json")
        else:
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
            # QUATRE etats, et l'oubli du quatrieme etait un faux
            # rapport. L'ancienne cascade tombait dans « aucune des deux
            # n'a fini » des que la force brute renoncait, SANS regarder
            # si le moteur avait abouti -- elle a donc declare inacheves
            # n=14, 16 et 20, dont les enumerations etaient closes par
            # epuisement. Elle se trompait precisement sur les instances
            # qui font l'interet de la sonde : celles hors de portee de
            # la force brute. L'etat « moteur seul a fini » est le
            # resultat cherche, pas un echec.
            # « desaccord » ne designe que le cas ou les DEUX ont fini et
            # ne s'accordent pas ; une enumeration interrompue rend un
            # sous-ensemble, ce qui n'est pas une contradiction.
            verdict=(("accord" if nb == len(r.vecteurs) else "desaccord")
                     if (nb is not None and r.complet)
                     else "moteur seul a fini" if r.complet
                     else "moteur incomplet, brute a fini"
                     if nb is not None
                     else "aucune des deux n'a fini"),
        ), ensure_ascii=False, indent=1), encoding="utf-8")
        part.replace(cible)
        # Le point de reprise ne survit pas au resultat : le garder
        # ferait repartir une enumeration deja consignee.
        if regions and r.complet:
            (SORTIE / f"{nom}.reprise.json").unlink(missing_ok=True)

        # LES VECTEURS, PAS SEULEMENT LEUR NOMBRE. La sonde ne consignait
        # que len(r.vecteurs) : elle repondait a sa propre question -- la
        # portee -- et jetait ce qu'une enumeration de vingt minutes avait
        # coute. Or le tableau des indicateurs se lit CONTRE ces vecteurs.
        # Sans eux, la meme enumeration devait etre refaite par un second
        # banc. Nous ecrivons donc le resultat complet dans le cache
        # partage, sous la cle que bench_livrable_grand.py attend, et le
        # banc des indicateurs le retrouvera sans rien recalculer.
        if r.complet:
            m = max(3, n // 2 + 1)
            CACHE.mkdir(parents=True, exist_ok=True)
            cle = CACHE / (f"n{n}_m{m}_p{p}_s1_c{int(corr * 100):03d}.json")
            if not cle.exists():
                tmp = cle.with_suffix(".part")
                tmp.write_text(json.dumps(dict(
                    complet=True, ilp=r.ilp, motif=r.motif,
                    regions=r.regions, secondes=round(tc, 1),
                    # float() explicite : les composantes sont des
                    # Fraction, que json refuse de serialiser. C'est le
                    # format exact que lit bench_livrable_grand.py.
                    vecteurs=[[float(c) for c in z] for z in r.vecteurs],
                ), ensure_ascii=False), encoding="utf-8")
                tmp.replace(cle)
                print(f"### {nom} : Z(E) ecrit dans le cache partage "
                      f"({cle.name})", flush=True)
        print(f"### {nom} : {len(r.vecteurs)} vecteurs, "
              f"complet={r.complet}, {tc:.1f} s "
              f"| brute {nb if nb is not None else '--'}", flush=True)
    print("### TOUS LES CAS TRAITES")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
