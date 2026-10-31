#!/usr/bin/env python3
"""
cache_brute.py
==============
LA VERITE TERRAIN PAR FORCE BRUTE, QUAND ELLE EST LA MOINS CHERE.

Pourquoi ce fichier existe. Le moteur a regions a ete ecrit pour les
instances dont S est hors d'atteinte -- n=14, 16, 20 a p=3, ou la force
brute bute sur ses deux millions de points realisables. Il n'est pas
meilleur partout. Sur n=12 p=4, S est parcourable et le front est epais
(1874 vecteurs) : la decomposition en regions y a depense 7500 s sans
clore, pendant que la force brute tranche en 139 s. Le front epais est
precisement le regime ou le nombre de regions explose, puisque chaque
vecteur trouve en engendre p nouvelles.

La regle est donc : la force brute quand S est parcourable, les regions
quand il ne l'est pas. Ce fichier applique la premiere moitie, et ecrit
dans le MEME cache que bench_livrable_grand.py lit -- un Z(E) exact est
un Z(E) exact, quelle que soit la methode qui l'a produit. Le fichier
consigne laquelle, pour que la provenance reste lisible.

Usage :  python cache_brute.py            tout le lot
         python cache_brute.py 12 4       une seule instance (n, p)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from molfp_instance import generate
from molfp_enum import ground_truth

RACINE = Path(__file__).resolve().parent
CACHE = RACINE / "results" / "cache_ZE"

# Les instances ou la force brute est la bonne methode : S parcourable.
# n=20 n'y est pas -- plus de deux millions de points realisables.
LOT = [(10, 4), (12, 4)]


def remplir(n: int, p: int, limite: int = 2_000_000) -> str:
    m = max(3, n // 2 + 1)
    cle = CACHE / f"n{n}_m{m}_p{p}_s1_c000.json"
    if cle.exists():
        d = json.loads(cle.read_text(encoding="utf-8"))
        return (f"n={n} p={p} : deja en cache "
                f"({len(d['vecteurs'])} vecteurs, {d.get('motif')})")

    inst = generate(n=n, m=m, p=p, seed=1, rhs_scale=1.0, corr=0.00)
    t0 = time.time()
    try:
        gt = ground_truth(inst, limit=limite)
    except Exception as e:
        return f"n={n} p={p} : HORS DE PORTEE -- {str(e)[:70]}"
    tc = time.time() - t0

    # Vecteurs DISTINCTS : c'est l'unite des deux cotes de la
    # proposition de couverture, et celle du cache.
    vect = sorted({tuple(float(c) for c in inst.criteria(x)) for x in gt.E})

    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cle.with_suffix(".part")
    tmp.write_text(json.dumps(dict(
        complet=True,
        motif=f"force brute exacte ({len(gt.S)} points realisables)",
        secondes=round(tc, 1),
        # Pas d'appels ILP ni de regions : la force brute n'en pose pas.
        # Mettre 0 serait lisible comme « gratuit » ; None dit « sans
        # objet », ce qui est le cas.
        ilp=None, regions=None,
        methode="force brute",
        vecteurs=[list(v) for v in vect],
    ), ensure_ascii=False), encoding="utf-8")
    tmp.replace(cle)
    return (f"n={n} p={p} : {len(vect)} vecteurs, {len(gt.S)} points "
            f"realisables, {tc:.1f} s -> {cle.name}")


def main() -> int:
    if len(sys.argv) >= 3:
        print(remplir(int(sys.argv[1]), int(sys.argv[2])), flush=True)
        return 0
    for n, p in LOT:
        print(remplir(n, p), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
