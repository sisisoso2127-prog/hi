#!/usr/bin/env python3
"""
bench_livrable_grand.py
=======================
LES CINQ INDICATEURS DE L'ARCHIVE, LA OU LA VERITE TERRAIN N'EXISTAIT PAS.

`bench_livrable.py` mesure l'archive contre E enumere -- purete,
cardinalite, couverture, hypervolume, epsilon+ -- et s'arrete a n <= 8.
La borne n'est pas un choix : au-dela, l'enumeration par force brute de
l'ensemble realisable echoue, et sans E il n'y a rien a comparer.
`bench_couverture.py` bute au meme endroit, avec une ligne qui le dit --
« |S| > 400000 : hors de portee de l'enumeration ».

`molfp_regions.py` deplace cette borne. Il enumere Z(E) sans jamais toucher
a S, et le sondage l'a mesure : n=14 en 1582 s (591 vecteurs), n=16 en
1165 s (811 vecteurs), la ou la force brute abandonne. Les indicateurs
deviennent donc calculables a des tailles ou ce memoire etait aveugle, et
c'est tout l'objet de ce banc.

CE QU'IL NE CHANGE PAS. Ni les indicateurs, ni leur code, ni la
configuration de l'archive : `indicateurs.py` est appele tel quel et la
matheuristique tourne sous son plafond habituel. Seule la SOURCE de la
verite terrain change -- regions au lieu de force brute -- et c'est
precisement ce qu'il faut pour que les nouvelles lignes se lisent a cote
des anciennes.

CE QU'IL COUTE. L'enumeration domine tout le reste, de loin. Elle est donc
MISE EN CACHE : les vecteurs criteres sont ecrits une fois pour toutes, et
une instance deja enumeree ne l'est jamais deux fois. Sans cela, un
redemarrage de conteneur -- il y en a eu cinq -- couterait une demi-heure
de calcul par instance a chaque fois.

Usage :  python bench_livrable_grand.py [plafond_ILP] [delai_enum_s]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P
from molfp_regions import enumere_regions
from indicateurs import indicateurs

RACINE = Path(__file__).resolve().parent
CACHE = RACINE / "results" / "cache_ZE"

# les tailles que la force brute n'atteint pas, plus deux temoins qu'elle
# atteint : sans eux, rien ne dirait que la nouvelle source de verite
# terrain s'accorde avec l'ancienne.
LOT = [dict(n=10, m=6, p=3, seed=1, rhs_scale=1.0, corr=0.00),
       dict(n=12, m=7, p=3, seed=1, rhs_scale=1.0, corr=0.00),
       dict(n=14, m=8, p=3, seed=1, rhs_scale=1.0, corr=0.00),
       dict(n=16, m=9, p=3, seed=1, rhs_scale=1.0, corr=0.00)]


def _cle(spec: dict) -> str:
    return (f"n{spec['n']}_m{spec['m']}_p{spec['p']}_s{spec['seed']}"
            f"_c{int(spec['corr'] * 100):03d}")


def verite_terrain(spec: dict, delai: float) -> Optional[List[tuple]]:
    """Z(E) par regions, mis en cache. None si l'enumeration n'a pas fini."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"{_cle(spec)}.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        return [tuple(v) for v in d["vecteurs"]] if d["complet"] else None

    inst = generate(**spec)
    t0 = time.time()
    r = enumere_regions(inst, time_budget=delai)
    part = f.with_suffix(".part")
    part.write_text(json.dumps(dict(
        complet=r.complet, motif=r.motif, secondes=round(time.time() - t0, 1),
        ilp=r.ilp, regions=r.regions,
        vecteurs=[[float(c) for c in z] for z in r.vecteurs]),
        ensure_ascii=False), encoding="utf-8")
    part.replace(f)                     # ecriture atomique
    return [tuple(float(c) for c in z) for z in r.vecteurs] if r.complet \
        else None


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 180
    delai = float(sys.argv[2]) if len(sys.argv) > 2 else 3600.0

    print("=" * 96)
    print("L'ARCHIVE CONTRE Z(E) ENUMERE PAR REGIONS -- "
          f"plafond ILP {cap}")
    print("=" * 96)
    print(f"\n{'instance':<18}{'|Z(E)|':>8}{'|Z(A)|':>8}{'purete':>9}"
          f"{'card %':>9}{'couv %':>9}{'HV %':>9}{'eps+':>9}{'z* in A':>9}")
    print("-" * 96)

    for spec in LOT:
        ZE = verite_terrain(spec, delai)
        etiq = f"n={spec['n']} p={spec['p']} c={spec['corr']:.2f}"
        if ZE is None:
            print(f"{etiq:<18}{'--':>8}   enumeration non terminee dans "
                  f"le budget : pas de verite terrain", flush=True)
            continue

        inst = generate(**spec)
        reset_oracle_counter()
        r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                           seed=spec["seed"], archive_cuts=True,
                           ilp_budget=cap, pool_alterne=True,
                           geom_bound=True)
        ZA = [tuple(float(c) for c in inst.criteria(a)) for a in r.archive]
        if not ZA:
            print(f"{etiq:<18}{len(set(ZE)):>8}   archive vide", flush=True)
            continue

        ind = indicateurs(ZA, ZE)
        dans = "oui" if tuple(
            float(c) for c in inst.criteria(r.x_best)) in set(ZE) else "NON"
        print(f"{etiq:<18}{ind['nE']:>8}{ind['nA']:>8}"
              f"{ind['purete']:>9.3f}{100 * ind['card']:>9.1f}"
              f"{100 * ind['couverture']:>9.1f}{100 * ind['hv']:>9.1f}"
              f"{ind['eps']:>9.3f}{dans:>9}", flush=True)

    print("-" * 96)
    print("« z* in A » verifie que le vecteur critere de l'incumbent est")
    print("EFFICACE -- c'est-a-dire present dans Z(E). C'est le controle")
    print("qui tombe en premier si l'archive ment sur son contenu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
