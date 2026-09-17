#!/usr/bin/env python3
"""
bench_confirm_beta.py
=====================
AVANT DE CHANGER UN DEFAUT, ELARGIR L'ECHANTILLON.

Le balayage de sensibilite donne `cut_batch = 80` et `cert_rounds = 4`
meilleurs que les valeurs de production 40 et 2 : -1,19 point d'ecart
median sur la strate B, quatre instances ameliorees contre zero degradee,
preuves inchangees. C'est le seul facteur du balayage dont l'effet soit
unanime, et c'est tentant.

Mais la strate B ne compte que HUIT executions, et ce memoire a deja
publie -- remarque « une mediane qui ment » -- qu'une mediane lue sur huit
executions n'est pas un juge : deux autres facteurs y paraissaient battre
la production de sept a neuf points, et le lot elargi les a renvoyes a
l'egalite. Changer le defaut de production sur huit lectures serait
exactement l'erreur que nous avons documentee. Nous appliquons donc a
$\\beta$ et $\\rho$ le traitement que nous avons applique a la fraction LNS.

Meme lot elargi que `bench_confirm_lns.py` : huit instances a n = 20 et
n = 30, plusieurs graines, sous plafond deterministe d'appels entiers.

CE QUI EST LU. L'ecart garanti median et moyen ; le COMPTE APPARIE (lignes
ameliorees contre degradees) ; le TEST DU SIGNE exact bilateral sur ce
compte, parce qu'un « 4 contre 0 » sur quatre lignes qui bougent n'est pas
le meme evenement qu'un « 12 contre 0 » sur douze ; et le nombre de
preuves, qui est la grandeur qu'on n'a pas le droit de degrader.

Usage :  python bench_confirm_beta.py [graines] [plafond]
"""

from __future__ import annotations

import math
import statistics
import sys
from typing import Dict, List

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P
from bench_sensibilite import PROD

LOT = [dict(n=n, m=max(3, n // 2 + 1), p=3, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for c in (0.00, 0.50) for s in (1, 2)]

CONFIGS = [
    ("production        b=40  r=2", {}),
    ("lot double        b=80  r=2", dict(cut_batch=80)),
    ("lot quadruple     b=160 r=2", dict(cut_batch=160)),
    ("tours doubles     b=40  r=4", dict(cert_rounds=4)),
    ("les deux          b=80  r=4", dict(cut_batch=80, cert_rounds=4)),
]


def signe_bilateral(mieux: int, pire: int) -> float:
    """Test du signe exact, bilateral. Les egalites sont ecartees, comme
    le veut le test : elles ne portent aucune information de direction."""
    n = mieux + pire
    if n == 0:
        return 1.0
    k = min(mieux, pire)
    queue = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * queue)


def campagne(graines: List[int], cap: int, **surcharge) -> Dict:
    cfg = dict(PROD)
    cfg.update(surcharge)
    ecarts, preuves = [], 0
    for spec in LOT:
        inst = generate(**spec)
        for g in graines:
            reset_oracle_counter()
            r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                               seed=g, ilp_budget=cap, **cfg)
            ecarts.append((r.gap * 100) if r.gap is not None
                          else float("nan"))
            preuves += int(r.proved_optimal)
    return dict(ecarts=ecarts, preuves=preuves, n=len(ecarts))


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    print("=" * 104)
    print(f"CONFIRMATION DE beta ET rho -- lot elargi : {len(LOT)} "
          f"instances (n = 20 et 30) x {len(graines)} graines, "
          f"plafond {cap}")
    print("=" * 104)

    base = campagne(graines, cap)
    e0 = base["ecarts"]
    fin0 = [x for x in e0 if x == x]
    print(f"\n{'configuration':<30}{'ecart med':>11}{'moyen':>9}"
          f"{'delta med':>11}{'mieux/pire':>12}{'signe p':>10}"
          f"{'preuves':>10}")
    print("-" * 104)
    for nom, sur in CONFIGS:
        r = base if not sur else campagne(graines, cap, **sur)
        e = r["ecarts"]
        fin = [x for x in e if x == x]
        paires = [(a, b) for a, b in zip(e, e0) if a == a and b == b]
        mieux = sum(1 for a, b in paires if a < b - 1e-9)
        pire = sum(1 for a, b in paires if a > b + 1e-9)
        dm = statistics.median(fin) - statistics.median(fin0)
        p = signe_bilateral(mieux, pire)
        print(f"{nom:<30}{statistics.median(fin):>11.2f}"
              f"{statistics.mean(fin):>9.2f}{dm:>+11.2f}"
              f"{f'{mieux}/{pire}':>12}{p:>10.3f}"
              f"{r['preuves']:>5}/{r['n']:<4}", flush=True)
    print("-" * 104)
    print("Lecture. Le defaut ne change que si TROIS choses concordent : la")
    print("mediane baisse, le compte apparie penche du meme cote, et le test")
    print("du signe ecarte le hasard -- sans qu'aucune preuve soit perdue.")
    print("Deux sur trois ne suffisent pas ; c'est la lecon de la remarque")
    print("sur la mediane bimodale.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
