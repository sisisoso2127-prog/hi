#!/usr/bin/env python3
"""
bench_temps.py
==============
CE QUE LE PROTOCOLE DETERMINISTE NE VOIT PAS : LE TEMPS.

Tous les bancs de ce depot lisent sous PLAFOND D'APPELS AU SOLVEUR ENTIER,
et c'est un choix defendable : le nombre d'appels ne depend ni de la
machine, ni de la charge, ni de la version du solveur, donc une mesure
publiee reste reproductible. Mais ce choix a un angle mort, et le memoire
le dit lui-meme (remarque « genou ») : POSER DEUX FOIS PLUS DE COUPES NE SE
VOIT PAS DANS UN COMPTEUR D'APPELS. Un lot de coupes plus gros alourdit
chaque programme entier -- plus de lignes, plus de binaires -- sans en
creer un seul de plus. Le plafond est respecte a l'identique et le cout
est invisible.

L'etude de sensibilite a conclu que `cut_batch = 80` et `cert_rounds = 4`
sont MEILLEURS que les valeurs de production 40 et 2 : -1,19 point d'ecart
median sur la strate B, quatre instances ameliorees contre zero degradee,
preuves inchangees. La production n'a pourtant pas ete changee, et pour une
seule raison : on ignorait ce que ces valeurs coutent. Ce banc mesure ce
cout.

PROTOCOLE.
  - Lecture APPARIEE et ENTRELACEE. Pour chaque (instance, graine) les
    configurations sont jouees l'une apres l'autre, dans le meme tour de
    boucle. On ne joue pas toute la production puis tout le variant : une
    machine partagee derive, et une derive lente se lirait comme un effet.
  - On publie le RAPPORT de temps, pas les secondes. Les secondes
    dependent de cette machine ; le rapport, beaucoup moins.
  - TEMOIN. La production est jouee DEUX FOIS par ligne. La methode est
    deterministe a graine fixee : les deux executions font exactement le
    meme travail, et l'ecart entre leurs deux temps est le BRUIT DU BANC.
    Tout rapport plus proche de 1 que ce bruit est illisible, et nous le
    disons au lieu de l'interpreter.
  - Le compteur d'appels entiers est rapporte a cote du temps. S'il ne
    bouge pas alors que le temps monte, le surcout est bien PAR APPEL --
    des programmes plus lourds -- et non un travail supplementaire.

Usage :  python bench_temps.py [strates] [graines]
         strates : « AB » (defaut), « A » ou « B »
         graines : nombre de graines d'algorithme (defaut 2)
"""

from __future__ import annotations

import statistics
import sys
import time
from typing import Dict, List, Tuple

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P

# memes lots que bench_sensibilite.py : on mesure le cout des reglages
# LA OU leur effet a ete lu, sinon les deux bancs ne parlent pas du meme
# regime.
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
DURS = [dict(n=20, m=11, p=3, seed=s, rhs_scale=1.0, corr=c)
        for c in (0.00, 0.50) for s in (1, 2)]

CAP_A, CAP_B = 80, 300

PROD = dict(pool_alterne=True, geom_bound=True, archive_cuts=True,
            k_elite=8, k_rand=4,
            menu_normal=("A", "A", "B", "C"),
            menu_divers=("B", "B", "C", "A"),
            lns_frac=(0.4, 0.6), keep_prob=(0.6, 0.4),
            max_stall=3, cut_batch=40, cert_rounds=2)

# La premiere entree est la reference ; la seconde est le TEMOIN, qui est
# la reference rejouee -- c'est volontaire, et c'est la mesure du bruit.
CONFIGS: List[Tuple[str, str, dict]] = [
    ("production  b=40 r=2", "b40 r2", {}),
    ("temoin      b=40 r=2", "temoin", {}),
    ("lot double  b=80 r=2", "b80 r2", dict(cut_batch=80)),
    ("tours x2    b=40 r=4", "b40 r4", dict(cert_rounds=4)),
    ("les deux    b=80 r=4", "b80 r4", dict(cut_batch=80, cert_rounds=4)),
]


def une(inst, cap: int, graine: int, **kw) -> Dict:
    cfg = dict(PROD)
    cfg.update(kw)
    reset_oracle_counter()
    t0 = time.perf_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=graine, ilp_budget=cap, **cfg)
    t = time.perf_counter() - t0
    return dict(t=t, ilp=r.ilp_calls, prouve=bool(r.proved_optimal),
                ecart=(r.gap * 100) if r.gap is not None else float("nan"))


def med(v: List[float]) -> float:
    v = [x for x in v if x == x]
    return statistics.median(v) if v else float("nan")


def strate(nom: str, specs: List[dict], cap: int,
           graines: List[int]) -> Dict[str, List[Dict]]:
    """Rend, par configuration, la liste APPARIEE des lectures : meme
    ordre (instance, graine) pour toutes les configurations."""
    par_config: Dict[str, List[Dict]] = {c: [] for c, _, _ in CONFIGS}
    print(f"\n{'='*104}")
    print(f"STRATE {nom}  --  {len(specs)} instances x {len(graines)} "
          f"graines, plafond {cap} appels entiers")
    print("=" * 104)
    entete = (f"{'instance':<26}{'gr':>3}" +
              "".join(f"{court:>13}" for _, court, _ in CONFIGS))
    print(entete)
    print("-" * 104)
    for spec in specs:
        inst = generate(**spec)
        etiq = (f"n={spec['n']} m={spec['m']} p={spec['p']} "
                f"s={spec['seed']} c={spec['corr']:.2f}")
        for g in graines:
            ligne = []
            for cnom, _, sur in CONFIGS:
                # entrelacement : toutes les configurations de CETTE ligne
                # sont jouees a la suite, donc sous la meme charge machine
                d = une(inst, cap, g, **sur)
                par_config[cnom].append(d)
                ligne.append(d)
            print(f"{etiq:<26}{g:>3}" +
                  "".join(f"{d['t']:>13.2f}" for d in ligne), flush=True)
    return par_config


def lire(nom: str, par_config: Dict[str, List[Dict]]) -> float:
    """Rapports apparies contre la production. Rend le bruit du banc."""
    ref = par_config[CONFIGS[0][0]]
    print(f"\n  LECTURE APPARIEE -- strate {nom}")
    print(f"    {'configuration':<24}{'rapport med':>13}{'min':>9}{'max':>9}"
          f"{'appels med':>13}{'ecart med':>12}{'preuves':>9}")
    bruit = 1.0
    for cnom, _, _ in CONFIGS:
        v = par_config[cnom]
        rapports = [d["t"] / r["t"] for d, r in zip(v, ref) if r["t"] > 0]
        rm = med(rapports)
        if cnom == CONFIGS[1][0]:           # le temoin
            bruit = max(abs(x - 1.0) for x in rapports) if rapports else 0.0
        print(f"    {cnom:<24}{rm:>13.3f}{min(rapports):>9.3f}"
              f"{max(rapports):>9.3f}"
              f"{med([d['ilp'] for d in v]):>13.1f}"
              f"{med([d['ecart'] for d in v]):>12.2f}"
              f"{sum(d['prouve'] for d in v):>6}/{len(v):<3}")
    print(f"\n    BRUIT DU BANC (temoin, ecart maximal a 1) : "
          f"{bruit*100:.1f} %")
    return bruit


def main() -> int:
    strates = sys.argv[1].upper() if len(sys.argv) > 1 else "AB"
    graines = list(range(int(sys.argv[2]) if len(sys.argv) > 2 else 2))

    print("=" * 104)
    print("COUT EN TEMPS DES REGLAGES QUE LA SENSIBILITE DIT MEILLEURS")
    print("  cut_batch 40 -> 80,  cert_rounds 2 -> 4,  et les deux ensemble")
    print("  temps de paroi en secondes, plafond d'appels entiers INCHANGE")
    print("=" * 104)

    bruits = []
    for nom, specs, cap in (("A", CIBLE, CAP_A), ("B", DURS, CAP_B)):
        if nom not in strates:
            continue
        pc = strate(nom, specs, cap, graines)
        bruits.append(lire(nom, pc))

    print("\n" + "=" * 104)
    print("CE QUE CE BANC ETABLIT. Le surcout en temps des valeurs que la")
    print("sensibilite prefere, a plafond d'appels entiers identique. Un")
    print("rapport superieur au bruit du temoin est un cout reel ; un")
    print("rapport dans le bruit ne se lit pas, et nous ne le lisons pas.")
    print("CE QU'IL N'ETABLIT PAS. Un rapport mesure sur cette machine,")
    print("avec ce solveur. Il ne se transporte pas tel quel ; ce qui se")
    print("transporte, c'est l'ordre de grandeur et le signe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
