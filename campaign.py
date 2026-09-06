"""
campaign.py
===========
Campagne de difficulte : plan d'experience CONTROLE sur 216 instances.

    n    in {5, 6, 7, 8}
    p    in {2, 3, 4}
    corr in {0, .25, .5, .75, .9, 1}
    3 graines
    -> 4 x 3 x 6 x 3 = 216 instances

`m = max(3, n//2 + 1)` et `rhs_scale` est calibre par taille pour garder |S|
dans un ordre de grandeur comparable, de sorte que l'enumeration exhaustive
reste praticable : c'est elle qui fournit la verite terrain.

Le parametre `corr` pilote la similarite entre criteres, donc la finesse de
`E`, **a domaine `S` rigoureusement identique** : le generateur construit `A`
et `b` independamment de `corr`, et la fonction d'utilite `f` aussi. On
manipule donc la cause supposee au lieu de l'observer.

Deux familles de descripteurs sont enregistrees, et il ne faut pas les
confondre :

  * BON MARCHE, calculables AVANT resolution -- `theta_S` (un seul ILP) et
    `depth_S` (quelques ILP). Ils partent du point `x_S` qui maximise `f` sur
    `S`, obtenu par Dinkelbach (Th. 3) sans aucune enumeration. Ce sont les
    seuls candidats a un predicteur utilisable en pratique.
  * DE MECANISME, qui font intervenir la reponse cherchee -- `relax_gap`
    utilise `q*`. Ils expliquent, ils ne predisent pas.

Sortie : `campaign.csv`, une ligne par instance. L'analyse est dans
`analyze.py` : la campagne ne conclut rien elle-meme, elle mesure.

Usage :  python campaign.py [limite_s] [csv]
"""

from __future__ import annotations

import csv
import sys
import time
from fractions import Fraction

import numpy as np

from molfp_core import (ORACLE_CALLS, efficiency_test, max_f_over_S,
                        reset_oracle_counter)
from molfp_enum import as_key, ground_truth
from molfp_instance import generate
from molfp_oracle import dedup_archive, solve_P

NS = [5, 6, 7, 8]
PS = [2, 3, 4]
CORRS = [0.00, 0.25, 0.50, 0.75, 0.90, 1.00]
SEEDS = [1, 2, 3]
RHS = {5: 1.8, 6: 1.5, 7: 1.2, 8: 1.0}
ENUM_LIMIT = 400_000

FIELDS = [
    "name", "n", "m", "p", "corr", "seed",
    "S", "E", "ratio",
    "q_star", "max_S", "relax_gap",
    "theta_S", "depth_S",
    "status", "outer", "cuts", "ilp", "time",
    "archive", "coverage", "false_pos", "mismatch",
]


def cheap_descriptors(inst) -> tuple:
    """
    `theta_S` et `depth_S`, calcules SANS verite terrain.

    `x_S` maximise `f` sur `S` (Dinkelbach, Th. 3). `theta_S = theta(x_S)`
    mesure a quelle distance de l'efficacite se trouve l'optimum de la
    relaxation ; `depth_S` compte les pas de la chaine de dominance qui l'y
    ramene. Les deux sont bon marche, donc utilisables comme predicteurs --
    c'est precisement ce qu'il s'agit de tester.
    """
    r = max_f_over_S(inst)
    if r.x_star is None:
        return None, None
    cur = r.x_star
    theta_S = depth = None
    for step in range(200):
        e = efficiency_test(inst, cur)
        if theta_S is None:
            theta_S = int(e.theta)
        if e.efficient:
            depth = step
            break
        cur = e.dominator
    return theta_S, depth


def run_one(n: int, m: int, p: int, corr: float, seed: int,
            limit: float) -> dict:
    inst = generate(n=n, m=m, p=p, seed=seed,
                    rhs_scale=RHS[n], corr=corr)
    gt = ground_truth(inst, limit=ENUM_LIMIT)

    theta_S, depth_S = cheap_descriptors(inst)
    relax = (float((gt.q_max_S - gt.q_star) / abs(gt.q_max_S))
             if gt.q_max_S else 0.0)

    reset_oracle_counter()
    t0 = time.time()
    r = solve_P(inst, time_limit=limit)
    elapsed = time.time() - t0

    arch = dedup_archive(r.archive)
    truth = {as_key(x) for x in gt.E}
    false_pos = sum(1 for x in arch if as_key(x) not in truth)
    coverage = 100.0 * sum(1 for x in arch if as_key(x) in truth) / max(1, len(gt.E))

    # DIVERGENCE : un statut 'optimal' qui ne rend pas q* est une faute grave.
    # Un statut 'limit' n'est pas une divergence : il n'affirme rien.
    mismatch = int(r.status == "optimal" and r.q_star != gt.q_star)

    return {
        "name": inst.name, "n": n, "m": m, "p": p, "corr": corr, "seed": seed,
        "S": len(gt.S), "E": len(gt.E),
        "ratio": len(gt.E) / max(1, len(gt.S)),
        "q_star": float(gt.q_star), "max_S": float(gt.q_max_S),
        "relax_gap": relax,
        "theta_S": theta_S, "depth_S": depth_S,
        "status": r.status, "outer": r.outer_iterations,
        "cuts": r.total_cuts, "ilp": r.ilp_calls, "time": elapsed,
        "archive": len(arch), "coverage": coverage,
        "false_pos": false_pos, "mismatch": mismatch,
    }


def main() -> int:
    limit = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    out = sys.argv[2] if len(sys.argv) > 2 else "campaign.csv"

    total = len(NS) * len(PS) * len(CORRS) * len(SEEDS)
    print(f"campagne : {total} instances, limite {limit:.0f} s par resolution")
    print(f"{'#':>4}  {'instance':<26}{'|S|':>7}{'|E|':>6}"
          f"{'statut':>9}{'ILP':>7}{'t(s)':>7}{'div.':>6}")

    t_start = time.time()
    n_mismatch = n_limit = n_fp = 0
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        i = 0
        for n in NS:
            m = max(3, n // 2 + 1)
            for p in PS:
                for corr in CORRS:
                    for seed in SEEDS:
                        i += 1
                        row = run_one(n, m, p, corr, seed, limit)
                        w.writerow(row)
                        fh.flush()
                        n_mismatch += row["mismatch"]
                        n_fp += row["false_pos"]
                        n_limit += int(row["status"] != "optimal")
                        print(f"{i:>4}  {row['name']:<26}{row['S']:>7}"
                              f"{row['E']:>6}{row['status']:>9}{row['ilp']:>7}"
                              f"{row['time']:>7.1f}"
                              f"{'!!' if row['mismatch'] else '':>6}",
                              flush=True)

    print(f"\n{total} instances en {(time.time()-t_start)/60:.1f} min")
    print(f"arrets sur limite de temps : {n_limit}/{total}")
    print(f"DIVERGENCES avec la verite terrain : {n_mismatch}")
    print(f"faux positifs dans les archives    : {n_fp}")
    print(f"csv : {out}")
    return 0 if (n_mismatch == 0 and n_fp == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
