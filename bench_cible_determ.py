"""
bench_cible_determ.py
=====================
LE RESULTAT PRINCIPAL, REJOUE SOUS PLAFOND D'APPELS.

Ce qui est en cause. Le chiffre qui ouvre le memoire -- optimalite prouvee de
58/80 a 78/80 sur le regime cible, 20 preuves gagnees et aucune perdue -- a
ete obtenu sous un budget en TEMPS (7+5 s). Or le memoire etablit lui-meme
qu'un budget en temps fait varier le nombre d'appels entiers de 125 % d'une
execution a l'autre, et il refuse cette unite partout ailleurs. Le chiffre
est donc annonce dans une unite que le protocole recuse.

Et l'objection va plus loin que la reproductibilite. Le texte attribue une
part du gain au theoreme de cloture par la hauteur : chaque cloture etablie
par programme LINEAIRE est un appel entier non depense, « et le temps ainsi
rendu retourne a la certification, qui ferme davantage de bornes ». Sous un
plafond compte en APPELS, ce temps rendu n'est plus une ressource. Le gain
annonce pourrait donc n'etre qu'un effet d'horloge.

Ce banc tranche. Memes 8 instances, memes graines, memes deux bras --
sans coupes d'archive contre avec -- mais plafonne en APPELS ENTIERS. Rien
d'autre ne change. Les deux bras portent la configuration courante (vivier
alterne, seconde route), de sorte que la seule difference entre eux soit la
coupe d'efficacite elle-meme.

Le protocole du memoire est applique a la lettre : lecture APPARIEE (meme
instance, meme graine), double execution de chaque configuration, et toute
ligne instable est exclue de la conclusion.

Usage :  python bench_cible_determ.py [n_graines] [cap1,cap2,...]
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

# les huit instances du regime cible, identiques a bench_archive.py PARTIE A
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

BRAS = [("sans", False), ("avec", True)]


def une_execution(inst, seed: int, arch: bool, cap: int) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=arch, ilp_budget=cap,
                       pool_alterne=True, geom_bound=True)
    return dict(prouve=bool(r.proved_optimal),
                lb=float(r.q_lb) if r.q_lb is not None else float("nan"),
                ub=r.q_ub, ilp=r.ilp_calls,
                gap=(r.gap * 100) if r.gap is not None else float("nan"))


def campagne(cap: int, n_seeds: int) -> Dict:
    print("=" * 104)
    print(f"REGIME CIBLE SOUS PLAFOND DETERMINISTE DE {cap} APPELS  "
          f"[V* S* D*]   {n_seeds} graines")
    print("=" * 104)
    print(f"{'instance':<24}{'|E|':>5}"
          f"{'ecart median':>22}{'optimalite prouvee':>24}"
          f"{'ILP med.':>18}{'det.':>7}")
    print(f"{'':<24}{'':>5}{'sans':>11}{'avec':>11}"
          f"{'sans':>11}{'avec':>13}{'sans':>9}{'avec':>9}")
    print("-" * 104)

    tot = {k: 0 for _, k in BRAS}
    paires = dict(gagne=0, perdu=0, egal=0)
    instables, runs, valides = 0, 0, 0
    d_ilp: List[int] = []
    lignes = []

    for cfg in CIBLE:
        inst = generate(**cfg)
        gt = ground_truth(inst, limit=200_000)
        q = float(gt.q_star)
        g = {k: [] for _, k in BRAS}
        ilps = {k: [] for _, k in BRAS}
        pv = {k: 0 for _, k in BRAS}
        det = {k: True for _, k in BRAS}

        for s in range(n_seeds):
            res = {}
            for _, arch in BRAS:
                a = une_execution(inst, s, arch, cap)
                b = une_execution(inst, s, arch, cap)   # determinisme verifie
                if (a["prouve"], a["ilp"], a["lb"]) != (b["prouve"], b["ilp"],
                                                        b["lb"]):
                    det[arch] = False
                res[arch] = a
                g[arch].append(a["gap"])
                ilps[arch].append(a["ilp"])
                pv[arch] += int(a["prouve"])
                tot[arch] += int(a["prouve"])
                # l'encadrement doit rester valide, c'est non negociable
                ok = (a["lb"] <= q + 1e-9) and (a["ub"] is None
                                                or a["ub"] >= q - 1e-9)
                valides += int(ok)
                runs += 1
            if res[True]["prouve"] and not res[False]["prouve"]:
                paires["gagne"] += 1
            elif res[False]["prouve"] and not res[True]["prouve"]:
                paires["perdu"] += 1
            else:
                paires["egal"] += 1
            d_ilp.append(res[True]["ilp"] - res[False]["ilp"])

        stable = det[False] and det[True]
        instables += int(not stable)
        lignes.append((inst.name, len(gt.E), pv[False], pv[True], stable))
        print(f"{inst.name:<24}{len(gt.E):>5}"
              f"{np.nanmedian(g[False]):>10.1f}%{np.nanmedian(g[True]):>10.1f}%"
              f"{pv[False]:>8}/{n_seeds}{pv[True]:>9}/{n_seeds}"
              f"{int(np.median(ilps[False])):>9}{int(np.median(ilps[True])):>9}"
              f"{'ok' if stable else 'KO':>7}", flush=True)

    print("-" * 104)
    n = n_seeds * len(CIBLE)
    print(f"  optimalite prouvee : sans {tot[False]}/{n}   "
          f"avec {tot[True]}/{n}")
    print(f"  lecture APPARIEE   : gagnees {paires['gagne']}   "
          f"perdues {paires['perdu']}   inchangees {paires['egal']}")
    d = np.asarray(d_ilp)
    print(f"  delta d'appels     : mediane {int(np.median(d)):+d}   "
          f"total {int(d.sum()):+d}   {int((d < 0).sum())}/{len(d)} en baisse")
    print(f"  encadrements valides : {valides}/{runs}")
    print(f"  lignes instables     : {instables}/{len(CIBLE)}"
          + ("   *** a exclure de la conclusion ***" if instables else ""))
    return dict(cap=cap, sans=tot[False], avec=tot[True], n=n, **paires)


if __name__ == "__main__":
    ns = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    caps = ([int(c) for c in sys.argv[2].split(",")]
            if len(sys.argv) > 2 else [100, 300])
    t0 = time.time()
    bilan = []
    for cap in caps:
        bilan.append(campagne(cap, ns))
        print()
    print("=" * 104)
    print("BILAN")
    for b in bilan:
        print(f"  plafond {b['cap']:>4} appels : "
              f"sans {b['sans']}/{b['n']}   avec {b['avec']}/{b['n']}   "
              f"apparie +{b['gagne']} / -{b['perdu']}")
    print(f"  ({time.time()-t0:.0f} s)")
    print("\n=== TERMINE ===")
