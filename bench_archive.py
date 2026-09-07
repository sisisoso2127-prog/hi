"""
bench_archive.py
================
Effet des COUPES D'EFFICACITE issues de l'archive (Th. 6 et 7).

Question posee. La borne du Th. 5' repose sur un relache R obtenu en coupant
autour de points DOMINES. Or la campagne etablit que la profondeur mediane
des chaines de reparation vaut 1 : dans le regime a E epais, les points
domines sont rares et l'archive est grande. Les coupes de dominance n'ont
donc presque rien a mordre la ou la borne en aurait le plus besoin.

Les coupes d'efficacite puisent dans l'archive. On mesure ici si cela
resserre reellement la borne, et surtout si elle reste VALIDE.

Deux parties :
  A  regime cible (E gros), 8 instances x plusieurs graines, verite terrain
     disponible : on verifie q_lb <= q* <= q_ub a chaque execution ;
  B  passage a l'echelle a corr = 0 (E epais), n de 20 a 40, sans verite
     terrain : on verifie la coherence des encadrements et le temoin max_S f.

Usage :  python bench_archive.py [n_graines]
"""

from __future__ import annotations

import sys
import time

import numpy as np

from molfp_core import (reset_oracle_counter, upper_bound_over_S)
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

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
ECHELLE = [20, 30, 40]
VARIANTES = [("sans", 0), ("avec", 40)]


def partie_a(n_seeds: int) -> bool:
    print("=" * 100)
    print(f"PARTIE A - REGIME CIBLE, verite terrain disponible, "
          f"{n_seeds} graines, budget 7 s + 5 s")
    print("=" * 100)
    print(f"{'instance':<24}{'|E|':>5}"
          f"{'ecart garanti median':>26}{'optimalite prouvee':>22}{'valide':>9}")
    print(f"{'':<24}{'':>5}{'sans':>12}{'avec':>14}{'sans':>10}{'avec':>12}")
    print("-" * 100)

    med = {k: [] for _, k in VARIANTES}
    prv = {k: 0 for _, k in VARIANTES}
    runs = 0
    tout_valide = True
    # comparaison APPARIEE : meme instance, MEME graine, les deux variantes.
    # C'est la seule lecture honnete d'un gain -- comparer deux medianes
    # independantes laisse croire a un gain par instance qui n'existe pas.
    paires = {"gagne": 0, "perdu": 0, "egal": 0}
    d_ilp = []
    for cfg in CIBLE:
        inst = generate(**cfg)
        gt = ground_truth(inst, limit=200_000)
        q = float(gt.q_star)
        gaps = {k: [] for _, k in VARIANTES}
        pv = {k: 0 for _, k in VARIANTES}
        valide = True
        for s in range(n_seeds):
            pr, il = {}, {}
            for _, ac in VARIANTES:
                reset_oracle_counter()
                r = matheuristic_P(inst, time_budget=7, bound_budget=5,
                                   seed=s, archive_cuts=ac)
                lo = float(r.q_lb)
                ok = (lo <= q + 1e-9) and (r.q_ub is None or r.q_ub >= q - 1e-9)
                valide &= ok
                gaps[ac].append(r.gap * 100 if r.gap is not None else np.nan)
                prv[ac] += int(r.proved_optimal)
                pv[ac] += int(r.proved_optimal)
                pr[ac], il[ac] = bool(r.proved_optimal), r.ilp_calls
            if pr[40] and not pr[0]:
                paires["gagne"] += 1
            elif pr[0] and not pr[40]:
                paires["perdu"] += 1
            else:
                paires["egal"] += 1
            d_ilp.append(il[40] - il[0])
        runs += n_seeds
        tout_valide &= valide
        for _, ac in VARIANTES:
            med[ac].append(float(np.nanmedian(gaps[ac])))
        print(f"{inst.name:<24}{len(gt.E):>5}"
              f"{np.nanmedian(gaps[0]):>11.1f}%{np.nanmedian(gaps[40]):>13.1f}%"
              f"{pv[0]:>7}/{n_seeds}{pv[40]:>8}/{n_seeds}"
              f"{'ok' if valide else 'KO':>9}", flush=True)

    print("-" * 100)
    for nom, ac in VARIANTES:
        print(f"  {nom:<5} : ecart garanti median (sur les medianes) = "
              f"{np.nanmedian(med[ac]):6.2f} %   "
              f"optimalite prouvee {prv[ac]:>3}/{runs}")
    print(f"  VALIDITE q_lb <= q* <= q_ub : "
          f"{'TOUT VALIDE' if tout_valide else 'ECHEC'}")
    print("  --- lecture APPARIEE (meme instance, meme graine) ---")
    print(f"  preuves gagnees {paires['gagne']:>3}   "
          f"perdues {paires['perdu']:>3}   inchangees {paires['egal']:>3}")
    d = np.asarray(d_ilp)
    print(f"  delta ILP par paire : mediane {np.median(d):+.0f}   "
          f"moyenne {d.mean():+.1f}   total {d.sum():+.0f}   "
          f"baisses {(d < 0).sum()}/{len(d)}")
    return tout_valide


def partie_b() -> bool:
    print()
    print("=" * 100)
    print("PARTIE B - PASSAGE A L'ECHELLE a corr = 0 (E epais), "
          "sans verite terrain, budget 18 s + 12 s")
    print("=" * 100)
    print(f"{'n':>4}{'max_S f':>12}"
          f"{'q_lb sans':>12}{'q_ub sans':>12}{'ecart':>9}"
          f"{'q_lb avec':>12}{'q_ub avec':>12}{'ecart':>9}{'coherent':>10}")
    print("-" * 100)
    tout_ok = True
    for n in ECHELLE:
        m = max(3, n // 2 + 1)
        inst = generate(n=n, m=m, p=3, seed=1, rhs_scale=1.0, corr=0.0)
        mS = upper_bound_over_S(inst)      # +inf si Dinkelbach n'a pas conclu
        res = {}
        for _, ac in VARIANTES:
            reset_oracle_counter()
            r = matheuristic_P(inst, time_budget=18, bound_budget=12,
                               seed=0, archive_cuts=ac)
            hi = min(r.q_ub if r.q_ub is not None else np.inf, mS)
            res[ac] = (float(r.q_lb), hi,
                       (hi - float(r.q_lb)) / max(1e-12, abs(hi)) * 100)
        # coherence : les deux encadrements doivent contenir le meme q*
        lo = max(res[0][0], res[40][0])
        hi = min(res[0][1], res[40][1])
        ok = lo <= hi + 1e-9
        tout_ok &= ok
        print(f"{n:>4}{mS:>12.4f}"
              f"{res[0][0]:>12.4f}{res[0][1]:>12.4f}{res[0][2]:>8.1f}%"
              f"{res[40][0]:>12.4f}{res[40][1]:>12.4f}{res[40][2]:>8.1f}%"
              f"{'ok' if ok else 'KO':>10}", flush=True)
    print("-" * 100)
    print(f"  COHERENCE des encadrements : "
          f"{'TOUT VALIDE' if tout_ok else 'ECHEC'}")
    return tout_ok


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    a = partie_a(n_seeds)
    b = partie_b()
    return 0 if (a and b) else 1


if __name__ == "__main__":
    raise SystemExit(main())
