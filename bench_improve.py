"""
bench_improve.py
================
Mesure de la matheuristique sur SON REGIME CIBLE (|E| gros, la ou la methode
exacte a echoue dans la campagne), a budget fixe et sur PLUSIEURS GRAINES.

Deux regles de methode, tirees des lecons de la campagne :

* jamais un run unique. La recherche est stochastique ; un tableau a une
  graine illustre un comportement, il ne le mesure pas. On rapporte mediane,
  min et max sur toutes les graines.
* la validite se verifie a chaque run, pas une fois pour toutes :
  q_lb <= q* <= q_ub contre la verite terrain par enumeration.

Le script n'utilise que l'API commune a toutes les versions du module
(`matheuristic_P(inst, time_budget=, bound_budget=, seed=, tightened=)`),
afin de pouvoir etre lance a l'identique sur une version anterieure du code
et servir de temoin d'A/B.

Usage :  python bench_improve.py [n_graines] [etiquette]
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

# instances ou la methode exacte a echoue dans la campagne (statut 'limit'),
# triees par |E| decroissant. Grille identique a campaign.py.
CFG = [
    dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.00),
    dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.25),
    dict(n=6, m=4, p=4, seed=3, rhs_scale=1.5, corr=0.00),
    dict(n=7, m=4, p=3, seed=1, rhs_scale=1.2, corr=0.25),
    dict(n=5, m=3, p=4, seed=2, rhs_scale=1.8, corr=0.00),
    dict(n=8, m=5, p=3, seed=1, rhs_scale=1.0, corr=0.00),
    dict(n=7, m=4, p=4, seed=3, rhs_scale=1.2, corr=0.50),
    dict(n=6, m=4, p=2, seed=1, rhs_scale=1.5, corr=0.25),
]

SEARCH_BUDGET = 7.0
BOUND_BUDGET = 5.0


def main() -> int:
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    label = sys.argv[2] if len(sys.argv) > 2 else "courant"
    seeds = list(range(n_seeds))

    print("=" * 108)
    print(f"MATHEURISTIQUE SUR LE REGIME CIBLE - {n_seeds} graines par instance"
          f"   [variante : {label}]")
    print(f"budget : {SEARCH_BUDGET:.0f} s recherche + {BOUND_BUDGET:.0f} s "
          f"certification")
    print("=" * 108)
    print(f"{'instance':<24}{'|E|':>5}{'q*':>10}"
          f"{'ecart reel %':>26}{'ecart garanti %':>26}"
          f"{'opt':>6}{'prouve':>8}{'valide':>8}")
    print(f"{'':<24}{'':>5}{'':>10}"
          f"{'med':>9}{'min':>8}{'max':>9}{'med':>9}{'min':>8}{'max':>9}"
          f"{'/n':>6}{'/n':>8}{'':>8}")
    print("-" * 108)

    records, all_valid = [], True
    med_real_all, med_gar_all = [], []
    n_opt_all = n_proved_all = n_runs_all = 0

    for cfg in CFG:
        inst = generate(**cfg)
        gt = ground_truth(inst, limit=200_000)
        q = float(gt.q_star)

        real, gar, n_opt, n_proved, valid = [], [], 0, 0, True
        for s in seeds:
            reset_oracle_counter()
            t0 = time.time()
            r = matheuristic_P(inst, time_budget=SEARCH_BUDGET,
                               bound_budget=BOUND_BUDGET, seed=s)
            lb = float(r.q_lb)
            real.append((q - lb) / abs(q) * 100.0)
            gar.append(r.gap * 100.0 if r.gap is not None else float("nan"))
            n_opt += int(abs(lb - q) < 1e-9)
            n_proved += int(r.proved_optimal)
            # VALIDITE : les deux bornes doivent encadrer q*
            ok = (lb <= q + 1e-9) and (r.q_ub is None or r.q_ub >= q - 1e-9)
            valid &= ok
            records.append({
                "instance": inst.name, "E": len(gt.E), "seed": s,
                "q_star": q, "q_lb": lb, "q_ub": r.q_ub,
                "real_gap": real[-1], "guaranteed_gap": gar[-1],
                "proved": bool(r.proved_optimal), "valid": bool(ok),
                "ilp": r.ilp_calls, "rounds": r.rounds,
                "time": time.time() - t0,
                "cert": {k: v for k, v in r.cert.items()
                         if k in ("n_cuts", "rounds", "Dmin", "Dplus",
                                  "restarts", "q_improved", "search_time",
                                  "cert_budget")},
            })
        all_valid &= valid

        rr, gg = np.array(real), np.array(gar)
        med_real_all.append(float(np.median(rr)))
        med_gar_all.append(float(np.nanmedian(gg)))
        n_opt_all += n_opt
        n_proved_all += n_proved
        n_runs_all += len(seeds)

        print(f"{inst.name:<24}{len(gt.E):>5}{q:>10.5f}"
              f"{np.median(rr):>9.1f}{rr.min():>8.1f}{rr.max():>9.1f}"
              f"{np.nanmedian(gg):>9.1f}{np.nanmin(gg):>8.1f}{np.nanmax(gg):>9.1f}"
              f"{n_opt:>4}/{len(seeds):<1}{n_proved:>6}/{len(seeds):<1}"
              f"{'ok' if valid else 'KO':>8}")

    print("-" * 108)
    print(f"ecart reel median (sur les medianes d'instance)     : "
          f"{np.median(med_real_all):.2f} %")
    print(f"ecart garanti median (sur les medianes d'instance)  : "
          f"{np.nanmedian(med_gar_all):.2f} %")
    print(f"optimum atteint       : {n_opt_all}/{n_runs_all} runs")
    print(f"optimalite PROUVEE    : {n_proved_all}/{n_runs_all} runs")
    print(f"VALIDITE q_lb <= q* <= q_ub : "
          f"{'TOUT VALIDE' if all_valid else 'ECHEC'}")

    os.makedirs("results", exist_ok=True)
    out = os.path.join("results", f"bench_improve_{label}.json")
    with open(out, "w") as fh:
        json.dump({"label": label, "seeds": seeds,
                   "search_budget": SEARCH_BUDGET,
                   "bound_budget": BOUND_BUDGET,
                   "median_real_gap": float(np.median(med_real_all)),
                   "median_guaranteed_gap": float(np.nanmedian(med_gar_all)),
                   "n_opt": n_opt_all, "n_proved": n_proved_all,
                   "n_runs": n_runs_all, "all_valid": bool(all_valid),
                   "runs": records}, fh, indent=1)
    print(f"detail par run : {out}")
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
