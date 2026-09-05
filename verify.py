"""
verify.py
=========
Validation V1-V4 des briques exactes de base, contre la verite terrain par
enumeration exhaustive (`molfp_enum`).

  V1  Test d'efficacite (Th. 2) : pour tout x teste, theta(x) == 0 <=> x in E.
      On verifie EN PLUS que le dominateur renvoye domine reellement x : un
      test qui repond juste mais rend un mauvais certificat serait inutilisable
      comme mouvement de recherche locale.
  V2  Dinkelbach (Th. 3) sur S retrouve exactement max_S f, en rationnels.
  V3  Validite de la borne de relaxation q_UB^(0) = max_S f >= q*.
  V4  Point ideal par Dinkelbach == point ideal par force brute.

Regle du projet : aucune brique n'est reputee correcte tant qu'elle n'a pas
ete confrontee a ce module.

Usage :  python verify.py
"""

from __future__ import annotations

import time

import numpy as np

from molfp_core import (ORACLE_CALLS, dinkelbach, efficiency_test,
                        feasibility_rows, ideal_nadir_estimates, max_f_over_S,
                        reset_oracle_counter)
from molfp_enum import as_key, ground_truth
from molfp_instance import generate

CONFIGS = [
    dict(n=4, m=3, p=2, seed=1),
    dict(n=4, m=3, p=3, seed=2),
    dict(n=5, m=3, p=3, seed=1),
    dict(n=5, m=4, p=4, seed=3),
    dict(n=6, m=4, p=3, seed=1),
    dict(n=6, m=4, p=5, seed=2),
    dict(n=8, m=4, p=3, seed=1),
    dict(n=8, m=5, p=4, seed=2),
]

# nombre de points de S soumis au test d'efficacite (V1). L'ensemble de S
# serait redondant et couteux ; on prend tout E -- le cas ou un faux negatif
# serait le plus grave -- plus un echantillon de S.
N_SAMPLE = 60


def v1_efficiency(inst, gt, rng) -> dict:
    """theta(x) == 0 <=> x in E, et le dominateur domine vraiment."""
    truth = {as_key(x) for x in gt.E}

    tested = list(gt.E)
    if len(gt.S) > 0:
        idx = rng.choice(len(gt.S), size=min(N_SAMPLE, len(gt.S)), replace=False)
        tested += [gt.S[i] for i in idx]

    n_bad, n_bad_dom = 0, 0
    for x in tested:
        r = efficiency_test(inst, x)
        if r.efficient != (as_key(x) in truth):
            n_bad += 1
        if not r.efficient:
            # le certificat doit reellement dominer x
            zx, zy = inst.criteria(x), inst.criteria(r.dominator)
            if not (all(a >= b for a, b in zip(zy, zx)) and zy != zx):
                n_bad_dom += 1
    return {"ok": n_bad == 0 and n_bad_dom == 0, "n_tested": len(tested),
            "n_bad": n_bad, "n_bad_dom": n_bad_dom}


def v2_dinkelbach(inst, gt) -> dict:
    """Dinkelbach sur S == max_S f (egalite EXACTE entre rationnels)."""
    r = max_f_over_S(inst)
    return {"ok": r.status == "optimal" and r.q_star == gt.q_max_S,
            "iters": r.iterations, "value": r.q_star}


def v3_bound(inst, gt) -> dict:
    """max_S f >= q* : la relaxation E -> S est une borne superieure valide."""
    gap = float((gt.q_max_S - gt.q_star) / abs(gt.q_max_S)) if gt.q_max_S else 0.0
    return {"ok": gt.q_max_S >= gt.q_star, "relax_gap": gap}


def v4_ideal(inst, gt) -> dict:
    """Point ideal par Dinkelbach == point ideal par force brute."""
    ideal, _ = ideal_nadir_estimates(inst)
    return {"ok": tuple(ideal) == gt.ideal, "ideal": tuple(ideal)}


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 100)
    print("VALIDATION V1-V4 - BRIQUES EXACTES CONTRE LA VERITE TERRAIN")
    print("=" * 100)
    print(f"{'instance':<24}{'|S|':>7}{'|E|':>7}{'|E|/|S|':>9}"
          f"{'V1':>5}{'V2':>5}{'V3':>5}{'V4':>5}"
          f"{'iters Dink.':>13}{'ecart relax.':>14}{'ILP':>7}{'t(s)':>7}")
    print("-" * 100)

    all_ok = True
    for cfg in CONFIGS:
        inst = generate(**cfg)
        t0 = time.time()
        gt = ground_truth(inst, limit=400_000)
        reset_oracle_counter()

        r1 = v1_efficiency(inst, gt, rng)
        r2 = v2_dinkelbach(inst, gt)
        r3 = v3_bound(inst, gt)
        r4 = v4_ideal(inst, gt)
        ok = r1["ok"] and r2["ok"] and r3["ok"] and r4["ok"]
        all_ok &= ok

        def mark(d):
            return "ok" if d["ok"] else "KO"

        print(f"{inst.name:<24}{len(gt.S):>7}{len(gt.E):>7}"
              f"{len(gt.E)/max(1,len(gt.S)):>9.3f}"
              f"{mark(r1):>5}{mark(r2):>5}{mark(r3):>5}{mark(r4):>5}"
              f"{r2['iters']:>13}{r3['relax_gap']*100:>12.2f} %"
              f"{ORACLE_CALLS['ilp']:>7}{time.time()-t0:>7.1f}")
        if not ok:
            print(f"    detail V1 : {r1}")

    print("-" * 100)
    print("V1-V4 : " + ("TOUT VALIDE" if all_ok else "ECHEC"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
