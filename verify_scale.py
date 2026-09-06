"""
verify_scale.py
===============
Validation AUX TAILLES OU L'ENUMERATION EXHAUSTIVE EST IMPRATICABLE.

Le socle de verite terrain (`molfp_enum`) plafonne vers n = 8 : au-dela, |S|
explose et l'enumeration n'est plus une reference. Tout ce qui precede est
donc mesure sur des instances minuscules, et rien n'y dit ce qui se passe a
n = 30. C'est la faiblesse dominante du projet.

Elle se leve en changeant de reference. Les garanties de la methode sont
AUTO-CERTIFIANTES : chacune se verifie sans connaitre q*.

  W1  LB sain. x_best passe le test d'efficacite (Th. 2, ILP exact), donc
      q_lb = f(x_best) <= q*. Aucune enumeration : c'est la propriete la plus
      importante et elle est verifiable a toute taille.
  W2  Archive saine. Chaque point archive passe le meme test (echantillon
      borne, les ILP devenant chers en n).
  W3  Encadrement coherent : q_lb <= q_ub.
  W4  Coupes sures. Tout point efficace CERTIFIE doit satisfaire la
      disjonction de chaque coupe posee : c'est l'invariant E inclus dans R,
      restreint a la partie connue de E. Arithmetique entiere pure, aucun
      solveur, et c'est le seul controle de surete des coupes qui survive a
      la disparition de l'enumeration.
  W5  Encadrements concordants. Plusieurs executions independantes encadrent
      le MEME q*, donc  max_r q_lb^r <= min_r q_ub^r. Une violation PROUVE un
      bug sans qu'on ait besoin de q*.
  W6  Concordance avec la relaxation. max_S f est une borne superieure valide
      et EXACTE de q*, obtenue par un seul Dinkelbach sur S (Th. 3, sans
      enumeration). Donc max_r q_lb^r <= max_S f. C'est un temoin totalement
      independant de la machinerie des coupes.

Ce que le protocole NE dit pas : il etablit la VALIDITE des bornes, pas leur
finesse, et ne peut pas detecter une borne superieure correcte mais inutile.
Pour cela, `bench_scale.py` compare a max_S f et a la methode exacte.

La grille croise `n` et `corr` : le generateur construit `A` et `b`
independamment de `corr`, donc le domaine `S` est identique d'un niveau a
l'autre. Verifier la validite sur le seul regime `corr = 0` laisserait sans
controle le regime ou `E` est mince, qui est structurellement different.

Usage :  python verify_scale.py
"""

from __future__ import annotations

import time

import numpy as np

from molfp_core import (ORACLE_CALLS, efficiency_test, max_f_over_S,
                        reset_oracle_counter)
from molfp_instance import generate
from molfp_matheuristic import e_row, matheuristic_P

SIZES = [10, 20, 30, 40]
CORRS = [0.00, 0.90]        # E epais / E mince, a domaine S identique
P = 3
SEEDS = [0, 1, 2]           # executions independantes pour W5
SEARCH_BUDGET = 8.0
BOUND_BUDGET = 6.0
N_ARCHIVE_CHECK = 8         # echantillon pour W2


def w4_cuts_preserve(inst, archive, cut_points) -> dict:
    """
    Invariant E inclus dans R sur la partie connue de E.

    Pour chaque coupe de base xbar, tout point efficace y doit verifier
    e_k(y) >= 1 pour au moins un k -- sinon la coupe l'aurait retire.
    Entierement entier, donc exact.
    """
    if not cut_points or not archive:
        return {"ok": True, "n_checks": 0, "n_bad": 0}
    n_bad = n_checks = 0
    for xbar in cut_points:
        rows = [e_row(inst, xbar, k) for k in range(inst.p)]
        for y in archive:
            n_checks += 1
            if not any(int(round(c @ y + cst)) >= 1 for c, cst in rows):
                n_bad += 1
    return {"ok": n_bad == 0, "n_checks": n_checks, "n_bad": n_bad}


def main() -> int:
    print("=" * 112)
    print("VALIDATION A L'ECHELLE - AUCUNE ENUMERATION, AUCUNE VERITE TERRAIN")
    print(f"budget par execution : {SEARCH_BUDGET:.0f} s + {BOUND_BUDGET:.0f} s"
          f"   |   {len(SEEDS)} executions independantes par instance")
    print("=" * 112)
    print(f"{'instance':<24}{'n':>4}{'corr':>6}"
          f"{'W1':>4}{'W2':>4}{'W3':>4}{'W4':>4}{'W5':>4}{'W6':>4}"
          f"{'max_r q_lb':>12}{'min_r q_ub':>12}{'max_S f':>11}"
          f"{'ecart%':>9}{'ILP':>7}{'t(s)':>7}")
    print("-" * 112)

    all_ok = True
    for n in SIZES:
      m = max(3, n // 2 + 1)
      for corr in CORRS:
        inst = generate(n=n, m=m, p=P, seed=1, rhs_scale=1.0, corr=corr)

        t0 = time.time()
        reset_oracle_counter()

        # W6 : temoin independant, exact, sans enumeration
        rS = max_f_over_S(inst)
        max_S = float(rS.q_star) if rS.q_star is not None else float("inf")

        lbs, ubs = [], []
        w1 = w2 = w3 = w4 = True
        for s in SEEDS:
            r = matheuristic_P(inst, time_budget=SEARCH_BUDGET,
                               bound_budget=BOUND_BUDGET, seed=s)
            lbs.append(float(r.q_lb))

            # W1
            w1 &= bool(efficiency_test(inst, r.x_best).efficient)
            # W2
            sample = r.archive[:N_ARCHIVE_CHECK]
            w2 &= all(efficiency_test(inst, y).efficient is True for y in sample)
            # W3
            if r.q_ub is not None:
                ubs.append(r.q_ub)
                w3 &= float(r.q_lb) <= r.q_ub + 1e-9
            # W4
            w4 &= w4_cuts_preserve(inst, r.archive, r.cut_points)["ok"]

        lo = max(lbs)
        hi = min(ubs) if ubs else float("inf")
        w5 = lo <= hi + 1e-9                       # encadrements concordants
        w6 = lo <= max_S + 1e-9                    # temoin independant
        best_ub = min(hi, max_S)
        gap = (best_ub - lo) / max(1e-12, abs(best_ub)) * 100

        ok = w1 and w2 and w3 and w4 and w5 and w6
        all_ok &= ok

        def mk(b):
            return "ok" if b else "KO"

        hi_s = f"{hi:.4f}" if np.isfinite(hi) else "aucune"
        print(f"{inst.name:<24}{n:>4}{corr:>6.2f}"
              f"{mk(w1):>4}{mk(w2):>4}{mk(w3):>4}{mk(w4):>4}{mk(w5):>4}{mk(w6):>4}"
              f"{lo:>12.4f}{hi_s:>12}{max_S:>11.4f}"
              f"{gap:>8.1f}%{ORACLE_CALLS['ilp']:>7}{time.time()-t0:>7.1f}",
              flush=True)

    print("-" * 112)
    print("W1-W6 : " + ("TOUT VALIDE" if all_ok else "ECHEC"))
    print("Lecture : W1-W6 etablissent la VALIDITE des bornes, pas leur")
    print("finesse. La colonne 'ecart%' se lit contre min(q_ub, max_S f),")
    print("c'est-a-dire contre la meilleure borne superieure disponible.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
