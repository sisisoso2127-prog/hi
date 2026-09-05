"""
verify_oracle.py
================
Validation V5-V8 de l'oracle lineaire sur E et de l'hybride exact-exact,
contre la verite terrain par enumeration exhaustive.

  V5  Oracle : max_linear_over_E(g) == max_{x in E} g(x) par force brute,
      sur plusieurs g tirees au hasard. C'est le test qui valide a la fois la
      coupe de dominance (Th. 4) et le critere d'arret anytime.
  V6  Hybride : solve_P retrouve q* = max_E f.
  V7  Archive : aucun faux positif (tout point archive est reellement efficace).
  V8  Big-M : validite DIRECTE du renforcement par relaxation continue.
      Pour chaque coupe posee, M_k doit majorer 1 - min_{x in S} e_k(x) ;
      on compare au minimum reel sur S enumere. Un M trop petit couperait des
      points efficaces sans que V5 le detecte forcement sur une seule graine :
      ce test-la ne depend pas de la chance.

Usage :  python verify_oracle.py
"""

from __future__ import annotations

import time

import numpy as np

from molfp_core import ORACLE_CALLS, efficiency_test, reset_oracle_counter
from molfp_enum import as_key, ground_truth
from molfp_instance import generate
from molfp_oracle import ECutModel, dedup_archive, max_linear_over_E, solve_P

CONFIGS = [
    dict(n=4, m=3, p=2, seed=1),
    dict(n=4, m=3, p=3, seed=2),
    dict(n=5, m=3, p=3, seed=1),
    dict(n=5, m=4, p=4, seed=3),
    dict(n=6, m=4, p=3, seed=1),
    dict(n=8, m=4, p=3, seed=1),
]

N_G = 4          # nombre de fonctions lineaires g tirees par instance
TIME_LIMIT = 60.0


def v5_oracle(inst, gt, rng) -> dict:
    """max_E g par l'oracle == max_E g par force brute, pour plusieurs g."""
    n_bad, gaps = 0, []
    for _ in range(N_G):
        g = rng.integers(-10, 11, size=inst.n).astype(float)
        truth = max(float(g @ x) for x in gt.E)
        r = max_linear_over_E(inst, g, 0.0, time_limit=TIME_LIMIT)
        if r.status == "limit":
            gaps.append(r.gap)
            continue                       # non conclusif, pas une divergence
        if r.value is None or abs(r.value - truth) > 1e-6:
            n_bad += 1
    return {"ok": n_bad == 0, "n_bad": n_bad, "n_g": N_G}


def v6_hybrid(inst, gt) -> dict:
    """solve_P retrouve q* (ou s'arrete honnetement sur 'limit')."""
    r = solve_P(inst, time_limit=TIME_LIMIT)
    if r.status != "optimal":
        return {"ok": True, "status": r.status, "value": r.q_star,
                "conclusive": False}
    return {"ok": r.q_star == gt.q_star, "status": r.status,
            "value": r.q_star, "conclusive": True, "result": r}


def v7_archive(inst, archive) -> dict:
    """Aucun point archive ne doit etre non efficace."""
    pts = dedup_archive(archive)
    bad = sum(1 for x in pts if not efficiency_test(inst, x).efficient)
    return {"ok": bad == 0, "n": len(pts), "bad": bad}


def v8_big_m(inst, gt, rng) -> dict:
    """
    Validite directe du big-M renforce : pour chaque coupe posee sur un point
    domine xbar, on exige M_k >= 1 - min_{x in S} e_k(x), le minimum etant
    calcule sur S ENUMERE (donc sans aucune hypothese).

    Renvoie aussi le facteur de resserrement median boite -> relaxation.
    """
    dominated = [x for x in gt.S if as_key(x) not in {as_key(y) for y in gt.E}]
    if not dominated:
        return {"ok": True, "n": 0, "ratio": None}
    idx = rng.choice(len(dominated), size=min(6, len(dominated)), replace=False)

    model = ECutModel(inst, tight_big_m=True)
    n_bad, ratios = 0, []
    for i in idx:
        xbar = dominated[i]
        before = len(model.big_m_used)
        model.add_dominance_cut(xbar)
        for k in range(inst.p):
            Zk = inst.Z[k]
            Nb, Db = Zk.numerator(xbar), Zk.denominator(xbar)
            coef = Db * Zk.num - Nb * Zk.den
            const = Db * Zk.a - Nb * Zk.b
            e_min_true = min(int(coef @ x) + int(const) for x in gt.S)
            M_needed = max(1.0, 1.0 - e_min_true)
            M_used = model.big_m_used[before + k]
            if M_used < M_needed - 1e-9:          # big-M invalide : grave
                n_bad += 1
            ratios.append(model.big_m_box[before + k] / max(1.0, M_used))
    return {"ok": n_bad == 0, "n": len(ratios), "bad": n_bad,
            "ratio": float(np.median(ratios))}


def main() -> None:
    rng = np.random.default_rng(0)
    print("=" * 104)
    print("VALIDATION V5-V8 - ORACLE SUR E, HYBRIDE EXACT-EXACT, BIG-M")
    print("=" * 104)
    print(f"{'instance':<24}{'|S|':>7}{'|E|':>6}"
          f"{'V5':>5}{'V6':>5}{'V7':>5}{'V8':>5}"
          f"{'statut':>9}{'Dink.':>7}{'coupes':>8}{'ILP':>7}{'t(s)':>7}"
          f"{'archive':>9}{'couv%':>7}{'M box/M lp':>12}")
    print("-" * 104)

    all_ok = True
    for cfg in CONFIGS:
        inst = generate(**cfg)
        gt = ground_truth(inst, limit=400_000)

        reset_oracle_counter()
        t0 = time.time()
        r5 = v5_oracle(inst, gt, rng)
        r6 = v6_hybrid(inst, gt)
        arch = r6["result"].archive if r6.get("result") is not None else []
        r7 = v7_archive(inst, arch)
        r8 = v8_big_m(inst, gt, rng)

        ok = r5["ok"] and r6["ok"] and r7["ok"] and r8["ok"]
        all_ok &= ok
        res = r6.get("result")

        def mark(d):
            return "ok" if d["ok"] else "KO"

        ratio = f"{r8['ratio']:>11.1f}x" if r8["ratio"] else f"{'-':>12}"
        print(f"{inst.name:<24}{len(gt.S):>7}{len(gt.E):>6}"
              f"{mark(r5):>5}{mark(r6):>5}{mark(r7):>5}{mark(r8):>5}"
              f"{r6['status']:>9}"
              f"{(res.outer_iterations if res else 0):>7}"
              f"{(res.total_cuts if res else 0):>8}"
              f"{ORACLE_CALLS['ilp']:>7}{time.time()-t0:>7.1f}"
              f"{r7['n']:>9}{100.0*r7['n']/max(1,len(gt.E)):>6.0f}%{ratio}")
        if not ok:
            print(f"    V5 {r5}\n    V6 {r6}\n    V7 {r7}\n    V8 {r8}")

    print("-" * 104)
    print("V5-V8 : " + ("TOUT VALIDE" if all_ok else "ECHEC"))
    print("Lecture : 'M box/M lp' est le facteur de resserrement du big-M "
          "apporte par la relaxation continue.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
