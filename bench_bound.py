"""
bench_bound.py
==============
Th. 5 (temoin) vs Th. 5' (resserre), a budget identique, sur les instances ou
la methode exacte a echoue. Verifie AUSSI la validite : q_ub >= q*.
Une borne plus fine mais fausse serait pire qu'inutile.
"""
import numpy as np
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

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

print(f"{'instance':<24}{'|E|':>5}{'ecart reel':>12}"
      f"{'Th.5':>10}{'Th.5p':>10}{'valide':>8}{'coupes':>8}{'Dmin':>7}{'D+':>7}")
print("-" * 92)
g5, g5p, real = [], [], []
allvalid = True
for c in CFG:
    inst = generate(**c)
    gt = ground_truth(inst, limit=200_000)
    q = float(gt.q_star)

    a = matheuristic_P(inst, time_budget=7, bound_budget=5, seed=0,
                       tightened=False)
    b = matheuristic_P(inst, time_budget=7, bound_budget=5, seed=0,
                       tightened=True)

    r = (q - float(b.q_lb)) / q * 100
    ga = a.gap * 100 if a.gap is not None else float('nan')
    gb = b.gap * 100 if b.gap is not None else float('nan')
    ok = (b.q_ub is None) or (b.q_ub >= q - 1e-9)
    allvalid &= ok
    g5.append(ga); g5p.append(gb); real.append(r)
    ci = b.cert
    print(f"{inst.name:<24}{len(gt.E):>5}{r:>11.1f}%{ga:>9.1f}%{gb:>9.1f}%"
          f"{'ok' if ok else 'KO':>8}{ci.get('n_cuts',0):>8}"
          f"{str(ci.get('Dmin')):>7}{str(ci.get('Dplus')):>7}")

g5, g5p, real = map(np.array, (g5, g5p, real))
print("-" * 92)
print(f"ecart garanti median : Th.5 = {np.nanmedian(g5):.1f} %   "
      f"Th.5' = {np.nanmedian(g5p):.1f} %   (ecart reel median {np.median(real):.1f} %)")
print(f"reduction mediane de l'ecart garanti : "
      f"{(1-np.nanmedian(g5p)/np.nanmedian(g5))*100:.1f} %")
print(f"VALIDITE (q_ub >= q*) : {'TOUT VALIDE' if allvalid else 'ECHEC'}")
