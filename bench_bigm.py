"""
bench_bigm.py
=============
Quantifie le renforcement du big-M des coupes de dominance (Th. 4) :
minorant sur la BOITE `0 <= x <= ub` (version d'origine) contre minorant sur
la RELAXATION CONTINUE de S (version actuelle), et contre le minimum REEL de
e_k sur S enumere.

Les trois quantites se lisent ensemble :

    M_boite  >=  M_lp  >=  M_reel  >=  1

`M_boite / M_lp` mesure le gain ; `M_lp >= M_reel` est la condition de
VALIDITE — un big-M sous le minimum reel couperait des points efficaces.
`M_lp / M_reel` dit ce qu'il reste a gagner : c'est l'ecart d'integralite,
qu'aucun calcul en temps polynomial ne fermera.

Usage :  python bench_bigm.py
"""

from __future__ import annotations

import numpy as np

from molfp_enum import as_key, enumerate_feasible
from molfp_instance import generate
from molfp_oracle import ECutModel

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

N_CUTS = 8          # points de base tires par instance


def main() -> int:
    rng = np.random.default_rng(0)
    print("=" * 88)
    print("RENFORCEMENT DU BIG-M DES COUPES DE DOMINANCE")
    print("=" * 88)
    print(f"{'instance':<24}{'|S|':>8}{'M boite':>12}{'M lp':>12}"
          f"{'M reel':>12}{'boite/lp':>10}{'lp/reel':>10}{'valide':>8}")
    print("-" * 88)

    gains, slacks, all_ok = [], [], True
    for cfg in CFG:
        inst = generate(**cfg)
        S = enumerate_feasible(inst, limit=400_000)
        Sarr = np.array(S)

        idx = rng.choice(len(S), size=min(N_CUTS, len(S)), replace=False)
        model = ECutModel(inst, tight_big_m=True)
        box, lp, real, ok = [], [], [], True
        for i in idx:
            xbar = S[i]
            start = len(model.big_m_used)
            model.add_dominance_cut(xbar)
            for k in range(inst.p):
                Zk = inst.Z[k]
                Nb, Db = Zk.numerator(xbar), Zk.denominator(xbar)
                coef = Db * Zk.num - Nb * Zk.den
                const = Db * Zk.a - Nb * Zk.b
                e_min = int((Sarr @ coef).min()) + int(const)
                m_real = max(1.0, 1.0 - e_min)
                m_box = model.big_m_box[start + k]
                m_lp = model.big_m_used[start + k]
                box.append(m_box); lp.append(m_lp); real.append(m_real)
                if m_lp < m_real - 1e-9:      # big-M invalide : coupe fausse
                    ok = False
        all_ok &= ok

        box, lp, real = map(np.array, (box, lp, real))
        gain = float(np.median(box / np.maximum(1.0, lp)))
        slack = float(np.median(lp / np.maximum(1.0, real)))
        gains.append(gain); slacks.append(slack)
        print(f"{inst.name:<24}{len(S):>8}"
              f"{np.median(box):>12.0f}{np.median(lp):>12.0f}"
              f"{np.median(real):>12.0f}{gain:>9.2f}x{slack:>9.2f}x"
              f"{'ok' if ok else 'KO':>8}")

    print("-" * 88)
    print(f"resserrement median boite -> relaxation : {np.median(gains):.2f}x")
    print(f"ecart d'integralite residuel median     : {np.median(slacks):.2f}x")
    print(f"VALIDITE (M_lp >= M_reel) : "
          f"{'TOUT VALIDE' if all_ok else 'ECHEC'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
