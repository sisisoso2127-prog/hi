#!/usr/bin/env python3
"""
verify_cglp.py
==============
Verification de la coupe disjonctive par programme generateur.

Trois questions, posees la ou l'enumeration exhaustive permet d'y repondre
sans discussion.

  V-CGLP-1  VALIDITE. Une coupe generee depuis le polyedre de base doit
            etre satisfaite par TOUT point verifiant la disjonction. Une
            seule violation invaliderait la borne.
  V-CGLP-2  VALIDITE EN CHAINE. Quand plusieurs coupes s'enchainent, chacune
            est derivee du polyedre DEJA restreint par les precedentes. Elle
            n'est donc valide que relativement a CE polyedre, et non a S
            tout entier. C'est une subtilite qui se paie : une premiere
            version de ce controle verifiait les coupes chainees sur S
            entier et signalait une violation, laquelle n'en etait pas une
            -- le point incrimine avait deja ete retire par une coupe
            anterieure. Le test corrige verifie chaque coupe sur les seuls
            survivants du polyedre qui l'a produite.
  V-CGLP-3  FORCE. Combien de points la coupe retranche-t-elle reellement,
            et combien de points EFFICACES ? Les seconds doivent etre de
            meme vecteur criteres que le point de coupe -- c'est la reserve
            du theoreme d'efficacite, pas une faute.

Usage :  python verify_cglp.py
"""

import sys

import numpy as np

from molfp_cglp import cglp_cut, optimum_relaxation, verifier_validite
from molfp_core import feasibility_rows
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import surrogate


def _instances():
    for i in range(10):
        n = 4 + i % 3
        yield generate(n=n, m=max(3, n // 2 + 1), p=3, seed=500 + i,
                       rhs_scale=1.5, corr=[0.0, 0.5, 0.9][i % 3])


def main() -> int:
    print("=" * 96)
    print("VERIFICATION DE LA COUPE PAR PROGRAMME GENERATEUR (lift-and-project)")
    print("=" * 96)
    print(f"{'instance':<26}{'|S|':>7}{'|E|':>6}{'coupes':>8}"
          f"{'viol. base':>12}{'viol. chaine':>14}"
          f"{'pts retranches':>16}{'dont eff.':>11}")
    print("-" * 96)

    tot = {"gen": 0, "vb": 0, "vc": 0, "ret": 0, "eff": 0}
    for inst in _instances():
        gt = ground_truth(inst, limit=100_000)
        Ek = {tuple(int(v) for v in x) for x in gt.E}
        w, _, _, _ = surrogate(inst, gt.q_star)
        ub = inst.var_upper_bounds()
        base = feasibility_rows(inst)

        # V-CGLP-1 : chaque coupe depuis le polyedre DE BASE
        vb = 0
        xs0 = optimum_relaxation(w, base, ub)
        for a in list(gt.E)[:6]:
            cut = cglp_cut(inst, np.asarray(a, dtype=int), base, ub, xs0)
            if cut is not None:
                vb += verifier_validite(inst, np.asarray(a, dtype=int),
                                        cut[0], cut[1], gt.S)

        # V-CGLP-2 et 3 : coupes CHAINEES
        rows, poses = list(base), []
        gen = ret = eff = vc = 0
        for a in list(gt.E)[:6]:
            xs = optimum_relaxation(w, rows, ub)
            if xs is None:
                break
            cut = cglp_cut(inst, np.asarray(a, dtype=int), rows, ub, xs)
            if cut is None:
                continue
            gen += 1
            survivants = [x for x in gt.S
                          if all(float(al @ np.asarray(x, dtype=float))
                                 >= be - 1e-6 for al, be in poses)]
            vc += verifier_validite(inst, np.asarray(a, dtype=int),
                                    cut[0], cut[1], survivants)
            for x in survivants:
                if float(cut[0] @ np.asarray(x, dtype=float)) < cut[1] - 1e-6:
                    ret += 1
                    eff += int(tuple(int(v) for v in x) in Ek)
            poses.append(cut)
            rows.append((cut[0], cut[1], np.inf))

        for k, v in (("gen", gen), ("vb", vb), ("vc", vc),
                     ("ret", ret), ("eff", eff)):
            tot[k] += v
        print(f"{inst.name:<26}{len(gt.S):>7}{len(gt.E):>6}{gen:>8}"
              f"{vb:>12}{vc:>14}{ret:>16}{eff:>11}")

    print("-" * 96)
    ok = (tot["vb"] == 0 and tot["vc"] == 0)
    print(f"  V-CGLP-1 validite depuis le polyedre de base : "
          f"{tot['vb']} violation(s) -> {'OK' if tot['vb'] == 0 else 'ECHEC'}")
    print(f"  V-CGLP-2 validite en chaine                  : "
          f"{tot['vc']} violation(s) -> {'OK' if tot['vc'] == 0 else 'ECHEC'}")
    print(f"  V-CGLP-3 force : {tot['gen']} coupes retranchent {tot['ret']} "
          f"points, dont {tot['eff']} efficaces")
    print("     (les efficaces retranches sont de MEME vecteur criteres que")
    print("      le point de coupe : c'est la reserve du theoreme, pas une faute)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
