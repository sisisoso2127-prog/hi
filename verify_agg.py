#!/usr/bin/env python3
"""
verify_agg.py
=============
Deux questions sur la coupe AGREGEE (forme sans binaire de la disjonction),
posees la ou l'enumeration exhaustive permet d'y repondre sans discussion.

  V-AGG-1  VALIDITE. Chaque ligne posee doit etre satisfaite par TOUT point
           efficace. Une seule violation invaliderait la borne.
  V-AGG-2  FORCE. Que retranche-t-elle vraiment, compare a la disjonction
           exacte posee sur les MEMES points ? C'est la question qui decide
           si elle merite d'exister : elle ne coute aucune binaire, donc
           aucun plafond ne s'y applique, mais elle n'a d'interet que si
           elle mord.

Usage :  python verify_agg.py
"""

import sys

import numpy as np

from molfp_enum import ground_truth
from molfp_instance import FracObj, MOILFP
from molfp_oracle import ECutModel


def _build(A, b, Zs, fo, name):
    return MOILFP(
        A=np.array(A), b=np.array(b),
        Z=[FracObj(np.array(c), a, np.array(d), bb) for c, a, d, bb in Zs],
        f=FracObj(np.array(fo[0]), fo[1], np.array(fo[2]), fo[3]), name=name)


CAS = [
    ("I2", [[3, 1], [2, 3]], [14, 15],
     [([0, 3], 2, [1, 1], 3), ([1, 3], 0, [0, 2], 3)], ([3, 1], 0, [0, 2], 1)),
    ("I3", [[3, 1, 1], [1, 2, 2]], [8, 9],
     [([0, 0, 3], 0, [0, 2, 1], 2), ([3, 3, 1], 1, [1, 2, 1], 3)],
     ([2, 2, 1], 1, [1, 0, 1], 1)),
    ("I4", [[3, 1], [1, 3]], [12, 12],
     [([1, 1], 2, [1, 2], 1), ([1, 2], 1, [2, 1], 2)], ([4, 2], 2, [0, 2], 1)),
    ("I5", [[2, 3], [3, 3]], [11, 13],
     [([2, 3], 1, [2, 2], 3), ([4, 1], 2, [0, 1], 3)], ([3, 0], 2, [1, 2], 3)),
]


def main() -> None:
    print("=" * 92)
    print("VERIFICATION DE LA COUPE AGREGEE (sans binaire)")
    print("=" * 92)
    print(f"{'instance':<8}{'|S|':>6}{'|E|':>5}{'domines':>9}"
          f"{'lignes':>8}{'rejetees':>10}"
          f"{'|R| agrege':>12}{'|R| disjonctif':>16}{'viol. sur E':>13}")
    print("-" * 92)

    viol_tot, lignes_tot = 0, 0
    for nom, A, b, Zs, fo in CAS:
        inst = _build(A, b, Zs, fo, nom)
        gt = ground_truth(inst, limit=50_000)
        Ek = {tuple(int(v) for v in x) for x in gt.E}
        dom = [np.asarray(x, dtype=int) for x in gt.S
               if tuple(int(v) for v in x) not in Ek]

        model = ECutModel(inst)
        posees = []
        for xb in dom:
            if model.add_aggregated_cut(xb):
                posees.append((xb, model._cut_rows[-1]))
        lignes_tot += len(posees)

        # V-AGG-1 : validite
        viol = 0
        for _, (row, lo, _hi) in posees:
            for x in gt.E:
                if float(row[:inst.n] @ x) < lo - 1e-9:
                    viol += 1
        viol_tot += viol

        # V-AGG-2 : force, contre la disjonction sur les memes points
        def survit_agrege(x):
            return all(float(row[:inst.n] @ x) >= lo - 1e-9
                       for _, (row, lo, _h) in posees)

        def survit_disjonctif(x):
            zx = inst.criteria(x)
            for xb, _ in posees:
                za = inst.criteria(xb)
                if not any(zx[k] > za[k] for k in range(inst.p)):
                    return False
            return True

        r_agg = sum(1 for x in gt.S if survit_agrege(x))
        r_dis = sum(1 for x in gt.S if survit_disjonctif(x))
        print(f"{nom:<8}{len(gt.S):>6}{len(gt.E):>5}{len(dom):>9}"
              f"{len(posees):>8}{len(dom) - len(posees):>10}"
              f"{r_agg:>12}{r_dis:>16}{viol:>13}")

    print("-" * 92)
    print(f"  V-AGG-1 validite : {lignes_tot} lignes, {viol_tot} violation(s) "
          f"-> {'OK' if viol_tot == 0 else 'ECHEC'}")
    print("  V-AGG-2 force    : la forme agregee retranche nettement MOINS "
          "que la disjonction")
    print("                     sur les memes points -- c'est un relachement, "
          "pas un equivalent.")
    print("                     Elle ne se justifie donc qu'AU-DELA du "
          "plafond, ou l'alternative")
    print("                     n'est pas la disjonction mais RIEN.")
    return 0 if viol_tot == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
