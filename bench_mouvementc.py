#!/usr/bin/env python3
"""
bench_mouvementc.py
===================
LE MOUVEMENT C : LE REPARER, OU LE RETIRER ?

CE QUE LA MESURE LUI REPROCHE. Sur douze instances, 4 succes pour 316
tentatives -- 1,3 %, contre 4,0 % pour le mouvement A et 3,0 % pour B --
soit pres d'un cinquieme du budget de recherche pour le plus faible
rendement des trois. Ce chiffre ne dit pas s'il faut le corriger ou s'en
passer ; ce banc tranche.

LE DIAGNOSTIC. C maximise LIBREMENT le substitut de f sur une sous-boite.
Or les points de f eleve sont DOMINES -- l'article l'etablit ailleurs -- et
la chaine de reparation les ramene vers des points efficaces de f faible.
Le mouvement gagne donc ce que la reparation lui reprend. A et B n'ont pas
ce defaut parce qu'ils CONTRAIGNENT les criteres et atterrissent pres de la
frontiere.

TROIS BRAS, sur les memes instances, a plafond identique :

  libre   C tel quel ;
  ancre   C avec des planchers e_j(x) >= 0 sur un sous-ensemble STRICT et
          aleatoire de criteres -- exactement ce qui fait marcher A ;
  sans    C retire du menu, son budget revenant a A et B.

CE QUE LE BANC PEUT CONCLURE. Si `sans` domine les deux autres, le
mouvement ne merite pas sa place et il faut le dire ; si `ancre` domine,
le defaut etait sa conception et non son principe. Les deux issues sont
publiables, et le banc ne prejuge pas.

Usage :  python bench_mouvementc.py [plafond] [graines]
"""

import sys
import time

import numpy as np

from molfp_core import reset_oracle_counter, upper_bound_over_S
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P
from molfp_oracle import CHAINES, reset_chaines

TAILLES = [10, 20, 30]
CORRS = [0.0, 0.5]
BRAS = ["libre", "ancre", "sans"]


def une(inst, cap, seed, mode, mS):
    reset_oracle_counter()
    reset_chaines()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=seed,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       geom_bound=True, mouvement_c=mode)
    ub = min(r.q_ub if r.q_ub is not None else np.inf, mS)
    lb = float(r.q_lb)
    mv, hi = r.cert.get("moves", {}), r.cert.get("hits", {})
    return {"lb": lb, "ecart": (ub - lb) / max(1e-12, abs(ub)) * 100,
            "mv": mv, "hits": hi, "arch": len(r.archive),
            "ilp": r.ilp_calls, "t": time.time() - t0}


def main() -> int:
    cap = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    n_gr = int(sys.argv[2]) if len(sys.argv) > 2 else 2

    entete = (f"{'n':>4}{'corr':>6}{'gr':>4}"
              + "".join(f"{'ecart ' + b:>13}" for b in BRAS)
              + f"{'q_lb libre/ancre/sans':>26}")
    print("=" * len(entete))
    print(f"MOUVEMENT C : REPARER OU RETIRER - plafond {cap}, {n_gr} graines")
    print("=" * len(entete))
    print(entete)
    print("-" * len(entete))

    ecarts = {b: [] for b in BRAS}
    rend = {b: [0, 0] for b in BRAS}
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in range(1, n_gr + 1):
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                mS = upper_bound_over_S(inst)
                res = {b: une(inst, cap, g, b, mS) for b in BRAS}
                for b in BRAS:
                    ecarts[b].append(res[b]["ecart"])
                    rend[b][0] += res[b]["hits"].get("C", 0)
                    rend[b][1] += res[b]["mv"].get("C", 0)
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      + "".join(f"{res[b]['ecart']:>12.1f}%" for b in BRAS)
                      + "  " + " / ".join(f"{res[b]['lb']:.3f}" for b in BRAS),
                      flush=True)

    print("-" * len(entete))
    ref = np.asarray(ecarts["libre"])
    for b in BRAS:
        v = np.asarray(ecarts[b])
        d = ref - v                       # > 0 : le bras est meilleur
        h, t = rend[b]
        print(f"  {b:<6} ecart median {np.median(v):6.1f}%   "
              f"moyen {v.mean():6.1f}%   "
              f"contre `libre` : mieux {int((d>1e-9).sum())}  "
              f"pire {int((d<-1e-9).sum())}  egal {int((abs(d)<=1e-9).sum())}"
              + (f"   rendement C {h}/{t}" if t else "   C retire"))
    print()
    print("  Lecture. `sans` meilleur => le mouvement ne merite pas sa place.")
    print("  `ancre` meilleur => le defaut etait sa conception, non son")
    print("  principe. Aucun des deux => il est neutre, et son cout est du")
    print("  gaspillage qu'il faut nommer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
