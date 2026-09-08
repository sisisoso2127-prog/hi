#!/usr/bin/env python3
"""
bench_recherche.py
==================
A n = 40, CE QUI RESTE EST-IL UNE BORNE TROP HAUTE OU UN INCUMBENT TROP BAS ?

D'ou vient la question. Avec la seconde route, la borne annoncee EST
max_R f -- le statut 'optimal' le certifie : il n'existe plus, dans la
region survivante, aucun point de valeur superieure. A n = 20 et n = 30 cela
fait tomber l'ecart garanti de 98,6 a 88,8 %, de 93,9 a 78,6 %, de 78,9 a
8,0 %. A n = 40, cela ne le fait pas bouger : 96,6 -> 96,5 %.

Or l'ecart s'ecrit (q_ub - q_lb)/|q_ub|. Si q_ub ne peut plus descendre --
et le statut 'optimal' dit qu'il ne le peut pas sans coupes
supplementaires -- alors ce qui reste tient au NUMERATEUR par son autre
terme : q_lb. Le banc de couverture a d'ailleurs deja etabli, la ou la
verite terrain existe, que l'incumbent MANQUE q* sur 3 lignes de 26, toutes
a n >= 12.

CE QUE MESURE CE BANC. La phase de recherche seule (certification coupee),
sous des plafonds croissants d'appels entiers. Si q_lb monte franchement
quand on lui donne davantage, le verrou a n = 40 est la RECHERCHE, et non
la borne -- ce qui deplacerait l'effort exactement a l'oppose de la ou nous
l'avons porte jusqu'ici. S'il ne bouge pas, la recherche est a son plafond
et le probleme est ailleurs.

Aucune verite terrain n'est disponible a ces tailles : on ne mesure donc
pas un ecart a q*, mais la PROGRESSION de q_lb, qui suffit a trancher la
question posee. Tout q_lb est valide de toute facon -- l'incumbent est
certifie efficace a chaque instant.

Usage :  python bench_recherche.py [graines] [plafonds separes par virgules]
"""

import os
import sys
import time

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P

TAILLES = [int(v) for v in os.environ.get("MOLFP_TAILLES", "20,30,40").split(",")]
CORRS = [0.0, 0.5]


def une(inst, cap, seed):
    reset_oracle_counter()
    t0 = time.time()
    # certification coupee : tout le plafond va a la recherche.
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=0.0, seed=seed,
                       archive_cuts=True, ilp_budget=cap, cut_diversify=False,
                       certify_bound=False, reallocate=False)
    return {"lb": float(r.q_lb), "ilp": r.ilp_calls, "arch": len(r.archive),
            "t": time.time() - t0}


def main() -> int:
    n_gr = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    caps = [int(v) for v in (sys.argv[2] if len(sys.argv) > 2
                             else "300,900").split(",")]
    graines = list(range(1, n_gr + 1))

    entete = (f"{'n':>4}{'corr':>6}{'gr':>4}"
              + "".join(f"{'q_lb @' + str(c):>14}" for c in caps)
              + f"{'progression':>14}{'archive':>9}")
    print("=" * len(entete))
    print(f"RECHERCHE SEULE - plafonds {caps} appels entiers, {n_gr} graines")
    print("=" * len(entete))
    print(entete)
    print("-" * len(entete))

    progres = []
    for n in TAILLES:
        m = max(3, n // 2 + 1)
        for corr in CORRS:
            for g in graines:
                inst = generate(n=n, m=m, p=3, seed=g, rhs_scale=1.0,
                                corr=corr)
                res = [une(inst, c, g) for c in caps]
                lb0, lb1 = res[0]["lb"], res[-1]["lb"]
                gain = (lb1 - lb0) / max(1e-12, abs(lb0)) * 100
                progres.append(gain)
                print(f"{n:>4}{corr:>6.2f}{g:>4}"
                      + "".join(f"{r['lb']:>14.4f}" for r in res)
                      + f"{gain:>+13.1f}%{res[-1]['arch']:>9}", flush=True)

    print("-" * len(entete))
    p = np.asarray(progres)
    n_bouge = int((p > 1e-9).sum())
    print(f"  lignes ou q_lb PROGRESSE en triplant le plafond : "
          f"{n_bouge}/{len(p)}")
    print(f"  progression : mediane {np.median(p):+.1f} %   "
          f"moyenne {p.mean():+.1f} %   max {p.max():+.1f} %")
    print()
    print("  Lecture. Une progression franche dit que la recherche n'etait")
    print("  pas a son plafond, donc qu'une part de l'ecart garanti lui est")
    print("  imputable et non a la borne. Une progression nulle dit le")
    print("  contraire, et renvoie la question a la construction du relache.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
