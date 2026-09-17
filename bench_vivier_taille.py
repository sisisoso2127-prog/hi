#!/usr/bin/env python3
"""
bench_vivier_taille.py
======================
POURQUOI beta = 160 REND EXACTEMENT CE QUE REND beta = 80.

`bench_confirm_beta.py` a mesure quelque chose que la mesure seule
n'explique pas : sur vingt-quatre executions, beta = 160 rend les SIX
memes colonnes que beta = 80 -- mediane, moyenne, delta, compte apparie,
test du signe, preuves. Une coincidence a six chiffres n'est pas une
coincidence. Nous en avions conclu « il y a un genou, et il est a 80 ».
C'est une lecture possible. En voici une autre, qui ne demande aucun
rendement decroissant :

  beta ne borne pas le nombre de coupes UTILES, il borne le nombre de
  coupes posees PAR TOUR, et le tour les prend dans un VIVIER FINI qui
  est consomme (`pending = pending[posees:]`). Si le vivier compte moins
  de beta * rho candidats, la configuration le VIDE, et deux
  configurations qui le vident posent le meme jeu de coupes.

Un sondage sur quatre instances de la strate B donne des viviers de 37 a
121 candidats. Si cet ordre de grandeur tient, beta = 160 ne borne jamais
rien -- ce ne serait pas un genou, ce serait une valeur qui ne mord pas.
Et ce n'est pas la meme phrase.

CE QUE CE BANC MESURE, par execution : la taille des viviers construits,
le nombre de coupes effectivement posees, le nombre de tours de
certification reellement joues, et si le vivier a ete VIDE. Puis il
confronte trois predictions a ce qu'il a vu.

CE QU'IL NE PEUT PAS FAIRE. Distinguer a lui seul les deux mecanismes en
jeu -- « avoir pose toutes les coupes » et « avoir paye moins de
resolutions entieres ». beta = 40 avec rho = 4 vide lui aussi le vivier
et ne rend pourtant pas ce que rend beta = 80 : le jeu de coupes ne
suffit donc pas a tout expliquer, et ce banc le montre au lieu de le
cacher.

Usage :  python bench_vivier_taille.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys
from typing import Dict, List

import molfp_hybride as mh
from molfp_core import reset_oracle_counter
from molfp_instance import generate
from bench_sensibilite import PROD

LOT = [dict(n=n, m=max(3, n // 2 + 1), p=3, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for c in (0.00, 0.50) for s in (1, 2)]

CONFIGS = [
    ("b=40  r=2   (production)", {}),
    ("b=80  r=2", dict(cut_batch=80)),
    ("b=160 r=2", dict(cut_batch=160)),
    ("b=40  r=4", dict(cert_rounds=4)),
]

# On espionne la construction du vivier sans toucher a la methode : le banc
# ne doit pas mesurer un code different de celui qui tourne en production.
_TAILLES: List[int] = []
_ORIG = mh.build_cut_pool


def _espion(*a, **k):
    r = _ORIG(*a, **k)
    _TAILLES.append(len(r))
    return r


mh.build_cut_pool = _espion


def une(inst, cap: int, graine: int, **surcharge) -> Dict:
    cfg = dict(PROD)
    cfg.update(surcharge)
    _TAILLES.clear()
    reset_oracle_counter()
    r = mh.matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                          seed=graine, ilp_budget=cap, **cfg)
    vivier = sum(_TAILLES)
    posees = r.cert.get("n_cuts", 0) if r.cert else 0
    return dict(viviers=list(_TAILLES), vivier=vivier, posees=posees,
                tours=r.cert.get("rounds", 0) if r.cert else 0,
                vide=posees >= vivier,
                ecart=(r.gap * 100) if r.gap is not None else float("nan"),
                prouve=bool(r.proved_optimal))


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("=" * 100)
    print("TAILLE DU VIVIER, COUPES POSEES, TOURS JOUES")
    print(f"  lot elargi : {len(LOT)} instances (n = 20 et 30) x "
          f"{len(graines)} graines, plafond {cap}")
    print("=" * 100)

    res: Dict[str, List[Dict]] = {}
    for nom, sur in CONFIGS:
        lectures = []
        for spec in LOT:
            inst = generate(**spec)
            for g in graines:
                lectures.append(une(inst, cap, g, **sur))
        res[nom] = lectures
        v = [d["vivier"] for d in lectures]
        p = [d["posees"] for d in lectures]
        print(f"\n{nom}")
        print(f"    vivier   min {min(v):>4}  med {statistics.median(v):>6.1f}"
              f"  max {max(v):>4}")
        print(f"    posees   min {min(p):>4}  med {statistics.median(p):>6.1f}"
              f"  max {max(p):>4}")
        print(f"    tours    med {statistics.median([d['tours'] for d in lectures]):>4.1f}"
              f"      vivier VIDE : {sum(d['vide'] for d in lectures)}"
              f"/{len(lectures)}")
        print(f"    ecart med {statistics.median([d['ecart'] for d in lectures if d['ecart'] == d['ecart']]):>7.2f}"
              f"   preuves {sum(d['prouve'] for d in lectures)}/{len(lectures)}",
              flush=True)

    print("\n" + "=" * 100)
    print("TROIS PREDICTIONS, CONFRONTEES")
    print("=" * 100)

    prod, b80, b160, r4 = (res[n] for n, _ in CONFIGS)

    jamais = all(max(d["viviers"] or [0]) < 160 for d in prod)
    print(f"\n 1. « beta = 160 ne mord jamais » -- aucun vivier n'atteint "
          f"160 candidats : {'OUI' if jamais else 'NON'}")
    if not jamais:
        gros = max(max(d['viviers'] or [0]) for d in prod)
        print(f"    (le plus gros vivier vu compte {gros} candidats)")

    memes = sum(1 for a, b in zip(b80, b160)
                if a["posees"] == b["posees"] and a["tours"] == b["tours"])
    print(f"\n 2. « beta = 80 et beta = 160 posent le meme jeu de coupes en "
          f"autant de tours » : {memes}/{len(b80)} executions")

    manque = sum(1 for d in prod if not d["vide"])
    print(f"\n 3. « la production ne VIDE pas le vivier » -- elle laisse des "
          f"candidats non poses sur {manque}/{len(prod)} executions")

    memes_r4 = sum(1 for a, b in zip(b80, r4) if a["posees"] == b["posees"])
    tours_r4 = sum(1 for a, b in zip(b80, r4) if a["tours"] == b["tours"])
    print(f"\n 4. CONTRE-EPREUVE. beta = 40, rho = 4 pose autant de coupes "
          f"que beta = 80 sur {memes_r4}/{len(r4)} executions,")
    print(f"    mais en autant de tours sur {tours_r4}/{len(r4)} seulement. "
          f"Si le premier compte est eleve et le second bas,")
    print("    alors le jeu de coupes n'explique pas tout, et c'est le "
          "NOMBRE DE RESOLUTIONS qui porte le reste.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
