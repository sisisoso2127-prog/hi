#!/usr/bin/env python3
"""
bench_prix_tours.py
===================
A PLAFOND DE COUPES CONSTANT, CE QUE COUTE UN TOUR DE PLUS.

`bench_vivier_taille.py` a laisse une hypothese debout et une seule. Elle
est nee d'un fait genant : $\\beta=40$ avec $\\rho=4$ pose DAVANTAGE de coupes
que $\\beta=80$ -- 97,5 contre 88,5 en mediane -- et fait pourtant moins bien
sur l'ecart comme sur les preuves. Le nombre de coupes n'est donc pas la
cause. Ce qui reste :

    sous un plafond d'appels entiers FIXE, chaque tour de certification
    paie une resolution de plus, et ce qu'elle consomme est autant de
    retire au reste.

PROTOCOLE. On tient le PRODUIT beta * rho CONSTANT, egal a 160, et on ne
fait varier que la repartition :

    beta = 160, rho = 1     un seul gros lot
    beta =  80, rho = 2     le reglage que la mesure prefere
    beta =  40, rho = 4     autant de coupes, quatre fois plus de tours
    beta =  20, rho = 8     huit tours

Le plafond de coupes est le meme dans les quatre colonnes ; seul le nombre
de tours change. C'est ce qui rend l'experience concluante la ou un
balayage a un facteur ne l'etait pas : si l'ecart se degrade en descendant
cette liste, ce n'est pas parce qu'on pose moins de coupes -- on en pose
autant -- c'est parce qu'on paie plus de tours.

LA GRANDEUR QUI MANQUAIT. Le banc precedent ne relevait pas les appels
entiers consommes par la CERTIFICATION, et c'est precisement ce que
l'hypothese met en cause. On enveloppe donc `certify` pour lire le compteur
avant et apres, sans toucher a la methode : le banc ne doit pas mesurer un
code different de celui qui tourne en production.

CE QUI REFUTERAIT L'HYPOTHESE. Que les quatre colonnes consomment le meme
nombre d'appels en certification, ou que l'ecart ne se degrade pas avec le
nombre de tours. Les deux sont possibles et le banc les afficherait tels
quels.

La ligne de production (beta = 40, rho = 2, produit 80) est jouee en plus,
hors de la serie a produit constant, pour que ce tableau se raccorde aux
autres.

Usage :  python bench_prix_tours.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys
from typing import Dict, List

import molfp_hybride as mh
from molfp_core import ORACLE_CALLS, reset_oracle_counter
from molfp_instance import generate
from bench_sensibilite import PROD

LOT = [dict(n=n, m=max(3, n // 2 + 1), p=3, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for c in (0.00, 0.50) for s in (1, 2)]

# produit beta * rho constant = 160, sauf la production, donnee pour repere
CONFIGS = [
    ("b=160 r=1   (1 tour)",   dict(cut_batch=160, cert_rounds=1)),
    ("b=80  r=2   (2 tours)",  dict(cut_batch=80,  cert_rounds=2)),
    ("b=40  r=4   (4 tours)",  dict(cut_batch=40,  cert_rounds=4)),
    ("b=20  r=8   (8 tours)",  dict(cut_batch=20,  cert_rounds=8)),
    ("b=40  r=2   (production, produit 80)", {}),
]

_CERT: List[int] = []
_ORIG = mh.certify


def _enveloppe(*a, **k):
    avant = ORACLE_CALLS["ilp"]
    r = _ORIG(*a, **k)
    _CERT.append(ORACLE_CALLS["ilp"] - avant)
    return r


mh.certify = _enveloppe


def une(inst, cap: int, graine: int, **surcharge) -> Dict:
    cfg = dict(PROD)
    cfg.update(surcharge)
    _CERT.clear()
    reset_oracle_counter()
    r = mh.matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                          seed=graine, ilp_budget=cap, **cfg)
    cert = sum(_CERT)
    return dict(posees=(r.cert or {}).get("n_cuts", 0),
                tours=(r.cert or {}).get("rounds", 0),
                ilp=r.ilp_calls, cert=cert, recherche=r.ilp_calls - cert,
                ecart=(r.gap * 100) if r.gap is not None else float("nan"),
                prouve=bool(r.proved_optimal))


def _m(v) -> float:
    v = [x for x in v if x == x]
    return statistics.median(v) if v else float("nan")


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("=" * 104)
    print("A PLAFOND DE COUPES CONSTANT (beta * rho = 160), "
          "CE QUE COUTE UN TOUR DE PLUS")
    print(f"  lot elargi : {len(LOT)} instances (n = 20 et 30) x "
          f"{len(graines)} graines, plafond {cap} appels entiers")
    print("=" * 104)
    print(f"\n{'configuration':<38}{'posees':>9}{'tours':>7}"
          f"{'appels':>8}{'dont cert.':>12}{'recherche':>11}"
          f"{'ecart med':>11}{'preuves':>10}")
    print("-" * 104)

    res: Dict[str, List[Dict]] = {}
    for nom, sur in CONFIGS:
        v = []
        for spec in LOT:
            inst = generate(**spec)
            for g in graines:
                v.append(une(inst, cap, g, **sur))
        res[nom] = v
        print(f"{nom:<38}{_m([d['posees'] for d in v]):>9.1f}"
              f"{_m([d['tours'] for d in v]):>7.1f}"
              f"{_m([d['ilp'] for d in v]):>8.1f}"
              f"{_m([d['cert'] for d in v]):>12.1f}"
              f"{_m([d['recherche'] for d in v]):>11.1f}"
              f"{_m([d['ecart'] for d in v]):>11.2f}"
              f"{sum(d['prouve'] for d in v):>6}/{len(v):<3}", flush=True)

    print("-" * 104)
    serie = [nom for nom, _ in CONFIGS[:4]]
    ec = [_m([d["ecart"] for d in res[n]]) for n in serie]
    ce = [_m([d["cert"] for d in res[n]]) for n in serie]
    po = [_m([d["posees"] for d in res[n]]) for n in serie]

    croit = all(b >= a - 1e-9 for a, b in zip(ce, ce[1:]))
    degrade = all(b >= a - 1e-9 for a, b in zip(ec, ec[1:]))
    stable = max(po) - min(po)

    print("\nCE QUE LA SERIE A PRODUIT CONSTANT MONTRE")
    print(f"  coupes posees, de 1 a 8 tours : "
          + " -> ".join(f"{x:.1f}" for x in po)
          + f"      amplitude {stable:.1f}")
    print(f"  appels en certification       : "
          + " -> ".join(f"{x:.1f}" for x in ce)
          + f"      {'CROISSANT' if croit else 'non monotone'}")
    print(f"  ecart garanti median          : "
          + " -> ".join(f"{x:.2f}" for x in ec)
          + f"      {'DEGRADE' if degrade else 'non monotone'}")
    print("\nLecture. L'hypothese demande DEUX choses a la fois : que le")
    print("nombre de coupes reste a peu pres constant le long de la serie")
    print("-- sinon la repartition n'est pas seule a varier -- et que")
    print("l'ecart se degrade a mesure que les tours se multiplient. Si")
    print("l'une des deux manque, l'hypothese tombe, et c'est un resultat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
