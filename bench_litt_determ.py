"""
bench_litt_determ.py
====================
CE QUE LA COUPE D'EFFICACITE RAPPORTE SUR LE TERRAIN DE CHACUN.

Precision qui commande ce banc. Les deux confrontations du memoire sont DEJA
comptees en appels au solveur entier, des deux cotes -- bench_zm.py et
bench_litterature.py passent time_budget = 1e6 et plafonnent par ilp_budget,
et le plafond de travail de Zerdani & Moulai porte sur leurs iterations et
leurs coupes, jamais sur des secondes. Il n'y a donc rien a « remesurer en
appels » la : c'est deja fait.

Ce qui n'a jamais ete mesure, en revanche, c'est ce que la COUPE
D'EFFICACITE elle-meme rapporte sur ces deux terrains. Le tableau du match
oppose leur methode a la notre ; celui de Drici ne rapporte que la notre.
Aucun des deux n'isole notre contribution.

Ce banc l'isole, avec le protocole applique aux trois autres lots : deux
bras en configuration courante ne differant que par archive_cuts, plafond
en appels, lecture APPARIEE, chaque configuration jouee deux fois, et les
trois axes de recul verifies -- preuve perdue, valeur plus faible, borne
plus lache.

La verite terrain est disponible sur le terrain de Zerdani & Moulai
(instances petites, enumeration exacte) et ne l'est pas sur celui de Drici
aux grandes tailles : le banc le signale au lieu de le taire.

Usage :  python bench_litt_determ.py [zm|drici|les-deux] [cap1,cap2]
"""

from __future__ import annotations

import sys
import time
from typing import Dict, List, Optional

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_matheuristic import matheuristic_P

from bench_zm import instance_zm
from bench_litterature import TAILLES_DRICI, instance_drici

BRAS = [("sans", False), ("avec", True)]
LIMITE_VT = 200_000


def une(inst, cap: int, arch: bool) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=0,
                       archive_cuts=arch, ilp_budget=cap,
                       pool_alterne=True, geom_bound=True)
    return dict(prouve=bool(r.proved_optimal),
                lb=float(r.q_lb) if r.q_lb is not None else None,
                ub=r.q_ub, ilp=r.ilp_calls,
                gap=(r.gap * 100) if r.gap is not None else float("nan"))


def campagne(nom: str, instances, cap: int, avec_vt: bool) -> Dict:
    print("=" * 96)
    print(f"{nom} -- plafond deterministe de {cap} appels  [V* S* D*]  "
          f"{len(instances)} instances")
    print("=" * 96)

    prv = {k: 0 for _, k in BRAS}
    gagne = perdu = r_val = r_borne = 0
    mieux = pire = egal = 0
    deltas: List[float] = []
    d_ilp: List[int] = []
    non_reprod: List[str] = []
    valides = controles = 0
    alertes: List[str] = []

    for etiq, inst in instances:
        q: Optional[float] = None
        if avec_vt:
            try:
                q = float(ground_truth(inst, limit=LIMITE_VT).q_star)
            except (MemoryError, ValueError):
                q = None
        res, reprod = {}, True
        for _, arch in BRAS:
            a = une(inst, cap, arch)
            b = une(inst, cap, arch)
            if (a["prouve"], a["ilp"], a["lb"]) != (b["prouve"], b["ilp"],
                                                    b["lb"]):
                reprod = False
            res[arch] = a
            prv[arch] += int(a["prouve"])
            if q is not None:
                controles += 1
                ok = (a["lb"] is not None and a["lb"] <= q + 1e-9
                      and (a["ub"] is None or a["ub"] >= q - 1e-9))
                valides += int(ok)
                if not ok:
                    alertes.append(f"  ENCADREMENT INVALIDE : {etiq} "
                                   f"({'avec' if arch else 'sans'})")
        if not reprod:
            non_reprod.append(etiq)
            continue
        A, B = res[False], res[True]
        if B["prouve"] and not A["prouve"]:
            gagne += 1
        elif A["prouve"] and not B["prouve"]:
            perdu += 1
            alertes.append(f"  PREUVE PERDUE : {etiq}")
        if (A["lb"] is not None and B["lb"] is not None
                and B["lb"] < A["lb"] - 1e-12):
            r_val += 1
            alertes.append(f"  VALEUR PLUS FAIBLE : {etiq}  "
                           f"{A['lb']:.4f} -> {B['lb']:.4f}")
        if (A["ub"] is not None and B["ub"] is not None
                and B["ub"] > A["ub"] + 1e-9):
            r_borne += 1
            alertes.append(f"  BORNE PLUS LACHE : {etiq}  "
                           f"{A['ub']:.3f} -> {B['ub']:.3f}")
        if not (np.isnan(A["gap"]) or np.isnan(B["gap"])):
            d = A["gap"] - B["gap"]
            deltas.append(d)
            if d > 1e-9:
                mieux += 1
            elif d < -1e-9:
                pire += 1
            else:
                egal += 1
        d_ilp.append(B["ilp"] - A["ilp"])

    n = len(instances)
    print(f"  reproductibilite        : {n - len(non_reprod)}/{n}")
    print(f"  optimalite prouvee      : sans {prv[False]}/{n}   "
          f"avec {prv[True]}/{n}")
    print(f"  apparie, preuves        : gagnees {gagne}   perdues {perdu}")
    print(f"  ecart garanti           : mieux {mieux}   pire {pire}   "
          f"egal {egal}")
    if deltas:
        d = np.asarray(deltas)
        print(f"  delta d'ecart (points)  : median {np.median(d):+.2f}   "
              f"moyen {d.mean():+.2f}   max {d.max():+.2f}")
    if d_ilp:
        di = np.asarray(d_ilp)
        print(f"  delta d'appels          : median {int(np.median(di)):+d}   "
              f"total {int(di.sum()):+d}")
    print(f"  recul : valeur {r_val}   borne {r_borne}")
    print(f"  encadrements valides    : "
          + (f"{valides}/{controles}" if controles else "-- sans verite terrain"))
    for a in alertes:
        print(a)
    return dict(nom=nom, cap=cap, n=n, sans=prv[False], avec=prv[True],
                gagne=gagne, perdu=perdu, rv=r_val, rb=r_borne,
                med=float(np.median(deltas)) if deltas else float("nan"),
                valides=valides, controles=controles)


def lot_zm(n_inst: int = 24):
    out = []
    for i in range(n_inst):
        n, m, p = 3 + i % 4, 2 + i % 3, 2 + i % 2
        out.append((f"zm{i:02d}_n{n}m{m}p{p}", instance_zm(n, m, p, 3000 + i)))
    return out


def lot_drici(par_taille: int = 10):
    out = []
    for (n, m, r, _cpu) in TAILLES_DRICI:
        for s in range(par_taille):
            out.append((f"drici_n{n}m{m}_s{s}", instance_drici(n, m, r, s)))
    return out


if __name__ == "__main__":
    quoi = sys.argv[1] if len(sys.argv) > 1 else "les-deux"
    caps = ([int(c) for c in sys.argv[2].split(",")]
            if len(sys.argv) > 2 else None)
    t0 = time.time()
    bilan = []
    if quoi in ("zm", "les-deux"):
        for cap in (caps or [100, 400]):
            bilan.append(campagne("TERRAIN DE ZERDANI & MOULAI", lot_zm(),
                                  cap, avec_vt=True))
            print()
    if quoi in ("drici", "les-deux"):
        for cap in (caps or [300, 1500]):
            bilan.append(campagne("TERRAIN DE DRICI, OUAIL & MOULAI",
                                  lot_drici(), cap, avec_vt=False))
            print()
    print("=" * 96)
    print("BILAN")
    for b in bilan:
        print(f"  {b['nom'][:28]:<28} plafond {b['cap']:>5} : "
              f"prouve {b['sans']}->{b['avec']} /{b['n']}   "
              f"apparie +{b['gagne']}/-{b['perdu']}   "
              f"ecart median {b['med']:+.2f}   reculs {b['rv']}+{b['rb']}")
    print(f"  ({time.time()-t0:.0f} s)")
    print("\n=== TERMINE ===")
