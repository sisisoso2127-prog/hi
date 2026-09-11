"""
bench_lot_determ.py
===================
LE LOT FIGE DE 90 INSTANCES, SOUS PLAFOND D'APPELS.

Meme question que bench_cible_determ.py, sur l'autre lot. Le tableau du lot
systematique (86/90 contre 84/90, ou 84 contre 73 selon le temoin) est lui
aussi arbitre par un budget en SECONDES -- 12 s -- c'est-a-dire par l'unite
que le memoire recuse partout ailleurs. Ce banc le rejoue en comptant les
APPELS au solveur entier.

Les deux bras portent la configuration courante ; la seule difference entre
eux est la coupe d'efficacite. Protocole du memoire applique a la lettre :
lecture APPARIEE (meme instance), chaque configuration jouee DEUX FOIS et
toute instance non reproductible exclue de la conclusion.

Trois axes de RECUL sont verifies, et pas seulement les preuves -- c'est la
definition que le banc deterministe du memoire retient :
  perdre une preuve, rendre une valeur plus faible, ou une borne superieure
  plus lache.

La validite de l'encadrement est verifiee partout ou la verite terrain est
calculable ; le manifeste dit sur quelles instances elle l'est.

Usage :  python bench_lot_determ.py [cap1,cap2,...] [--sous-lot k]
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import MOILFP
from molfp_matheuristic import matheuristic_P

LOT = os.path.join("instances", "lot_v1")
LIMITE_VT = 200_000
BRAS = [("sans", False), ("avec", True)]


def une(inst: MOILFP, cap: int, arch: bool) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6, seed=0,
                       archive_cuts=arch, ilp_budget=cap,
                       pool_alterne=True, geom_bound=True)
    return dict(prouve=bool(r.proved_optimal),
                lb=r.q_lb, ub=r.q_ub, ilp=r.ilp_calls,
                gap=(r.gap * 100) if r.gap is not None else float("nan"))


def verite(inst: MOILFP, dispo: bool) -> Optional[float]:
    if not dispo:
        return None
    try:
        return float(ground_truth(inst, limit=LIMITE_VT).q_star)
    except (MemoryError, ValueError):
        return None


def campagne(cap: int, noms: List[str], vt: Dict[str, bool]) -> Dict:
    print("=" * 96)
    print(f"LOT FIGE SOUS PLAFOND DETERMINISTE DE {cap} APPELS  [V* S* D*]  "
          f"{len(noms)} instances")
    print("=" * 96)

    prv = {k: 0 for _, k in BRAS}
    gagne = perdu = 0
    recul_val = recul_borne = 0
    non_reprod: List[str] = []
    sans_vt = 0
    valides = controles = 0
    d_ilp: List[int] = []
    detail: List[str] = []

    for nom in noms:
        inst = MOILFP.load(os.path.join(LOT, nom + ".json"))
        q = verite(inst, vt.get(nom, False))
        if q is None:
            sans_vt += 1
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
                lo = float(a["lb"]) if a["lb"] is not None else float("-inf")
                ok = (lo <= q + 1e-9) and (a["ub"] is None
                                           or a["ub"] >= q - 1e-9)
                valides += int(ok)
                if not ok:
                    detail.append(f"  ENCADREMENT INVALIDE : {nom} "
                                  f"({'avec' if arch else 'sans'})")
        if not reprod:
            non_reprod.append(nom)
            continue                      # exclue de la conclusion
        A, B = res[False], res[True]
        if B["prouve"] and not A["prouve"]:
            gagne += 1
        elif A["prouve"] and not B["prouve"]:
            perdu += 1
            detail.append(f"  PREUVE PERDUE : {nom}")
        if A["lb"] is not None and B["lb"] is not None and B["lb"] < A["lb"]:
            recul_val += 1
            detail.append(f"  VALEUR PLUS FAIBLE : {nom}")
        if (A["ub"] is not None and B["ub"] is not None
                and B["ub"] > A["ub"] + 1e-9):
            recul_borne += 1
            detail.append(f"  BORNE PLUS LACHE : {nom}")
        d_ilp.append(B["ilp"] - A["ilp"])

    n = len(noms)
    ret = len(non_reprod)
    print(f"  instances                     : {n}  "
          f"(dont {sans_vt} sans vérité terrain)")
    print(f"  reproductibilité              : {n - ret}/{n}"
          + (f"   exclues : {', '.join(non_reprod)}" if non_reprod else ""))
    print(f"  optimalité prouvée            : sans {prv[False]}/{n}   "
          f"avec {prv[True]}/{n}")
    print(f"  lecture APPARIÉE, preuves     : gagnées {gagne}   perdues {perdu}")
    print(f"  recul, valeur plus faible     : {recul_val}")
    print(f"  recul, borne plus lâche       : {recul_borne}")
    if d_ilp:
        d = np.asarray(d_ilp)
        print(f"  delta d'appels par paire      : médiane {int(np.median(d)):+d}"
              f"   total {int(d.sum()):+d}   {int((d < 0).sum())}/{len(d)} "
              f"en baisse")
    print(f"  encadrements valides          : {valides}/{controles}")
    for x in detail:
        print(x)
    return dict(cap=cap, n=n, sans=prv[False], avec=prv[True], gagne=gagne,
                perdu=perdu, rv=recul_val, rb=recul_borne,
                valides=valides, controles=controles, non_reprod=ret)


if __name__ == "__main__":
    caps = ([int(c) for c in sys.argv[1].split(",")]
            if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
            else [100, 300])
    pas = 1
    if "--sous-lot" in sys.argv:
        pas = int(sys.argv[sys.argv.index("--sous-lot") + 1])

    man = json.load(open(os.path.join(LOT, "manifest.json")))
    vt = {e["name"]: bool(e["ground_truth"]) for e in man["instances"]}
    noms = sorted(f[:-5] for f in os.listdir(LOT)
                  if f.endswith(".json") and f != "manifest.json")[::pas]

    t0 = time.time()
    bilan = []
    for cap in caps:
        bilan.append(campagne(cap, noms, vt))
        print()
    print("=" * 96)
    print("BILAN")
    for b in bilan:
        print(f"  plafond {b['cap']:>4} : sans {b['sans']}/{b['n']}   "
              f"avec {b['avec']}/{b['n']}   apparié +{b['gagne']} / -{b['perdu']}"
              f"   reculs {b['rv']}+{b['rb']}   "
              f"encadrements {b['valides']}/{b['controles']}")
    print(f"  ({time.time()-t0:.0f} s)")
    print("\n=== TERMINE ===")
