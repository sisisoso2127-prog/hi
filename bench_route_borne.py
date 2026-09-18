#!/usr/bin/env python3
"""
bench_route_borne.py
====================
LAQUELLE DES DEUX ROUTES PORTE REELLEMENT LA BORNE ?

La certification produit DEUX bornes supérieures valides et retient la
plus petite :

  -- celle du theoreme 5', q_ref + U / (Q * D), lue sur le relache coupe ;
  -- celle de la SECONDE ROUTE (borne geometrique), calculee a part.

Ce memoire mesure le GAIN de la seconde route, et il le mesure bien. Il ne
dit nulle part LAQUELLE DES DEUX porte la borne finalement publiee, et
c'est une question differente : un gain moyen de quelques points est
compatible aussi bien avec « la seconde route resserre un peu » qu'avec
« la seconde route fait tout le travail ».

La question n'est pas academique. Un sondage sur quatre executions a
montre un ecart garanti de 0 % -- donc une ligne FERMEE -- pendant que le
terme du theoreme 5' valait 7,8 ou 24,2. Autrement dit, sur ces lignes, la
borne du theoreme 5' n'a joue AUCUN role : tout venait de la seconde
route. Si ce cas est le cas general, alors plusieurs pages de ce memoire
decrivent un mecanisme qui ne decide presque jamais, et il faut le dire.

CE QUE CE BANC RELEVE, par execution : la meilleure borne du theoreme 5'
sur tous ses tours, la borne de la seconde route, celle qui est finalement
publiee, et laquelle des deux l'egale. Plus le cas ou le theoreme 5' n'a
produit AUCUNE borne -- son relache n'ayant pas rendu de majorant fini
dans le budget imparti.

CE QU'IL NE FAIT PAS. Juger la seconde route. Qu'elle porte la borne ne la
rend pas meilleure dans l'absolu : les deux sont valides, et le minimum de
deux bornes valides est une borne valide. Il etablit une repartition des
roles, et rien d'autre.

Usage :  python bench_route_borne.py [graines] [plafond]
"""

from __future__ import annotations

import statistics
import sys
from typing import Dict, List, Optional

from molfp_core import reset_oracle_counter
from molfp_instance import generate
from molfp_hybride import matheuristic_P
from bench_sensibilite import PROD

LOT = [dict(n=n, m=max(3, n // 2 + 1), p=p, seed=s, rhs_scale=1.0, corr=c)
       for n in (20, 30) for p in (3, 4) for c in (0.00, 0.50)
       for s in (1, 2)]

TOL = 1e-9


def une(inst, cap: int, graine: int) -> Dict:
    reset_oracle_counter()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=graine, ilp_budget=cap, **PROD)
    info = r.cert or {}
    trace = info.get("trace") or []
    th5 = min((d["cand"] for d in trace), default=None)
    geom = info.get("geom_ub")
    ub = r.q_ub
    return dict(th5=th5, geom=geom, ub=ub, statut=info.get("geom"),
                ecart=(r.gap * 100) if r.gap is not None else float("nan"),
                prouve=bool(r.proved_optimal))


def qui(d: Dict) -> str:
    """Quelle route egale la borne publiee."""
    ub, th5, geom = d["ub"], d["th5"], d["geom"]
    if ub is None:
        return "aucune borne"
    a = th5 is not None and abs(th5 - ub) <= TOL * max(1.0, abs(ub))
    b = geom is not None and abs(geom - ub) <= TOL * max(1.0, abs(ub))
    if a and b:
        return "les deux (egales)"
    if b:
        return "seconde route"
    if a:
        return "theoreme 5'"
    return "ni l'une ni l'autre"


def main() -> int:
    graines = list(range(int(sys.argv[1]) if len(sys.argv) > 1 else 3))
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300

    print("=" * 96)
    print("QUELLE ROUTE PORTE LA BORNE PUBLIEE ?")
    print(f"  {len(LOT)} instances x {len(graines)} graines, plafond {cap}")
    print("=" * 96)
    print(f"\n{'n':>3}{'p':>3}{'corr':>6}{'gr':>4}{'Th5':>13}{'seconde':>13}"
          f"{'publiee':>13}{'porteuse':>20}{'ecart':>9}{'pr':>4}")
    print("-" * 96)

    lect: List[Dict] = []
    for spec in LOT:
        inst = generate(**spec)
        for g in graines:
            d = une(inst, cap, g)
            d["qui"] = qui(d)
            lect.append(d)
            print(f"{spec['n']:>3}{spec['p']:>3}{spec['corr']:>6.2f}{g:>4}"
                  f"{_x(d['th5']):>13}{_x(d['geom']):>13}{_x(d['ub']):>13}"
                  f"{d['qui']:>20}{d['ecart']:>9.2f}"
                  f"{'o' if d['prouve'] else '.':>4}", flush=True)

    print("\n" + "=" * 96)
    n = len(lect)
    for etiq in ("seconde route", "theoreme 5'", "les deux (egales)",
                 "aucune borne", "ni l'une ni l'autre"):
        c = sum(1 for d in lect if d["qui"] == etiq)
        if c:
            print(f"  {etiq:<22}{c:>4}/{n}")
    sans = sum(1 for d in lect if d["th5"] is None)
    print(f"\n  le theoreme 5' n'a produit AUCUNE borne sur {sans}/{n} "
          f"executions")

    # de combien la porteuse bat-elle l'autre, quand les deux existent
    duo = [d for d in lect if d["th5"] is not None and d["geom"] is not None
           and d["ub"] not in (None, 0)]
    if duo:
        rap = [d["th5"] / d["geom"] for d in duo if d["geom"]]
        print(f"  quand les deux existent ({len(duo)} lignes), "
              f"Th5 / seconde route : mediane {statistics.median(rap):.3g}, "
              f"max {max(rap):.3g}")
        print("  Un rapport tres superieur a 1 dit que la borne du "
              "theoreme 5' est, sur ces lignes,")
        print("  sans effet sur le resultat publie.")
    return 0


def _x(v) -> str:
    return "--" if v is None else f"{v:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
