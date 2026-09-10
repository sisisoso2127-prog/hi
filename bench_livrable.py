"""
bench_livrable.py
=================
L'ARCHIVE MESUREE COMME LIVRABLE, ET NON COMME SOUS-PRODUIT.

Le probleme pose demande « un sous-ensemble de solutions efficaces engendre
par des algorithmes hybrides ». Notre methode en produit un : l'archive A,
dont chaque point est CERTIFIE efficace (Th. 2) avant d'y entrer. Jusqu'ici
elle n'etait mesuree nulle part -- elle servait a amorcer la recherche et a
poser des coupes, jamais a etre rendue. Ce banc la mesure.

Il ne s'agit pas de flatter la methode. L'archive est engendree par une
recherche PILOTEE PAR f : rien dans l'algorithme ne lui demande de couvrir
le front. Si la mesure montre qu'elle ne le couvre pas, c'est un resultat, et
il doit etre ecrit tel quel -- avec la nuance que la partie du front que f
designe est justement celle que le sujet demande.

Cinq indicateurs (indicateurs.py), contre la verite terrain E enumeree :

  purete        part des points de A reellement efficaces. Doit valoir 1
                EXACTEMENT, sans quoi la certification est fausse : c'est
                d'abord un test de correction, ensuite un indicateur.
  |Z(A)|/|Z(E)| cardinalite, en vecteurs criteres distincts.
  C(A,E)        couverture : part de E faiblement dominee par un point de A.
  HV(A)/HV(E)   hypervolume exact, reference = nadir de E abaisse de 10 %.
  eps+          decalage normalise a ajouter a A pour qu'elle domine E.

PARTIE A -- un lot d'instances enumerables, a budget fixe. On y lit ce que
l'archive vaut telle qu'elle sort aujourd'hui, et si le point f-optimal en
fait partie (il doit : c'est l'incumbent).

PARTIE B -- le meme lot a trois budgets ILP. La question decisive pour un
livrable : l'archive s'AMELIORE-t-elle quand on paie ? Un front approche qui
ne bouge pas avec l'effort n'est pas un livrable, c'est un artefact.

Usage :  python bench_livrable.py [partie]      partie dans {A, B, AB}
"""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, List

from molfp_core import reset_oracle_counter
from molfp_enum import ground_truth
from molfp_instance import generate
from molfp_matheuristic import matheuristic_P
from indicateurs import indicateurs

# ---------------------------------------------------------------------------
# CONFIGURATION MESUREE  (meme convention que les autres bancs)
# ---------------------------------------------------------------------------
CFG = os.environ.get("MOLFP_CFG", "eee").lower()
CFG_ALT = CFG[0:1] in ("e", "*", "1")
CFG_GEOM = CFG[1:2] in ("e", "*", "1")


def marqueur() -> str:
    return f"[V{'*' if CFG_ALT else 'o'} S{'*' if CFG_GEOM else 'o'} D*]"


LIMITE_S = 400_000

# Instances enumerables : la verite terrain E doit etre calculable, sinon
# aucun de ces indicateurs n'a de sens. corr = 0 donne un front epais,
# corr = 0.5 un front mince : les deux regimes ne se lisent pas pareil.
LOT = [
    dict(n=6,  m=4, p=3, seed=1, rhs_scale=1.5, corr=0.00),
    dict(n=6,  m=4, p=3, seed=2, rhs_scale=1.5, corr=0.00),
    dict(n=7,  m=4, p=3, seed=1, rhs_scale=1.2, corr=0.00),
    dict(n=7,  m=4, p=4, seed=3, rhs_scale=1.2, corr=0.00),
    dict(n=8,  m=4, p=3, seed=1, rhs_scale=1.0, corr=0.00),
    dict(n=6,  m=4, p=3, seed=1, rhs_scale=1.5, corr=0.50),
    dict(n=7,  m=4, p=3, seed=1, rhs_scale=1.2, corr=0.50),
    dict(n=8,  m=4, p=3, seed=1, rhs_scale=1.0, corr=0.50),
]

BUDGETS = [60, 180, 540]


def mesurer(spec: Dict, cap: int) -> Dict[str, object]:
    inst = generate(**spec)
    gt = ground_truth(inst, limit=LIMITE_S)

    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=1e6, bound_budget=1e6,
                       seed=spec["seed"], archive_cuts=True, ilp_budget=cap,
                       pool_alterne=CFG_ALT, geom_bound=CFG_GEOM)
    dt = time.time() - t0

    ZE = [tuple(float(c) for c in z) for z in gt.ZE]
    ZA = [tuple(float(c) for c in inst.criteria(a)) for a in r.archive]
    ind = indicateurs(ZA, ZE) if ZA else dict(
        purete=1.0, card=0.0, nA=0, nE=len({tuple(z) for z in ZE}),
        couverture=0.0, hv=0.0, eps=float("inf"))

    # l'incumbent doit etre DANS l'archive : c'est le point que le sujet
    # designe, et le seul dont l'optimalite soit en jeu.
    z_star = tuple(float(c) for c in inst.criteria(gt.x_star))
    dedans = z_star in {tuple(z) for z in ZA}
    exact = (r.q_lb is not None and r.q_lb == gt.q_star)

    return dict(spec=spec, cap=cap, S=len(gt.S), E=len(gt.E),
                ilp=r.ilp_calls, t=dt, opt=exact, zstar=dedans, **ind)


def ligne(d: Dict[str, object]) -> str:
    s = d["spec"]
    eps = d["eps"]
    return (f"{s['n']:4d} {s['p']:3d} {s['corr']:5.2f} {d['S']:7d} "
            f"{d['nE']:5d} {d['nA']:5d} {100*d['card']:7.1f} "
            f"{100*d['couverture']:8.1f} {100*d['hv']:7.1f} "
            f"{eps:7.3f} {d['purete']:6.2f} "
            f"{'oui' if d['zstar'] else 'NON':>5} {d['ilp']:5d}")


ENTETE = ("   n   p  corr      |S|  |ZE|  |ZA|  card%  couv%    HV%   eps+ "
          "purete  z*∈A   ILP")
BARRE = "-" * len(ENTETE)


def partie_A() -> List[Dict]:
    cap = BUDGETS[1]
    print("=" * len(ENTETE))
    print(f"PARTIE A -- l'archive comme front approche, budget ILP = {cap}  "
          f"{marqueur()}")
    print("=" * len(ENTETE))
    print(ENTETE)
    print(BARRE)
    res = []
    for spec in LOT:
        d = mesurer(spec, cap)
        res.append(d)
        print(ligne(d), flush=True)
    print(BARRE)
    med = lambda k: sorted(x[k] for x in res)[len(res) // 2]
    print(f"  medianes : card {100*med('card'):.1f}%   couv "
          f"{100*med('couverture'):.1f}%   HV {100*med('hv'):.1f}%   "
          f"eps+ {med('eps'):.3f}")
    mauvais = [x for x in res if x["purete"] < 1.0]
    print(f"  purete = 1 sur {len(res)-len(mauvais)}/{len(res)} instances"
          + ("" if not mauvais else "   *** CERTIFICATION EN DEFAUT ***"))
    absents = [x for x in res if not x["zstar"]]
    print(f"  z* present dans l'archive : {len(res)-len(absents)}/{len(res)}")
    return res


def partie_B() -> None:
    print()
    print("=" * len(ENTETE))
    print(f"PARTIE B -- l'archive s'ameliore-t-elle avec le budget ?  "
          f"{marqueur()}")
    print("=" * len(ENTETE))
    print(f"{'   n   p  corr':>15}" + "".join(
        f"{'|ZA|@'+str(c):>10}" for c in BUDGETS)
        + "".join(f"{'couv@'+str(c):>10}" for c in BUDGETS)
        + "".join(f"{'HV@'+str(c):>10}" for c in BUDGETS))
    print(BARRE)
    croiss_card = croiss_hv = plats = 0
    for spec in LOT:
        ds = [mesurer(spec, c) for c in BUDGETS]
        print(f"{spec['n']:4d} {spec['p']:3d} {spec['corr']:6.2f} "
              + "".join(f"{d['nA']:10d}" for d in ds)
              + "".join(f"{100*d['couverture']:9.1f}%" for d in ds)
              + "".join(f"{100*d['hv']:9.1f}%" for d in ds), flush=True)
        if ds[-1]["nA"] > ds[0]["nA"]:
            croiss_card += 1
        if ds[-1]["hv"] > ds[0]["hv"] + 1e-9:
            croiss_hv += 1
        if ds[-1]["nA"] == ds[0]["nA"] and abs(ds[-1]["hv"] - ds[0]["hv"]) < 1e-9:
            plats += 1
    print(BARRE)
    print(f"  archive plus grande en triplant trois fois le budget : "
          f"{croiss_card}/{len(LOT)}")
    print(f"  hypervolume en hausse                                : "
          f"{croiss_hv}/{len(LOT)}")
    print(f"  aucun mouvement du tout                              : "
          f"{plats}/{len(LOT)}")


if __name__ == "__main__":
    quoi = (sys.argv[1] if len(sys.argv) > 1 else "AB").upper()
    if "A" in quoi:
        partie_A()
    if "B" in quoi:
        partie_B()
    print("\n=== TERMINE ===")
