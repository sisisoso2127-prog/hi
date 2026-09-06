"""
bench_compare.py
================
Banc de comparaison entre methodes, sur le lot fige `instances/lot_v1`.

RAISON D'ETRE. Comparer deux methodes suppose trois choses qui n'existaient
pas dans ce depot : un lot d'instances identique pour toutes (`make_lot.py`),
une interface commune, et surtout une VERIFICATION INDEPENDANTE de ce que
chaque methode renvoie. Ce dernier point est le plus important : on ne
compare pas des nombres auto-declares. Toute valeur rendue est recontrolee
ici, pour toutes les methodes y compris les notres.

INTERFACE. Une methode est une fonction

    solve(inst, time_limit) -> MethodResult(q, x, status, ilp, seconds)

`status` vaut 'optimal' quand la methode PROUVE l'optimalite, 'heuristic'
quand elle rend une valeur sans preuve, 'limit' quand elle s'est arretee sans
conclure. Un statut 'optimal' engage la methode : il est verifie ci-dessous.

VERIFICATION appliquee a chaque resultat :

  C1  `x` est realisable ;
  C2  `x` est EFFICACE (Th. 2, ILP exact) -- sans quoi `q` ne minore meme pas
      `q*` et la valeur ne veut rien dire ;
  C3  `q = f(x)` -- la valeur annoncee est bien celle du point rendu ;
  C4  si la verite terrain existe : `q <= q*`, et `q = q*` des que le statut
      est 'optimal'. Un statut 'optimal' faux est une faute, pas un ecart.

COUT. On rapporte les appels ILP (`ORACLE_CALLS`), reproductibles et
independants de la machine, ET les secondes. Les deux sont necessaires : une
methode qui appelle peu mais resout des ILP enormes ne se lit que sur le
temps.

TEMOINS EXTERNES. `zerdani_moulai` (2011) et `drici` (2018) ne sont PAS
implementes : les articles n'ont pas pu etre obtenus dans cet environnement,
et reimplementer de memoire un algorithme qu'on n'a pas lu produirait un
homme de paille -- pire qu'aucun temoin, parce qu'on le battrait a coup sur.
Les emplacements sont prevus dans `METHODS` ; il suffit d'y brancher une
fonction respectant l'interface ci-dessus pour que tout le reste fonctionne,
y compris la verification.

Usage :  python bench_compare.py [budget_s] [methodes separees par des virgules]
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from fractions import Fraction
from typing import Callable, Dict, List, Optional

import numpy as np

from molfp_core import (ORACLE_CALLS, efficiency_test, reset_oracle_counter)
from molfp_enum import ground_truth
from molfp_instance import MOILFP
from molfp_matheuristic import matheuristic_P
from molfp_oracle import solve_P

LOT = os.path.join("instances", "lot_v1")
ENUM_LIMIT = 400_000


@dataclass
class MethodResult:
    q: Optional[Fraction]
    x: Optional[np.ndarray]
    status: str                 # 'optimal' | 'heuristic' | 'limit' | 'error'
    ilp: int = 0
    seconds: float = 0.0
    q_ub: Optional[float] = None      # borne sup annoncee, si la methode en a


# ----------------------------------------------------------------------------
# Methodes
# ----------------------------------------------------------------------------

def m_enum(inst: MOILFP, time_limit: float) -> MethodResult:
    """Reference : enumeration exhaustive de E puis maximisation de f.

    Aucune finesse, aucun risque : c'est la definition du probleme. Ne passe
    pas l'echelle -- c'est precisement pourquoi tout le reste existe.
    """
    t0 = time.time()
    gt = ground_truth(inst, limit=ENUM_LIMIT)
    return MethodResult(gt.q_star, gt.x_star, "optimal", 0, time.time() - t0)


def m_exact(inst: MOILFP, time_limit: float) -> MethodResult:
    """Hybride exact-exact : Dinkelbach externe + oracle lineaire sur E."""
    reset_oracle_counter()
    t0 = time.time()
    r = solve_P(inst, time_limit=time_limit)
    return MethodResult(r.q_star, r.x_star,
                        "optimal" if r.status == "optimal" else "limit",
                        ORACLE_CALLS["ilp"], time.time() - t0)


def m_matheuristic(inst: MOILFP, time_limit: float) -> MethodResult:
    """Matheuristique certifiee : recherche + borne du Th. 5'."""
    reset_oracle_counter()
    t0 = time.time()
    r = matheuristic_P(inst, time_budget=time_limit * 0.6,
                       bound_budget=time_limit * 0.4, seed=0)
    return MethodResult(r.q_lb, r.x_best,
                        "optimal" if r.proved_optimal else "heuristic",
                        ORACLE_CALLS["ilp"], time.time() - t0, r.q_ub)


METHODS: Dict[str, Callable[[MOILFP, float], MethodResult]] = {
    "enum": m_enum,
    "exact": m_exact,
    "matheuristic": m_matheuristic,
    # "zerdani_moulai": ...,   <- a brancher, cf. en-tete du module
    # "drici": ...,
}


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------

def check(inst: MOILFP, res: MethodResult, gt) -> dict:
    """C1-C4. Renvoie un dict de verdicts ; `ok` est leur conjonction."""
    out = {"C1": None, "C2": None, "C3": None, "C4": None}
    if res.x is None or res.q is None:
        # une methode qui ne rend rien n'est pas fautive, elle est muette
        out["ok"] = res.status in ("limit", "error")
        return out

    x = np.asarray(res.x, dtype=int)
    out["C1"] = bool(inst.is_feasible(x))
    out["C2"] = bool(efficiency_test(inst, x).efficient) if out["C1"] else False
    out["C3"] = bool(inst.f.value(x) == res.q) if out["C1"] else False
    if gt is not None:
        le = res.q <= gt.q_star
        eq = res.q == gt.q_star
        out["C4"] = bool(le and (eq or res.status != "optimal"))
    out["ok"] = all(v for v in out.values() if v is not None)
    return out


def main() -> int:
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0
    names = (sys.argv[2].split(",") if len(sys.argv) > 2
             else ["exact", "matheuristic"])
    for nm in names:
        if nm not in METHODS:
            print(f"methode inconnue : {nm} (disponibles : "
                  f"{', '.join(METHODS)})")
            return 2

    with open(os.path.join(LOT, "manifest.json")) as fh:
        manifest = json.load(fh)

    print("=" * 104)
    print(f"COMPARAISON SUR {manifest['version']} - budget {budget:.0f} s par "
          f"methode et par instance")
    print(f"methodes : {', '.join(names)}")
    print("=" * 104)
    head = f"{'instance':<26}{'|E|':>6}"
    for nm in names:
        head += f"{nm[:13]:>16}{'':>10}"
    print(head)
    print(f"{'':<26}{'':>6}" + "".join(f"{'valeur':>10}{'statut':>6}"
                                       f"{'ILP':>6}{'t(s)':>4}" for _ in names))
    print("-" * 104)

    agg = {nm: {"proved": 0, "best": 0, "ilp": [], "t": [], "bad": 0}
           for nm in names}
    n_inst = 0
    all_ok = True

    for meta in manifest["instances"]:
        inst = MOILFP.load(os.path.join(LOT, meta["name"] + ".json"))
        gt = ground_truth(inst, limit=ENUM_LIMIT) if meta["ground_truth"] else None
        n_inst += 1

        results = {}
        for nm in names:
            try:
                res = METHODS[nm](inst, budget)
            except Exception as exc:                      # pragma: no cover
                res = MethodResult(None, None, "error", 0, 0.0)
                print(f"  {nm} a leve {type(exc).__name__}: {exc}")
            ver = check(inst, res, gt)
            if not ver["ok"]:
                agg[nm]["bad"] += 1
                all_ok = False
            results[nm] = (res, ver)
            agg[nm]["proved"] += int(res.status == "optimal")
            agg[nm]["ilp"].append(res.ilp)
            agg[nm]["t"].append(res.seconds)

        vals = [float(r.q) for r, _ in results.values() if r.q is not None]
        best = max(vals) if vals else None
        line = f"{meta['name']:<26}{(len(gt.E) if gt else -1):>6}"
        for nm in names:
            r, v = results[nm]
            if r.q is None:
                line += f"{'-':>10}{r.status[:5]:>6}{r.ilp:>6}{r.seconds:>4.0f}"
                continue
            if best is not None and abs(float(r.q) - best) < 1e-12:
                agg[nm]["best"] += 1
            mark = "" if v["ok"] else "!"
            line += (f"{float(r.q):>10.4f}{r.status[:5] + mark:>6}"
                     f"{r.ilp:>6}{r.seconds:>4.0f}")
        print(line, flush=True)

    print("-" * 104)
    print(f"{'methode':<16}{'optimalite prouvee':>20}{'meilleure valeur':>19}"
          f"{'ILP median':>13}{'t median':>11}{'verifications KO':>19}")
    for nm in names:
        a = agg[nm]
        print(f"{nm:<16}{a['proved']:>12}/{n_inst:<7}{a['best']:>13}/{n_inst:<5}"
              f"{np.median(a['ilp']):>13.0f}{np.median(a['t']):>10.1f}s"
              f"{a['bad']:>19}")
    print("\nVERIFICATION C1-C4 : " + ("TOUT VALIDE" if all_ok else "ECHEC"))
    print("Un '!' accole a un statut signale un resultat qui n'a pas passe la")
    print("verification : la valeur annoncee ne doit alors pas etre lue.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
