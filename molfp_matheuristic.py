"""
molfp_matheuristic.py
=====================
Volet exact-metaheuristique, cible sur le regime identifie par la campagne :
|E| GROS, ou l'oracle exact derive vers un comportement d'enumeration et
n'aboutit pas dans le temps imparti.

--------------------------------------------------------------------------
THEOREME 5 — d'une borne sur le sous-probleme a une borne sur q*
--------------------------------------------------------------------------
Soit q = P/Q la valeur courante (atteinte par un point EFFICACE, donc
q <= q*), et U une borne superieure valide de

        F(q) = max_{x in E} { Q N(x) - P D(x) }.

Posons Dmin = min_{x in S} D(x) > 0 sous (A1). Alors

        q*  <=  q + U / (Q * Dmin).

Preuve. Pour tout x de E : Q N(x) - P D(x) <= U. En divisant par
Q D(x) > 0 : f(x) - q <= U / (Q D(x)) <= U / (Q Dmin), la derniere
inegalite valant car U >= 0 (U majore F(q) >= 0, l'incumbent etant dans E).
En passant au max sur E : q* - q <= U / (Q Dmin). []

Portee. C'est ce qui rend la matheuristique CERTIFIEE : il n'est plus
necessaire de resoudre le sous-probleme a l'optimum. Un oracle interrompu
fournit un U, donc un ecart d'optimalite garanti sur q*. Dmin coute un seul
ILP.

--------------------------------------------------------------------------
THEOREME 5' — version resserree
--------------------------------------------------------------------------
Le Th. 5 est valide mais lache d'un facteur ~59 en pratique. Il gaspille
trois informations. Soit R un relache quelconque de E (R contient E), obtenu
par accumulation de coupes de dominance (Th. 4). Posons

        U   >=  F_R(q) = max_{x in R} { Q N(x) - P D(x) }
        D+  =   min { D(x) : x in R,  Q N(x) - P D(x) >= 0 }

Alors

        q*  <=  q + U / (Q * D+)     et si U = 0, q* = q.

Preuve. Soit x dans E, donc dans R. Si Q N(x) - P D(x) < 0 alors f(x) < q.
Sinon x appartient a la region definissant D+, donc D(x) >= D+ et
f(x) - q = (Q N(x) - P D(x)) / (Q D(x)) <= U / (Q D+). []

Trois gains sur le Th. 5 :
  1. R au lieu de S     -> U plus petit : les coupes retirent des zones sans
                           aucune solution efficace ;
  2. D+ au lieu de Dmin -> on ne minimise D que la ou la borne agit, c'est-a
                           dire la ou f depasse q. Le minimum portant sur un
                           sous-ensemble, D+ >= Dmin : borne plus fine ;
  3. les points DOMINES traverses par les chaines de reparation pendant la
     phase heuristique sont autant de coupes DEJA PAYEES. On les recycle,
     en priorite ceux de plus grande valeur du substitut : ce sont eux qui
     tirent U vers le haut.

Remarque. La region definissant D+ n'est jamais vide : l'incumbent y
appartient, son residu valant exactement 0. Une infaisabilite signalerait
donc un bug, pas une preuve d'optimalite.

--------------------------------------------------------------------------
RECHERCHE : mouvements exacts dans l'espace des criteres
--------------------------------------------------------------------------
Le Th. 4 fournit e_k(x) = Dbar_k D_k(x) (Z_k(x) - Z_k(xbar)), entiere, donc

        Z_k(x) >= Z_k(xbar)  <=>  e_k(x) >= 0
        Z_k(x) >  Z_k(xbar)  <=>  e_k(x) >= 1

Le pas "+1" est exact : aucun epsilon a regler, contrairement aux schemas
epsilon-contrainte usuels.

ERREUR DE CONCEPTION CORRIGEE. Une premiere version prenait pour voisinage
   V_k(x^r) = { x : e_k(x) >= 1, e_j(x) >= 0 pour j != k }
c'est-a-dire "mieux sur k sans rien perdre ailleurs". Or c'est exactement
l'ensemble des points qui DOMINENT x^r : il est VIDE des que x^r est
efficace. Le voisinage etait donc vide par construction et la recherche ne
progressait pas. Un arbitrage sur les autres criteres est indispensable.

Trois mouvements sont utilises :

  A  epsilon partiel : e_k(x) >= 1, et des planchers e_j(x) >= 0 sur un
     SOUS-ENSEMBLE aleatoire J des autres criteres (J peut etre vide).
     J = {} donne le mouvement le plus libre, J = tout donne le voisinage
     vide ci-dessus : la taille de J regle l'intensification.
  B  plancher absolu : Z_j(x) >= eps_j avec eps_j tire entre nadir et ideal,
     pour explorer une region non encore visitee (Th. 1).
  C  LNS / fix-and-optimize : on fige une fraction des variables a leur
     valeur dans l'incumbent et on resout exactement le reste.

Dans les trois cas le sous-probleme est un ILP resolu EXACTEMENT, puis le
point obtenu est certifie efficace par le Th. 2. Choix des regions =
heuristique, resolution dans chaque region = exacte.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from molfp_core import (INF, ORACLE_CALLS, efficiency_test, feasibility_rows,
                        solve_ilp)
from molfp_instance import MOILFP
from molfp_oracle import ECutModel, max_linear_over_E, repair_to_efficient

Row = Tuple[np.ndarray, float, float]


# ----------------------------------------------------------------------------
# Outils
# ----------------------------------------------------------------------------

def d_min(inst: MOILFP) -> int:
    """Dmin = min_{x in S} d^T x + beta  (un seul ILP). Strictement positif."""
    res = solve_ilp(inst.f.den.astype(float), feasibility_rows(inst),
                    inst.var_upper_bounds(), maximize=False,
                    obj_const=float(inst.f.b))
    return max(1, int(round(res.obj))) if res.ok else int(inst.f.b)


def d_plus(inst: MOILFP, model: ECutModel,
           w: np.ndarray, w0: float) -> Optional[int]:
    """
    D+ = min { D(x) : x dans R,  w^T x + w0 >= 0 }   (Th. 5')

    Un seul ILP. La region n'est jamais vide (l'incumbent y est), donc un
    statut infaisable signale un probleme et on renvoie None pour laisser
    l'appelant retomber sur Dmin.
    """
    extra = [(np.asarray(w, dtype=float), -float(w0), INF)]
    res = model.optimize(inst.f.den.astype(float), float(inst.f.b),
                         maximize=False, extra_rows=extra)
    if not res.ok:
        return None
    return max(1, int(round(res.obj)))


def build_certification_model(inst: MOILFP, dominated: Sequence[np.ndarray],
                              w: np.ndarray, max_cuts: int = 40) -> ECutModel:
    """
    Construit R en recyclant les points DOMINES rencontres pendant la phase
    heuristique (Th. 5', gain 3).

    Priorite aux points de plus grande valeur du substitut w : ce sont eux qui
    tirent U vers le haut, donc les couper est ce qui resserre le plus la
    borne. Le nombre de coupes est plafonne, chacune ajoutant p binaires.
    """
    model = ECutModel(inst)
    seen = set()
    uniq = []
    for x in dominated:
        key = tuple(int(v) for v in x)
        if key not in seen:
            seen.add(key)
            uniq.append(x)
    uniq.sort(key=lambda x: -float(np.asarray(w, dtype=float) @ x))
    for x in uniq[:max_cuts]:
        model.add_dominance_cut(x)
    return model


def certify(inst: MOILFP, q: Fraction, dominated: Sequence[np.ndarray],
            budget: float, use_tightened: bool = True
            ) -> Tuple[Optional[float], bool, dict]:
    """
    Convertit un budget de calcul en borne superieure valide sur q*.

    Renvoie (q_ub, optimalite_prouvee, diagnostic).
    `use_tightened=False` applique le Th. 5 d'origine : sert de temoin pour
    mesurer le gain apporte par le Th. 5'.
    """
    w, w0, P, Q = surrogate(inst, q)
    info: dict = {"n_cuts": 0, "U": None, "Dmin": None, "Dplus": None}

    model = build_certification_model(inst, dominated, w) if use_tightened \
        else ECutModel(inst)
    info["n_cuts"] = model.n_cuts

    r = max_linear_over_E(inst, w, w0, time_limit=budget, model=model,
                          collect=False)

    # optimalite prouvee : le sous-probleme est resolu et F(q) <= 0
    if r.status == "optimal" and r.value is not None and r.value <= 1e-9:
        return float(q), True, info

    if r.ub is None:
        return None, False, info
    U = max(0.0, float(r.ub))
    info["U"] = U
    if U <= 1e-9:
        return float(q), True, info

    Dm = d_min(inst)
    info["Dmin"] = Dm
    denom = Dm
    if use_tightened:
        Dp = d_plus(inst, model, w, w0)
        info["Dplus"] = Dp
        if Dp is not None:
            denom = max(Dm, Dp)          # D+ >= Dmin par construction

    return float(q) + U / (Q * denom), False, info


def e_row(inst: MOILFP, xbar: np.ndarray, k: int) -> Tuple[np.ndarray, float]:
    """Coefficients et constante de e_k (Th. 4), entiers."""
    Zk = inst.Z[k]
    Nb, Db = Zk.numerator(xbar), Zk.denominator(xbar)
    return (Db * Zk.num - Nb * Zk.den).astype(float), float(Db * Zk.a - Nb * Zk.b)


def surrogate(inst: MOILFP, q: Fraction) -> Tuple[np.ndarray, float, int, int]:
    """Substitut lineaire de Dinkelbach en q = P/Q : (w, w0, P, Q)."""
    f = inst.f
    P, Q = q.numerator, q.denominator
    return (Q * f.num - P * f.den).astype(float), float(Q * f.a - P * f.b), P, Q


# ----------------------------------------------------------------------------
# Archive de points efficaces certifies
# ----------------------------------------------------------------------------

class Archive:
    """Points efficaces certifies, indexes par vecteur criteres (exact)."""

    def __init__(self, inst: MOILFP):
        self.inst = inst
        self._by_z: Dict[Tuple[Fraction, ...], np.ndarray] = {}

    def add(self, x: np.ndarray) -> bool:
        z = self.inst.criteria(x)
        if z in self._by_z:
            return False
        self._by_z[z] = np.array(x, dtype=int)
        return True

    def points(self) -> List[np.ndarray]:
        return list(self._by_z.values())

    def __len__(self) -> int:
        return len(self._by_z)


# ----------------------------------------------------------------------------
# Voisinage exact dans l'espace des criteres
# ----------------------------------------------------------------------------

def move_epsilon_partial(inst: MOILFP, xr: np.ndarray, k: int,
                         keep: Sequence[int],
                         w: np.ndarray, w0: float) -> Optional[np.ndarray]:
    """
    Mouvement A.  max w^T x + w0  s.c.  x in S,  e_k(x) >= 1,
                  e_j(x) >= 0 pour j dans `keep` (j != k).

    `keep` vide  -> on accepte n'importe quelle degradation ailleurs
                    (mouvement le plus explorateur) ;
    `keep` plein -> voisinage vide (les points dominant x^r) : a eviter.
    """
    rows = feasibility_rows(inst)
    coef, const = e_row(inst, xr, k)
    rows.append((coef, 1.0 - const, INF))
    for j in keep:
        if j == k:
            continue
        cj, kj = e_row(inst, xr, j)
        rows.append((cj, -kj, INF))
    res = solve_ilp(w, rows, inst.var_upper_bounds(), maximize=True,
                    obj_const=w0)
    return res.x if res.ok else None


def move_epsilon_absolute(inst: MOILFP, eps: Sequence[Fraction],
                          w: np.ndarray, w0: float) -> Optional[np.ndarray]:
    """Mouvement B.  max w  s.c. x in S, Z_k(x) >= eps_k (Th. 1)."""
    from molfp_core import threshold_row
    rows = feasibility_rows(inst)
    for k, v in enumerate(eps):
        if v is not None:
            rows.append(threshold_row(inst.Z[k], v))
    res = solve_ilp(w, rows, inst.var_upper_bounds(), maximize=True,
                    obj_const=w0)
    return res.x if res.ok else None


def move_lns(inst: MOILFP, xr: np.ndarray, free_idx: Sequence[int],
             w: np.ndarray, w0: float) -> Optional[np.ndarray]:
    """Mouvement C.  Fige x_j = xr_j hors de `free_idx`, resout le reste."""
    lb = np.array(xr, dtype=float)
    ub = np.array(xr, dtype=float)
    box = inst.var_upper_bounds().astype(float)
    for j in free_idx:
        lb[j], ub[j] = 0.0, box[j]
    from molfp_core import solve_milp
    rows = [(inst.A[i].astype(float), -INF, float(inst.b[i]))
            for i in range(inst.m)]
    res = solve_milp(w, rows, lb, ub, maximize=True, obj_const=w0)
    return res.x if res.ok else None


# ----------------------------------------------------------------------------
# Matheuristique
# ----------------------------------------------------------------------------

@dataclass
class MatheurResult:
    q_lb: Optional[Fraction]        # meilleure valeur CERTIFIEE (x_best in E)
    x_best: Optional[np.ndarray]
    q_ub: Optional[float]           # borne superieure valide sur q* (Th. 5)
    archive: List[np.ndarray] = field(default_factory=list)
    ilp_calls: int = 0
    time: float = 0.0
    rounds: int = 0
    proved_optimal: bool = False
    status: str = "heuristic"
    cert: dict = field(default_factory=dict)   # diagnostic du Th. 5'

    @property
    def gap(self) -> Optional[float]:
        if self.q_lb is None or self.q_ub is None:
            return None
        lb = float(self.q_lb)
        return (self.q_ub - lb) / max(1e-12, abs(self.q_ub))


def matheuristic_P(inst: MOILFP,
                   time_budget: float = 20.0,
                   bound_budget: float = 10.0,
                   seed: int = 0,
                   certify_bound: bool = True,
                   tightened: bool = True,
                   verbose: bool = False) -> MatheurResult:
    """
    Phase 1 (heuristique) : VNS dans l'espace des criteres, sous-problemes
                            exacts sur chaque voisinage V_k. Produit un
                            incumbent EFFICACE certifie (Th. 2) donc un LB sur.
    Phase 2 (certification) : oracle exact lance avec un budget borne au point
                            q obtenu ; sa borne superieure U sur F(q) est
                            convertie en borne sur q* par le Th. 5.

    L'incumbent est toujours un vrai point efficace : le LB n'est jamais
    optimiste, meme si la recherche est interrompue.
    """
    t0 = time.time()
    calls0 = ORACLE_CALLS["ilp"]
    rng = np.random.default_rng(seed)
    arch = Archive(inst)

    # --- amorcage : un point efficace quelconque --------------------------
    dominated: List[np.ndarray] = []      # recyclage pour le Th. 5' (gain 3)
    x0 = repair_to_efficient(inst, np.zeros(inst.n, dtype=int),
                             dominated_out=dominated)
    arch.add(x0)
    q = inst.f.value(x0)
    x_best = x0

    # --- bornes de l'espace des criteres (pour le mouvement B) ------------
    from molfp_core import ideal_nadir_estimates
    ideal, nadir = ideal_nadir_estimates(inst)

    rounds = 0
    n_moves = {"A": 0, "B": 0, "C": 0}
    n_hits = {"A": 0, "B": 0, "C": 0}

    # --- phase 1 : recherche dans l'espace des criteres --------------------
    while time.time() - t0 < time_budget:
        rounds += 1
        w, w0, _, _ = surrogate(inst, q)
        improved = False

        pool = arch.points()
        pool.sort(key=lambda x: -float(inst.f.value(x)))
        base_pool = pool[:8]
        if len(pool) > 8:
            idx = rng.choice(len(pool), size=min(4, len(pool)), replace=False)
            base_pool += [pool[i] for i in idx]

        for xr in base_pool:
            for k in rng.permutation(inst.p):
                if time.time() - t0 >= time_budget:
                    break
                mv = rng.choice(["A", "A", "B", "C"])   # A privilegie
                k = int(k)

                if mv == "A":
                    # sous-ensemble STRICT des autres criteres : jamais tous,
                    # sinon le voisinage est vide (cf. en-tete du module)
                    others = [j for j in range(inst.p) if j != k]
                    n_keep = int(rng.integers(0, max(1, len(others))))
                    keep = list(rng.choice(others, size=n_keep, replace=False)) \
                        if n_keep else []
                    y = move_epsilon_partial(inst, xr, k, keep, w, w0)
                elif mv == "B":
                    t = rng.random(inst.p)
                    eps = [nadir[j] + Fraction(float(t[j])).limit_denominator(64)
                           * (ideal[j] - nadir[j]) for j in range(inst.p)]
                    eps = [eps[j] if rng.random() < 0.6 else None
                           for j in range(inst.p)]
                    y = move_epsilon_absolute(inst, eps, w, w0)
                else:
                    n_free = max(1, int(0.4 * inst.n))
                    free = rng.choice(inst.n, size=n_free, replace=False)
                    y = move_lns(inst, xr, free, w, w0)

                n_moves[mv] += 1
                if y is None:
                    continue
                y = repair_to_efficient(inst, y,       # certification Th. 2
                                        dominated_out=dominated)
                arch.add(y)
                fy = inst.f.value(y)
                if fy > q:
                    q, x_best, improved = fy, y, True
                    n_hits[mv] += 1
                    w, w0, _, _ = surrogate(inst, q)
            if time.time() - t0 >= time_budget:
                break

        if verbose:
            print(f"  tour {rounds}: q = {float(q):.6f}  |archive| = {len(arch)}"
                  f"  mouvements {n_moves} succes {n_hits}"
                  f"  ({time.time()-t0:.1f}s)")

        if not improved and rounds > 2:
            break            # optimum local en espace des criteres

    # --- phase 2 : certification (Th. 5') --------------------------------
    q_ub, proved, cert_info = None, False, {}
    if certify_bound and bound_budget > 0:
        q_ub, proved, cert_info = certify(inst, q, dominated, bound_budget,
                                          use_tightened=tightened)

    return MatheurResult(
        q_lb=q, x_best=x_best, q_ub=q_ub, archive=arch.points(),
        ilp_calls=ORACLE_CALLS["ilp"] - calls0, time=time.time() - t0,
        rounds=rounds, proved_optimal=proved,
        status="optimal" if proved else "heuristic", cert=cert_info,
    )
