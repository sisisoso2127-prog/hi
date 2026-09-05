"""
molfp_core.py
=============
Briques exactes de base, toutes ENTIERES (aucune tolerance numerique).

  Theoreme 1 (linearisation de seuil)
      Sous (A1),  Z_k(x) >= v  <=>  (c_k - v d_k)^T x >= v beta_k - alpha_k.
      Si v = N/D avec D > 0 entier, on multiplie par D :
          (D c_k - N d_k)^T x >= N beta_k - D alpha_k
      => contrainte a coefficients ENTIERS.

  Theoreme 2 (test d'efficacite, forme integrale)
      Pour xbar realisable, N_k = N_k(xbar), D_k = D_k(xbar) > 0 :

        theta(xbar) = max  sum_k [ (D_k c_k - N_k d_k)^T x + D_k alpha_k - N_k beta_k ]
                      s.c. (D_k c_k - N_k d_k)^T x >= N_k beta_k - D_k alpha_k,  k=1..p
                           A x <= b,  x in Z^n_+

      Le terme k vaut D_k(xbar) * D_k(x) * (Z_k(x) - Z_k(xbar)), de meme signe
      que Z_k(x) - Z_k(xbar) car les denominateurs sont > 0.
      Donc :  xbar efficace  <=>  theta(xbar) = 0.
      theta est un ENTIER : le test est exact, sans epsilon.

  Theoreme 3 (Dinkelbach sur un ensemble fini)
      F(q) = max_{x in X} { N(x) - q D(x) } est convexe, lineaire par morceaux,
      strictement decroissante, de racine unique q* = max_{x in X} f(x).
      X etant FINI, l'iteration de Newton q_{t+1} = f(x_t) converge en un
      nombre FINI d'iterations. Avec q = P/Q rationnel on resout
          max (Q num - P den)^T x + (Q a - P b)
      a coefficients entiers : critere d'arret exact "valeur == 0".

Solveur : scipy.optimize.milp (HiGHS). Remplacable par Gurobi/CPLEX en
reimplementant uniquement solve_ilp().
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from molfp_instance import MOILFP, FracObj

INF = np.inf


# ----------------------------------------------------------------------------
# Couche solveur (unique point a remplacer pour changer de solveur)
# ----------------------------------------------------------------------------

@dataclass
class ILPResult:
    status: str                    # 'optimal' | 'infeasible' | 'unbounded' | 'error'
    x: Optional[np.ndarray]
    obj: Optional[float]
    n_calls: int = 1

    @property
    def ok(self) -> bool:
        return self.status == "optimal"


# compteurs globaux d'appels au solveur : l'unite de cout a rapporter dans
# les experiences (bien plus reproductible que les secondes CPU). Les LP de
# renforcement du big-M sont comptes a part : ils sont d'un ordre de grandeur
# moins chers qu'un ILP et les melanger fausserait la comparaison.
ORACLE_CALLS = {"ilp": 0, "lp": 0}


def reset_oracle_counter() -> None:
    ORACLE_CALLS["ilp"] = 0
    ORACLE_CALLS["lp"] = 0


def solve_ilp(obj: np.ndarray,
              rows: Sequence[Tuple[np.ndarray, float, float]],
              var_ub: np.ndarray,
              maximize: bool = True,
              obj_const: float = 0.0) -> ILPResult:
    """Cas particulier : variables x uniquement, 0 <= x <= var_ub, entieres."""
    return solve_milp(obj, rows, np.zeros(len(obj)), var_ub,
                      maximize=maximize, obj_const=obj_const)


def solve_milp(obj: np.ndarray,
               rows: Sequence[Tuple[np.ndarray, float, float]],
               var_lb: np.ndarray,
               var_ub: np.ndarray,
               integrality: Optional[np.ndarray] = None,
               maximize: bool = True,
               obj_const: float = 0.0) -> ILPResult:
    """
    Version generale de solve_ilp acceptant des bornes et une integralite par
    variable. Necessaire pour les coupes de dominance, qui introduisent des
    variables binaires auxiliaires.

    integrality = None  ->  toutes les variables sont entieres (cas courant
    ici : x entier et les auxiliaires u binaires).
    """
    ORACLE_CALLS["ilp"] += 1
    cost = -np.asarray(obj, dtype=float) if maximize else np.asarray(obj, dtype=float)
    if integrality is None:
        integrality = np.ones(len(obj))

    cons = []
    if rows:
        Amat = np.array([r[0] for r in rows], dtype=float)
        lbs = np.array([r[1] for r in rows], dtype=float)
        ubs = np.array([r[2] for r in rows], dtype=float)
        cons.append(LinearConstraint(Amat, lbs, ubs))

    res = milp(c=cost, constraints=cons,
               integrality=np.asarray(integrality, dtype=float),
               bounds=Bounds(np.asarray(var_lb, dtype=float),
                             np.asarray(var_ub, dtype=float)))

    if res.status == 0:
        x = np.rint(res.x).astype(int)
        return ILPResult("optimal", x, float(obj @ x) + obj_const)
    if res.status == 2:
        return ILPResult("infeasible", None, None)
    if res.status == 3:
        return ILPResult("unbounded", None, None)
    return ILPResult("error", None, None)


# ----------------------------------------------------------------------------
# Minorant lineaire par relaxation continue (renforcement du big-M)
# ----------------------------------------------------------------------------

def min_over_relaxation(coef: np.ndarray,
                        const: float,
                        rows: Sequence[Tuple[np.ndarray, float, float]],
                        var_ub: np.ndarray) -> float:
    """
    Minorant VALIDE de  min { coef^T x + const : x in S }  obtenu sur la
    relaxation continue de S = { x entier, 0 <= x <= var_ub, rows }.

    Sert a renforcer le big-M des coupes de dominance. Deux raisons de le
    prendre ici plutot que sur la boite seule :

    * la relaxation continue contient S, donc son minimum minore celui sur S :
      la borne reste valide ;
    * elle est contenue dans la boite, donc son minimum est >= celui sur la
      boite : la borne est mecaniquement plus fine, et souvent de beaucoup
      des que A x <= b mord.

    Les coefficients etant ENTIERS et x entier, coef^T x + const est entier :
    on remonte donc au plafond entier du minorant continu, ce qui resserre
    encore sans rien supposer. La marge 1e-6 absorbe l'erreur du simplexe --
    elle ne peut que relacher la borne, donc pas invalider le big-M.

    Renvoie -inf si le LP echoue : l'appelant retombe alors sur la boite.
    """
    ORACLE_CALLS["lp"] += 1
    bounds = [(0.0, float(u)) for u in var_ub]
    A_ub, b_ub = [], []
    for r_coef, lo, hi in rows:
        r = np.asarray(r_coef, dtype=float)
        if np.isfinite(hi):
            A_ub.append(r)
            b_ub.append(float(hi))
        if np.isfinite(lo):
            A_ub.append(-r)
            b_ub.append(-float(lo))

    res = linprog(c=np.asarray(coef, dtype=float),
                  A_ub=np.array(A_ub) if A_ub else None,
                  b_ub=np.array(b_ub) if b_ub else None,
                  bounds=bounds, method="highs")
    if not res.success:
        return -np.inf

    c = np.asarray(coef, dtype=float)
    integral = bool(np.all(c == np.rint(c)))     # coef^T x entier pour x entier
    val = float(np.ceil(res.fun - 1e-6)) if integral else float(res.fun)
    return val + const


# ----------------------------------------------------------------------------
# Theoreme 1 : contraintes de seuil integrales
# ----------------------------------------------------------------------------

def threshold_row(obj: FracObj, v: Fraction) -> Tuple[np.ndarray, float, float]:
    """
    Renvoie la ligne (coef, lb, ub) codant  obj(x) >= v,  a coefficients entiers.

        v = N/D, D > 0  =>  (D*num - N*den)^T x >= N*b - D*a
    """
    N, D = v.numerator, v.denominator      # Fraction : D > 0 toujours
    coef = D * obj.num - N * obj.den
    rhs = N * obj.b - D * obj.a
    return coef.astype(float), float(rhs), INF


def epsilon_rows(inst: MOILFP, eps: Sequence[Fraction],
                 skip: Optional[int] = None) -> List[Tuple[np.ndarray, float, float]]:
    """Contraintes  Z_k(x) >= eps_k  pour tout k (sauf 'skip' eventuellement)."""
    return [threshold_row(inst.Z[k], eps[k])
            for k in range(inst.p) if k != skip]


def feasibility_rows(inst: MOILFP) -> List[Tuple[np.ndarray, float, float]]:
    """Contraintes  A x <= b."""
    return [(inst.A[i].astype(float), -INF, float(inst.b[i]))
            for i in range(inst.m)]


# ----------------------------------------------------------------------------
# Theoreme 2 : test d'efficacite exact
# ----------------------------------------------------------------------------

@dataclass
class EfficiencyResult:
    efficient: bool
    theta: Optional[int]           # entier ; 0 <=> efficace
    dominator: Optional[np.ndarray]  # solution dominante trouvee si non efficace


def efficiency_test(inst: MOILFP, xbar: np.ndarray) -> EfficiencyResult:
    """
    Test d'efficacite integral de xbar pour (MOILFP).  Un seul ILP.
    Renvoie theta et, si xbar n'est pas efficace, une solution qui le domine.
    """
    if not inst.is_feasible(xbar):
        raise ValueError("xbar n'est pas realisable.")

    ub = inst.var_upper_bounds()
    obj = np.zeros(inst.n, dtype=float)
    const = 0.0
    rows = feasibility_rows(inst)

    for Zk in inst.Z:
        Nk = Zk.numerator(xbar)          # entier
        Dk = Zk.denominator(xbar)        # entier > 0 par (A1)
        coef = (Dk * Zk.num - Nk * Zk.den).astype(float)
        rhs = float(Nk * Zk.b - Dk * Zk.a)
        rows.append((coef, rhs, INF))    # Z_k(x) >= Z_k(xbar)
        obj += coef
        const += float(Dk * Zk.a - Nk * Zk.b)

    res = solve_ilp(obj, rows, ub, maximize=True, obj_const=const)
    if not res.ok:
        # xbar est toujours realisable pour ce programme : infaisable = bug
        raise RuntimeError(f"Test d'efficacite : statut {res.status}")

    theta = int(round(res.obj))
    if theta == 0:
        return EfficiencyResult(True, 0, None)
    return EfficiencyResult(False, theta, res.x)


# ----------------------------------------------------------------------------
# Theoreme 3 : Dinkelbach exact a parametre rationnel
# ----------------------------------------------------------------------------

@dataclass
class DinkelbachResult:
    q_star: Optional[Fraction]
    x_star: Optional[np.ndarray]
    iterations: int
    status: str                       # 'optimal' | 'infeasible'
    trace: List[Tuple[Fraction, int]]  # (q_t, F(q_t))  -- F(q_t) entier


def dinkelbach(fobj: FracObj,
               rows: Sequence[Tuple[np.ndarray, float, float]],
               var_ub: np.ndarray,
               x0: Optional[np.ndarray] = None,
               max_iter: int = 200) -> DinkelbachResult:
    """
    Maximise fobj sur { x entier, 0 <= x <= var_ub, rows }.

    Utilise q rationnel exact : la valeur optimale du sous-probleme est un
    ENTIER, donc le test d'arret F(q) == 0 est exact (pas d'epsilon).

    Remarque de mise en oeuvre : les coefficients (Q*num - P*den) grossissent
    avec la taille du numerateur/denominateur de q. Sur de grandes instances,
    borner q (Fraction.limit_denominator) ou repasser en flottant avec
    tolerance.
    """
    rows = list(rows)

    # -- point de depart --------------------------------------------------
    if x0 is None:
        r0 = solve_ilp(np.zeros(len(var_ub)), rows, var_ub, maximize=True)
        if not r0.ok:
            return DinkelbachResult(None, None, 0, "infeasible", [])
        x0 = r0.x

    q = fobj.value(x0)
    x_best = x0
    trace: List[Tuple[Fraction, int]] = []

    for it in range(1, max_iter + 1):
        P, Q = q.numerator, q.denominator          # Q > 0
        obj = (Q * fobj.num - P * fobj.den).astype(float)
        const = float(Q * fobj.a - P * fobj.b)

        res = solve_ilp(obj, rows, var_ub, maximize=True, obj_const=const)
        if not res.ok:
            return DinkelbachResult(None, None, it, "infeasible", trace)

        Fq = int(round(res.obj))                   # = Q * (N(x) - q D(x))
        trace.append((q, Fq))

        if Fq <= 0:                                # racine atteinte
            return DinkelbachResult(q, x_best, it, "optimal", trace)

        x_best = res.x
        q = fobj.value(x_best)                     # pas de Newton

    raise RuntimeError("Dinkelbach : max_iter atteint (ne devrait pas arriver "
                       "sur un ensemble fini).")


def max_f_over_S(inst: MOILFP) -> DinkelbachResult:
    """
    Borne superieure initiale q_UB^(0) = max_{x in S} f(x)  (relaxation E -> S).

    Valide car E est inclus dans S, donc max_S f >= max_E f = q*.
    """
    return dinkelbach(inst.f, feasibility_rows(inst), inst.var_upper_bounds())


def ideal_nadir_estimates(inst: MOILFP) -> Tuple[List[Fraction], List[Fraction]]:
    """
    Point ideal exact (max de chaque Z_k sur S) et estimation du nadir
    (min des Z_k sur les p solutions ideales -- borne inferieure du vrai nadir).
    """
    ub = inst.var_upper_bounds()
    rows = feasibility_rows(inst)
    argmax_pts = []
    ideal = []
    for k in range(inst.p):
        r = dinkelbach(inst.Z[k], rows, ub)
        ideal.append(r.q_star)
        argmax_pts.append(r.x_star)
    nadir_est = [min(inst.Z[k].value(x) for x in argmax_pts)
                 for k in range(inst.p)]
    return ideal, nadir_est
