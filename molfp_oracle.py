"""
molfp_oracle.py
===============
Piece manquante de l'hybride exact-exact : maximisation d'une fonction
LINEAIRE sur l'ensemble efficace entier E de (MOILFP), puis assemblage
Dinkelbach + oracle pour resoudre (P).

--------------------------------------------------------------------------
COUPE DE DOMINANCE EXACTE
--------------------------------------------------------------------------
Soit xbar realisable, non efficace. Posons Nbar_k = N_k(xbar),
Dbar_k = D_k(xbar) > 0 et

        e_k(x) = Dbar_k N_k(x) - Nbar_k D_k(x)
               = (Dbar_k c_k - Nbar_k d_k)^T x + (Dbar_k a_k - Nbar_k b_k)

Alors  e_k(x) = Dbar_k * D_k(x) * (Z_k(x) - Z_k(xbar)),  de meme signe que
Z_k(x) - Z_k(xbar) car les denominateurs sont > 0. De plus e_k(x) est un
ENTIER, donc :

        Z_k(x) > Z_k(xbar)   <=>   e_k(x) >= 1

Aucun pas minimal delta n'est a estimer : c'est la version fractionnaire de
la coupe "+1" du cas lineaire entier, et elle est exacte.

Lemme (validite). Si xbar est domine, alors { x in S : Z(x) <= Z(xbar) } ne
contient aucune solution efficace.
  Preuve : soit y dominant xbar et x tel que Z(x) <= Z(xbar). Alors
  Z(y) >= Z(xbar) >= Z(x). Si Z(y) = Z(x) alors Z(xbar) est coince entre les
  deux, donc Z(y) = Z(xbar), ce qui contredit Z(y) != Z(xbar). Donc y domine
  x. []

On peut donc retirer exactement cette region, via la disjonction
"il existe k tel que e_k(x) >= 1", modelisee par p binaires u_k et un
big-M valide calcule sur la boite :

        M_k = 1 - min_{0<=x<=ub} e_k(x)     (borne atteinte coin par coin)

        e_k(x) >= 1 - M_k (1 - u_k),   sum_k u_k >= 1

En xbar on a e_k(xbar) = 0 pour tout k : xbar est bien exclu.

--------------------------------------------------------------------------
ALGORITHME (oracle lineaire sur E)
--------------------------------------------------------------------------
    R <- S
    repeter
        xbar <- argmax g sur R                 -> UB = g(xbar)
        si xbar efficace : xbar est optimal sur E, stop
        sinon :
            reparer (suivre la chaine de dominance) -> point efficace -> LB
            ajouter la coupe de dominance de xbar
    jusqu'a UB - LB <= tol  ou  R vide

Correction. Invariant E inclus dans R (lemme ci-dessus). A l'arret avec xbar
efficace : xbar in E inclus dans R et g(xbar) = max_R g >= max_E g, donc xbar
est optimal sur E. Terminaison : chaque coupe retire au moins xbar et S est
fini.

Comportement ANYTIME : a chaque iteration on dispose de
    LB = meilleur g sur les points efficaces certifies deja trouves
    UB = g(xbar) = max_R g  >=  max_E g
donc d'un ecart d'optimalite garanti a tout instant. C'est precisement ce
qui manque a la litterature exacte MOILFP.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

import numpy as np

from molfp_core import (INF, ORACLE_CALLS, efficiency_test, feasibility_rows,
                        min_over_relaxation, solve_milp)
from molfp_instance import MOILFP

Row = Tuple[np.ndarray, float, float]

# Renforcement du big-M des coupes par relaxation continue (cf. ECutModel).
# Drapeau de module pour que les etudes d'ablation puissent le desactiver
# globalement sans avoir a passer l'option de main en main.
TIGHT_BIG_M = True


# ----------------------------------------------------------------------------
# Modele a coupes
# ----------------------------------------------------------------------------

class ECutModel:
    """Relaxation R de E, enrichie de coupes de dominance exactes."""

    def __init__(self, inst: MOILFP, tight_big_m: Optional[bool] = None):
        self.inst = inst
        self.n = inst.n
        self.p = inst.p
        self.ub_x = inst.var_upper_bounds()
        self._rows_x: List[Row] = feasibility_rows(inst)   # espace x
        self._cut_rows: List[Row] = []                     # espace etendu
        self.n_cuts = 0
        self.tight_big_m = TIGHT_BIG_M if tight_big_m is None else tight_big_m
        # diagnostic : big-M boite vs big-M relaxation continue, pour mesurer
        # le resserrement au lieu de le postuler
        self.big_m_box: List[float] = []
        self.big_m_used: List[float] = []

    # -- dimensions --------------------------------------------------------
    @property
    def nvar(self) -> int:
        return self.n + self.p * self.n_cuts

    def _bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        lb = np.zeros(self.nvar)
        ub = np.concatenate([self.ub_x.astype(float),
                             np.ones(self.p * self.n_cuts)])
        return lb, ub

    def _pad(self, row: Row) -> Row:
        coef, lo, hi = row
        out = np.zeros(self.nvar)
        out[:len(coef)] = coef
        return (out, lo, hi)

    # -- ajout d'une coupe -------------------------------------------------
    def add_dominance_cut(self, xbar: np.ndarray) -> None:
        """Retire exactement { x : Z(x) <= Z(xbar) }. xbar doit etre domine."""
        inst = self.inst
        t = self.n_cuts
        base = self.n + self.p * t          # indice de u_{t,0}
        self.n_cuts += 1
        nvar = self.nvar

        # les coupes deja posees doivent etre re-elargies au nouvel espace
        self._cut_rows = [self._pad_to(r, nvar) for r in self._cut_rows]

        sum_row = np.zeros(nvar)
        for k in range(self.p):
            Zk = inst.Z[k]
            Nbar = Zk.numerator(xbar)
            Dbar = Zk.denominator(xbar)
            a = (Dbar * Zk.num - Nbar * Zk.den).astype(float)   # entiers
            b = float(Dbar * Zk.a - Nbar * Zk.b)

            # big-M valide : M_k >= 1 - min_{x in S} e_k(x).
            # La boite seule ignore A x <= b et donne un M tres lache ; le
            # minorant par relaxation continue est valide (la relaxation
            # contient S) et plus fin (elle est contenue dans la boite).
            e_min_box = float(np.sum(np.minimum(a, 0.0) * self.ub_x)) + b
            e_min = e_min_box
            if self.tight_big_m:
                e_min_lp = min_over_relaxation(a, b, self._rows_x, self.ub_x)
                if np.isfinite(e_min_lp):
                    e_min = max(e_min_box, e_min_lp)
            M = max(1.0, 1.0 - e_min)
            self.big_m_box.append(max(1.0, 1.0 - e_min_box))
            self.big_m_used.append(M)

            #  a^T x - M u_k >= 1 - M - b
            row = np.zeros(nvar)
            row[:self.n] = a
            row[base + k] = -M
            self._cut_rows.append((row, 1.0 - M - b, INF))

            sum_row[base + k] = 1.0

        self._cut_rows.append((sum_row, 1.0, INF))   # sum_k u_k >= 1

    @staticmethod
    def _pad_to(row: Row, nvar: int) -> Row:
        coef, lo, hi = row
        if len(coef) == nvar:
            return row
        out = np.zeros(nvar)
        out[:len(coef)] = coef
        return (out, lo, hi)

    # -- resolution --------------------------------------------------------
    def optimize(self, g_coef: np.ndarray, g_const: float = 0.0,
                 maximize: bool = True, extra_rows=None):
        """
        Optimise g sur R (= S prive des zones coupees), avec d'eventuelles
        contraintes supplementaires en espace x. Renvoie un ILPResult dont
        res.x est en espace etendu.
        """
        nvar = self.nvar
        obj = np.zeros(nvar)
        obj[:self.n] = g_coef
        rows = [self._pad(r) for r in self._rows_x] + \
               [self._pad_to(r, nvar) for r in self._cut_rows]
        for r in (extra_rows or []):
            rows.append(self._pad_to(r, nvar))
        lb, ub = self._bounds()
        return solve_milp(obj, rows, lb, ub, maximize=maximize,
                          obj_const=g_const)

    def maximize_linear(self, g_coef: np.ndarray, g_const: float = 0.0):
        """max g sur R. Renvoie ILPResult ; res.x est en espace etendu."""
        return self.optimize(g_coef, g_const, maximize=True)


# ----------------------------------------------------------------------------
# Reparation par chaine de dominance
# ----------------------------------------------------------------------------

def repair_to_efficient(inst: MOILFP, x: np.ndarray,
                        max_steps: int = 100,
                        dominated_out: Optional[list] = None) -> np.ndarray:
    """
    Suit la chaine de dominance jusqu'a atteindre un point efficace certifie.

    Termine : chaque pas ameliore strictement le vecteur critere au sens de
    Pareto, S est fini, donc pas de cycle.

    `dominated_out` : si une liste est fournie, tous les points DOMINES
    traverses y sont ajoutes. Ils sont exactement les points sur lesquels le
    Th. 4 autorise une coupe de dominance ; les jeter serait gaspiller une
    information deja payee.
    """
    cur = x
    for _ in range(max_steps):
        r = efficiency_test(inst, cur)
        if r.efficient:
            return cur
        if dominated_out is not None:
            dominated_out.append(np.array(cur, dtype=int))
        cur = r.dominator
    raise RuntimeError("Chaine de dominance trop longue (bug probable).")


# ----------------------------------------------------------------------------
# Oracle : max d'une fonction lineaire sur E
# ----------------------------------------------------------------------------

@dataclass
class OracleResult:
    status: str                    # 'optimal' | 'gap' | 'empty' | 'limit'
    x_star: Optional[np.ndarray]
    value: Optional[float]         # meilleure valeur certifiee (LB)
    ub: Optional[float]            # borne superieure valide
    iterations: int
    n_cuts: int
    ilp_calls: int
    time: float
    incumbents: List[np.ndarray] = field(default_factory=list)

    @property
    def gap(self) -> Optional[float]:
        if self.value is None or self.ub is None:
            return None
        denom = max(1e-12, abs(self.ub))
        return (self.ub - self.value) / denom


def max_linear_over_E(inst: MOILFP,
                      g_coef: np.ndarray,
                      g_const: float = 0.0,
                      tol: float = 0.0,
                      max_iter: int = 2000,
                      time_limit: float = 300.0,
                      model: Optional[ECutModel] = None,
                      collect: bool = True,
                      verbose: bool = False) -> OracleResult:
    """
    Maximise  g(x) = g_coef^T x + g_const  sur l'ensemble efficace E.

    tol : ecart relatif accepte. tol = 0 -> resolution exacte.
    model : un ECutModel deja garni de coupes peut etre passe pour etre
            reutilise d'un appel a l'autre (crucial dans Dinkelbach : les
            coupes restent valides puisqu'elles ne dependent que de E).
    """
    t0 = time.time()
    calls0 = ORACLE_CALLS["ilp"]
    R = model if model is not None else ECutModel(inst)

    LB, x_best, UB = -np.inf, None, np.inf
    incumbents: List[np.ndarray] = []

    def g_of(x: np.ndarray) -> float:
        return float(g_coef @ x) + g_const

    for it in range(1, max_iter + 1):
        if time.time() - t0 > time_limit:
            return OracleResult("limit", x_best, LB if x_best is not None else None,
                                UB, it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)

        res = R.maximize_linear(g_coef, g_const)
        if not res.ok:
            # R vide : plus aucun candidat non coupe
            status = "optimal" if x_best is not None else "empty"
            return OracleResult(status, x_best,
                                LB if x_best is not None else None,
                                LB if x_best is not None else None,
                                it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)

        xbar = res.x[:inst.n]
        UB = res.obj

        eff = efficiency_test(inst, xbar)
        if eff.efficient:
            # optimal : xbar maximise g sur R qui contient E
            if collect:
                incumbents.append(xbar)
            return OracleResult("optimal", xbar, g_of(xbar), UB, it, R.n_cuts,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)

        # xbar domine : on en tire quand meme un point efficace (borne inf)
        x_eff = repair_to_efficient(inst, eff.dominator)
        if collect:
            incumbents.append(x_eff)
        if g_of(x_eff) > LB:
            LB, x_best = g_of(x_eff), x_eff

        if verbose:
            print(f"   it {it:>3}  UB={UB:>12.4f}  LB={LB:>12.4f}  "
                  f"coupes={R.n_cuts}")

        if x_best is not None:
            gap = (UB - LB) / max(1e-12, abs(UB))
            # gap nul : la borne superieure rejoint l'incumbent, donc x_best
            # est PROUVE optimal -- sans que xbar ait eu besoin d'etre efficace.
            # C'est le gain propre au schema anytime : l'arret peut survenir
            # bien avant que la relaxation ne produise un point efficace.
            if gap <= 1e-9:
                return OracleResult("optimal", x_best, LB, UB, it, R.n_cuts,
                                    ORACLE_CALLS["ilp"] - calls0,
                                    time.time() - t0, incumbents)
            if gap <= tol:
                return OracleResult("gap", x_best, LB, UB, it, R.n_cuts,
                                    ORACLE_CALLS["ilp"] - calls0,
                                    time.time() - t0, incumbents)

        R.add_dominance_cut(xbar)

    return OracleResult("limit", x_best, LB if x_best is not None else None,
                        UB, max_iter, R.n_cuts,
                        ORACLE_CALLS["ilp"] - calls0, time.time() - t0,
                        incumbents)


# ----------------------------------------------------------------------------
# Hybride exact-exact : Dinkelbach + oracle lineaire sur E
# ----------------------------------------------------------------------------

@dataclass
class HybridResult:
    status: str
    q_star: Optional[Fraction]
    x_star: Optional[np.ndarray]
    outer_iterations: int          # iterations de Dinkelbach
    total_cuts: int
    ilp_calls: int
    time: float
    trace: List[Tuple[Fraction, float]] = field(default_factory=list)
    archive: List[np.ndarray] = field(default_factory=list)


def solve_P(inst: MOILFP,
            max_outer: int = 60,
            time_limit: float = 600.0,
            reuse_cuts: bool = True,
            verbose: bool = False) -> HybridResult:
    """
    Resout  (P)  max f(x) s.c. x in E  par l'hybride exact-exact.

    Boucle externe : Dinkelbach a parametre rationnel (Th. 3).
    Boucle interne : max_linear_over_E (oracle exact ci-dessus).

    reuse_cuts : les coupes de dominance ne dependent que de E, pas de la
    fonction objectif. Les conserver d'une iteration Dinkelbach a l'autre
    evite de reconstruire la relaxation a chaque fois -- c'est le point
    d'hybridation qui rend l'ensemble economique.
    """
    t0 = time.time()
    calls0 = ORACLE_CALLS["ilp"]
    f = inst.f
    R = ECutModel(inst) if reuse_cuts else None

    # amorcage : un point efficace quelconque
    first = max_linear_over_E(inst, np.zeros(inst.n), 0.0,
                              model=R, time_limit=time_limit)
    if first.x_star is None and not first.incumbents:
        return HybridResult("empty", None, None, 0, 0,
                            ORACLE_CALLS["ilp"] - calls0, time.time() - t0)
    x_cur = first.x_star if first.x_star is not None else first.incumbents[0]

    q = f.value(x_cur)
    archive = list(first.incumbents)
    trace: List[Tuple[Fraction, float]] = []

    for it in range(1, max_outer + 1):
        if time.time() - t0 > time_limit:
            return HybridResult("limit", q, x_cur, it,
                                R.n_cuts if R else 0,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, trace, archive)

        P_, Q_ = q.numerator, q.denominator
        coef = (Q_ * f.num - P_ * f.den).astype(float)
        const = float(Q_ * f.a - P_ * f.b)

        r = max_linear_over_E(inst, coef, const, model=R,
                              time_limit=time_limit - (time.time() - t0))
        archive.extend(r.incumbents)

        if r.x_star is None:
            return HybridResult("empty", q, x_cur, it,
                                R.n_cuts if R else 0,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, trace, archive)

        Fq = r.value                      # borne INFERIEURE de F(q) en general
        trace.append((q, Fq))
        if verbose:
            print(f" [Dinkelbach {it}] q = {float(q):.6f}   F(q) = {Fq:.4f}   "
                  f"coupes = {r.n_cuts}   [{r.status}]")

        # ------------------------------------------------------------------
        # CORRECTION DE SURETE.
        # r.value n'est la valeur exacte de F(q) que si l'oracle a PROUVE
        # l'optimalite. Sinon ce n'est qu'une borne inferieure, et le test
        # "F(q) <= 0" ne prouve rien : c'est ainsi que l'on concluait a tort
        # a l'optimalite quand l'oracle atteignait sa limite de temps.
        #
        #   Fq > 0  : valide quel que soit le statut, car F(q) >= Fq > 0
        #             donc la racine n'est pas atteinte -> on continue.
        #   Fq <= 0 : conclusion valide UNIQUEMENT si r.status == 'optimal'.
        # ------------------------------------------------------------------
        if Fq > 1e-9:
            x_cur = r.x_star
            q = f.value(x_cur)
            continue

        if r.status == "optimal":         # racine atteinte et prouvee
            return HybridResult("optimal", q, x_cur, it,
                                R.n_cuts if R else 0,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, trace, archive)

        # borne non prouvee : on ne peut rien conclure
        return HybridResult("limit", q, x_cur, it,
                            R.n_cuts if R else 0,
                            ORACLE_CALLS["ilp"] - calls0,
                            time.time() - t0, trace, archive)

    return HybridResult("limit", q, x_cur, max_outer,
                        R.n_cuts if R else 0,
                        ORACLE_CALLS["ilp"] - calls0, time.time() - t0,
                        trace, archive)


def dedup_archive(archive: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Retire les doublons de l'archive de solutions efficaces certifiees."""
    seen, out = set(), []
    for x in archive:
        k = tuple(int(v) for v in x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out
