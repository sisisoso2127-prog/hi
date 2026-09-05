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

def rank_dominated(dominated: Sequence[np.ndarray],
                   w: np.ndarray) -> List[np.ndarray]:
    """
    Points DOMINES uniques, tries par valeur decroissante du substitut w.

    Ce sont exactement les points sur lesquels le Th. 4 autorise une coupe, et
    ils ont deja ete payes par les chaines de reparation de la phase de
    recherche. L'ordre compte : ceux de plus grande valeur du substitut sont
    ceux qui tirent U vers le haut, donc les couper est ce qui resserre le
    plus la borne du Th. 5'.
    """
    seen, uniq = set(), []
    for x in dominated:
        key = tuple(int(v) for v in x)
        if key not in seen:
            seen.add(key)
            uniq.append(np.asarray(x, dtype=int))
    wv = np.asarray(w, dtype=float)
    uniq.sort(key=lambda x: -float(wv @ x))
    return uniq


@dataclass
class CertResult:
    """Sortie de la phase de certification."""
    q_ub: Optional[float] = None        # meilleure borne sup VALIDE sur q*
    proved: bool = False
    q_lb: Optional[Fraction] = None     # incumbent, eventuellement AMELIORE
    x_best: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)


def certify(inst: MOILFP, q: Fraction, x_cur: np.ndarray,
            dominated: Sequence[np.ndarray],
            budget: float,
            use_tightened: bool = True,
            archive: Optional["Archive"] = None,
            cut_batch: int = 10,
            max_rounds: int = 6) -> CertResult:
    """
    Convertit un budget de calcul en borne superieure VALIDE sur q*.

    Deux differences avec la version a plafond de coupes fige :

    1. PLAFOND DE COUPES ADAPTATIF. Les coupes recyclees sont ajoutees par
       lots (`cut_batch`), et un lot n'est ajoute que si le budget reste et
       que la borne n'a pas ferme. Un plafond fixe se trompe des deux cotes :
       trop bas il laisse U inutilement grand, trop haut il alourdit chaque
       ILP (p binaires par coupe) au point que l'oracle n'a plus le temps de
       conclure. Le budget arbitre a la place du reglage.

    2. LA CERTIFICATION AMELIORE AUSSI L'INCUMBENT. L'oracle interne renvoie
       des points efficaces certifies ; s'ils battent q, c'est un vrai pas de
       Dinkelbach (Th. 3) offert par la phase de certification. On relance
       alors sur le nouveau substitut : le LB monte pendant que l'UB descend.

    VALIDITE. Chaque tour produit une borne valide sur q* pour le q de ce
    tour ; comme q* ne depend pas de q, le MINIMUM des bornes obtenues reste
    une borne valide. Un q ameliore ne peut donc pas invalider une borne
    posee plus tot.
    """
    t0 = time.time()
    info: dict = {"n_cuts": 0, "U": None, "Dmin": None, "Dplus": None,
                  "rounds": 0, "q_improved": False}

    best_ub: Optional[float] = None
    proved = False
    Dm: Optional[int] = None

    model = ECutModel(inst)
    pending: List[np.ndarray] = []
    if use_tightened:
        w0_coef, _, _, _ = surrogate(inst, q)
        pending = rank_dominated(dominated, w0_coef)

    for rnd in range(1, max_rounds + 1):
        left = budget - (time.time() - t0)
        if left <= 0.05:
            break
        info["rounds"] = rnd

        # q_ref : le q qui engendre le substitut de ce tour. La borne du
        # Th. 5' se lit sur CE q -- si la certification ameliore l'incumbent
        # en cours de tour, lire la borne sur le nouveau q resterait valide
        # mais la relacherait pour rien.
        q_ref = q
        w, w0, P, Q = surrogate(inst, q_ref)

        # --- plafond adaptatif : un lot de coupes de plus a chaque tour ----
        if pending:
            for x in pending[:cut_batch]:
                model.add_dominance_cut(x)
            pending = pending[cut_batch:]
        info["n_cuts"] = model.n_cuts

        # le dernier tour recoit tout le reste : inutile de garder du budget
        # pour un tour qu'on ne fera pas
        slice_ = left if (rnd == max_rounds or not pending) else left / 2.0
        r = max_linear_over_E(inst, w, w0, time_limit=slice_, model=model,
                              collect=True)

        # -- l'oracle a-t-il produit un meilleur point efficace ? -----------
        improved = False
        for y in r.incumbents:
            if archive is not None:
                archive.add(y)
            fy = inst.f.value(y)
            if fy > q:
                q, x_cur, improved = fy, y, True
        if improved:
            info["q_improved"] = True

        # -- optimalite prouvee : F(q) resolu et <= 0 ----------------------
        if r.status == "optimal" and r.value is not None and r.value <= 1e-9 \
                and not improved:
            return CertResult(float(q), True, q, x_cur, info)

        # -- borne du Th. 5 / 5' -------------------------------------------
        # une UB infinie (oracle interrompu avant sa premiere relaxation)
        # n'est pas une borne : mieux vaut ne rien annoncer qu'annoncer inf.
        if r.ub is not None and np.isfinite(r.ub):
            U = max(0.0, float(r.ub))
            info["U"] = U
            if U <= 1e-9 and not improved:
                return CertResult(float(q_ref), True, q, x_cur, info)

            if Dm is None:
                Dm = d_min(inst)
                info["Dmin"] = Dm
            denom = Dm
            if use_tightened:
                Dp = d_plus(inst, model, w, w0)
                info["Dplus"] = Dp
                if Dp is not None:
                    denom = max(Dm, Dp)      # D+ >= Dmin par construction
            cand = float(q_ref) + U / (Q * denom)
            best_ub = cand if best_ub is None else min(best_ub, cand)

        # Un tour sans amelioration, sans coupe en reserve ET dont l'oracle
        # a conclu se repeterait a l'identique : on s'arrete.
        # En revanche un oracle INTERROMPU ('limit') laisse du travail : il a
        # pose ses propres coupes dans `model`, donc R s'est resserre et le
        # tour suivant repart d'une relaxation strictement meilleure.
        if not improved and not pending and r.status != "limit":
            break

    if best_ub is not None and best_ub <= float(q) + 1e-12:
        proved = True
        best_ub = float(q)
    return CertResult(best_ub, proved, q, x_cur, info)


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

def _select_pool(arch: "Archive", usage: Dict[Tuple[int, ...], int],
                 rng: np.random.Generator, inst: MOILFP,
                 diversify: bool, k_elite: int = 8, k_rand: int = 4
                 ) -> List[np.ndarray]:
    """
    Choisit les points de depart du tour.

    Regime normal : intensification autour des meilleurs points pour f.
    Regime DIVERSIFICATION (declenche par la stagnation) : on repart des
    points de l'archive les MOINS souvent utilises comme base. Un redemarrage
    aleatoire jetterait le travail deja fait ; l'archive, elle, ne contient
    que des points efficaces certifies, donc des bases legitimes et
    gratuites. C'est ce qui remplace le redemarrage aveugle.
    """
    pool = arch.points()
    if not pool:
        return []
    if diversify:
        pool.sort(key=lambda x: (usage.get(tuple(int(v) for v in x), 0),
                                 -float(inst.f.value(x))))
    else:
        pool.sort(key=lambda x: -float(inst.f.value(x)))
    base = pool[:k_elite]
    if len(pool) > k_elite:
        idx = rng.choice(len(pool), size=min(k_rand, len(pool)), replace=False)
        base += [pool[i] for i in idx]
    return base


def matheuristic_P(inst: MOILFP,
                   time_budget: float = 20.0,
                   bound_budget: float = 10.0,
                   seed: int = 0,
                   certify_bound: bool = True,
                   tightened: bool = True,
                   max_stall: int = 3,
                   reallocate: bool = True,
                   cut_batch: int = 10,
                   verbose: bool = False) -> MatheurResult:
    """
    Phase 1 (recherche) : VNS dans l'espace des criteres, sous-problemes
                          exacts sur chaque voisinage. L'incumbent est
                          toujours certifie efficace (Th. 2), donc le LB
                          n'est JAMAIS optimiste, meme interrompu.
    Phase 2 (certification) : oracle exact a budget borne, dont la borne
                          superieure U sur F(q) devient une borne sur q*
                          par le Th. 5'.

    Trois reglages remplacent des constantes qui etaient figees :

    `max_stall`  la recherche ne s'arrete plus au premier tour sans
                 amelioration ; elle redemarre depuis les points les moins
                 exploites de l'archive, et n'abandonne qu'apres `max_stall`
                 tours steriles consecutifs.
    `reallocate` le budget que la recherche n'a pas consomme (arret sur
                 stagnation) est REVERSE a la certification au lieu d'etre
                 perdu. C'est du temps deja alloue, et la certification est
                 precisement ce qui manquait de budget.
    `cut_batch`  taille des lots de coupes recyclees ; le plafond n'est plus
                 fixe, il est arbitre par le budget (cf. `certify`).
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
    stall = 0
    n_restarts = 0
    usage: Dict[Tuple[int, ...], int] = {}
    n_moves = {"A": 0, "B": 0, "C": 0}
    n_hits = {"A": 0, "B": 0, "C": 0}

    # --- phase 1 : recherche dans l'espace des criteres --------------------
    while time.time() - t0 < time_budget:
        rounds += 1
        w, w0, _, _ = surrogate(inst, q)
        improved = False
        diversify = stall > 0
        if diversify:
            n_restarts += 1

        # en diversification on tire davantage vers B (plancher absolu) et C
        # (LNS) : A reste ancre sur le point de base, donc explore peu
        menu = ["B", "B", "C", "A"] if diversify else ["A", "A", "B", "C"]

        for xr in _select_pool(arch, usage, rng, inst, diversify):
            usage[tuple(int(v) for v in xr)] = \
                usage.get(tuple(int(v) for v in xr), 0) + 1
            for k in rng.permutation(inst.p):
                if time.time() - t0 >= time_budget:
                    break
                mv = str(rng.choice(menu))
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
                    keep_prob = 0.4 if diversify else 0.6
                    eps = [eps[j] if rng.random() < keep_prob else None
                           for j in range(inst.p)]
                    y = move_epsilon_absolute(inst, eps, w, w0)
                else:
                    frac = 0.6 if diversify else 0.4
                    n_free = max(1, int(frac * inst.n))
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

        stall = 0 if improved else stall + 1

        if verbose:
            print(f"  tour {rounds}: q = {float(q):.6f}  |archive| = {len(arch)}"
                  f"  stagnation {stall}  mouvements {n_moves} succes {n_hits}"
                  f"  ({time.time()-t0:.1f}s)")

        if stall >= max_stall:
            break            # optimum local confirme sur plusieurs tours

    search_time = time.time() - t0

    # --- phase 2 : certification (Th. 5') --------------------------------
    # le budget de recherche non consomme est reverse ici : la recherche a
    # conclu, la certification non.
    budget = bound_budget
    if reallocate:
        budget += max(0.0, time_budget - search_time)

    q_ub, proved, cert_info = None, False, {}
    if certify_bound and budget > 0:
        c = certify(inst, q, x_best, dominated, budget,
                    use_tightened=tightened, archive=arch,
                    cut_batch=cut_batch)
        q_ub, proved, cert_info = c.q_ub, c.proved, c.info
        if c.q_lb is not None and c.q_lb > q:
            q, x_best = c.q_lb, c.x_best      # pas de Dinkelbach offert
    cert_info["search_time"] = search_time
    cert_info["cert_budget"] = budget
    cert_info["restarts"] = n_restarts
    cert_info["moves"] = dict(n_moves)
    cert_info["hits"] = dict(n_hits)

    return MatheurResult(
        q_lb=q, x_best=x_best, q_ub=q_ub, archive=arch.points(),
        ilp_calls=ORACLE_CALLS["ilp"] - calls0, time=time.time() - t0,
        rounds=rounds, proved_optimal=proved,
        status="optimal" if proved else "heuristic", cert=cert_info,
    )
