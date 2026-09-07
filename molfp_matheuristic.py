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
                        max_over_relaxation, solve_ilp)
from molfp_instance import MOILFP
from molfp_oracle import (ECutModel, HybridResult, max_linear_over_E,
                          repair_to_efficient)

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

def same_criteria_improves(inst: MOILFP, a: np.ndarray,
                           q: Fraction) -> Optional[np.ndarray]:
    """
    UN SEUL ILP. Existe-t-il x de MEME vecteur criteres que `a` avec
    f(x) > q ?

        trouver x dans S  tel que  e_k^a(x) = 0 pour tout k
                                   Q N(x) - P D(x) >= 1

    Les e_k etant entieres, `e_k^a(x) = 0` equivaut a `Z_k(x) = Z_k(a)`, et
    `Q N - P D >= 1` a `f(x) > q` (le membre de gauche est entier).

    Deux usages, et c'est ce qui rend l'appel economique :

      * INFAISABLE -> aucun point de meme vecteur criteres que `a` ne bat
        l'incumbent : la condition de cloture du Th. 7 est etablie pour `a`,
        et l'on peut couper en `a` sans risque ;
      * FAISABLE -> le point rendu est efficace (meme vecteur criteres qu'un
        point efficace) ET meilleur que l'incumbent : on l'empoche.

    Poser la question comme une FAISABILITE plutot que comme un maximum
    evite un Dinkelbach complet par point d'archive.
    """
    P, Q = q.numerator, q.denominator
    rows = feasibility_rows(inst)
    for k in range(inst.p):
        coef, cst = e_row(inst, a, k)
        rows.append((coef, -cst, -cst))            # e_k^a(x) = 0
    w = (Q * inst.f.num - P * inst.f.den).astype(float)
    w0 = float(Q * inst.f.a - P * inst.f.b)
    rows.append((w, 1.0 - w0, INF))                # f(x) > q
    res = solve_ilp(np.zeros(inst.n), rows, inst.var_upper_bounds(),
                    maximize=True)
    return None if not res.ok else res.x


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


def build_cut_pool(dominated: Sequence[np.ndarray],
                   archive: Sequence[np.ndarray],
                   w: np.ndarray) -> List[tuple]:
    """
    UN SEUL vivier de coupes, alimente par DEUX sources.

    Les deux coupes ont exactement la meme forme -- la disjonction
    "il existe k tel que e_k(x) >= 1" -- donc le meme cout : p binaires
    chacune. Seule leur precondition differe (Th. 4 pour un point domine,
    Th. 6 pour un point efficace). Il n'y a donc aucune raison de leur
    allouer des budgets separes, et une bonne raison de ne pas le faire : la
    mesure du plafond de coupes montre qu'au-dela d'une quarantaine, chaque
    coupe supplementaire coute plus qu'elle ne rapporte. Poser 40 coupes
    d'archive EN PLUS des 40 de dominance a d'ailleurs fait chuter
    l'optimalite prouvee de 84 a 57 sur 90 instances -- exactement l'effet de
    saturation deja documente.

    On classe donc les candidats des deux sources ensemble, par valeur
    decroissante du substitut : ce sont ceux qui tirent U vers le haut, quelle
    que soit leur origine. Le plafond global reste celui qui a ete regle.

    Cette mise en commun a un effet automatique et souhaitable : la ou les
    points domines abondent, ils occupent le vivier ; la ou ils sont rares --
    le regime a E epais, ou la profondeur des chaines de reparation vaut 1 --
    l'archive le remplit a leur place.

    Renvoie une liste de couples (point, origine) avec origine dans
    {"dom", "arch"}.
    """
    wv = np.asarray(w, dtype=float)
    seen, pool = set(), []
    for pts, kind in ((dominated, "dom"), (archive or [], "arch")):
        for x in pts:
            key = (kind, tuple(int(v) for v in x))
            if key in seen:
                continue
            seen.add(key)
            pool.append((np.asarray(x, dtype=int), kind))
    pool.sort(key=lambda t: -float(wv @ t[0]))
    return pool


def region_height(inst: MOILFP, a: np.ndarray, q: Fraction) -> float:
    """
    Hauteur de la REGION que retranche une coupe posee en `a` :

        h(a) = max { Q N(x) - P D(x)  :  x in S,  Z(x) <= Z(a) }

    majoree sur la relaxation continue (donc valide), et ramenee au plancher
    entier puisque Q N - P D est entiere.

    Pourquoi cette quantite, et pas w^T a comme jusqu'ici. Le classement du
    vivier par valeur decroissante du substitut se justifiait pour les points
    DOMINES : un tel point est lui-meme dans R et tire U vers le haut, donc le
    couper fait mecaniquement baisser U. Ce raisonnement ne vaut PAS pour un
    point d'archive : `a` est efficace, donc f(a) <= q, donc son substitut est
    <= 0 par construction -- w^T a le classe systematiquement en queue, alors
    que la valeur d'une coupe d'efficacite ne tient pas au point mais a la
    REGION qu'il domine. Sur l'instance I2 de l'illustration, la coupe la plus
    profitable de toute l'instance est posee en un point d'archive de
    substitut exactement nul.

    Deux usages, tous deux exacts :

      * h(a) <= 0 CERTIFIE que la region ne contient aucun point entier
        battant q. La couper ne peut ni faire baisser U (un maximum ne bouge
        pas quand on retire des points sous son niveau) ni faire monter D+
        (dont le minimum ne porte que sur la zone active). La coupe est donc
        SANS EFFET sur la borne : la poser gaspille un ILP de cloture et une
        place sous le plafond.
      * sinon h(a) classe les candidats des deux sources sur la meme echelle.

    Cout : un seul PL, sur S (et non sur R) -- la region ne depend donc pas
    des coupes deja posees, et h reste un majorant valide puisque R est
    inclus dans S.
    """
    P, Q = q.numerator, q.denominator
    rows = feasibility_rows(inst)
    for k in range(inst.p):
        coef, cst = e_row(inst, a, k)
        rows.append((coef, -INF, -cst))            # e_k^a(x) <= 0
    w = (Q * inst.f.num - P * inst.f.den).astype(float)
    w0 = float(Q * inst.f.a - P * inst.f.b)
    return max_over_relaxation(w, w0, rows, inst.var_upper_bounds())


def select_cuts(inst: MOILFP,
                dominated: Sequence[np.ndarray],
                archive: Sequence[np.ndarray],
                q: Fraction,
                cap: int,
                max_lp: Optional[int] = None) -> Tuple[List[tuple], dict]:
    """
    Choisit et ordonne les coupes a poser, les deux sources sur la MEME
    echelle : la hauteur de region `region_height`.

    Renvoie des TRIPLETS (point, origine, hauteur). La hauteur n'est pas
    qu'une cle de tri : elle dispense parfois de l'ILP de cloture, par le

    LEMME (cloture par la hauteur). Si h(a) <= 0, la condition de cloture
    est etablie en `a`.
      Preuve. { x : Z(x) = Z(a) } est inclus dans { x : Z(x) <= Z(a) }. Si
      le maximum de Q N - P D sur le second est <= 0, il l'est sur le
      premier, c'est-a-dire f(x) <= q pour tout x de meme vecteur criteres
      que a. []
    Comme h est majoree sur la relaxation continue, un h <= 0 calcule par PL
    suffit. UN PL remplace donc UN ILP -- et c'etait la, exactement, le
    surcout mesure : sur les instances a E epais dont le vivier tient sous
    le plafond, la lecture appariee relevait +21 a +27 appels au solveur
    entier, pour 21 a 26 coupes d'archive posees. Un ILP de cloture chacune.

    Deux regimes, et la condition qui les separe est la TAILLE DU VIVIER :

      * vivier plus petit que le plafond -> on garde TOUT, y compris les
        candidats de hauteur nulle. Couper l'archive entiere est a portee,
        et c'est la seule facon de VIDER le relache, donc d'obtenir la
        preuve de la proposition du relache vide, qu'aucune borne ne
        remplace. Ces coupes-la ne coutent plus d'ILP, par le lemme.
      * vivier plus grand -> vider le relache est hors de portee de toute
        facon (l'archive ne couvre pas un E epais) et chaque place compte :
        on ne retient que les candidats de hauteur strictement positive.

    NOTE. Les hauteurs sont calculees une fois, pour le q d'entree. Si la
    certification ameliore l'incumbent, elles restent des majorants -- donc
    le lemme reste valide -- mais moins fins.

    Renvoie (vivier ordonne, diagnostic).
    """
    w, w0, _, _ = surrogate(inst, q)
    pool = build_cut_pool(rank_dominated(dominated, w), archive or [], w)
    serre = len(pool) > cap

    if not serre and not any(k == "arch" for _, k in pool):
        # rien a arbitrer et aucune cloture a etablir : pas un seul PL.
        return ([(x, k, None) for x, k in pool],
                {"pool": len(pool), "evalues": 0, "utiles": None,
                 "vains": None, "vains_arch": 0, "filtre_actif": False,
                 "cloture_gratuite": 0})

    budget_lp = max_lp if max_lp is not None else max(2 * cap, 40)
    scored = [(x, k, region_height(inst, x, q)) for x, k in pool[:budget_lp]]
    reste = [(x, k, None) for x, k in pool[budget_lp:]]

    utiles = sorted([t for t in scored if t[2] > 0], key=lambda t: -t[2])
    vains = sorted([t for t in scored if t[2] <= 0], key=lambda t: -t[2])

    retenus = utiles if serre else utiles + vains + reste
    info = {"pool": len(pool), "evalues": len(scored),
            "utiles": len(utiles), "vains": len(vains),
            "vains_arch": sum(1 for t in vains if t[1] == "arch"),
            "filtre_actif": serre,
            # coupes d'archive dont la cloture est acquise par le lemme,
            # donc posees SANS ILP
            "cloture_gratuite": sum(1 for t in retenus
                                    if t[1] == "arch" and t[2] is not None
                                    and t[2] <= 0)}
    return retenus, info


def diversify_over_archive(inst: MOILFP,
                           archive: "Archive",
                           q: Fraction,
                           x_best: np.ndarray,
                           budget: float,
                           cap: int = 40,
                           dominated_out: Optional[List[np.ndarray]] = None
                           ) -> Tuple[Fraction, np.ndarray, int]:
    """
    La coupe d'efficacite comme OPERATEUR DE RECHERCHE, et non comme outil de
    borne.

    L'idee. Poser une coupe d'efficacite autour de chaque point de l'archive
    retire { x : Z(x) <= Z(a) } pour tout a deja trouve. Ce qui reste est
    donc, par construction, l'ensemble des points que l'archive ne domine ni
    n'egale. Maximiser le substitut sur ce reste rend un point GARANTI NEUF
    en vecteur criteres -- ce qu'aucun redemarrage aleatoire ne garantit.
    On le repare en un point efficace certifie, on l'archive, on coupe autour
    de lui, et on recommence.

    POINT ESSENTIEL : ici la condition de cloture n'est PAS requise. Elle
    n'est necessaire que pour tirer une BORNE d'un relache ampute. Employee
    comme operateur de recherche, la coupe ne sert qu'a diriger l'exploration
    vers du neuf ; la validite du resultat ne tient qu'au test d'efficacite,
    qui certifie chaque point rendu. Aucun ILP de cloture, donc, et aucune
    precondition a etablir.

    QUAND. La campagne montre qu'a n >= 20 la borne ne bouge pas : les ecarts
    garantis restent a 92-98 % quoi qu'on fasse du budget de certification.
    Or ce budget est preleve sur du temps que la recherche, elle, aurait su
    employer. Rendre ce temps a la recherche est donc le seul progres
    disponible dans ce regime -- non pas sur la borne, mais sur la SOLUTION.

    Renvoie (q, x_best, nombre de points neufs certifies).
    """
    t0 = time.time()
    model = ECutModel(inst)
    for a in archive.points()[:cap]:
        model.add_efficiency_cut(a)          # aucune cloture a etablir
    neufs = 0
    while time.time() - t0 < budget:
        w, w0, _, _ = surrogate(inst, q)
        reste = budget - (time.time() - t0)
        r = model.optimize(w, w0, maximize=True,
                           time_limit=max(0.05, min(reste, budget / 3.0)))
        if not r.ok or r.x is None:
            break                            # relache vide : E est epuise
        y = repair_to_efficient(inst, np.asarray(r.x[:inst.n], dtype=int),
                                dominated_out=dominated_out,
                                deadline=t0 + budget)
        if y is None:
            break                            # plus le temps de certifier
        if archive.add(y):
            neufs += 1
        fy = inst.f.value(y)
        if fy > q:
            q, x_best = fy, np.asarray(y, dtype=int)
        model.add_efficiency_cut(y)          # interdire d'y revenir
    return q, x_best, neufs


@dataclass
class CertResult:
    """Sortie de la phase de certification."""
    q_ub: Optional[float] = None        # meilleure borne sup VALIDE sur q*
    proved: bool = False
    q_lb: Optional[Fraction] = None     # incumbent, eventuellement AMELIORE
    x_best: Optional[np.ndarray] = None
    info: dict = field(default_factory=dict)
    cut_points: List[np.ndarray] = field(default_factory=list)


def certify(inst: MOILFP, q: Fraction, x_cur: np.ndarray,
            dominated: Sequence[np.ndarray],
            budget: float,
            use_tightened: bool = True,
            archive: Optional["Archive"] = None,
            cut_batch: int = 10,
            max_rounds: int = 6,
            archive_cuts: bool = False,
            closure_lemma: bool = True,
            lemma_strikes: int = 3,
            height_rank: bool = False,
            agg_extra: int = 0) -> CertResult:
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
                  "rounds": 0, "q_improved": False, "archive_cuts": 0,
                  "select": None, "cloture_lemme": 0, "agg_cuts": 0}

    best_ub: Optional[float] = None
    proved = False
    # Le lemme de cloture est BIMODAL : sur les instances ou il fonctionne il
    # etablit presque toutes les clotures (24 sur 24, 22 sur 22, 21 sur 21) ;
    # sur les autres il n'en etablit AUCUNE. Continuer a payer un PL par
    # candidat quand il a echoue plusieurs fois de suite est donc du gaspillage
    # pur -- et c'est ce gaspillage qui a fait la seule instance en HAUSSE de
    # l'isolement sur 90 (+3 appels, avec zero cloture etablie par le lemme).
    echecs = 0
    Dm: Optional[int] = None

    model = ECutModel(inst)
    pending: List[tuple] = []
    if use_tightened:
        arch_pts = archive.points() if (archive is not None and archive_cuts) \
            else []
        if height_rank:
            # RECLASSEMENT des deux sources sur la hauteur de region, et
            # rejet des candidats sans effet quand la place manque.
            # Mesure sur les 90 instances : gain NET (-393 appels au total)
            # mais de SIGNE VARIABLE par instance (+25 sur l'une, -83 sur une
            # autre), parce qu'il change quelles coupes sont posees. Il n'est
            # donc pas actif par defaut : la ou l'on veut un progres sur
            # CHAQUE instance, on ne veut pas d'un levier qui parie.
            # `cap` = le plafond qui ARBITRE reellement, c'est-a-dire le lot
            # pose en un tour, et non le total sur tous les tours.
            pending, sel = select_cuts(inst, dominated, arch_pts, q,
                                       cap=cut_batch)
            info["select"] = sel
        else:
            w0_coef, _, _, _ = surrogate(inst, q)
            pending = [(x, k, None) for x, k in
                       build_cut_pool(rank_dominated(dominated, w0_coef),
                                      arch_pts, w0_coef)]

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
        # Le vivier melange points domines (Th. 4) et points d'archive
        # (Th. 6). Pour ces derniers, un ILP etablit d'abord la condition de
        # cloture du Th. 7 ; s'il rend un point, celui-ci est efficace ET
        # meilleur que l'incumbent, et on repart de la.
        improved_by_closure = None
        if pending:
            posees = 0
            for cand in pending:
                x, kind = cand[0], cand[1]
                h = cand[2] if len(cand) > 2 else None
                if posees >= cut_batch or time.time() > t0 + budget:
                    break
                if kind == "arch":
                    # LEMME DE CLOTURE PAR LA HAUTEUR :
                    #   h(a) <= 0  =>  la cloture est etablie en a.
                    # {Z(x) = Z(a)} est inclus dans {Z(x) <= Z(a)} ; si le
                    # maximum de Q N - P D sur le second est <= 0, il l'est
                    # sur le premier. h etant un MAJORANT calcule par PL,
                    # h <= 0 suffit : UN PL remplace UN ILP.
                    #
                    # Point essentiel : le lemme ne change RIEN au jeu de
                    # coupes pose. Meme vivier, meme ordre, memes coupes ;
                    # seul le moyen d'etablir la cloture change. Le nombre
                    # d'appels au solveur entier ne peut donc que BAISSER,
                    # jamais monter -- par construction, et pas seulement
                    # en moyenne.
                    if h is None and closure_lemma and echecs < lemma_strikes:
                        h = region_height(inst, x, q_ref)
                        echecs = echecs + 1 if h > 0 else 0
                    if h is None or h > 0:
                        better = same_criteria_improves(inst, x, q_ref)
                        if better is not None:
                            improved_by_closure = better
                            break
                    else:
                        info["cloture_lemme"] = info.get("cloture_lemme", 0) + 1
                    model.add_efficiency_cut(x)
                    info["archive_cuts"] = info.get("archive_cuts", 0) + 1
                else:
                    model.add_dominance_cut(x)
                posees += 1
            pending = pending[posees + (1 if improved_by_closure is not None
                                        else 0):]
        # --- au-dela du plafond : la forme AGREGEE, sans binaire --------
        # Le plafond ne tient pas au nombre de coupes mais a leur cout en
        # binaires. Les candidats qui n'ont pas eu de place peuvent quand
        # meme etre poses sous forme agregee : une ligne, zero binaire, donc
        # aucun plafond. Beaucoup plus faible, mais gratuit.
        # On s'en tient aux candidats dont la cloture ne coute rien : points
        # domines (aucune cloture requise) et points d'archive dont le lemme
        # de hauteur l'a deja etablie.
        if agg_extra > 0 and pending:
            n_agg = 0
            for cand in pending:
                if n_agg >= agg_extra or time.time() > t0 + budget:
                    break
                xa, ka = cand[0], cand[1]
                ha = cand[2] if len(cand) > 2 else None
                if ka == "arch" and not (ha is not None and ha <= 0):
                    continue
                if model.add_aggregated_cut(xa):
                    n_agg += 1
        info["agg_cuts"] = model.n_agg_cuts
        info["n_cuts"] = model.n_cuts

        if improved_by_closure is not None:
            fb = inst.f.value(improved_by_closure)
            if fb > q:
                q = fb
                x_cur = np.asarray(improved_by_closure, dtype=int)
                if archive is not None:
                    archive.add(x_cur)
                info["q_improved"] = True
                continue

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

        # -- RELACHE VIDE : la preuve la plus forte -------------------------
        # Le Th. 6 donne E inclus dans R_A union {x : Z(x) = Z(a), a dans A}.
        # Si R_A est vide, alors E est tout entier dans le second ensemble, et
        # la condition de cloture -- etablie point par point avant chaque
        # coupe d'efficacite -- dit qu'aucun de ses elements ne bat q. Donc
        # q* <= q, et comme q <= q*, q* = q.
        #
        # C'est une preuve que la coupe de DOMINANCE ne peut jamais produire :
        # elle preserve E, donc R ne se vide pas. Seule la coupe d'efficacite
        # peut epuiser le relache. Le cas se presente exactement quand E est
        # mince -- le regime a corr eleve -- ou l'archive couvre vite tout E.
        if r.status == "empty" and info.get("archive_cuts", 0) > 0:
            return CertResult(float(q), True, q, x_cur, info)

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
    return CertResult(best_ub, proved, q, x_cur, info,
                      cut_points=list(model.cut_points))


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
    cut_points: List[np.ndarray] = field(default_factory=list)

    @property
    def gap(self) -> Optional[float]:
        if self.q_lb is None or self.q_ub is None:
            return None
        lb = float(self.q_lb)
        return (self.q_ub - lb) / max(1e-12, abs(self.q_ub))

# ----------------------------------------------------------------------------
# Hybride : la matheuristique amorce et alimente la methode exacte
# ----------------------------------------------------------------------------

def solve_P_warm(inst: MOILFP,
                 time_limit: float = 30.0,
                 warm_frac: float = 0.35,
                 max_preload: int = 40,
                 seed: int = 0,
                 verbose: bool = False):
    """
    Hybride a proprement parler : une PHASE HEURISTIQUE amorce la METHODE
    EXACTE, au lieu que les deux soient deux methodes concurrentes.

    Ce que la methode exacte gaspille sans amorcage. `solve_P` demarre au
    point efficace obtenu en reparant x = 0, dont la valeur de f est
    arbitraire, puis remonte vers q* par iterations de Dinkelbach. Or chaque
    iteration externe coute un appel COMPLET a l'oracle, et l'oracle est la
    partie chere. Partir loin de q* se paie donc en appels a l'oracle, pas en
    arithmetique.

    Ce que la matheuristique fournit, et pourquoi c'est licite :

      * un point efficace CERTIFIE de valeur proche de q*. Le Th. 3 n'impose
        rien au point de depart sinon d'appartenir a l'ensemble optimise :
        partir de la donne q <= q* et l'iteration reste croissante. Le
        demarrage a chaud ne peut donc pas fausser le resultat, seulement
        raccourcir le chemin ;
      * des coupes de dominance deja payees. Le Th. 4 ne depend que de E, pas
        de l'objectif : les coupes posees pendant la recherche restent valides
        pour la methode exacte. Les lui transmettre, c'est lui offrir un
        relache R deja resserre.

    DEUX VOIES DE PREUVE. Le Th. 5' certifie l'optimalite sans fermer le
    sous-probleme, la phase exacte la prouve en le fermant : ce sont deux
    routes independantes vers le meme statut 'optimal', et la premiere est
    strictement moins chere. L'hybride prend celle qui aboutit -- si la phase
    heuristique a deja prouve, il s'arrete la. Ne pas le faire coute cher :
    mesure a l'appui, exiger systematiquement la preuve par fermeture faisait
    tomber l'hybride a 75 preuves sur 90, contre 84 pour la matheuristique
    seule.

    Ce que l'hybride ne change pas : le sens du statut. 'optimal' signifie
    toujours optimalite PROUVEE, par l'une ou l'autre voie. L'amorcage
    accelere ou ne fait rien ; il ne peut pas faire conclure a tort.

    `warm_frac` est la part du budget confiee a la phase heuristique.
    """
    t0 = time.time()
    calls0 = ORACLE_CALLS["ilp"]

    # --- phase heuristique -------------------------------------------------
    wb = warm_frac * time_limit
    mh = matheuristic_P(inst, time_budget=wb * 0.7, bound_budget=wb * 0.3,
                        seed=seed)

    # --- DEUX VOIES DE PREUVE, on prend celle qui aboutit -----------------
    # Mesure : sans ce test, l'hybride prouvait 75 fois sur 90 la ou la
    # matheuristique seule en prouvait 84. Il jetait en effet la preuve deja
    # obtenue pour en exiger une plus chere. Les deux voies sont pourtant
    # independantes et egalement valides :
    #   * Th. 5' certifie SANS fermer le sous-probleme -- il suffit que la
    #     borne U rejoigne l'incumbent ;
    #   * la phase exacte prouve en fermant F(q) <= 0, ce qui est
    #     strictement plus difficile.
    # Une preuve deja en main n'a aucune raison d'etre refaite.
    if mh.proved_optimal:
        return HybridResult(
            "optimal", mh.q_lb, mh.x_best, 0, 0,
            ORACLE_CALLS["ilp"] - calls0, time.time() - t0,
            [], list(mh.archive))

    # --- transfert : relache pre-garni des coupes deja payees --------------
    R = ECutModel(inst)
    for x in mh.cut_points[:max_preload]:
        R.add_dominance_cut(x)

    if verbose:
        print(f"  amorcage : q_lb = {float(mh.q_lb):.6f}, "
              f"{R.n_cuts} coupes transmises, "
              f"{ORACLE_CALLS['ilp'] - calls0} ILP consommes "
              f"({time.time() - t0:.1f} s)")

    # --- phase exacte, amorcee a chaud -------------------------------------
    from molfp_oracle import solve_P
    left = max(0.1, time_limit - (time.time() - t0))
    r = solve_P(inst, time_limit=left, model=R, x0=mh.x_best,
                verbose=verbose)

    # l'incumbent de la phase heuristique ne peut pas etre perdu : la phase
    # exacte demarre dessus et Dinkelbach est croissante
    r.ilp_calls = ORACLE_CALLS["ilp"] - calls0
    r.time = time.time() - t0
    r.archive = list(r.archive) + list(mh.archive)
    return r


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
                   cut_batch: int = 40,
                   cert_rounds: int = 2,
                   archive_cuts: bool = False,
                   closure_lemma: bool = True,
                   lemma_strikes: int = 3,
                   height_rank: bool = False,
                   agg_extra: int = 0,
                   cut_diversify: bool = True,
                   gap_hopeless: float = 0.5,
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
    `cut_batch`  taille des lots de coupes recyclees et `cert_rounds` leur
                 nombre : le plafond n'est plus fixe, il est arbitre par le
                 budget (cf. `certify`).
    """
    t0 = time.time()
    calls0 = ORACLE_CALLS["ilp"]
    rng = np.random.default_rng(seed)
    arch = Archive(inst)

    # --- amorcage : un point efficace quelconque --------------------------
    dominated: List[np.ndarray] = []      # recyclage pour le Th. 5' (gain 3)
    # l'amorcage n'a pas de garde-temps : sans un premier point efficace
    # certifie il n'y a pas de LB du tout, donc rien a rapporter
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
                                        dominated_out=dominated,
                                        deadline=t0 + time_budget)
                if y is None:
                    continue       # non certifie : ni archive ni incumbent
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
    leftover = max(0.0, time_budget - search_time) if reallocate else 0.0
    budget = bound_budget + leftover

    q_ub, proved, cert_info = None, False, {}
    cut_points: List[np.ndarray] = []
    n_neufs = 0

    def _cert(bud: float, rounds: int):
        return certify(inst, q, x_best, dominated, bud,
                       use_tightened=tightened, archive=arch,
                       cut_batch=cut_batch, max_rounds=rounds,
                       archive_cuts=archive_cuts, closure_lemma=closure_lemma,
                       lemma_strikes=lemma_strikes, height_rank=height_rank,
                       agg_extra=agg_extra)

    if certify_bound and budget > 0:
        sonde = None
        # SONDE. Un seul tour de certification, pour SAVOIR si la borne est
        # en train de se fermer, au lieu de le supposer. Toute borne obtenue
        # reste valide : la sonde ne peut donc rien gater, et quand elle
        # repond non elle fait gagner tout le reste du budget.
        if cut_diversify and leftover > 0.3:
            part = min(0.3 * budget, leftover)
            sonde = _cert(part, 1)
            if sonde.q_lb is not None and sonde.q_lb > q:
                q, x_best = sonde.q_lb, sonde.x_best
            q_ub, proved = sonde.q_ub, sonde.proved
            cert_info, cut_points = sonde.info, sonde.cut_points
            budget -= part
            ecart = None if q_ub is None else \
                (q_ub - float(q)) / max(1e-12, abs(q_ub))
            # borne qui ne ferme pas : a n >= 20 la campagne mesure 92-98 %
            # quoi qu'on fasse. Le temps restant vaut alors plus a la
            # RECHERCHE qu'a la certification, et la coupe d'efficacite sert
            # ici d'operateur de diversification -- sans cloture a etablir.
            if not proved and (ecart is None or ecart > gap_hopeless):
                part_div = 0.6 * budget
                q, x_best, n_neufs = diversify_over_archive(
                    inst, arch, q, x_best, part_div, cut_batch, dominated)
                budget -= part_div

        if budget > 0.05 and not proved:
            c = _cert(budget, cert_rounds)
            if c.q_lb is not None and c.q_lb > q:
                q, x_best = c.q_lb, c.x_best   # pas de Dinkelbach offert
            # le MINIMUM de deux bornes valides est une borne valide : q* ne
            # depend pas du q qui a servi a l'obtenir.
            if c.q_ub is not None:
                q_ub = c.q_ub if q_ub is None else min(q_ub, c.q_ub)
            proved = proved or c.proved
            cert_info, cut_points = c.info, c.cut_points
    cert_info["search_time"] = search_time
    cert_info["diversify_new"] = n_neufs
    cert_info["cert_budget"] = budget
    cert_info["restarts"] = n_restarts
    cert_info["moves"] = dict(n_moves)
    cert_info["hits"] = dict(n_hits)

    return MatheurResult(
        q_lb=q, x_best=x_best, q_ub=q_ub, archive=arch.points(),
        ilp_calls=ORACLE_CALLS["ilp"] - calls0, time=time.time() - t0,
        rounds=rounds, proved_optimal=proved,
        status="optimal" if proved else "heuristic", cert=cert_info,
        cut_points=cut_points,
    )
