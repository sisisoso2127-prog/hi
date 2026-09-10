#!/usr/bin/env python3
"""
hybride_complet.py
==================
LA METHODE HYBRIDE, EN UN SEUL FICHIER.

    (MOILFP)  max Z_k(x) = (c_k^T x + a_k) / (d_k^T x + b_k),  k = 1..p
              s.c. A x <= b,  x dans Z^n_+
    (P)       max f(x) = (c^T x + a) / (d^T x + b)   sur E, l'ensemble
              des solutions EFFICACES de (MOILFP)

Ce fichier est autonome : numpy et scipy suffisent. Il est ASSEMBLE a partir
des sources du projet, sans retranscription -- ce qui est mesure est ce qui
est livre.

    python hybride_complet.py demo      les deux illustrations, pas a pas
    python hybride_complet.py verify    controles de validite (verite terrain)
    python hybride_complet.py ab        A/B des leviers, sur instances generees
    python hybride_complet.py scale     n = 20, 30, 40 : ce qui resiste encore

--------------------------------------------------------------------------
CE QUI FAIT L'HYBRIDATION
--------------------------------------------------------------------------
Deux moities, et le sens de circulation de l'information entre elles.

  Phase 1, RECHERCHE.  Voisinages definis dans l'espace des criteres, chaque
    sous-probleme resolu EXACTEMENT par un solveur entier, chaque point
    retenu CERTIFIE efficace (Th. 2). La borne inferieure q n'est donc
    jamais optimiste, meme si l'on interrompt.
  Phase 2, CERTIFICATION.  Un relache R contenant E, sur lequel on maximise
    un substitut lineaire. La borne du Th. 5' convertit un budget borne en
    ecart d'optimalite GARANTI sur q*.

L'hybridation n'est pas l'assemblage des deux -- mesure faite, cela ne rend
rien. Elle est que l'ARCHIVE de points efficaces produite par la phase 1
devienne la SOURCE DES COUPES de la phase 2.

--------------------------------------------------------------------------
LES CINQ RESULTATS QUE CE FICHIER MET EN OEUVRE
--------------------------------------------------------------------------
Th. 2   TEST D'EFFICACITE INTEGRAL.  theta(xbar) entier, xbar efficace ssi
        theta = 0. Exact, sans tolerance ; et quand il echoue il rend
        GRATUITEMENT un point dominant, reutilise comme mouvement de
        recherche et comme base de coupe.        -> `efficiency_test`

Th. 4   COUPE DE DOMINANCE.  Pour xbar DOMINE, { x : Z(x) <= Z(xbar) } ne
        contient aucun point efficace. Retiree par la disjonction
        "il existe k tel que e_k(x) >= 1", p binaires.
                                                 -> `ECutModel.add_dominance_cut`

Th. 5'  BORNE RESSERREE.  q* <= q + U / (Q D+), avec U majorant du substitut
        sur R et D+ le minimum de D sur la seule zone active.
                                                 -> `certify`

Th. 6   COUPE D'EFFICACITE.  a et x EFFICACES avec Z(x) != Z(a) => il existe
        k tel que e_k^a(x) >= 1. On peut donc couper autour d'un point
        EFFICACE : la region retiree ne contient, parmi les efficaces, que
        ceux de MEME vecteur criteres. C'est ce qui ouvre l'archive comme
        source de coupes.                        -> `ECutModel.add_efficiency_cut`

Prop.   RELACHE VIDE.  Si R_A se vide, q* = q. Preuve INACCESSIBLE a la coupe
        de dominance, qui preserve E et ne peut donc jamais vider R. Seule la
        coupe d'efficacite peut EPUISER le relache.
                                                 -> branche `status == "empty"`

--------------------------------------------------------------------------
LA RESERVE, ET COMMENT ELLE SE LEVE
--------------------------------------------------------------------------
Couper en un point efficace `a` retire les points de meme vecteur criteres.
Il faut donc la CONDITION DE CLOTURE : aucun d'eux ne bat l'incumbent q.
Deux moyens de l'etablir, et le second est ce qui a renverse le cout :

  (1) UN PROGRAMME DE FAISABILITE.  Trouver x avec e_k^a(x) = 0 pour tout k
      et Q N(x) - P D(x) >= 1. Infaisable => cloture acquise. Faisable => le
      point rendu est efficace ET bat q : on l'empoche, la borne inferieure
      monte. Aucune branche perdante.            -> `same_criteria_improves`

  (2) LEMME DE CLOTURE PAR LA HAUTEUR.  Soit
          h(a) = max { Q N(x) - P D(x) : x dans S, Z(x) <= Z(a) }.
      Si h(a) <= 0 la cloture est acquise, car { Z(x) = Z(a) } est inclus
      dans { Z(x) <= Z(a) }. Et h se MAJORE par un PL : un PL remplace donc
      un PLNE.                                   -> `region_height`

--------------------------------------------------------------------------
LA COUPE D'EFFICACITE COMME OPERATEUR DE RECHERCHE
--------------------------------------------------------------------------
Couper autour de TOUTE l'archive laisse exactement les points que l'archive
ne domine ni n'egale. Maximiser le substitut sur ce reste rend un point
GARANTI NEUF en vecteur criteres -- ce qu'aucun redemarrage aleatoire ne
garantit. Employee ainsi la coupe n'exige AUCUNE cloture : celle-ci n'est
requise que pour tirer une BORNE d'un relache ampute. La validite ne tient
alors qu'au test d'efficacite, qui certifie chaque point rendu.
                                                 -> `diversify_over_archive`

--------------------------------------------------------------------------
CE QUI EST MESURE, ET CE QUI NE L'EST PAS
--------------------------------------------------------------------------
Lecture APPARIEE seulement (meme instance, meme graine, les deux variantes).
Comparer deux medianes independantes laisse croire a un gain par instance qui
n'existe pas : sur le lot de 90, la mediane du delta d'appels vaut +1 la ou
la comparaison des medianes annoncait -22 %.

  Regime cible (8 instances x 10 graines, budget identique) :
      preuves 58/80 -> 78/80,  appariee +20 / -0
      delta ILP par paire : mediane -12, total -1579, 54/80 en baisse
      Deux instances changent d'ETAT : 0/10 preuves -> 10/10 et -> 8/10.
  Lot fige (90 instances, budget identique) :
      valeur inchangee 90/90 -- la recherche y trouve deja l'optimum, il n'y
      a pas de place pour un progres sur la SOLUTION.
  A n >= 20 : ecarts garantis 98,3 / 97,1 / 92,5 %, INCHANGES. La borne ne
      bouge pas, quoi qu'on fasse du budget de certification. C'est la
      frontiere honnete de la methode, et c'est pourquoi `cut_diversify`
      rend ce budget a la recherche quand la sonde dit que la borne ne
      fermera pas.

AVERTISSEMENT DE MESURE. Les bancs sont a budget de TEMPS. Deux executions
concurrentes se volent le processeur et produisent des ecarts qui ne sont pas
ceux de la methode : un run lance en parallele d'un autre a fait apparaitre
une degradation a n = 40 qui n'existait pas. Mesurer SEUL.

--------------------------------------------------------------------------
LES REGLAGES QUI COMPTENT (`matheuristic_P`)
--------------------------------------------------------------------------
  archive_cuts    coupes d'efficacite depuis l'archive (Th. 6). Le levier.
  closure_lemma   cloture par PL quand h <= 0. Ne change AUCUNE coupe posee,
                  seulement le moyen de les autoriser : le nombre d'appels
                  entiers ne peut donc que baisser. Actif par defaut.
  height_rank     reclassement du vivier par hauteur de region. Gain NET sur
                  le lot (-393 appels) mais de SIGNE VARIABLE par instance
                  (+25 ici, -83 la). INACTIF par defaut : la ou l'on veut un
                  progres sur chaque instance, on ne garde pas un levier qui
                  parie. A rallumer pour experimenter.
  cut_diversify   sonde d'un tour ; si la borne ne ferme pas, le reste du
                  budget va a la diversification par coupes d'efficacite.
  agg_extra       coupes AGREGEES sans binaire, au-dela du plafond. Validite
                  verifiee (0 violation), mais force faible : elles retirent
                  le quart au tiers de ce que retire la disjonction sur les
                  memes points. Inactif par defaut.
  cut_batch       plafond de coupes par tour. Au-dela d'une quarantaine,
                  chaque coupe coute plus qu'elle ne rapporte : elle alourdit
                  le programme entier (p binaires) plus qu'elle ne resserre.

--------------------------------------------------------------------------
PISTES POUR ALLER PLUS LOIN
--------------------------------------------------------------------------
1. DESAGREGER LA DISJONCTION. Le verrou a grande taille n'est plus la
   disponibilite des coupes mais leur COUT : p binaires chacune. La forme
   agregee (`add_aggregated_cut`) est gratuite mais faible. La voie
   serieuse est un CGLP (lift-and-project) : generer, par programmation
   lineaire, l'inegalite valide pour la disjonction la plus violee par
   l'optimum courant du relache. Une ligne, zero binaire, et coupante par
   construction.
2. CLOTURE STRUCTURELLE. Sur 143 220 paires de points efficaces verifiees
   par enumeration, aucune ne partage de vecteur criteres. Une condition
   sur l'instance (verifiee une fois pour toutes plutot qu'une fois par
   point d'archive) rendrait la coupe d'efficacite entierement gratuite.
3. REGIME D'ACTIVATION. Le gain depend de l'epaisseur du front. Un critere
   mesure en cours d'execution -- taille de l'archive contre nombre de
   points domines collectes -- pourrait allumer ou eteindre `archive_cuts`
   tout seul, au lieu d'un reglage fixe.
4. D+ PAR COUPE. D+ est recalcule globalement ; le calculer sur le relache
   apres chaque lot resserrerait la borne sans ILP supplementaire notable.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field, asdict
from fractions import Fraction
from itertools import product
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linprog, milp, LinearConstraint, Bounds

INF = np.inf
Row = Tuple[np.ndarray, float, float]


# ==========================================================================
# extrait de molfp_instance.py
# ==========================================================================


@dataclass
class FracObj:
    """Une fonction fractionnaire lineaire  (num^T x + a) / (den^T x + b)."""
    num: np.ndarray      # vecteur entier de taille n
    a: int
    den: np.ndarray      # vecteur entier de taille n, >= 0
    b: int               # >= 1

    def numerator(self, x: np.ndarray) -> int:
        return int(self.num @ x) + self.a

    def denominator(self, x: np.ndarray) -> int:
        return int(self.den @ x) + self.b

    def value(self, x: np.ndarray) -> Fraction:
        """Valeur EXACTE (rationnelle) de la fonction en x."""
        return Fraction(self.numerator(x), self.denominator(x))

    def value_float(self, x: np.ndarray) -> float:
        return float(self.value(x))


@dataclass
class MOILFP:
    """Instance complete : le multi-objectif + la fonction d'utilite f."""
    A: np.ndarray            # m x n, entiers >= 0
    b: np.ndarray            # m,    entiers > 0
    Z: List[FracObj]         # p criteres fractionnaires
    f: FracObj               # fonction d'utilite a optimiser sur E
    name: str = "unnamed"
    seed: Optional[int] = None
    ub: Optional[np.ndarray] = None   # bornes explicites, si A n'est pas >= 0

    # -- dimensions ---------------------------------------------------------
    @property
    def n(self) -> int:
        return self.A.shape[1]

    @property
    def m(self) -> int:
        return self.A.shape[0]

    @property
    def p(self) -> int:
        return len(self.Z)

    # -- bornes explicites sur les variables --------------------------------
    def var_upper_bounds(self) -> np.ndarray:
        """
        ub_j = min_{i : A[i,j] > 0} floor(b_i / A[i,j]).

        Valide car A >= 0, b >= 0, x >= 0 : toute contrainte i active sur j
        majore x_j. (A2) garantit qu'au moins un A[i,j] > 0 par colonne.

        Si des bornes explicites ont ete fournies (champ `ub`), elles priment :
        c'est le cas des instances de la litterature, dont la matrice A peut
        comporter des coefficients negatifs et pour lesquelles ce calcul
        n'a plus de sens.
        """
        if self.ub is not None:
            return np.asarray(self.ub, dtype=int)
        ub = np.full(self.n, np.inf)
        for j in range(self.n):
            for i in range(self.m):
                if self.A[i, j] > 0:
                    ub[j] = min(ub[j], self.b[i] // self.A[i, j])
        if not np.all(np.isfinite(ub)):
            raise ValueError("Domaine non borne : une colonne de A est nulle.")
        return ub.astype(int)

    # -- evaluation ---------------------------------------------------------
    def is_feasible(self, x: np.ndarray) -> bool:
        return bool(np.all(x >= 0) and np.all(self.A @ x <= self.b))

    def criteria(self, x: np.ndarray) -> tuple:
        """Vecteur EXACT (Z_1(x), ..., Z_p(x)) en Fractions."""
        return tuple(Zk.value(x) for Zk in self.Z)

    def check_assumptions(self, strict: bool = True) -> None:
        """
        Verifie les hypotheses. Leve une exception si elles sont violees.

        `strict=True` (defaut) impose (A1) et (A2) sous leur forme
        SUFFISANTE et facile a verifier : `d_k >= 0`, `b_k >= 1`, `A >= 0`.
        C'est ce que garantit notre generateur.

        `strict=False` impose la forme REELLEMENT NECESSAIRE aux theoremes :
        le domaine est borne, et chaque denominateur reste strictement positif
        sur le domaine. Les preuves des theoremes 1, 2 et 4 n'utilisent en
        effet que `D_k(x) > 0`, jamais `d_k >= 0` : (A1) n'est qu'une
        condition suffisante commode. Ce mode permet de lire les instances de
        la litterature, dont les coefficients peuvent etre negatifs -- c'est
        exactement l'hypothese de Zerdani et Moulai (2011), qui demandent
        `q^i x + beta^i > 0` sur le domaine.
        """
        if strict:
            if np.any(self.A < 0):
                raise ValueError("(A2) violee : A doit etre >= 0.")
            if np.any(self.b < 0):
                raise ValueError("(A2) violee : b doit etre >= 0.")
            for k, Zk in enumerate(self.Z):
                if np.any(Zk.den < 0) or Zk.b < 1:
                    raise ValueError(f"(A1) violee pour le critere {k}.")
            if np.any(self.f.den < 0) or self.f.b < 1:
                raise ValueError("(A1) violee pour la fonction d'utilite f.")
            self.var_upper_bounds()  # leve si non borne
            return

        # --- mode non strict : positivite des denominateurs, verifiee ------
        ub = self.var_upper_bounds()
        if not np.all(np.isfinite(ub)):
            raise ValueError("Domaine non borne.")
        from molfp_core import min_over_relaxation, feasibility_rows
        rows = feasibility_rows(self)
        for k, obj in enumerate(list(self.Z) + [self.f]):
            lo = min_over_relaxation(obj.den.astype(float), float(obj.b),
                                     rows, ub)
            if not (lo > 0):
                who = f"critere {k}" if k < len(self.Z) else "la fonction f"
                raise ValueError(
                    f"Denominateur non strictement positif sur le domaine "
                    f"pour {who} (minorant {lo}).")

    # -- serialisation ------------------------------------------------------
    def to_dict(self) -> dict:
        def fo(o: FracObj) -> dict:
            return {"num": o.num.tolist(), "a": int(o.a),
                    "den": o.den.tolist(), "b": int(o.b)}
        return {
            "name": self.name, "seed": self.seed,
            "n": self.n, "m": self.m, "p": self.p,
            "A": self.A.tolist(), "b": self.b.tolist(),
            "Z": [fo(z) for z in self.Z], "f": fo(self.f),
            "ub": None if self.ub is None else np.asarray(self.ub).tolist(),
        }

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=1)

    @staticmethod
    def load(path: str) -> "MOILFP":
        with open(path) as fh:
            d = json.load(fh)
        def fo(o: dict) -> FracObj:
            return FracObj(np.array(o["num"], dtype=int), o["a"],
                           np.array(o["den"], dtype=int), o["b"])
        return MOILFP(
            A=np.array(d["A"], dtype=int), b=np.array(d["b"], dtype=int),
            Z=[fo(z) for z in d["Z"]], f=fo(d["f"]),
            name=d["name"], seed=d["seed"],
            ub=(None if d.get("ub") is None
                else np.array(d["ub"], dtype=int)),
        )


def generate(n: int, m: int, p: int, seed: int,
             coef_max: int = 20, den_max: int = 5,
             rhs_scale: float = 2.0,
             corr: float = 0.0,
             name: Optional[str] = None) -> MOILFP:
    """
    Genere une instance aleatoire de (MOILFP) + fonction d'utilite f.

    Parametres
    ----------
    n, m, p     : nb de variables, de contraintes, de criteres
    seed        : graine (REPRODUCTIBILITE : toujours l'enregistrer)
    coef_max    : borne sup des coefficients des numerateurs
    den_max     : borne sup des coefficients des denominateurs (petits = ratios
                  plus contrastes, donc front plus interessant)
    rhs_scale   : controle la taille du domaine realisable. Plus grand
                  => domaine plus large => |S| et |E| plus grands.
                  Calibration mesuree (p=3, 3 graines) :
                      n=4,m=3 : 1.0 -> |S| ~ 20-55    2.0 -> ~150-400
                      n=6,m=4 : 1.0 -> |S| ~ 200-400  2.0 -> ~4600-8400
                      n=8,m=4 : 1.0 -> |S| ~ 2700-4100
                  Au-dela, l'enumeration exhaustive devient impraticable.

    corr        : correlation entre les numerateurs des criteres, dans [0, 1].
                  0 -> criteres independants (beaucoup de compromis, E epais)
                  1 -> criteres quasi identiques (peu de compromis, E mince)
                  Ce parametre permet de MANIPULER la finesse de E au lieu de
                  seulement l'observer : c'est ce qui rend l'etude de
                  difficulte causale et non purement correlationnelle.

    Garanties : (A1) et (A2) verifiees, S non vide (x = 0 realisable).
    """
    rng = np.random.default_rng(seed)

    # --- contraintes : A >= 0, chaque colonne non nulle -> domaine borne ----
    A = rng.integers(0, 10, size=(m, n))
    for j in range(n):                      # garantit une colonne non nulle
        if A[:, j].sum() == 0:
            A[rng.integers(0, m), j] = rng.integers(1, 10)
    # b choisi proportionnellement a la somme des lignes : controle |S|
    b = np.maximum(1, (rhs_scale * A.sum(axis=1)).astype(int))

    # --- composantes communes aux criteres (numerateur ET denominateur) ----
    # correler les seuls numerateurs ne suffit pas : des denominateurs
    # independants recreent du conflit entre les ratios, et l'effet sur la
    # finesse de E reste faible. On correle donc les deux.
    corr = float(np.clip(corr, 0.0, 1.0))
    base_num = rng.uniform(1, coef_max, size=n)
    base_den = rng.uniform(0, den_max, size=n)
    base_a = rng.uniform(0, 10)
    base_b = rng.uniform(1, 10)

    def mix(base, own, lo, hi):
        v = corr * base + (1.0 - corr) * own
        return np.clip(np.rint(v), lo, hi).astype(int)

    def make_criterion() -> FracObj:
        num = mix(base_num, rng.uniform(1, coef_max, size=n), 1, coef_max)
        den = mix(base_den, rng.uniform(0, den_max, size=n), 0, den_max)
        a = int(np.clip(round(corr * base_a + (1 - corr) * rng.uniform(0, 10)), 0, 10))
        b = int(np.clip(round(corr * base_b + (1 - corr) * rng.uniform(1, 10)), 1, 10))
        return FracObj(num=num, a=a, den=den, b=b)   # b >= 1 -> (A1)

    def make_utility() -> FracObj:
        # la fonction d'utilite f reste independante des criteres
        return FracObj(
            num=rng.integers(1, coef_max + 1, size=n),
            a=int(rng.integers(0, 10)),
            den=rng.integers(0, den_max + 1, size=n),
            b=int(rng.integers(1, 10)),
        )

    inst = MOILFP(
        A=A.astype(int), b=b.astype(int),
        Z=[make_criterion() for _ in range(p)],
        f=make_utility(),
        name=name or f"molfp_n{n}_m{m}_p{p}_c{int(100*corr):03d}_s{seed}",
        seed=seed,
    )
    inst.check_assumptions()
    return inst



# ==========================================================================
# extrait de molfp_core.py
# ==========================================================================


_MIN_TIME_LIMIT = 0.05


ORACLE_CALLS = {"ilp": 0, "lp": 0}


ORACLE_BUDGET: dict = {"ilp": None}


def set_ilp_budget(n: Optional[int]) -> None:
    """Plafond d'appels au solveur ENTIER, ou None pour l'illimite."""
    ORACLE_BUDGET["ilp"] = n


def ilp_budget_left() -> float:
    cap = ORACLE_BUDGET["ilp"]
    return np.inf if cap is None else max(0, cap - ORACLE_CALLS["ilp"])


def reset_oracle_counter() -> None:
    ORACLE_CALLS["ilp"] = 0
    ORACLE_CALLS["lp"] = 0


@dataclass
class ILPResult:
    """
    Resultat d'un appel au solveur.

    `obj`   valeur de l'incumbent : une borne du cote PESSIMISTE (minorant en
            maximisation, majorant en minimisation). None si aucun incumbent.
    `bound` borne du cote OPTIMISTE, toujours VALIDE : majorant en
            maximisation, minorant en minimisation. Egale a `obj` quand le
            statut est 'optimal'. C'est elle qui rend un ILP interrompu
            exploitable au lieu d'etre perdu.
    """
    status: str          # 'optimal' | 'infeasible' | 'unbounded' | 'limit' | 'error'
    x: Optional[np.ndarray]
    obj: Optional[float]
    bound: Optional[float] = None
    n_calls: int = 1

    @property
    def ok(self) -> bool:
        return self.status == "optimal"


def solve_milp(obj: np.ndarray,
               rows: Sequence[Tuple[np.ndarray, float, float]],
               var_lb: np.ndarray,
               var_ub: np.ndarray,
               integrality: Optional[np.ndarray] = None,
               maximize: bool = True,
               obj_const: float = 0.0,
               time_limit: Optional[float] = None) -> ILPResult:
    """
    Version generale de solve_ilp acceptant des bornes et une integralite par
    variable. Necessaire pour les coupes de dominance, qui introduisent des
    variables binaires auxiliaires.

    integrality = None  ->  toutes les variables sont entieres (cas courant
    ici : x entier et les auxiliaires u binaires).

    `time_limit` (secondes) est transmis a HiGHS. Sans lui, un seul ILP peut
    depasser a lui seul le budget de l'algorithme appelant, qui ne verifie
    l'heure qu'ENTRE deux appels : la limite de temps d'un schema anytime
    n'en serait pas une. Interrompu, le solveur rend tout de meme
    `mip_dual_bound`, borne optimiste VALIDE : c'est elle que l'on remonte
    dans `bound`. Un ILP interrompu informe donc encore, au lieu d'etre perdu.
    """
    if ilp_budget_left() <= 0:
        # ressource epuisee : on rend le statut d'un ILP interrompu, SANS
        # consulter le solveur et SANS incrementer le compteur. Le plafond
        # est ainsi exactement respecte, et la trajectoire ne depend plus de
        # l'horloge.
        return ILPResult("limit", None, None, None)

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

    options = None
    if time_limit is not None and np.isfinite(time_limit):
        options = {"time_limit": max(_MIN_TIME_LIMIT, float(time_limit))}

    res = milp(c=cost, constraints=cons,
               integrality=np.asarray(integrality, dtype=float),
               bounds=Bounds(np.asarray(var_lb, dtype=float),
                             np.asarray(var_ub, dtype=float)),
               options=options)

    if res.status == 0:
        x = np.rint(res.x).astype(int)
        val = float(obj @ x) + obj_const
        return ILPResult("optimal", x, val, val)

    if res.status == 1:                     # limite de temps ou d'iterations
        dual = getattr(res, "mip_dual_bound", None)
        bound = None
        if dual is not None and np.isfinite(dual):
            # HiGHS minimise `cost` ; on repasse dans le sens de `obj`
            bound = (-float(dual) if maximize else float(dual)) + obj_const
        x = None
        if res.x is not None:
            cand = np.asarray(res.x, dtype=float)
            # un incumbent non entier ne serait pas une solution : on l'ignore
            if np.allclose(cand, np.rint(cand), atol=1e-6):
                x = np.rint(cand).astype(int)
        obj_val = float(obj @ x) + obj_const if x is not None else None
        return ILPResult("limit", x, obj_val, bound)

    if res.status == 2:
        return ILPResult("infeasible", None, None, None)
    if res.status == 3:
        return ILPResult("unbounded", None, None, None)
    return ILPResult("error", None, None, None)


def solve_ilp(obj: np.ndarray,
              rows: Sequence[Tuple[np.ndarray, float, float]],
              var_ub: np.ndarray,
              maximize: bool = True,
              obj_const: float = 0.0,
              time_limit: Optional[float] = None) -> ILPResult:
    """Cas particulier : variables x uniquement, 0 <= x <= var_ub, entieres."""
    return solve_milp(obj, rows, np.zeros(len(obj)), var_ub,
                      maximize=maximize, obj_const=obj_const,
                      time_limit=time_limit)


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


def max_over_relaxation(coef: np.ndarray,
                       const: float,
                       rows: Sequence[Tuple[np.ndarray, float, float]],
                       var_ub: np.ndarray) -> float:
    """
    Majorant VALIDE de  max { coef^T x + const : x in S }  sur la relaxation
    continue de S = { x entier, 0 <= x <= var_ub, rows }.

    Symetrique de `min_over_relaxation`, a une difference pres qui compte :
    ici l'ensemble peut etre VIDE (on l'appelle sur des regions definies par
    des contraintes supplementaires, pas sur S entier). Un maximum sur le vide
    vaut -inf, et c'est l'information la plus utile qui soit -- pas un echec.
    On distingue donc les trois cas :

        -inf  la relaxation est vide, donc S l'est aussi ;
        +inf  le LP a echoue pour une autre raison, ou l'ensemble est non
              borne : l'appelant doit retomber sur le comportement prudent ;
        fini  majorant valide.

    Les coefficients etant ENTIERS et x entier, coef^T x est entier : on
    descend au plancher entier du majorant continu. La marge 1e-6 absorbe
    l'erreur du simplexe et ne peut que relacher le majorant.
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

    c = np.asarray(coef, dtype=float)
    res = linprog(c=-c,
                  A_ub=np.array(A_ub) if A_ub else None,
                  b_ub=np.array(b_ub) if b_ub else None,
                  bounds=bounds, method="highs")
    if not res.success:
        # status 2 = infaisable : le vide, pas une panne.
        return -np.inf if getattr(res, "status", None) == 2 else np.inf

    integral = bool(np.all(c == np.rint(c)))
    val = -float(res.fun)
    if integral:
        val = float(np.floor(val + 1e-6))
    return val + const


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


def reduce_row(coef: np.ndarray, lo: float, hi: float) -> Tuple[np.ndarray, float, float]:
    """
    Divise une ligne ENTIERE par le pgcd de ses coefficients et de ses bornes.

    L'ensemble realisable est INCHANGE -- diviser une inegalite par un entier
    positif est une identite -- mais les amplitudes baissent, et avec elles le
    risque qu'un solveur se trompe sur un sommet degenere. Or le test
    d'efficacite place precisement xbar sur un tel sommet : il y SATURE les p
    contraintes a la fois, e_k(xbar) valant 0 pour tout k. C'est la
    configuration ou un statut « infaisable » errone a ete observe, sur un
    programme dont xbar est pourtant une solution realisable.

    Sans effet si la ligne n'est pas entiere, ou si le pgcd vaut 1.
    """
    vals = [v for v in np.asarray(coef, dtype=float)]
    for b in (lo, hi):
        if np.isfinite(b):
            vals.append(float(b))
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0 or not np.all(arr == np.rint(arr)):
        return coef, lo, hi
    g = 0
    for v in np.rint(arr).astype(np.int64):
        g = np.gcd(g, abs(int(v)))
    if g <= 1:
        return coef, lo, hi
    return (np.asarray(coef, dtype=float) / g,
            lo / g if np.isfinite(lo) else lo,
            hi / g if np.isfinite(hi) else hi)


def feasibility_rows(inst: MOILFP) -> List[Tuple[np.ndarray, float, float]]:
    """Contraintes  A x <= b."""
    return [(inst.A[i].astype(float), -INF, float(inst.b[i]))
            for i in range(inst.m)]


@dataclass
class EfficiencyResult:
    """
    `efficient` vaut True/False quand le test CONCLUT, et None quand l'ILP a
    ete interrompu sans permettre de trancher. Ne jamais lire None comme
    "non efficace" : le seul certificat d'efficacite du projet passe par ce
    test, et un incumbent declare efficace a tort rendrait le LB optimiste,
    c'est-a-dire la matheuristique inutilisable.
    """
    efficient: Optional[bool]
    theta: Optional[int]           # entier ; 0 <=> efficace
    dominator: Optional[np.ndarray]  # solution dominante trouvee si non efficace
    status: str = "proved"         # 'proved' | 'limit' | 'solveur'

    @property
    def conclusive(self) -> bool:
        return self.efficient is not None


def efficiency_test(inst: MOILFP, xbar: np.ndarray,
                    time_limit: Optional[float] = None) -> EfficiencyResult:
    """
    Test d'efficacite integral de xbar pour (MOILFP).  Un seul ILP.
    Renvoie theta et, si xbar n'est pas efficace, une solution qui le domine.

    Si l'ILP est interrompu, deux cas seulement :
      * l'incumbent donne theta >= 1 : xbar est PROUVE non efficace et le
        certificat de dominance est valide -- un incumbent realisable suffit ;
      * l'incumbent donne theta = 0 : on ne sait rien, car theta n'est
        minore que par 0. Le resultat est alors non concluant.
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
        # la CONTRAINTE est reduite par son pgcd (ensemble realisable
        # inchange, amplitudes plus petites) ; l'OBJECTIF ne l'est pas, car
        # c'est lui qui porte theta, dont l'integralite fait l'exactitude du
        # test « theta = 0 ».
        rows.append(reduce_row(coef, rhs, INF))   # Z_k(x) >= Z_k(xbar)
        obj += coef
        const += float(Dk * Zk.a - Nk * Zk.b)

    res = solve_ilp(obj, rows, ub, maximize=True, obj_const=const,
                    time_limit=time_limit)

    if res.status not in ("optimal", "limit"):
        # xbar est REALISABLE pour ce programme : e_k(xbar) = 0 pour tout k.
        # Un statut « infaisable » ne peut donc venir que du solveur, jamais
        # du modele -- et cela s'observe, selon la version de HiGHS, sur ce
        # sommet degenere ou xbar sature les p contraintes a la fois.
        # On retente UNE fois sans limite de temps.
        res = solve_ilp(obj, rows, ub, maximize=True, obj_const=const)

    if res.status == "limit":
        theta_lb = int(round(res.obj)) if res.obj is not None else 0
        if theta_lb > 0 and res.x is not None:
            # un point qui domine strictement suffit a conclure : pas besoin
            # de connaitre le maximum de theta
            return EfficiencyResult(False, theta_lb, res.x, status="limit")
        return EfficiencyResult(None, None, None, status="limit")

    if not res.ok:
        # Apres la seconde chance, on renonce -- sans lever. Un test non
        # concluant ne coute que de ne pas certifier CE point : les appelants
        # le traitent deja (`repair_to_efficient` rend None, l'oracle rend
        # les bornes acquises), et le LB reste valide. Lever, au contraire,
        # detruirait une execution dont tout le reste etait bon.
        return EfficiencyResult(None, None, None, status="solveur")

    theta = int(round(res.obj))
    if theta == 0:
        return EfficiencyResult(True, 0, None)
    return EfficiencyResult(False, theta, res.x)


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

    # `max_iter` atteint. La convergence finie est demontree -- q croit
    # strictement a chaque tour et S est fini -- donc y arriver signale une
    # anomalie, en pratique un solveur qui rend un statut aberrant. On rend
    # le meilleur point atteint avec le statut 'limit' plutot que de lever :
    # x_best est REALISABLE, donc q en est une borne inferieure valide, et
    # les appelants savent lire un statut non optimal.
    return DinkelbachResult(q, x_best, max_iter, "limit", trace)


def max_f_over_S(inst: MOILFP) -> DinkelbachResult:
    """
    Borne superieure initiale q_UB^(0) = max_{x in S} f(x)  (relaxation E -> S).

    Valide car E est inclus dans S, donc max_S f >= max_E f = q*.
    """
    return dinkelbach(inst.f, feasibility_rows(inst), inst.var_upper_bounds())


def upper_bound_over_S(inst: MOILFP) -> float:
    """
    MAJORANT VALIDE de q* : le maximum de f sur S, puisque E est inclus dans S.

    Le detour par cette fonction n'est pas cosmetique. `max_f_over_S` rend un
    `DinkelbachResult` dont `q_star` est le maximum cherche UNIQUEMENT si le
    statut est 'optimal' ; dans les autres cas c'est le meilleur point
    atteint, donc un MINORANT du maximum -- et s'en servir comme majorant de
    q* serait faux. On rend alors +inf, qui est un majorant valide et le dit.
    """
    r = max_f_over_S(inst)
    if r.status != "optimal" or r.q_star is None:
        return float("inf")
    return float(r.q_star)


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



# ==========================================================================
# extrait de molfp_oracle.py
# ==========================================================================


TIGHT_BIG_M = True


CHAINES: List[int] = []


def reset_chaines() -> None:
    CHAINES.clear()


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
        self.n_agg_cuts = 0          # coupes agregees, sans binaire
        self.n_lin_rows = 0          # coupes lineaires en x, sans binaire
        self.tight_big_m = TIGHT_BIG_M if tight_big_m is None else tight_big_m
        # points de base des coupes posees. Les conserver permet de verifier
        # l'invariant E inclus dans R sans connaitre E : tout point EFFICACE
        # certifie doit satisfaire la disjonction de chaque coupe. C'est le
        # seul controle de surete des coupes qui reste possible quand
        # l'enumeration exhaustive n'est plus praticable.
        self.cut_points: List[np.ndarray] = []
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

    # -- ligne e_k et minorant, mutualises ---------------------------------
    def _e_row(self, xbar: np.ndarray, k: int):
        """Coefficients, constante et minorant valide de e_k autour de xbar."""
        Zk = self.inst.Z[k]
        Nbar, Dbar = Zk.numerator(xbar), Zk.denominator(xbar)
        a = (Dbar * Zk.num - Nbar * Zk.den).astype(float)
        b = float(Dbar * Zk.a - Nbar * Zk.b)
        e_min_box = float(np.sum(np.minimum(a, 0.0) * self.ub_x)) + b
        e_min = e_min_box
        if self.tight_big_m:
            e_min_lp = min_over_relaxation(a, b, self._rows_x, self.ub_x)
            if np.isfinite(e_min_lp):
                e_min = max(e_min_box, e_min_lp)
        return a, b, e_min, e_min_box

    # -- coupe LINEAIRE en espace x, sans binaire --------------------------
    def add_linear_row(self, coef: np.ndarray, lo: float = -INF,
                       hi: float = INF) -> None:
        """
        Ajoute une inegalite valide portant sur les seules variables x.

        Sert aux coupes disjonctives produites par programme generateur
        (`molfp_cglp`) : une ligne, aucune binaire, donc aucune place
        consommee sous le plafond de coupes. Elle entre dans `_rows_x`, si
        bien qu'elle resserre aussi le calcul du big-M des coupes
        disjonctives posees ENSUITE -- un effet secondaire souhaitable.

        L'appelant repond de la validite : la ligne doit etre satisfaite par
        tout point efficace que l'on veut conserver.
        """
        self._rows_x.append((np.asarray(coef, dtype=float), lo, hi))
        self.n_lin_rows = getattr(self, "n_lin_rows", 0) + 1

    # -- coupe AGREGEE : la meme region, sans une seule binaire -------------
    def add_aggregated_cut(self, xbar: np.ndarray) -> bool:
        """
        Relachement AGREGE de la disjonction, a ZERO variable binaire.

        Si x echappe a la region retranchee, il existe j tel que
        e_j(x) >= 1. En minorant les autres termes par m_k <= min_S e_k :

            sum_k e_k(x)  >=  1 + sum_{k != j} m_k                (pour ce j)
                          >=  1 + min_j sum_{k != j} m_k
                          =   1 + sum_k m_k - max_k m_k

        La derniere ligne ne depend plus de j : elle est valide pour toute
        la disjonction, donc pour E. UNE seule ligne, AUCUNE binaire.

        POURQUOI. Le plafond de coupes ne vient pas du nombre de coupes mais
        de leur cout : p binaires chacune. A n >= 20 la mesure montre que
        c'est ce plafond, et non la disponibilite des coupes, qui bloque la
        borne. Une coupe sans binaire n'a, elle, pas de plafond.

        EN CONTREPARTIE elle est beaucoup plus FAIBLE que la disjonction :
        elle n'exclut meme pas xbar en general, puisque sum_k e_k(xbar) = 0
        et que le membre de droite est negatif des que les m_k le sont.
        C'est un relachement, pas un equivalent -- et c'est pourquoi elle ne
        remplace pas la forme disjonctive, elle la complete au-dela du
        plafond.

        Renvoie True si la ligne a ete posee, False si elle est vide de sens
        (membre de droite -inf, ou coupe trivialement satisfaite sur la
        boite).
        """
        nvar = self.nvar
        coef = np.zeros(nvar)
        cst = 0.0
        mins = []
        for k in range(self.p):
            a, b, e_min, _ = self._e_row(xbar, k)
            coef[:self.n] += a
            cst += b
            mins.append(e_min)
        if not all(np.isfinite(m) for m in mins):
            return False
        rhs = 1.0 + sum(mins) - max(mins) - cst
        # La ligne n'a d'interet que si elle peut mordre : si le minimum de
        # sum_k e_k sur la boite lui est deja superieur, elle est redondante.
        borne_boite = float(np.sum(np.minimum(coef[:self.n], 0.0) * self.ub_x))
        if borne_boite >= rhs - 1e-9:
            return False
        self._cut_rows.append((coef, rhs, INF))
        self.n_agg_cuts += 1
        return True

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

        self.cut_points.append(np.array(xbar, dtype=int))

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

    # -- coupe d'EFFICACITE (Th. 6) ---------------------------------------
    def add_efficiency_cut(self, a: np.ndarray) -> None:
        """
        Retire { x : Z(x) <= Z(a) } pour `a` EFFICACE certifie.

        Structurellement identique a `add_dominance_cut`, et pour une raison
        plus forte que la ressemblance : c'est LE MEME THEOREME.

        La demonstration ne suppose que `a` REALISABLE. Elle etablit que
        pour tout x EFFICACE de vecteur criteres different de Z(a), il
        existe k tel que e_k(x) >= 1 -- la contradiction porte sur
        l'efficacite de x, jamais sur celle de a. Les deux methodes ne
        different donc que par le REGIME de la condition de cloture :

          * `add_dominance_cut` : `a` domine. Aucun point efficace ne
            partage Z(a) -- sinon son dominateur dominerait ce point
            efficace -- donc la cloture est AUTOMATIQUE.
          * `add_efficiency_cut` : `a` efficace. Les points efficaces de
            meme vecteur criteres sont retires, et l'appelant doit avoir
            etabli qu'aucun ne bat l'incumbent.

        CE QUE L'EFFICACITE DE `a` NE SERT PAS. Elle n'intervient pas dans
        la validite de la coupe. Elle ne sert qu'a la branche FAISABLE du
        programme de cloture : c'est parce que `a` est efficace qu'un point
        de meme vecteur criteres l'est aussi, et peut etre empoche comme
        incumbent. Un point de S au statut INDETERMINE reste donc un centre
        de coupe legitime, sa cloture etant etablie -- possibilite ouverte
        par le theoreme et que la methode n'exploite pas encore.

        Pourquoi cette coupe manquait. Les coupes de dominance ne se posent
        que sur des points domines, c'est-a-dire sur ce que les chaines de
        reparation traversent -- or leur profondeur mediane vaut 1. Dans le
        regime a E epais, ou l'archive est grande et les points domines rares,
        il n'y avait donc presque rien a couper. Cette coupe puise dans
        l'archive, qui y est justement abondante.
        """
        self.add_dominance_cut(a)

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
                 maximize: bool = True, extra_rows=None,
                 time_limit: Optional[float] = None):
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
                          obj_const=g_const, time_limit=time_limit)

    def maximize_linear(self, g_coef: np.ndarray, g_const: float = 0.0,
                        time_limit: Optional[float] = None):
        """max g sur R. Renvoie ILPResult ; res.x est en espace etendu."""
        return self.optimize(g_coef, g_const, maximize=True,
                             time_limit=time_limit)


def repair_to_efficient(inst: MOILFP, x: np.ndarray,
                        max_steps: int = 100,
                        dominated_out: Optional[list] = None,
                        deadline: Optional[float] = None,
                        indetermine_out: Optional[list] = None
                        ) -> Optional[np.ndarray]:
    """
    Suit la chaine de dominance jusqu'a atteindre un point efficace certifie.

    Termine : chaque pas ameliore strictement le vecteur critere au sens de
    Pareto, S est fini, donc pas de cycle.

    `dominated_out` : si une liste est fournie, tous les points DOMINES
    traverses y sont ajoutes. Ils sont exactement les points sur lesquels le
    Th. 4 autorise une coupe de dominance ; les jeter serait gaspiller une
    information deja payee.

    `deadline` : instant (time.time()) au-dela duquel on renonce. Renvoie
    alors None -- JAMAIS le dernier point atteint. Ce point n'a pas ete
    certifie efficace, et le rendre comme s'il l'etait rendrait le LB
    optimiste : c'est la seule propriete que la matheuristique ne peut pas
    perdre. Les points domines deja traverses restent acquis dans
    `dominated_out`.

    `indetermine_out` : si une liste est fournie, le dernier point atteint
    sans avoir pu etre classe y est depose. Il ne doit JAMAIS servir de
    minorant -- son efficacite est inconnue -- mais le Th. 1 generalise
    l'autorise comme CENTRE DE COUPE : sa demonstration ne suppose du centre
    que d'etre realisable, et la reserve (les points efficaces de meme
    vecteur criteres) se leve par la meme condition de cloture que pour un
    point d'archive. La difference est dans la branche faisable : elle rend
    alors un point dont on ne sait PAS s'il est efficace, donc qu'on ne peut
    pas empocher.
    """
    cur = x
    for pas in range(max_steps):
        tl = None
        if deadline is not None:
            tl = deadline - time.time()
            if tl <= 0:
                if indetermine_out is not None:
                    indetermine_out.append(np.array(cur, dtype=int))
                return None
        r = efficiency_test(inst, cur, time_limit=tl)
        if not r.conclusive:
            if indetermine_out is not None:
                indetermine_out.append(np.array(cur, dtype=int))
            return None
        if r.efficient:
            CHAINES.append(pas)      # 0 = le point etait deja efficace
            return cur
        if dominated_out is not None:
            dominated_out.append(np.array(cur, dtype=int))
        cur = r.dominator
    # Chaine anormalement longue. Chaque pas passe a un point qui DOMINE
    # strictement le precedent, donc la chaine est finie et courte en
    # pratique (profondeur mediane mesuree : 1). Depasser `max_steps` signale
    # une anomalie -- mais on rend None plutot que de lever : un point non
    # certifie n'est simplement pas retenu, et le LB reste valide.
    return None


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
                      ilp_grace: float = 1.0,
                      verbose: bool = False) -> OracleResult:
    """
    Maximise  g(x) = g_coef^T x + g_const  sur l'ensemble efficace E.

    tol : ecart relatif accepte. tol = 0 -> resolution exacte.
    model : un ECutModel deja garni de coupes peut etre passe pour etre
            reutilise d'un appel a l'autre (crucial dans Dinkelbach : les
            coupes restent valides puisqu'elles ne dependent que de E).
    ilp_grace : sursis accorde a un ILP demarre pres de l'echeance, en
            fraction de `time_limit`. Majore le depassement total ; a 0 on
            coupe pile a l'echeance, ce qui respecte le budget a la seconde
            mais degrade la borne (cf. commentaire dans la boucle).
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

        deadline = t0 + time_limit
        # POLITIQUE DE COUPURE. Couper l'ILP pile a l'echeance est perdant :
        # une relaxation resolue a l'optimum donne l'argmax, donc une COUPE,
        # alors qu'une resolution interrompue ne donne qu'une borne duale, et
        # la coupe vaut bien plus que la borne (mesure : UB 369 contre 474 a
        # 10 s sur n5 m3 p4). On laisse donc l'ILP en cours finir, dans la
        # limite d'un sursis borne : le depassement total reste majore par
        # `grace`, et aucun ILP ne peut s'emballer indefiniment -- ce qui
        # etait le vrai risque, un appel unique n'ayant aucune limite.
        # Le sursis vaut pour TOUTE l'iteration -- relaxation, test
        # d'efficacite et chaine de reparation. Le donner a la seule
        # relaxation revient a laisser l'iteration s'interrompre juste apres,
        # donc a perdre la coupe qu'elle allait produire : le meme gaspillage,
        # deplace d'un cran.
        slack = deadline + ilp_grace * time_limit
        res = R.maximize_linear(g_coef, g_const,
                                time_limit=slack - time.time())

        if res.status == "infeasible":
            # R vide : plus aucun candidat non coupe
            status = "optimal" if x_best is not None else "empty"
            return OracleResult(status, x_best,
                                LB if x_best is not None else None,
                                LB if x_best is not None else None,
                                it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)

        if res.status == "limit":
            # La relaxation n'a pas ete resolue a l'optimum. `res.bound` reste
            # une borne superieure VALIDE de max_R g, donc de max_E g : c'est
            # tout ce qu'il faut au schema anytime. En revanche res.x n'est
            # plus l'argmax sur R : on ne peut ni conclure a l'optimalite si
            # ce point est efficace, ni couper si l'incumbent manque.
            if res.bound is not None and np.isfinite(res.bound):
                UB = min(UB, res.bound)
            if res.x is None:
                return OracleResult("limit", x_best,
                                    LB if x_best is not None else None,
                                    UB if np.isfinite(UB) else None,
                                    it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                    time.time() - t0, incumbents)
            xbar = res.x[:inst.n]
            truncated = True
        elif not res.ok:
            return OracleResult("limit", x_best,
                                LB if x_best is not None else None,
                                UB if np.isfinite(UB) else None,
                                it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)
        else:
            xbar = res.x[:inst.n]
            UB = res.obj
            truncated = False

        eff = efficiency_test(inst, xbar, time_limit=slack - time.time())
        if not eff.conclusive:
            # sans certificat on ne peut ni couper ni archiver : on rend les
            # bornes acquises, qui restent valides
            return OracleResult("limit", x_best,
                                LB if x_best is not None else None,
                                UB if np.isfinite(UB) else None,
                                it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)

        if eff.efficient:
            if collect:
                incumbents.append(xbar)
            if truncated:
                # xbar est efficace mais n'est PAS prouve argmax sur R :
                # il ne borne que par en dessous. Si la borne superieure le
                # rejoint tout de meme, l'optimalite est acquise malgre tout.
                if g_of(xbar) > LB:
                    LB, x_best = g_of(xbar), xbar
                proved = np.isfinite(UB) and \
                    (UB - LB) / max(1e-12, abs(UB)) <= 1e-9
                return OracleResult("optimal" if proved else "limit",
                                    x_best, LB,
                                    UB if np.isfinite(UB) else None,
                                    it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                    time.time() - t0, incumbents)
            # optimal : xbar maximise g sur R qui contient E
            return OracleResult("optimal", xbar, g_of(xbar), UB, it, R.n_cuts,
                                ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)

        # xbar domine : on en tire quand meme un point efficace (borne inf)
        x_eff = repair_to_efficient(inst, eff.dominator, deadline=slack)
        if x_eff is None:
            # la chaine n'a pas abouti dans le temps : la coupe sur xbar reste
            # licite (xbar est prouve domine), mais aucun LB nouveau
            R.add_dominance_cut(xbar)
            return OracleResult("limit", x_best,
                                LB if x_best is not None else None,
                                UB if np.isfinite(UB) else None,
                                it, R.n_cuts, ORACLE_CALLS["ilp"] - calls0,
                                time.time() - t0, incumbents)
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


def dedup_archive(archive: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Retire les doublons de l'archive de solutions efficaces certifiees."""
    seen, out = set(), []
    for x in archive:
        k = tuple(int(v) for v in x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out



# ==========================================================================
# extrait de molfp_cglp.py
# ==========================================================================


def optimum_relaxation(coef: np.ndarray, rows: Sequence[Row],
                       ub: np.ndarray) -> Optional[np.ndarray]:
    """argmax de coef^T x sur la relaxation continue. None si echec."""
    n = len(coef)
    A, b = _rows_en_Ax_le_b(rows, n)
    ORACLE_CALLS["lp"] += 1
    res = linprog(c=-np.asarray(coef, dtype=float),
                  A_ub=A if A.size else None, b_ub=b if A.size else None,
                  bounds=[(0.0, float(u)) for u in ub], method="highs")
    return np.asarray(res.x, dtype=float) if res.success else None


def cglp_cut(inst: MOILFP, a: np.ndarray, rows: Sequence[Row],
             ub: np.ndarray, x_star: np.ndarray,
             tol: float = 1e-7) -> Optional[Tuple[np.ndarray, float]]:
    """
    Programme generateur de coupes. Renvoie (alpha, beta) tel que
    alpha^T x >= beta soit valide pour la reunion des P_k et VIOLEE en
    x_star, ou None si aucune coupe violee n'existe.

    Le cout est d'UN programme lineaire. Sa taille : n + 1 + p(m+1)
    variables, p*n + p + 1 contraintes -- soit une centaine de chaque a
    n = 40, m = 21, p = 3.
    """
    from molfp_matheuristic import e_row

    n, p = inst.n, inst.p
    A, b = _rows_en_Ax_le_b(rows, n)
    # les bornes de boite font partie du polyedre : x <= u
    A = np.vstack([A, np.eye(n)]) if A.size else np.eye(n)
    b = np.concatenate([b, ub.astype(float)])
    m = A.shape[0]

    gam, dlt = [], []
    for k in range(p):
        g, d = e_row(inst, a, k)
        gam.append(np.asarray(g, dtype=float))
        dlt.append(float(d))

    # variables : alpha (n, libre) | beta (1, libre) | lambda^k (m) | mu^k (1)
    nv = n + 1 + p * (m + 1)
    def i_lam(k): return n + 1 + k * (m + 1)
    def i_mu(k): return n + 1 + k * (m + 1) + m

    A_ub, b_ub = [], []
    for k in range(p):
        # -alpha_j - (lambda^k A)_j + mu^k gamma_{k,j} <= 0
        for j in range(n):
            row = np.zeros(nv)
            row[j] = -1.0
            row[i_lam(k):i_lam(k) + m] = -A[:, j]
            row[i_mu(k)] = gam[k][j]
            A_ub.append(row)
            b_ub.append(0.0)
        # beta + lambda^k b - mu^k (1 - delta_k) <= 0
        row = np.zeros(nv)
        row[n] = 1.0
        row[i_lam(k):i_lam(k) + m] = b
        row[i_mu(k)] = -(1.0 - dlt[k])
        A_ub.append(row)
        b_ub.append(0.0)

    # normalisation : somme des multiplicateurs = 1 (sinon non borne)
    A_eq = np.zeros((1, nv))
    for k in range(p):
        A_eq[0, i_lam(k):i_lam(k) + m + 1] = 1.0

    c = np.zeros(nv)
    c[:n] = np.asarray(x_star, dtype=float)     # on MINIMISE alpha^T x* - beta
    c[n] = -1.0
    bounds = [(None, None)] * (n + 1) + [(0.0, None)] * (p * (m + 1))

    ORACLE_CALLS["lp"] += 1
    res = linprog(c=c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=np.array([1.0]), bounds=bounds,
                  method="highs")
    if not res.success:
        return None
    violation = -float(res.fun)                 # = beta - alpha^T x*
    if violation <= tol:
        return None
    alpha = np.asarray(res.x[:n], dtype=float)
    beta = float(res.x[n])
    return alpha, beta


def verifier_validite(inst: MOILFP, a: np.ndarray,
                      alpha: np.ndarray, beta: float,
                      points: Sequence[np.ndarray],
                      tol: float = 1e-6) -> int:
    """
    Compte les points de `points` qui VIOLENT alpha^T x >= beta alors qu'ils
    devraient la satisfaire, c'est-a-dire ceux qui verifient la disjonction
    (il existe k tel que e_k^a(x) >= 1). Doit valoir 0.

    C'est le controle de surete de la coupe : on ne peut pas se contenter de
    la derivation, il faut la confronter aux points que l'on pretend garder.
    """
    from molfp_matheuristic import e_row

    gam, dlt = [], []
    for k in range(inst.p):
        g, d = e_row(inst, a, k)
        gam.append(np.asarray(g, dtype=float))
        dlt.append(float(d))
    viol = 0
    for x in points:
        xv = np.asarray(x, dtype=float)
        dans_union = any(gam[k] @ xv + dlt[k] >= 1 - 1e-9
                         for k in range(inst.p))
        if dans_union and float(alpha @ xv) < beta - tol:
            viol += 1
    return viol



# ==========================================================================
# extrait de molfp_matheuristic.py
# ==========================================================================


TOURS_GEOM = 12


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

    UN ILP INTERROMPU RESTE EXPLOITABLE, et une premiere version le jetait.
    Elle exigeait `res.ok`, c'est-a-dire le statut 'optimal'. Or ce
    programme-ci est pose sur le modele de coupes complet, sans limite de
    temps, et il arrive DERNIER dans le tour : sous plafond deterministe le
    budget est deja epuise quand on l'appelle. Diagnostic sur neuf executions
    a n = 20, 30, 40 : `Dplus = None` sur les NEUF, aux deux plafonds. Le
    gain 2 du Th. 5' -- D+ au lieu de Dmin -- n'etait donc jamais realise en
    pratique, alors que l'article le compte parmi ses trois apports.

    Ce qu'il faut retenir d'un ILP interrompu depend du SENS de
    l'optimisation. Ici on MINIMISE : `res.bound` est le cote optimiste,
    c'est-a-dire un MINORANT valide de min D sur la region. Or c'est
    exactement ce dont le Th. 5' a besoin : la borne s'ecrit
    q + U/(Q*denom) et n'exige de `denom` que de minorer D(x) sur toute la
    region ou elle agit. Un minorant interrompu est donc aussi legitime que
    l'optimum, seulement moins fin -- et toujours meilleur que rien, puisque
    l'appelant prend max(Dmin, D+), maximum de deux minorants valides.
    """
    extra = [(np.asarray(w, dtype=float), -float(w0), INF)]
    res = model.optimize(inst.f.den.astype(float), float(inst.f.b),
                         maximize=False, extra_rows=extra)
    if res.status == "infeasible":
        return None
    val = res.obj if res.ok else res.bound
    if val is None or not np.isfinite(val):
        return None
    # minorant : on arrondit VERS LE BAS, jamais vers le haut -- un `denom`
    # surestime rendrait la borne fausse.
    return max(1, int(np.floor(val + 1e-9)))


def borne_geometrique(inst: MOILFP, model: ECutModel, q: Fraction,
                      budget: float, max_iter: int = TOURS_GEOM,
                      Dm: Optional[int] = None
                      ) -> Tuple[Optional[float], str, int]:
    """
    LE MEME TH. 5 PRIME, EVALUE A UN MEILLEUR SEUIL.

    Ce que fait la certification aujourd'hui. Elle pose q = P/Q, resout
    U = max_R F_q avec F_q = Q N - P D, et conclut

        q* <= q + U / (Q Dmin).                                        (5')

    Le seuil est donc TOUJOURS l'incumbent. Or (5') n'a rien de particulier
    a l'incumbent : pour tout seuil t, en posant G_t = N - t D,

        max_R G_t <= v   =>   pour tout x de R, N(x) - t D(x) <= v
                         =>   f(x) <= t + v / D(x) <= t + v / Dmin,

    donc  max_R f <= t + v/Dmin, et (5') est le cas t = q. Rien n'oblige a
    s'arreter la : chaque solution x de R rencontree en chemin fournit un
    seuil t = f(x) STRICTEMENT MEILLEUR, et l'on peut recommencer. C'est un
    Dinkelbach ordinaire, mais mene SUR LE RELACHE et non sur S.

    Pourquoi la methode ne le faisait pas. La boucle de certification
    n'avance son seuil que sur des points certifies EFFICACES -- c'etait un
    reflexe correct pour le LB, ou seul un point efficace peut compter. Mais
    pour une BORNE SUPERIEURE l'efficacite est inutile : n'importe quel point
    du relache fait un seuil valide. Le refus d'avancer le seuil sur un point
    non certifie coutait donc tout l'ecart entre q et max_R f, sans rien
    acheter.

    VALIDITE. A la convergence (max_R G_t <= 0) on a max_R f <= t. Les coupes
    de dominance preservent E ; chaque coupe d'efficacite n'est posee qu'apres
    etablissement de sa condition de cloture, donc aucun point efficace retire
    ne bat l'incumbent. D'ou

        q* <= max(q, max_R f).

    Interrompue, la procedure reste valide : `res.bound` majore max_R G_t
    meme quand le solveur n'a pas conclu, et chaque tour produit un candidat
    valide ; on renvoie le plus petit.

    INCOMPARABLE a (5') telle qu'appliquee aujourd'hui, et non superieure :
    (5') utilise D+ >= Dmin, plus fin, et la structure du test d'efficacite.
    Les deux se prennent donc au MINIMUM, ce qui reste valide.

    Renvoie (borne, statut, tours) ; statut 'optimal' quand max_R f est
    atteint, 'empty' quand R est vide (alors q* = q), 'limit' sinon.
    """
    f = inst.f
    if Dm is None:
        Dm = d_min(inst)
    t = Fraction(q)
    best: Optional[float] = None
    fin = time.time() + budget
    tours = 0

    for it in range(max_iter):
        reste = fin - time.time()
        if reste <= 0.02 or ilp_budget_left() <= 0:
            break
        # Un seul ILP ne doit pas pouvoir consommer tout le budget : le temps
        # restant est reparti sur les tours restants. Sans cela, la premiere
        # relaxation d'une instance a n = 20 tourne des heures et la procedure
        # n'est plus « anytime » que de nom -- c'est exactement ce qui s'est
        # produit au premier essai, avec un time_limit herite de 10^6 s.
        par_appel = reste / max(1, max_iter - it)
        tours += 1
        P, Q = t.numerator, t.denominator
        coef = (Q * f.num - P * f.den).astype(float)
        const = float(Q * f.a - P * f.b)
        res = model.optimize(coef, const, maximize=True, time_limit=par_appel)

        if res.status == "infeasible":
            # R vide : plus aucun point non retire, donc q* <= t.
            return float(t), "empty", tours
        if res.bound is None or not np.isfinite(res.bound):
            break

        U = max(0.0, float(res.bound))          # majorant VALIDE de max_R F_t
        cand = float(t) + U / (Q * Dm)
        best = cand if best is None else min(best, cand)
        if U <= 1e-9:
            return float(t), "optimal", tours          # max_R f <= t, exactement

        # avancer le seuil sur le point trouve : aucune efficacite requise
        if res.x is None or res.obj is None or res.obj <= 1e-9:
            break
        x = np.rint(np.asarray(res.x)[:inst.n]).astype(int)
        if f.denominator(x) <= 0:
            break
        t_new = f.value(x)
        if t_new <= t:
            break
        t = t_new

    return best, "limit", tours


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
                   w: np.ndarray,
                   alterne: bool = True) -> List[tuple]:
    """
    UN SEUL vivier de coupes, alimente par DEUX sources.

    Les deux coupes ont exactement la meme forme -- la disjonction
    "il existe k tel que e_k(x) >= 1" -- donc le meme cout : p binaires
    chacune. Seule leur precondition differe (Th. 4 pour un point domine,
    Th. 6 pour un point efficace). Il n'y a donc aucune raison de leur
    allouer des budgets separes. Nous invoquions ici une mesure -- poser 40
    coupes d'archive EN PLUS des 40 de dominance aurait fait chuter
    l'optimalite prouvee de 84 a 57 sur 90 instances. CE CHIFFRE EST RETIRE.
    Ce fichier en portait 57 et l'article 56 pour la meme experience ; il a
    ete obtenu sous budget en TEMPS, qui varie de 125 % d'une execution a
    l'autre ; et le chemin de code correspondant avait ete supprime.

    Remesure sous plafond DETERMINISTE (bench_budgets.py, 18 instances,
    trois graines) : ecart median 27,6 % contre 27,4 %, optimalite prouvee
    5/18 DANS LES DEUX BRAS, mieux 3 / pire 3 / egal 12. Aucun effondrement,
    alors que le bras a budgets separes pose jusqu'a 60 % de coupes en plus
    sur cinq lignes. La regle du plafond unique est conservee pour sa
    simplicite, non sur la foi d'une mesure qui ne se reproduit pas.

    On classe donc les candidats des deux sources ensemble, par valeur
    decroissante du substitut : ce sont ceux qui tirent U vers le haut, quelle
    que soit leur origine. Le plafond global reste celui qui a ete regle.

    LE CLASSEMENT COMMUN AFFAME L'ARCHIVE, ET C'EST MESURE. Nous avons
    ecrit que l'effet du vivier commun etait « automatique et souhaitable :
    la ou les points domines abondent, ils occupent le vivier ; la ou ils
    sont rares, l'archive le remplit a leur place ». Le diagnostic du banc
    de budget dit le contraire. En triplant le plafond -- donc en allongeant
    la recherche, donc en recoltant plus de points domines -- sur six
    executions a n = 20, 30, 40 :

        archive        39->80   26->50   24->52   50->99   19->36   29->63
        coupes totales 105->167 115->176 101->161 107->161 90->171  95->240
        coupes ARCHIVE  28->12   26->0    20->2    33->10   16->0    22->15

    L'archive double, les coupes posees augmentent, et les coupes D'ARCHIVE
    s'effondrent -- jusqu'a ZERO sur deux lignes. Or ce sont les SEULES qui
    peuvent vider le relache (Prop. du relache vide) ; les coupes de
    dominance, elles, preservent E par construction et ne le videront
    jamais. Les affamer, c'est renoncer a la seule preuve forte.

    L'ecart garanti suit : il EMPIRE sur cinq de ces six lignes quand on
    triple le budget. C'est le mecanisme de la non-monotonie en budget.

    LA FAUTE N'EST PAS LE PLAFOND UNIQUE, C'EST L'ECHELLE UNIQUE. Les deux
    sources sont classees sur le meme critere, w^T x, alors que leurs
    CAPACITES different en nature : une coupe de dominance retrecit R, une
    coupe d'efficacite peut le VIDER. Trier sur une seule echelle traite
    comme equivalent ce qui ne l'est pas.

    LA CORRECTION, SANS PARAMETRE NOUVEAU. On classe a l'interieur de
    chaque source, puis on ALTERNE. Aucune fraction a regler : l'alternance
    donne la moitie des places a chaque source quand les deux ont des
    candidats, et la TOTALITE a celle qui reste quand l'autre est epuisee --
    l'effet « automatique » que le vivier commun promettait sans le tenir.

    Renvoie une liste de couples (point, origine) avec origine dans
    {"dom", "arch"}.
    """
    wv = np.asarray(w, dtype=float)
    seen, par_source = set(), {"dom": [], "arch": []}
    for pts, kind in ((dominated, "dom"), (archive or [], "arch")):
        for x in pts:
            key = (kind, tuple(int(v) for v in x))
            if key in seen:
                continue
            seen.add(key)
            par_source[kind].append((np.asarray(x, dtype=int), kind))
    for k in par_source:
        par_source[k].sort(key=lambda t: -float(wv @ t[0]))

    if not alterne:
        pool = par_source["dom"] + par_source["arch"]
        pool.sort(key=lambda t: -float(wv @ t[0]))
        return pool

    d, a = par_source["dom"], par_source["arch"]
    pool, i, j = [], 0, 0
    while i < len(d) or j < len(a):
        if i < len(d):
            pool.append(d[i])
            i += 1
        if j < len(a):
            pool.append(a[j])
            j += 1
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
                max_lp: Optional[int] = None,
                alterne: bool = True) -> Tuple[List[tuple], dict]:
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
    pool = build_cut_pool(rank_dominated(dominated, w), archive or [], w,
                          alterne=alterne)
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
                           dominated_out: Optional[List[np.ndarray]] = None,
                           steriles_max: int = 2
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

    SONDE DE PRODUCTIVITE. La mesure a n = 20 a 50 est nette : quand la
    diversification produit, elle produit beaucoup -- une instance passe de
    q_lb = 2,90 a 7,84 (+170 %) avec 36 points neufs. Quand elle ne produit
    pas, elle ne produit RIEN : les deux instances en recul de la campagne
    ont toutes deux zero point neuf, et leur perte est exactement le budget
    pris a la certification. Le mecanisme est donc bon et c'est la POLITIQUE
    d'activation qui coutait. On abandonne apres `steriles_max` tours sans
    aucun point neuf, et le budget non consomme retourne a l'appelant.

    Renvoie (q, x_best, nombre de points neufs certifies).
    """
    t0 = time.time()
    model = ECutModel(inst)
    for a in archive.points()[:cap]:
        model.add_efficiency_cut(a)          # aucune cloture a etablir
    neufs = 0
    steriles = 0
    while time.time() - t0 < budget and steriles < steriles_max:
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
            steriles = 0
        else:
            steriles += 1        # deja connu : ce tour n'a rien produit
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
            agg_extra: int = 0,
            cglp_extra: int = 0,
            geom_bound: bool = False,
            geom_share: float = 0.25,
            geom_gate: bool = False,
            geom_gap: float = 0.5,
            pool_alterne: bool = True,
            budgets_separes: bool = False) -> CertResult:
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
                  "select": None, "cloture_lemme": 0, "agg_cuts": 0,
                  "cglp_cuts": 0, "geom_ub": None, "geom": None,
                  "geom_ilp": 0, "geom_tours": 0}

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

    # -- reservation pour la SECONDE ROUTE, DECIDEE AU BON MOMENT ---------
    # Elle coute des ILP comme tout le reste : ce qu'on lui donne, la boucle
    # de coupes ne l'a pas. Lui laisser le simple reliquat ne marche pas --
    # la boucle consomme le plafond en entier et la seconde route ne
    # s'execute jamais, exactement la panne deja mesuree sur la
    # diversification. Mais reserver EN AVEUGLE est perdant la ou la
    # premiere route ferme deja : mesure, -0,66 point sur une instance ou
    # elle suffisait.
    #
    # On reserve donc APRES LE PREMIER TOUR, quand la premiere route a
    # produit sa premiere borne et que l'ecart est donc CONNU. Le seuil est
    # le meme que celui qui sert deja ailleurs a declarer une borne qui ne
    # ferme pas. La decision se prend ainsi sur une mesure, non sur un pari,
    # et elle se prend assez tot pour que la reservation ait un sens.
    # LA RESERVATION DOIT PRECEDER LA BOUCLE. Une version l'a placee apres le
    # premier tour, pour decider sur une mesure plutot que sur un pari. Elle
    # ne pouvait pas fonctionner : le premier tour consomme a lui seul la
    # TOTALITE du plafond -- l'oracle interne tourne jusqu'a epuisement --
    # de sorte qu'a la fin du tour 1 il ne reste rien a reserver. Mesure du
    # symptome : statut 'limit', zero tour, zero appel, gain +0,00 la ou la
    # reservation en amont gagnait 9,82 points.
    #
    # On reserve donc EN AMONT, et le declencheur devient une LIBERATION :
    # si le premier tour montre que la premiere route ferme convenablement,
    # la part reservee est RENDUE et le tour suivant en profite. La decision
    # reste prise sur une mesure ; c'est le sens de la reservation qui
    # s'inverse, pas le moment de la decision.
    _bud_ext = ORACLE_BUDGET["ilp"]
    budget_coupes = budget
    geom_actif = geom_bound
    reserve_faite = not geom_bound      # rien a liberer si la route est off

    def _reserver() -> None:
        nonlocal budget_coupes
        budget_coupes = (time.time() - t0) + (1.0 - geom_share) * \
            max(0.0, budget - (time.time() - t0))
        if _bud_ext is not None:
            libre = max(0, _bud_ext - ORACLE_CALLS["ilp"])
            # On ne reserve pas une PART, on reserve ce que la route peut
            # effectivement depenser : au plus TOURS_GEOM appels. Une reserve
            # en pourcentage immobilisait 28 appels pour une route qui en
            # consomme 2, et les retirait aux coupes sans contrepartie.
            garde = min(int(geom_share * libre), TOURS_GEOM)
            set_ilp_budget(ORACLE_CALLS["ilp"] + max(1, libre - garde))

    def _liberer() -> None:
        """
        REFUTE PAR SON PROPRE BANC, et desactive par defaut.

        L'idee etait de ne pas financer la seconde route la ou la premiere
        ferme deja. Elle ne peut pas tenir, pour une raison de sequence : la
        reserve est prise AVANT la boucle -- elle doit l'etre, le premier
        tour consommant a lui seul tout le plafond -- et le premier tour a
        donc DEJA tourne sous le plafond reduit quand la liberation se
        decide. Rendre la part ne rend pas les coupes que ce tour n'a pas
        posees.

        Mesure appariee, douze lignes, plafond 300 :

                              ameliorees  degradees  inchangees   median
          sans declencheur         8          1          3        +3,19
          avec declencheur         6          2          4        +0,03

        Le detail est plus net que le resume. Sur n=30, corr=0, graine 2, le
        declencheur refuse d'engager la route -- et la ligne perd tout de
        meme 0,26 point, contre un GAIN de 6,04 sans lui. Il paie le cout et
        refuse le benefice. Sur n=20, corr=0, graine 2, il abandonne +0,33.

        Le code reste, desactive : un mecanisme dont on a mesure qu'il nuit
        se documente mieux en place qu'efface.
        """
        nonlocal budget_coupes, geom_actif
        budget_coupes = budget
        geom_actif = False
        set_ilp_budget(_bud_ext)
        info["geom"] = "non engagee"

    if geom_bound:
        _reserver()

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
                                       cap=2 * cut_batch if budgets_separes
                                       else cut_batch,
                                       alterne=pool_alterne)
            info["select"] = sel
        else:
            w0_coef, _, _, _ = surrogate(inst, q)
            pending = [(x, k, None) for x, k in
                       build_cut_pool(rank_dominated(dominated, w0_coef),
                                      arch_pts, w0_coef,
                                      alterne=pool_alterne)]

    for rnd in range(1, max_rounds + 1):
        left = budget_coupes - (time.time() - t0)
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
            # BUDGETS SEPARES : chaque source recoit son propre plafond, au
            # lieu de les mettre en concurrence sous un plafond unique. C'est
            # la variante que la regle du vivier unique rejette ; elle est
            # conservee pour que ce rejet soit MESURE et non postule.
            cap_tour = 2 * cut_batch if budgets_separes else cut_batch
            for cand in pending:
                x, kind = cand[0], cand[1]
                h = cand[2] if len(cand) > 2 else None
                if posees >= cap_tour or time.time() > t0 + budget_coupes:
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
                if n_agg >= agg_extra or time.time() > t0 + budget_coupes:
                    break
                xa, ka = cand[0], cand[1]
                ha = cand[2] if len(cand) > 2 else None
                if ka == "arch" and not (ha is not None and ha <= 0):
                    continue
                if model.add_aggregated_cut(xa):
                    n_agg += 1
        # --- au-dela du plafond : coupes DISJONCTIVES SANS BINAIRE -----
        # Programme generateur (lift-and-project) : une ligne en espace x,
        # aucune binaire, donc aucune place consommee sous le plafond. C'est
        # le seul levier disponible dans le regime ou le plafond sature.
        # On s'en tient aux candidats dont le retrait est licite sans frais :
        # points domines (aucune cloture requise) et points d'archive dont le
        # lemme de hauteur a deja etabli la cloture.
        if cglp_extra > 0 and pending:
            xs = optimum_relaxation(w, model._rows_x, model.ub_x)
            n_cg = 0
            for cand in pending:
                if n_cg >= cglp_extra or xs is None \
                        or time.time() > t0 + budget_coupes:
                    break
                xc, kc = cand[0], cand[1]
                hc = cand[2] if len(cand) > 2 else None
                if kc == "arch" and not (hc is not None and hc <= 0):
                    continue
                cut = cglp_cut(inst, xc, model._rows_x, model.ub_x, xs)
                if cut is None:
                    continue
                model.add_linear_row(cut[0], cut[1], INF)
                n_cg += 1
                xs = optimum_relaxation(w, model._rows_x, model.ub_x)
            info["cglp_cuts"] = model.n_lin_rows

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
            set_ilp_budget(_bud_ext)   # part reservee rendue
            return CertResult(float(q), True, q, x_cur, info)

        # -- optimalite prouvee : F(q) resolu et <= 0 ----------------------
        if r.status == "optimal" and r.value is not None and r.value <= 1e-9 \
                and not improved:
            set_ilp_budget(_bud_ext)   # part reservee rendue
            return CertResult(float(q), True, q, x_cur, info)

        # -- borne du Th. 5 / 5' -------------------------------------------
        # une UB infinie (oracle interrompu avant sa premiere relaxation)
        # n'est pas une borne : mieux vaut ne rien annoncer qu'annoncer inf.
        if r.ub is not None and np.isfinite(r.ub):
            U = max(0.0, float(r.ub))
            info["U"] = U
            if U <= 1e-9 and not improved:
                set_ilp_budget(_bud_ext)   # part reservee rendue
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

        # -- decision de LIBERER, a la fin du premier tour ------------------
        # Hors du bloc de la borne, et non a l'interieur : un premier tour
        # peut ne produire AUCUNE borne (oracle interrompu avant sa premiere
        # relaxation, `r.ub` vaut None), et ce cas doit etre traite.
        if not reserve_faite:
            reserve_faite = True
            # Une absence de borne vaut borne qui ne ferme pas : on garde la
            # reserve. On ne la rend que sur une borne EFFECTIVEMENT serree.
            ecart_1 = 1.0 if best_ub is None else \
                (best_ub - float(q)) / max(1e-12, abs(best_ub))
            if geom_gate and ecart_1 <= geom_gap:
                # la premiere route ferme : lui prendre des coupes pour
                # financer la seconde serait un troc perdant.
                _liberer()

        # Un tour sans amelioration, sans coupe en reserve ET dont l'oracle
        # a conclu se repeterait a l'identique : on s'arrete.
        # En revanche un oracle INTERROMPU ('limit') laisse du travail : il a
        # pose ses propres coupes dans `model`, donc R s'est resserre et le
        # tour suivant repart d'une relaxation strictement meilleure.
        if not improved and not pending and r.status != "limit":
            break

    # -- SECONDE ROUTE : le meme Th. 5' a un seuil avance sur le relache --
    # Les deux routes sont incomparables (la premiere utilise D+ et la
    # structure du test d'efficacite, la seconde un meilleur seuil), et le
    # minimum de deux bornes valides est une borne valide.
    if geom_actif and not proved:
        set_ilp_budget(_bud_ext)               # on rend la part reservee
        reste = budget - (time.time() - t0)
        if reste > 0.05:
            avant = ORACLE_CALLS["ilp"]
            g, st, tours = borne_geometrique(inst, model, q, reste, Dm=Dm)
            info["geom"] = st
            info["geom_ub"] = g
            info["geom_ilp"] = ORACLE_CALLS["ilp"] - avant
            info["geom_tours"] = tours
            if g is not None:
                best_ub = g if best_ub is None else min(best_ub, g)

    if geom_bound:
        set_ilp_budget(_bud_ext)

    if best_ub is not None and best_ub <= float(q) + 1e-12:
        proved = True
        best_ub = float(q)
    return CertResult(best_ub, proved, q, x_cur, info,
                      cut_points=list(model.cut_points))


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
             w: np.ndarray, w0: float,
             keep: Optional[Sequence[int]] = None) -> Optional[np.ndarray]:
    """
    Mouvement C.  Fige x_j = xr_j hors de `free_idx`, resout le reste.

    `keep` : criteres sur lesquels le point produit ne doit RIEN perdre par
    rapport a la base, au sens e_j(x) >= 0. Sans eux, le mouvement maximise
    librement le substitut de f -- et le substitut de f est maximal sur des
    points DOMINES. Le detail compte, parce qu'il explique un rendement.

    CE QUE LA MESURE REPROCHE A LA VERSION LIBRE. Sur douze instances,
    4 succes pour 316 tentatives, soit 1,3 % -- trois fois moins que le
    mouvement A, pour pres d'un cinquieme du budget de recherche. Le
    diagnostic tient en une phrase deja etablie ailleurs dans ce travail :
    les points de f eleve sont domines, donc hors de E, et leurs voisins
    efficaces ont reellement un f faible. La reparation defait donc ce que
    le mouvement vient de gagner.

    A et B n'ont pas ce defaut parce qu'ils CONTRAIGNENT les criteres --
    e_k(x) >= 1 pour A, Z_k(x) >= eps_k pour B -- et atterrissent donc pres
    de la frontiere efficace. Ancrer C de la meme facon n'est pas un reglage
    de plus : c'est lui donner ce qui fait marcher les deux autres.
    """
    lb = np.array(xr, dtype=float)
    ub = np.array(xr, dtype=float)
    box = inst.var_upper_bounds().astype(float)
    for j in free_idx:
        lb[j], ub[j] = 0.0, box[j]
    from molfp_core import solve_milp
    rows = [(inst.A[i].astype(float), -INF, float(inst.b[i]))
            for i in range(inst.m)]
    for j in (keep or []):
        cj, kj = e_row(inst, xr, j)
        rows.append((cj, -kj, INF))
    res = solve_milp(w, rows, lb, ub, maximize=True, obj_const=w0)
    return res.x if res.ok else None


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
                   cglp_extra: int = 0,
                   geom_bound: bool = False,
                   geom_gate: bool = False,
                   pool_alterne: bool = True,
                   budgets_separes: bool = False,
                   mouvement_c: str = "libre",
                   cut_diversify: bool = True,
                   gap_hopeless: float = 0.5,
                   ilp_budget: Optional[int] = None,
                   ilp_search_share: float = 0.6,
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

    # --- budget DETERMINISTE, en nombre d'appels au solveur entier --------
    # Quand `ilp_budget` est donne, c'est LUI qui arbitre : les budgets de
    # temps sont laisses larges et l'horloge sort de la boucle de decision.
    # Une execution devient alors reproductible, et une difference entre deux
    # variantes est un effet, non un alea. Le plafond est reparti entre les
    # deux phases dans la meme proportion que le temps, sans quoi la
    # recherche consommerait tout et la certification n'aurait rien.
    _budget_exterieur = ORACLE_BUDGET["ilp"]
    if ilp_budget is not None:
        set_ilp_budget(calls0 + max(1, int(ilp_search_share * ilp_budget)))

    # --- amorcage : un point efficace quelconque --------------------------
    dominated: List[np.ndarray] = []      # recyclage pour le Th. 5' (gain 3)
    # Points dont la chaine de reparation n'a pas abouti : statut INCONNU.
    # Le Th. 1 generalise en fait des centres de coupe legitimes ; on les
    # collecte d'abord pour savoir COMBIEN il y en a, avant de decider s'il
    # vaut la peine de les exploiter.
    indetermines: List[np.ndarray] = []
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
        if mouvement_c == "sans":
            menu = [m for m in menu if m != "C"] or ["A"]

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
                    keep_c = None
                    if mouvement_c == "ancre":
                        # meme regle que le mouvement A : un sous-ensemble
                        # STRICT, sinon le voisinage ne contient que les
                        # points qui dominent la base, donc rien.
                        autres = [j for j in range(inst.p)]
                        n_keep = int(rng.integers(0, max(1, len(autres))))
                        keep_c = list(rng.choice(autres, size=n_keep,
                                                 replace=False)) \
                            if n_keep else []
                    y = move_lns(inst, xr, free, w, w0, keep=keep_c)

                n_moves[mv] += 1
                if y is None:
                    continue
                y = repair_to_efficient(inst, y,       # certification Th. 2
                                        dominated_out=dominated,
                                        deadline=t0 + time_budget,
                                        indetermine_out=indetermines)
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
    if ilp_budget is not None:
        set_ilp_budget(calls0 + ilp_budget)   # le reste va a la certification

    # --- phase 2 : certification (Th. 5') --------------------------------
    # le budget de recherche non consomme est reverse ici : la recherche a
    # conclu, la certification non.
    leftover = max(0.0, time_budget - search_time) if reallocate else 0.0
    budget = bound_budget + leftover

    q_ub, proved, cert_info = None, False, {}
    cut_points: List[np.ndarray] = []
    n_neufs = 0

    def _cert(bud: float, rounds: int, geom: Optional[bool] = None):
        return certify(inst, q, x_best, dominated, bud,
                       use_tightened=tightened, archive=arch,
                       cut_batch=cut_batch, max_rounds=rounds,
                       archive_cuts=archive_cuts, closure_lemma=closure_lemma,
                       lemma_strikes=lemma_strikes, height_rank=height_rank,
                       agg_extra=agg_extra, cglp_extra=cglp_extra,
                       geom_bound=geom_bound if geom is None else geom,
                       geom_gate=geom_gate, geom_gap=gap_hopeless,
                       pool_alterne=pool_alterne,
                       budgets_separes=budgets_separes)

    # --- repartition du plafond d'APPELS entre les phases -----------------
    # Sans elle, la sonde -- dont la regle d'arret est TEMPORELLE -- consomme
    # la totalite des appels restants, et la diversification ne s'execute
    # jamais. Le banc deterministe l'a montre sans ambiguite : `neufs` valait
    # 0 sur les 23 lignes, non parce que la diversification echouait mais
    # parce qu'elle ne tournait pas. Un plafond en appels exige donc une
    # repartition en appels ; melanger les deux unites ne mesure rien.
    def _cap(fraction: float) -> None:
        """Plafonne la phase suivante a `fraction` des appels encore libres."""
        if ilp_budget is None:
            return
        utilises = ORACLE_CALLS["ilp"] - calls0
        reste = max(0, ilp_budget - utilises)
        set_ilp_budget(calls0 + utilises + max(1, int(fraction * reste)))

    if certify_bound and budget > 0:
        sonde = None
        # SONDE. Un seul tour de certification, pour SAVOIR si la borne est
        # en train de se fermer, au lieu de le supposer. Toute borne obtenue
        # reste valide : la sonde ne peut donc rien gater, et quand elle
        # repond non elle fait gagner tout le reste du budget.
        if cut_diversify and (leftover > 0.3 or ilp_budget is not None):
            part = min(0.3 * budget, leftover) if leftover > 0.3 else budget
            _cap(0.3)
            sonde = _cert(part, 1, geom=False)
            if sonde.q_lb is not None and sonde.q_lb > q:
                q, x_best = sonde.q_lb, sonde.x_best
            q_ub, proved = sonde.q_ub, sonde.proved
            cert_info, cut_points = sonde.info, sonde.cut_points
            budget -= part
            ecart = None if q_ub is None else \
                (q_ub - float(q)) / max(1e-12, abs(q_ub))
            _diag = {"sonde_prouve": bool(proved), "sonde_ecart": ecart,
                     "ilp_apres_sonde": ORACLE_CALLS["ilp"] - calls0}
            # borne qui ne ferme pas : a n >= 20 la campagne mesure 92-98 %
            # quoi qu'on fasse. Le temps restant vaut alors plus a la
            # RECHERCHE qu'a la certification, et la coupe d'efficacite sert
            # ici d'operateur de diversification -- sans cloture a etablir.
            if not proved and (ecart is None or ecart > gap_hopeless):
                part_div = 0.6 * budget
                t_div = time.time()
                _cap(0.6)
                _diag["divers_tentee"] = True
                _diag["ilp_avant_divers"] = ORACLE_CALLS["ilp"] - calls0
                q, x_best, n_neufs = diversify_over_archive(
                    inst, arch, q, x_best, part_div, cut_batch, dominated)
                _diag["ilp_apres_divers"] = ORACLE_CALLS["ilp"] - calls0
                # on ne retire que ce qui a ETE CONSOMME. La sonde de
                # productivite interrompt la diversification des qu'elle
                # tourne a vide ; sans cette ligne le budget ainsi libere
                # serait perdu au lieu de revenir a la certification -- ce
                # qui etait exactement la perte mesuree sur les instances a
                # zero point neuf.
                budget -= min(part_div, time.time() - t_div)

        if ilp_budget is not None:
            set_ilp_budget(calls0 + ilp_budget)   # le reste a la certification
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
    try:
        cert_info["diag"] = _diag
    except NameError:
        cert_info["diag"] = None
    cert_info["cert_budget"] = budget
    cert_info["restarts"] = n_restarts
    cert_info["indetermines"] = len(indetermines)
    cert_info["moves"] = dict(n_moves)
    cert_info["hits"] = dict(n_hits)

    if ilp_budget is not None:
        set_ilp_budget(_budget_exterieur)      # on rend le plafond appelant

    return MatheurResult(
        q_lb=q, x_best=x_best, q_ub=q_ub, archive=arch.points(),
        ilp_calls=ORACLE_CALLS["ilp"] - calls0, time=time.time() - t0,
        rounds=rounds, proved_optimal=proved,
        status="optimal" if proved else "heuristic", cert=cert_info,
        cut_points=cut_points,
    )



# ==========================================================================
# extrait de molfp_enum.py
# ==========================================================================


def enumerate_feasible(inst: MOILFP, limit: int = 2_000_000) -> List[np.ndarray]:
    """
    Enumere tous les points entiers de S = {x in Z^n_+ : Ax <= b}.

    ELAGAGE, ET LE BOGUE QU'IL A CACHE. Une premiere version elaguait ainsi :
    des qu'une somme partielle A[:, :j] @ x[:j] depassait b, elle coupait le
    sous-arbre et sortait de la boucle sur la valeur de x_j. Ce raisonnement
    suppose A >= 0 -- les sommes partielles ne peuvent alors que croitre. Le
    generateur garantit cette hypothese (A2), donc rien ne s'est jamais vu
    sur les instances tirees.

    Mais les instances de la LITTERATURE ont des coefficients NEGATIFS. Une
    variable ultERIEURE peut alors faire REDESCENDRE une ligne au-dessous de
    b, et l'elagage retire des points REALISABLES. Sur l'exemple de Drici,
    Ouail & Moulai (2018), il retirait tous les points a x1 = 2, dont
    (2,3,0,0) -- l'un des deux optima publies. La reference elle-meme etait
    donc fausse, precisement sur les instances servant a la validation
    externe.

    Elagage CORRECT. Pour chaque ligne i, on minore la contribution des
    variables encore libres par sum_{k >= j} min(0, A[i,k]) * ub[k]. Ajoutee
    a la somme partielle, elle donne le plus petit membre de gauche encore
    atteignable : si celui-ci depasse deja b, le sous-arbre est reellement
    infaisable. Et l'on ne sort de la boucle sur x_j (`break`) que si la
    colonne est >= 0 ; sinon on passe a la valeur suivante (`continue`), une
    valeur plus grande pouvant redevenir realisable.
    """
    n, m = inst.n, inst.m
    ub = inst.var_upper_bounds()
    A, b = inst.A, inst.b

    # contribution MINIMALE encore atteignable par les variables j..n-1
    neg = np.minimum(A, 0) * ub                       # m x n
    suffix_min = np.zeros((n + 1, m), dtype=int)
    for j in range(n - 1, -1, -1):
        suffix_min[j] = suffix_min[j + 1] + neg[:, j]

    out: List[np.ndarray] = []
    x = np.zeros(n, dtype=int)

    def rec(j: int, partial: np.ndarray) -> None:
        if len(out) > limit:
            raise MemoryError(f"Plus de {limit} points realisables : instance "
                              f"trop grande pour l'enumeration exhaustive.")
        if j == n:
            if np.all(partial <= b):
                out.append(x.copy())
            return
        col = A[:, j]
        montante = bool(np.all(col >= 0))
        for v in range(ub[j] + 1):
            new_partial = partial + v * col
            if np.any(new_partial + suffix_min[j + 1] > b):
                if montante:
                    break          # colonne >= 0 : les v suivants sont pires
                continue           # sinon un v plus grand peut redevenir bon
            x[j] = v
            rec(j + 1, new_partial)
        x[j] = 0

    rec(0, np.zeros(m, dtype=int))
    return out


def dominates(u: Tuple[Fraction, ...], v: Tuple[Fraction, ...]) -> bool:
    """u domine v (maximisation) : u >= v composante par composante, u != v."""
    ge = all(ui >= vi for ui, vi in zip(u, v))
    return ge and u != v


def pareto_filter(vectors: List[Tuple[Fraction, ...]]) -> List[int]:
    """Indices des vecteurs non domines. Comparaison exacte (Fractions)."""
    uniq = sorted(set(vectors), reverse=True)     # tri lexicographique decroissant
    nd_set = []
    for v in uniq:
        # un vecteur ne peut etre domine que par un vecteur lexico-superieur,
        # donc deja traite et conserve dans nd_set
        if not any(dominates(u, v) for u in nd_set):
            nd_set.append(v)
    nd = set(nd_set)
    return [i for i, v in enumerate(vectors) if v in nd]


@dataclass
class GroundTruth:
    S: List[np.ndarray]                       # tous les points realisables
    E: List[np.ndarray]                       # ensemble efficace exact
    ZE: List[Tuple[Fraction, ...]]            # vecteurs criteres des points de E
    q_star: Fraction                          # max f sur E   <-- reference de (P)
    x_star: np.ndarray                        # un argmax
    q_max_S: Fraction                         # max f sur S   <-- borne sup valide
    ideal: Tuple[Fraction, ...]
    nadir: Tuple[Fraction, ...]               # vrai nadir (calcule sur E)

    def summary(self) -> str:
        return (f"|S| = {len(self.S):>7}   |E| = {len(self.E):>6}   "
                f"|E|/|S| = {len(self.E)/max(1,len(self.S)):.4f}\n"
                f"q*      = {float(self.q_star):.6f}  (max f sur E)\n"
                f"max_S f = {float(self.q_max_S):.6f}  (borne sup, relaxation)\n"
                f"ecart relaxation = "
                f"{float((self.q_max_S - self.q_star)/abs(self.q_max_S))*100:.2f} %")


def ground_truth(inst: MOILFP, limit: int = 2_000_000) -> GroundTruth:
    """Calcule S, E, q* et les points ideal/nadir par force brute exacte."""
    S = enumerate_feasible(inst, limit=limit)
    Zvals = [inst.criteria(x) for x in S]

    idx_E = pareto_filter(Zvals)
    E = [S[i] for i in idx_E]
    ZE = [Zvals[i] for i in idx_E]

    fS = [inst.f.value(x) for x in S]
    q_max_S = max(fS)

    fE = [inst.f.value(x) for x in E]
    j = int(np.argmax([float(v) for v in fE]))
    # argmax exact (evite les egalites mal tranchees en flottant)
    best = max(fE)
    j = fE.index(best)

    p = inst.p
    ideal = tuple(max(z[k] for z in Zvals) for k in range(p))
    nadir = tuple(min(z[k] for z in ZE) for k in range(p))

    return GroundTruth(S=S, E=E, ZE=ZE, q_star=best, x_star=E[j],
                       q_max_S=q_max_S, ideal=ideal, nadir=nadir)


def as_key(x: np.ndarray) -> Tuple[int, ...]:
    return tuple(int(v) for v in x)

# ============================================================================
# LES DEUX INSTANCES DE L'ILLUSTRATION
# ============================================================================

def instance_I2() -> MOILFP:
    """
    max Z1 = (3 x2 + 2) / (x1 + x2 + 3),   Z2 = (x1 + 3 x2) / (2 x2 + 3)
    max f  = (3 x1 + x2) / (2 x2 + 1)   sur E
    s.c.   3 x1 +   x2 <= 14
           2 x1 + 3 x2 <= 15,   x entier >= 0

    |S| = 22, |E| = 5, q* = 14/5 en (4,2). Le maximum de f sur S vaut 12, en
    (4,0) -- un point DOMINE : borner q* par max_S f se trompe de 77 %.
    """
    return MOILFP(
        A=np.array([[3, 1], [2, 3]]), b=np.array([14, 15]),
        Z=[FracObj(np.array([0, 3]), 2, np.array([1, 1]), 3),
           FracObj(np.array([1, 3]), 0, np.array([0, 2]), 3)],
        f=FracObj(np.array([3, 1]), 0, np.array([0, 2]), 1),
        name="I2", seed=0)


def instance_I3() -> MOILFP:
    """
    max Z1 = 3 x3 / (2 x2 + x3 + 2)
        Z2 = (3 x1 + 3 x2 + x3 + 1) / (x1 + 2 x2 + x3 + 3)
    max f  = (2 x1 + 2 x2 + x3 + 1) / (x1 + x3 + 1)   sur E
    s.c.   3 x1 +   x2 +   x3 <= 8
             x1 + 2 x2 + 2 x3 <= 9,   x entier >= 0

    |S| = 36, |E| = 5, q* = 3 en (2,2,0), max_S f = 9 en (0,4,0) -- domine.
    L'archive n'y trouve que 4 des 5 points efficaces : le relache NE SE VIDE
    PAS, il finit exactement sur le point manquant -- et le Th. 5' conclut
    quand meme. Les deux voies de preuve sont bien independantes.
    """
    return MOILFP(
        A=np.array([[3, 1, 1], [1, 2, 2]]), b=np.array([8, 9]),
        Z=[FracObj(np.array([0, 0, 3]), 0, np.array([0, 2, 1]), 2),
           FracObj(np.array([3, 3, 1]), 1, np.array([1, 2, 1]), 3)],
        f=FracObj(np.array([2, 2, 1]), 1, np.array([1, 0, 1]), 1),
        name="I3", seed=0)


# ============================================================================
# OUTILS DE LECTURE : ce que la METHODE ne fait pas, mais qui permet de la
# verifier point par point sur une instance enumerable.
# ============================================================================

def dans_relache(inst: MOILFP, x, coupes) -> bool:
    """x survit-il a toutes les coupes posees ? (disjonction du Th. 4 / Th. 6)"""
    zx = inst.criteria(x)
    for a in coupes:
        za = inst.criteria(np.asarray(a, dtype=int))
        if not any(zx[k] > za[k] for k in range(inst.p)):
            return False
    return True


def borne_th5(inst: MOILFP, R, q: Fraction):
    """
    Les quantites EXACTES du Th. 5' sur un relache R enumere :
        U = max_R { Q N - P D },  D+ = min D sur la zone active,
        borne = q + U / (Q D+).
    Renvoie (U, D+, borne) ; (None, None, None) si R est VIDE.
    """
    P, Q = q.numerator, q.denominator
    if not R:
        return None, None, None
    vals = [(Q * inst.f.numerator(x) - P * inst.f.denominator(x),
             inst.f.denominator(x)) for x in R]
    U = max(0, max(v for v, _ in vals))
    pos = [d for v, d in vals if v >= 0]
    if not pos:
        return U, None, q
    Dp = min(pos)
    return U, Dp, q + Fraction(U, Q * Dp)


def _fmt(x) -> str:
    return "(" + ",".join(str(int(v)) for v in x) + ")"


# ============================================================================
# COMMANDES
# ============================================================================

def cmd_demo() -> int:
    """Le deroule complet sur les deux illustrations."""
    for inst in (instance_I2(), instance_I3()):
        print("=" * 78)
        print(f"INSTANCE {inst.name}   n = {inst.n}, m = {inst.m}, p = {inst.p}")
        print("=" * 78)
        inst.check_assumptions()
        gt = ground_truth(inst, limit=50_000)
        q_star = gt.q_star
        E_keys = {tuple(int(v) for v in x) for x in gt.E}
        print(f"  |S| = {len(gt.S)}   |E| = {len(gt.E)}   "
              f"q* = {q_star} = {float(q_star):.4f} en {_fmt(gt.x_star)}")
        x_maxS = max(gt.S, key=lambda x: inst.f.value(x))
        est_eff = tuple(int(v) for v in x_maxS) in E_keys
        print(f"  max_S f = {gt.q_max_S} en {_fmt(x_maxS)} "
              f"({'EFFICACE' if est_eff else 'DOMINE'}) -> la borne naive se "
              f"trompe de {float((gt.q_max_S - q_star) / gt.q_max_S) * 100:.0f} %")

        print("\n-- Phase 1 : la recherche -------------------------------------")
        res = matheuristic_P(inst, time_budget=3.0, bound_budget=2.0, seed=0,
                             archive_cuts=True, cut_batch=40, cert_rounds=2)
        archive = list(res.archive)
        print(f"  archive ({len(archive)} points) : "
              + "  ".join(_fmt(x) for x in archive))
        print(f"  incumbent q = {res.q_lb}")
        print(f"  points DOMINES collectes : {len(res.cut_points)}"
              f"   <- le verrou : la coupe de dominance n'a rien a mordre")

        q = q_star
        w, _, _, _ = surrogate(inst, q)
        vivier = sorted(archive, key=lambda x: -float(w @ x))
        domines = sorted([x for x in gt.S
                          if tuple(int(v) for v in x) not in E_keys],
                         key=lambda x: -float(w @ x))

        print("\n-- Condition de cloture : un programme par point ---------------")
        reset_oracle_counter()
        for a in vivier:
            r = same_criteria_improves(inst, np.asarray(a, dtype=int), q)
            h = region_height(inst, np.asarray(a, dtype=int), q)
            par_lemme = " (acquise par le LEMME, sans PLNE)" if h <= 0 else ""
            verdict = "FAISABLE : on empoche le point" if r is not None \
                else f"infaisable -> cloture etablie{par_lemme}"
            print(f"  a = {_fmt(a):>10}   h = {h:>6.0f}   {verdict}")
        print(f"  cout : {ORACLE_CALLS['ilp']} appels au solveur ENTIER "
              f"(zero si le lemme suffit partout)")

        for titre, source, dispo in (
                ("COUPE D'EFFICACITE (source : archive)", vivier, len(vivier)),
                ("coupe de dominance (source : reparations)", domines, 0)):
            print(f"\n-- {titre} --")
            print(f"   {'#':>2} {'coupe en':>10} {'|R|':>4} {'U':>5} "
                  f"{'D+':>4} {'borne':>10}")
            coupes, R = [], list(gt.S)
            U, Dp, bd = borne_th5(inst, R, q)
            print(f"   {0:>2} {'--':>10} {len(R):>4} {U:>5} {str(Dp):>4} "
                  f"{str(bd):>10}")
            for i, a in enumerate(source, 1):
                coupes.append(a)
                R = [x for x in gt.S if dans_relache(inst, x, coupes)]
                if not R:
                    print(f"   {i:>2} {_fmt(a):>10} {0:>4}   RELACHE VIDE "
                          f"-> q* = q = {q}   (Proposition)")
                    break
                U, Dp, bd = borne_th5(inst, R, q)
                marque = "  <- U = 0, optimalite prouvee" if U == 0 else ""
                print(f"   {i:>2} {_fmt(a):>10} {len(R):>4} {U:>5} "
                      f"{str(Dp):>4} {str(bd):>10}{marque}")
                if i >= 8:
                    print("   ...")
                    break
            print(f"   points reellement DISPONIBLES : {dispo}")

        print("\n-- Ce que la methode rend --------------------------------------")
        for ac in (False, True):
            r = matheuristic_P(inst, time_budget=3.0, bound_budget=2.0, seed=0,
                               archive_cuts=ac, cut_batch=40, cert_rounds=2)
            assert r.q_lb <= q_star, "borne inferieure INVALIDE"
            assert r.q_ub is None or r.q_ub >= float(q_star) - 1e-9, \
                "borne superieure INVALIDE"
            print(f"  archive_cuts={str(ac):<5} : q_lb = {r.q_lb}   "
                  f"q_ub = {r.q_ub}   prouve = {r.proved_optimal}   "
                  f"coupes d'efficacite = {r.cert.get('archive_cuts', 0)}   "
                  f"clotures par le lemme = {r.cert.get('cloture_lemme', 0)}")
        print("  VALIDITE q_lb <= q* <= q_ub : verifiee\n")
    return 0


def cmd_verify(n_inst: int = 12) -> int:
    """
    Controles de validite contre la verite terrain, sur des instances tirees.
    C'est ici qu'il faut ajouter un test avant de croire a une amelioration.
    """
    print("=" * 92)
    print(f"VALIDITE contre enumeration exhaustive - {n_inst} instances tirees")
    print("=" * 92)
    print(f"{'instance':<26}{'|S|':>7}{'|E|':>6}{'q*':>10}"
          f"{'q_lb':>10}{'q_ub':>10}{'prouve':>8}{'verdict':>10}")
    print("-" * 92)
    ko = 0
    for i in range(n_inst):
        n = 5 + i % 4
        inst = generate(n=n, m=max(3, n // 2 + 1), p=2 + i % 3,
                        seed=100 + i, rhs_scale=1.6,
                        corr=[0.0, 0.5, 0.9][i % 3])
        inst.check_assumptions()
        gt = ground_truth(inst, limit=200_000)
        r = matheuristic_P(inst, time_budget=3.0, bound_budget=2.0, seed=0,
                           archive_cuts=True)
        lb_ok = r.q_lb <= gt.q_star + Fraction(1, 10**9)
        ub_ok = r.q_ub is None or r.q_ub >= float(gt.q_star) - 1e-9
        eff_ok = efficiency_test(inst, r.x_best).efficient is not False
        pr_ok = (not r.proved_optimal) or r.q_lb == gt.q_star
        bon = lb_ok and ub_ok and eff_ok and pr_ok
        ko += 0 if bon else 1
        print(f"{inst.name:<26}{len(gt.S):>7}{len(gt.E):>6}"
              f"{float(gt.q_star):>10.4f}{float(r.q_lb):>10.4f}"
              f"{(r.q_ub if r.q_ub is not None else float('nan')):>10.4f}"
              f"{str(r.proved_optimal):>8}{'ok' if bon else 'KO':>10}")
    print("-" * 92)
    print(f"  q_lb <= q*  |  q* <= q_ub  |  x_best EFFICACE  |  "
          f"prouve => q_lb = q*")
    print(f"  {n_inst - ko}/{n_inst} instances conformes"
          f"{'' if ko == 0 else '   *** ECHEC ***'}")
    return 0 if ko == 0 else 1


def cmd_ab(n_inst: int = 10, budget: float = 8.0) -> int:
    """
    A/B APPARIE des leviers. C'est le seul mode de comparaison honnete :
    meme instance, meme graine, un seul reglage qui change.
    """
    # Chaque variante n'ajoute QU'UN levier a la precedente : sans cela on
    # ne saurait pas a quoi attribuer un ecart.
    variantes = [
        ("base   ", dict(archive_cuts=False, closure_lemma=False,
                         cut_diversify=False)),
        ("+arch  ", dict(archive_cuts=True, closure_lemma=False,
                         cut_diversify=False)),
        ("+lemme ", dict(archive_cuts=True, closure_lemma=True,
                         cut_diversify=False)),
        ("+rang  ", dict(archive_cuts=True, closure_lemma=True,
                         height_rank=True, cut_diversify=False)),
        ("+divers", dict(archive_cuts=True, closure_lemma=True,
                         cut_diversify=True)),
    ]
    print("=" * 98)
    print(f"A/B APPARIE - {n_inst} instances, budget {budget:g} s par variante")
    print("=" * 98)
    entetes = "".join(f"{nom:>17}" for nom, _ in variantes)
    print(f"{'instance':<24}{entetes}")
    print(f"{'':<24}" + "".join(f"{'q_lb / ILP':>17}" for _ in variantes))
    print("-" * 98)
    tot = {nom: [0, 0] for nom, _ in variantes}          # [preuves, ILP]
    for i in range(n_inst):
        n = 6 + i % 5
        inst = generate(n=n, m=max(3, n // 2 + 1), p=3,
                        seed=200 + i, rhs_scale=1.6,
                        corr=[0.0, 0.5, 0.9][i % 3])
        ligne = f"{inst.name:<24}"
        for nom, kw in variantes:
            r = matheuristic_P(inst, time_budget=budget * 0.6,
                               bound_budget=budget * 0.4, seed=0, **kw)
            tot[nom][0] += int(r.proved_optimal)
            tot[nom][1] += r.ilp_calls
            marque = "*" if r.proved_optimal else " "
            ligne += f"{float(r.q_lb):>10.3f}{marque}{r.ilp_calls:>6}"
        print(ligne, flush=True)
    print("-" * 98)
    for nom, _ in variantes:
        print(f"  {nom} : optimalite prouvee {tot[nom][0]:>3}/{n_inst}   "
              f"appels entiers cumules {tot[nom][1]:>6}")
    print("  ( * = optimalite prouvee )")
    print("\n  Rappel de lecture : ne comparez PAS deux medianes independantes."
          "\n  Le gain se lit par PAIRES, instance par instance.")
    return 0


def cmd_scale() -> int:
    """
    n = 20, 30, 40 : le regime ou la borne ne bouge pas. Sans verite terrain
    -- l'enumeration de E y est hors de portee -- mais les encadrements
    restent verifiables entre eux, et q_lb reste comparable puisque les deux
    valeurs sont des points EFFICACES CERTIFIES.
    """
    print("=" * 100)
    print("PASSAGE A L'ECHELLE (corr = 0, E epais) - budget 18 + 12 s")
    print("=" * 100)
    print(f"{'n':>4}{'max_S f':>12}{'q_lb sans':>12}{'q_ub sans':>12}"
          f"{'ecart':>9}{'q_lb avec':>12}{'q_ub avec':>12}{'ecart':>9}"
          f"{'coherent':>10}")
    print("-" * 100)
    ok_tout = True
    for n in (20, 30, 40):
        inst = generate(n=n, m=max(3, n // 2 + 1), p=3, seed=1,
                        rhs_scale=1.0, corr=0.0)
        mS = upper_bound_over_S(inst)   # +inf si Dinkelbach n'a pas conclu
        res = {}
        for ac in (False, True):
            r = matheuristic_P(inst, time_budget=18, bound_budget=12,
                               seed=0, archive_cuts=ac)
            hi = min(r.q_ub if r.q_ub is not None else np.inf, mS)
            res[ac] = (float(r.q_lb), hi,
                       (hi - float(r.q_lb)) / max(1e-12, abs(hi)) * 100)
        lo = max(res[False][0], res[True][0])
        hi = min(res[False][1], res[True][1])
        ok = lo <= hi + 1e-9
        ok_tout &= ok
        print(f"{n:>4}{mS:>12.4f}"
              f"{res[False][0]:>12.4f}{res[False][1]:>12.4f}{res[False][2]:>8.1f}%"
              f"{res[True][0]:>12.4f}{res[True][1]:>12.4f}{res[True][2]:>8.1f}%"
              f"{'ok' if ok else 'KO':>10}", flush=True)
    print("-" * 100)
    print(f"  COHERENCE : {'TOUT VALIDE' if ok_tout else 'ECHEC'}")
    print("  Les ecarts restent enormes : c'est la frontiere honnete de la"
          " methode.\n  Le verrou n'est plus la disponibilite des coupes mais"
          " leur COUT en binaires.")
    return 0 if ok_tout else 1


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "demo"
    if cmd == "demo":
        return cmd_demo()
    if cmd == "verify":
        return cmd_verify(int(sys.argv[2]) if len(sys.argv) > 2 else 12)
    if cmd == "ab":
        return cmd_ab(int(sys.argv[2]) if len(sys.argv) > 2 else 10)
    if cmd == "scale":
        return cmd_scale()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
