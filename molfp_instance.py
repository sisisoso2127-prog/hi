"""
molfp_instance.py
=================
Structures de donnees et generateur d'instances pour le probleme

    (MOILFP)  max  Z_k(x) = (c_k^T x + alpha_k) / (d_k^T x + beta_k),  k = 1..p
              s.c. A x <= b,  x in Z^n_+

et pour la fonction d'utilite fractionnaire

    (P)       max  f(x) = (c^T x + alpha) / (d^T x + beta)   s.c.  x in E

ou E est l'ensemble des solutions efficaces de (MOILFP).

Le generateur garantit par construction :
  (A1) D_k(x) = d_k^T x + beta_k >= 1 > 0 pour tout x realisable  (d_k >= 0, beta_k >= 1)
  (A2) S = {x in Z^n_+ : Ax <= b} est non vide et borne  (A >= 0, colonnes non nulles)

Toutes les donnees sont ENTIERES : c'est ce qui permet plus loin un test
d'efficacite entierement integral, donc sans tolerance numerique.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from fractions import Fraction
from typing import List, Optional

import numpy as np


# ----------------------------------------------------------------------------
# Structures
# ----------------------------------------------------------------------------

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
        """
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

    def check_assumptions(self) -> None:
        """Verifie (A1) et (A2). Leve une exception si violees."""
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
        )


# ----------------------------------------------------------------------------
# Generateur
# ----------------------------------------------------------------------------

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


def generate_calibrated(n: int, m: int, p: int, seed: int,
                        target_S: int = 2000, tol: float = 0.5,
                        max_tries: int = 14, **kw) -> MOILFP:
    """
    Genere une instance dont |S| est proche de target_S, par recherche
    dichotomique sur rhs_scale. Utile pour construire un lot d'instances
    de difficulte homogene (comparaisons equitables entre methodes).
    """
    from molfp_enum import enumerate_feasible
    lo, hi = 0.2, 8.0
    best, best_err = None, float("inf")
    for _ in range(max_tries):
        mid = (lo + hi) / 2
        inst = generate(n, m, p, seed, rhs_scale=mid, **kw)
        try:
            size = len(enumerate_feasible(inst, limit=20 * target_S))
        except MemoryError:
            hi = mid
            continue
        err = abs(size - target_S) / target_S
        if err < best_err:
            best, best_err = inst, err
        if err <= tol:
            return inst
        if size < target_S:
            lo = mid
        else:
            hi = mid
    return best


def generate_suite(sizes, seeds, out_dir: str) -> List[str]:
    """
    Genere un lot d'instances et les ecrit sur disque.

    sizes : liste de triplets (n, m, p)
    seeds : liste de graines
    Retourne la liste des chemins ecrits.
    """
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for (n, m, p) in sizes:
        for s in seeds:
            inst = generate(n, m, p, s)
            path = os.path.join(out_dir, inst.name + ".json")
            inst.save(path)
            paths.append(path)
    return paths
