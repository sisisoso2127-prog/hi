#!/usr/bin/env python3
"""
verify_protocole.py
===================
LE PROTOCOLE PUBLIE SUFFIT-IL A REGENERER LES INSTANCES ?

Le memoire donne le generateur en toutes lettres (tableau « Le generateur
d'instances, tirage par tirage » et les deux paragraphes qui le suivent).
Un protocole publie n'a de valeur que s'il se reimplemente. Ce module le
reimplemente A PARTIR DU SEUL TEXTE -- sans importer molfp_instance.generate
dans `depuis_le_texte` -- et compare coefficient par coefficient.

Ce n'est pas un test unitaire de plus : c'est le controle que la section
« protocole » du memoire dit la verite. S'il echoue, c'est le TEXTE qui est
faux, pas le code.
"""

from __future__ import annotations

import numpy as np

from molfp_instance import generate


def depuis_le_texte(n: int, m: int, p: int, seed: int,
                    kappa: float, corr: float):
    """Le generateur, tel que le memoire le decrit, et rien d'autre."""
    rng = np.random.default_rng(seed)

    # contraintes : A dans {0..9}, colonne nulle corrigee dans {1..9}
    A = rng.integers(0, 10, size=(m, n))
    for j in range(n):
        if A[:, j].sum() == 0:
            A[rng.integers(0, m), j] = rng.integers(1, 10)
    # second membre POSE, non tire : b_i = max(1, floor(kappa * somme ligne))
    b = np.maximum(1, (kappa * A.sum(axis=1)).astype(int))

    # composante latente commune, partagee par les p criteres
    base_num = rng.uniform(1, 20, size=n)
    base_den = rng.uniform(0, 5, size=n)
    base_a = rng.uniform(0, 10)
    base_b = rng.uniform(1, 10)

    def mix(u, v, lo, hi):
        return np.clip(np.rint(corr * u + (1 - corr) * v), lo, hi).astype(int)

    Z = []
    for _ in range(p):
        num = mix(base_num, rng.uniform(1, 20, size=n), 1, 20)
        den = mix(base_den, rng.uniform(0, 5, size=n), 0, 5)
        a = int(np.clip(round(corr * base_a
                              + (1 - corr) * rng.uniform(0, 10)), 0, 10))
        bk = int(np.clip(round(corr * base_b
                               + (1 - corr) * rng.uniform(1, 10)), 1, 10))
        Z.append((num, a, den, bk))

    # utilite : entiere, et INDEPENDANTE des criteres (aucun melange)
    f = (rng.integers(1, 21, size=n), int(rng.integers(0, 10)),
         rng.integers(0, 6, size=n), int(rng.integers(1, 10)))
    return A, b, Z, f


def grille():
    for n, kappa in ((6, 1.5), (8, 1.0), (20, 1.0), (40, 1.0)):
        m = max(3, n // 2 + 1)
        for p in (2, 3, 4):
            for seed in (1, 2, 3):
                for corr in (0.0, 0.25, 0.5, 0.9, 1.0):
                    yield n, m, p, seed, kappa, corr


def main() -> int:
    ecarts = 0
    total = 0
    for (n, m, p, seed, kappa, corr) in grille():
        total += 1
        ref = generate(n=n, m=m, p=p, seed=seed,
                       rhs_scale=kappa, corr=corr)
        A, b, Z, f = depuis_le_texte(n, m, p, seed, kappa, corr)
        ok = np.array_equal(ref.A, A) and np.array_equal(ref.b, b)
        for Zk, (num, a, den, bk) in zip(ref.Z, Z):
            ok &= (np.array_equal(Zk.num, num) and Zk.a == a
                   and np.array_equal(Zk.den, den) and Zk.b == bk)
        ok &= (np.array_equal(ref.f.num, f[0]) and ref.f.a == f[1]
               and np.array_equal(ref.f.den, f[2]) and ref.f.b == f[3])
        if not ok:
            ecarts += 1
            print(f"  ECART : n={n} m={m} p={p} graine={seed} "
                  f"kappa={kappa} corr={corr}")

    # l'invariance que la remarque du memoire annonce : (A,b) ne bouge pas
    # quand seul corr change
    inv = True
    for n, p, s in ((6, 3, 1), (8, 3, 2), (20, 3, 1), (40, 3, 2)):
        m = max(3, n // 2 + 1)
        ref = generate(n=n, m=m, p=p, seed=s, rhs_scale=1.0, corr=0.0)
        for c in (0.25, 0.5, 0.9, 1.0):
            g = generate(n=n, m=m, p=p, seed=s, rhs_scale=1.0, corr=c)
            inv &= (np.array_equal(ref.A, g.A) and np.array_equal(ref.b, g.b))

    print(f"{total} instances regenerees a partir du seul texte publie ; "
          f"ecarts : {ecarts}")
    print(f"(A, b) invariants quand seul corr change : {inv}")
    return 0 if (ecarts == 0 and inv) else 1


if __name__ == "__main__":
    raise SystemExit(main())
