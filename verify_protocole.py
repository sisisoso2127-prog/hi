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

DEUX TEXTES sont controles, parce qu'il y en a deux :

  MEMOIRE  le tableau « Le generateur d'instances, tirage par tirage »
           et les paragraphes qui le suivent ;
  ARTICLE  le paragraphe « Le generateur » du plan d'experience, plus
           condense -- donc le vrai test : c'est le texte le plus court qui
           dit si l'essentiel a ete garde.

Et un CONTROLE NEGATIF, `lecture_insuffisante`, qui reimplemente l'article
tel qu'il etait AVANT correction : les criteres y etaient donnes par un
domaine et non par une loi, et « toutes les donnees sont entieres » invitait
a les tirer en entiers. Cette lecture reproduit A et b puis diverge sur tous
les criteres. Elle est gardee ici pour que la precision « continus puis
arrondis » ne puisse pas etre reperdue sans que ce module s'en apercoive.
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


def depuis_l_article(n: int, m: int, p: int, seed: int,
                     kappa: float, corr: float):
    """
    Le generateur tel que l'ARTICLE le decrit, en n'utilisant que son
    paragraphe : moteur PCG64, flux unique, ordre des tirages annonce, lois
    continues pour les criteres et entieres pour l'utilite.
    """
    rng = np.random.default_rng(seed)

    A = rng.integers(0, 10, size=(m, n))
    for j in range(n):
        if A[:, j].sum() == 0:
            A[rng.integers(0, m), j] = rng.integers(1, 10)
    b = np.maximum(1, (kappa * A.sum(axis=1)).astype(int))

    # « les quatre tirages latents, continus », dans cet ordre
    lat = (rng.uniform(1, 20, size=n), rng.uniform(0, 5, size=n),
           rng.uniform(0, 10), rng.uniform(1, 10))

    def melange(u, v, lo, hi):
        return np.clip(np.rint(corr * u + (1 - corr) * v), lo, hi).astype(int)

    Z = []
    for _ in range(p):
        # « pour chacun des p criteres et dans cet ordre : c, d, alpha, beta »
        ck = melange(lat[0], rng.uniform(1, 20, size=n), 1, 20)
        dk = melange(lat[1], rng.uniform(0, 5, size=n), 0, 5)
        ak = int(np.clip(round(corr * lat[2]
                               + (1 - corr) * rng.uniform(0, 10)), 0, 10))
        bk = int(np.clip(round(corr * lat[3]
                               + (1 - corr) * rng.uniform(1, 10)), 1, 10))
        Z.append((ck, ak, dk, bk))

    # « f vient en dernier, en entiers et sans melange : c, alpha, d, beta »
    f = (rng.integers(1, 21, size=n), int(rng.integers(0, 10)),
         rng.integers(0, 6, size=n), int(rng.integers(1, 10)))
    return A, b, Z, f


def lecture_insuffisante(n: int, m: int, p: int, seed: int,
                         kappa: float, corr: float):
    """
    CONTROLE NEGATIF. L'article avant correction donnait les criteres par un
    domaine, « c_k dans [1,20] », en annoncant par ailleurs que toutes les
    donnees sont entieres. Un lecteur sans le code tire donc en ENTIERS.
    Cette fonction doit DIVERGER : c'est ce qui prouve que la precision
    « continus puis arrondis » est necessaire et non decorative.
    """
    rng = np.random.default_rng(seed)
    A = rng.integers(0, 10, size=(m, n))
    for j in range(n):
        if A[:, j].sum() == 0:
            A[rng.integers(0, m), j] = rng.integers(1, 10)
    b = np.maximum(1, (kappa * A.sum(axis=1)).astype(int))
    lat = (rng.integers(1, 21, size=n), rng.integers(0, 6, size=n),
           rng.integers(0, 11), rng.integers(1, 11))

    def melange(u, v, lo, hi):
        return np.clip(np.rint(corr * u + (1 - corr) * v), lo, hi).astype(int)

    Z = []
    for _ in range(p):
        ck = melange(lat[0], rng.integers(1, 21, size=n), 1, 20)
        dk = melange(lat[1], rng.integers(0, 6, size=n), 0, 5)
        ak = int(np.clip(round(corr * lat[2]
                               + (1 - corr) * rng.integers(0, 11)), 0, 10))
        bk = int(np.clip(round(corr * lat[3]
                               + (1 - corr) * rng.integers(1, 11)), 1, 10))
        Z.append((ck, ak, dk, bk))
    f = (rng.integers(1, 21, size=n), int(rng.integers(0, 10)),
         rng.integers(0, 6, size=n), int(rng.integers(1, 10)))
    return A, b, Z, f


def identique(ref, produit) -> bool:
    A, b, Z, f = produit
    ok = np.array_equal(ref.A, A) and np.array_equal(ref.b, b)
    for Zk, (num, a, den, bk) in zip(ref.Z, Z):
        ok &= (np.array_equal(Zk.num, num) and Zk.a == a
               and np.array_equal(Zk.den, den) and Zk.b == bk)
    ok &= (np.array_equal(ref.f.num, f[0]) and ref.f.a == f[1]
           and np.array_equal(ref.f.den, f[2]) and ref.f.b == f[3])
    return bool(ok)


def grille():
    for n, kappa in ((6, 1.5), (8, 1.0), (20, 1.0), (40, 1.0)):
        m = max(3, n // 2 + 1)
        for p in (2, 3, 4):
            for seed in (1, 2, 3):
                for corr in (0.0, 0.25, 0.5, 0.9, 1.0):
                    yield n, m, p, seed, kappa, corr


def main() -> int:
    textes = [("MEMOIRE  (tableau tirage par tirage)", depuis_le_texte),
              ("ARTICLE  (paragraphe du plan d'experience)", depuis_l_article)]
    total = sum(1 for _ in grille())
    verdict = 0

    for nom, reimpl in textes:
        ecarts = []
        for (n, m, p, seed, kappa, corr) in grille():
            ref = generate(n=n, m=m, p=p, seed=seed,
                           rhs_scale=kappa, corr=corr)
            if not identique(ref, reimpl(n, m, p, seed, kappa, corr)):
                ecarts.append((n, m, p, seed, kappa, corr))
        etat = "AUCUN ECART" if not ecarts else f"{len(ecarts)} ECARTS"
        print(f"{nom:<44} {total:>4} instances -> {etat}")
        for e in ecarts[:5]:
            print(f"    n={e[0]} m={e[1]} p={e[2]} graine={e[3]} "
                  f"kappa={e[4]} corr={e[5]}")
        if ecarts:
            verdict = 1

    # controle negatif : cette lecture DOIT diverger
    div = sum(1 for (n, m, p, seed, kappa, corr) in grille()
              if not identique(generate(n=n, m=m, p=p, seed=seed,
                                        rhs_scale=kappa, corr=corr),
                               lecture_insuffisante(n, m, p, seed,
                                                    kappa, corr)))
    ok_neg = (div == total)
    print(f"{'CONTROLE NEGATIF (criteres lus en entiers)':<44} "
          f"{total:>4} instances -> {div} divergences"
          f"{'' if ok_neg else '  <-- ATTENDU : toutes'}")
    if not ok_neg:
        verdict = 1

    # l'invariance que la remarque du memoire annonce : (A,b) ne bouge pas
    # quand seul corr change
    inv = True
    for n, p, s_ in ((6, 3, 1), (8, 3, 2), (20, 3, 1), (40, 3, 2)):
        m = max(3, n // 2 + 1)
        ref = generate(n=n, m=m, p=p, seed=s_, rhs_scale=1.0, corr=0.0)
        for c in (0.25, 0.5, 0.9, 1.0):
            g = generate(n=n, m=m, p=p, seed=s_, rhs_scale=1.0, corr=c)
            inv &= (np.array_equal(ref.A, g.A) and np.array_equal(ref.b, g.b))
    print(f"{'(A, b) invariants quand seul corr change':<44} {str(inv):>21}")
    if not inv:
        verdict = 1

    print("OK" if verdict == 0 else "ECHEC : le TEXTE est faux, pas le code")
    return verdict


if __name__ == "__main__":
    raise SystemExit(main())
