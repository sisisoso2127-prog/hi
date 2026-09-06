"""
analyze.py
==========
Analyse de `campaign.csv`. La campagne mesure, l'analyse conclut -- et la
separation compte, parce que la lecture naive de ces donnees est fausse.

TRAITEMENT DE LA CENSURE (point de methode central). Les instances arretees
sur limite de temps ont un cout CENSURE : leur nombre d'appels ILP est un
minorant, pas une mesure. Les melanger aux autres remplace le cout des
instances les plus dures par une valeur tronquee, et **les strates qui
echouent le plus paraissent alors les plus faciles**. C'est un biais de
survie. On separe donc deux questions qui n'ont pas la meme reponse :

  Q1  parmi les instances RESOLUES, qu'est-ce qui fait monter le cout ?
      (correlations de Spearman, censurees exclues)
  Q2  qu'est-ce qui fait qu'une instance n'est pas resolue du tout ?
      (Mann-Whitney entre censurees et resolues)

  Q3  experience controlee sur `corr`, le seul facteur manipule
  Q4  la taille predit-elle quelque chose ?

Usage :  python analyze.py [csv]
"""

from __future__ import annotations

import csv
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

NUM = {"n", "m", "p", "seed", "S", "E", "outer", "cuts", "ilp",
       "archive", "false_pos", "mismatch", "theta_S", "depth_S"}
FLOAT = {"corr", "ratio", "q_star", "max_S", "relax_gap", "time", "coverage"}

PREDICTORS = [
    ("relax_gap", "relax_gap = (max_S f - q*)/|max_S f|"),
    ("theta_S",   "theta_S (un seul ILP)"),
    ("depth_S",   "depth_S (quelques ILP)"),
    ("ratio",     "ratio |E|/|S|"),
    ("E",         "|E| absolu"),
    ("p",         "p"),
    ("n",         "n"),
    ("S",         "|S|"),
]

SPLIT_VARS = [("E", "|E|"), ("ratio", "ratio |E|/|S|"), ("n", "n"),
              ("S", "|S|"), ("theta_S", "theta_S"), ("depth_S", "depth_S")]


def stars(pv: float) -> str:
    return "***" if pv < 1e-3 else "**" if pv < 1e-2 else \
           "*" if pv < 5e-2 else "(n.s.)"


def load(path: str) -> list:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            for k, v in list(r.items()):
                if k in NUM:
                    r[k] = int(v) if v not in ("", "None") else None
                elif k in FLOAT:
                    r[k] = float(v)
            rows.append(r)
    return rows


def col(rows, key):
    return np.array([r[key] for r in rows], dtype=float)


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "campaign.csv"
    rows = load(path)
    solved = [r for r in rows if r["status"] == "optimal"]
    censored = [r for r in rows if r["status"] != "optimal"]

    print("=" * 78)
    print(f"CAMPAGNE DE DIFFICULTE - {len(rows)} instances")
    print("=" * 78)
    print(f"resolues a l'optimum : {len(solved)}")
    print(f"arretees sur limite  : {len(censored)}  (cout CENSURE, exclu de Q1)")
    n_mis = sum(r["mismatch"] for r in rows)
    n_fp = sum(r["false_pos"] for r in rows)
    print(f"\nVALIDATION contre la verite terrain")
    print(f"  divergences (statut 'optimal' mais q != q*) : {n_mis}")
    print(f"  faux positifs dans les archives             : {n_fp}")
    if n_mis or n_fp:
        print("  >>> ECHEC : au moins une garantie est violee.")

    # ---------------------------------------------------------------- Q1
    print("\n" + "=" * 78)
    print("Q1 - COUT PARMI LES INSTANCES RESOLUES (censurees exclues)")
    print("=" * 78)
    print(f"{'predicteur':<44}{'Spearman avec les appels ILP':>34}")
    y = col(solved, "ilp")
    res = []
    for key, label in PREDICTORS:
        x = col(solved, key)
        ok = np.isfinite(x) & np.isfinite(y)
        # une colonne constante (strate unique) n'a pas de correlation definie ;
        # scipy renvoie nan avec un avertissement, on le dit explicitement
        if ok.sum() < 3 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0:
            res.append((-1.0, key, label, float("nan"), float("nan")))
            continue
        rho, pv = spearmanr(x[ok], y[ok])
        res.append((abs(rho), key, label, rho, pv))
    for _, key, label, rho, pv in sorted(res, reverse=True):
        if np.isnan(rho):
            print(f"{label:<44}{'constant sur ce lot':>28} {'':<5}")
        else:
            print(f"{label:<44}{rho:>+28.3f} {stars(pv):<5}")
    print("\n`relax_gap` fait intervenir q*, donc la reponse cherchee : c'est un")
    print("indicateur de MECANISME, jamais un predicteur utilisable avant")
    print("resolution.")

    # ---------------------------------------------------------------- Q2
    print("\n" + "=" * 78)
    print("Q2 - ECHEC TOTAL : censurees contre resolues")
    print("=" * 78)
    if not censored:
        print("aucune instance censuree : Q2 sans objet a cette limite de temps.")
    else:
        print(f"{'variable':<18}{'mediane censuree':>18}{'mediane resolue':>18}"
              f"{'Mann-Whitney':>22}")
        for key, label in SPLIT_VARS:
            a, b = col(censored, key), col(solved, key)
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if len(a) < 2 or len(b) < 2:
                continue
            _, pv = mannwhitneyu(a, b, alternative="two-sided")
            print(f"{label:<18}{np.median(a):>18.4g}{np.median(b):>18.4g}"
                  f"{f'p = {pv:.1e}':>16} {stars(pv):<5}")
        print("\nLire le SENS de chaque ligne : si une variable monte chez les")
        print("censurees alors qu'elle correle NEGATIVEMENT avec le cout en Q1,")
        print("les deux regimes sont de sens opposes et aucun reglage a priori")
        print("ne protege des deux a la fois.")

    # ---------------------------------------------------------------- Q3
    print("\n" + "=" * 78)
    print("Q3 - EXPERIENCE CONTROLEE SUR corr (domaine S identique)")
    print("=" * 78)
    print(f"{'corr':>6}{'|E| median':>13}{'timeouts':>11}"
          f"{'ILP median (resolues)':>24}")
    by_corr = defaultdict(list)
    for r in rows:
        by_corr[r["corr"]].append(r)
    for c in sorted(by_corr):
        v = by_corr[c]
        sv = [r for r in v if r["status"] == "optimal"]
        med = np.median([r["ilp"] for r in sv]) if sv else float("nan")
        print(f"{c:>6.2f}{np.median([r['E'] for r in v]):>13.1f}"
              f"{sum(1 for r in v if r['status'] != 'optimal'):>11}"
              f"{med:>24.1f}")
    print("\nLa colonne des couts se lit APRES celle des timeouts : a faible")
    print("corr, les instances cheres sont censurees et sortent de la mediane.")

    # ---------------------------------------------------------------- Q4
    print("\n" + "=" * 78)
    print("Q4 - LA TAILLE PREDIT-ELLE QUELQUE CHOSE ?")
    print("=" * 78)
    print(f"{'n':>4}{'ILP median':>13}{'min':>8}{'max':>8}{'% timeout':>12}")
    by_n = defaultdict(list)
    for r in rows:
        by_n[r["n"]].append(r)
    for n in sorted(by_n):
        v = by_n[n]
        sv = [r["ilp"] for r in v if r["status"] == "optimal"]
        pct = 100.0 * sum(1 for r in v if r["status"] != "optimal") / len(v)
        if sv:
            print(f"{n:>4}{np.median(sv):>13.0f}{min(sv):>8}{max(sv):>8}"
                  f"{pct:>11.1f}%")
        else:
            print(f"{n:>4}{'-':>13}{'-':>8}{'-':>8}{pct:>11.1f}%")

    # ------------------------------------------------------- confirmations
    print("\n" + "=" * 78)
    print("CONFIRMATIONS")
    print("=" * 78)
    outer = col(rows, "outer")
    outer = outer[outer > 0]
    print(f"Dinkelbach : {int(outer.min())} a {int(outer.max())} iterations "
          f"externes (mediane {np.median(outer):.0f}, moyenne {outer.mean():.2f})")
    cov = col(rows, "coverage")
    print(f"Archive    : couverture mediane de E = {np.median(cov):.1f} %, "
          f"aucun faux positif" if n_fp == 0 else
          f"Archive    : couverture mediane {np.median(cov):.1f} %, "
          f"{n_fp} FAUX POSITIFS")
    return 0 if (n_mis == 0 and n_fp == 0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
