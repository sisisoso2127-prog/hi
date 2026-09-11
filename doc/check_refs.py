#!/usr/bin/env python3
"""
check_refs.py
=============
VERIFIE QUE CHAQUE RENVOI DESIGNE UN OBJET DU BON TYPE.

Pourquoi cet outil existe. La numerotation des enonces a change plusieurs fois
en cours de travail. Les renvois ecrits en CHIFFRES DURS -- « Th. 4 », « le
theoreme 2 » -- ne bougent pas avec elle et finissent par designer autre chose,
ou rien. Trois d'entre eux designaient un theoreme INEXISTANT, et deux
attribuaient la coupe de dominance au corollaire de la coupe d'efficacite.

Un renvoi faux ne casse pas la compilation : LaTeX ne verifie que l'existence
de l'etiquette, jamais la NATURE de ce qu'elle designe. « corollaire~\\ref{x} »
ou x etiquette un theoreme compile sans un mot.

Trois controles :
  1. chiffres durs    -- « Th. 4 », « theoreme 2 » hors \\ref, hors citation
                         d'une numerotation externe ;
  2. type du renvoi   -- le mot qui precede \\ref{x} doit correspondre a
                         l'environnement ou \\label{x} est pose ;
  3. etiquettes mortes-- \\label sans \\ref, \\ref sans \\label.

Usage :  python check_refs.py [fichier.tex ...]     (defaut : les deux memoires)
"""

from __future__ import annotations

import re
import sys
from typing import Dict, List, Tuple

ENVS = ("theorem", "proposition", "lemma", "corollary", "remark",
        "definition", "algorithm", "table", "figure", "lstlisting")

# le mot francais attendu devant \ref, par environnement
# valeurs au singulier depouille de son « s » final, comme le mot teste
ATTENDU = {
    "theorem":     {"théorème", "theoreme", "th"},
    "proposition": {"proposition", "prop"},
    "lemma":       {"lemme"},
    "corollary":   {"corollaire", "cor"},
    "remark":      {"remarque"},
    "algorithm":   {"algorithme", "alg"},
    "table":       {"tableau", "table"},
    "figure":      {"figure", "fig"},
    "lstlisting":  {"listing"},
}

# numerotations EXTERNES, citees telles que les auteurs les ecrivent
EXEMPT = (r"corollaire~2\.3", r"théorème~3\.6", r"section~6", r"leur ")


MATH = ("equation", "align", "gather", "multline", "eqnarray")


def labels_math(src: str) -> set:
    """Etiquettes posees sur une EQUATION : elles ne designent pas l'enonce
    qui les contient, et un « la condition~(3) » est alors legitime."""
    out = set()
    for env in MATH:
        for m in re.finditer(r"\\begin\{" + env + r"\*?\}", src):
            fin = src.find("\\end{" + env, m.end())
            out |= set(re.findall(r"\\label\{([^}]+)\}",
                                  src[m.end(): fin if fin > 0 else len(src)]))
    return out


def labels_par_env(src: str) -> Dict[str, str]:
    """etiquette -> environnement qui la contient (equations exclues)."""
    math = labels_math(src)
    out: Dict[str, str] = {}
    for env in ENVS:
        for m in re.finditer(r"\\begin\{" + env + r"\}", src):
            fin = src.find("\\end{" + env + "}", m.end())
            bloc = src[m.end(): fin if fin > 0 else len(src)]
            for lab in re.findall(r"\\label\{([^}]+)\}", bloc):
                if lab not in math:
                    out.setdefault(lab, env)
    return out


def controle(path: str) -> List[str]:
    src = open(path, encoding="utf-8").read()
    # on ignore les commentaires LaTeX
    sans_comm = re.sub(r"(?<!\\)%.*", "", src)
    labs = labels_par_env(sans_comm)
    pbs: List[str] = []

    # -- 1. chiffres durs ----------------------------------------------------
    dur = re.compile(r"(Th\.|[Tt]héorème|[Pp]roposition|[Cc]orollaire|[Ll]emme)"
                     r"~? ?(\d+)")
    for m in dur.finditer(sans_comm):
        ctx = sans_comm[max(0, m.start() - 40): m.end() + 20]
        if any(re.search(e, ctx) for e in EXEMPT):
            continue
        pbs.append(f"chiffre dur : « {m.group(0)} »  ...{' '.join(ctx.split())[-70:]}")

    # -- 2. type du renvoi ---------------------------------------------------
    ren = re.compile(r"(\w+)\.?~?\\(?:eq)?ref\{([^}]+)\}")
    for m in ren.finditer(sans_comm):
        mot, lab = m.group(1).lower().rstrip(".").rstrip("s"), m.group(2)
        env = labs.get(lab)
        if env is None or env not in ATTENDU:
            continue
        if mot in ("et", "ou", "le", "la", "de", "du", "des", "voir", "cf",
                   "dans", "a", "à", "par", "au", "aux", "l", "d", "s"):
            continue
        if mot not in {a.rstrip("s") for a in ATTENDU[env]}:
            pbs.append(f"type : « {m.group(1)}~\\ref{{{lab}}} » mais "
                       f"{lab} etiquette un(e) {env}")

    # -- 3. etiquettes mortes ------------------------------------------------
    tous_labs = set(re.findall(r"\\label\{([^}]+)\}", sans_comm))
    tous_labs |= set(re.findall(r"label=\{?([A-Za-z0-9:_-]+)", sans_comm))
    refs = set(re.findall(r"\\(?:eq|page|auto)?ref\{([^}]+)\}", sans_comm))
    for lab in sorted(refs - tous_labs):
        pbs.append(f"renvoi vers une etiquette INEXISTANTE : {lab}")
    enonces = {"theorem", "proposition", "lemma", "corollary", "remark"}
    for lab in sorted(tous_labs - refs):
        if labs.get(lab) in enonces:
            pbs.append(f"enonce etiquete mais jamais invoque : {lab}")
    return pbs


if __name__ == "__main__":
    cibles = sys.argv[1:] or ["hybride.tex", "article.tex"]
    total = 0
    for c in cibles:
        pbs = controle(c)
        total += len(pbs)
        print(f"=== {c} : {len(pbs)} probleme(s)")
        for p in pbs:
            print("   ", p)
    print(f"\nTOTAL : {total}")
    sys.exit(1 if total else 0)
