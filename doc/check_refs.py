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

Un quatrieme controle, CROISE, existe pour le cas que rien d'autre ne peut
attraper : pseudocode.tex cite la numerotation de article.tex en chiffres
durs, alors que ce sont DEUX documents separes. Aucun compilateur ne verra
jamais la derive. Ce controle simule le compteur de theoremes de l'article et
confronte chaque « Theoreme N » du pseudo-code a ce que N designe reellement.

Usage :  python check_refs.py [fichier.tex ...]     (defaut : les deux memoires)
         python check_refs.py --cross pseudocode.tex article.tex
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
    # Un document peut declarer qu'il CITE la numerotation d'un autre. La
    # regle ne s'y applique alors pas : c'est le controle croise qui la
    # remplace, et il est plus fort puisqu'il verifie ce que le numero
    # designe vraiment.
    ext = re.search(r"%%\s*check_refs:\s*numerotation externe\s*=\s*(\S+)", src)
    if ext:
        pbs.append(f"[info] numerotation externe declaree : "
                   f"lancer  python check_refs.py --cross {path} {ext.group(1)}")

    # les variantes sans accent sont acceptees : le controle ne doit pas
    # dependre de l'hygiene typographique de celui qui ecrit le renvoi faux
    dur = re.compile(r"(Th\.|[Tt]h[ée]or[èe]me|[Pp]roposition|[Cc]orollaire"
                     r"|[Ll]emme)~? ?(\d+)")
    for m in dur.finditer(sans_comm):
        ctx = sans_comm[max(0, m.start() - 40): m.end() + 20]
        if ext or any(re.search(e, ctx) for e in EXEMPT):
            continue
        pbs.append(f"chiffre dur : « {m.group(0)} »  ...{' '.join(ctx.split())[-70:]}")

    # -- 2. type du renvoi ---------------------------------------------------
    ren = re.compile(r"(\w+)\.?~?\\(?:eq)?ref\{([^}]+)\}")
    for m in ren.finditer(sans_comm):
        # pluriels francais : « tableaux », « corollaires », « theoremes »
        mot = m.group(1).lower().rstrip(".")
        if mot.endswith("aux"):
            mot = mot[:-3] + "au"
        mot, lab = mot.rstrip("s"), m.group(2)
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


def numerotation(path: str) -> Dict[int, str]:
    """Simule le compteur `theorem` de LaTeX (partage avec `lemma`)."""
    src = re.sub(r"(?<!\\)%.*", "", open(path, encoding="utf-8").read())
    out, n = {}, 0
    for m in re.finditer(r"\\begin\{(theorem|lemma)\}(\[[^\]]*\])?", src):
        n += 1
        out[n] = (m.group(2) or "[]")[1:-1]
    return out


def _mots(t: str) -> set:
    t = re.sub(r"\\[a-zA-Z]+|[^\w\s]", " ", t.lower())
    petits = {"de", "la", "le", "un", "une", "sur", "du", "des", "a", "au",
              "et", "en", "d", "l", "pour", "par"}
    return {w for w in t.split() if len(w) > 2 and w not in petits}


def croise(cite: str, source: str) -> List[str]:
    """Verifie les renvois en chiffres durs de `cite` vers les enonces de
    `source`. Imprime la correspondance, signale ce qui ne colle pas."""
    num = numerotation(source)
    src = re.sub(r"(?<!\\)%.*", "", open(cite, encoding="utf-8").read())
    pbs: List[str] = []
    print(f"    ({source} definit {len(num)} theoremes)")
    vus = set()
    for m in re.finditer(r"(?:Th\.|Théorème)~? ?(\d+)([^\n]{0,60})", src):
        n, suite = int(m.group(1)), m.group(2)
        if n not in num:
            pbs.append(f"« Théorème {n} » : {source} n'en compte que {len(num)}")
            continue
        titre = num[n]
        # une glose suit-elle le numero ?
        g = re.match(r"\s*(?:---|--|—|\()\s*([^)\n]{3,60})", suite)
        etat = "?"
        if g:
            glose = g.group(1)
            etat = "ok" if _mots(glose) & _mots(titre) else "DIVERGE"
            if etat == "DIVERGE":
                pbs.append(f"« Théorème {n} --- {glose.strip()} » mais "
                           f"{source} y met : « {titre} »")
        cle = (n, etat)
        if cle not in vus:
            vus.add(cle)
            print(f"      Th. {n:<2} -> « {titre} »  [{etat}]")

    # Un renvoi sans glose n'est PAS verifie : le controle sait seulement que
    # le numero existe. Le taire serait pire que de ne pas controler, puisque
    # la sortie ressemble alors a une verification. On dit donc ce qui a ete
    # verifie, et on signale tout numero dont AUCUNE occurrence ne l'est.
    cites = {n for (n, _) in vus}
    verifies = {n for (n, e) in vus if e == "ok"}
    aveugles = sorted(cites - verifies)
    print(f"    {len(verifies)}/{len(cites)} numeros cites avec au moins un "
          f"renvoi glose, donc reellement verifies")
    for n in aveugles:
        pbs.append(f"« Théorème {n} » n'est cite que sans glose : le controle "
                   f"ne peut pas verifier qu'il designe « {num[n]} »")
    return pbs


if __name__ == "__main__":
    if sys.argv[1:2] == ["--cross"]:
        cite, source = sys.argv[2], sys.argv[3]
        print(f"=== controle croise : {cite} contre {source}")
        pbs = croise(cite, source)
        for x in pbs:
            print("   ", x)
        print(f"\nTOTAL : {len(pbs)}")
        sys.exit(1 if pbs else 0)

    cibles = sys.argv[1:] or ["hybride.tex", "article.tex"]
    total = 0
    for c in cibles:
        pbs = controle(c)
        durs = [x for x in pbs if not x.startswith("[info]")]
        total += len(durs)
        print(f"=== {c} : {len(durs)} probleme(s)")
        for p in pbs:
            print("   ", p)
    print(f"\nTOTAL : {total}")
    sys.exit(1 if total else 0)
