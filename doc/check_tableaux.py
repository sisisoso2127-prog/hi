#!/usr/bin/env python3
"""
check_tableaux.py -- les pieds de tableau contre les cellules du tableau.

POURQUOI. Un tableau porte deux choses de nature differente : des cellules,
qui sont un releve, et un pied, qui est un CALCUL sur ces cellules --
« ameliorees 4, degradees 0 », « median +0,01 », « maximum +38,77 ». Quand
un banc est rejoue, les cellules sont regenerees en bloc depuis le journal ;
le pied, lui, est recopie a la main. Il se perime donc seul, et rien ne le
signale : le document compile, les renvois resolvent, et le lecteur lit un
total qui ne correspond plus a ce qu'il a sous les yeux.

Quatre defauts de cette nature ont ete trouves par relecture externe dans ce
projet. Ce programme les trouve sans relecture.

METHODE, ET SA PRUDENCE. Un tableau peut porter plusieurs colonnes de gain
-- deux demi-tableaux cote a cote, ou deux configurations comparees -- et le
pied peut parler de l'une, de l'autre, ou de leur reunion. Le programme
construit donc toutes ces lectures et ne signale une affirmation que si
AUCUNE ne la verifie. Il prefere manquer un defaut a en inventer un : un
verificateur qui crie a tort cesse d'etre lu.

CE QU'IL NE FAIT PAS. Il ne verifie pas les cellules : elles viennent du
journal, et c'est au banc d'en repondre. Il ne juge que la coherence INTERNE
du tableau, ce qui suffit a attraper le pied oublie.
"""

import re
import statistics
import sys
from typing import Dict, List, Optional

TOL = 0.011          # les pieds sont arrondis au centieme


def _nombre(cellule: str) -> Optional[float]:
    """Valeur d'une cellule LaTeX, ou None.

    On retire les NOMS de commande puis TOUTES les accolades, au lieu
    d'apparier \\mathbf{...} : le corps contient lui-meme des accolades,
    comme dans \\mathbf{+17{,}03}, et un appariement naif s'arrete a la
    premiere fermante -- il lisait « +17{, » et rendait None. C'est la
    meme faute que celle corrigee dans check_refs, commise deux fois.
    """
    t = re.sub(r"\\(?:mathbf|textbf|emph|mathit|bm)\b", " ", cellule)
    t = t.replace("\\,", "").replace("\\ ", "").replace("%", "")
    t = t.replace("$", "").replace("{,}", ".")
    t = t.replace("{", "").replace("}", "")
    m = re.fullmatch(r"\s*([+-]?\d+(?:\.\d+)?)\s*", t)
    return float(m.group(1)) if m else None


def _lignes(bloc: str) -> List[List[str]]:
    corps = bloc.split(r"\midrule", 1)[-1].split(r"\bottomrule")[0]
    out = []
    for r in corps.split("\\\\"):
        r = r.replace("\\midrule", " ")
        if "multicolumn" in r or "cmidrule" in r:
            continue
        c = [x.strip().replace("\n", " ") for x in r.split("&")]
        if len(c) > 1:
            out.append(c)
    return out


def _colonnes_gain(bloc: str) -> List[int]:
    tete = bloc.split(r"\midrule", 1)[0]
    for r in reversed(tete.split("\\\\")):
        c = [x.strip().lower() for x in r.split("&")]
        if len(c) < 2:
            continue
        idx = [i for i, x in enumerate(c)
               if re.search(r"\bgain\b|\\delta|\bdelta\b", x)]
        if idx:
            return idx
    return []


def _pied(bloc: str) -> str:
    """Le pied, aplati pour la recherche d'affirmations.

    Les noms de commande sont retires puis TOUTES les accolades, pour la
    raison dite dans _nombre : \\mathbf{+131{,}31} contient lui-meme des
    accolades, et un appariement naif s'arrete a la premiere fermante. Une
    premiere version de ce fichier a fait la faute ici aussi -- deux
    mutations sur quatre passaient au travers, silencieusement.
    """
    p = bloc.split(r"\midrule")[-1] if r"\midrule" in bloc else ""
    p = p.replace("{,}", ",")
    p = re.sub(r"\\(?:mathbf|textbf|emph|mathit|bm)\b", " ", p)
    p = p.replace("{", " ").replace("}", " ")
    return " ".join(p.split())


def _lectures(bloc: str, lignes: List[List[str]]) -> List[List[float]]:
    """Toutes les series de gains que le pied peut legitimement resumer."""
    out = []
    for i in _colonnes_gain(bloc):
        g = [_nombre(L[i]) for L in lignes if len(L) > i]
        g = [x for x in g if x is not None]
        if len(g) >= 3:
            out.append(g)
    if len(out) > 1:
        out.append([x for s in out for x in s])
    return out


def _affirmations(pied: str) -> Dict[str, float]:
    """Ce que le pied affirme, sous forme normalisee."""
    dit: Dict[str, float] = {}
    for mot, cle in (("am[eé]lior[eé]es", "mieux"), ("mieux", "mieux"),
                     ("d[eé]grad[eé]es", "pire"), ("pire", "pire"),
                     ("inchang[eé]es", "egal"), ("[eé]gal", "egal")):
        m = re.search(mot + r"\s*\$?\s*(\d+)", pied)
        if m and cle not in dit:
            dit[cle] = float(m.group(1))
    for mot, cle in (("maximum", "max"), ("minimum", "min"),
                     ("m[eé]dian", "med"), ("moyen", "moy")):
        m = re.search(mot + r"[a-zé]*\s*(?:de\s*)?\$?\s*([+-]?\d+(?:,\d+)?)",
                      pied)
        if m:
            dit[cle] = float(m.group(1).replace(",", "."))
    return dit


def _calcul(g: List[float]) -> Dict[str, float]:
    return {"mieux": sum(1 for x in g if x > 1e-9),
            "pire": sum(1 for x in g if x < -1e-9),
            "egal": sum(1 for x in g if abs(x) <= 1e-9),
            "max": max(g), "min": min(g),
            "med": statistics.median(g), "moy": sum(g) / len(g)}


def controle(path: str) -> List[str]:
    src = re.sub(r"(?<!\\)%.*", "", open(path, encoding="utf-8").read())
    pbs: List[str] = []
    for bloc in re.findall(r"\\begin\{table\}.*?\\end\{table\}", src, re.S):
        lab = re.search(r"\\label\{(tab:[^}]+)\}", bloc)
        nom = lab.group(1) if lab else "(sans etiquette)"
        pied = _pied(bloc)

        # -- arithmetique interne du pied, independante des cellules -------
        mg = re.search(r"somme des gains\s*\$?\s*([+-]?\d+(?:,\d+)?)", pied)
        mp = re.search(r"somme des pertes\s*\$?\s*([+-]?\d+(?:,\d+)?)", pied)
        mn = re.search(r"net\s*\$?\s*([+-]?\d+(?:,\d+)?)", pied)
        if mg and mp and mn:
            g, p, nt = (float(x.group(1).replace(",", ".")) for x in (mg, mp, mn))
            if abs((g + p) - nt) > TOL:
                pbs.append(f"{nom} : net {nt:+.2f}, alors que gains {g:+.2f} "
                           f"et pertes {p:+.2f} donnent {g + p:+.2f}")

        dit = _affirmations(pied)
        if not dit:
            continue
        lectures = _lectures(bloc, _lignes(bloc))
        if not lectures:
            continue
        calculs = [_calcul(g) for g in lectures]
        for cle, v in dit.items():
            if any(abs(c[cle] - v) <= (TOL if cle in ("max", "min", "med",
                                                      "moy") else 0.5)
                   for c in calculs):
                continue
            vus = sorted({round(c[cle], 2) for c in calculs})
            pbs.append(f"{nom} : pied dit {cle} = {v:g}, "
                       f"les cellules donnent {vus}")
    return pbs


def main() -> int:
    total = 0
    for path in (sys.argv[1:] or ["hybride.tex", "article.tex"]):
        pbs = controle(path)
        total += len(pbs)
        print(f"=== {path} : {len(pbs)} probleme(s)")
        for p in pbs:
            print("   ", p)
    print(f"\nTOTAL : {total}")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
