"""Test par mutation de check_sensib.py.

On fabrique un journal QUI CONCORDE avec le tableau publie, on verifie que
le fascheur ne dit rien, puis on abime le tableau d'une seule cellule a la
fois et on verifie qu'il le dit. Un fascheur qui passe ce test detecte au
moins les fautes qu'on sait commettre ; il ne prouve rien sur les autres.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, ".")
import check_sensib as cs

DOC = Path("hybride.tex")
SCR = Path(tempfile.mkdtemp())


def journal_depuis(rangs) -> Path:
    """Le format exact de bench_sensibilite.py, colonne pour colonne."""
    out, vu = [], set()
    out.append("=" * 108)
    out.append("SENSIBILITE AUX REGLAGES")
    out.append("=" * 108)
    out.append(f"{'facteur':<30}{'valeur':>22}{'A preuves':>11}"
               f"{'A ecart':>9}{'B ecart':>9}{'B delta':>9}"
               f"{'B mieux/pire':>14}")
    out.append("-" * 108)
    for fac, val, pre, eb, db, mp, prod in rangs:
        nom = "" if fac in vu else fac
        vu.add(fac)
        delta = 0.0 if prod else db
        compte = "0/0" if prod else mp
        marque = "  <- production" if prod else ""
        out.append(f"{nom:<30}{val:>22}{pre:>5}/{24:<5}"
                   f"{0.0:>9.2f}{eb:>9.2f}{delta:>+9.2f}"
                   f"{compte:>14}{marque}")
    p = SCR / "journal.out"
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


def lance(journal: Path, doc: Path) -> tuple[int, str]:
    r = subprocess.run([sys.executable, "check_sensib.py",
                        str(journal), str(doc)],
                       capture_output=True, text=True)
    return r.returncode, r.stdout


def mute(nom: str, vieux: str, neuf: str) -> Path:
    txt = DOC.read_text(encoding="utf-8")
    if txt.count(vieux) != 1:
        raise SystemExit(f"[{nom}] motif non unique : {txt.count(vieux)}")
    p = SCR / f"{nom}.tex"
    p.write_text(txt.replace(vieux, neuf), encoding="utf-8")
    return p


rangs = cs.lire_tableau(str(DOC), "tab:sensib")
journal = journal_depuis(rangs)

code, sortie = lance(journal, DOC)
print("TEMOIN (journal concordant) :", "OK" if code == 0 else "ECHEC")
if code != 0:
    print(sortie)
    raise SystemExit("le temoin echoue : le fascheur ou le generateur est faux")

MUTATIONS = [
    ("ecart_B",     r" & $2$ & $14/24$ & $8{,}11$",
                    r" & $2$ & $14/24$ & $8{,}12$"),
    ("preuves_A",   r" & $2$ & $14/24$ & $8{,}11$",
                    r" & $2$ & $15/24$ & $8{,}11$"),
    ("delta_B",     r"$8{,}11$ & $-4{,}03$ & $1/3$",
                    r"$8{,}11$ & $-4{,}13$ & $1/3$"),
    ("mieux_pire",  r"$8{,}11$ & $-4{,}03$ & $1/3$",
                    r"$8{,}11$ & $-4{,}03$ & $1/2$"),
    ("etiquette",   r" & $2$ & $14/24$ & $8{,}11$",
                    r" & $3$ & $14/24$ & $8{,}11$"),
    ("signe_delta", r" & $16$ & $13/24$ & $12{,}93$ & $+0{,}79$",
                    r" & $16$ & $13/24$ & $12{,}93$ & $-0{,}79$"),
    ("cellule_gras",
     r" & $\mathbf{8}$ & $13/24$ & $12{,}14$ & --- & \textbf{production}",
     r" & $\mathbf{8}$ & $14/24$ & $12{,}14$ & --- & \textbf{production}"),
    ("production_perdue",
     r" & $\mathbf{8}$ & $13/24$ & $12{,}14$ & --- & \textbf{production}",
     r" & $\mathbf{8}$ & $13/24$ & $12{,}14$ & $+0{,}00$ & $0/0$"),
    ("ligne_supprimee",
     "\n & $32$ & $13/24$ & $12{,}93$ & $+0{,}79$ & $1/3$\\\\", ""),
    ("beta_80",     r" & $80$ & $13/24$ & $10{,}94$ & $-1{,}19$ & $4/0$",
                    r" & $80$ & $13/24$ & $10{,}94$ & $-1{,}19$ & $3/0$"),
    ("rho_4",       r" & $4$ & $13/24$ & $10{,}94$ & $-1{,}19$ & $4/0$",
                    r" & $4$ & $13/24$ & $11{,}94$ & $-1{,}19$ & $4/0$"),
    # deux lignes echangees : l'appariement etant positionnel, il doit
    # signaler les DEUX, pas se recaler silencieusement
    ("echange_de_lignes",
     " & $16$ & $13/24$ & $12{,}93$ & $+0{,}79$ & $1/2$\\\\\n"
     " & $32$ & $13/24$ & $12{,}93$ & $+0{,}79$ & $1/3$\\\\",
     " & $32$ & $13/24$ & $12{,}93$ & $+0{,}79$ & $1/3$\\\\\n"
     " & $16$ & $13/24$ & $12{,}93$ & $+0{,}79$ & $1/2$\\\\"),
]

# Mutations que ce fascheur NE PEUT PAS attraper, par construction. On les
# joue quand meme : un test par mutation qui ne montre que ses reussites
# laisse croire a une couverture qu'il n'a pas.
AVEUGLES = [
    ("nom_de_facteur",
     r"\multirow{5}{*}{$k_{\text{hasard}}$}",
     r"\multirow{5}{*}{$k_{\text{bidon}}$}"),
    ("les_deux_ensemble", None, None),
]

attrapes = 0
for nom, vieux, neuf in MUTATIONS:
    doc = mute(nom, vieux, neuf)
    code, sortie = lance(journal, doc)
    ok = code != 0
    attrapes += ok
    detail = ""
    if ok:
        m = re.search(r"TOTAL : (\d+)", sortie)
        n = re.search(r"nombre de lignes different", sortie)
        detail = ("lignes" if n else f"{m.group(1)} divergence(s)"
                  if m else "")
    print(f"  {nom:<20}{'ATTRAPEE' if ok else 'MANQUEE ':<10}{detail}")

print(f"\nSCORE : {attrapes}/{len(MUTATIONS)}")

print("\nANGLES MORTS -- attendus, et verifies comme tels :")
for nom, vieux, neuf in AVEUGLES:
    if vieux is None:
        # la meme faute commise dans le journal ET dans le tableau : ils
        # concordent, et aucune comparaison ne peut les departager
        r2 = [list(x) for x in rangs]
        r2[0][3] = 99.99
        j2 = journal_depuis([tuple(x) for x in r2])
        doc2 = mute("les_deux", r" & $2$ & $14/24$ & $8{,}11$",
                    r" & $2$ & $14/24$ & $99{,}99$")
        code, _ = lance(j2, doc2)
    else:
        doc2 = mute(nom, vieux, neuf)
        code, _ = lance(journal, doc2)
    print(f"  {nom:<20}{'non vue (attendu)' if code == 0 else 'VUE -- le test est a revoir'}")

print("""
Le premier angle mort est assume : rapprocher les intitules du journal
(« bases au hasard    k_rand ») de ceux du tableau ($k_{hasard}$)
demanderait une table de correspondance a maintenir, donc une source de
faux negatifs. L'appariement reste positionnel, et le fichier le dit.
Le second est indepassable : un journal perime et un tableau perime
concordent. Seul rejouer le banc y remedie.""")
raise SystemExit(0 if attrapes == len(MUTATIONS) else 1)
