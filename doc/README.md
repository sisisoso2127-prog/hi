# doc/

`pseudocode.tex` / `pseudocode.pdf` — pseudo-code des dix algorithmes de la
methode, en francais, pret a etre inclus dans le manuscrit.

Chaque algorithme est autonome : on peut copier un seul environnement
`algorithm` sans emporter le reste. Le preambule minimal a reprendre est

    \usepackage[ruled,vlined,linesnumbered]{algorithm2e}

plus les redefinitions de mots-cles francais en tete du fichier (l'option
`[french]` d'algorithm2e ne traduit que le titre dans TeX Live 2023).

Compilation : `pdflatex pseudocode.tex` (deux passes).
Paquets Debian/Ubuntu : `texlive-latex-base texlive-latex-recommended
texlive-latex-extra texlive-science texlive-lang-french`.

| # | algorithme | correspond a |
|---|---|---|
| 1 | `TestEfficacite` | Th. 2 — `molfp_core.efficiency_test` |
| 2 | `Dinkelbach` | Th. 3 — `molfp_core.dinkelbach` |
| 3 | `AjouterCoupe` | Th. 4 + big-M par relaxation continue — `ECutModel.add_dominance_cut` |
| 4 | `Reparer` | `molfp_oracle.repair_to_efficient` |
| 5 | `MaxLineaire` | oracle anytime — `molfp_oracle.max_linear_over_E` |
| 6 | `ResoudreP` | hybride exact-exact — `molfp_oracle.solve_P` |
| 7 | `Certifier` | Th. 5' — `molfp_matheuristic.certify` |
| 8 | `Matheuristique` | `molfp_matheuristic.matheuristic_P` |
| 9 | mouvements A / B / C | `move_epsilon_partial`, `move_epsilon_absolute`, `move_lns` |
| 10 | `ValiderALEchelle` | protocole W1-W6 — `verify_scale.py` |
