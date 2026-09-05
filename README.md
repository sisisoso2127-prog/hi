# Matheuristique pour l'optimisation fractionnaire sur l'ensemble efficace d'un MOILFP

Socle logiciel de la these. Cette premiere livraison contient **uniquement les
briques exactes validees** : generateur d'instances, verite terrain par
enumeration exhaustive, test d'efficacite integral et Dinkelbach exact.
Aucune heuristique n'est encore implementee — c'est volontaire.

## Probleme

    (MOILFP)  max  Z_k(x) = (c_k^T x + a_k)/(d_k^T x + b_k),  k = 1..p
              s.c. A x <= b,  x in Z^n_+

    (P)       max  f(x) = (c^T x + a)/(d^T x + b)   s.c.  x in E

ou `E` est l'ensemble des solutions efficaces de (MOILFP).

## Fichiers

| Fichier | Role |
|---|---|
| `molfp_instance.py` | Structures `FracObj` / `MOILFP`, generateur aleatoire, calibration, E/S JSON |
| `molfp_core.py` | Theoremes 1-3 : seuils integraux, test d'efficacite, Dinkelbach exact, minorant LP |
| `molfp_enum.py` | Verite terrain : enumeration de S, filtre de Pareto exact, `q*` de reference |
| `molfp_oracle.py` | Oracle lineaire sur E (coupes de dominance, bornes anytime) + hybride exact-exact |
| `molfp_matheuristic.py` | Volet exact-metaheuristique : recherche certifiee (Th. 2) + bornes (Th. 5/5') |
| `verify.py` | Validation V1-V4 des briques de base |
| `verify_oracle.py` | Validation V5-V8 de l'oracle, de l'hybride et du big-M |
| `verify_math.py` | Validation V8-V11 de la matheuristique + comparaison a budget egal |
| `bench_bound.py` | Th. 5 (temoin) vs Th. 5' (resserre), a budget identique |
| `bench_improve.py` | Mesure multi-graines sur le regime cible ; sert aussi de temoin d'A/B |
| `legacy/` | Prototype initial a deux variables, conserve pour tracabilite |

Lancer : `python verify.py` puis `python verify_oracle.py`
(dependances : `pip install -r requirements.txt` — numpy, scipy >= 1.9).

`scaling.py`, `campaign.py` et `analyze.py`, dont les resultats sont
rapportes plus bas, ne sont pas encore dans ce depot : les tableaux de la
section « Campagne de difficulte » proviennent de la campagne d'origine et
sont conserves ici comme resultats acquis, non comme sorties reproductibles
en l'etat.

## Hypotheses garanties par construction

* **(A1)** `d_k >= 0` et `b_k >= 1` donc `D_k(x) >= 1 > 0` sur tout le domaine.
* **(A2)** `A >= 0` avec colonnes non nulles donc `S` non vide et borne
  (`x = 0` realisable, bornes explicites via `var_upper_bounds()`).

Ces deux hypotheses ne sont pas des commodites : elles conditionnent la
validite des theoremes 1 et 2. Toute instance externe doit passer
`check_assumptions()`.

## Les trois resultats implementes

### Theoreme 1 — linearisation de seuil integrale

Sous (A1), pour `v = N/D` avec `D > 0` entier :

    Z_k(x) >= v   <=>   (D c_k - N d_k)^T x  >=  N b_k - D a_k

Les contraintes de type epsilon sont donc **entierement lineaires et a
coefficients entiers**. C'est ce qui rend possible l'usage d'un solveur ILP
standard a chaque appel, au lieu d'un solveur fractionnaire.

### Theoreme 2 — test d'efficacite exact, un seul ILP

Pour `xbar` realisable, `N_k = N_k(xbar)`, `D_k = D_k(xbar) > 0` :

    theta(xbar) = max  sum_k [ (D_k c_k - N_k d_k)^T x + D_k a_k - N_k b_k ]
                  s.c. (D_k c_k - N_k d_k)^T x >= N_k b_k - D_k a_k,  k = 1..p
                       A x <= b,  x in Z^n_+

Le terme `k` vaut `D_k(xbar) * D_k(x) * (Z_k(x) - Z_k(xbar))`, de meme signe que
`Z_k(x) - Z_k(xbar)` puisque les deux denominateurs sont strictement positifs.
Donc **`xbar` efficace <=> `theta(xbar) = 0`**.

Point de mise en oeuvre important : tous les coefficients etant entiers,
`theta` est un **entier**. Le test est donc exact, sans tolerance numerique —
contrairement a une formulation directe en `v_k = N_k/D_k` flottant. En cas de
non-efficacite, le programme fournit gratuitement une solution dominante,
exploitable comme mouvement de recherche locale.

### Theoreme 3 — Dinkelbach a parametre rationnel

`F(q) = max_{x in X} { N(x) - q D(x) }` est le sup d'un nombre **fini** de
fonctions affines de `q` : convexe, lineaire par morceaux, strictement
decroissante, de racine unique `q* = max_X f`. L'iteration `q_{t+1} = f(x_t)`
converge en un **nombre fini** d'iterations — resultat plus fort que la
convergence superlineaire du cas continu, et qui vient precisement de la
finitude de `E`.

Avec `q = P/Q` rationnel, le sous-probleme s'ecrit

    max (Q num - P den)^T x + (Q a - P b)

a coefficients entiers, de valeur optimale entiere : le critere d'arret
`F(q) == 0` est teste **exactement**.

Consequence structurante : chaque iteration revient a **maximiser une fonction
lineaire sur l'ensemble efficace entier** — exactement le probleme traite par
Zerdani & Moulai (2011). L'oracle interne de l'hybride exact-exact existe deja
dans la litterature.

## Validation

`verify.py` verifie sur chaque instance :

* **V1** — pour tout `x` teste : `theta(x) == 0 <=> x in E` (force brute).
  Verifie aussi que le dominateur renvoye domine reellement.
* **V2** — Dinkelbach sur `S` retrouve exactement `max_S f`.
* **V3** — validite de la borne `q_UB^(0) = max_S f >= q*`.
* **V4** — point ideal par Dinkelbach == point ideal par force brute.

### Resultats (8 instances, tout valide)

| instance | \|S\| | \|E\| | \|E\|/\|S\| | iterations Dinkelbach | ecart relaxation |
|---|---|---|---|---|---|
| n4 m3 p2 | 817 | 1 | 0.001 | 2 | **92.55 %** |
| n4 m3 p3 | 415 | 50 | 0.120 | 3 | 0.00 % |
| n5 m3 p3 | 1833 | 187 | 0.102 | 3 | 40.71 % |
| n5 m4 p4 | 1080 | 32 | 0.030 | 3 | **86.81 %** |
| n6 m4 p3 | 2878 | 29 | 0.010 | 2 | 36.00 % |
| n6 m4 p5 | 3932 | 1324 | 0.337 | 3 | 0.00 % |
| n8 m4 p3 | 3329 | 67 | 0.020 | 4 | 22.06 % |
| n8 m5 p4 | 6968 | 763 | 0.110 | 3 | 0.00 % |

## Deux constats a exploiter dans la these

**1. Dinkelbach converge en 2 a 4 iterations.** La convergence finie n'est pas
seulement theorique, elle est tres rapide en pratique. Le cout total de
l'hybride exact-exact est donc essentiellement celui de **3 a 4 appels** a
l'oracle « lineaire sur l'ensemble efficace ». C'est l'argument central du
premier article.

**2. L'ecart de la relaxation `E -> S` est erratique : de 0 % a 92 %.**
C'est le resultat le plus utile de cette campagne. La borne superieure triviale
`max_S f` est parfois exacte, parfois inutilisable. Elle ne peut donc pas
servir seule de certificat de qualite : il faut construire des bornes
superieures **resserrees progressivement** par accumulation de coupes
efficaces. C'est la justification empirique du volet « garantie de qualite /
comportement anytime » de la these — volet absent de toute la litterature
MOILFP existante.

On note aussi que `|E|/|S|` varie de 0.001 a 0.34 : la difficulte d'une
instance n'est pas donnee par `n` seul. Prevoir un descripteur de difficulte
(densite du front, `p`, `|E|` estime) pour construire un lot d'instances
equilibre.

> Ces deux constats reposent sur 6 instances. Ils sont repris, corriges et
> quantifies sur 216 instances dans la section « Campagne de difficulte »
> plus bas — qui **invalide** au passage l'hypothese « `E` mince = instance
> dure » prise isolement.

## Theoreme 4 — coupe de dominance fractionnaire exacte

Soit `xbar` realisable et **domine**. Avec `Nbar_k = N_k(xbar)`, `Dbar_k = D_k(xbar)` :

    e_k(x) = (Dbar_k c_k - Nbar_k d_k)^T x + (Dbar_k a_k - Nbar_k b_k)
           = Dbar_k * D_k(x) * ( Z_k(x) - Z_k(xbar) )

`e_k` est a valeurs **entieres**, donc `Z_k(x) > Z_k(xbar) <=> e_k(x) >= 1`.
Aucun pas minimal `delta` n'est a estimer : c'est la transposition exacte au
cas fractionnaire de la coupe « +1 » du cas lineaire entier.

**Lemme de validite.** Si `xbar` est domine, `{ x in S : Z(x) <= Z(xbar) }` ne
contient aucune solution efficace.
*Preuve.* Soit `y` dominant `xbar`, et `x` avec `Z(x) <= Z(xbar)`. Alors
`Z(y) >= Z(xbar) >= Z(x)`. Si `Z(y) = Z(x)`, `Z(xbar)` est coince entre les
deux donc `Z(y) = Z(xbar)`, contredisant que `y` domine `xbar`. Donc `y`
domine `x`. ∎

C'est ce lemme qui rend la coupe sure : on ne coupe **jamais** autour d'un
point efficace, donc la question des solutions alternatives de meme vecteur
criteres ne se pose pas.

## Oracle lineaire sur E

    R <- S
    repeter
        xbar <- argmax g sur R                    ->  UB = g(xbar)
        si xbar efficace (Th. 2) : optimal, stop
        sinon : remonter la chaine de dominance   ->  point efficace, LB
                ajouter la coupe de dominance de xbar (Th. 4)
    jusqu'a  UB - LB <= tol  ou  R vide

*Correction* : invariant `E ⊆ R`. A l'arret avec `xbar` efficace,
`g(xbar) = max_R g >= max_E g` et `xbar in E`, donc `xbar` est optimal.
*Terminaison* : chaque coupe retire au moins `xbar`, et `S` est fini.

**Bornes anytime** : a chaque iteration `LB <= max_E g <= UB`. L'algorithme
peut s'arreter a tout instant avec un ecart d'optimalite garanti — et se
termine souvent parce que `UB` rejoint `LB`, sans que la relaxation ait eu
besoin de produire un point efficace.

## Hybride exact-exact (`solve_P`)

Boucle externe Dinkelbach (Th. 3) ; boucle interne l'oracle ci-dessus, car le
sous-probleme `max { Q N(x) - P D(x) }` est **lineaire**. Les coupes de
dominance ne dependent que de `E`, pas de l'objectif : elles sont **conservees
d'une iteration Dinkelbach a l'autre** (`reuse_cuts=True`). C'est le point
d'hybridation qui rend l'ensemble economique.

## Validation

`verify.py` : V1 test d'efficacite vs force brute — V2 Dinkelbach sur S —
V3 validite de la borne — V4 point ideal.
`verify_oracle.py` : V5 oracle vs `max_E g` force brute (plusieurs `g`
aleatoires) — V6 hybride vs `q*` force brute — V7 archive sans faux positif —
V8 big-M des coupes contre le minimum reel de `e_k` sur `S` enumere.

**Tout est valide sur 8 instances (V1-V4) et 6 instances (V5-V8)**
(`results/verify.out`, `results/verify_oracle.out`).

### Oracle et hybride

| instance | \|S\| | \|E\| | Dinkelbach | coupes | ILP | t(s) | archive | couverture de E |
|---|---|---|---|---|---|---|---|---|
| n4 m3 p2 | 817 | 1 | 1 | 111 | 338 | 6.4 | 1 | 100 % |
| n4 m3 p3 | 415 | 50 | 2 | 0 | 7 | 0.01 | 2 | 4 % |
| n5 m3 p3 | 1833 | 187 | 2 | 11 | 47 | 0.4 | 8 | 4 % |
| n5 m4 p4 | 1080 | 32 | 2 | 45 | 145 | 5.6 | 18 | 56 % |
| n6 m4 p3 | 2878 | 29 | 2 | 58 | 182 | 9.6 | 6 | 21 % |
| n8 m4 p3 | 3329 | 67 | 3 | 78 | 245 | 15.2 | 15 | 22 % |

### Passage a l'echelle (limite 45 s par instance)

| instance | statut | Dinkelbach | coupes | ILP | t(s) |
|---|---|---|---|---|---|
| n8 m5 p3 | **limite** | 2 | 135 | 414 | 45.0 |
| n10 m5 p3 | optimal | 2 | 0 | 7 | **0.1** |
| n12 m6 p3 | optimal | 3 | 50 | 161 | 9.5 |
| n12 m6 p4 | **limite** | 2 | 85 | 342 | 45.7 |
| n15 m6 p3 | optimal | 1 | 129 | 391 | 45.4 |
| n20 m8 p3 | **limite** | 2 | 103 | 354 | 45.1 |

## Campagne de difficulte (216 instances)

`campaign.py` / `analyze.py` / `campaign.csv` / `analyze.out`

**Plan d'experience controle.** Le parametre `corr` du generateur pilote la
similarite entre criteres, donc la finesse de `E`, **a domaine `S`
rigoureusement identique** (`A` et `b` ne dependent pas de `corr`). On
manipule donc la cause supposee au lieu de l'observer.

Grille : `n` in {5,6,7,8} x `p` in {2,3,4} x `corr` in {0, .25, .5, .75, .9, 1}
x 3 graines = 216 instances, verite terrain par enumeration exhaustive,
limite de 12 s par resolution.

**Validation : 0 divergence avec la force brute** sur les 199 instances
resolues a l'optimum (17 arrets sur limite de temps, honnetement rapportes).

### Un bug de surete decouvert par la campagne

Trois instances ont d'abord ete signalees `MISMATCH`. Diagnostic : quand
l'oracle interne atteignait sa limite de temps, il renvoyait `value = LB`,
une simple **borne inferieure** de `F(q)`. La boucle Dinkelbach la traitait
comme la valeur exacte et concluait `F(q) <= 0` donc « optimal » — a tort.

Correctif (`solve_P`) :
* `Fq > 0` reste valide quel que soit le statut, car `F(q) >= Fq > 0` ;
* `Fq <= 0` ne conclut que si l'oracle a **prouve** l'optimalite ;
* sinon le statut `limit` est propage.

C'est exactement le type de faute que le socle de verite terrain existe pour
attraper : sans lui, la methode aurait publie des optima faux avec un statut
« optimal ».

### Traitement de la censure (point methodologique)

Les 17 instances arretees sur limite de temps ont un cout **censure** : leur
nombre d'appels ILP est une borne inferieure, pas une mesure. Les inclure dans
une moyenne ou une correlation revient a remplacer le cout des instances les
plus dures par une valeur tronquee, et **les strates qui echouent le plus
paraissent alors les plus faciles**. C'est un biais de survie, et il a
d'abord fausse la lecture de cette campagne.

On separe donc deux questions, qui n'ont pas la meme reponse :

* **Q1** parmi les instances resolues, qu'est-ce qui fait monter le cout ?
* **Q2** qu'est-ce qui fait qu'une instance n'est pas resolue du tout ?

### Q1 — cout parmi les instances resolues (censurees exclues)

| predicteur | Spearman avec les appels ILP |
|---|---|
| `relax_gap` = (max_S f − q*)/\|max_S f\| | +0.606 *** |
| `theta_S` (un seul ILP) | +0.410 *** |
| `depth_S` (quelques ILP) | +0.373 *** |
| ratio \|E\|/\|S\| | **−0.353** *** |
| \|E\| absolu | **−0.307** *** |
| `p` | −0.272 *** |
| `n` | +0.268 *** |
| \|S\| | +0.173 * |

Le cout monte quand `E` est **mince** : la relaxation est alors loin de `E` et
il faut beaucoup de coupes pour l'y ramener.

`relax_gap` fait intervenir `q*`, donc la reponse cherchee : c'est un
indicateur de **mecanisme**, jamais un predicteur utilisable avant resolution.

### Q2 — echec total : le sens inverse

| variable | mediane censuree | mediane resolue | Mann-Whitney |
|---|---|---|---|
| \|E\| | **26.0** | **4.0** | p = 1.2e−03 ** |
| ratio \|E\|/\|S\| | 0.0139 | 0.0017 | p = 7.7e−03 ** |
| `n` | 7 | 6 | p = 3.2e−02 * |
| \|S\| | 2387 | 1931 | p = 5.3e−02 (n.s.) |
| `theta_S` | 3229 | 2819 | p = 0.42 (n.s.) |
| `depth_S` | 1 | 1 | p = 0.80 (n.s.) |

**Deux regimes distincts, de sens opposes.** Un `E` mince rend chaque
resolution moderement plus chere ; un `E` **gros** rend la resolution
impossible dans le temps imparti, car l'oracle doit couper un a un un grand
nombre de points non domines et derive vers un comportement d'enumeration.

Consequence lourde : `theta_S` et `depth_S`, les deux predicteurs bon marche,
correlent avec le cout des instances resolues mais **n'ont aucun pouvoir de
prediction sur l'echec** (p = 0.42 et 0.80). Ils detectent le facile, pas le
catastrophique. C'est un resultat negatif, et il est utile : il delimite ce
qu'il reste a trouver.

### Q3 — experience controlee sur `corr`

| `corr` | \|E\| median | timeouts | ILP median (resolues) |
|---|---|---|---|
| 0.00 | 44.5 | **6** | 29.5 |
| 0.25 | 47.0 | **4** | 34.5 |
| 0.50 | 18.5 | **6** | 59.5 |
| 0.75 | 2.0 | 1 | 83.0 |
| 0.90 | 1.0 | 0 | 65.5 |
| 1.00 | 1.0 | 0 | 53.0 |

Des criteres correles font **s'effondrer `E`** (mediane 44 -> 1) et le timeout
**disparait** : les 17 echecs sont tous a `corr <= 0.75`. La correlation entre
criteres est donc le levier structurel de la difficulte — mais son effet
n'apparait que si l'on regarde le taux d'echec, pas la moyenne des couts.
Lue sur la seule colonne des couts, elle semble au contraire rendre les
instances plus cheres (29.5 -> 83.0), ce qui n'est que l'effet du biais de
survie : a faible `corr`, les instances cheres sont censurees et sortent de
la mediane.

### Q4 — la taille ne predit rien

| `n` | ILP median | min | max | % timeout |
|---|---|---|---|---|
| 5 | 41 | 5 | 214 | 1.9 |
| 6 | 44 | 7 | 236 | 5.6 |
| 7 | 57 | 5 | **359** | 13.0 |
| 8 | 91 | 10 | 211 | 11.1 |

A `n` fixe le cout couvre deux ordres de grandeur. Rapporter les performances
en fonction de la seule taille ne permet aucune prediction utile.

### Confirmations

* **Dinkelbach : 1 a 4 iterations externes** (mediane 1, moyenne 1.53) sur
  216 instances. La convergence finie du Th. 3 est rapide en pratique.
* **Archive** : couverture mediane de `E` de 66.7 %, sans aucun faux positif.

## Ce que la campagne change pour la these

**1. L'argument en faveur de la matheuristique est renforce, mais reformule.**
Le cout est gouverne par une quantite inaccessible avant resolution, il varie
de deux ordres de grandeur a taille fixe, et le risque d'echec total suit un
axe **oppose** a celui du cout. Aucun reglage a priori ne peut donc proteger
des deux regimes a la fois. C'est exactement le regime ou un schema **anytime
avec garantie d'ecart** vaut mieux qu'une methode exacte tout-ou-rien.

**2. La cible de la matheuristique est identifiee : le regime a `E` gros.**
C'est la que la methode exacte echoue, et c'est aussi la que l'archive de
solutions efficaces a le plus de valeur pour le decideur. Le volet
metaheuristique doit etre concu et evalue **sur ce regime**, pas sur la
moyenne du lot.

**3. Un descripteur de difficulte publiable reste a construire.** `theta_S` et
`depth_S` trient le facile ; rien ne trie le catastrophique. Contribution
methodologique ouverte et bien delimitee.

**4. Le protocole de comparaison doit inclure `corr`.** Deux lots de meme
`(n, m, p)` mais de `corr` different ont des `|E|` dans un rapport de 1 a 40.
Toute comparaison entre methodes qui ignore ce facteur est ininterpretable.

## Volet exact-metaheuristique (matheuristique)

`molfp_matheuristic.py` / `verify_math.py`

Cible : le regime identifie par la campagne, `|E|` gros, ou la methode exacte
derive vers l'enumeration et n'aboutit pas.

### Theoreme 5 — d'une borne sur le sous-probleme a une borne sur `q*`

Soit `q = P/Q` atteinte par un point efficace (donc `q <= q*`), `U` une borne
superieure valide de `F(q) = max_{x in E} { Q N(x) - P D(x) }`, et
`Dmin = min_{x in S} D(x) > 0` sous (A1). Alors

        q*  <=  q + U / (Q * Dmin)

*Preuve.* Pour tout `x` de `E`, `Q N(x) - P D(x) <= U`. En divisant par
`Q D(x) > 0` : `f(x) - q <= U/(Q D(x)) <= U/(Q Dmin)`, la derniere inegalite
valant car `U >= 0`. Passage au max sur `E`. ∎

Portee : il n'est plus necessaire de resoudre le sous-probleme a l'optimum.
Un oracle **interrompu** fournit un `U`, donc un ecart garanti sur `q*`.
`Dmin` coute un seul ILP. C'est ce qui rend la matheuristique *certifiee*.

### Une erreur de conception, et sa correction

La premiere version prenait pour voisinage

        V_k(x^r) = { x : e_k(x) >= 1,  e_j(x) >= 0 pour j != k }

soit « mieux sur `k` sans rien perdre ailleurs ». C'est **exactement
l'ensemble des points qui dominent `x^r`** : vide des que `x^r` est efficace.
Le voisinage etait donc vide par construction, verifie sur instance :
`V_0, V_1, V_2` tous vides. Resultat, la recherche ne bougeait pas — optimum
atteint 1 fois sur 6, ecarts reels jusqu'a 95 %.

Correction : un arbitrage sur les autres criteres est indispensable. Trois
mouvements, tous a sous-probleme ILP resolu exactement puis certifie par le
Th. 2 :

| | mouvement | contraintes |
|---|---|---|
| **A** | epsilon partiel | `e_k(x) >= 1` + planchers `e_j(x) >= 0` sur un sous-ensemble **strict** des autres criteres |
| **B** | plancher absolu | `Z_j(x) >= eps_j` tire entre nadir et ideal (Th. 1) |
| **C** | LNS fix-and-optimize | fige 60 % des variables a l'incumbent, resout le reste |

La taille du sous-ensemble dans A regle l'intensification : vide = mouvement
le plus explorateur, plein = voisinage vide.

### Validation (8 instances, verite terrain par enumeration)

| test | resultat |
|---|---|
| **V8** incumbent reellement efficace et `q_lb <= q*` | 8/8 |
| **V9** borne du Th. 5 valide : `q_ub >= q*` | 8/8 |
| **V10** archive sans faux positif | 8/8 |
| **V11** optimum atteint | 8/8 |

Le LB n'est **jamais** optimiste : l'incumbent est un point efficace certifie
par le Th. 2, meme si la recherche est interrompue.

### Matheuristique vs exact, budget egal (12 s), sur le regime cible

Les 8 instances sont celles ou la methode exacte a **echoue** dans la
campagne (statut `limit`), reprises a l'identique.

| instance | \|E\| | `q*` | exact | matheuristique |
|---|---|---|---|---|
| n7 m4 p4 c0.00 | 218 | 8.87500 | 1.73684 | **8.87500** |
| n7 m4 p4 c0.25 | 198 | 4.73333 | 3.92308 | **4.73333** |
| n6 m4 p4 c0.00 | 185 | 5.41667 | 5.41667 | **5.41667** |
| n7 m4 p3 c0.25 | 112 | 4.75000 | 3.96296 | **4.75000** |
| n5 m3 p4 c0.00 | 54 | 3.30612 | 3.18519 | **3.30612** |
| n8 m5 p3 c0.00 | 46 | 3.17391 | 2.84000 | **3.17391** |
| n7 m4 p4 c0.50 | 30 | 4.73333 | 2.27586 | **4.73333** |
| n6 m4 p2 c0.25 | 26 | 2.38462 | 1.94444 | **2.38462** |

*La colonne exacte est en statut `limit` sur les 8 : ces valeurs ne sont pas
prouvees. La colonne matheuristique, elle, est certifiee efficace (Th. 2) sur
les 8.*

| | exact | matheuristique |
|---|---|---|
| ecart reel a `q*` (mediane) | 16.8 % | **0.0 %** |
| ecart reel a `q*` (moyenne) | 24.8 % | **0.0 %** |
| optimum atteint | 1/8 | **8/8** |

> Ce tableau a ete regenere apres l'iteration decrite plus bas (« Iteration
> suivante »). La version qui l'avait produit la premiere fois atteignait
> l'optimum sur 6 instances sur 8 et perdait sur une, avec un ecart reel
> moyen de 8.3 %.

### Variance stochastique : ne jamais rapporter un run unique

Sur l'instance la plus dure du lot (`n7 m4 p4 c0.00`, `|E| = 218`,
`q* = 8.875`), la version d'alors donnait, sur 6 graines a budget identique :

    8.875   4.529   8.875   8.875   8.875   4.529
    optimum atteint 4/6, ecart median 0.0 %, ecart maximal 49.0 %

Un tableau a **une seule graine** illustre un comportement, il ne le mesure
pas : toute campagne definitive doit rapporter mediane et dispersion sur au
moins 10 graines par instance. C'est cette lecture qui a fait de la variance
une cible a part entiere ; `bench_improve.py` l'applique systematiquement, et
la section « Iteration suivante » rapporte 10/10 sur cette meme instance.

### Theoreme 5' — resserrement de la borne

Le Th. 5 etait valide mais lache d'un facteur ~59. Il gaspillait trois
informations. Soit `R` un relache de `E` (R contient E) obtenu par
accumulation de coupes de dominance (Th. 4). Posons

        U   >=  F_R(q) = max_{x in R} { Q N(x) - P D(x) }
        D+  =   min { D(x) : x in R,  Q N(x) - P D(x) >= 0 }

Alors

        q*  <=  q + U / (Q * D+)     et si U = 0, alors q* = q.

*Preuve.* Soit `x` dans `E`, donc dans `R`. Si `Q N(x) - P D(x) < 0` alors
`f(x) < q`. Sinon `x` appartient a la region definissant `D+`, donc
`D(x) >= D+` et `f(x) - q = (Q N(x) - P D(x)) / (Q D(x)) <= U / (Q D+)`. ∎

Trois gains :

1. **`R` au lieu de `S`** — `U` plus petit, les coupes retirant des zones
   sans aucune solution efficace.
2. **`D+` au lieu de `Dmin`** — on ne minimise `D` que la ou la borne agit,
   c'est-a-dire la ou `f` depasse `q`. Le minimum portant sur un
   sous-ensemble, `D+ >= Dmin` : la borne est mecaniquement plus fine.
   Mesure sur le lot : `Dmin -> D+` passe de 1 a 12, 1 a 13, 5 a 10, 7 a 24.
3. **Recyclage des points domines** — les chaines de reparation du Th. 2
   traversent des points **domines**, seuls points sur lesquels le Th. 4
   autorise une coupe. Les jeter gaspillait une information deja payee. On
   les rejoue comme coupes, **tries par valeur decroissante du substitut** :
   ce sont eux qui tirent `U` vers le haut, donc les couper resserre le plus.

La region definissant `D+` n'est jamais vide (l'incumbent y est, son residu
valant 0) : une infaisabilite signalerait un bug, pas une preuve.

### Effet mesure (memes 8 instances, budget identique)

| instance | \|E\| | ecart reel | Th. 5 | **Th. 5'** | valide | coupes | `Dmin` -> `D+` |
|---|---|---|---|---|---|---|---|
| n7 m4 p4 c0.00 | 218 | 0.0 % | 0.0 % | **0.0 %** | ok | 40 | prouve |
| n7 m4 p4 c0.25 | 198 | 0.0 % | 51.8 % | **27.3 %** | ok | 70 | 5 -> 11 |
| n6 m4 p4 c0.00 | 185 | 0.0 % | 16.2 % | **0.0 %** | ok | 50 | 7 -> 23 |
| n7 m4 p3 c0.25 | 112 | 0.0 % | 66.7 % | **0.9 %** | ok | 86 | 1 -> 12 |
| n5 m3 p4 c0.00 | 54 | 0.0 % | 74.5 % | **5.1 %** | ok | 38 | 1 -> 27 |
| n8 m5 p3 c0.00 | 46 | 0.0 % | 48.2 % | **0.0 %** | ok | 59 | 6 -> 14 |
| n7 m4 p4 c0.50 | 30 | 0.0 % | 0.0 % | **0.0 %** | ok | 34 | prouve |
| n6 m4 p2 c0.25 | 26 | 0.0 % | 12.3 % | **0.0 %** | ok | 22 | prouve |

| | Th. 5 | Th. 5' |
|---|---|---|
| ecart garanti median | 32.2 % | **0.0 %** |
| optimalite **prouvee** | 0/8 | **5/8** |

`coupes` compte ici toutes les coupes portees par le modele au dernier tour :
celles recyclees des chaines de reparation **et** celles que l'oracle pose
lui-meme. Les deux colonnes `Th. 5` / `Th. 5'` sont produites par le meme
code au meme budget, seule la formule de la borne differe.

**Reduction mediane de l'ecart garanti : 100 %** — la borne resserree ferme
completement l'ecart sur la mediane du lot, validite `q_ub >= q*` verifiee
contre la verite terrain sur les 8 instances.

> Ce tableau a lui aussi ete regenere apres l'iteration decrite plus bas. La
> version qui l'avait produit la premiere fois donnait un ecart garanti
> median de 1.5 % pour le Th. 5' (53.2 % pour le Th. 5) et prouvait
> l'optimalite sur 4 instances sur 8.

Le changement de nature compte plus que le chiffre : sur 5 instances la
matheuristique **prouve** desormais l'optimalite, sur des instances que la
methode exacte seule n'avait pas su resoudre dans le meme budget. La
formulation « trouve sans pouvoir prouver » ne tient plus.

### Ce qui restait ouvert a ce stade

Deux instances gardaient un ecart notable (33.4 % et 20.9 %) alors que leur
ecart reel etait nul : le budget de certification y etait absorbe avant que
l'oracle ne referme la borne. Pistes annoncees : plafond de coupes adaptatif
(fixe a 40), allocation dynamique du budget entre recherche et certification,
et big-M des coupes moins lache. La section suivante les traite.

## Iteration suivante : trois verrous, mesures a budget egal

`bench_improve.py` — 8 instances du regime cible, **10 graines chacune**,
budget identique (7 s de recherche + 5 s de certification). Le meme script
tourne sans modification sur la version precedente du code et sert de temoin.

### 1. Big-M des coupes, calcule sur la relaxation continue

Le big-M du Th. 4 valide toute borne inferieure de `min_{x in S} e_k(x)`. Il
etait calcule sur la **boite** `0 <= x <= ub`, donc en ignorant `A x <= b`.
On le calcule desormais sur la **relaxation continue** de `S` :

* valide, puisque la relaxation contient `S` ;
* plus fine, puisqu'elle est contenue dans la boite ;
* les coefficients etant entiers et `x` entier, `e_k(x)` est entier : on
  remonte au plafond entier du minorant continu, ce qui resserre encore.

Cout : un LP par critere et par coupe, compte separement des ILP
(`ORACLE_CALLS["lp"]`) pour ne pas polluer l'unite de cout.

La validite ne se postule pas : **V8** (`verify_oracle.py`) et
`bench_bigm.py` comparent le `M` retenu au minimum REEL de `e_k` sur `S`
enumere. Ce test ne depend pas de la chance d'une graine — un `M` trop petit
couperait des points efficaces et serait detecte immediatement.

Mesure (`bench_bigm.py`, 8 instances du regime cible, 8 coupes chacune) :

| | mediane |
|---|---|
| resserrement `M_boite -> M_lp` | **1.14x** (jusqu'a 1.47x) |
| ecart residuel `M_lp / M_reel` | **1.00x** (au pire 1.05x) |
| validite `M_lp >= M_reel` | 8/8 instances |

La seconde ligne est la plus informative : le minorant continu **atteint
pratiquement le minimum entier**. Ce qui reste a gagner sur le big-M lui-meme
est nul ou presque — la piste est fermee, et c'est utile de le savoir avant
d'y consacrer du travail. Le gain de 1.14x est reel mais modeste ; il ne
suffit pas a expliquer les resultats ci-dessous, qui viennent des points 2 a 4.

### 2. Plafond de coupes adaptatif — et un contre-resultat mesure

Le plafond de coupes recyclees etait fige a 40. Il est maintenant pilote par
le budget : les coupes sont ajoutees par lots, un lot de plus n'etant pose
que si la borne n'a pas ferme et qu'il reste du temps.

La premiere version de ce mecanisme, a petits lots et nombreux tours, a
**degrade** l'ecart garanti median (2.94 % -> 6.17 %). Diagnostic par les
diagnostics par run : le modele etant conserve d'un tour a l'autre, les
coupes s'accumulent — celles des lots **et** celles que l'oracle pose
lui-meme — et atteignaient ~90 coupes, soit `p` binaires chacune. Chaque ILP
devenait trop lourd pour que l'oracle referme quoi que ce soit, alors meme
qu'il disposait de deux fois plus de temps.

Reglage retenu apres mesure (4 instances x 4 graines) :

| schema | ecart garanti median | optimalite prouvee |
|---|---|---|
| lots de 10, 6 tours | 9.91 % | 4/16 |
| lots de 40, 2 tours | **0.00 %** | 11/16 |
| lot unique de 40 | 0.00 % | 10/16 |
| lot unique de 80 | 0.00 % | 12/16 |

La lecon n'est pas le chiffre mais le sens : sur ce schema, **plus de coupes
n'est pas mieux**. Le nombre de binaires croit en `p` par coupe et finit par
coûter plus que ce que la coupe rapporte. C'est la meme limite que celle
deja signalee dans « Note sur le solveur », ici quantifiee.

### 3. La certification ne fait plus que certifier

Deux changements rendent la seconde phase productive :

* **le budget de recherche non consomme lui est reverse.** La recherche
  s'arretait au premier tour sans amelioration et abandonnait le reste de son
  budget. Ce temps est deja alloue ; il va desormais a la certification, qui
  en manquait.
* **l'oracle de certification remonte l'incumbent.** Les points efficaces
  qu'il certifie au passage sont compares a `q` : s'ils le battent, c'est un
  pas de Dinkelbach obtenu gratuitement. Le LB monte pendant que l'UB
  descend. Chaque tour lit sa borne sur le `q` qui a engendre son substitut,
  et l'on garde le **minimum** des bornes obtenues — valide, puisque `q*` ne
  depend pas de `q`.

### 4. Redemarrages guides par l'archive

Un seul tour sterile mettait fin a la recherche. Elle repart maintenant des
points de l'archive **les moins exploites** comme bases, et n'abandonne
qu'apres `max_stall` tours steriles consecutifs. L'archive ne contient que
des points efficaces certifies : ce sont des bases legitimes et deja payees,
la ou un redemarrage aleatoire jetterait le travail fait.

C'est ce qui traite le point « reduire la variance » : sur l'instance la plus
dure (`n7 m4 p4 c0.00`, `|E| = 218`), l'ecart reel maximal sur 10 graines
passe de **49.0 % a 0.0 %**.

### Resultat — 8 instances x 10 graines, budget identique

| | version precedente | **version actuelle** |
|---|---|---|
| ecart garanti median | 2.94 % | **0.00 %** |
| optimum atteint | 59/80 runs | **78/80 runs** |
| optimalite **prouvee** | 31/80 runs | **57/80 runs** |
| validite `q_lb <= q* <= q_ub` | 80/80 | **80/80** |

Par instance (mediane sur 10 graines ; `max` = pire graine, c'est la colonne
qui mesure la variance) :

| instance | \|E\| | ecart reel med / max | ecart garanti med / max | optimum | prouve |
|---|---|---|---|---|---|
| n7 m4 p4 c0.00 | 218 | 0.0 / **0.0** *(av. 0.0 / 49.0)* | 0.0 / **0.0** *(av. 0.0 / 60.6)* | **10**/10 *(8)* | **10**/10 *(8)* |
| n7 m4 p4 c0.25 | 198 | 0.0 / 0.0 | **19.4** / 27.0 *(av. 31.3 / 34.1)* | 10/10 | 0/10 |
| n6 m4 p4 c0.00 | 185 | **0.0** / 0.0 *(av. 3.0 / 17.1)* | **0.0** / 0.0 *(av. 4.0 / 35.7)* | **10**/10 *(0)* | **10**/10 *(0)* |
| n7 m4 p3 c0.25 | 112 | 0.0 / **0.0** *(av. 0.0 / 2.6)* | **0.0** / 0.9 *(av. 1.3 / 39.8)* | **10**/10 *(8)* | **9**/10 *(5)* |
| n5 m3 p4 c0.00 | 54 | 0.0 / 0.0 | **5.4** / 9.8 *(av. 15.0 / 21.0)* | 10/10 | 0/10 |
| n8 m5 p3 c0.00 | 46 | 0.0 / **10.5** *(av. 0.0 / 18.1)* | **0.0** / 52.1 *(av. 1.9 / 45.8)* | **8**/10 *(6)* | **8**/10 *(5)* |
| n7 m4 p4 c0.50 | 30 | 0.0 / **0.0** *(av. 0.0 / 61.5)* | **0.0** / 0.0 *(av. 7.0 / 65.7)* | **10**/10 *(8)* | **10**/10 *(4)* |
| n6 m4 p2 c0.25 | 26 | 0.0 / **0.0** *(av. 0.0 / 18.7)* | 0.0 / **0.0** *(av. 0.0 / 29.9)* | 10/10 *(9)* | **10**/10 *(9)* |

*(entre parentheses : la version precedente, quand elle differe)*

La version actuelle ameliore les quatre indicateurs agreges et la mediane de
chaque instance. **Une** case sur trente-deux est en retrait : sur
`n8 m5 p3 c0.00`, la pire graine donne 52.1 % d'ecart garanti contre 45.8 %
avant — la meme graine isolee que signale la section suivante. Partout
ailleurs le gain le plus net n'est pas la mediane mais la colonne `max` : la
dispersion entre graines, que la version precedente laissait monter jusqu'a
60 %, disparait presque entierement.

### Ce qui reste ouvert

* **`n7 m4 p4 c0.25` (`|E| = 198`) : ~19 % d'ecart garanti pour un ecart reel
  nul, et 0/10 preuves.** L'optimum est trouve a tous les coups sans jamais
  etre prouve. Mesure directe du blocage : au point `q*`, `F(q*)` vaut 0 sur
  `E` mais **959** sur `S` sans coupe, et prouver `F(q*) <= 0` demande
  **433 s, 177 coupes et 533 ILP** — soit 43 fois le budget de certification.
  Ce n'est donc ni un reglage ni une graine malheureuse : c'est le prix
  reel de la certification sur cette instance, et il tient a la qualite de
  la relaxation, pas au budget.
* **`n8 m5 p3 c0.00` : une graine sur dix a 52 % d'ecart garanti** alors que
  les neuf autres sont a 0. Une seule graine defavorable subsiste ; la
  variance est reduite, pas eliminee.
* Le contre-resultat du point 2 delimite la piste suivante : tant que chaque
  coupe coute `p` binaires, en ajouter davantage se paie. Desagreger la
  disjonction, ou remplacer le schema de type Sylva-Crema, conditionne tout
  gain supplementaire sur `U`.

## La limite de temps en etait-elle une ? — une premisse mesuree, puis dementie

`solve_milp` ne transmettait aucune limite de temps a HiGHS. Chaque ILP
tournait donc jusqu'a l'optimum, et l'heure n'etait verifiee qu'**entre**
deux appels : en principe, un seul ILP pouvait absorber tout le budget d'un
schema qui se presente comme anytime.

### Ce que la mesure a dementi

Le depassement reel, mesure sur quatre instances a deux budgets, plafonne a
**+1.3 s** — les ILP de cette famille sont trop rapides pour que le probleme
morde. La premisse etait donc surestimee.

Pire, couper l'ILP pile a l'echeance **degrade** la borne : une relaxation
resolue a l'optimum livre l'argmax donc une **coupe**, alors qu'une
resolution interrompue ne livre qu'une borne duale, et la coupe vaut
davantage. Mesure sur `n5 m3 p4` a 10 s : `UB` = 474 en coupant, contre 369
en laissant finir. Le meme gaspillage se reproduit si le sursis n'est accorde
qu'a la relaxation et pas au test d'efficacite ni a la chaine de reparation
qui la suivent : l'iteration s'interrompt un cran plus loin et la coupe est
perdue quand meme.

### Ce qui a ete retenu

La limite est transmise, mais avec un sursis (`ilp_grace`, defaut 1.0)
couvrant **toute l'iteration**. Autrement dit : on ne coupe jamais un calcul
en cours pour respecter l'horloge, mais aucun ILP ne peut s'emballer. Le
depassement total est majore par `ilp_grace x budget` au lieu d'etre non
majore.

Deux proprietes en decoulent, toutes deux de correction et non de
performance :

* un ILP interrompu rend desormais `mip_dual_bound`, borne optimiste
  **valide** : il informe encore au lieu d'etre perdu ;
* le test d'efficacite distingue « prouve efficace », « prouve non efficace »
  et « non concluant ». Un incumbent non certifie n'entre ni dans l'archive
  ni dans le LB. `repair_to_efficient` renvoie `None` plutot que le dernier
  point atteint : le LB ne peut pas devenir optimiste, et c'est la seule
  propriete que la matheuristique n'a pas le droit de perdre.

### Effet mesure : aucun

8 instances x 10 graines, budget identique, avant / apres :

| | sans limite ILP | avec limite ILP |
|---|---|---|
| ecart garanti median | 0.00 % | 0.00 % |
| optimum atteint | 78/80 | 78/80 |
| optimalite prouvee | 58/80 | 58/80 |
| validite `q_lb <= q* <= q_ub` | 80/80 | 80/80 |

Strictement rien. Le changement est une **assurance**, pas un resultat : il
ne se paie pas, et il borne un risque qui n'existe pas encore sur ce lot mais
qui apparaitra des que `n` grandira — c'est-a-dire sur la faiblesse n°1
ci-dessous. Le rapporter comme un gain serait malhonnete.

Portee exacte : la borne concerne la boucle de l'oracle. `d_min` et `d_plus`
restent des ILP uniques sans limite.

## Les vraies faiblesses, par ordre de gravite

Elles ne sont pas des reglages ; les nommer vaut mieux que les laisser
trouver par un rapporteur.

1. **Le calibre est minuscule.** `n` de 5 a 8, `m` de 3 a 5, `|S|` de 1300 a
   5300. La cause est structurelle : la verite terrain vient d'une
   enumeration exhaustive, qui devient impraticable au-dela de `n = 8`. Rien
   ici ne dit ce qui se passe a `n = 30`. C'est la faiblesse dominante, et
   elle impose un choix : renoncer a l'enumeration comme reference et se
   contenter de verifier la VALIDITE des bornes (verifiable sans connaitre
   `q*`) au lieu de leur exactitude.
2. **Le lot de 8 est choisi, pas echantillonne.** Ces instances ont ete
   retenues *parce que* la methode exacte y echouait. « La matheuristique bat
   l'exact » y est donc presque tautologique. La formulation defendable est
   plus etroite : *sur le regime ou l'exact ne conclut pas, elle transforme
   une absence de resultat en resultat certifie*.
3. **Aucun temoin issu de la litterature.** Ni Zerdani & Moulai (2011) ni
   Drici et al. (2018) ne sont implementes. La seule comparaison disponible
   est avec une version anterieure du meme code.
4. **Le budget de 12 s est arbitraire.** Et la mesure du point precedent le
   situe deux ordres de grandeur sous le cout d'une certification exacte.

## Prochaines etapes

Les trois premiers points de la liste precedente sont traites et mesures
ci-dessus (allocation du budget et plafond adaptatif, variance, big-M). Ce
qui reste :

0. **Monter en taille** (faiblesse n°1 ci-dessus) : abandonner l'enumeration
   comme reference et ne verifier que la validite des bornes, ce qui est
   possible sans connaitre `q*`. C'est le prealable a toute affirmation
   au-dela de `n = 8`.
1. **Desagreger la coupe de dominance.** Le contre-resultat mesure plus haut
   est net : au-dela d'une quarantaine de coupes, les `p` binaires par coupe
   coutent plus qu'elles ne rapportent. Tant que ce cout n'est pas reduit,
   aucune strategie de coupes ne fera mieux — c'est le verrou suivant, et il
   commande le gain restant sur `U`.
2. **Fermer la borne sur `n7 m4 p4 c0.25`.** Optimum trouve 10 fois sur 10,
   jamais prouve. C'est le cas ou `U` reste grand independamment du budget :
   il isole ce que la relaxation courante ne sait pas faire.
3. **Campagne complete de la matheuristique** — les 216 instances, au moins
   10 graines chacune, mediane et dispersion. Le lot mesure ici n'en couvre
   que 8, choisies parce que la methode exacte y echouait ; ce choix est
   assume mais il n'est pas un echantillon.
4. **Remettre `campaign.py`, `analyze.py` et `scaling.py` dans le depot.**
   Les tableaux de la section « Campagne de difficulte » ne sont pas
   reproductibles en l'etat.
5. **Comparaison avec la litterature** — reimplementer Zerdani & Moulai
   (2011) et Drici et al. (2018) sur le meme lot. Point 4 du plan de these.
6. **Protocole** — stratifier par `corr` : a `(n,m,p)` fixe, `|E|` varie d'un
   facteur 40 selon ce parametre.

## Note sur le solveur

`solve_ilp()` et `solve_milp()` dans `molfp_core.py` sont les **seuls** points
a reecrire pour passer a Gurobi ou CPLEX. Le compteur `ORACLE_CALLS` est
l'unite de cout a rapporter dans les experiences : reproductible et
independante de la machine et du solveur.

Limites connues :
* les coefficients `Q*num - P*den` grossissent avec le numerateur et le
  denominateur de `q` — sur de grandes instances, borner `q` via
  `Fraction.limit_denominator` ou repasser en flottant avec tolerance ;
* le nombre de binaires croit en `p` par coupe (defaut classique du schema
  Sylva-Crema). Le big-M est desormais calcule sur la relaxation continue et
  non plus sur la boite ; le cout en binaires, lui, demeure, et la mesure de
  la section « Iteration suivante » le chiffre : au-dela d'une quarantaine de
  coupes il annule le gain de la coupe. C'est la premiere cible restante.

`ORACLE_CALLS` distingue `"ilp"` et `"lp"` : les LP de renforcement du big-M
sont d'un ordre de grandeur moins chers qu'un ILP et les melanger fausserait
la comparaison entre methodes.
