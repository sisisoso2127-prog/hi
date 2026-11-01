#!/bin/sh
# Rebuild every shipped PDF in docs/.  Run from this directory: sh build.sh
#
# The engines are not interchangeable.  summary-arabic needs a Unicode engine
# for polyglossia/Amiri; the markdown conversions need DejaVu, because Latin
# Modern has no U+2264/U+2265/U+21D2/U+21D4 and pandoc drops them silently --
# a missing "<=" in a claim is not a typesetting detail.
set -e
cd "$(dirname "$0")"

PDFLATEX="pdflatex -interaction=nonstopmode -halt-on-error"
MD="--pdf-engine=lualatex -V geometry:margin=2.3cm -V fontsize=10pt \
    -V mainfont=DejaVuSerif -V monofont=DejaVuSansMono"

# LaTeX, pdflatex: twice for cross-references.
for f in results worked-examples algorithm-walkthrough theory \
         hybrid-methods-note memoire-fr; do
  echo "== $f"; $PDFLATEX $f.tex >/dev/null; $PDFLATEX $f.tex >/dev/null
done

# The article also has a bibliography.
echo "== hybrid-article"
$PDFLATEX hybrid-article.tex >/dev/null
bibtex hybrid-article >/dev/null
$PDFLATEX hybrid-article.tex >/dev/null
$PDFLATEX hybrid-article.tex >/dev/null

# Arabic: lualatex, and the rendered pages are worth looking at -- bidi
# defects (reversed page numbers, unshaped text in tabulars) never reach the
# exit code.
echo "== summary-arabic"
lualatex -interaction=nonstopmode -halt-on-error summary-arabic.tex >/dev/null
lualatex -interaction=nonstopmode -halt-on-error summary-arabic.tex >/dev/null

# The two READMEs, via pandoc.
echo "== README / package-notes"
pandoc ../README.md            -o README.pdf        $MD
pandoc ../lfp_efficient/README.md -o package-notes.pdf $MD

echo "done"
