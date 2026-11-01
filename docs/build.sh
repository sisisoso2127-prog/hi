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
         hybrid-methods-note; do
  echo "== $f"; $PDFLATEX $f.tex >/dev/null; $PDFLATEX $f.tex >/dev/null
done

# These two carry a bibliography: latex, bibtex, then latex twice more so the
# \cite labels and the page numbers both settle.
for f in hybrid-article memoire-fr; do
  echo "== $f"
  $PDFLATEX $f.tex >/dev/null
  bibtex $f >/dev/null
  $PDFLATEX $f.tex >/dev/null
  $PDFLATEX $f.tex >/dev/null
done

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

# The Arabic reading list: a Unicode engine, an Arabic face, and RTL.  Look
# at the rendered pages -- bidi defects never reach the exit code.
echo "== reading-list-ar"
pandoc reading-list-ar.md -o reading-list-ar.pdf --pdf-engine=lualatex \
  -V geometry:margin=2.3cm -V fontsize=11pt \
  -V mainfont=Amiri -V monofont=DejaVuSansMono -V dir=rtl -V lang=ar

echo "done"
