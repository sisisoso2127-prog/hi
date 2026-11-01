#!/bin/sh
# Build memoire-code.zip: the self-contained archive of everything the thesis
# needs -- the package, the tests, the campaign, the examples, the results and
# the theory -- with README-ar.md as its guide.
#
# The archive is derived, so it is not committed.  Rebuild it with: sh pack.sh
set -e
cd "$(dirname "$0")"

OUT=memoire-code.zip
STAGE=$(mktemp -d)
ROOT=$STAGE/memoire-code

mkdir -p "$ROOT"/lfp_efficient "$ROOT"/tests "$ROOT"/studies \
         "$ROOT"/examples "$ROOT"/docs

cp lfp_efficient/*.py lfp_efficient/README.md "$ROOT"/lfp_efficient/
cp tests/*.py                                 "$ROOT"/tests/
cp studies/*.py studies/campaign-results.json "$ROOT"/studies/
cp examples/*.py                              "$ROOT"/examples/
cp docs/results.pdf docs/theory.pdf docs/theory.tex "$ROOT"/docs/
cp pyproject.toml LICENSE README-ar.md        "$ROOT"/

# The archive must stand on its own: run the suite from inside it.
( cd "$ROOT" && python3 tests/test_lfp_efficient.py >/dev/null ) \
  || { echo "the staged copy does not pass its own tests"; exit 1; }
find "$ROOT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

rm -f "$OUT"
( cd "$STAGE" && zip -r -q "$OLDPWD/$OUT" memoire-code )
rm -rf "$STAGE"
echo "$OUT  $(du -h "$OUT" | cut -f1)  $(unzip -l "$OUT" | awk '/files$/{print $2" files"}')"
