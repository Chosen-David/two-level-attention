#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../output ../qa ../texmf-var
# Standard TeX Live installs need none of these fallbacks. The current
# workspace contains TeX files but lacks a generated system filename DB.
if [ -d /usr/share/texlive/texmf-dist ]; then
  export TEXINPUTS=".:/usr/share/texlive/texmf-dist/tex//:"
  export TFMFONTS="/usr/share/texlive/texmf-dist/fonts/tfm//:/usr/share/texmf/fonts/tfm//:"
  export OPENTYPEFONTS="/usr/share/texmf/fonts/opentype//:/usr/share/fonts/opentype//:"
  export ENCFONTS="/usr/share/texlive/texmf-dist/fonts/enc//:"
  export TEXFONTMAPS="/usr/share/texlive/texmf-dist/fonts/map//:"
  export T1FONTS="/usr/share/texlive/texmf-dist/fonts/type1//:/usr/share/fonts/type1//:"
  export FONTFEATURES="/usr/share/texlive/texmf-dist/fonts/fea//:"
  export TEXFONTMAPS="/usr/share/texlive/texmf-dist/fonts/map//:"
  export TEXFONTS="/usr/share/texlive/texmf-dist/fonts//:/usr/share/texmf/fonts//:"
  export TEXMFCACHE="$(pwd)/../texmf-var"
  export XDG_CACHE_HOME="$(pwd)/../texmf-var/cache"
  export TEXMFVAR="$(pwd)/../texmf-var"
  export TEXFORMATS="$(pwd)/../texmf-var:"
  if [ ! -f ../texmf-var/xelatex.fmt ]; then
    (cd ../texmf-var && xetex -ini -etex -interaction=nonstopmode -jobname=xelatex xelatex.ini > init-xelatex.log 2>&1)
  fi
fi
papers=(paper_en paper_zh evidence_register)
if (( $# > 0 )); then papers=("$@"); fi
for paper in "${papers[@]}"; do
  case "$paper" in paper_en|paper_zh|evidence_register|supplement_en|supplement_zh) ;; *) echo "Expected paper_en, paper_zh or evidence_register" >&2; exit 2;; esac
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=../output "$paper.tex" > "../qa/${paper}-compile.log" 2>&1
  xelatex -interaction=nonstopmode -halt-on-error -output-directory=../output "$paper.tex" >> "../qa/${paper}-compile.log" 2>&1
done
