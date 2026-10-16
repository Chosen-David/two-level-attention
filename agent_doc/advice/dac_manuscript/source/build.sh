#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ../output ../qa ../texmf-var
# Optional relocatable TeX/font roots. Defaults retain the original host layout.
texmf_dist="${PAPER_TEXMF_DIST:-/usr/share/texlive/texmf-dist}"
texmf_extra="${PAPER_TEXMF_EXTRA:-/usr/share/texmf}"
font_root="${PAPER_FONT_ROOT:-/usr/share/fonts}"
xelatex_bin="${PAPER_XELATEX:-xelatex}"
xetex_bin="${PAPER_XETEX:-xetex}"
# Standard TeX Live installs need none of these fallbacks. The current
# workspace contains TeX files but lacks a generated system filename DB.
if [ -d "$texmf_dist" ]; then
  export TEXINPUTS=".:${texmf_dist}/tex//:"
  export TFMFONTS="${texmf_dist}/fonts/tfm//:${texmf_extra}/fonts/tfm//:"
  export OPENTYPEFONTS="${texmf_extra}/fonts/opentype//:${font_root}/opentype//:"
  export ENCFONTS="${texmf_dist}/fonts/enc//:"
  export TEXFONTMAPS="${texmf_dist}/fonts/map//:"
  export T1FONTS="${texmf_dist}/fonts/type1//:${font_root}/type1//:"
  export FONTFEATURES="${texmf_dist}/fonts/fea//:"
  export TEXFONTS="${texmf_dist}/fonts//:${texmf_extra}/fonts//:"
  export TEXMFCACHE="$(pwd)/../texmf-var"
  export XDG_CACHE_HOME="$(pwd)/../texmf-var/cache"
  export TEXMFVAR="$(pwd)/../texmf-var"
  export TEXFORMATS="$(pwd)/../texmf-var:"
  if [ ! -f ../texmf-var/xelatex.fmt ]; then
    (cd ../texmf-var && "$xetex_bin" -ini -etex -interaction=nonstopmode -jobname=xelatex xelatex.ini > init-xelatex.log 2>&1)
  fi
fi
papers=(paper_en paper_zh supplement_en supplement_zh)
if (( $# > 0 )); then papers=("$@"); fi
for paper in "${papers[@]}"; do
  case "$paper" in paper_en|paper_zh|evidence_register|supplement_en|supplement_zh) ;; *) echo "Expected a paper/supplement stem (en or zh)" >&2; exit 2;; esac
  [ -f "$paper.tex" ] || { echo "Missing TeX source: $paper.tex" >&2; exit 2; }
  "$xelatex_bin" -interaction=nonstopmode -halt-on-error -output-directory=../output "$paper.tex" > "../qa/${paper}-compile.log" 2>&1
  "$xelatex_bin" -interaction=nonstopmode -halt-on-error -output-directory=../output "$paper.tex" >> "../qa/${paper}-compile.log" 2>&1
done
