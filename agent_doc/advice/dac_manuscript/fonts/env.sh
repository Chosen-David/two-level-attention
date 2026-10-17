#!/usr/bin/env bash
# Source in the shell used for XeLaTeX/Inkscape/Python. No system or venv writes.
PAPER_FONT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PAPER_FONT_DIR
export PAPER_FONT_CACHE="${PAPER_FONT_CACHE:-${PAPER_FONT_DIR}/../.font-cache}"
export FONTCONFIG_FILE="${PAPER_FONT_CACHE}/fonts.conf"
export OPENTYPEFONTS="${PAPER_FONT_DIR}//:${OPENTYPEFONTS:-}"
export TTFONTS="${PAPER_FONT_DIR}//:${TTFONTS:-}"
export TEXFONTS="${PAPER_FONT_DIR}//:${TEXFONTS:-}"
export INKSCAPE_PROFILE_DIR="${PAPER_FONT_CACHE}/inkscape"
mkdir -p "$INKSCAPE_PROFILE_DIR"
export MPLCONFIGDIR="${PAPER_FONT_CACHE}/matplotlib"
python3 "${PAPER_FONT_DIR}/check_fonts.py" --prepare "$PAPER_FONT_CACHE" || return 1 2>/dev/null || exit 1
python3 "${PAPER_FONT_DIR}/check_fonts.py" --check || return 1 2>/dev/null || exit 1
