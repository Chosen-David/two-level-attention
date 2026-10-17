#!/usr/bin/env bash
# Portable project-font build. No fake Nimbus aliases or fixed HOME/TeX install.
set -euo pipefail
source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$source_dir/../fonts/env.sh"
command -v xelatex >/dev/null || { echo 'XeLaTeX missing: activate the host TeX Live PATH first.' >&2; exit 1; }
exec bash "$source_dir/build.sh" "$@"
