#!/usr/bin/env bash
# Download only, from signed Debian indexes. Never installs or modifies /usr.
set -euo pipefail
if (( $# != 1 )); then echo 'usage: download_debian13.sh OUTPUT_DIR' >&2; exit 2; fi
out=$(realpath -m "$1")
mkdir -p "$out/lists/partial" "$out/cache/archives/partial"
cat > "$out/sources.list" <<'EOF'
deb https://deb.debian.org/debian trixie main
deb https://deb.debian.org/debian trixie-updates main
deb https://security.debian.org/debian-security trixie-security main
EOF
cat > "$out/no-host-hooks.conf" <<'EOF'
#clear APT::Update::Post-Invoke;
#clear APT::Update::Post-Invoke-Success;
#clear DPkg::Post-Invoke;
EOF
: > "$out/empty-status"
opts=(-c "$out/no-host-hooks.conf" -o "Dir::Etc::sourcelist=$out/sources.list"
      -o Dir::Etc::sourceparts=- -o "Dir::State::lists=$out/lists"
      -o "Dir::State::status=$out/empty-status" -o "Dir::Cache=$out/cache"
      -o APT::Architecture=amd64 -o APT::Get::List-Cleanup=0)
apt-get "${opts[@]}" update
apt-get "${opts[@]}" -y --download-only --no-install-recommends install \
    texlive-xetex texlive-latex-extra inkscape poppler-utils fontconfig \
    fonts-noto-cjk fonts-urw-base35 fonts-lmodern
# An empty dpkg status requests the dependency closure instead of omitting
# packages already installed on the packaging host. This is Debian 13 amd64,
# NOT an installer for Ubuntu/older Debian/ARM or a relocatable Linux runtime.
