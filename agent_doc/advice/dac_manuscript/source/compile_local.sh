#!/usr/bin/env bash
# 本地编译包装（2026-10-16 by Claude）：
# 本机沙箱会清空 PATH/HOME 等环境变量，build.sh 直接跑会找不到 xelatex，
# 此脚本用绝对路径显式设置环境后调用标准 build.sh。
# 用法：bash compile_local.sh [paper_en|paper_zh|supplement_en|supplement_zh|evidence_register ...]
# 字体说明：Nimbus OTF 暂用 TeX Gyre Termes/Heros 符号链接兜底（Times 同源克隆，
# 本地验证编译用；权威 PDF 仍以远端 Nimbus 构建为准）。
set -euo pipefail
export PATH="/home/wangyuanshuo02/texlive/2026/bin/x86_64-linux:/usr/local/bin:/usr/bin:/bin"
export HOME="${HOME:-/home/wangyuanshuo02}"
export TEXMFCACHE="/home/wangyuanshuo02/texlive/2026/texmf-var"
export XDG_CACHE_HOME="/home/wangyuanshuo02/texlive/2026/texmf-var/cache"
export TEXMFVAR="/home/wangyuanshuo02/texlive/2026/texmf-var"
# 让 fontspec 按文件名找到 texmf-local 中的 Nimbus 兜底字体
export OPENTYPEFONTS="/home/wangyuanshuo02/texlive/texmf-local/fonts/opentype/urwbase35:"
export TEXFONTS="/home/wangyuanshuo02/texlive/texmf-local/fonts/opentype/urwbase35:"
cd /home/wangyuanshuo02/two-level-attention/agent_doc/advice/dac_manuscript/source
exec bash build.sh "$@"
