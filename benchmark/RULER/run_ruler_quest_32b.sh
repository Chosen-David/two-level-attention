#!/bin/bash
# 旧 quest_vendor 32B 派单已停用。官方 RULER runner 当前仅验收 Qwen3-8B；
# 32B 必须在官方 Quest 原生 GQA、显存容量和 YaRN 对拍完成后单独启用。
set -u
echo "拒绝运行旧 quest_vendor；Qwen3-32B 官方 RULER 尚未验收。" >&2
exit 2
