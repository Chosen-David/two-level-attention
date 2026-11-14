#!/bin/bash
# 旧 quest_vendor 32B TP2 派单已停用。官方 RULER runner 当前仅验收
# Qwen3-8B 单卡，禁止把未经验证的接口适配套用到 32B/TP2。
set -u
echo "拒绝运行旧 quest_vendor；Qwen3-32B TP2 官方 RULER 尚未验收。" >&2
exit 2
