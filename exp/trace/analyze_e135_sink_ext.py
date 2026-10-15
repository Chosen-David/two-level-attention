#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E135 扩展臂：sink 细粒度下探 {0,4}（swa 固定 128），回答"sink=4 是否就够"。

不改动归档分析器 analyze_e135_sinkswa_grid.py（其输出 e135_sinkswa_grid.json
已被 E135 verdict 引用）。本驱动 monkey-patch 其模块常量：
  SINK_GRID = [0, 4, 16, 64, 128]（新增 0/4 两臂；0 = 无强制 sink，
  检验 tail32 打分器能否自行发现 sink token）
  SWA_GRID  = [128]（sink 是本案唯一问题维度）
  OUT       = e135_sink_ext.json（独立产物，归档 JSON 不动）

行过滤沿用原器硬编码最坏点 (16,512) → mid_len≥4096，与归档网格同一行集，
sk16/64/128 三重叠臂可与 e135_sinkswa_grid.json 逐值交叉验证（应逐位复现）。
锚点臂 anchor_e98_sk128_sw1024 保留自校验。

用法：E135_DEVICE=cuda:0 python3 analyze_e135_sink_ext.py [--smoke]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_e135_sinkswa_grid as e135

e135.SINK_GRID = [0, 4, 16, 64, 128]
e135.SWA_GRID = [128]
e135.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results", "e135_sink_ext.json")

if __name__ == "__main__":
    e135.main()
