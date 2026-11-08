# 本轮准备的V1完整独验

仅3750完整配对/11250臂/128done且主控生产validate/analyze/export结束后执行。当前无质量授信。

1. run_logged.py PREPARE python -B prepare_review.py（0615副本加全量前置门禁，保存diff，复制生产导出）。
2. tools/recheck.py --run fixed/v1 --source /mnt/dolphinfs/ssd_pool/docker/user/hadoop-scale-llm/tla/frozen005_v1_16_source_20261030 --output v1_checked --require-full --no-snapshot
3. tools/cache_receipts_check.py --snapshot fixed/v1 --output tools/results/v1_cache.json；official_scores_verified.py；audit_scope.py。
4. tools/closeout_check_v2.py --run fixed/v1 --output tools/results/closeout。配套extra_checks.py绑定所有导出及B5000逐任务/等权AVG16。
5. 最终报告正文、六类artifact及六项独验绑定；tools/prepare_spec.py及seal_artifacts.py仅封存完整范围。

脚本路径均在本目录。reference_dependency已成套复制；没有调用部分汇总路径。V1本轮若未齐，不写FINAL_REPORT_ZH.md。
