# CPU 红绿与回归原始短日志

基线60f469e8d085e81904417971f4367128d9d9d5f7；环境/输入/实现范围见同任务advice及独立verify.py。所有命令OMP_NUM_THREADS=1，timeout120。耗时仅日志时长，不作性能对照。完整traceback私有保留。

```text
python -m unittest sparse_attn.ops.tests.test_prefill_sigma_capacity -v
before: Ran 3 tests in 8.000s; FAILED (failures=10), exit1
after:  Ran 3 tests in 12.641s; OK, exit0
python -m unittest discover -s sparse_attn/ops/tests -v
Ran 87 tests in 30.550s; OK
python -O -m unittest discover -s sparse_attn/ops/tests -v
Ran 87 tests in 9.724s; OK
PYTHONPATH=.:tests python -m unittest discover -s tests -p 'test_e151c*.py' -v
Ran 21 tests in 2.667s; OK
PYTHONPATH=.:tests python -O -m unittest discover -s tests -p 'test_e151c*.py' -v
Ran 21 tests in 1.113s; OK
```

原始日志SHA256：

```text
before.log 00cd00adbd985bfa857a093078ec1eaa02365d7f39159adf4f16517cb334473f
after.log e67c517a6cb0fb7d671a0033cffe5975080d6de04c33c88fafb0ef6146562132
ops.log ba2c70c9b145b5b9d8c974968df215d61dc66669480239575f0fe85c5de1c1f3
ops_optimized.log 305c96d0b6ef2be6b45e0a8f763d1c55b3d1ccc7c06c1e7ac22a0007ad74e767
adjacent.log 89162065f596dce749b5b0b181625b2c40e6ff3a34a62521f05179fd0241dfbf
adjacent_optimized.log 3c7e214e332b16f08b0c910541c6f7d52dfca0ef3dfcec0fb0f81749616674ed
test_prefill_sigma_capacity.py f4634221035029e2e22366d51c56d6c6b421d8c0e08169a6df36f77d0284e8a0
```

独立结果见independent_review.md和可复现verify.py。未运行CUDA、模型质量、端到端性能或显存测量；不把CPU oracle当生产GPU验证。
