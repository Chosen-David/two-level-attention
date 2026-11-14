from types import SimpleNamespace

from benchmark.efficiency import benchmark_longbench_two_point as harness


class _FakeTokenizer:
    eos_token_id = 99


class _FakeEngine:
    instances = []

    def __init__(self, args, model, tokenizer, eos_ids):
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.eos_ids = eos_ids
        self.runtime = {"execution": args.execution, "batch_size": 1}
        self.request_receipts = []
        self.calls = []
        type(self).instances.append(self)

    def generate(self, ids, max_new_tokens, ignore_eos):
        if not ignore_eos:
            raise AssertionError("性能路径必须忽略EOS")
        self.calls.append((tuple(ids), max_new_tokens))
        receipt = {"generated_tokens": max_new_tokens}
        self.request_receipts.append(receipt)
        return [11] * max_new_tokens, max_new_tokens * 0.01


def test_longbench_two_point_isolates_task_arenas_and_closes_cells(
        tmp_path, monkeypatch):
    monkeypatch.setattr(harness, "source_receipt", lambda: {"test": True})
    monkeypatch.setattr(harness, "load_rows", lambda root, task: ([{"row": task}], "data-" + task))
    monkeypatch.setattr(
        harness, "build_input",
        lambda row, task, fmt, max_length, tokenizer: (
            [len(task), 2, 3], {"input_tokens": 3, "task": task}))
    monkeypatch.setattr(harness.torch.cuda, "empty_cache", lambda: None)
    _FakeEngine.instances = []
    args = SimpleNamespace(
        method="nfa", execution="graph", model_path="/model",
        data_root="/data", output_dir=str(tmp_path), tasks=["narrativeqa", "qasper"],
        sample_indices=None, warmup_requests=1, repeats=2, max_new_tokens=33,
        visible_len=8192)
    model = SimpleNamespace(_nfa_graph_session_pool=object())
    payload = harness.run_performance(
        args, model, _FakeTokenizer(), (99,), {"narrativeqa": "", "qasper": ""},
        31500, engine_factory=_FakeEngine)

    if payload["status"] != "complete" or len(payload["cells"]) != 4:
        raise AssertionError("两任务×两repeat未闭集")
    if len(_FakeEngine.instances) != 2:
        raise AssertionError("不同任务必须使用独立固定arena/Graph session")
    expected_calls = [(1, 33), (33, 33)] * 3
    for engine, task in zip(_FakeEngine.instances, args.tasks):
        actual = [tokens for _, tokens in engine.calls]
        if actual != [item[0] for item in expected_calls]:
            raise AssertionError(f"{task} warmup/repeat请求日程错误: {actual}")
        if len({ids for ids, _ in engine.calls}) != 1:
            raise AssertionError("同一任务的所有pair必须复用相同输入")
    if set(payload["request_receipts"]) != set(args.tasks):
        raise AssertionError("逐任务真实请求回执缺失")
    if any(len(payload["request_receipts"][task]) != 6 for task in args.tasks):
        raise AssertionError("每任务应有6个请求回执")
    if model._nfa_graph_session_pool is not None:
        raise AssertionError("任务边界必须释放Graph pool")
    for cell in payload["cells"]:
        timing = cell["request_timing"]
        if timing["actual_generated_tokens"] != 33 or timing["eos_early_stop"]:
            raise AssertionError("固定33 token/EOS协议漂移")
        if timing["tpot_ms"] != 10.0:
            raise AssertionError("两点法TPOT计算错误")


def test_longbench_two_point_rejects_early_eos_generation(tmp_path, monkeypatch):
    class EarlyStopEngine(_FakeEngine):
        def generate(self, ids, max_new_tokens, ignore_eos):
            return [11] * max(1, max_new_tokens - 1), 0.1

    monkeypatch.setattr(harness, "source_receipt", lambda: {"test": True})
    monkeypatch.setattr(harness, "load_rows", lambda root, task: ([{"row": task}], "data"))
    monkeypatch.setattr(
        harness, "build_input",
        lambda row, task, fmt, max_length, tokenizer: ([1, 2, 3], {"input_tokens": 3}))
    monkeypatch.setattr(harness.torch.cuda, "empty_cache", lambda: None)
    args = SimpleNamespace(
        method="flashinfer", execution="graph", model_path="/model",
        data_root="/data", output_dir=str(tmp_path), tasks=["qasper"],
        sample_indices=None, warmup_requests=0, repeats=1, max_new_tokens=33,
        visible_len=8192)
    try:
        harness.run_performance(
            args, SimpleNamespace(), _FakeTokenizer(), (99,), {"qasper": ""},
            31500, engine_factory=EarlyStopEngine)
    except RuntimeError as exc:
        if "固定长度生成失败" not in str(exc):
            raise
    else:
        raise AssertionError("提前EOS/短输出必须fail-closed")
