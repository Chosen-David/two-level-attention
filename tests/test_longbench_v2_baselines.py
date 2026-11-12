import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmark.LongBench.pred import parse_args, shard_longbench_v2
from sparse_attn.info import get_method_name_with_info, get_treatment_manifest_json


def test_longbench_v2_shards_are_deterministic_disjoint_and_complete():
    data = [{"_id": f"sample-{i}"} for i in range(503)]
    for world_size in (1, 2, 3, 7, 16, 32):
        shards = [
            shard_longbench_v2(data, "lbv2", rank, world_size)
            for rank in range(world_size)
        ]
        merged = [row["_id"] for shard in shards for row in shard]
        assert sorted(merged, key=lambda value: int(value.split("-")[-1])) == [
            row["_id"] for row in data
        ]
        for rank, shard in enumerate(shards):
            assert shard == data[rank::world_size]
            assert len(shard) == len(range(rank, 503, world_size))


def test_longbench_v2_shard_rejects_bad_rank_and_other_tasks():
    data = [{"_id": "x"}]
    for rank, world_size in ((-1, 1), (1, 1), (0, 0), (2, 2)):
        with pytest.raises(ValueError):
            shard_longbench_v2(data, "lbv2", rank, world_size)
    with pytest.raises(ValueError):
        shard_longbench_v2(data, "qasper", 1, 2)


def test_longbench_v2_shard_cli_defaults_and_explicit_values():
    args = parse_args(["--task", "lbv2"])
    assert args.sample_rank == 0
    assert args.sample_world_size == 1
    args = parse_args([
        "--task", "lbv2", "--sample-rank", "2",
        "--sample-world-size", "7", "--pred_postfix", "_r2of7",
    ])
    assert (args.sample_rank, args.sample_world_size) == (2, 7)


def test_jssa_official_and_quest_kernel_treatment_identity(tmp_path):
    basis = tmp_path / "basis.pt"
    basis.write_bytes(b"first basis")
    jssa = SimpleNamespace(
        method="jssa_official", nfa_enable_layer_skip=False,
        jssa_budget=1024, jssa_rank=32, jssa_basis_path=str(basis))
    name = get_method_name_with_info(jssa)
    manifest = json.loads(get_treatment_manifest_json(jssa))
    assert name.startswith("jssa_official_basis_k1024_r32_h")
    assert manifest["jssa_basis_path"]["sha256"] == hashlib.sha256(
        b"first basis").hexdigest()
    basis.write_bytes(b"second basis")
    assert get_method_name_with_info(jssa) != name

    quest = SimpleNamespace(
        method="quest_vendor_kernel", nfa_enable_layer_skip=False,
        quest_token_budget=1024, quest_chunk_size=16,
        quest_page_size=16, quest_max_seq_len=40960)
    quest_manifest = json.loads(get_treatment_manifest_json(quest))
    assert quest_manifest["quest_page_size"] == 16
    assert quest_manifest["quest_max_seq_len"] == 40960
    quest.quest_page_size = 32
    assert get_method_name_with_info(quest) != get_method_name_with_info(
        SimpleNamespace(**{**vars(quest), "quest_page_size": 16}))


def test_jssa_official_and_torch_semantics_have_distinct_names(tmp_path):
    basis = tmp_path / "basis.pt"
    basis.write_bytes(b"basis")
    common = dict(
        nfa_enable_layer_skip=False, jssa_budget=1024, jssa_rank=32,
        jssa_basis_path=str(basis))
    official = get_method_name_with_info(
        SimpleNamespace(method="jssa_official", **common))
    semantic = get_method_name_with_info(
        SimpleNamespace(method="jssa_vendor", **common))
    assert official.startswith("jssa_official_")
    assert semantic.startswith("jssa_vendor_")
    assert official != semantic
