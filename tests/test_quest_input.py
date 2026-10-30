import pytest

from benchmark.LongBench.quest_input import (
    canonical_quest_input, quest_character_boundary,
)


class FakeTokenizer:
    def __init__(self, pieces, bos=False):
        self.pieces = pieces
        self.bos = bos

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        if not return_offsets_mapping:
            raise AssertionError("canonical path must request offsets")
        ids, offsets = [], []
        if add_special_tokens and self.bos:
            ids.append(1)
            offsets.append((0, 0))
        cursor = 0
        for token, piece in self.pieces:
            start = text.index(piece, cursor)
            end = start+len(piece)
            ids.append(token)
            offsets.append((start, end))
            cursor = end
        return {"input_ids": ids, "offset_mapping": offsets}


def test_single_tokenization_crossing_bpe_token_belongs_to_suffix():
    text = "abcdeQuestion: answer"
    tokenizer = FakeTokenizer([(10, "abcde"), (11, "Question:"), (12, " answer")])
    result = canonical_quest_input(tokenizer, text, "qasper")
    assert result.prefix_ids == (10,)
    assert result.suffix_ids == (11, 12)
    assert result.prefix_ids+result.suffix_ids == result.full_ids
    assert not result.metadata["crossing_token_adjusted"]

    # 人为让末尾 100 字符边界落入 "Question:" 内，不能切开 BPE token。
    short = "abQuestion:"
    tokenizer = FakeTokenizer([(10, "ab"), (11, "Question:")])
    result = canonical_quest_input(tokenizer, short, "qmsum")
    boundary = len(short)-100
    assert boundary < 0 or result.metadata["requested_character_boundary"] == 0
    assert result.prefix_ids == ()
    assert result.suffix_ids == result.full_ids


def test_crossing_token_moves_whole_token_not_text_slice():
    text = "x"*100+"abcdef"+"y"*96
    tokenizer = FakeTokenizer([(10, "x"*100), (11, "abcdef"), (12, "y"*96)])
    result = canonical_quest_input(tokenizer, text, "qmsum")
    # 字符边界 102 落在 abcdef(100..106) 内，整个 token 必须归入 suffix。
    assert result.metadata["requested_character_boundary"] == 102
    assert result.metadata["crossing_token_adjusted"]
    assert result.metadata["crossing_token_offsets"] == [100, 106]
    assert result.prefix_ids == (10,)
    assert result.suffix_ids == (11, 12)
    assert result.prefix_ids+result.suffix_ids == result.full_ids


def test_bos_is_not_guessed_or_dropped_and_prefix_only_bos_is_rejected():
    text = "x"*120
    pieces = [(10+i, "x") for i in range(120)]
    result = canonical_quest_input(FakeTokenizer(pieces, bos=True), text, "qmsum",
                                   add_special_tokens=True,
                                   require_material_prefix=True)
    assert result.full_ids[0] == 1
    assert result.prefix_ids[0] == 1
    assert result.prefix_ids+result.suffix_ids == result.full_ids

    empty_material = canonical_quest_input(
        FakeTokenizer([(10, "short")], bos=True), "short", "qmsum",
        add_special_tokens=True)
    assert empty_material.prefix_ids == (1,)
    with pytest.raises(ValueError, match="nonempty material prefix"):
        canonical_quest_input(FakeTokenizer([(10, "short")], bos=True), "short", "qmsum",
                              add_special_tokens=True, require_material_prefix=True)


def test_task_anchor_and_hash_metadata_are_deterministic():
    text = "context "*30+"Question: final?"
    assert text[quest_character_boundary(text, "qasper"):].startswith("Question:")
    tokenizer = FakeTokenizer([(10, "context "), (11, "Question: final?")])
    # FakeTokenizer 只映射首个 context；改用覆盖全文的简单片段。
    tokenizer = FakeTokenizer([(10, text[:quest_character_boundary(text, "qasper")]),
                               (11, "Question: final?")])
    first = canonical_quest_input(tokenizer, text, "qasper")
    second = canonical_quest_input(tokenizer, text, "qasper")
    assert first == second
    assert first.metadata["full_ids_sha256"] == second.metadata["full_ids_sha256"]
    assert first.metadata["suffix_tokens"] == 1
