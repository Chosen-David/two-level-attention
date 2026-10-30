"""Quest 风格材料前缀/问题后缀的唯一分词与边界映射。"""
from dataclasses import dataclass
import hashlib
import json


@dataclass(frozen=True)
class QuestTokenInput:
    full_ids: tuple
    prefix_ids: tuple
    suffix_ids: tuple
    metadata: dict


def quest_character_boundary(text, dataset):
    """返回任务锚点与末尾 100 字符中较晚的字符边界。"""
    if dataset in ("qasper", "hotpotqa"):
        anchor = text.rfind("Question:")
    elif dataset == "lbv2":
        anchor = text.rfind("Choices:")
    elif dataset in ("multifieldqa_en", "gov_report"):
        anchor = text.rfind("Now,")
    elif dataset == "triviaqa":
        anchor = text.rfind("Answer the question")
    elif dataset == "narrativeqa":
        anchor = text.rfind("Do not provide")
    else:
        anchor = -1
    return max(0, len(text)-100, anchor)


def _ids_sha256(ids):
    return hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()


def canonical_quest_input(tokenizer, text, dataset, *, add_special_tokens=False,
                          require_material_prefix=False):
    """完整文本只分词一次，跨界 token 归入 suffix，保证 IDs 拼接逐位相等。"""
    encoding = tokenizer(text, add_special_tokens=add_special_tokens,
                         return_offsets_mapping=True)
    raw_ids = encoding["input_ids"]
    if hasattr(raw_ids, "tolist"):
        raw_ids = raw_ids.tolist()
    if raw_ids and isinstance(raw_ids[0], (list, tuple)):
        if len(raw_ids) != 1:
            raise ValueError("canonical Quest input only supports B=1")
        raw_ids = raw_ids[0]
    full_ids = tuple(int(token) for token in raw_ids)
    raw_offsets = encoding["offset_mapping"]
    if hasattr(raw_offsets, "tolist"):
        raw_offsets = raw_offsets.tolist()
    offsets = [tuple(map(int, offset)) for offset in raw_offsets]
    if not full_ids or len(full_ids) != len(offsets):
        raise ValueError("canonical Quest input requires nonempty fast-tokenizer offsets")
    boundary = quest_character_boundary(text, dataset)
    split = 0
    while split < len(full_ids) and offsets[split][1] <= boundary:
        split += 1
    prefix_ids, suffix_ids = full_ids[:split], full_ids[split:]
    if prefix_ids+suffix_ids != full_ids:
        raise ValueError("canonical Quest token partition is not lossless")
    material_tokens = sum(end > start for start, end in offsets[:split])
    if require_material_prefix and not material_tokens:
        raise ValueError("Quest prefix freeze requires a nonempty material prefix")
    crossing = bool(split < len(offsets)
                    and offsets[split][0] < boundary < offsets[split][1])
    metadata = dict(
        schema="quest-canonical-tokenization-v1",
        add_special_tokens=bool(add_special_tokens),
        requested_character_boundary=boundary,
        realized_token_boundary=split,
        crossing_token_adjusted=crossing,
        crossing_token_offsets=(list(offsets[split]) if crossing else None),
        full_tokens=len(full_ids), prefix_tokens=len(prefix_ids),
        suffix_tokens=len(suffix_ids), material_prefix_tokens=material_tokens,
        full_ids_sha256=_ids_sha256(full_ids),
        prefix_ids_sha256=_ids_sha256(prefix_ids),
        suffix_ids_sha256=_ids_sha256(suffix_ids),
    )
    return QuestTokenInput(full_ids, prefix_ids, suffix_ids, metadata)
