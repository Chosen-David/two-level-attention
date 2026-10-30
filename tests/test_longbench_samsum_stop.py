from types import SimpleNamespace

import torch

from benchmark.LongBench.pred import _manual_generate


class StopModel:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        logits = torch.zeros(1, 1, 16)
        if self.calls == 0:
            logits[0, -1, 7] = 5
            logits[0, -1, 2] = 10
        else:
            logits[0, -1, 10] = 10
        self.calls += 1
        return SimpleNamespace(logits=logits, past_key_values=object())


def test_samsum_manual_path_suppresses_eos_for_first_token_then_stops_on_newline():
    model = StopModel()
    generated = _manual_generate(model, torch.tensor([[3]]), torch.empty(1, 0, dtype=torch.long),
                                 4, [2, 10], min_new_tokens=1)
    assert generated == [7, 10]
    assert model.calls == 2
