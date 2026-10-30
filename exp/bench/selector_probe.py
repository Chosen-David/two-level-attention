"""Capture selector inputs at one real decode layer/step, without altering them.

Experimental observer only. Extra copies/readback invalidate timing. A caller
must independently verify that generated tokens match an uninstrumented run.
"""
from contextlib import contextmanager
from pathlib import Path
import torch


@contextmanager
def record_selector_probe(directory, layer, step):
    from sparse_attn.indexer.tli_indexer import TLIIndexer
    if layer < 0 or step < 0:
        raise ValueError('nonnegative layer and step required')
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    score, mask = TLIIndexer.compute_score, TLIIndexer.compute_mask_graph
    replay = torch.cuda.CUDAGraph.replay
    captures, graphs = [], {}
    eager_step = 0

    def snapshot(self, q, index, scale, length, q_ids=None):
        tensors = {k: v.detach().clone() for k, v in index.items()
                   if isinstance(v, torch.Tensor)}
        tensors['q'] = q.detach().clone()
        if isinstance(length, torch.Tensor):
            tensors['probe_length'] = length.detach().clone()
        else:
            # No pageable host tensor construction inside capture.
            tensors['probe_length'] = length
        if q_ids is not None:
            tensors['probe_q_ids'] = q_ids.detach().clone()
        return dict(tensors=tensors, layer=self.layer_idx, skip_far=self.skip_far,
                    scale=scale)

    def save(name, row):
        result = dict(row)
        result['tensors'] = {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                             for k, v in row['tensors'].items()}
        with (root / name).open('xb') as f:
            torch.save(result, f)

    def observe_score(self, q, q_ids, index_dict, softmax_scale):
        nonlocal eager_step
        if self.layer_idx == layer and q.shape[1] == 1:
            if eager_step == step:
                save('eager.pt', snapshot(self, q, index_dict, softmax_scale,
                                          index_dict.get('valid_length'), q_ids))
            eager_step += 1
        return score(self, q, q_ids, index_dict, softmax_scale)

    def observe_mask(self, q, index_dict, softmax_scale, valid_len, **kwargs):
        if self.layer_idx == layer and torch.cuda.is_current_stream_capturing():
            captures.append(snapshot(self, q, index_dict, softmax_scale, valid_len))
        return mask(self, q, index_dict, softmax_scale, valid_len, **kwargs)

    def observe_replay(graph):
        result = replay(graph)
        if graph not in graphs:
            graphs[graph] = [0, list(captures)]
            captures.clear()
        current, rows = graphs[graph]
        if current == step:
            if len(rows) != 1:
                raise RuntimeError(f'expected one captured layer, got {len(rows)}')
            save(f'graph_{len(graphs)-1:03d}.pt', rows[0])
        graphs[graph][0] += 1
        return result

    TLIIndexer.compute_score = observe_score
    TLIIndexer.compute_mask_graph = observe_mask
    torch.cuda.CUDAGraph.replay = observe_replay
    try:
        yield
    finally:
        TLIIndexer.compute_score = score
        TLIIndexer.compute_mask_graph = mask
        torch.cuda.CUDAGraph.replay = replay
