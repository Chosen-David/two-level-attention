"""Read-only allocator evidence, collected outside request timing boundaries."""
import os
import torch


def observe_allocator():
    result = {'requested': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
              'scope': 'post-request or failure observation; not timed'}
    try:
        snapshot = torch.cuda.memory._snapshot()  # Pinned torch 2.8 diagnostic API.
        segments = snapshot['segments']
        result.update(
            backend=torch.cuda.memory.get_allocator_backend(),
            settings=snapshot.get('allocator_settings'),
            allocated_bytes=torch.cuda.memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved(),
            segment_count=len(segments),
            expandable_segment_count=(sum(s['is_expandable'] for s in segments)
                                      if segments and all('is_expandable' in s for s in segments)
                                      else None),
            inactive_split_bytes=torch.cuda.memory_stats().get('inactive_split_bytes.all.current'))
    except Exception as exc:
        # A diagnostic failure must not hide the original OOM/CUDA exception.
        result['observation_error'] = f'{type(exc).__name__}: {exc}'
    return result
