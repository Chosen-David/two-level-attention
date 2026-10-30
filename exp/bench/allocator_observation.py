"""Read-only allocator evidence, collected outside request timing boundaries."""
import os
import json
import sys
import torch


def observe_allocator(phase='request_complete'):
    result = {'requested': os.environ.get('PYTORCH_CUDA_ALLOC_CONF'),
              'scope': phase + ' observation; outside request timing'}
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


def write_failure_diagnostic(path, error):
    try:
        path.write_text(json.dumps(dict(error=f'{type(error).__name__}: {error}',
                                       allocator=observe_allocator('failure')), indent=2))
    except Exception as diagnostic_error:
        print(f'Could not write allocator failure diagnostic: {diagnostic_error}', file=sys.stderr)
