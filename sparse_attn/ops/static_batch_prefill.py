"""Consume independent unpadded prefills into a fresh static batch, layer by layer.

Keeping only the rows already prefilled avoids allocating the entire batch KV
beside each later prefill's dynamic KV and MLP activations. Conversion holds at
most one extra layer of KV; the input caches are consumed and cannot be reused.
"""
from .tli_static_cache import TLIStaticCache


def consume_prefill_rows(cache, rows, lengths):
    if (type(cache) is not TLIStaticCache or not cache.layers
            or len(rows) != cache.batch_size):
        raise ValueError('fresh static cache and one dynamic cache per batch row required')
    if len(lengths) != len(rows) or any(
            type(n) is not int or not 0 < n <= cache.max_cache_len for n in lengths):
        raise ValueError('invalid unpadded prefill lengths')
    if any(layer.keys is not None for layer in cache.layers):
        raise ValueError('destination cache must be fresh')
    if any(len(row.layers) != len(cache.layers) for row in rows):
        raise ValueError('prefill layer count mismatch')
    # Validate the complete transfer before consuming any source tensor.
    for li in range(len(cache.layers)):
        geometry = None
        for row, n in zip(rows, lengths):
            src = row.layers[li]
            k, v = src.keys, src.values
            if (k is None or v is None or k.ndim != 4 or k.shape[0] != 1
                    or k.shape[-2] != n or v.shape != k.shape
                    or v.dtype != k.dtype or v.device != k.device
                    or k.requires_grad or v.requires_grad):
                raise ValueError('source must be an unpadded single-row KV pair')
            current = (k.shape[1], k.shape[-1], k.dtype, k.device)
            if geometry is not None and geometry != current:
                raise ValueError('incompatible KV rows in a layer')
            geometry = current
    # Do not retain the validation loop's last tensor during conversion.
    del k, v, src
    for li, dest in enumerate(cache.layers):
        dest.lazy_initialization(rows[0].layers[li].keys.expand(len(rows), -1, -1, -1))
        for b, (row, n) in enumerate(zip(rows, lengths)):
            src = row.layers[li]
            dest.keys[b, :, :n].copy_(src.keys[0])
            dest.values[b, :, :n].copy_(src.values[0])
        # Release only after the entire layer's copies have been submitted.
        # Same-stream allocator ordering keeps asynchronous CUDA copies safe.
        for row in rows:
            row.layers[li].keys = None
            row.layers[li].values = None
        dest.cumulative_length = max(lengths)
