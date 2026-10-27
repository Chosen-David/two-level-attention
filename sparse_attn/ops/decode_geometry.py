"""Geometry shared only within one model forward or capture invocation."""
from contextlib import contextmanager
from contextvars import ContextVar


_scope = ContextVar('tli_decode_geometry', default=None)


@contextmanager
def decode_geometry_scope():
    # Never inherit warmup results into capture, or a previous request's graph.
    token = _scope.set({})
    try:
        yield
    finally:
        _scope.reset(token)


def scoped_geometry(length, key, build):
    cache = _scope.get()
    if cache is None:
        return None
    identity = (length.device, length.data_ptr(), tuple(length.shape),
                tuple(length.stride()), key)
    if identity not in cache:
        # Pin the input until the scope exits; no pointer ABA or global cache
        # eviction at B>8. The caller keeps lengths immutable within a forward.
        cache[identity] = (length, build())
    return cache[identity][1]
