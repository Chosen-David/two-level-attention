"""Shared region geometry for paged and gate selection."""
from dataclasses import dataclass


@dataclass(frozen=True)
class RegionGeometry:
    sink: int = 4
    near: int = 4096
    window: int = 128
    page: int = 64
    max_new_tokens: int = 512

    def __post_init__(self):
        for name in ('sink', 'near', 'window', 'page', 'max_new_tokens'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < (0 if name == 'sink' else 1):
                raise ValueError(f'invalid {name}: {value}')

    def bounds(self, length):
        sink_hi = min(self.sink, length)
        near_hi = max(sink_hi, length - self.window)
        far_hi = min(near_hi, max(sink_hi, ((length - self.window - self.near) // self.page) * self.page))
        return sink_hi, far_hi, near_hi
