"""Host-only selector option validation before a model mutates its KV cache."""
import os


def validate_selector_options(capacity=None):
    values=[]
    for name in ('TLI_SPARSE_SORT_INT16','TLI_COMPACT_SCAN_TILED'):
        value=os.environ.get(name,'0')
        if value not in ('0','1'):
            raise ValueError(name+' must be 0 or 1')
        values.append(value=='1')
    if values[0] and capacity is not None and (type(capacity) is not int or not 0<capacity<65536):
        raise ValueError('packed sparse sort requires integer capacity <65536')
    return tuple(values)
