"""Host facade regression for cache invalidation, not a native Tensor/GPU test."""
import ast
import os
from pathlib import Path
import unittest

SOURCE = Path(os.environ.get('TLI_REVISION_SOURCE', Path(__file__).resolve().parents[1] / 'sparse_attn/indexer/tli_indexer.py'))

class TensorFacade:
    def __init__(self, ptr=100, base=None, inference=False):
        self.ptr, self._base, self.inference = ptr, base, inference
        self.version = 0
    @property
    def _version(self):
        if self.inference:
            raise RuntimeError('no inference version')
        return self.version if self._base is None else self._base._version
    def data_ptr(self):
        return self.ptr
    def row(self, row):
        root = self if self._base is None else self._base
        return TensorFacade(self.ptr + 8 * row, root, self.inference)

class DeviceRevisionTests(unittest.TestCase):
    def setUp(self):
        functions = [n for n in ast.parse(SOURCE.read_text()).body
                     if isinstance(n, ast.FunctionDef) and n.name in ('_graph_geo_lookup', '_graph_geo_store')]
        self.ns = {'_GRAPH_GEO_CACHE': {}}
        exec(compile(ast.Module(functions, []), str(SOURCE), 'exec'), self.ns)
        self.lookup, self.store = self.ns['_graph_geo_lookup'], self.ns['_graph_geo_store']

    def test_fresh_and_nested_views_across_device_steps(self):
        for batch in (1, 2, 8):
            for nested in (False, True):
                self.ns['_GRAPH_GEO_CACHE'].clear()
                root = TensorFacade()
                for step in range(4):
                    root._tli_device_revision = step
                    source = root.row(0) if nested else root
                    for row in range(batch):
                        with self.subTest(batch=batch,nested=nested,step=step,row=row):
                            # A fresh view has the same pointer but no copied attributes.
                            self.assertIsNone(self.lookup(source.row(row), 'geometry'))
                            value = object()
                            self.store(source.row(row), 'geometry', value)
                            self.assertIs(self.lookup(source.row(row), 'geometry'), value)

    def test_direct_and_view_local_revisions(self):
        for tensor in (TensorFacade(), TensorFacade().row(0)):
            self.ns['_GRAPH_GEO_CACHE'].clear()
            value = object(); self.store(tensor, 'g', value)
            tensor._tli_device_revision = 1
            self.assertIsNone(self.lookup(tensor, 'g'))

    def test_torch_version_and_geometry_keys_remain_binding(self):
        root = TensorFacade(); value = object()
        self.store(root.row(0), ('capture1','stream1'), value)
        self.assertIsNone(self.lookup(root.row(0), ('capture2','stream1')))
        self.assertIsNone(self.lookup(root.row(0), ('capture1','stream2')))
        root.version += 1
        self.assertIsNone(self.lookup(root.row(0), ('capture1','stream1')))

    def test_inference_still_never_reuses(self):
        root = TensorFacade(inference=True); root._tli_device_revision = 1
        value = object()
        self.assertIs(self.store(root.row(0), 'g', value), value)
        self.assertIsNone(self.lookup(root.row(0), 'g'))
        self.assertEqual(self.ns['_GRAPH_GEO_CACHE'], {})

if __name__ == '__main__': unittest.main()
