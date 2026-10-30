"""Exercise actual page-reclaim metadata without importing Torch or allocating KV."""
import ast
from dataclasses import dataclass
import heapq
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('TLI_PAGED_RECLAIM_SOURCE', ROOT/'sparse_attn/ops/paged_region_cache.py'))
ns = {'dataclass': dataclass, 'CacheLayerMixin': object, 'heapq': heapq}
for path, name in ((ROOT/'sparse_attn/ops/frozen_region_cache.py', 'RegionGeometry'),
                   (SOURCE, 'PagedRegionLayer')):
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == name]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
Layer, Geometry = ns['PagedRegionLayer'], ns['RegionGeometry']

class Pages(dict):
    scans = 0
    def __iter__(self):
        self.scans += 1
        return super().__iter__()

class ReclaimTests(unittest.TestCase):
    def make_layer(self, geometry, length, near, far):
        layer=Layer(geometry)
        layer.logical_length=length
        layer.skip_near,layer.skip_far=near,far
        layer.frozen=True
        layer.pages=Pages({lp:(lp,object(),object()) for lp in range((length+geometry.page-1)//geometry.page)})
        layer._tables=(object(),object())
        return layer

    def test_kept_far_does_not_traverse_or_invalidate(self):
        for near in (False,True):
            for length in (8192,32768,131072):
                with self.subTest(near=near,length=length):
                    layer=self.make_layer(Geometry(),length,near,False)
                    pages,tables=layer.pages.copy(),layer._tables
                    layer._reclaim()
                    self.assertEqual(layer.pages,pages)
                    self.assertIs(layer._tables,tables)
                    self.assertEqual(layer.free_ids,[])
                    self.assertEqual(layer.reclaimed_pages,0)
                    self.assertEqual(layer.pages.scans,0)

    def test_all_gates_page_boundaries_and_staging(self):
        for page in (4,16,64):
            for sink in (0,2):
                g=Geometry(sink=sink,near=page*2,window=page+1,page=page)
                for near,far in ((False,False),(True,False),(False,True),(True,True)):
                    for length in (1,page-1,page,page+1,page*5-1,page*5,page*5+1,page*8+3):
                        with self.subTest(page=page,sink=sink,near=near,far=far,length=length):
                            layer=self.make_layer(g,length,near,far)
                            before=layer.pages.copy();tables=layer._tables
                            s,f,n=g.bounds(length)
                            # Independent per-token resident oracle includes
                            # invisible near staging when far is kept.
                            resident={i//page for i in range(length)
                                      if i<s or i>=n or (i<f and not far) or (f<=i<n and (not near or not far))}
                            layer._reclaim()
                            self.assertEqual(set(layer.pages),resident)
                            removed=set(before)-resident
                            self.assertEqual(sorted(layer.free_ids),sorted(removed))
                            self.assertEqual(layer.reclaimed_pages,len(removed))
                            for lp in resident:self.assertIs(layer.pages[lp],before[lp])
                            if removed:self.assertIsNone(layer._tables)
                            else:self.assertIs(layer._tables,tables)
                            layer._reclaim()
                            self.assertEqual(layer.reclaimed_pages,len(removed))
                            self.assertEqual(sorted(layer.free_ids),sorted(removed))

if __name__=='__main__':unittest.main()
