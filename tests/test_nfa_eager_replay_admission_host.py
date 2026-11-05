"""在无 GPU 环境执行 eager replay 准入与重绑的生产 AST。"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace as NS
import unittest


SOURCE = Path(os.environ.get('NFA_SELECTOR_SOURCE', str(
    Path(__file__).resolve().parents[1] /
    'sparse_attn/indexer/nfa_layer_batch_selector.py')))
TREE = ast.parse(SOURCE.read_text())


def function(name):
    return next(node for node in TREE.body
                if isinstance(node, ast.FunctionDef) and node.name == name)


def execute(nodes, env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)


class Tensor:
    shape = (2, 8, 128)
    dtype = 'bf16'

    def contiguous(self):
        return self

    def unsqueeze(self, dim):
        return self

    def data_ptr(self):
        return 16

    def stride(self, dim):
        return (2048, 1024, 128)[dim]


class ReplayAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.fused, self.prepare, self.write = object(), object(), object()
        values = ([2, 1, 8] + [0] * 9 + [64] + [0] * 14
                  + ['bf16', 0, 2048, 1024, 128, False, False])
        geometry_ast = next(
            node.value for node in function('_layer_batched_select_impl').body
            if isinstance(node, ast.Assign)
            and ast.unparse(node.targets[0]) == 'geometry')
        self.geometry_size = len(geometry_ast.elts)
        self.geometry = tuple(values[:self.geometry_size])
        self.old, self.new = Tensor(), Tensor()
        self.key = ('eager_selector_call_plan', 0)

    def with_geometry(self, *, protected=False, writes=False):
        values = list(self.geometry)
        values[32] = protected
        if self.geometry_size > 33:
            values[33] = writes
        return tuple(values)

    def make_plan(self, fused, writes):
        impl = function('_layer_batched_select_impl')
        starts = [index for index, node in enumerate(impl.body)
                  if isinstance(node, ast.Assign)
                  and ast.unparse(node.targets[0]) == 'record']
        args = tuple([self.old] * 24)
        kernel = self.fused if fused else self.prepare
        record = [('producer', kernel, (2,), args, {})]
        if not fused and writes:
            record.append(('paged_write', self.write, (2,),
                           (self.old,) * 14, {}))
        geometry = self.with_geometry(writes=writes)
        workspace = {'_nfa_eager_record': record,
                     '_nfa_launch_geometry': geometry}
        env = dict(workspace=workspace, geometry=geometry,
                   plan_key=self.key, fuse_prepare_q=fused,
                   decode_write=(self.old,) * 3 if writes else None)
        execute([function('_store_eager_selector_call_plan')], env)
        execute(impl.body[starts[-1]:], env)
        return workspace

    def public_entry(self, workspace, routes, writes=None):
        cache = NS(_nfa_selector_workspace=workspace, active_visible_bucket=64,
                   layers=[NS(head_dim_k=128)], states={},
                   write_decode=lambda *args: routes.append(('write', args)))
        env = dict(_has_triton=lambda: True,
                   torch=NS(cuda=NS(is_current_stream_capturing=lambda: False)),
                   _replay_eager_selector_call_plan=lambda *args: routes.append('replay'),
                   _layer_batched_select_impl=lambda *args, **kwargs: routes.append('normal'))
        execute([function('select_topk_for_layer_batched')], env)
        return cache, env

    def test_nonfused_write_plan_is_not_retained(self):
        workspace = self.make_plan(fused=False, writes=True)
        self.assertNotIn(self.key, workspace)
        self.assertNotIn('_nfa_eager_record', workspace)

    def test_rebindable_plans_are_retained(self):
        for fused, writes in ((False, False), (True, False), (True, True)):
            with self.subTest(fused=fused, writes=writes):
                workspace = self.make_plan(fused=fused, writes=writes)
                self.assertIn(self.key, workspace)
                self.assertNotIn('_nfa_eager_record', workspace)

    def test_second_nonfused_write_uses_normal_dispatch(self):
        workspace = self.make_plan(fused=False, writes=True)
        workspace['_nfa_launch_req_ids'] = (0, 1)
        routes = []
        cache, env = self.public_entry(workspace, routes)
        env['select_topk_for_layer_batched'](
            cache, 0, [0, 1], self.new, decode_write=(self.new,) * 3)
        self.assertEqual(routes, ['normal'])

    def test_write_presence_mismatch_uses_normal_dispatch(self):
        workspace = self.make_plan(fused=False, writes=False)
        workspace['_nfa_launch_req_ids'] = (0, 1)
        routes = []
        cache, env = self.public_entry(workspace, routes)
        env['select_topk_for_layer_batched'](
            cache, 0, [0, 1], self.new, decode_write=(self.new,) * 3)
        self.assertEqual(routes, ['normal'])

    def test_write_presence_is_part_of_specialization_geometry(self):
        impl = function('_layer_batched_select_impl')
        value = next(node.value for node in impl.body
                     if isinstance(node, ast.Assign)
                     and ast.unparse(node.targets[0]) == 'geometry')
        code = compile(ast.Expression(value.elts[-1]), str(SOURCE), 'eval')
        self.assertIs(eval(code, {'decode_write': None, 'protected_only': False}), False)
        self.assertIs(eval(code, {'decode_write': (self.new,) * 3,
                                  'protected_only': False}), True)

    def test_fused_replay_updates_all_live_inputs(self):
        for compiled in (False, True):
            with self.subTest(compiled=compiled):
                seen = []

                def launch(workspace, key, kernel, grid, *args, **kwargs):
                    seen.append(args)

                def run(*args):
                    seen.append(args[9:])

                env = {'_nfa_page_stats_prepare_q_kernel': self.fused,
                       '_prepare_q_kernel': object(),
                       '_prepare_q_grouped_kernel': object(),
                       '_fast_launch': launch}
                execute([function('_replay_eager_selector_call_plan')], env)
                args = tuple([self.old] * 24)
                plan = (self.with_geometry(writes=True),
                        [('producer', self.fused, (2,), args, {},
                          run if compiled else None, None, None, None, args, 1, 1)])
                keys, values, slots = Tensor(), Tensor(), Tensor()
                env['_replay_eager_selector_call_plan'](
                    {'_nfa_launch_context': (0, None, None, False)}, plan,
                    self.new, (keys, values, slots))
                self.assertIs(seen[0][10], self.new)
                self.assertEqual(seen[0][17:20], (keys, values, slots))

    def test_protected_replay_still_writes_current_kv(self):
        geometry = self.with_geometry(protected=True, writes=True)
        workspace = {self.key: (geometry, []),
                     '_nfa_launch_geometry': geometry,
                     '_nfa_launch_req_ids': (0, 1)}
        routes = []
        cache, env = self.public_entry(workspace, routes)
        cache.states = {rid: NS(skip_far_by_layer=[True],
                                skip_near_by_layer=[True]) for rid in (0, 1)}
        decode_write = (self.new, self.new, self.new)
        env['select_topk_for_layer_batched'](
            cache, 0, [0, 1], self.new, decode_write=decode_write)
        self.assertEqual(routes, [('write', (0,) + decode_write), 'replay'])


if __name__ == '__main__':
    unittest.main()
