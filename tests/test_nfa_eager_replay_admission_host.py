"""Execute production plan admission/replay AST without Torch or a GPU.

These tests check argument ownership, not native kernels or model quality.
NFA_SELECTOR_SOURCE can point to a retained pre-fix source for red/green runs.
"""
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
    return next(n for n in TREE.body if isinstance(n, ast.FunctionDef) and n.name == name)

def execute(nodes, env):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), env)

class Tensor:
    shape = (2, 8, 128)
    dtype = 'bf16'
    def contiguous(self): return self
    def unsqueeze(self, dim): return self
    def data_ptr(self): return 16
    def stride(self, dim): return (2048, 1024, 128)[dim]

class ReplayAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.fused, self.prepare, self.write = object(), object(), object()
        self.geometry = tuple([2, 1, 8] + [0]*9 + [64] + [0]*14 +
                              ['bf16', 0, 2048, 1024, 128, True])
        geometry_ast = next(n.value for n in function('_layer_batched_select_impl').body
                            if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'geometry')
        self.geometry = self.geometry[:len(geometry_ast.elts)]
        self.old, self.new = Tensor(), Tensor()
        self.key = ('eager_selector_call_plan', 0)

    def make_plan(self, fused, writes):
        # Execute the actual producer's final plan-construction statements.
        impl = function('_layer_batched_select_impl')
        start = next(i for i,n in enumerate(impl.body) if isinstance(n,ast.Assign)
                     and ast.unparse(n.targets[0]) == 'record')
        args = tuple([self.old]*24)
        kernel = self.fused if fused else self.prepare
        record = [('producer', kernel, (2,), args, {})]
        if not fused and writes:
            record.append(('paged_write', self.write, (2,), (self.old,)*14, {}))
        workspace = {'_nfa_eager_record': record}
        env = dict(workspace=workspace, geometry=self.geometry, plan_key=self.key,
                   fixed_candidate_buffers=True, fuse_prepare_q=fused,
                   decode_write=(self.old,)*3 if writes else None)
        execute(impl.body[start:], env)
        return workspace

    def test_only_rebindable_plans_are_retained(self):
        for fused in (False, True):
            for writes in (False, True):
                with self.subTest(fused=fused, writes=writes):
                    workspace = self.make_plan(fused, writes)
                    self.assertEqual(self.key in workspace, fused and writes)
                    self.assertNotIn('_nfa_eager_record', workspace)

    def test_fused_replay_updates_all_live_inputs(self):
        for compiled in (False, True):
            for hook in (None, object()):
                with self.subTest(compiled=compiled, hook=hook):
                    seen = []
                    def launch(workspace,key,kernel,grid,*args,**kwargs): seen.append(args)
                    def run(*args): seen.append(args[9:])
                    env = {'_nfa_page_stats_prepare_q_kernel': self.fused,
                           '_fast_launch': launch}
                    execute([function('_replay_eager_selector_call_plan')], env)
                    args = tuple([self.old]*24)
                    plan = (self.geometry, [('producer',self.fused,(2,),args,{},
                        run if compiled else None,None,None,None,args,1,1)])
                    keys,values,slots = Tensor(),Tensor(),Tensor()
                    env['_replay_eager_selector_call_plan'](
                        {'_nfa_launch_context':(0,hook,None,False)}, plan,
                        self.new, (keys,values,slots))
                    self.assertIs(seen[0][10], self.new)
                    self.assertEqual(seen[0][17:20], (keys,values,slots))
                    self.assertIs(seen[0][0], self.old)  # static cache metadata

    def test_write_to_no_write_call_uses_normal_dispatch(self):
        workspace = self.make_plan(True, True)
        workspace['_nfa_launch_geometry'] = self.geometry
        cache = NS(_nfa_selector_workspace=workspace, active_visible_bucket=64,
                   layers=[NS(head_dim_k=128)])
        routes = []
        env = dict(_has_triton=lambda:True,
            torch=NS(cuda=NS(is_current_stream_capturing=lambda:False)),
            _replay_eager_selector_call_plan=lambda *a: routes.append('replay'),
            _layer_batched_select_impl=lambda *a,**k: routes.append('normal'))
        execute([function('select_topk_for_layer_batched')], env)
        env['select_topk_for_layer_batched'](cache,0,[0,1],self.new,decode_write=None)
        self.assertEqual(routes,['normal'])
        routes.clear()
        env['select_topk_for_layer_batched'](cache,0,[0,1],self.new,
                                            decode_write=(self.new,)*3)
        self.assertEqual(routes,['replay'])

    def test_write_presence_is_part_of_specialization_geometry(self):
        impl = function('_layer_batched_select_impl')
        value = next(n.value for n in impl.body if isinstance(n, ast.Assign)
                     and ast.unparse(n.targets[0]) == 'geometry')
        # Execute the actual final tuple element; other geometry dimensions
        # are held fixed. HAS_WRITE must not reuse the previous specialization.
        code = compile(ast.Expression(value.elts[-1]), str(SOURCE), 'eval')
        self.assertIs(eval(code, {'decode_write': None, 's2': 128}), False)
        self.assertIs(eval(code, {'decode_write': (self.new,)*3, 's2': 128}), True)

    def test_current_nonfused_replay_would_retain_old_inputs(self):
        # This intentionally documents why admission must reject such plans.
        seen=[]
        env=dict(_nfa_page_stats_prepare_q_kernel=self.fused,
                 _fast_launch=lambda w,k,f,g,*a,**kw:seen.append(a))
        execute([function('_replay_eager_selector_call_plan')],env)
        entries=[('q',self.prepare,(2,),(self.old,)*24,{},None,None,None,None,(),1,1),
                 ('paged_write',self.write,(2,),(self.old,)*14,{},None,None,None,None,(),1,1)]
        env['_replay_eager_selector_call_plan'](
            {'_nfa_launch_context':(0,None,None,False)},(self.geometry,entries),
            self.new,(self.new,)*3)
        self.assertIs(seen[0][0], self.old)
        self.assertEqual(seen[1][:3],(self.old,)*3)

if __name__ == '__main__': unittest.main()
