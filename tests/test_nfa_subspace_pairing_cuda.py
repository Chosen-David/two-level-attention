"""Native gates for dimension pairing; run on CUDA before adopting the fix."""
import importlib.util
from types import SimpleNamespace
import unittest

if importlib.util.find_spec('torch') is not None:
    import torch
else:
    torch = None

CUDA = (torch is not None and torch.cuda.is_available()
        and importlib.util.find_spec('triton') is not None)


@unittest.skipUnless(CUDA, 'requires Torch, Triton and CUDA')
class PairingCudaTest(unittest.TestCase):
    def test_paged_fine_matches_independent_coordinate_reference(self):
        from sparse_attn.indexer.nfa_layer_batch_selector import (
            _fused_fine_probabilities_paged, _prepare_q_kernel)
        import triton

        # D=192 exercises the padded D_BLOCK=256 query loads under memcheck.
        for d in (128, 192, 256):
            for use_full in (False, True):
                with self.subTest(d=d, full=use_full):
                    b, hk, g, page, blocks, max_req = 2, 2, 2, 4, 3, 3
                    pool = torch.zeros((6, hk, page, d), device='cuda',
                                       dtype=torch.bfloat16)
                    for physical in range(6):
                        pool[physical, :, :, 0] = physical / 8
                        pool[physical, :, :, d-1] = (5-physical) / 8
                    layer = SimpleNamespace(kv_pool_k=pool, num_kv_heads=hk,
                                            head_dim_k=d)
                    table = torch.zeros((2, max_req, blocks), device='cuda',
                                        dtype=torch.int32)
                    table[1, 2] = torch.tensor([5, 3, 1], device='cuda')
                    table[1, 0] = torch.tensor([0, 2, 4], device='cuda')
                    cache = SimpleNamespace(layers=[layer, layer],
                        geometry=SimpleNamespace(page=page), max_blocks=blocks,
                        max_requests=max_req, block_table_buf=table,
                        layer_kv_ptrs_buf=torch.tensor([pool.data_ptr()]*2,
                            dtype=torch.int64, device='cuda'))
                    req_ids = torch.tensor([2, 0], device='cuda', dtype=torch.int64)
                    dims = torch.arange(0 if use_full else d-32, d,
                                        device='cuda', dtype=torch.int64)
                    q_src = torch.zeros((1, b, hk*g, d), device='cuda',
                                        dtype=torch.bfloat16)
                    q_src[..., 0] = 1
                    q_src[..., d-1] = -1
                    q = torch.empty((b, hk*g, dims.numel()), device='cuda')
                    rows = torch.arange(b, device='cuda', dtype=torch.int64)
                    scales = torch.ones(1, device='cuda')
                    _prepare_q_kernel[(b*hk*g,)](
                        q_src, rows, scales, dims, q, *q_src.stride()[:3],
                        HQ=hk*g, D=d, DQ=dims.numel(), B=b,
                        DQ_BLOCK=triton.next_power_of_2(dims.numel()), num_warps=4)
                    ids = torch.tensor([0, 3, 4, 7, 8, 9, 10, 11],
                        device='cuda', dtype=torch.int64).expand(b, hk, -1).contiguous()
                    counts = torch.tensor([[5, 5], [8, 8]], device='cuda',
                                          dtype=torch.int64)
                    keep = torch.arange(8, device='cuda')[None, None] < counts[..., None]
                    out, unique = _fused_fine_probabilities_paged(
                        cache, req_ids, [pool.data_ptr()], dims, q, ids, keep,
                        counts, blocks*page, g, layer_offset=1)
                    for row, rid in enumerate((2, 0)):
                        for h in range(hk):
                            count = (5, 8)[row]
                            logical = ids[row, h, :count]
                            physical = table[1, rid, logical // page].long()
                            keys = pool[physical, h, logical % page][:, dims].float()
                            queries = q_src[0, row, h*g:(h+1)*g][:, dims].float()
                            logits = (queries @ keys.T).bfloat16().float()
                            ref = logits.softmax(-1).bfloat16().float().mean(0).bfloat16()
                            torch.testing.assert_close(out[row, h, :count], ref,
                                                       atol=2e-3, rtol=1e-2)
                            self.assertEqual(out[row, h, :count].argmax().item(),
                                             ref.argmax().item())
                            self.assertTrue(unique[row, h, :count].all().item())
                    torch.cuda.synchronize()

    def test_actual_selector_modes_eager_and_graph(self):
        from sparse_attn.cache import GraphReadyPagedRegionCache
        from sparse_attn.indexer.nfa_indexer import NFAIndexer
        from sparse_attn.indexer.nfa_layer_batch_selector import select_topk_for_layer_batched
        from sparse_attn.ops.paged_region_kernels_nfa import nfa_args
        from sparse_attn.policies.region_geometry import RegionGeometry

        geometry = RegionGeometry(sink=4, near=128, window=64,
                                  page=64, max_new_tokens=8)
        # Four frozen policies, unequal lengths and noncontiguous request IDs.
        req_ids = [1, 3, 6, 7]
        for subspace, l2, enable in (
                ('tail', 'same', True), ('tail', 'full', True),
                ('full', 'same', True), ('full', 'full', True),
                ('highfreq', 'same', False)):
            with self.subTest(subspace=subspace, l2=l2, enable=enable):
                args = nfa_args()
                args.tia_level1_topk, args.tia_level2_topk = 4, 128
                args.nfa_near_pages, args.nfa_near_tokens = 2, 'off'
                args.nfa_near_len, args.nfa_swa_size = 128, 64
                args.nfa_sink_tokens, args.tia_level2_cmp_ratio = 4, 4
                args.nfa_subspace, args.nfa_l2_subspace = subspace, l2
                args.nfa_enable_subspace = enable
                attention = SimpleNamespace(indexer=NFAIndexer(args), scaling=1.0)
                cache = GraphReadyPagedRegionCache(
                    1, geometry, max_requests=8, max_physical_pages=64,
                    max_blocks=16, max_visible_len=640, max_query_tokens=8,
                    num_kv_heads=1, num_query_heads=2, head_dim_k=128,
                    head_dim_v=128, dtype=torch.bfloat16, device='cuda')
                torch.manual_seed(20261102)
                for rid, length, gates in zip(req_ids, [448, 449, 510, 513],
                        [(False, False), (True, False), (False, True), (True, True)]):
                    cache.add_request(rid)
                    k = torch.randn(1, 1, length, 128, device='cuda', dtype=torch.bfloat16)
                    cache.write_prefill(0, rid, k, torch.randn_like(k))
                    cache.freeze(rid, [gates[0]], [gates[1]])
                cache.new_step(req_ids)
                cache.plan_common(req_ids)
                k = torch.randn(4, 1, 1, 128, device='cuda', dtype=torch.bfloat16)
                cache.write_decode(0, k, torch.randn_like(k),
                                   cache.slot_mapping_for_layer(0)[:4])
                query = torch.randn(4, 2, 128, device='cuda', dtype=torch.bfloat16)

                def run():
                    select_topk_for_layer_batched(cache, 0, req_ids, query,
                                                 attention=attention)

                warmup = torch.cuda.Stream()
                warmup.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(warmup):
                    run()
                    run()
                torch.cuda.current_stream().wait_stream(warmup)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run()
                query.neg_()
                graph.replay()
                graph_ids = cache.gather_indices_for_layer(0).clone()
                graph_counts = cache.visible_lens_for_layer(0).clone()
                run()
                self.assertTrue(torch.equal(graph_counts, cache.visible_lens_for_layer(0)))
                for row in range(4):
                    count = int(graph_counts[row])
                    self.assertGreater(count, 0)
                    self.assertTrue(torch.equal(graph_ids[row, :count],
                        cache.gather_indices_for_layer(0)[row, :count]))
                torch.cuda.synchronize()


if __name__ == '__main__':
    unittest.main()
