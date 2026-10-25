
比如128K的数据，我们方法 K1 = 128, K2 = 1024
compute_score(sink_k, swa_k, q) # sink和swa直接拼起来上tensorcore?

page_ids_near = indexerL1_near(near_k, nb, q) # 对于nl长度的near_k粗筛选中了哪些page_ids
if not skip_far:
    page_ids_far = indexerL1_far(far_k, K1-nb, q) # far区域选中了哪些page_ids

token_ids = indexerL2(page_ids, mid_k, K2, q) # 细筛选中了哪些token
gather_and_compute_score(token_ids, q) # gather到一起算分



