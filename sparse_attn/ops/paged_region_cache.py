"""Request-local paged KV: region aging changes metadata, never copies old KV.

Each logical page owns independently allocated K/V tensors. Reclaiming a page
releases its tensor references to the Torch allocator, and recycles its page ID.
Allocator-reserved memory is not reported as freed driver memory. Eager B1 only.
"""
import heapq
import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from .frozen_region_cache import RegionGeometry


class PagedRegionLayer(CacheLayerMixin):
    is_compileable = False
    is_sliding = False

    def __init__(self, geometry):
        super().__init__()
        self.geometry = geometry
        self.pages = {}
        self.free_ids = []
        self.next_id = 0
        self.logical_length = self.prompt_length = 0
        self.frozen = self.skip_near = self.skip_far = False
        self.reclaimed_pages = self.reused_page_ids = 0
        self.aging_copy_bytes = 0
        self.write_bytes = 0
        self._tables = None
        self.tli_receipt = None
        self._stream_id = None

    def _check_stream(self, device):
        if device.type == "cuda":
            stream = torch.cuda.current_stream(device).cuda_stream
            if self._stream_id is not None and stream != self._stream_id:
                raise ValueError("paged request must stay on its owning CUDA stream")
            self._stream_id = stream

    def lazy_initialization(self, key_states):
        pass

    def intervals(self, length=None, resident=False):
        length = self.logical_length if length is None else length
        s, f, n = self.geometry.bounds(length)
        spans = [(0, s), (n, length)]
        if not self.skip_far: spans.append((s, f))
        if not self.skip_near or (resident and not self.skip_far): spans.append((f, n))
        return [(a,b) for a,b in sorted(spans) if a < b]

    def visible_positions(self):
        return [p for a,b in self.intervals() for p in range(a,b)]

    def _reclaim(self):
        # Keeping far also keeps near-only staging: every logical page is
        # resident, so there can be no reclaimable page to scan for.
        if not self.skip_far:
            return
        spans = self.intervals(resident=True)
        size = self.geometry.page
        for lp in list(self.pages):
            lo, hi = lp*size, min((lp+1)*size,self.logical_length)
            if not any(lo < b and a < hi for a,b in spans):
                physical, _, _ = self.pages.pop(lp)
                heapq.heappush(self.free_ids, physical)
                self.reclaimed_pages += 1
                self._tables = None

    def freeze(self, skip_near=False, skip_far=False):
        if self.frozen:
            if (skip_near,skip_far)!=(self.skip_near,self.skip_far):
                raise ValueError('cannot change frozen gates')
            return
        if not self.logical_length: raise ValueError('prefill before freeze')
        self._check_stream(next(iter(self.pages.values()))[1].device)
        self.skip_near,self.skip_far=bool(skip_near),bool(skip_far)
        self.frozen=True
        self._reclaim()

    def _allocate(self, lp, keys, values):
        if self.free_ids:
            physical=heapq.heappop(self.free_ids);self.reused_page_ids+=1
        else:
            physical=self.next_id;self.next_id+=1
        p=self.geometry.page
        k=torch.empty((1,keys.shape[1],p,keys.shape[-1]),dtype=keys.dtype,device=keys.device)
        v=torch.empty((1,values.shape[1],p,values.shape[-1]),dtype=values.dtype,device=values.device)
        self.pages[lp]=(physical,k,v)
        self._tables=None

    def update(self, keys, values, cache_kwargs=None):
        if keys.ndim!=4 or values.ndim!=4 or keys.shape[:3]!=values.shape[:3] or keys.shape[0]!=1:
            raise ValueError('paged cache requires unpadded B1 and matching KV heads/tokens')
        if keys.device!=values.device or keys.dtype!=values.dtype or keys.shape[-1]<1 or values.shape[-1]<1:
            raise ValueError('invalid KV dtype/device/width')
        if keys.requires_grad or values.requires_grad: raise ValueError('inference-only cache')
        self._check_stream(keys.device)
        n=keys.shape[-2]
        if n<=0: raise ValueError('empty update')
        positions=(cache_kwargs or {}).get('cache_position')
        if positions is not None and positions.tolist()!=list(range(self.logical_length,self.logical_length+n)):
            raise ValueError('nonmonotonic cache positions')
        prefill=not self.logical_length
        if not prefill:
            if not self.frozen or n!=1: raise ValueError('freeze after one-shot prefill; decode one token')
            if self.logical_length-self.prompt_length>=self.geometry.max_new_tokens:
                raise ValueError('generation capacity exceeded')
            _,samplek,samplev=next(iter(self.pages.values()))
            if keys.dtype!=samplek.dtype or values.dtype!=samplev.dtype or keys.device!=samplek.device or values.device!=samplev.device:
                raise ValueError('KV dtype/device changed across request')
            if keys.shape[1:]!=(samplek.shape[1],1,samplek.shape[-1]) or values.shape[-1]!=samplev.shape[-1]:
                raise ValueError('KV shape changed across request')
        p=self.geometry.page
        start=self.logical_length
        new_length=start+n
        allocated=[]
        touched={}
        free_before=list(self.free_ids)
        next_before=self.next_id
        reused_before=self.reused_page_ids
        tables_before=self._tables
        write_before=self.write_bytes
        try:
            cursor=0
            while cursor<n:
                absolute=start+cursor;lp,offset=divmod(absolute,p)
                count=min(p-offset,n-cursor)
                if lp not in self.pages:
                    self._allocate(lp,keys,values)
                    allocated.append(lp)
                _,k,v=self.pages[lp]
                if lp not in allocated and lp not in touched:
                    # 仅备份本次覆盖的尾部槽位；分配、拷贝或回收前异常均可原子回滚。
                    touched[lp]=(offset,count,
                        k[...,offset:offset+count,:].clone(),v[...,offset:offset+count,:].clone())
                k[...,offset:offset+count,:].copy_(keys[...,cursor:cursor+count,:])
                v[...,offset:offset+count,:].copy_(values[...,cursor:cursor+count,:])
                self.write_bytes+=(k.shape[1]*k.shape[-1]+v.shape[1]*v.shape[-1])*count*k.element_size()
                cursor+=count
        except Exception:
            for lp,(offset,count,old_k,old_v) in touched.items():
                _,k,v=self.pages[lp]
                k[...,offset:offset+count,:].copy_(old_k)
                v[...,offset:offset+count,:].copy_(old_v)
            for lp in allocated:
                self.pages.pop(lp,None)
            self.free_ids=free_before
            heapq.heapify(self.free_ids)
            self.next_id=next_before
            self.reused_page_ids=reused_before
            self._tables=tables_before
            self.write_bytes=write_before
            raise
        # 只有全部页写入成功才提交逻辑边界并回收旧页，避免半追加状态。
        self.logical_length=new_length
        if prefill:self.prompt_length=new_length
        if self.frozen:self._reclaim()
        # Prefill returns original complete tensors. Decode is consumed by the
        # paged adapter, which must not interpret these as a gathered history.
        return keys,values

    def pointer_tables(self):
        self._check_stream(next(iter(self.pages.values()))[1].device)
        n=(self.logical_length+self.geometry.page-1)//self.geometry.page
        if self._tables is None or self._tables[0].numel()!=n:
            device=next(iter(self.pages.values()))[1].device
            self._tables=tuple(torch.tensor([self.pages[i][j].data_ptr() if i in self.pages else 0 for i in range(n)],device=device,dtype=torch.int64) for j in (1,2))
        return self._tables

    def project_visible(self, dims, block_size):
        """Only low-dimensional index scratch; never a full-width KV gather."""
        from .paged_region_kernels import project_visible
        return project_visible(self,dims,block_size)

    def get_seq_length(self):return self.logical_length
    def get_max_cache_shape(self):return -1
    def get_mask_sizes(self,cache_position):
        q=cache_position if isinstance(cache_position,int) else cache_position.shape[0]
        return self.logical_length+q,0
    def reset(self):raise ValueError('create a fresh request cache')
    def reorder_cache(self,beam_idx):raise ValueError('beam/reorder unsupported')

    def report(self):
        spans=self.intervals(resident=True)
        logical_retained=sum(b-a for a,b in spans)
        occupied=sum(min(self.geometry.page,self.logical_length-lp*self.geometry.page) for lp in self.pages)
        kv=sum(k.numel()*k.element_size()+v.numel()*v.element_size() for _,k,v in self.pages.values())
        return dict(logical_length=self.logical_length,skip_near=self.skip_near,skip_far=self.skip_far,
            live_pages=len(self.pages),logical_to_physical={lp:page[0] for lp,page in self.pages.items()},
            resident_tokens=logical_retained,physical_token_slots=len(self.pages)*self.geometry.page,
            inactive_retained_slots=len(self.pages)*self.geometry.page-logical_retained,
            attention_tokens=sum(b-a for a,b in self.intervals()),kv_bytes=kv,
            device_page_table_bytes=0 if self._tables is None else sum(t.numel()*t.element_size() for t in self._tables),
            metadata_scope='GPU pointer tables exact; Python dict/list overhead excluded',
            reclaimed_pages=self.reclaimed_pages,reused_page_ids=self.reused_page_ids,
            aging_copy_bytes=self.aging_copy_bytes,kv_write_bytes=self.write_bytes,
            allocator_scope='active tensor allocations; caching allocator reserved memory is separate')


class PagedRegionCache(Cache):
    tli_regions=True
    paged_regions=True

    def __init__(self,n_layers,geometry=RegionGeometry()):
        super().__init__(layers=[PagedRegionLayer(geometry) for _ in range(n_layers)])
    def update(self,key_states,value_states,layer_idx,cache_kwargs=None):
        if self.layers[layer_idx].frozen:
            # 通用 HF attention 会把返回值当作完整历史；只允许直接分页消费者写入。
            raise ValueError('frozen paged decode requires the patched direct-page consumer')
        return super().update(key_states,value_states,layer_idx,cache_kwargs)
    def update_for_paged_decode(self,key_states,value_states,layer_idx,cache_kwargs=None):
        if not self.layers[layer_idx].frozen:
            raise ValueError('freeze gates before direct-page decode')
        return super().update(key_states,value_states,layer_idx,cache_kwargs)
    def freeze(self,decisions):
        if set(decisions)!=set(range(len(self.layers))):raise ValueError('all layers need frozen decisions')
        for li,layer in enumerate(self.layers):layer.freeze(**decisions[li])
    def report(self):
        layers=[x.report() for x in self.layers]
        return dict(layers=layers,kv_bytes=sum(x['kv_bytes'] for x in layers),
                    device_page_table_bytes=sum(x['device_page_table_bytes'] for x in layers))

