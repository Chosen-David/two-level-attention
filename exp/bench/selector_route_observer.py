"""Count real optional-selector kernel submissions without changing GPU work."""
class CountedKernel:
    def __init__(self,kernel,counts,capturing,required=None):
        self.kernel,self.counts,self.capturing,self.required=kernel,counts,capturing,required or {}
    def __getitem__(self,grid):
        launch=self.kernel[grid]
        def call(*args,**kwargs):
            if any(kwargs.get(key)!=value for key,value in self.required.items()):
                raise RuntimeError('selector kernel specialization does not match measured route')
            phase='capture' if self.capturing() else 'eager'
            self.counts[phase]+=1
            return launch(*args,**kwargs)
        return call


def install(packed,tiled,capturing):
    counters={}
    def wrap(owner,name,label,required=None):
        counts={'capture':0,'eager':0};counters[label]=counts
        setattr(owner,name,CountedKernel(getattr(owner,name),counts,capturing,required))
    if packed:
        from sparse_attn.ops import sparse_normalization as norm
        wrap(norm,'_prepare_keys','packed_prepare',{'PACKED':True})
        wrap(norm,'_offsets_and_keys','packed_offsets',{'PACKED':True})
    if tiled:
        from sparse_attn.ops import tiled_compact as compact
        wrap(compact,'_count','tiled_count')
        wrap(compact,'_partition','tiled_partition')
    return counters


def checked_delta(counters,before,expected_capture):
    result={name:{phase:counts[phase]-before[name][phase] for phase in counts}
            for name,counts in counters.items()}
    if any(counts['capture']!=expected_capture for counts in result.values()):
        raise RuntimeError('optional selector kernel missing from actual model capture')
    return result
