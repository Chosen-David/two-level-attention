"""Read-only completed-shard integrity inventory, not scientific acceptance."""
import pathlib,json,hashlib,time,collections,sys
r=pathlib.Path(__file__).resolve().parents[2]
label=sys.argv[1];assert label.isidentifier()
ref=json.loads((r/'manifest_rank0.json').read_text());proto=ref['protocol'];ident=ref['identity']
expected=[(ds,row) for ds in proto['datasets'] for row in proto['selected_rows'][ds]]
errors=[];files=[];ranks=[];coverage=collections.Counter()
def read(p):
 b=p.read_bytes();files.append(dict(path=str(p.relative_to(r)),sha256=hashlib.sha256(b).hexdigest()));return json.loads(b)
for done in sorted(r.glob('worker_done_rank*.json')):
 rank=int(done.stem.split('rank')[1]);d=read(done);m=read(r/f'manifest_rank{rank}.json')
 if d['identity']!=ident or m['identity']!=ident or m['protocol']!=proto:errors.append(f'identity/protocol rank {rank}')
 pairs=expected[rank::proto['world_size']]
 for ds,row in pairs:
  p=r/'requests'/ds/f'{row:04d}.json'
  if not p.exists():errors.append(f'missing {ds}:{row}');continue
  q=read(p)
  if (q['dataset'],q['row_index'])!=(ds,row) or q['identity']!=ident or q['status']!='complete' or set(q['arms'])!={'full','far_0.05','dual_0.05'}:errors.append(f'invalid {ds}:{row}')
  coverage[ds]+=1
 ranks.append(dict(rank=rank,expected_pairs=len(pairs)))
result=dict(time=time.time(),scope='completed-shard integrity inventory only; independent scientific verification pending',identity=ident,done_ranks=ranks,coverage=dict(coverage),pairs=sum(coverage.values()),errors=errors,files=files)
p=pathlib.Path(__file__).parent/f'integrity_{label}.json';p.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
