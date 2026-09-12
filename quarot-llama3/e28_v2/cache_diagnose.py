"""Isolate warm-up/nondeterminism from metadata changes without timing."""
import torch
from .common import *
from .cache_adapter import make_cache,clear,snapshot

def main():
 a=arguments(__doc__).parse_args();initialize()
 with torch.inference_mode():
  m=build('3b','hadamard');tokens=torch.load(WORK/'cache/tokenized/llama32_3b-wikitext2-test-full-l2048.pt',weights_only=True).reshape(-1)
  saved=[];prefills=[];order=[True,True,False,True]
  for legacy in order:
   cache=make_cache(m,1,256,legacy=legacy,page_size=64);clear(cache)
   m(tokens[:128].cuda().int().view(1,-1),past_key_values=cache)
   prefills.append((cache.pages.cpu().clone(),cache.scales.cpu().clone(),snapshot(cache)))
   m(tokens[128:129].cuda().int().view(1,1),past_key_values=cache)
   saved.append((cache.pages.cpu().clone(),cache.scales.cpu().clone(),snapshot(cache)))
  pairs=[]
  for u,v in [(0,1),(1,2),(2,3),(0,3)]:
   row={'runs':[u,v],'legacy':[order[u],order[v]],'prefill':[prefills[u][2],prefills[v][2]],'decode':[saved[u][2],saved[v][2]],'differences':[]}
   for i in [0,1]:
    aa=saved[u][i];bb=saved[v][i];inds=(aa!=bb).nonzero()
    row['differences'].append({'tensor':i,'count':int((aa!=bb).sum()),'prefill_count':int((prefills[u][i]!=prefills[v][i]).sum()),'first_indices':inds[:20].tolist(),'first_values':[[float(aa[tuple(ind)]),float(bb[tuple(ind)])] for ind in inds[:20]]})
   pairs.append(row);print('CACHE PAIR',u,v,[d['count'] for d in row['differences']],flush=True)
  write(a.run/'correctness/cache_diagnose.json',{'order_legacy':order,'pairs':pairs})
if __name__=='__main__':main()
