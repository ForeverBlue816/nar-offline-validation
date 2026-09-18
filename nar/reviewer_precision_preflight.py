import sys,torch,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from nar import reviewer_ablations as a
a.setup()
from pathlib import Path
r=a.Rotation('llama32_3b',0)
rows=[]
for site,layer,n in a.keys(r.m):
 d=r.data[site,layer];v=d['vectors'].T.float();sign=r.signs[site,layer]
 target=torch.zeros_like(v)
 for i in range(d['k']):target[i,i*128:(i+1)*128]=128**-.5
 outputs={}
 for mode in ['stored_fp32','stored_fp64_matmul','recovered_fp64']:
  x=v
  if mode=='stored_fp32':z=x-(x@d['w'])@d['y'].T
  else:
   if mode=='stored_fp64_matmul':w,y=d['w'].double(),d['y'].double()
   else:_,w,y=a.wy64(d['vectors'])
   xx=x.double();z=(xx-(xx@w)@y.T).float()
  perm=torch.empty_like(z);perm[:,d['target']]=z[:,d['source']]
  rotated=a.act.ext._fast_walsh_hadamard((perm*sign).reshape(-1,n//128,128)).reshape(-1,n)
  outputs[mode]=float(torch.minimum((rotated-target).norm(dim=1),(rotated+target).norm(dim=1)).max())
 rows.append(dict(site=site,layer=layer,**outputs))
print(json.dumps(rows,indent=2))
a.savej(a.REPO/'experiments/e29_precision_preflight.json',{'status':'NUMERICAL DIAGNOSTIC ONLY','PPL_rows_used':False,'rows':rows,'maxima':{m:max(r[m] for r in rows) for m in outputs}})
