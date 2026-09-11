"""Full-resolution diagnostics and exact group-range distributions; no model reload."""
from __future__ import annotations
import argparse, collections, csv, json
from pathlib import Path
import numpy as np
import torch
from nar import experiment as base
from .capture import write

def ratio(a,b): return a/b if b>0 else None

def distribution(x,prefix):
    x=x.double().flatten()
    q=torch.quantile(x,torch.tensor([.5,.95],dtype=torch.float64,device=x.device))
    return {prefix+'_mean':float(x.mean()),prefix+'_median':float(q[0]),prefix+'_p95':float(q[1]),prefix+'_max':float(x.max())}

def measure(y):
    assert y.dtype==torch.float32 and torch.isfinite(y).all()
    group=base.group_view(y,128); mu=group.mean(-1,keepdim=True); residual=group-mu
    ranges=group.amax(-1)-group.amin(-1)
    erange=residual.amax(-1)-residual.amin(-1)
    range_error=float(((ranges-erange).abs()/ranges.abs().clamp_min(1)).max())
    assert range_error<=2e-6,range_error
    qdq,scale,offset,codes=base.dynamic_asym_int4(y,128)
    assert torch.isfinite(qdq).all() and scale.dtype==offset.dtype==torch.float16
    yd=y.double();energy=float(yd.square().sum());common=float(mu.double().square().sum()*128)
    residual_energy=float(residual.double().square().sum());error=float((qdq.double()-yd).square().sum())
    conservation=abs((common+residual_energy)/energy-1) if energy else 0
    assert conservation<=2e-6,conservation
    row={'absmax':float(y.abs().max()),'rms':(energy/y.numel())**.5,
         'elements':y.numel(),'energy':energy,'common_energy':common,'residual_energy':residual_energy,
         'error_energy':error,'rho':ratio(common,energy),'f':ratio(residual_energy,energy),
         'nmse':ratio(error,energy),'range_identity_relative_error':range_error,'energy_decomposition_error':conservation,
         'degenerate_groups':int(((ranges/15).half()==0).sum()),'nonfinite':0,
         **distribution(ranges,'range'),**distribution(scale,'step'),**distribution(offset,'offset')}
    row['absmax_over_rms']=ratio(row['absmax'],row['rms'])
    for label,sl in [('token0',slice(0,1)),('remaining',slice(1,None))]:
        part=yd[sl];pe=float(part.square().sum());err=float((qdq[sl].double()-part).square().sum())
        row.update({label+'_absmax':float(part.abs().max()),label+'_energy':pe,
          label+'_energy_fraction':ratio(pe,energy),label+'_error_energy':err,label+'_nmse':ratio(err,pe),
          **distribution(ranges[sl],label+'_range')})
    return row,residual.reshape_as(y),ranges,scale

def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def run(root,device='cpu'):
    root=Path(root);torch.set_num_threads(4)
    inventory=json.loads((root/'activation_inventory.json').read_text())
    rows=[]; groups=collections.defaultdict(list); exact=collections.defaultdict(dict); checks=[]
    for item in inventory:
        y=torch.load(root/item['file'],map_location=device,weights_only=True)
        row,residual,ranges,scale=measure(y)
        row={**{k:item[k] for k in ('mode','method','sample','layer','site')},**row};rows.append(row)
        key=(item['mode'],item['method'],item['layer'],item['site']);groups[key].append((row,ranges.cpu().numpy(),scale.cpu().numpy()))
        del y,residual,ranges,scale
    summaries=[]
    for key,items in groups.items():
        rr=[a[0] for a in items]; row=dict(zip(('mode','method','layer','site'),key));row['samples']=len(rr)
        for name in ('energy','common_energy','residual_energy','error_energy','elements','token0_energy','remaining_energy','token0_error_energy','remaining_error_energy','degenerate_groups'):
            row[name]=sum(r[name] for r in rr)
        row['absmax']=max(r['absmax'] for r in rr);row['rms']=(row['energy']/row['elements'])**.5
        row['absmax_over_rms']=ratio(row['absmax'],row['rms'])
        for name,num in [('rho','common_energy'),('f','residual_energy'),('nmse','error_energy')]:
            row[name]=ratio(row[num],row['energy'])
            vals=[r[name] for r in rr if r[name] is not None]
            for stat,fn in [('sample_mean',np.mean),('sample_sd',lambda a:np.std(a,ddof=1) if len(a)>1 else 0),('sample_min',np.min),('sample_max',np.max)]:
                row[name+'_'+stat]=float(fn(vals)) if vals else None
        for part in ('token0','remaining'):
            row[part+'_absmax']=max(r[part+'_absmax'] for r in rr)
            row[part+'_energy_fraction']=ratio(row[part+'_energy'],row['energy'])
            row[part+'_nmse']=ratio(row[part+'_error_energy'],row[part+'_energy'])
        ranges=np.concatenate([a[1].reshape(-1) for a in items]);steps=np.concatenate([a[2].reshape(-1) for a in items])
        for prefix,values in [('range',ranges),('step',steps)]:row.update(distribution(torch.from_numpy(values),prefix))
        for part in ('token0','remaining'):
            values=np.concatenate([(a[1][:1] if part=='token0' else a[1][1:]).reshape(-1) for a in items])
            row.update(distribution(torch.from_numpy(values),part+'_range'))
        values, multiplicity = np.unique(ranges, return_counts=True)
        mode,method,layer,site=key
        exact[(mode,site)][f'{method}__{layer}__values']=values
        exact[(mode,site)][f'{method}__{layer}__counts']=multiplicity.astype(np.uint32)
        summaries.append(row)
    csv_write(root/'metrics_per_sample.csv',rows);csv_write(root/'metrics_summary.csv',summaries)
    (root/'exact_ecdf').mkdir(exist_ok=True)
    for (mode,site),arrays in exact.items():
        np.savez_compressed(root/'exact_ecdf'/f'{mode}__{site}.npz',**arrays)
    paired=[i for i in inventory if i['mode']=='paired_local']
    identity=collections.defaultdict(set)
    for i in paired:identity[(i['sample'],i['layer'],i['site'])].add(i['canonical_tensor_sha256'])
    assert all(len(v)==1 and None not in v for v in identity.values())
    assert len(inventory)==448 and len(rows)==448 and all(r['samples']==8 for r in summaries)
    report=json.loads((root/'validation_report.json').read_text())
    report['full_resolution_metrics']={'passed':True,'rows':len(rows),'summary_rows':len(summaries),
        'shared_canonical_input_groups':len(identity),'max_range_identity_relative_error':max(r['range_identity_relative_error'] for r in rows),
        'max_energy_decomposition_error':max(r['energy_decomposition_error'] for r in rows),
        'nonfinite':sum(r['nonfinite'] for r in rows),'zero_energy_policy':'ratios null; JSON/CSV preserve missing values',
        'aggregate_ratio':'sum numerators / sum denominators; sample mean/sd/min/max also reported',
        'unrotated_end_to_end':'not a W4A4KV4 checkpoint; paired reference is not duplicated or labelled quantized'}
    write(root/'validation_report.json',report)
    print('METRICS COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--device',default='cpu');a=p.parse_args();run(a.root,a.device)
