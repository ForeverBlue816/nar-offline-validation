#!/usr/bin/env python3
"""Append offline E1c DuQuant measurements; never load or rerun a model."""
import argparse,csv,hashlib,io,json,sys,os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

def sha(data): return hashlib.sha256(data).hexdigest()
def frozen_rows(ext,wide,capture,site,layer,stride,device):
    # Concurrent exact row reads avoid serial network-filesystem page faults.
    shape=ext._site_shape(capture,site);path=ext._site_path(wide,site,layer)
    ext._validate_dump_file(path,shape);row_bytes=shape[-1]*2
    fd=os.open(path,os.O_RDONLY)
    def sequence(index):
        values=[os.pread(fd,row_bytes,(index*shape[1]+position)*row_bytes) for position in range(0,shape[1],stride)]
        assert all(len(value)==row_bytes for value in values)
        return b''.join(values)
    try:
        with ThreadPoolExecutor(max_workers=32) as pool: bits=b''.join(pool.map(sequence,range(shape[0])))
    finally: os.close(fd)
    array=np.frombuffer(bits,dtype=np.uint16).reshape(-1,shape[-1])
    return ext._bits_to_tensor(array,device)

def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--code-root',type=Path,required=True);p.add_argument('--model',required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    sys.path.insert(0,str(a.code_root))
    from nar import experiment as base,extended_experiment as ext,e11_fair_baselines as e11
    result=a.repo/'results'/a.model;source=result/'e1c_per_layer.csv';done=result/'E1C_DONE.json'
    if not source.exists() or not done.exists(): raise FileNotFoundError(f'Frozen E1c source missing for {a.model}')
    old=source.read_bytes();original=pd.read_csv(source);meta=json.loads(done.read_text())
    if original.method.eq('duquant').any(): raise RuntimeError('DuQuant rows already present; refusing duplicate append')
    wide=a.assets/'activations'/a.model/meta['settings']['wide_tag'];capture=json.loads((wide/'DONE.json').read_text())
    device=torch.device(a.device)
    if a.device=='cuda': assert torch.cuda.is_available()
    torch.set_num_threads(a.threads);torch.set_float32_matmul_precision('highest')
    stats_path=e11.calibration_dir(a.assets,a.model)/'channel_stats.pt'
    stats=torch.load(stats_path,map_location='cpu',weights_only=True)
    seed=20260902;group=128;stride=int(meta['settings']['evaluation_token_stride']);rows=[];checks=[]
    partial=result/'e1c_duquant_addendum.partial.csv';audit_path=result/'e1c_duquant_sanity.csv'
    if partial.exists() and audit_path.exists():
        rows=pd.read_csv(partial).to_dict('records');checks=pd.read_csv(audit_path).to_dict('records')
        assert len(rows)==len(checks)
    completed={(r['site'],int(r['layer'])) for r in rows}
    assert int(meta['seed'])==seed and int(meta['settings']['group_size'])==group
    for site_index,(site,du_site) in [(1,('down_input','down')),(0,('q_input','qkv'))]:
        for layer in range(int(capture['num_layers'])):
            if (site,layer) in completed: continue
            n=int(capture['hidden_size'] if site=='q_input' else capture['intermediate_size'])
            x=frozen_rows(ext,wide,capture,site,layer,stride,device)
            reference=original[original.site.eq(site)&original.layer.eq(layer)].set_index('method')
            assert x.shape[0]==int(reference.loc['hadamard_full','evaluation_tokens'])
            scores=stats['activation_absmax'][e11._key(du_site,layer)]
            construction_seed=seed+100000*(du_site=='down')+1000*layer
            permutation,blocks=e11._duquant_blocks(scores,group,construction_seed,device)
            values=x[:,permutation]
            rotated=torch.einsum('ngi,gij->ngj',values.reshape(-1,blocks.shape[0],group),blocks).reshape_as(x)
            value,nmse,_=base.quant_metrics(rotated,group)
            generator=torch.Generator(device='cpu').manual_seed(seed+1000*layer+10*site_index+group)
            signs=torch.randint(0,2,(n,),generator=generator,dtype=torch.int64).float().mul_(2).sub_(1).to(device)
            had=ext._full_hadamard_rows(x,signs);hr,hn,_=base.quant_metrics(had,group)
            np.testing.assert_allclose([hr,hn],[reference.loc['hadamard_full','mean_group_range'],reference.loc['hadamard_full','relative_quantization_error_nmse']],rtol=2e-5,atol=1e-8)
            had_range=float(reference.loc['hadamard_full','mean_group_range']);had_nmse=float(reference.loc['hadamard_full','relative_quantization_error_nmse'])
            prism=float(reference.loc['nar_kmax','mean_group_range'])
            rows.append(dict(model=a.model,site=site,layer=layer,n=n,b=group,method='duquant',mean_group_range=value,
                relative_quantization_error_nmse=nmse,range_reduction_vs_hadamard=(had_range-value)/had_range,
                nmse_delta_vs_hadamard=nmse-had_nmse,evaluation_tokens=int(x.shape[0])))
            groups=rotated.reshape(x.shape[0],-1,group)
            capture_fraction=float((groups.sum(-1).square()/group).double().sum()/rotated.square().double().sum())
            tol=2e-5*max(had_range,prism)
            inside= min(prism,had_range)-tol<=value<=max(prism,had_range)+tol
            checks.append(dict(model=a.model,site=site,layer=layer,duquant_range=value,hadamard_range=had_range,
                prismquant_range=prism,duquant_over_hadamard=value/had_range,null_space_capture=capture_fraction,
                inside_paired_bracket=inside,construction_seed=construction_seed,
                permutation_sha256=sha(permutation.cpu().numpy().tobytes()),
                sampled_bf16_sha256=sha(x.to(torch.bfloat16).view(torch.uint16).cpu().numpy().tobytes()),
                hadamard_replay_range=hr,hadamard_replay_nmse=hn,
                frozen_hadamard_source_csv_line=int(original.index[(original.site.eq(site))&(original.layer.eq(layer))&(original.method.eq('hadamard_full'))][0])+2))
            pd.DataFrame(rows).to_csv(result/'e1c_duquant_addendum.partial.csv',index=False)
            pd.DataFrame(checks).to_csv(result/'e1c_duquant_sanity.csv',index=False)
            print(json.dumps(checks[-1]),flush=True)
            del x,values,rotated,had,groups,blocks,permutation
    fieldnames=list(original.columns);buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=fieldnames,lineterminator='\r\n');writer.writerows(rows)
    assert source.read_bytes()==old,'Source changed concurrently'
    suffix=buffer.getvalue().encode();assert old.endswith(b'\n')
    source.write_bytes(old+suffix);assert source.read_bytes()[:len(old)]==old
    audit=pd.DataFrame(checks);violations=audit[~audit.inside_paired_bracket]
    addendum={'timestamp':datetime.now(timezone.utc).isoformat(),'method':'duquant','seed':seed,'group_size':group,
        'construction':'E16 uses e11._duquant_blocks: descending absmax zigzag permutation, tie by channel index; seeded block QR complement; peak channel maps to DC',
        'construction_seed_formula':'20260902 + 100000*(site == down) + 1000*layer',
        'score_source':str(stats_path),'score_source_sha256':sha(stats_path.read_bytes()),
        'evaluation_token_stride':stride,'evaluation_tokens_per_layer_site':int(rows[0]['evaluation_tokens']),
        'sites':['q_input','down_input'],'layers':int(capture['num_layers']),'appended_rows':len(rows),
        'original_csv_bytes':len(old),'original_csv_sha256':sha(old),'original_rows_unchanged':True,
        'no_model_load_or_rerun':True,'quantizer':'experiment.quant_metrics; fp16 scale and offset dynamic asymmetric INT4',
        'code_source_sha256':{name:sha((a.code_root/'nar'/name).read_bytes()) for name in ['e11_fair_baselines.py','extended_experiment.py','experiment.py']},
        'device':a.device,'hardware':base.hardware_info(),'sanity_source':'e1c_duquant_sanity.csv','bracket_violations':len(violations),
        'plotting_gate':'FAIL' if len(violations) else 'PASS','bracket_tolerance_relative':2e-5}
    meta['duquant_addendum']=addendum;base.atomic_json(done,meta)
    print(json.dumps(addendum,indent=2),flush=True)
if __name__=='__main__':main()
