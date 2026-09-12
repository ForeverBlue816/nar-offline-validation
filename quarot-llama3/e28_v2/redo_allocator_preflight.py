"""One-time, explicit replacement of the two pre-fix records; preserve originals."""
from .common import *

def main():
 a=arguments(__doc__).parse_args();out=a.run
 previous=out/'raw_runs/allocator_preflight_attempt';previous.mkdir(exist_ok=True)
 keys=[f'3b_fp16_eager_sequence_{phase}_s1' for phase in ['prefill1','prefill16']]
 source=command(['git','-C',str(ROOT),'show','bff19d9:quarot-llama3/e28_v2/benchmark.py'])+'\n'
 original=[]
 for key in keys:
  path=out/'raw_runs'/f'{key}.json'
  record=read(path)
  assert record['status'] in ('PASS','FAIL','INVALID'), 'Do not archive a running process'
  expected=record['environment']['execution_python_sha256']['quarot-llama3/e28_v2/benchmark.py']
  assert hashlib.sha256(source.encode()).hexdigest()==expected
  original.append({'key':key,'sha256':sha(path),'status':record['status'],'execution_source_sha256':expected})
  path.rename(previous/path.name)
 (previous/'benchmark.py').write_text(source)
 write(out/'allocator_preflight_revision.json',{'timestamp':now(),'originals':original,'reason':'Explicitly clear allocator-only correctness-preflight residue before10 complete inference warmups; both affected processes are repeated in full, no selection by speed.','after_commit':command(['git','-C',str(ROOT),'rev-parse','HEAD'])})
 for phase in ['prefill1','prefill16']:
  cmd=[sys.executable,'-m','e28_v2.benchmark','--run',str(out),'--model','3b','--method','fp16','--phase',phase,'--mode','eager_sequence','--session','1']
  with (out/'commands.jsonl').open('a') as f:f.write(json.dumps({'timestamp':now(),'argv':cmd,'reason':'allocator preflight correction'})+'\n')
  subprocess.run(cmd,check=True)
if __name__=='__main__':main()
