"""Submit only the remaining fixed E30 jobs when the user's queue has capacity.

No job is cancelled, requeued, or automatically retried by this controller.
"""
from pathlib import Path
import datetime,fcntl,json,subprocess,time
ROOT=Path(__file__).resolve().parent.parent
STATE=ROOT/'experiments/e30_queue_state.json'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(state):
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(state,indent=2)+'\n');temp.replace(STATE)
def queue():
    raw=subprocess.check_output(['squeue','-h','-u','yanlongc','-o','%i|%j|%T'],text=True)
    return [line.split('|') for line in raw.splitlines() if line]
def main():
    # A single controller owns this fixed three-job schedule.
    lock=(ROOT/'runs/e30_queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=json.loads(STATE.read_text()) if STATE.exists() else {'started_utc':utc(),'jobs':{}}
    tasks=[('seed2','nar-e30-prep-s2',['slurm_reviewer_e30_prepare.sh','2'],'e30_preparation_seed2.json'),('seed0','nar-e30-prep-s0',['slurm_reviewer_e30_prepare.sh','0'],'e30_preparation_seed0.json'),('evaluation','nar-e30-final',['slurm_reviewer_eval.sh','qwen3_4b_base','30'],'e30_DONE.json')]
    while True:
        live=queue();state['updated_utc']=utc();state['queue_job_count']=len(live)
        for key,name,command,done in tasks:
            done_path=ROOT/'results/qwen3_4b_base'/done
            if done_path.exists() and json.loads(done_path.read_text())['status']=='COMPLETE':
                state['jobs'].setdefault(key,{})['status']='COMPLETE';continue
            if key in state['jobs']:
                job=state['jobs'][key]
                if 'job_id' in job and not any(int(r[0])==job['job_id'] for r in live):
                    status=subprocess.check_output(['sacct','-X','-n','-P','-j',str(job['job_id']),'--format=State'],text=True).strip().splitlines()
                    if status and any(s.rstrip('|') not in ['RUNNING','PENDING','COMPLETING','COMPLETED'] for s in status):
                        job['status']='NEEDS_INSPECTION';job['slurm_status']=status;save(state);raise RuntimeError(job)
                continue
            existing=[r for r in live if r[1]==name]
            if existing:
                assert len(existing)==1;state['jobs'][key]={'job_id':int(existing[0][0]),'status':'SUBMITTED','recovered_by_unique_name':True};save(state);continue
            if len(live)>=5:continue
            if key=='evaluation' and not all((ROOT/'results/qwen3_4b_base'/f'e30_preparation_seed{s}.json').exists() for s in [0,1,2]):continue
            if key=='evaluation' and not (ROOT/'results/qwen3_4b_base/e29_DONE.json').exists():continue
            result=subprocess.run(['sbatch','--parsable',f'--job-name={name}',*command],cwd=ROOT,capture_output=True,text=True)
            if result.returncode:
                state['last_submission_error']=result.stderr;save(state)
                if 'QOSMaxSubmitJobPerUserLimit' in result.stderr:break
                raise RuntimeError(result.stderr)
            jid=int(result.stdout.strip().split(';')[0]);state['jobs'][key]={'job_id':jid,'status':'SUBMITTED','submitted_utc':utc(),'command':command};save(state)
            print(utc(),key,jid,flush=True);live.append([str(jid),name,'SUBMITTED'])
        save(state)
        if all(state['jobs'].get(key,{}).get('status')=='COMPLETE' for key,*_ in tasks):
            state['status']='COMPLETE';save(state);return
        time.sleep(30)
if __name__=='__main__':main()
