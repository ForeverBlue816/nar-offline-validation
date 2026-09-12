#!/bin/bash
#SBATCH --job-name=nar-e28-v2-graph
#SBATCH --nodelist=gpu-a40-1
#SBATCH --gpus=a40:1
#SBATCH --qos=rose
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
cd "${E28_CODE_ROOT:?Set E28_CODE_ROOT}"
: "${E28_RUN:?Set E28_RUN}"
export E28_REQUIRED_GPU_UUID=GPU-97605b97-fbc5-631d-fd5d-348d2f30ba22
/projects/nar/nar-validation/e28-env/bin/python - <<'PY'
import datetime,json,os,pathlib,subprocess
out=pathlib.Path(os.environ['E28_RUN'])
if not (out/'pipeline_finished.json').exists():
    raise RuntimeError('Primary pipeline has not finished; do not start the graph timing panel')
uuids=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).splitlines()
expected=os.environ['E28_REQUIRED_GPU_UUID']
if uuids != [expected]:
    raise RuntimeError(f'Require the original single A40 UUID, observed {uuids}')
record={'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job_id':os.environ['SLURM_JOB_ID'],'expected_gpu_uuid':expected,'observed_gpu_uuids':uuids,'affinity':sorted(os.sched_getaffinity(0)),'cgroup':pathlib.Path('/proc/self/cgroup').read_text(),'scope':'The full private-backend eager/graph panel runs in this allocation; no ratios cross into original-binary core records.'}
(out/'stream_allocation.json').write_text(json.dumps(record,indent=2)+'\n')
PY
bash quarot-llama3/e28_v2/run.sh stream_pipeline --run "$E28_RUN"
