#!/bin/bash
#SBATCH --job-name=nar-e14-3b-proto
#SBATCH --gpus=pro6000:1
#SBATCH --constraint=highmem
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/e14-3b-proto-%j.out
#SBATCH --error=runs/e14-3b-proto-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
code_dir="${NAR_CODE_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}"
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
python_bin="$NAR_WORKDIR/venv/bin/python"
export PYTHONPATH="${E13_SITEPACKAGES:-$HOME/.e13_packages}${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$NAR_WORKDIR/cache/huggingface" HF_DATASETS_CACHE="$NAR_WORKDIR/cache/datasets"
export XDG_CACHE_HOME="$NAR_WORKDIR/cache/xdg" TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR="$NAR_WORKDIR/tmp"
mkdir -p "$TMPDIR" "$code_dir/runs"

model=llama32_3b
artifact_root="${NAR_E14_ARTIFACT_ROOT:-$NAR_WORKDIR/artifacts/e14}"
protocols="${NAR_PROTOCOLS:-g128_asym g128 act_order}"
run() { "$python_bin" "$code_dir/nar/e14_w4a4kv4.py" --workdir "$NAR_WORKDIR" \
        --artifact-root "$artifact_root" --seed 0 "$@"; }

# Three attempts to move the W4A4KV4 rows on the one checkpoint the published
# tables share with this repository, each following from the E19 diagnosis
# that the k-dependence lives in the weight quantizer.  Every step skips its
# own completed artifact, so a requeue resumes.

# (3) Does the KV quantizer explain NAR k=8's BoolQ loss?  BoolQ has the
# longest prompts of the eight tasks and so is the one where most requests
# actually quantize the K cache.  The extra set is small, so this is minutes.
echo "########## phase 0: extra tasks with the KV cache left in bf16 ##########"
for row in nar_k8_asym_g128 hadamard_asym_g128 nar_kmax_asym_g128; do
    echo "===== kv off: $row (extra) ====="
    run evaluate --model "$model" --row "$row" --metrics zero_shot --task-set extra --kv off
done

# (1)(2) Alternative GPTQ protocols, perplexity first so the ordering lands
# early. g128_asym is the mechanism-matched one; g128 is what E19 ran on Qwen3;
# act_order is bit-neutral and the protocol E19 selected there.
echo "########## phase 1: GPTQ protocols, perplexity ##########"
for protocol in $protocols; do
    for rotation in hadamard nar_k8 nar_kmax; do
        echo "===== $protocol: GPTQ $rotation ====="
        run gptq --model "$model" --rotation "$rotation" --protocol "$protocol"
        echo "===== $protocol: perplexity ${rotation}_asym_g128 ====="
        run evaluate --model "$model" --row "${rotation}_asym_g128" --metrics ppl --protocol "$protocol"
    done
done

echo "########## phase 2: GPTQ protocols, zero-shot ##########"
for protocol in $protocols; do
    for rotation in nar_kmax nar_k8 hadamard; do
        for set in frozen extra; do
            echo "===== $protocol: zero-shot ${rotation}_asym_g128 ($set) ====="
            run evaluate --model "$model" --row "${rotation}_asym_g128" \
                --metrics zero_shot --task-set "$set" --protocol "$protocol"
        done
    done
done
echo "===== done ====="
