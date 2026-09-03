#!/bin/bash
#SBATCH --job-name=e14-quarot-anchor
#SBATCH --gpus=rtx5090:1
#SBATCH --time=3-00:00:00
#SBATCH --output=runs/e14-quarot-anchor-%j.out
#SBATCH --error=runs/e14-quarot-anchor-%j.err
#SBATCH --requeue
set -euo pipefail
umask 0007
: "${NAR_WORKDIR:?Set NAR_WORKDIR}"
quarot_dir="${QUAROT_CODE_DIR:-/home/yanlongc/QuaRot-upstream}"
python_bin="$NAR_WORKDIR/quarot-env/bin/python"
anchor_root="${NAR_E14_ANCHOR_ROOT:-$NAR_WORKDIR/artifacts/e14/quarot_release_anchor}"
checkpoint="$anchor_root/quarot_release_llama2_7b.pt"
mkdir -p "$(dirname "$checkpoint")"
qmodel_args=(--save_qmodel_path "$checkpoint")
if [[ -s "$checkpoint" ]]; then
    qmodel_args=(--load_qmodel_path "$checkpoint")
    echo "Resuming anchor evaluation from $checkpoint"
fi
export HF_HOME="$anchor_root/huggingface"
export HUGGINGFACE_HUB_CACHE="$anchor_root/huggingface/hub"
export HF_DATASETS_CACHE="$anchor_root/datasets"
export XDG_CACHE_HOME="$anchor_root/xdg"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 WANDB_DISABLED=true
cd "$quarot_dir/fake_quant"
echo "QuaRot upstream base: 5008669b08c1f11f9b64d52d16fddd47ca754c5a"
echo "QuaRot compatibility commit: $(git -C "$quarot_dir" rev-parse HEAD)"
"$python_bin" -c "import torch,transformers,datasets; print('torch',torch.__version__,'transformers',transformers.__version__,'datasets',datasets.__version__)"
exec "$python_bin" main.py \
    --model NousResearch/llama-2-7b-hf \
    --rotate \
    --a_bits 4 \
    --v_bits 4 \
    --k_bits 4 \
    --w_bits 4 \
    --w_clip \
    "${qmodel_args[@]}" \
    --save_name e14_quarot_release_anchor
