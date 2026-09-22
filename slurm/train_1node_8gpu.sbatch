#!/bin/bash
#SBATCH --job-name=astar-0921
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=gamma-huge-long
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --signal=USR1@600

set -euo pipefail

# Slurm executes a spooled copy of this script, so BASH_SOURCE does not point
# back into the repository inside a batch job.
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

: "${ASTAR_DATA_ROOT:?Set ASTAR_DATA_ROOT to the prepared dataset directory}"
if [[ -x "${REPO_ROOT}/.venv/bin/astar-train" ]]; then
  TRAIN_COMMAND=("${REPO_ROOT}/.venv/bin/astar-train")
else
  UV_BIN="${UV_BIN:-uv}"
  command -v "${UV_BIN}" >/dev/null 2>&1 || {
    echo "Run 'uv sync' first or set UV_BIN to the uv executable." >&2
    exit 1
  }
  TRAIN_COMMAND=("${UV_BIN}" run astar-train)
fi

# NCCL requires enough pinned host memory for multi-GPU collectives.
if ! ulimit -Sl unlimited; then
  echo "Cannot raise the soft memlock limit required by NCCL." >&2
  exit 1
fi

export MPLBACKEND=Agg
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/outputs/wandb}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/astar_jax_${SLURM_JOB_ID}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/astar_mpl_${SLURM_JOB_ID}}"

WANDB_ARGS=(--no-wandb-enabled)
if [[ "${WANDB_ENABLED:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb-enabled --wandb-entity "${WANDB_ENTITY:-yohan-UMD}")
fi

# One JAX process with eight-way FSDP. TrainConfig forms checkpoint paths as
# checkpoint_base_dir/config_name/exp_name, so these values resolve exactly to
# ${REPO_ROOT}/checkpoints/astar_0921 with the default experiment name.
srun --kill-on-bad-exit=1 --gpu-bind=none \
  "${TRAIN_COMMAND[@]}" \
  --config-name checkpoints \
  --exp-name "${EXP_NAME:-astar_0921}" \
  --project-name "${WANDB_PROJECT:-ASTAR}" \
  --data-root "${ASTAR_DATA_ROOT}" \
  --train-split-ids-path "${REPO_ROOT}/configs/splits/train_clip_ids.txt" \
  --eval-split-ids-path "${REPO_ROOT}/configs/splits/eval_clip_ids.txt" \
  --pretrained-params "${REPO_ROOT}/checkpoints/pi05_base/params" \
  --checkpoint-base-dir "${REPO_ROOT}" \
  --batch-size 16 --fsdp-devices 8 \
  --action-dim 2 --action-horizon 16 --train-scope adapter \
  --aggregation-rounds 0 --post-aggregation-updates 0 \
  --aggregation-batches 8 --eval-batches 2 --updates-per-round 2500 \
  --benefit-min-rounds 6 --benefit-patience 4 --eval-plot-count 4 \
  --field-time 1.0 \
  --aggregation-step-size-start 0.1 --aggregation-step-size 1.0 \
  --oracle-gradient-floor 1e-4 --max-field-direction-rms 1.0 \
  --rollout-backtracks 10 --backtrack-factor 0.5 --energy-increase-tolerance 1e-6 \
  --train-particles 1 --diversity-direction-weight 0 \
  --prior-dt-s 0.5 --prior-min-v-mps 0.8 --prior-max-v-mps 2.0 \
  --prior-length-scale 0.25 \
  --prior-max-omega-radps 0.5 --prior-forward-only \
  --prior-goal-heading-fraction 1.0 --prior-goal-heading-limit-rad 1.0471975512 \
  --sampled-goal-fraction 1.0 --segment-samples 15 \
  --collision-weight 0 --clearance-weight 1 --clearance-cap-m 0.2 \
  --goal-weight 30 --progress-weight 10 \
  --early-heading-weight 0.25 --smoothness-weight 0.5 \
  --max-step-length-m 1.0 --required-progress-fraction 0.9 \
  --obstacle-safety-margin-m 0.25 \
  --warmup-steps 1000 --peak-lr 1e-4 --decay-steps 30000 --decay-lr 1e-5 \
  --clip-gradient-norm 1.0 \
  --convergence-min-rounds 3 --convergence-patience 2 \
  --convergence-median-path-change-m 0.01 --convergence-p95-path-change-m 0.03 \
  --convergence-relative-energy-change 1e-3 --convergence-oracle-grad-rms 1e-4 \
  --convergence-max-collision-rate 0.05 --convergence-min-progress-ratio 0.9 \
  --save-interval 5000 --trainable-snapshot-interval 0 --log-interval 100 \
  --aggregation-plot-interval 1 --train-plot-num-images 12 \
  "${WANDB_ARGS[@]}" "$@"
