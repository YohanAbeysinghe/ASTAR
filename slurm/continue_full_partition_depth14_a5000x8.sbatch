#!/bin/bash
#SBATCH --job-name=astar-full-d14-a5k8
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=gamma-huge-long
#SBATCH --mem=120G
#SBATCH --time=5-00:00:00
#SBATCH --signal=USR1@600

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

if ! ulimit -Sl unlimited; then
  echo "Cannot raise the soft memlock limit required by NCCL." >&2
  exit 1
fi

export MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/astar_continuation_jax_${SLURM_JOB_ID}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/astar_continuation_mpl_${SLURM_JOB_ID}}"

SOURCE_ROOT="${REPO_ROOT}/checkpoints/astar_clearance025_prefix_a6000x4_20260924_eval35000"
OUTPUT_DIR="${REPO_ROOT}/checkpoints/astar_fulltrain_depth14_from35000_a5000x8_20260930"
RESUME_ARGS=()
if [[ -d "${OUTPUT_DIR}" ]]; then
  RESUME_ARGS=(--resume)
fi

srun --kill-on-bad-exit=1 --gpu-bind=none \
  "${REPO_ROOT}/.venv/bin/python" -m astar.training.continuation \
  --source-root "${SOURCE_ROOT}" \
  --source-step 35000 \
  --output-dir "${OUTPUT_DIR}" \
  --depth-cap 14 \
  --additional-steps "${ADDITIONAL_STEPS:-50000}" \
  --batch-size 16 --fsdp-devices 8 \
  --seed 42 --save-interval 2500 --eval-interval 2500 \
  --source-wandb-run yohanab/astar/4wzq1hru \
  --wandb-enabled \
  "${RESUME_ARGS[@]}" "$@"
