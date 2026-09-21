#!/bin/bash
#SBATCH --job-name=astar-eval
#SBATCH --output=slurm-%x-%j.out
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=huge-long
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

: "${ASTAR_DATA_ROOT:?Set ASTAR_DATA_ROOT to the prepared dataset directory}"
: "${CHECKPOINT_ROOT:?Set CHECKPOINT_ROOT to an ASTAR experiment directory}"
if [[ -x "${REPO_ROOT}/.venv/bin/astar-eval" ]]; then
  EVAL_COMMAND=("${REPO_ROOT}/.venv/bin/astar-eval")
else
  UV_BIN="${UV_BIN:-uv}"
  command -v "${UV_BIN}" >/dev/null 2>&1 || {
    echo "Run 'uv sync' first or set UV_BIN to the uv executable." >&2
    exit 1
  }
  EVAL_COMMAND=("${UV_BIN}" run astar-eval)
fi

read -r -a CHECKPOINT_STEP_ARGS <<< "${CHECKPOINT_STEPS:?Set CHECKPOINT_STEPS, for example 30000}"
read -r -a MODE_ARGS <<< "${EVAL_MODES:-prior model_only guarded oracle}"

export MPLBACKEND=Agg
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/astar_eval_jax_${SLURM_JOB_ID}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/astar_eval_mpl_${SLURM_JOB_ID}}"

"${EVAL_COMMAND[@]}" \
  --checkpoint-root "${CHECKPOINT_ROOT}" \
  --data-root "${ASTAR_DATA_ROOT}" \
  --eval-split-ids "${REPO_ROOT}/configs/splits/eval_clip_ids.txt" \
  --steps "${CHECKPOINT_STEP_ARGS[@]}" \
  --modes "${MODE_ARGS[@]}" \
  --parameter-sources raw \
  --goal-kind sampled \
  --examples-per-clip "${EXAMPLES_PER_CLIP:-128}" \
  --batch-size "${BATCH_SIZE:-8}" \
  --fsdp-devices 4 \
  --rollout-rounds "${ROLLOUT_ROUNDS:-12}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-2000}" \
  --num-plots "${NUM_PLOTS:-12}" \
  "$@"
