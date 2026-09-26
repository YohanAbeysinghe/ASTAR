#!/bin/bash
#SBATCH --job-name=astar-ckpt-watch
#SBATCH --output=slurm-%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --qos=gamma-huge-long
#SBATCH --mem=2G
#SBATCH --time=5-00:00:00

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${REPO_ROOT}"

: "${TRAINING_JOB_ID:?Set TRAINING_JOB_ID to the Slurm training job to watch}"
: "${RUN_DIR:?Set RUN_DIR to the active experiment checkpoint directory}"

KEEP_INTERVAL="${KEEP_INTERVAL:-10000}"
POLL_SECONDS="${POLL_SECONDS:-30}"

PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${REPO_ROOT}/.venv/bin/python" -m astar.checkpoint_milestones \
  --run-dir "${RUN_DIR}" \
  --interval "${KEEP_INTERVAL}" \
  --training-job-id "${TRAINING_JOB_ID}" \
  --poll-seconds "${POLL_SECONDS}"
