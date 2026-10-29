# GEAR-Nav

**Geometry-guided trajectory refinement for navigation**

GEAR-Nav is a vision-language navigation research stack that teaches a
pi0.5-based policy to iteratively improve local robot trajectories without
using recorded robot actions or reference paths as training targets.

The central idea is to use geometry as a teacher during training, not as a
required policy input at deployment. A Euclidean signed-distance field
(ESDF), a goal location, and robot geometry define a differentiable trajectory
energy. Its projected negative gradient says how a candidate waypoint path
should change to become safer, smoother, and more goal-directed. GEAR-Nav trains
pi0.5 to predict that local refinement direction from the robot observation,
the selected goal modality, and the current candidate path.

At inference time, the learned field can repeatedly refine a dynamically
feasible waypoint prior. The model itself does not consume an ESDF, a map, an
A* path, or a demonstrated action sequence.

> **In one sentence:** GEAR-Nav distills a map-based geometric optimizer into a
> vision-language trajectory-refinement policy.

> **Naming and compatibility:** GEAR-Nav was previously called ASTAR. The Python package is `gear_nav`; the legacy `astar` package path and CLI commands remain available for existing scripts and checkpoints. `GEAR_NAV_DATA_ROOT` is preferred, with `ASTAR_DATA_ROOT` supported as a fallback. Historical checkpoint directories and experiment links retain their original names.

The workspace on the original machine also has an `ASTAR` symlink pointing to
`GEAR-Nav`. This machine-local link is not created by cloning the repository.
Use one Python import spelling consistently within a process: loading the same
module through both package names creates separate Python module objects.

## What this project is trying to achieve

Many navigation policies are trained by behavior cloning: collect an expert
trajectory, then ask a network to reproduce the expert's actions. This creates
three practical problems:

1. expert action collection is expensive and tied to a particular robot and
   controller;
2. a single demonstrated path gives only one answer, even when many safe paths
   are valid; and
3. behavior-cloned policies receive weak supervision on the off-demonstration
   states they encounter after making their own mistakes.

GEAR-Nav replaces action imitation with **trajectory improvement learning**. It
starts from candidate local paths, asks a differentiable geometric objective
how each path should improve, trains a neural field on those directions, and
then uses the learned field to generate the next set of states. Repeating this
process moves training data toward the states produced by the policy itself.

The intended result is a policy that can:

- produce a 16-waypoint local path from a forward RGB observation;
- condition navigation on language, an object crop, a relative metric
  waypoint, or combinations of those modalities;
- make progress toward a goal while respecting robot footprint clearance;
- refine its own candidate paths over multiple iterations; and
- run without the training-time ESDF being passed through the model.

GEAR-Nav is not a replacement for the low-level controller. It predicts a local
metric path; a path manager and controller are still responsible for turning
that path into robot velocity commands.

## System at a glance

```text
                         TRAINING ONLY

 RGB observation -----+--------------------------+
 goal text/image/xy ---+--> pi0.5 refinement field|--> predicted direction
 candidate waypoints --+--------------------------+            |
                                                               v
 ESDF + robot radius + goal xy --> trajectory energy --> oracle direction
                                      dE/d(path)              |
                                                               v
                           replay of (state, oracle direction)
                                                               |
                                  field-regression training <--+
                                                               |
                                  learned rollout creates -----+
                                  the next candidate paths

                         DEPLOYMENT / ROLLOUT

 observation + goal + feasible prior
                  |
                  v
        learned refinement field, repeated N times
                  |
                  v
        16 local (x, y) waypoints --> path manager --> controller
```

There are two important boundaries in this diagram:

- **The model input** contains perception, goal conditioning, and the current
  candidate path. It does not contain the ESDF.
- **The training oracle** may use the ESDF, robot radius, and metric goal to
  calculate a supervision direction. Those tensors are deliberately kept
  outside the policy observation.

## Core representation

The canonical energy-aggregation configuration uses:

| Item | Representation |
| --- | --- |
| Policy backbone | pi0.5 / OpenPI |
| Observation | 224 x 224 forward RGB image plus robot state |
| Goal | text, object-image crop, relative `(x, y)` waypoint, or a combination |
| Candidate/action state | 16 absolute local-frame `(x, y)` waypoints |
| Fixed origin | waypoint 0 is always `(0, 0)` |
| Learned output | a direction in the same 16 x 2 waypoint space |
| Geometric teacher | differentiable ESDF trajectory energy |
| Canonical train scope | action and goal adapters only |

The waypoint action adapter preserves the pi0.5 action-stack interface: it
projects waypoint states into action tokens, injects the time embedding used by
the action expert, and decodes the hidden state back into waypoint-space
directions. A separate MLP projects metric waypoint goals into soft prefix
tokens in the PaliGemma/Gemma embedding space.

The model predicts a **field**, not a final path in one shot. If `p_k` is the
current path and `F_theta` is the learned direction, a simplified rollout is:

```text
p_(k+1) = Project[p_k + alpha * F_theta(observation, goal, p_k)]
```

`Project` fixes the origin, enforces finite local waypoints, and limits segment
lengths. The actual implementation can also cap direction RMS and backtrack an
update until the geometric energy does not increase.

## Why the training is action-label-free

The general dataset loader understands recorded `path_xytheta`, `path_xy`, and
`actions_vw` arrays because they are useful to older loaders, diagnostics, and
evaluation. The canonical aggregation trainer does not use them as labels.

`OracleOnlyNavigationDataset` loads only the information needed for a
condition:

- current RGB image;
- text/object metadata and optional object crop;
- goal coordinate;
- ESDF metadata and grid; and
- robot radius.

It inserts zero trajectory placeholders only to satisfy the inherited OpenPI
batch interface. Before a condition enters aggregation,
`sanitize_metric_tensors` checks that no nonzero ground-truth trajectory tensor
is present and removes all path/action keys. The training target is then
computed from the current candidate path by differentiating the energy.

This means **action-label-free does not mean supervision-free**. The method
uses geometric supervision, but it does not imitate an expert's recorded
control sequence or planned path.

## The trajectory objective

For a candidate path `p`, GEAR-Nav minimizes the weighted objective

```text
E(p) = lambda_clear * E_clearance
     + lambda_goal  * E_goal
     + lambda_prog  * E_retreat
     + lambda_head  * E_early_heading
     + lambda_smooth * E_smoothness
```

The canonical launchers use weights `1, 5, 10, 0.25, 0.5` respectively.

### Clearance

The ESDF is sampled along every path segment, not only at the 16 discrete
waypoints. Raw ESDF distance is converted to robot-footprint clearance:

```text
footprint_clearance = ESDF_distance - robot_radius
```

The optimized clearance term is a squared hinge below the requested margin:

```text
E_clearance = mean(max(0, safety_margin - footprint_clearance)^2)
```

The canonical safety margin is 0.25 m outside the robot footprint. For the
common 0.50 m dataset robot radius, this asks the robot center to remain 0.75 m
from obstacles. Physical collision remains a separate diagnostic defined by
`footprint_clearance < 0`; collision is not a second term in the optimized
energy.

### Goal progress

GEAR-Nav does not supervise only the endpoint. Every future waypoint receives a
linearly increasing goal-progress weight, from `1 / 15` at the first future
waypoint to `1` at the endpoint. This supplies a dense forward signal across
the complete path.

The required progress is bounded by both the goal distance and the feasible
motion budget. A separate retreat penalty activates whenever a segment
increases distance to the goal.

### Heading and smoothness

The early-heading term aligns near-robot segments with the goal direction. The
smoothness term penalizes second differences in waypoint position, discouraging
sharp oscillations and jagged local paths.

### Feasibility projection

Candidate paths are decoded as origin-fixed absolute local waypoints. Each
segment is radially clipped to a goal-aware limit:

```text
segment_cap = min(
    physical_max_step,
    detour_factor * min(goal_distance, physical_horizon) / num_segments,
)
```

The canonical run uses a 1.0 m physical cap and a 1.25x detour allowance. This
keeps short-goal paths from spending an unnecessarily large motion budget while
still allowing obstacle detours.

## Aggregation and self-training

Training is an outer-loop dataset-aggregation procedure:

1. **Cache conditions.** Build a deterministic pool of RGB/goal/ESDF
   conditions. The canonical Slurm jobs use 64 batches x 16 examples = 1,024
   fixed conditions.
2. **Initialize paths.** Sample forward-only unicycle random walks, rotate them
   toward the goal bearing, and project them into the allowed local region.
3. **Query the oracle once.** Differentiate the energy with respect to every
   path. Normalize and cap the negative gradient, project the proposal, and
   backtrack until it is finite and non-energy-increasing.
4. **Append replay.** Store the visited candidate path and its accepted oracle
   direction. The large observation and ESDF pool is reconstructed
   deterministically instead of duplicated in the replay archive.
5. **Train the field.** Regress the pi0.5 field onto oracle directions sampled
   from every retained aggregation round.
6. **Advance with the student.** Apply the learned field to the current paths.
   These student-produced states become the next round's oracle queries.
7. **Validate and stop.** Evaluate model-only rollouts on clip-disjoint
   validation data. Stop on geometric convergence, a validation plateau, or an
   explicit resource limit.

This is DAgger-like in spirit: the model is trained on the state distribution
it creates, while the teacher labels those states only when they are first
visited.

Replay selection is deterministic from the optimizer step, so an interrupted
and resumed run sees the same replay sequence as an uninterrupted run. Full
resumption restores model parameters, optimizer state, aggregation frontier,
cached oracle labels, convergence state, and validation history.

### Full-partition continuation

`gear_nav.training.continuation` supports a second training phase after a fixed
aggregation depth has been learned. A frozen source checkpoint rolls each new
training example to a reproducibly sampled depth, the analytic oracle labels
that state, and only the student is updated. This exposes the student to the
full training partition without caching millions of images or maps and without
letting the teacher depth drift during continuation.

## Goal conditioning

GEAR-Nav supports two goal families:

- **Object goals** can expose language, an object crop, a metric waypoint, or
  any independently sampled combination of those inputs.
- **Sampled free-space goals** expose a metric waypoint because they have no
  object label or crop.

The canonical 1,024-condition pool uses approximately 1/6 sampled goals and
5/6 object goals. For object goals, text and image are each enabled with
probability 1/3 and waypoint conditioning with probability 1/6. If all three
draws are false, text is enabled as a fallback. The standard launchers require
the realized pool to contain every modality before training starts.

Goal modality masking applies to the policy observation. The geometric oracle
still retains the metric goal needed to define progress and supervise examples
whose policy input contains only text or an image.

## Training-time versus inference-time information

| Information | Geometric teacher | Learned model | Needed by a model-only rollout |
| --- | :---: | :---: | :---: |
| Forward RGB image | no | yes | yes |
| Goal text | no | optional | optional |
| Goal image crop | no | optional | optional |
| Relative goal waypoint | yes | optional | required only for waypoint-conditioned runs or goal-biased wrapper logic |
| Current candidate path | yes | yes | yes |
| ESDF / obstacle map | yes | no | no |
| Robot radius | yes | no | no |
| Demonstrated path/actions | no | no | no |

`guarded` offline evaluation is intentionally different from a pure
model-only rollout: it consults the energy to reject an increasing learned
update. Use `model_only` to measure the deployable learned refinement without
that geometric guard.

## Repository layout

```text
GEAR-Nav/
├── src/gear_nav/
│   ├── adapters/              # metric-goal to prefix-token projector
│   ├── heads/                 # pi0.5 waypoint action adapter
│   ├── training/              # aggregation, replay, continuation, monitoring
│   ├── evaluation/            # held-out evaluation and multistep rendering
│   ├── dataloader.py          # manifest/index, modalities, OpenPI batches
│   ├── path_sampler.py        # unicycle prior and path projection
│   ├── waypoint_energy.py     # differentiable geometric objective
│   └── esdf.py                # batched ESDF sampling
├── configs/splits/            # versioned clip-disjoint train/eval IDs
├── slurm/                     # canonical training/evaluation launchers
├── tests/                     # geometry, replay, resume, eval, distributed tests
├── docs/training.md           # operational training notes
├── data/README.md             # external dataset schema
├── checkpoints/README.md      # checkpoint download and layout
├── NavigationStack/           # optional ROS2 deployment/evaluation gitlink
└── third_party/openpi/        # pinned OpenPI/pi0.5 submodule
```

Large or machine-specific artifacts are intentionally ignored by Git:

- datasets under `data/`;
- pretrained and trained checkpoints under `checkpoints/`;
- W&B caches and generated outputs;
- local `eval/` visualizations;
- local deployment bundles; and
- simulator runs under `SimResults/`.

The tracked repository is therefore the reproducible training and offline
evaluation implementation, not an all-inclusive artifact release.

## Installation

GEAR-Nav requires Linux, Python 3.11, and an NVIDIA GPU for realistic training.
Install [`uv`](https://docs.astral.sh/uv/), clone the repository, and initialize
the OpenPI submodule:

```bash
git clone https://github.com/YohanAbeysinghe/GEAR-Nav.git GEAR-Nav
cd GEAR-Nav
git submodule update --init --recursive third_party/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --extra dev
```

`NavigationStack/` is stored as an optional Git link but currently has no URL
entry in the root `.gitmodules`; initialize OpenPI by its explicit path as
shown above. This does not affect core training or evaluation.

The working OpenPI fork and pinned revision are:

```text
https://github.com/gershom96/openpi
a324ec445007031ed5b11d708ad764406b1f55ee
```

Do not silently replace this revision with OpenPI `main` when reproducing a
run; GEAR-Nav relies on the goal and waypoint adapter interfaces in the pinned
fork.

### Download pi0.5 parameters

The roughly 12 GB pi0.5 base checkpoint is an external artifact:

```bash
uv run gear-nav-download-pi05
```

This downloads:

```text
gs://openpi-assets/checkpoints/pi05_base/params
```

and exposes it at:

```text
checkpoints/pi05_base/params
```

The default is a symlink to the OpenPI cache, avoiding a duplicate copy. Set
`OPENPI_DATA_HOME=/path/to/cache` before downloading to move the cache, or pass
`--copy` when a physical local copy is required.

Verify the environment and checkpoint with:

```bash
uv run gear-nav-verify --require-checkpoint
```

## Dataset

The dataset is external to Git. Set its root explicitly:

```bash
export GEAR_NAV_DATA_ROOT=/absolute/path/to/dataset
```

The root contains a dataset manifest, a JSONL sample index, RGB images, ESDF
arrays, and per-clip geometry metadata. Object records may include labels,
captions, and bounding boxes used to create goal-image crops. The generic
loader also supports planned path arrays for diagnostics and non-aggregation
workflows.

A typical layout is:

```text
/path/to/dataset/
├── dataset_manifest.json
├── samples.jsonl
├── <clip-id>/
│   ├── manifest.json
│   ├── images/
│   ├── esdf/
│   └── paths/                  # optional for oracle-only aggregation
└── ...
```

The canonical aggregation condition requires each usable sample/goal to
resolve:

- `clip_id`, `sample_id`, and `image_path`;
- `goal_xy_m` for an object or sampled goal;
- `esdf_path`;
- ESDF bounds and resolution; and
- the robot radius, with a validated nonnegative value.

See [`data/README.md`](data/README.md) for the complete shared-loader schema.
The checked-in clip IDs under [`configs/splits/`](configs/splits/) are used
explicitly by the Slurm launchers. The evaluator verifies that the evaluation
clips do not overlap the training split.

## Training

### Canonical multi-node run

The primary launcher targets two nodes with eight RTX A5000 GPUs per node, one
JAX controller per node, and eight-way FSDP within each node:

```bash
GEAR_NAV_DATA_ROOT=/absolute/path/to/dataset \
  sbatch slurm/train_2node_16gpu.sbatch
```

Give every clean experiment a new name:

```bash
GEAR_NAV_DATA_ROOT=/absolute/path/to/dataset \
EXP_NAME=astar_clean_v1 \
  sbatch slurm/train_2node_16gpu.sbatch
```

For one node with four RTX A6000 GPUs:

```bash
GEAR_NAV_DATA_ROOT=/absolute/path/to/dataset \
EXP_NAME=astar_a6000_v1 \
  sbatch slurm/train_1node_4gpu_a6000.sbatch
```

Inspect the complete direct CLI before constructing a custom run:

```bash
uv run gear-nav-train --help
```

The canonical jobs use a global batch size of 16, 64 aggregation batches,
2,500 optimizer updates per aggregation round, 16 x 2 waypoint paths, adapter
fine-tuning, and a target of up to 64 aggregation rounds while retaining
geometric convergence checks.

### Clean runs, resume, and migration

A clean run must use a new experiment name and must not pass `--resume` or a
relabel/migration option. To continue an interrupted run, use the same name and
resume-critical configuration with `--resume`.

Changing the path objective, ESDF cutoff, or goal-conditioning distribution
changes the meaning of cached oracle labels. The trainer therefore guards such
changes behind explicit migration/relabel flags. These flags are for controlled
experiments, not normal fresh training.

Training outputs are written below:

```text
checkpoints/astar_pi05_energy_aggregation/<experiment-name>/
```

Important run artifacts include:

- a full model/optimizer checkpoint;
- `aggregation_state_<step>.npz`, containing the matching replay/frontier;
- `monitoring/configuration.json`, containing resolved arguments and
  provenance;
- training and validation JSONL diagnostics; and
- path-history and validation plots.

An aggregation checkpoint and its replay archive are a matched pair. Copying
only the model parameters is sufficient for inference but not for exact
training resumption.

### Weights & Biases

W&B is opt-in in the Slurm launchers:

```bash
WANDB_ENABLED=1 \
WANDB_ENTITY=<entity> \
WANDB_PROJECT=astar \
GEAR_NAV_DATA_ROOT=/absolute/path/to/dataset \
  sbatch slurm/train_2node_16gpu.sbatch
```

The logged configuration includes requested and realized modality counts,
condition-pool size, energy settings, JAX process/device counts, aggregation
depth, training diagnostics, model-only validation metrics, and path plots.

## Offline evaluation

Evaluate a completed experiment with:

```bash
uv run gear-nav-eval \
  --checkpoint-root checkpoints/astar_pi05_energy_aggregation/astar_clean_v1 \
  --data-root "$GEAR_NAV_DATA_ROOT" \
  --eval-split-ids configs/splits/eval_clip_ids.txt \
  --steps <checkpoint-step>
```

Or use the four-GPU Slurm launcher:

```bash
GEAR_NAV_DATA_ROOT=/absolute/path/to/dataset \
CHECKPOINT_ROOT=/absolute/path/to/experiment \
CHECKPOINT_STEPS="30000" \
  sbatch slurm/evaluate.sbatch
```

The evaluator reads the saved training configuration and compares four modes:

| Mode | Meaning |
| --- | --- |
| `prior` | structured goal-biased path before learned refinement |
| `model_only` | learned iterative refinement with projection only |
| `guarded` | learned refinement plus ESDF energy backtracking |
| `oracle` | analytic projected energy-gradient refinement |

It uses deterministic, clip-balanced example selection and can evaluate raw or
EMA parameters at multiple checkpoints and rollout depths. Safety is rescored
independently with dense segment sampling and conservative ESDF-coverage
semantics.

Primary metrics include:

- goal-progress and safe-success rates;
- collision and requested-clearance violation rates;
- final goal error and achieved progress;
- minimum footprint clearance;
- trajectory energy and per-term energies;
- field/oracle cosine similarity and RMSE;
- update acceptance and energy non-increase; and
- finite-action and valid-ESDF coverage rates.

Evaluation outputs default to `outputs/evaluation/eval_<timestamp>/` and
include:

```text
evaluation_config.json       # provenance and resolved settings
selected_examples.json       # deterministic held-out sample list
examples.jsonl               # per-example, per-mode, per-depth records
summary.json / summary.csv   # aggregate metrics with bootstrap CIs
paired_summary.*             # paired mode/checkpoint differences
per_clip_summary.csv         # clip-level breakdown
final_paths.npz              # optional final waypoint arrays
plots/                       # worst-case visualizations
```

The validation split is used for aggregation stopping and model selection; it
should not be described as an untouched final test set.

## Simulation and deployment artifacts in this checkout

The core Git repository does not version simulator results or packaged
deployment checkpoints. This working checkout may additionally contain:

- `SimResults/astar_57500_oracle_goal/` — a 150-trial, three-scene simulator
  diagnostic for step 57,500 at refinement depth 14;
- `SimResults/astar_57500_oracle_goal_100/` — a resumable extension and video
  export workflow;
- `deployment/` — a laptop ROS2 bundle that serves GEAR-Nav and preserves the
  NavigationStack `/path` interface; and
- `eval/` — generated offline path plots and summaries.

These directories are ignored because they contain machine-specific paths,
large checkpoints, videos, and generated evidence. They are useful experiment
records, but they are not automatically reproduced by cloning the repository.

The completed 50-trial-per-scene oracle-goal diagnostic reports 149 successes
and one contact collision across 150 trials:

| Scene | Trials | Success | Contact collision | Normalized goal progress |
| --- | ---: | ---: | ---: | ---: |
| Scenario 1 | 50 | 100% | 0% | 0.831 |
| Scenario 2 | 50 | 98% | 2% | 0.785 |
| Scenario 3 | 50 | 100% | 0% | 0.799 |

This result is explicitly an **oracle-goal upper-bound diagnostic**. The policy
receives the simulator's exact frozen goal transformed into the current robot
frame, so these numbers must not be presented as language-only navigation
performance. The local manifests also record that scene selection involved
development screening; reported rates are conditional on that protocol.

For robot deployment, the optional ROS2 path is:

```text
camera + odometry + goal
          |
          v
GEAR-Nav inference server / model node
          |
          v
     /path (nav_msgs/Path)
          |
          v
path manager --> /next_goal --> DWA/PID controller --> /cmd_vel
```

See the local deployment runbook when that ignored bundle is available. The
tracked [`NavigationStack/README.md`](NavigationStack/README.md) documents the
generic ROS2 topic contract and offline robotics evaluation skeleton.

## Testing

Run the complete repository test suite with:

```bash
uv run pytest
```

The suite covers ESDF interpolation and validity, energy geometry, goal-aware
segment caps, replay determinism and serialization, exact resume behavior,
goal-conditioning/objective migrations, checkpoint milestones, continuation
sampling, evaluation selection and summaries, monitoring, multistep paths, and
multi-process distributed startup. The distributed integration test runs with
simulated JAX controllers on CPU but is more expensive than the unit tests.

For a lightweight installation check before allocating GPUs:

```bash
uv run gear-nav-verify
```

## Reproducibility and interpretation

Each run records its resolved arguments, split paths, condition fingerprints,
goal-modality counts, JAX topology, energy configuration, and aggregation
algorithm version. Evaluation additionally records Git/source hashes, split
hashes, checkpoint source, deterministic sample selection, and scorer settings.

When reporting a GEAR-Nav result, include at least:

- checkpoint step and raw/EMA parameter source;
- aggregation or rollout depth;
- goal kind and active modality;
- `model_only`, `guarded`, or `oracle` mode;
- train/evaluation split hashes;
- clearance margin and robot radius convention;
- dense safety sampling spacing; and
- whether the metric is validation, offline test, simulation, or real-robot
  evaluation.

This avoids three common misinterpretations:

1. guarded performance is not the same as map-free model-only performance;
2. an oracle waypoint is not a language-only goal; and
3. clearance below the requested 0.25 m margin is not necessarily physical
   collision, which begins below zero footprint clearance.

## Command-line entry points

| Command | Purpose |
| --- | --- |
| `gear-nav-train` | single-process/multi-GPU energy aggregation |
| `gear-nav-train-distributed` | initialize JAX across multiple Slurm nodes, then train |
| `gear-nav-eval` | held-out prior/model/guarded/oracle evaluation |
| `gear-nav-download-pi05` | fetch and validate the base pi0.5 parameters |
| `gear-nav-verify` | verify imports, devices, and optionally the checkpoint |

Additional utilities under `src/gear_nav/` archive checkpoint milestones, render
multistep path evolution, and continue a fixed-depth teacher on the complete
training partition.

## Limitations

- ESDFs and metric goal coordinates are still required to generate geometric
  training targets.
- The canonical implementation learns local 2D waypoint refinement; it does
  not directly output velocity commands.
- Safety constraints are learned and empirically evaluated, not formally
  guaranteed by the model-only policy.
- `guarded` inference needs geometric energy information and therefore has a
  different deployment assumption from `model_only`.
- The public repository does not include the dataset, trained checkpoints,
  simulator environment, or ROS2 laptop bundle.
- The default training recipe is computationally heavy and was designed for
  multi-GPU Slurm systems.

## License

GEAR-Nav is released under the MIT License. The OpenPI submodule, pi0.5 model
artifacts, datasets, simulator assets, and optional deployment dependencies
retain their own upstream licenses and terms.
