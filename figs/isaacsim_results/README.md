# Isaac Sim results figure

Separate review draft for the three recorded Isaac Sim scenarios. The existing
`GEAR-Nav Overleaf/fig/astar_realworld.pdf` and `main.tex` are unchanged.
Keep these drafts in `figs/`; do not copy them into the paper repository or
insert them into the paper unless the user explicitly requests it.

- [Full figure](isaacsim_results.pdf): third-person images, all 150 recorded
  trajectories, and repeated-trial outcomes with nominal 95% Wilson intervals.
- [PNG preview](isaacsim_results.png) and [editable SVG](isaacsim_results.svg).
- [Compact scene row](isaacsim_scene_strip.pdf): three images and concise
  statistics for a possible later composition with the real-world figure.
- [Paper-size proof](review/paper_proof.pdf): full figure at 7.12 inches wide
  with the proposed caption. Use the full figure across two paper columns.
- [Caption](caption.tex) and [optional LaTeX placement](paper_integration.tex).

The photos are original recorded JPEGs, with target-person stars added as
manual image annotations. Other elements are editable vectors. Orange, yellow,
pale blue panels, and serif type follow the real-world and shared figure styles.
No generated imagery, fabricated routes, model inference, or Figma calls are used.

## Verified experiment scope

| Scene | Completed metric trials | Successes | Contact trials | Success 95% interval |
| --- | ---: | ---: | ---: | --- |
| 1: Hospital lobby | 50 | 50 | 0 | 92.9–100.0% |
| 2: Hospital corridor | 50 | 49 | 1 | 89.5–99.6% |
| 3: Warehouse | 50 | 50 | 0 | 92.9–100.0% |

The raw checkpoint is step 57,500 with 14 refinement rounds. Every policy
request receives the **exact simulator goal transformed into the robot frame**.
This is an **oracle-goal diagnostic**, not a language-only comparison. Trial
outcomes are conditional on three fixed scene geometries and their pedestrian
timings. Scene selection followed VEGA development screening, as disclosed by
the original report. These are custom Isaac Sim trials, not official benchmark
scores. There are no matched baseline results in this figure, so no claim of
comparative statistical significance is made.

`astar_57500_oracle_goal_100/README.md` describes a planned extension to 100
trials per scene. Its extension result directories contain no completed episode
JSONs. The cumulative summary is stale (113 trials); the validated base report
and all 150 raw results agree. Neither the planned extension, selected success
reruns, nor challenge variants in `SimResults/picked` enter these statistics.

## Evidence and interpretation

- `trial_outcomes.csv`: one row per validated metric trial.
- `evidence.json`: frozen outcomes, all saved pose samples, scene coordinate
  transforms, and selected image metadata; allows rendering without SimResults.
- `provenance.json`: source hashes for all 150 results and their pose sequences,
  experiment protocol, source image hashes, protected real-world PDF hash, and
  interpretation limits. Every raw result matches the validated report hash;
  frozen configuration hashes also match the evaluation manifest.
- `verification.json`: output hashes, counts, trajectory bounds, and protected
  real-world PDF preservation.
- `style.json`: portable snapshot of relevant shared colors and serif font.

Trajectories join saved poses every 10 simulator frames without smoothing. One
rigid frame per scene uses the configured start and the first trial's initial
heading. The last recorded state may precede termination slightly. The cross on
the red path is that last pose; it does not locate a contact event. Shaded goal
disks use the frozen 1.5 m success threshold. All trajectories and complete goal
disks fit within the shared axes. Repeated paths are nearly coincident because
geometry is fixed; the rendering preserves this measured behavior.

Photo selection is the earliest episode ID among each scene's available seeded
random collision-free success reruns: scene 1 episode 21, scene 2 episode 01,
and scene 3 episode 12. Observations 8, 16, and 8 respectively avoid startup noise
and expose the target person (occluded earlier in the corridor). These separate
reruns illustrate scene appearance, not the plotted metric trajectories.

The figure contains success and actual PhysX contact counts. FOV human-surface
clearance is preserved in the evidence but is not labelled all-obstacle clearance
or mixed with real-world safety metrics. Wilson intervals are nominal descriptive
intervals, not a significance test or evidence of generalization to new maps.

## Rebuild

From the repository root, using the frozen portable evidence:

```sh
.venv/bin/python figs/isaacsim_results/render.py
```

To refresh from original local results (requires SimResults and the frozen
benchmark configuration paths):

```sh
.venv/bin/python figs/isaacsim_results/prepare.py
.venv/bin/python figs/isaacsim_results/render.py
```

To compile the paper-size proof, run `pdflatex -interaction=nonstopmode
-halt-on-error paper_proof.tex` from this folder's `review/` directory.

The five figure and caption files previously copied into the paper repository
were removed at the user's request. No Isaac Sim draft assets remain in that
repository. All editable masters, previews, evidence, and the optional placement
snippet remain here for review; no figure was inserted into `main.tex`.
