# Isaac Sim evaluation figure

For the main-paper layout, the selected proposal is now an
[images-only scene strip](space_review/compact_proposal.pdf). Evaluation values
stay in the paper's existing table; the figure does not repeat them.
Gold stars identify the target people in these
illustrative recordings. The recorded instructions, source hashes, and manual
image coordinates are saved in `space_review/goal_annotations.json`.

The current [PDF](isaacsim_results.pdf), [PNG](isaacsim_results.png), and
[editable SVG](isaacsim_results.svg) visualize **all 36 simulation values** in
the paper's updated `tab:isaac_sim`. GEAR-Nav is orange; OmniVLA, TIC-VLA, and
NaVILA retain the blue, magenta, and purple colors of the real-world figure.
The figure shows success, collision rate, and clearance for each scenario.

**Keep these drafts in `figs/`. Do not copy them into the paper repository or
insert them into the paper unless the user explicitly requests it.** No paper
files, including `main.tex`, `supplementary.tex`, or the real-world figure,
were changed for this update.

| Scenario | GEAR-Nav success | Collision | Clearance |
| --- | ---: | ---: | ---: |
| 1 | 92% | 6% | 0.78 m |
| 2 | 84% | 10% | 0.46 m |
| 3 | 86% | 12% | 0.71 m |

- [Compact scene row](isaacsim_scene_strip.pdf): three recorded views and the
  current GEAR-Nav aggregate results.
- [Paper-size proof](review/paper_proof.pdf): full figure at 7.12 inches wide
  with the proposed [caption](caption.tex).
- [Optional LaTeX placement](paper_integration.tex) is a local review snippet;
  it has not been inserted into the paper.

## Source and interpretation

`prepare.py` extracts the table directly from `GEAR-Nav Overleaf/main.tex`.
All 27 baseline values are independently checked against the active simulation
table in `Overleaf--VEGA/sec/5_results.tex`. They are previously reported VEGA
results. GEAR-Nav's nine values were supplied by the user and added to the paper
in commit `a38487f`. `GEAR-Nav Overleaf/BASELINE_RESULTS.md` documents the source
and the outstanding verification of run provenance and protocol matching.

Rates are converted from 0–1 to percentages; clearance stays in meters. The
source defines clearance as average minimum clearance over non-colliding
trials. Success and collision are plotted independently. No counts, error bars,
confidence intervals, or significance claims are inferred from the rounded
aggregate values. The reference baseline trial count is 50 per scenario;
the new GEAR-Nav count and per-trial logs have not been supplied.

The three photos are unchanged recorded third-person JPEGs from the earlier
local diagnostic: hospital lobby, hospital corridor, and warehouse. They are
explicitly **illustrative views from separate runs**, not evidence that the
new metrics use those exact scenes or goals. In the full evaluation chart,
the old goal markers, trajectory
plots, contact event labels, trial counts, and Wilson intervals are excluded.
No imagery or trajectories were generated or retouched.

The superseded 150-trial oracle-goal figure and its evidence are retained in
[`oracle_goal_diagnostic/`](oracle_goal_diagnostic/README.md). Its values
(100%, 98%, 100% success) are distinct from the new supplied results and do not
contribute to this figure.

## Rebuild and validation

To refresh values from the paper and check the reference baseline entries:

```sh
.venv/bin/python figs/isaacsim_results/prepare.py
.venv/bin/python figs/isaacsim_results/render.py
```

`render.py` alone uses the frozen `evidence.json`. `aggregate_results.csv`
contains the 12 method/scenario rows. `provenance.json` records source-file
hashes, image provenance, source commit, and protected paper-file hashes.
`verification.json` records all 36 plotted metric groups and output hashes.

To rebuild the paper-size proof, run `pdflatex -interaction=nonstopmode
-halt-on-error paper_proof.tex` from `figs/isaacsim_results/review/`.

Only the GEAR-Nav workspace repository is updated and pushed. The paper
repository is read for values and left unchanged.
