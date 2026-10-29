# Space recommendation for the current paper

**Use one single-column simulation block containing a shallow three-image
strip and the complete comparison table. Replace the existing simulation
table with that block if the images are retained. Put the nine-chart version
in the supplement or presentation.** For the strictest space budget, retain
the current table alone and put all simulation images in the supplement.

[Compact layout proposal](compact_proposal.pdf) · [PNG](compact_proposal.png)
· [editable SVG](compact_proposal.svg)

## What the current paper actually occupies

The current source was compiled twice with the supplement disabled **only in
an isolated review copy**. The main paper plus references occupies eight pages.
The class gives a 7.00-inch text width, 3.40-inch columns, and 9.46-inch text
height. The last page is partly empty, but the bibliography currently has only
four entries and several unresolved citations. Real-world GEAR-Nav values and
the proposed refinement ablation are also pending. The eight-page count is
therefore a current-layout measurement, not evidence that the completed paper
will fit without further editing.

| Visual | Width in paper | Image height, excluding caption |
| --- | ---: | ---: |
| Current nine-chart simulation figure, if added full width | 7.00 in | 5.31 in |
| Proposed compact simulation block | 3.40 in | 2.45 in |
| Current training pipeline | 7.00 in | 6.37 in |
| Current correction-model figure | 3.40 in | 4.96 in |
| Current teaser | 7.00 in | 3.09 in |
| Current real-world figure | 3.40 in | 1.97 in |

The compact proposal uses about **78% less figure area** than the full-width
nine-chart version. This comparison excludes external captions and float
spacing. It is not a claim that the proposal saves 78% versus the existing
table: adding the scene strip to that table increases its height modestly.

## Recommended use of the available space

1. **Simulation: one compact block.** Preserve all 36 numeric entries and the
   three scenarios. Use the existing method names and a subtle orange highlight
   for GEAR-Nav. Main numeric labels in the mockup are 7.8 pt at native column
   width. Do not retain a second table with the same values. Simply reducing
   the nine-chart image to one column would make its numeric labels about
   3.4 pt, which is not a useful print layout.
2. **Real-world: keep the current qualitative figure and result table.** Its
   image is already compact. Add the measured GEAR-Nav row when available;
   avoid another chart repeating those same values.
3. **Spend new results space on the contribution.** A compact prior-versus-model
   or refinement-depth ablation would provide evidence for the learned
   refinement mechanism and dataset aggregation. The current analysis section
   describes planned comparisons; it does not yet supply these measurements.
   Reserve space for actual results rather than a second rendering of the
   simulation table.
4. **Recover space from the pipeline first.** The full-width training figure
   is the largest single visual. Keep the training/aggregation loop in the
   main paper and move the lower expanded energy-oracle walkthrough to the
   supplement, where the architecture is already expanded. This can recover
   several inches of full-width height without shrinking all labels. The
   actual saving depends on the redesigned panel and caption.

## What the results support visually

In the supplied table, GEAR-Nav has the highest success in Scenarios 1 and 3
(92% and 86%). In Scenario 2, its 84% trails TIC-VLA and NaVILA at 90%.
Its collision rates are not lowest across the scenarios, and it does not
uniformly lead clearance. Therefore, keep success, collision, and clearance
together instead of using a success-only headline graphic. In Scenario 3,
NaVILA's high clearance accompanies zero success, illustrating why clearance
alone is not a complete navigation result.

The new GEAR-Nav trial count and episode logs remain unspecified, and the
paper flags protocol matching for verification. The mockup contains aggregate
values only. Its existing hospital/warehouse photos remain explicitly
illustrative separate recordings. Matched evaluation frames would be preferable
for a final main-paper scene strip; if they are unavailable, the table-only
option is the most efficient and least ambiguous choice.

## Files and preservation

- `review_main.pdf`: current eight-page submission-mode proof.
- `review_main.tex`: isolated source snapshot; the only layout changes are
  disabling the supplement and printing class dimensions to the build log.
- `current_page-5.png`, `current_page-7.png`, `current_page-8.png`: inspected
  pipeline, results, and final pages.
- `space_measurements.json`: measured widths and calculated image footprints.
- `protected_files.json`: hashes of the actual paper source, supplement, and
  paper PDFs. All remain unchanged.
- `build_mockup.py`: creates the compact proposal from the frozen comparison
  evidence and verifies protected files.

No proposal was inserted into the paper, and the existing full simulation
figure remains available at `../isaacsim_results.pdf`.
