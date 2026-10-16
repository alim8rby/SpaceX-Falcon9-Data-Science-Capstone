# Executed results

Input: 90 course-snapshot launches; 2010-06-04 to 2020-11-05.
Course target: 60 successes, 30 non-successes.

Models are selected by training-only five-fold stratified CV balanced accuracy.
The held-out test set is evaluated only after selection, once per declared split.

| Evaluation | Selected model | Test n | Accuracy | Balanced accuracy | Baseline accuracy |
|---|---|---:|---:|---:|---:|
| Random holdout | SVM | 18 | 0.778 | 0.667 | 0.667 |
| Chronological sensitivity | Decision tree | 18 | 0.889 | 0.667 | 0.833 |

## Interpretation and limits

- The selected SVM correctly classified 14/18 random-holdout launches, including all 12 successes but only 2/6 non-successes. Accuracy alone hides this asymmetry.
- Small historical educational datasets; these are not deployment performance estimates.
- Random splits mix historical eras. The chronological check exposes time-shift sensitivity; its training CV remains stratified, not rolling-origin forecasting.
- Class 1 includes controlled ocean landings. Class 0 includes no-attempt/unknown outcomes; this is not a pure attempted-landing failure label.
- No causal effects or launch-cost savings are established.
- PayloadMass is upstream-imputed and excluded from modeling. Cumulative ReusedCount and target-derived Outcome are also excluded.
- Block, Flights, and hardware attributes are assumed available before launch; retrospective source revisions can still create point-in-time bias.
- JSON files record class counts, flight IDs, parameters, confusion matrices, and runtime versions.
- The original presentation remains historical course work; it is not the source of these newly executed metrics.
