"""Run from repository root: python -m scripts.run_analysis."""
import json
from falcon9.analysis import ROOT, load_data, save_results, group_summary, sql_analysis


def main():
    df = load_data()
    output = save_results(df)
    for column in ['LaunchSite', 'Orbit', 'Year']:
        group_summary(df, column).to_csv(output / f'{column.lower()}_summary.csv', index=False)
    for name, frame in sql_analysis(df).items():
        frame.to_csv(output / f'sql_{name}.csv', index=False)
    (ROOT / 'data/processed').mkdir(exist_ok=True)
    df.to_csv(ROOT / 'data/processed/launches.csv', index=False)
    random = json.loads((output / 'random_metrics.json').read_text())
    temporal = json.loads((output / 'temporal_metrics.json').read_text())
    lines = ['# Executed results', '',
             f'Input: {len(df)} course-snapshot launches; {df.Date.min().date()} to {df.Date.max().date()}.',
             f'Course target: {df.Class.sum()} successes, {len(df)-df.Class.sum()} non-successes.', '',
             'Models are selected by training-only five-fold stratified CV balanced accuracy.',
             'The held-out test set is evaluated only after selection, once per declared split.', '',
             '| Evaluation | Selected model | Test n | Accuracy | Balanced accuracy | Baseline accuracy |',
             '|---|---|---:|---:|---:|---:|']
    for label, result in [('Random holdout', random), ('Chronological sensitivity', temporal)]:
        lines.append(f"| {label} | {result['selected_model']} | {result['test_n']} | "
                     f"{result['selected_test']['accuracy']:.3f} | "
                     f"{result['selected_test']['balanced_accuracy']:.3f} | "
                     f"{result['baseline_test']['accuracy']:.3f} |")
    lines += ['', '## Interpretation and limits', '',
              '- The selected SVM correctly classified 14/18 random-holdout launches, including all 12 successes but only 2/6 non-successes. Accuracy alone hides this asymmetry.',
              '- Small historical educational datasets; these are not deployment performance estimates.',
              '- Random splits mix historical eras. The chronological check exposes time-shift sensitivity; its training CV remains stratified, not rolling-origin forecasting.',
              '- Class 1 includes controlled ocean landings. Class 0 includes no-attempt/unknown outcomes; this is not a pure attempted-landing failure label.',
              '- No causal effects or launch-cost savings are established.',
              '- PayloadMass is upstream-imputed and excluded from modeling. Cumulative ReusedCount and target-derived Outcome are also excluded.',
              '- Block, Flights, and hardware attributes are assumed available before launch; retrospective source revisions can still create point-in-time bias.',
              '- JSON files record class counts, flight IDs, parameters, confusion matrices, and runtime versions.',
              '- The original presentation remains historical course work; it is not the source of these newly executed metrics.', '']
    (output / 'SUMMARY.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:14]))


if __name__ == '__main__':
    main()
