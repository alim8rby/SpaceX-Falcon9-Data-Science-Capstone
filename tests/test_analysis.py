import hashlib
import json
import unittest
from unittest.mock import patch
from io import StringIO
import pandas as pd
from falcon9.analysis import ROOT, FEATURES, load_data, prepare_data, make_pipeline, sql_analysis
from falcon9.collection import extract_launch_tables
from sklearn.linear_model import LogisticRegression


class DataAndValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = load_data()

    def test_snapshot_integrity(self):
        for source in json.loads((ROOT / 'data/source_manifest.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT / source['file']).read_bytes()).hexdigest(), source['sha256'])

    def test_unknown_outcomes_fail(self):
        bad = self.df.copy()
        bad.loc[0, 'Outcome'] = 'unknown label'
        with self.assertRaises(ValueError):
            prepare_data(bad)

    def test_duplicate_launches_fail(self):
        with self.assertRaises(ValueError):
            prepare_data(pd.concat([self.df, self.df.iloc[[0]]]))

    def test_derived_labels_match_course(self):
        reference = pd.read_csv(ROOT / 'data/raw/dataset_part_2.csv')
        expected = reference.set_index('FlightNumber').Class.sort_index()
        actual = self.df.set_index('FlightNumber').Class.sort_index()
        pd.testing.assert_series_equal(actual, expected)

    def test_sql_reconciles(self):
        sites = sql_analysis(self.df)['site_summary']
        self.assertEqual(sites.launches.sum(), len(self.df))
        self.assertEqual(sites.successes.sum(), self.df.Class.sum())

    def test_pipeline_ignores_extreme_unseen_test_values(self):
        train = self.df.iloc[:60][FEATURES]
        pipeline = make_pipeline(LogisticRegression(max_iter=2000)).fit(train, self.df.iloc[:60].Class)
        scaler = pipeline.named_steps['preprocess'].named_transformers_['numeric'].named_steps['scale']
        self.assertAlmostEqual(scaler.mean_[0], train.FlightNumber.mean())
        test = self.df.iloc[60:][FEATURES].copy()
        test['FlightNumber'] = 1000000
        test['Orbit'] = 'unseen orbit'
        pipeline.predict(test)
        self.assertAlmostEqual(scaler.mean_[0], train.FlightNumber.mean())

    def test_saved_splits_are_disjoint_and_exclude_leaky_columns(self):
        for name in ['random','temporal']:
            result = json.loads((ROOT / f'results/{name}_metrics.json').read_text())
            self.assertFalse(set(result['train_flights']) & set(result['test_flights']))
            self.assertEqual(result['train_n'] + result['test_n'], len(self.df))
            self.assertFalse({'Outcome','Class','ReusedCount','PayloadMass'} & set(result['features']))
            predictions = pd.read_csv(ROOT / f'results/{name}_predictions.csv')
            self.assertEqual(len(predictions), result['test_n'])
            self.assertAlmostEqual((predictions.actual == predictions.prediction).mean(), result['selected_test']['accuracy'])
            if name == 'temporal':
                self.assertLess(max(result['train_flights']), min(result['test_flights']))

    def test_html_table_selection_synthetic_fixture(self):
        # Tests parsing logic only; this is not evidence of a successful live scrape.
        html = '<table><tr><th>Flight</th><th>Launch site</th><th>Payload</th></tr><tr><td>1</td><td>Demo</td><td>100</td></tr></table>'
        self.assertEqual(len(extract_launch_tables(html)), 1)
        with self.assertRaises(ValueError):
            extract_launch_tables('<table><tr><th>Other</th></tr><tr><td>1</td></tr></table>')


class DashboardTests(unittest.TestCase):
    def test_filters_and_empty_results(self):
        from dashboard.spacex_dash_app import update_dashboard, df, minimum, maximum, sites, app
        bar, scatter, text = update_dashboard('ALL', [minimum, maximum])
        self.assertEqual(sum(bar.data[0].y), len(df))
        bar, _, text = update_dashboard(sites[0], [minimum, maximum])
        self.assertEqual(sum(bar.data[0].y), int((df['Launch Site'] == sites[0]).sum()))
        bar, _, text = update_dashboard('ALL', [maximum+1, maximum+2])
        self.assertEqual(text, '0 launches selected')
        self.assertEqual(app.server.test_client().get('/').status_code, 200)
        self.assertEqual(app.server.test_client().get('/_dash-layout').status_code, 200)

if __name__ == '__main__':
    unittest.main()
