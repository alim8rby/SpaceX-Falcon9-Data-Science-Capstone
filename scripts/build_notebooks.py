"""Regenerate the seven notebooks with a deterministic, inspectable cell structure."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SETUP = '''from pathlib import Path
import sys
ROOT = Path.cwd() if (Path.cwd() / 'falcon9').exists() else Path.cwd().parent
if not (ROOT / 'falcon9').exists():
    raise RuntimeError('Start Jupyter in the repository root or notebooks directory')
sys.path.insert(0, str(ROOT))
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from falcon9.analysis import load_data, group_summary
plt.rcParams.update({'figure.figsize': (9, 4.5), 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'axes.grid': False})
df = load_data()
'''


def write(name, title, intro, cells, takeaway):
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}
    notebook.cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n## tl;dr\n{intro}\n\n## Context & Methods\nIBM course project, reproduced from attributed historical snapshots. See `data/source_manifest.json` for URLs and hashes.\n\n### Key Assumptions\nCourse success labels include controlled ocean landings; descriptive associations are not causal effects.'),
                      nbf.v4.new_markdown_cell('## Data\nRun from a clean kernel. All ordinary analysis uses included source snapshots.'),
                      nbf.v4.new_code_cell(SETUP)]
    for kind, text in cells:
        notebook.cells.append(nbf.v4.new_markdown_cell(text) if kind == 'md' else nbf.v4.new_code_cell(text))
    notebook.cells.append(nbf.v4.new_markdown_cell('## Takeaways\n' + takeaway))
    nbf.validate(notebook)
    nbf.write(notebook, ROOT / 'notebooks' / name)


write('01_spacex_api_data_collection.ipynb', 'SpaceX Data Sources and Optional API Collection',
      'The repeatable analysis uses the included IBM normalized course snapshot. Live API collection is optional and is not claimed as validated by this saved run.', [
('md', '### Source provenance'),
('code', "import hashlib, json\nmanifest = json.loads((ROOT / 'data/source_manifest.json').read_text())\nfor item in manifest:\n    assert hashlib.sha256((ROOT / item['file']).read_bytes()).hexdigest() == item['sha256']\ndisplay(pd.DataFrame(manifest)[['file', 'url', 'bytes']])\nprint('All cached source hashes verified')"),
('md', '## Results\n### Inspect the normalized launch snapshot'),
('code', "display(df[['FlightNumber','Date','LaunchSite','Orbit','Outcome']].head())\nprint(f'{len(df)} unique launches; {df.Date.min().date()} to {df.Date.max().date()}')"),
('md', '### Optional live API collection\nSet `REFRESH_API=True` to request raw launch records. Raw IDs require rocket/payload/core enrichment before comparison with the normalized course snapshot. This notebook does not claim that a fresh raw response recreates the historical dataset.'),
('code', "from falcon9.collection import collect_api\nREFRESH_API = False\nif REFRESH_API:\n    raw_launches = collect_api()\n    target = ROOT / 'data/live'\n    target.mkdir(parents=True, exist_ok=True)\n    raw_launches.to_json(target / 'api_launches.json', orient='records', indent=2)\n    display(raw_launches[['id','flight_number','date_utc','rocket']].head())\nelse:\n    print('Live API: NOT RUN. Analysis source: IBM cached normalized CSV, not a fresh API extraction.')")],
'Cached source integrity is verified. The live SpaceX service timed out during maintenance; rerun the optional cell in a network-enabled environment to validate live collection. No refreshed API dataset is substituted silently.')

write('02_web_scraping.ipynb', 'Historical Course Tables and Optional Wikipedia Scraping',
      'The included IBM geographic reference table is inspectable offline. The live scraper is implemented separately; no successful live scrape is claimed in this saved notebook.', [
('md', '### Inspect the course geographic reference table\nThis course-provided CSV is not represented as HTML scraped during this run.'),
('code', "reference = pd.read_csv(ROOT / 'data/raw/spacex_launch_geo.csv')\ndisplay(reference.head(4))\nprint('Reference shape:', reference.shape)"),
('md', '## Results\n### Optional schema-aware HTML extraction\nThe collector selects tables by flight, payload, and launch column names rather than a table index. Upstream schema errors fail explicitly. Wikipedia and the normalized API-derived snapshot have different grains and scopes and are not joined without reconciliation.'),
('code', "from falcon9.collection import collect_wikipedia\nREFRESH_WIKIPEDIA = False\nif REFRESH_WIKIPEDIA:\n    tables = collect_wikipedia()\n    target = ROOT / 'data/live'\n    target.mkdir(parents=True, exist_ok=True)\n    for number, table in enumerate(tables):\n        table.to_csv(target / f'wikipedia_table_{number}.csv', index=False)\n    print('Extracted tables:', len(tables))\n    display(tables[0].head())\nelse:\n    print('Live HTML collection: NOT RUN. Course reference CSV inspection only.')")],
'The live Wikipedia request timed out during maintenance. Enable the refresh cell to verify the upstream page when reachable. Offline analytical and modeling findings do not depend on a successful scrape.')

write('03_data_wrangling.ipynb', 'Data Quality and Landing Labels',
      'Validate launch keys and known outcomes, derive the binary target, and distinguish missing landing-pad information from missing labels.', [
('md', '### Validate row grain and source missingness'),
('code', "raw = pd.read_csv(ROOT / 'data/raw/dataset_part_1.csv')\ndisplay(raw.isna().sum().rename('missing_rows').to_frame())\nassert df.FlightNumber.is_unique\nassert len(df) == len(raw)\nprint('Unique launches:', len(df))"),
('md', '## Results\n### Audit label mapping'),
('code', "from falcon9.analysis import OUTCOMES\naudit = df.groupby('Outcome').agg(launches=('Class','size'), Class=('Class','first'))\ndisplay(audit)\nprovided = pd.read_csv(ROOT / 'data/raw/dataset_part_2.csv')\njoined = df[['FlightNumber','Class']].merge(provided[['FlightNumber','Class']], on='FlightNumber', validate='one_to_one', suffixes=('_derived','_course'))\nassert len(joined) == len(df)\nassert joined.Class_derived.equals(joined.Class_course)\nprint('Derived labels match the course snapshot for all launches.')"),
('md', '### Class balance'),
('code', "counts = df.Class.value_counts().reindex([0,1])\nax = counts.plot.bar(color='#245a81', rot=0)\nax.set_xticklabels(['Non-success','Success'])\nax.set(title='Course landing target: historical launch counts', xlabel='Target class', ylabel='Launches')\nfor i, value in enumerate(counts): ax.text(i, value+0.5, str(value), ha='center')\nplt.tight_layout(); plt.show()"),
('md', '### Export the validated analytical table'),
('code', "target = ROOT / 'data/processed'\ntarget.mkdir(exist_ok=True)\ndf.to_csv(target / 'launches.csv', index=False)\nprint('Saved data/processed/launches.csv')")],
'Class 0 includes unknown/no-attempt outcomes; Class 1 includes controlled ocean landings. PayloadMass is already imputed upstream, so it is retained for descriptive analysis but excluded from primary modeling. Cumulative ReusedCount, target-derived Outcome, and identifiers are also excluded from predictors.')

write('04_eda_visualization.ipynb', 'Exploratory Launch Analysis',
      'Compare historical outcomes across launch sites and years, and inspect payload associations with explicit sample sizes.', [
('md', '## Results\n### Site success rates and denominators'),
('code', "sites = group_summary(df, 'LaunchSite').sort_values('success_rate')\ndisplay(sites)\nax = sites.plot.barh(x='LaunchSite', y='success_rate', color='#245a81', legend=False)\nax.set(xlim=(0,1), xlabel='Course success proportion', ylabel='Launch site', title='Landing outcomes by launch site')\nfor i, row in enumerate(sites.itertuples()): ax.text(row.success_rate+0.01, i, f'n={row.launches}', va='center')\nplt.tight_layout(); plt.show()"),
('md', '### Yearly rates and observation counts\nA changing mix of rockets, missions, and landing attempts can explain associations; these charts do not isolate a site or year effect.'),
('code', "yearly = group_summary(df, 'Year')\ndisplay(yearly)\nfig, axes = plt.subplots(2,1,figsize=(9,6),sharex=True)\naxes[0].plot(yearly.Year, yearly.success_rate, marker='o', color='#245a81')\naxes[0].set(ylabel='Success proportion', ylim=(0,1.05), title='Historical landing outcomes by year')\naxes[1].bar(yearly.Year, yearly.launches, color='#245a81')\naxes[1].set(xlabel='Year', ylabel='Launches')\naxes[1].set_xticks(yearly.Year)\nplt.tight_layout(); plt.show()"),
('md', '### Payload mass and target\nSome payload values were mean-imputed by the course upstream. Treat the pattern as descriptive, not an unbiased missingness analysis.'),
('code', "fig, ax = plt.subplots()\nfor label, marker in [(0,'x'),(1,'o')]:\n    subset = df[df.Class == label]\n    ax.scatter(subset.FlightNumber, subset.PayloadMass, marker=marker, color={0:'#245a81',1:'#ba6c2d'}[label], alpha=.7, label=f'Class {label} (n={len(subset)})')\nax.set(xlabel='Historical flight sequence', ylabel='Payload mass (kg)', title='Payload mass across historical launches')\nax.legend(); plt.tight_layout(); plt.show()"),
('md', '### Orbit summary'),
('code', "display(group_summary(df, 'Orbit').sort_values('launches', ascending=False))")],
'See the displayed tables for exact denominators. Site, orbit, payload, and era are confounded; small groups do not support stable rankings or causal claims. Future launch performance cannot be inferred directly from historical aggregate rates.')

write('05_eda_sql.ipynb', 'SQL Exploration with SQLite',
      'Execute SQL summaries against the same validated launch table and reconcile counts with Pandas.', [
('md', '## Results\n### Inspect and execute SQL'),
('code', "from falcon9.analysis import SQL_QUERIES, sql_analysis\noutputs = sql_analysis(df)\nfor name, result in outputs.items():\n    print(name, '\\n', SQL_QUERIES[name])\n    display(result)"),
('md', '### Reconcile independent implementations'),
('code', "assert outputs['site_summary'].launches.sum() == len(df)\nassert outputs['site_summary'].successes.sum() == df.Class.sum()\nexpected = group_summary(df, 'LaunchSite').sort_values('LaunchSite').reset_index(drop=True)\nactual = outputs['site_summary'].sort_values('LaunchSite').reset_index(drop=True)\npd.testing.assert_series_equal(actual.launches, expected.launches, check_names=False)\nprint('SQL launch and success counts reconcile with Pandas.')")],
'SQL and Pandas summaries use the same launch-level population. Payload means inherit the course snapshot imputation; the table does not support customer-specific queries because customer is absent from this source schema.')

write('06_folium_launch_sites.ipynb', 'Launch-Site Geography with Folium',
      'Map actual snapshot coordinates and summarize the launches associated with each site. No unsourced proximity distances are asserted.', [
('md', '## Results\n### Validate and map coordinates'),
('code', "import folium\nassert df.Latitude.between(-90,90).all() and df.Longitude.between(-180,180).all()\ncoordinates = df.groupby('LaunchSite').agg(latitude=('Latitude','first'), longitude=('Longitude','first'), launches=('Class','size'), successes=('Class','sum')).reset_index()\ndisplay(coordinates)\nlaunch_map = folium.Map(location=[31,-92], zoom_start=4, tiles='OpenStreetMap')\nfrom folium.plugins import MarkerCluster\ncluster = MarkerCluster().add_to(launch_map)\nfor row in coordinates.itertuples():\n    label = f'{row.LaunchSite}: {row.successes}/{row.launches} course successes'\n    folium.Marker([row.latitude,row.longitude], tooltip=label, popup=label).add_to(cluster)\n(ROOT / 'results').mkdir(exist_ok=True)\nlaunch_map.save(ROOT / 'results/launch_sites.html')\ndisplay(launch_map)"),
('md', '### Static coordinate view for portable inspection\nThis scatterplot is a coordinate view, not a basemap. Folium map tiles require internet access.'),
('code', "fig, ax = plt.subplots()\nax.scatter(coordinates.longitude, coordinates.latitude, color='#245a81', s=70)\noffsets = {'CCAFS SLC 40': (-120,-25), 'KSC LC 39A': (-120,25), 'VAFB SLC 4E': (7,7)}\nfor row in coordinates.itertuples():\n    ax.annotate(row.LaunchSite, (row.longitude,row.latitude), xytext=offsets[row.LaunchSite], textcoords='offset points', arrowprops={'arrowstyle':'-', 'color':'#666666'})\nax.set(xlabel='Longitude (degrees)', ylabel='Latitude (degrees)', title='Launch-site coordinates from the course snapshot', xlim=(-125,-75), ylim=(25,38))\nplt.tight_layout(); plt.show()")],
'Coordinates are historical course source values. No railway, highway, or coastline distance is claimed without a sourced reference geometry and a declared measurement method.')

write('07_machine_learning_prediction.ipynb', 'Falcon 9 Landing Classification',
      'Compare four classifiers using training-only preprocessing and cross-validation. Select a model before evaluating its held-out test set, and compare with a majority-class baseline.', [
('md', '### Modeling contract\nPredictors: flight sequence, prior flight count, block, orbit, site, grid fins, reused status, legs. Exclude upstream-imputed payload, cumulative ReusedCount, outcome-derived columns and serial/landing identifiers. Random splitting is the main educational comparison; a separate chronological split is a sensitivity check. Neither is production validation.'),
('code', "from falcon9.analysis import FEATURES, run_experiment\ndisplay(df[FEATURES].head())\nprint('Target counts:', df.Class.value_counts().to_dict())"),
('md', '## Results\n### Train, tune, select, then evaluate\nEach classifier has a full preprocessing pipeline fitted inside five-fold training CV. Selection uses CV balanced accuracy. Only the selected model and the baseline are scored on each held-out set.'),
('code', "random_result, random_predictions = run_experiment(df)\ncomparison = pd.DataFrame(random_result['cv_comparison'])\ndisplay(comparison[['model','cv_balanced_accuracy','cv_std','parameters']])\nprint('Selected:', random_result['selected_model'])\ndisplay(pd.DataFrame({'Selected': random_result['selected_test'], 'Baseline': random_result['baseline_test']}).drop(index='confusion_matrix'))"),
('md', '### Cross-validation comparison\nError bars show fold-score standard deviation, not confidence intervals.'),
('code', "ax = comparison.plot.barh(x='model', y='cv_balanced_accuracy', xerr='cv_std', color='#245a81', legend=False)\nax.set(xlim=(0,1), xlabel='Training CV balanced accuracy (mean ± fold SD)', ylabel='Model', title='Training-only classifier comparison')\nplt.tight_layout(); plt.show()"),
('md', '### Held-out confusion matrix and individual errors'),
('code', "from sklearn.metrics import ConfusionMatrixDisplay\nConfusionMatrixDisplay(__import__('numpy').array(random_result['selected_test']['confusion_matrix']), display_labels=['Non-success','Success']).plot(cmap='Blues', colorbar=False)\nplt.title(f\"Selected {random_result['selected_model']}: {random_result['test_n']} held-out launches\")\nplt.tight_layout(); plt.show()\ndisplay(random_predictions)"),
('md', '### Chronological sensitivity\nTrain on the first 80% of launches, evaluate the last 20%. Training-only CV is stratified and shuffled, not rolling-origin forecasting. Report the temporal test class distribution because class imbalance changes interpretation.'),
('code', "temporal_result, temporal_predictions = run_experiment(df, temporal=True)\nprint('Temporal test classes:', temporal_result['test_class_counts'])\ndisplay(pd.DataFrame([{'split': r['split'], 'selected': r['selected_model'], 'test_n': r['test_n'], 'accuracy': r['selected_test']['accuracy'], 'balanced_accuracy': r['selected_test']['balanced_accuracy'], 'baseline_accuracy': r['baseline_test']['accuracy']} for r in [random_result,temporal_result]]))"),
('md', '### Save exact experiment evidence'),
('code', "import json\nfor name, result, predictions in [('random',random_result,random_predictions),('temporal',temporal_result,temporal_predictions)]:\n    (ROOT / f'results/{name}_metrics.json').write_text(json.dumps(result, indent=2)+'\\n')\n    predictions.to_csv(ROOT / f'results/{name}_predictions.csv', index=False)\nprint('Saved parameters, splits, environment versions, metrics, and predictions.')")],
'The comparison and metrics above are measured outputs, not course-report figures copied into notebooks. A test set of 18 launches yields unstable estimates. Random and chronological results answer different questions; report both. Do not claim causal effects, modern launch forecasting accuracy, or recovered-booster economics.')

if __name__ == '__main__':
    print('Generated seven notebooks; execute with python -m scripts.verify_notebooks')
