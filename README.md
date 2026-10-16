# SpaceX Falcon 9 Data Science Capstone

A reproducible IBM Data Science capstone exploring historical Falcon 9 launches with Python, SQL, interactive visualization, and classification.

**Question:** Which launch characteristics are associated with the course's first-stage landing-success label, and how well do simple classifiers generalize to held-out historical launches?

## What is complete

- Attributed IBM snapshots with source URLs, retrieval timestamps, and SHA-256 checksums.
- Launch-level validation, missingness checks, explicit target mapping, and reconciliation with course labels.
- Executed Python EDA and SQLite analysis with saved figures and tables in notebooks.
- Folium launch-site map and an offline-first Plotly Dash dashboard.
- Four classifier families: logistic regression, SVM, decision tree, and KNN.
- Train-fold preprocessing, training-only hyperparameter selection, a majority baseline, and a chronological sensitivity check.
- Saved split IDs, selected hyperparameters, metrics, confusion matrices, and predictions.

**Collection boundary:** optional live API and Wikipedia collectors are implemented, but live requests timed out during this maintenance run. Their notebook refresh flags are off by default and explicitly print `NOT RUN`. Executed analysis uses the included IBM course snapshots, not a claimed fresh scrape or a newly reconstructed API dataset.

## Results

The analytical snapshot contains **90 launches from June 2010 to November 2020**, with 60 course successes and 30 non-successes.

| Evaluation | Selected by training CV | Test launches | Accuracy | Balanced accuracy | Majority baseline accuracy |
|---|---|---:|---:|---:|---:|
| Stratified random holdout | SVM | 18 | 77.8% | 66.7% | 66.7% |
| Chronological sensitivity | Decision tree | 18 | 88.9% | 66.7% | 83.3% |

See [executed results and limitations](results/SUMMARY.md) and [model notebook](notebooks/07_machine_learning_prediction.ipynb). These are newly executed results; the original course presentation's figures are retained as historical material, not substituted as current evidence.

## Quick start

Tested with Python 3.12. A fresh virtual environment is recommended.

```bash
git clone https://github.com/alim8rby/SpaceX-Falcon9-Data-Science-Capstone.git
cd SpaceX-Falcon9-Data-Science-Capstone
python -m venv .venv
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Install and reproduce:

```bash
python -m pip install -r requirements.txt
python -m scripts.run_analysis
python -m unittest discover -s tests -v
python -m scripts.verify_notebooks
```

The notebook verifier executes clean kernels and exports local HTML previews to `results/notebook_previews/`. If the environment disallows Jupyter socket binding, use the verified alternative:

```bash
python -m scripts.verify_notebooks --in-process
```

That option uses a separate IPython subprocess per notebook, executes cells in order, and captures actual text, table, and figure outputs. It does not simulate notebook results. The maintenance environment required this mode because both TCP and IPC kernel sockets were unavailable.

Run the dashboard:

```bash
python -m dashboard.spacex_dash_app
```

Open **http://127.0.0.1:8050**. Filter launch sites and payload ranges; empty selections are handled. The dashboard uses a separate 56-launch IBM course dataset and must not be conflated with the 90-launch modeling population. It reads local data and does not require a remote CSV at startup.

## Notebook guide

| Notebook | Purpose |
|---|---|
| [01](notebooks/01_spacex_api_data_collection.ipynb) | Source integrity, normalized snapshot inspection, optional raw API collection |
| [02](notebooks/02_web_scraping.ipynb) | Course reference table inspection and optional schema-aware HTML scraping |
| [03](notebooks/03_data_wrangling.ipynb) | Data quality, missingness, target mapping, course-label reconciliation |
| [04](notebooks/04_eda_visualization.ipynb) | Site, year, orbit, and payload exploration with denominators |
| [05](notebooks/05_eda_sql.ipynb) | Executed SQLite queries reconciled with Pandas |
| [06](notebooks/06_folium_launch_sites.ipynb) | Sourced coordinates, interactive map, static coordinate view |
| [07](notebooks/07_machine_learning_prediction.ipynb) | Training-only model selection, baseline, holdout results, temporal sensitivity |

## Modeling decisions

- Split the data **before fitting preprocessing**. Imputation, scaling, and one-hot encoding stay inside a scikit-learn `Pipeline`/`ColumnTransformer` fitted within each CV fold.
- Compare models by five-fold stratified training CV **balanced accuracy**. Freeze the selection before evaluating the holdout; do not select the model with the best test accuracy.
- Primary features: `FlightNumber`, `Flights`, `Block`, `Orbit`, `LaunchSite`, `GridFins`, `Reused`, `Legs`.
- Exclude `Outcome`/`Class` leakage, identifiers, cumulative `ReusedCount`, and upstream-imputed `PayloadMass` from modeling. Payload remains descriptive only.
- `OneHotEncoder(handle_unknown='ignore')` supports unseen test categories.
- Report accuracy, balanced accuracy, precision, recall, F1, confusion matrices, and exact test predictions, plus a majority-class baseline.
- The chronological sensitivity trains on the first 80% and evaluates the last 20%. Its **internal training CV is still stratified**, so it is not a rolling-origin forecast validation.

## Interpretation limits

1. **Target semantics:** course success includes controlled ocean landings. Non-success includes no-attempt/unknown outcomes. This is not simply recovered booster versus failed attempted landing.
2. **Small sample:** each holdout has 18 launches. Results depend on the split, historical era, and class balance. Higher temporal accuracy does not imply better minority-class performance.
3. **Point-in-time uncertainty:** the source is retrospective. Hardware and prior-flight attributes are assumed available before launch; historical revisions may still introduce bias.
4. **Upstream preparation:** payload values were filled before this repository's analysis. The original missingness cannot be reconstructed from the snapshot.
5. **Descriptive scope:** site and orbit associations are confounded by time and mission mix. No causal, financial-savings, or operational deployment claim is made.
6. **Geography:** map coordinates are sourced; proximity distances to roads, railways, or coastlines are not claimed without reviewed reference geometry.
7. **Live collection:** optional collectors need validation against reachable upstream services and may need schema updates. Fresh raw records require enrichment and scope reconciliation before reuse in this historical analysis.

## Reproducibility and provenance

- [Data sources](data/README.md) and [checksum manifest](data/source_manifest.json)
- `falcon9/analysis.py`: data contracts, SQL, preprocessing, and experiments
- `scripts/run_analysis.py`: one-command reproduction and results summary
- `scripts/build_notebooks.py`: notebook source generator (regeneration clears outputs; execute afterward)
- `tests/test_analysis.py`: integrity, label, split, preprocessing, SQL, scraper-logic, and dashboard checks
- `requirements.txt`: exact top-level package versions used; transitive dependencies are not fully locked

## Attribution and authorship

Ali El Maghraby completed the IBM Data Science Professional Certificate in October 2026. This repository follows that course's SpaceX capstone and uses IBM/Skills Network educational snapshots. The October 2026 reproducibility upgrade was implemented with AI assistance. This is educational portfolio work, not professional SpaceX employment or a production forecasting system.

The repository's MIT license applies to original repository code. Third-party course datasets and the original course presentation retain their respective rights and attribution; inclusion does not relicense them under MIT. Upstream data origins include SpaceX and Wikipedia; see the source manifest and collector URLs.
