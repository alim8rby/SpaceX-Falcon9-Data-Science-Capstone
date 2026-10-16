"""Data contracts, descriptive analysis, and training-only model selection."""
from pathlib import Path
import json
import platform
import sqlite3

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score, confusion_matrix)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ['FlightNumber', 'Flights', 'Block']
CATEGORICAL = ['Orbit', 'LaunchSite', 'GridFins', 'Reused', 'Legs']
FEATURES = NUMERIC + CATEGORICAL
OUTCOMES = {'True ASDS': 1, 'True RTLS': 1, 'True Ocean': 1,
            'False ASDS': 0, 'False RTLS': 0, 'False Ocean': 0,
            'None ASDS': 0, 'None None': 0}


def load_data():
    return prepare_data(pd.read_csv(ROOT / 'data/raw/dataset_part_1.csv'))


def prepare_data(raw):
    """Retain source values; never infer success from unknown labels."""
    df = raw.copy()
    required = FEATURES + ['Date', 'Outcome', 'PayloadMass', 'Longitude', 'Latitude']
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.FlightNumber.isna().any() or df.FlightNumber.duplicated().any():
        raise ValueError('FlightNumber must be a non-null unique launch key')
    if df.Outcome.isna().any() or not set(df.Outcome).issubset(OUTCOMES):
        raise ValueError('Missing or unrecognized landing outcome')
    df['Date'] = pd.to_datetime(df.Date, errors='raise')
    df['Class'] = df.Outcome.map(OUTCOMES).astype(int)
    df['Year'] = df.Date.dt.year
    if (df.PayloadMass.dropna() < 0).any():
        raise ValueError('Payload mass cannot be negative')
    return df.sort_values(['Date', 'FlightNumber']).reset_index(drop=True)


def group_summary(df, column):
    return df.groupby(column, dropna=False).Class.agg(
        launches='size', successes='sum', success_rate='mean').reset_index()


SQL_QUERIES = {
    'site_summary': '''SELECT LaunchSite, COUNT(*) AS launches,
        SUM(Class) AS successes, ROUND(AVG(Class), 4) AS success_rate
        FROM launches GROUP BY LaunchSite ORDER BY launches DESC''',
    'year_summary': '''SELECT Year, COUNT(*) AS launches, SUM(Class) AS successes,
        ROUND(AVG(Class), 4) AS success_rate FROM launches GROUP BY Year ORDER BY Year''',
    'orbit_summary': '''SELECT Orbit, COUNT(*) AS launches,
        ROUND(AVG(Class), 4) AS success_rate FROM launches GROUP BY Orbit ORDER BY launches DESC''',
    'first_ground_success': '''SELECT MIN(Date) AS first_ground_success
        FROM launches WHERE Outcome = 'True RTLS' ''',
    'payload_summary': '''SELECT LaunchSite, COUNT(*) AS launches,
        ROUND(AVG(PayloadMass), 2) AS mean_payload_kg
        FROM launches GROUP BY LaunchSite ORDER BY launches DESC''',
}


def sql_analysis(df):
    with sqlite3.connect(':memory:') as connection:
        df.to_sql('launches', connection, index=False)
        return {name: pd.read_sql_query(query, connection)
                for name, query in SQL_QUERIES.items()}


def make_pipeline(model):
    # Every fitted operation stays inside each training fold.
    preprocessor = ColumnTransformer([
        ('numeric', Pipeline([('impute', SimpleImputer(strategy='median')),
                              ('scale', StandardScaler())]), NUMERIC),
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL),
    ])
    return Pipeline([('preprocess', preprocessor), ('model', model)])


def candidates():
    return {
        'Logistic regression': (LogisticRegression(max_iter=2000, random_state=42),
                                {'model__C': [0.1, 1, 10]}),
        'SVM': (SVC(), {'model__C': [0.1, 1, 10], 'model__kernel': ['linear', 'rbf']}),
        'Decision tree': (DecisionTreeClassifier(random_state=42),
                          {'model__max_depth': [2, 4, None],
                           'model__min_samples_leaf': [1, 3, 5]}),
        'KNN': (KNeighborsClassifier(), {'model__n_neighbors': [3, 5, 7],
                                       'model__weights': ['uniform', 'distance']}),
    }


def evaluate(model, X, y):
    predicted = model.predict(X)
    return {'accuracy': float(accuracy_score(y, predicted)),
            'balanced_accuracy': float(balanced_accuracy_score(y, predicted)),
            'precision': float(precision_score(y, predicted, zero_division=0)),
            'recall': float(recall_score(y, predicted, zero_division=0)),
            'f1': float(f1_score(y, predicted, zero_division=0)),
            'confusion_matrix': confusion_matrix(y, predicted, labels=[0, 1]).tolist()}


def run_experiment(df, temporal=False):
    X, y = df[FEATURES], df.Class
    if temporal:
        cut = int(len(df) * 0.8)
        train_idx, test_idx = np.arange(cut), np.arange(cut, len(df))
    else:
        train_idx, test_idx = train_test_split(
            np.arange(len(df)), test_size=0.2, random_state=42, stratify=y)
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    folds = min(5, int(y_train.value_counts().min()))
    if folds < 2:
        raise ValueError('Not enough training examples in both classes for stratified CV')
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    searches, rows = {}, []
    for name, (model, parameters) in candidates().items():
        search = GridSearchCV(make_pipeline(clone(model)), parameters, cv=cv,
                              scoring='balanced_accuracy', n_jobs=1, error_score='raise')
        search.fit(X_train, y_train)
        searches[name] = search
        rows.append({'model': name, 'cv_balanced_accuracy': float(search.best_score_),
                     'cv_std': float(search.cv_results_['std_test_score'][search.best_index_]),
                     'parameters': search.best_params_})
    # Freeze selection BEFORE looking at any test predictions.
    ranking = sorted(rows, key=lambda row: (-row['cv_balanced_accuracy'], row['model']))
    winner = ranking[0]['model']
    selected = searches[winner].best_estimator_
    baseline = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    results = {
        'split': 'chronological_last_20_percent' if temporal else 'stratified_random_80_20',
        'seed': 42, 'cv_folds': folds, 'selection_metric': 'cv_balanced_accuracy',
        'features': FEATURES, 'selected_model': winner, 'cv_comparison': ranking,
        'train_n': len(train_idx), 'test_n': len(test_idx),
        'train_flights': df.iloc[train_idx].FlightNumber.tolist(),
        'test_flights': df.iloc[test_idx].FlightNumber.tolist(),
        'train_class_counts': {str(k): int(v) for k,v in y_train.value_counts().items()},
        'test_class_counts': {str(k): int(v) for k,v in y_test.value_counts().items()},
        'selected_test': evaluate(selected, X_test, y_test),
        'baseline_test': evaluate(baseline, X_test, y_test),
        'environment': {'python': platform.python_version(), 'sklearn': sklearn.__version__,
                        'pandas': pd.__version__, 'numpy': np.__version__},
    }
    predictions = pd.DataFrame({'FlightNumber': df.iloc[test_idx].FlightNumber.to_numpy(),
                                'actual': y_test.to_numpy(),
                                'prediction': selected.predict(X_test)})
    return results, predictions


def save_results(df):
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    for key, temporal in [('random', False), ('temporal', True)]:
        results, predictions = run_experiment(df, temporal)
        (output / f'{key}_metrics.json').write_text(json.dumps(results, indent=2) + '\n')
        predictions.to_csv(output / f'{key}_predictions.csv', index=False)
    return output
