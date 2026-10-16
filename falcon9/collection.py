"""Optional live collectors; cached analysis does not require these services."""
from io import StringIO
import pandas as pd
import requests

API_URL = 'https://api.spacexdata.com/v4/launches/past'
WIKI_URL = ('https://en.wikipedia.org/w/index.php?title='
            'List_of_Falcon_9_and_Falcon_Heavy_launches&oldid=1027686922')


def collect_api(url=API_URL):
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    records = response.json()
    if not isinstance(records, list) or not records:
        raise ValueError('Expected a nonempty list of launch records')
    frame = pd.json_normalize(records)
    if not {'id', 'flight_number', 'date_utc', 'rocket'}.issubset(frame.columns):
        raise ValueError('Unexpected SpaceX API schema')
    return frame


def extract_launch_tables(html):
    tables = pd.read_html(StringIO(html))
    selected = []
    for table in tables:
        table.columns = [' '.join(str(v) for v in c if str(v) != 'nan')
                         if isinstance(c, tuple) else str(c) for c in table.columns]
        columns = ' '.join(table.columns).lower()
        if 'flight' in columns and 'payload' in columns and 'launch' in columns:
            selected.append(table)
    if not selected:
        raise ValueError('No launch tables found; inspect upstream page schema')
    return selected


def collect_wikipedia(url=WIKI_URL):
    response = requests.get(url, timeout=30,
                            headers={'User-Agent': 'Falcon9EducationalCapstone/1.0'})
    response.raise_for_status()
    return extract_launch_tables(response.text)
