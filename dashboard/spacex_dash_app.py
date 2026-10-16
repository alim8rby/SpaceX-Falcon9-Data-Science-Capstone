"""Offline dashboard. Run: python -m dashboard.spacex_dash_app."""
from pathlib import Path
import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

DATA_PATH = Path(__file__).resolve().parents[1] / 'data/raw/spacex_launch_dash.csv'
df = pd.read_csv(DATA_PATH)
sites = sorted(df['Launch Site'].dropna().unique())
minimum = float(df['Payload Mass (kg)'].min())
maximum = float(df['Payload Mass (kg)'].max())
app = Dash(__name__)
app.title = 'Falcon 9 Launch Analytics'
app.layout = html.Main([
    html.H1('Falcon 9 Launch Analytics'),
    html.P('IBM course dashboard dataset: historical sample, separate from the 90-launch modeling dataset.'),
    html.Label('Launch site', htmlFor='site-dropdown'),
    dcc.Dropdown(id='site-dropdown', options=[{'label': 'All sites', 'value': 'ALL'}] +
                 [{'label': s, 'value': s} for s in sites], value='ALL', clearable=False),
    html.P('Payload mass (kg)'),
    dcc.RangeSlider(id='payload-slider', min=minimum, max=maximum, step=100,
                    value=[minimum, maximum], tooltip={'placement': 'bottom'}),
    html.P(id='sample-summary'),
    dcc.Graph(id='success-pie-chart'),
    dcc.Graph(id='success-payload-scatter-chart'),
    html.P('Descriptive associations; no causal effects or financial savings are established.'),
], style={'maxWidth': '1050px', 'margin': '32px auto', 'fontFamily': 'Arial, sans-serif', 'padding': '16px'})


def empty_figure(title):
    fig = go.Figure()
    fig.add_annotation(text='No launches match these filters', showarrow=False)
    fig.update_layout(title=title, template='plotly_white')
    return fig


@app.callback(Output('success-pie-chart', 'figure'),
              Output('success-payload-scatter-chart', 'figure'),
              Output('sample-summary', 'children'),
              Input('site-dropdown', 'value'), Input('payload-slider', 'value'))
def update_dashboard(site, payload_range):
    low, high = payload_range
    filtered = df[df['Payload Mass (kg)'].between(low, high)].copy()
    if site != 'ALL':
        filtered = filtered[filtered['Launch Site'] == site]
    if filtered.empty:
        return empty_figure('Landing outcomes'), empty_figure('Payload and outcome'), '0 launches selected'
    counts = filtered['class'].value_counts().reindex([0, 1], fill_value=0)
    chart_data = pd.DataFrame({'Outcome': ['Non-success', 'Success'], 'Launches': counts.to_numpy()})
    outcomes = px.bar(chart_data, x='Outcome', y='Launches', text='Launches',
                       title='Landing outcomes in selected launches', template='plotly_white')
    outcomes.update_traces(marker_color='#245a81')
    outcomes.update_yaxes(rangemode='tozero')
    scatter = px.scatter(filtered, x='Payload Mass (kg)', y='class',
                         symbol='Booster Version Category', hover_data=['Launch Site'],
                         title='Payload mass and binary landing outcome', template='plotly_white')
    scatter.update_traces(marker={'color': '#245a81', 'size': 9, 'opacity': 0.7})
    scatter.update_yaxes(tickvals=[0, 1], ticktext=['Non-success', 'Success'], title='Landing outcome')
    summary = f"{len(filtered)} launches selected; {int(filtered['class'].sum())} successes ({filtered['class'].mean():.1%})"
    return outcomes, scatter, summary


if __name__ == '__main__':
    app.run(debug=False)
