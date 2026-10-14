import pandas as pd
from dash import Dash, dcc, html, Input, Output
import plotly.express as px

DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/jupyterlite/coursera/IBM-DS0321EN-SkillsNetwork/labs/module_3/spacex_launch_dash.csv"

df = pd.read_csv(DATA_URL)

app = Dash(__name__)
app.title = "SpaceX Falcon 9 Launch Dashboard"

sites = sorted(df["Launch Site"].dropna().unique())

app.layout = html.Div([
    html.H1("SpaceX Falcon 9 Launch Records Dashboard", style={"textAlign": "center"}),
    dcc.Dropdown(
        id="site-dropdown",
        options=[{"label": "All Sites", "value": "ALL"}] + [
            {"label": s, "value": s} for s in sites
        ],
        value="ALL",
        clearable=False,
    ),
    dcc.RangeSlider(
        id="payload-slider",
        min=0,
        max=10000,
        step=1000,
        marks={i: str(i) for i in range(0, 10001, 2000)},
        value=[0, 10000],
    ),
    dcc.Graph(id="success-pie-chart"),
    dcc.Graph(id="success-payload-scatter-chart"),
])

@app.callback(
    Output("success-pie-chart", "figure"),
    Output("success-payload-scatter-chart", "figure"),
    Input("site-dropdown", "value"),
    Input("payload-slider", "value"),
)
def update_dashboard(site, payload_range):
    filtered = df[(df["Payload Mass (kg)"] >= payload_range[0]) & (df["Payload Mass (kg)"] <= payload_range[1])]

    if site == "ALL":
        pie_data = filtered.groupby("class").size().reset_index(name="count")
        pie = px.pie(pie_data, values="count", names="class", title="Landing Outcomes — All Sites")
    else:
        filtered = filtered[filtered["Launch Site"] == site]
        pie_data = filtered.groupby("class").size().reset_index(name="count")
        pie = px.pie(pie_data, values="count", names="class", title=f"Landing Outcomes — {site}")

    scatter = px.scatter(
        filtered,
        x="Payload Mass (kg)",
        y="class",
        color="Booster Version Category",
        hover_data=["Launch Site"],
        title="Payload Mass vs. Landing Outcome",
    )
    return pie, scatter

if __name__ == "__main__":
    app.run(debug=True)
