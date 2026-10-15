# SpaceX Falcon 9 Data Science Capstone

IBM Data Science Capstone project analyzing Falcon 9 launches, first-stage landing outcomes, launch-site geography, and machine-learning classification.

## Project Goal

The project investigates which factors are associated with successful Falcon 9 first-stage landings and builds classification models to predict whether a landing will succeed.

## Workflow

1. Collect launch data using the SpaceX REST API.
2. Scrape Falcon 9 launch history from Wikipedia.
3. Clean and wrangle the collected data.
4. Perform exploratory data analysis with Python visualizations.
5. Perform exploratory data analysis with SQL.
6. Analyze launch-site geography with Folium.
7. Build an interactive Plotly Dash dashboard.
8. Train and evaluate classification models.

## Repository Structure

```text
notebooks/
  01_spacex_api_data_collection.ipynb
  02_web_scraping.ipynb
  03_data_wrangling.ipynb
  04_eda_visualization.ipynb
  05_eda_sql.ipynb
  06_folium_launch_sites.ipynb
  07_machine_learning_prediction.ipynb
dashboard/
  spacex_dash_app.py
data/
presentation/
requirements.txt
```

## Main Technologies

Python, Pandas, NumPy, Matplotlib, Seaborn, BeautifulSoup, Requests, SQLite, Folium, Plotly, Dash, and scikit-learn.

## Analysis Topics

- Flight number vs. launch site
- Payload mass vs. launch site
- Success rate by orbit type
- Flight number vs. orbit type
- Payload mass vs. orbit type
- Yearly landing-success trend
- SQL analysis of launch sites, payloads, boosters, and landing outcomes
- Launch-site proximity analysis with Folium
- Interactive launch dashboard with Plotly Dash
- Classification using Logistic Regression, SVM, Decision Tree, and KNN

## Capstone Presentation

The final report is titled **Winning Space Race with Data Science** and summarizes the complete workflow, EDA findings, interactive analytics, and predictive modeling results.

## Author

Ali El Maghraby
