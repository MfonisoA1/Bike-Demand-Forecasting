# Bike-Demand-Forecasting

Objective: Predicting hourly bike rental demand from time, calendar, and weather features using XGBoost, evaluated with a time-based split against a naive baseline.

I previously worked through a tutorial-based retail demand forecasting pipeline; this project is my independent rebuild on a new dataset, with a focus on fixing the biggest weakness of that first version — random train/test splitting on time-series data.

## Data

[UCI Bike Sharing Dataset](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) — 17,379 hourly records of bike rentals in Washington, D.C. (2011–2012), with calendar features (hour, working day, holiday, season) and weather features (temperature, humidity, windspeed, weather situation). Target: `cnt`, the total rentals in each hour.

The dataset is loaded directly from the UCI repository in the notebook, so no data files are committed to this repo.

## Exploratory Findings

1. [Two rider populations] Working day demand shows sharp commuter spikes around 8am and between 5 or 6pm, while non working day demand is a single broad midday peak. Plotting `casual` vs `registered` riders by hour confirms it: registered riders drive the high commute spikes, casual riders the weekend leisure curve.
2. [Temperature helps until it doesn't] Demand climbs with temperature but flattens at the hottest bins rather than rising forever, so the relationship is nonlinear.
3. [Demand is hour dominated] The shape of demand across the day is the strongest pattern in the data, and it changes depending on day type — an interaction a tree-based model is well suited to capture.

## Why a Time-Based Split

Demand data is a time series. A random split would train the model on hours from the future and test it on the past, leaking information and inflating scores in a way that wouldn't survive real deployment. Instead, the data is sorted chronologically and the first 80% of the timeline is used for training, the final 20% for testing: making sure the model is always evaluated on a future it never saw.

## Leakage Check

The dataset includes `casual` and `registered` rider counts, which sum exactly to the target. These are outcomes measured after each hour happens, they would not exist at prediction time. so they are excluded from the feature set and used only for exploratory analysis. Every feature used by the model (hour, working day, holiday, season, weather, temperature, humidity, windspeed) is knowable in advance.

## Results

| Model | Test RMSE |
|---|---|
| Naive baseline (predict training mean) | 232.61 |
| XGBoost (default parameters) | **120.20** |

The model cuts error roughly in half relative to the baseline. The baseline is notably weak here for an instructive reason: the test period (late 2012) has substantially higher demand than the training period, because the bike program grew over time. The future genuinely differs from the past — which is exactly why the honest time based evaluation matters.

## Next Steps

- Hyperparameter tuning (randomized search with time-series-aware cross-validation)
- Lag and rolling features (e.g., demand at the same hour the previous day/week) to capture the growth trend the current model misses
- Test an engineered rush-hour flag against the baseline feature set
- Quantify the year-over-year growth effect (e.g., include `yr`/month features and compare)
- Deploy as a Streamlit app for interactive predictions



## Acknowledgments

Dataset: Fanaee-T & Gama, UCI Machine Learning Repository. The original pipeline skeleton (encode → split → train → evaluate) was adapted from an online XGBoost tutorial and extended here with time-based validation, a leakage analysis, and baseline comparison.
