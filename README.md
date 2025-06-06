# Tesla Stock Reversal Prediction via Velocity & Acceleration-Based ML

An applied machine learning project predicting momentum reversals in Tesla stock using 1st and 2nd derivatives of price (velocity and acceleration). The model output is visualized in both Python and Tableau for interpretability.

---

## Data Analyzed by  
**Aidan O'Halloran**  
[linkedin.com/in/aohallx](https://www.linkedin.com/in/aohallx/)  
[Live Tableau Dashboard](https://public.tableau.com/views/TeslaStockReversalPredictionAnalysis/Sheet1?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)

---

## Project Hypothesis

**Can we detect trend reversals in Tesla's stock price using mathematical features derived from historical data (velocity and acceleration)?**

---

## Dataset Overview

- Source: [Kaggle - Tesla Historical Stock Dataset](https://www.kaggle.com/datasets/muhammadatiflatif/tesla-complete-stocks-dataset)
- Date Range: Jan 2024 – May 2025
- Columns used:
  - `date`, `open`, `high`, `low`, `volume`, `adj_close` (used as `close`)

### Features Engineered (in SQL + Python):

- `velocity` = change in close price from previous day (1st derivative)
- `acceleration` = change in velocity (2nd derivative)
- `daily_change` = close − open (intra-day movement)
- `percent_change` = (close − open) / open × 100
- `range` = high − low
- `reversal_binary` = 1 when a momentum flip is detected (used as target)

---

## ML Pipeline (scikit-learn)

- **Model**: RandomForestClassifier  
- **Cross-Validation**: TimeSeriesSplit (5-fold)
- **Target**: reversal_binary (1 = reversal detected, 0 = normal trend)

### Classifier Evaluation

- Precision / Recall / F1-score reported across time-aware splits
- Red dots overlaid on price chart to show reversal predictions

---

## Visualizations

### Python (matplotlib):
- Line chart of Tesla’s adjusted close price
- Red dots plotted at reversal predictions
- All logic computed directly from cleaned Pandas DataFrame

### Tableau:
- [Live Dashboard Here](https://public.tableau.com/views/TeslaStockReversalPredictionAnalysis/Sheet1)
- Dual-axis chart:
  - Line: full close price over time
  - Dots: predicted reversal points
- Filtered to range from **January 5, 2024 – May 5, 2025**
- Tooltip shows reversal metadata (future addition: velocity, acceleration)

---

## Techniques Used

- SQL-based data preprocessing (with DuckDB & SQLite)
- Feature engineering via Pandas
- Outlier detection (IQR-based, optional for volatility)
- ML with scikit-learn (TimeSeriesSplit, classification report)
- Visual storytelling using Tableau dual-axis filtering
- Date-range slicing, calculated fields, reversal overlay logic

---

## Key Findings

- Velocity and acceleration provide effective signals for reversal classification
- Reversal predictions closely track true inflection points in Tesla stock
- Visualizations help bridge model logic and business interpretability

---

## Project Files

```bash
├── tesla_features_for_tableau.csv       # Cleaned + feature-engineered dataset
├── TeslaStockReversalPrediction.ipynb   # Jupyter Notebook (Python + ML pipeline)
├── Tesla_Stock_Analysis.twbx            # Tableau project (optional)
├── README.md
