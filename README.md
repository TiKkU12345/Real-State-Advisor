# Real Estate Investment Advisor: Predicting Property Profitability and Future Value

Classifies whether a property is a **Good Investment** and predicts its **price after 5 years**, with a Streamlit app and MLflow experiment tracking.

Read the "Honest limitations" section before presenting results. The dataset is synthetic and that changes how every number below should be read.

## Project layout

| Path | Purpose |
| :--- | :--- |
| `src/config.py` | Paths, feature lists, growth and label assumptions |
| `src/preprocess.py` | Cleaning, feature engineering, label and future-price creation |
| `src/eda.py` | 20 EDA questions from the brief, figures and `reports/eda_findings.md` |
| `src/train.py` | Trains models, logs to MLflow, registers the best, saves `models/*.joblib` |
| `app.py` | Streamlit app (Predict, Property Search, Market Insights, Model Performance) |
| `data/processed/` | Cleaned dataset (`housing_processed.csv.gz`) and a 20,000-row sample used by the app |
| `reports/` | Cleaning report, EDA findings and 21 figures |
| `models/` | Deployed pipelines, metrics and reference statistics |
| `mlflow.db`, `mlruns/` | MLflow tracking store and registered models |

## How to run

```bash
pip install -r requirements.txt
# put india_housing_prices.csv in data/raw/
python -m src.preprocess     # clean + features + labels
python -m src.eda            # figures + findings
python -m src.train          # about 2 minutes, logs to MLflow
mlflow ui --backend-store-uri sqlite:///mlflow.db
streamlit run app.py
```

MLflow stores absolute artifact paths. If you move the folder, re-run `python -m src.train` to regenerate the tracking store.

## Methodology

**1. Preprocessing.** 250,000 rows, no missing values, no duplicates. In 46.5% of rows (116,304) `Floor_No` was higher than `Total_Floors`; the pair is swapped so floor is never above the building height. `Price_per_SqFt` was in lakhs per sq ft, so it is recomputed in rupees. Age is checked against `Year_Built`. Outliers are detected with the IQR rule and flagged (`Outlier_PPS`: 19,723 rows), not deleted, because they are valid listings. Numeric features are standardised and categoricals one-hot encoded inside the sklearn pipeline, so the app applies exactly the same transforms.

**2. Feature engineering.** Price per sq ft (Rs), School Density Score, Hospital Score, Transport Score, Infrastructure Score (0.3 schools + 0.3 hospitals + 0.4 transport), amenity flags and count, parking and security flags, floor ratio, size per BHK.

**3. Targets.**

* **Good Investment** (multi-factor rule, one of the three options in the brief). One point each for: price per sq ft at or below the city median, 3+ BHK, ready to move, infrastructure score of 60+, 3+ amenities, assumed growth of 8%+ per year. Score of 4 or more is Good. Result: 39.5% Good.
* **Future price** = current price x (1 + r)^5, where r is location and feature based (the third option in the brief): city tier base rate (9 / 7.5 / 6%), plus adjustments for property type, transport access, under-construction status, age and infrastructure, clipped to 4 to 13%. Mean r is 7.9%.

**4. Modelling.** Stratified 70 / 15 / 15 train / validation / test split. Models are compared on the validation set; the table below is the untouched test set. The final models are registered in the MLflow Model Registry with the alias `champion`.

## Results (test set, 37,500 listings)

**Classification** (classifier never sees price columns)

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.830 | 0.797 | 0.765 | 0.781 | 0.914 |
| Random Forest | 0.874 | 0.852 | 0.823 | 0.838 | 0.952 |
| **XGBoost (depth 6), deployed** | **0.896** | **0.866** | **0.871** | **0.868** | **0.970** |
| XGBoost (depth 4, slow) | 0.890 | 0.863 | 0.858 | 0.860 | 0.967 |
| XGBoost with price features (leakage demo, not deployed) | 0.993 | 0.991 | 0.992 | 0.991 | 1.000 |

Confusion matrix of the deployed model: 20,678 true negatives, 1,997 false positives, 1,918 false negatives, 12,907 true positives.

**Regression** (5-year price, in lakhs)

| Model | RMSE | MAE | R2 |
| :--- | ---: | ---: | ---: |
| Baseline: fixed 8% growth | 23.11 | 16.38 | 0.988 |
| Linear Regression | 11.19 | 7.85 | 0.997 |
| Random Forest | 14.16 | 10.48 | 0.995 |
| **XGBoost (depth 6), deployed** | **1.55** | **1.18** | **1.000** |
| XGBoost (depth 8) | 1.67 | 1.23 | 1.000 |

## EDA summary

Full numbers in `reports/eda_findings.md`, charts in `reports/figures/`.

* Price and size are both uniformly distributed. The correlation between size and price is -0.003.
* Average price by city only ranges from 250.8 to 258.5 lakhs (about 3%). Mumbai (252.1 L) and Silchar (252.6 L) are priced the same.
* Price per sq ft, furnishing, facing, parking, amenities, schools and hospitals all show gaps under 3% between groups. None of them has a real effect on price.
* BHK is about 20% each of 1 to 5 in every city.
* Good-Investment rate rises from 21.5% (low transport) to 60.5% (high transport), but only because transport is an input of the label.
* No crime-rate column exists, so that brief item cannot be analysed. No sale date exists, so true price trends over time cannot be built.

## Honest limitations

1. **The data is synthetic.** Price has no relationship with size, city, amenities or infrastructure. Any "insight" about the housing market from this file would be wrong.
2. **The classifier learns my rule, not the market.** The label is a formula. The model reaches 89.6% because it can see five of the six factors; it cannot see the price factor. `City` is its top feature (53% importance) only because city decides the growth tier and the median-price comparison.
3. **Why price is hidden from the classifier.** Feeding `Price_in_Lakhs` and `Price_per_SqFt` takes accuracy to 99.3%, but that is leakage of the label definition, not skill. It is logged in MLflow as a demonstration and not deployed.
4. **The regression R2 means nothing.** The target is a formula of the inputs, and even a fixed 8% guess scores R2 = 0.988. The useful comparison is RMSE against that baseline (1.55 vs 23.11 lakhs), and it only shows that the model recovered my formula. It does not show real forecasting ability. The app shows a plus or minus RMSE band for that reason.
5. **No price history exists**, so growth rates are assumptions, listed in `src/config.py`. Change them there and re-run to test other scenarios.

## Business use cases covered

* Investors get a verdict, confidence, 5-year price estimate and a plain-language score breakdown per property.
* Buyers can filter listings by state, type, BHK, price, area and availability, and sort by investment score.
* Platforms can reuse `build_row` and the saved pipelines in `app.py` to score listings in bulk.

## Deploying on Streamlit Cloud

Push everything except `data/raw/` to GitHub, set `app.py` as the main file, and keep `requirements.txt` as is. The pinned scikit-learn and xgboost versions must match the ones used to train, otherwise loading the `.joblib` files can fail.
